"""Decoder — Python readout (02_decoding/pyport)

Unit tests that need no recordings: the MATLAB-semantics helpers and synthetic sessions, against MATLAB ground truth in tests/mref_*.mat

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import os
import sys

import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from pyport import decoder
from pyport.config import Config
from pyport.mcompat import mround, randperm, parse_grid, rand_twister, fold_ids, mcorr

REF = os.path.join(HERE, 'mref_units.mat')


def _ref():
    if not os.path.exists(REF):
        return None
    return scipy.io.loadmat(REF, squeeze_me=False)


def test_mround_and_lags():
    assert np.array_equal(mround([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]), [-3, -2, -1, 1, 2, 3])
    lags = Config().stage1_lags(250)
    assert lags.size == 41 and lags[0] == -500 and lags[1] == -488 and lags[3] == -463 and lags[-2] == -13
    assert np.array_equal(Config().stage2_lags(250), lags)
    R = _ref()
    if R is not None:
        assert np.array_equal(lags, R['lag_round'].ravel())
        assert np.array_equal(Config(lag_ms='-2000:25:0').stage1_lags(250), R['lag_round_25'].ravel())
        assert np.array_equal(mround([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -462.5, -12.5]), R['round_halves'].ravel())
    assert np.array_equal(parse_grid('[1e-3 1e-2 0.1 1 10 100]'), [1e-3, 1e-2, 0.1, 1, 10, 100])


def test_randperm():
    R = _ref()
    if R is None:
        print('  (no MATLAB reference file; randperm checked against the spec example only)')
    else:
        n_checked = 0
        for k in R:
            if k.startswith('perm_'):
                _, s, n = k.split('_')
                assert np.array_equal(randperm(int(s), int(n)), R[k].ravel()), k
                n_checked += 1
        assert n_checked == 6
        assert np.array_equal(rand_twister(20260505, 5), R['rand5'].ravel())
    f = fold_ids(20261753, 59, 10)
    assert list(np.flatnonzero(f == 1) + 1) == [27, 43, 48, 54, 56, 58]
    assert 20260505 + sum(map(ord, 'subj01_sess01')) + ord('M') == 20261753


def _tw_case(R, c, mode):
    X = [np.asarray(x, dtype=float) for x in R['tw%d_X' % c].ravel()]
    Y = [np.asarray(y, dtype=float) for y in R['tw%d_Y' % c].ravel()]
    Z = np.ascontiguousarray(np.vstack([np.hstack(X), np.ones((1, sum(x.shape[1] for x in X)))]).T)
    m = decoder.train_wiener(Z, np.arange(Z.shape[0]), np.vstack(Y), [1e-3, 1e-2, 0.1, 1, 10, 100], 10, mode)
    return m, R['tw%d_W' % c], R['tw%d_cv_mse' % c].ravel(), float(R['tw%d_lambda' % c].ravel()[0])


def test_train_wiener_vs_matlab():
    R = _ref()
    if R is None:
        print('  SKIP: no MATLAB reference')
        return
    for c in (1, 2):
        for mode in ('exact', 'fast'):
            m, W, cv, lam = _tw_case(R, c, mode)
            assert m['lambda'] == lam, (c, mode, m['lambda'], lam)
            assert np.array_equal(np.isnan(m['cv_mse']), np.isnan(cv))
            ok = np.isfinite(cv)
            if ok.any():
                rel = np.max(np.abs(m['cv_mse'][ok] - cv[ok]) / np.abs(cv[ok]))
                assert rel < 1e-10, (c, mode, rel)
            assert np.array_equal(np.isnan(m['W']), np.isnan(W))
            f = np.isfinite(W)
            rel = np.max(np.abs(m['W'][f] - W[f])) / np.max(np.abs(W[f]))
            assert rel < 1e-9, (c, mode, rel)
            print('  train_wiener case %d %s: lambda %g, cv_mse rel %.2g, W rel %.2g, NaN cols %s' % (
                c, mode, lam, (np.max(np.abs(m['cv_mse'][ok] - cv[ok]) / np.abs(cv[ok])) if ok.any() else 0),
                rel, list(np.flatnonzero(np.isnan(W).all(axis=0)))))


def test_ridge_vs_direct():
    rng = np.random.default_rng(0)
    P, N = 12, 400
    X = rng.standard_normal((P, N))
    Y = rng.standard_normal((N, 2))
    Z = np.vstack([X, np.ones((1, N))])
    lam = 0.3
    A = Z @ Z.T + lam * np.diag(np.r_[np.ones(P), 0])
    W_direct = np.linalg.solve(A, Z @ Y)
    assert np.allclose(decoder.chol_solve(A, Z @ Y), W_direct, rtol=1e-12, atol=1e-12)
    Xc = X - X.mean(axis=1, keepdims=True)
    Wc = np.linalg.solve(Xc @ Xc.T + lam * np.eye(P), Xc @ (Y - Y.mean(axis=0)))
    assert np.allclose(W_direct[:P], Wc, rtol=1e-10, atol=1e-12)
    Yn = Y.copy()
    Yn[:, 1] = np.nan
    W = decoder.chol_solve(A, Z @ Yn)
    assert np.isnan(W[:, 1]).all() and np.allclose(W[:, 0], W_direct[:, 0], rtol=1e-12)
    x = np.cumsum(rng.standard_normal(3000))
    lags = np.arange(-40, 1, 4)
    Xl, idx = decoder.fir_lag_matrix(x, lags)
    assert idx[0] == 40 and Xl.shape == (3000 - 40, lags.size) and np.array_equal(Xl[:, -1], x[40:])
    assert np.array_equal(Xl[:, 0], x[:3000 - 40])
    prep, nv = decoder.fir_prepare([x[:, None]], [x[:, None] * 2 + rng.standard_normal((3000, 1))], 0, lags)
    w = decoder.fir_solve(prep, 1.0)
    Aref = prep['A'] + np.diag(np.r_[np.ones(lags.size), 0])
    assert np.allclose(w, np.linalg.solve(Aref, prep['b']), rtol=1e-10, atol=1e-12)


def test_corr():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(500), rng.standard_normal(500)
    assert abs(mcorr(a, b) - np.corrcoef(a, b)[0, 1]) < 1e-14


def _synthetic_session(seed=3, n_ev=14, n_ch=24, nan_joint=True):
    rng = np.random.default_rng(seed)
    U, Yf, et0, Lk = [], [], [], []
    pos = 1000
    mix = rng.standard_normal((n_ch, 4))
    for t in range(n_ev):
        L = int(rng.integers(900, 1300))
        if t == 5:
            L = 450
        tt = np.arange(L) / 250.0
        phase = 2 * np.pi * 0.9 * tt + rng.uniform(0, 6)
        src = np.vstack([np.sin(phase), np.cos(phase), np.sin(2 * phase), rng.standard_normal(L) * 0.3])
        u = mix @ src + 0.5 * rng.standard_normal((n_ch, L))
        y = np.column_stack([np.sin(phase + 0.3 * j) * (1 + 0.1 * j) for j in range(6)]) + 0.3 * rng.standard_normal((L, 6))
        if nan_joint:
            y[:, 2] = np.nan
        U.append(u)
        Yf.append(y)
        et0.append(pos)
        Lk.append(L)
        pos += L + 400
    s = decoder.Session('synth_sess01', 250, pos + 100, U, Yf, et0, Lk, None, 20261753, decoder.JOINTS_DEFAULT, {})
    basis = np.linalg.qr(rng.standard_normal((n_ch, n_ch)))[0]
    return s, basis


def test_fast_equals_exact_synthetic():
    s, basis = _synthetic_session()
    out = {}
    for mode in ('exact', 'fast'):
        cfg = Config(pca_k=6, mode=mode, outer_k=5)
        rows, infos, tr, geo = decoder.run_session(s, basis[:, :6], cfg, trace=True)
        out[mode] = (rows, infos, tr)
    re, rf = out['exact'][0], out['fast'][0]
    assert len(re) == len(rf) and len(re) > 0
    d = 0.0
    for a, b in zip(re, rf):
        assert a[:6] == b[:6]
        assert np.isnan(a[7]) == np.isnan(b[7])
        if np.isfinite(a[7]):
            d = max(d, abs(a[7] - b[7]))
    lam_e = [(i['lambda_stage1'], i['stage2_selected_lambda']) for i in out['exact'][1]]
    lam_f = [(i['lambda_stage1'], i['stage2_selected_lambda']) for i in out['fast'][1]]
    assert lam_e == lam_f, (lam_e, lam_f)
    dmu = max(np.max(np.abs(a['feature_mu'] - b['feature_mu'])) for a, b in zip(out['exact'][2], out['fast'][2]))
    dcv = max(np.nanmax(np.abs(a['cv_score'] - b['cv_score'])) for a, b in zip(out['exact'][2], out['fast'][2]))
    assert d < 1e-9, d
    n_nan_joint = sum(1 for r in re if r[4] == 'LAnkW X' and np.isnan(r[7]))
    assert n_nan_joint == sum(1 for r in re if r[4] == 'LAnkW X')
    print('  synthetic exact vs fast: %d rows, max |dr| %.2g, max |d feature_mu| %.2g, max |d cv_score| %.2g, '
          'lambdas %s' % (len(re), d, dmu, dcv, lam_e))


def test_loto_fold_numbering():
    s, basis = _synthetic_session(seed=4, n_ev=8)
    f = decoder.outer_folds(s, Config(outer_k=50))
    assert sorted(f) == list(range(1, 9))
    assert np.array_equal(f, fold_ids(20261753, 8, 8))


def test_block_folds_vs_matlab():
    p = os.path.join(HERE, 'mref_block.mat')
    if not os.path.exists(p):
        print('  SKIP: no mref_block.mat')
        return
    R = scipy.io.loadmat(p)
    n_checked = 0
    for k, v in R.items():
        if k.startswith('fold_'):
            _, n, K = k.split('_')
            assert np.array_equal(decoder.block_folds(int(n), int(K)), v.ravel()), k
            n_checked += 1
    assert n_checked == 450
    f = decoder.block_folds(45, 10)
    assert [int(np.sum(f == k)) for k in range(1, 11)] == [5, 4, 5, 4, 5, 4, 5, 4, 5, 4]


def test_chrono_expand_sets():
    n = 59
    fa = decoder.block_folds(n, 10)
    assert [int(np.sum(fa == k)) for k in range(1, 11)] == [6, 6, 6, 6, 6, 5, 6, 6, 6, 6]
    Tk = np.full(n, 100)
    Tk[[2, 20]] = 0
    cfg = Config(split='chrono_expand')
    assert decoder.default_folds(fa, cfg) == [4, 5, 6, 7, 8, 9, 10]
    tested = []
    for b in decoder.default_folds(fa, cfg):
        tr, te = decoder.fold_sets(fa, b, Tk, 'chrono_expand')
        assert max(tr) < min(te)
        assert all(fa[t] < b for t in tr) and all(fa[t] == b for t in te)
        assert 2 not in tr and 20 not in te
        tested += te
    assert sorted(tested) == [t for t in range(18, 59) if t != 20]
    tr4, te4 = decoder.fold_sets(fa, 4, Tk, 'chrono_expand')
    assert (len(tr4), len(te4)) == (17, 5)
    tr10, te10 = decoder.fold_sets(fa, 10, Tk, 'chrono_expand')
    assert (len(tr10), len(te10)) == (51, 6)
    tr, te = decoder.fold_sets(fa, 4, Tk, 'block')
    assert len(tr) + len(te) == n - 2


def test_chrono_expand_runs_synthetic():
    s, basis = _synthetic_session(seed=5, n_ev=20)
    cfg = Config(pca_k=6, split='chrono_expand')
    rows, infos, _, geo = decoder.run_session(s, basis[:, :6], cfg)
    assert [i['fold'] for i in infos] == [4, 5, 6, 7, 8, 9, 10]
    assert all(i['n_train'] >= 2 for i in infos)
    assert infos[-1]['n_train'] > infos[0]['n_train']


def test_randi_vs_matlab():
    p = os.path.join(HERE, 'mref_randi.mat')
    if not os.path.exists(p):
        print('  SKIP: no mref_randi.mat')
        return
    d = scipy.io.loadmat(p)
    R, seeds, im = d['R'], d['seeds'].ravel(), d['imaxs'].ravel()
    for i, s in enumerate(seeds):
        u = rand_twister(int(s), 2)
        for k, m in enumerate(im):
            assert np.array_equal(np.floor(u * m) + 1, R[i, k, :]), (s, m)


def test_circular_shift_vs_matlab():
    p = os.path.join(HERE, 'mref_cshift.mat')
    if not os.path.exists(p):
        print('  SKIP: no mref_cshift.mat')
        return
    C = scipy.io.loadmat(p, squeeze_me=True, struct_as_record=False)['C']
    lags = Config().stage1_lags(250)
    n_shifted = 0
    for c in C:
        n = int(c.n)
        perm = decoder.circular_shift_perm(n, lags, int(c.seed), int(c.et0), 500, int(c.T_total))
        ref = np.atleast_1d(np.asarray(c.col, dtype=np.int64))
        got = np.arange(n) if perm is None else perm
        assert np.array_equal(got, ref), (n, int(c.seed))
        n_shifted += perm is not None
    assert n_shifted == len(C) - 6


def test_controls_synthetic():
    s, basis = _synthetic_session(seed=6, n_ev=14)
    s.base_mu = [np.zeros((u.shape[0], 1)) for u in s.U]
    s.baseline_mode = 'mean'
    out = {}
    for mode in ('exact', 'fast'):
        out[mode] = decoder.run_session(s, basis[:, :6], Config(pca_k=6, mode=mode, outer_k=5,
                                                                input_control='circular_shift'))[0]
    d = max(abs(a[7] - b[7]) for a, b in zip(out['exact'], out['fast']) if np.isfinite(a[7]))
    assert d < 1e-9, d
    base = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5))[0]
    assert max(abs(a[7] - b[7]) for a, b in zip(base, out['fast']) if np.isfinite(a[7])) > 1e-3
    s.U_rest = [None if u is None else u.copy() for u in s.U]
    rr = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5, input_control='rest_test_only'))[0]
    assert [r[:6] for r in rr] == [r[:6] for r in base]
    assert max(abs(a[7] - b[7]) for a, b in zip(rr, base) if np.isfinite(a[7])) == 0
    s.U_rest[3] = None
    rr = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5, input_control='rest_test_only'))[0]
    assert 4 not in {r[2] for r in rr} and len(rr) == len(base) - 6
    import copy
    s.cross = copy.copy(s)
    xr = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5, input_control='cross_task'))[0]
    assert [(r[1], r[2] - s.n_ev, r[4]) for r in xr] == [(r[1], r[2], r[4]) for r in base]
    assert max(abs(a[7] - b[7]) for a, b in zip(xr, base) if np.isfinite(a[7])) == 0


def test_cross_scope_all():
    """Checks the index sets / seeds and that the predictions equal an independent hand-wired call chain of the
    gated stage functions on those sets (fold-loop order: fold scaler -> transform -> stage 1 -> OOF -> stage 2)."""
    import copy
    s, basis = _synthetic_session(seed=8, n_ev=12)
    s.base_mu = [np.zeros((u.shape[0], 1)) for u in s.U]
    s.baseline_mode = 'mean'
    xs, _ = _synthetic_session(seed=9, n_ev=9)
    xs.label, xs.T_total = s.label, s.T_total
    xs.base_mu = [np.zeros((u.shape[0], 1)) for u in xs.U]
    s.cross = xs
    cfg = Config(pca_k=6, input_control='cross_task', cross_scope='all')
    rows, infos, tr_, _ = decoder.run_session(s, basis[:, :6], cfg, trace=True)
    assert [i['fold'] for i in infos] == [1]
    T = tr_[0]
    lags1 = cfg.stage1_lags(250)
    comb = decoder.combine_cross(s, xs)
    crop, Tk = comb.geometry(lags1)
    tr = [t for t in range(s.n_ev) if Tk[t] > 0]
    te = [s.n_ev + i for i in range(xs.n_ev) if Tk[s.n_ev + i] > 0]
    assert list(T['tr_idx'] - 1) == tr and list(T['te_idx'] - 1) == te
    assert len(tr) == s.n_ev - 1 and len(te) == xs.n_ev - 1
    assert np.array_equal(T['oof_fold_id'], fold_ids(20260518 + 1, len(tr), min(10, len(tr))))
    Y = [None] * comb.n_ev
    for t in tr + te:
        Y[t] = comb.Y_full[t][crop:crop + Tk[t], :]
    mu, sd = decoder.fold_scaler(comb, tr, lags1, crop, Tk, 'fast')
    K, L = 6, lags1.size
    offs = np.zeros(comb.n_ev, dtype=np.int64)
    pos = 0
    for t in sorted(tr + te):
        offs[t] = pos
        pos += Tk[t]
    Z = np.empty((pos, K * L + 1))
    Z[:, -1] = 1.0
    bT = np.ascontiguousarray(basis[:, :6].T)
    for t in sorted(tr + te):
        decoder.transform_trial(comb.U[t], mu, sd, bT, lags1, crop, Tk[t], 'fast', Z[offs[t]:offs[t] + Tk[t], :K * L])
    A_te, _, _, _ = decoder.fit_predict_angle(Z, offs, Tk, Y, tr, te, cfg, 'fast')
    A_oof = decoder.angle_oof_predictions(Z, offs, Tk, Y, tr, 1, cfg, 'fast')
    C_te, _ = decoder.post_prediction_fir_cv_lambda(A_oof, [Y[t] for t in tr], A_te, cfg.post_lambda_grid,
                                                    cfg.stage2_lags(250), cfg.post_inner_k, 'fast')
    k = 0
    for ti, t in enumerate(te):
        for j in range(6):
            r = decoder.single_trial_r(Y[t], C_te[ti], j, (crop + np.arange(Tk[t])) / 250.0)
            assert rows[k][2] == t + 1 and (np.isnan(r[1]) and np.isnan(rows[k][7]) or r[1] == rows[k][7])
            k += 1
    assert k == len(rows)


def test_zero_eeg_synthetic():
    """input_control zero_eeg (Xseg(:) = 0 in every trial): same folds / rows as the full run, fold scaler mu 0 /
    sd 1, and an intercept-only stage 1, so every finite final prediction is constant within its trial."""
    s, basis = _synthetic_session(seed=10, n_ev=14)
    s.base_mu = [np.zeros((u.shape[0], 1)) for u in s.U]
    s.baseline_mode = 'mean'
    s.U[2][0, :5] = np.nan
    base = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5))[0]
    for mode in ('exact', 'fast'):
        rows, _, T, _ = decoder.run_session(s, basis[:, :6], Config(pca_k=6, outer_k=5, mode=mode,
                                                                    input_control='zero_eeg'), trace=True)
        assert [r[:6] for r in rows] == [r[:6] for r in base]
        for Tf in T:
            assert np.all(Tf['feature_mu'] == 0) and np.all(Tf['feature_sd'] == 1)
            W = Tf['W1'][:-1]
            assert np.all(W[:, np.isfinite(W).all(axis=0)] == 0)
            for C in Tf['C_te'] + Tf['A_te']:
                for j in range(C.shape[1]):
                    c = C[np.isfinite(C[:, j]), j]
                    assert c.size == 0 or np.all(c == c[0])
        assert all(np.isnan(r[7]) or abs(r[7]) < 1e-12 for r in rows)
    assert s.U[2][0, 0] != 0


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:
                fails += 1
                print('FAIL', name, repr(e))
    print('%d failure(s)' % fails)
    sys.exit(1 if fails else 0)
