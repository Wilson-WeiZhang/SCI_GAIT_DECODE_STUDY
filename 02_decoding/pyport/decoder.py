"""Decoder — Python readout (02_decoding/pyport)

Session loader, fold transform onto the shared PCA basis, stage-1 ridge readout, optional causal low-pass of the stage-1 output (stage1_lpf_hz), stage-2 causal FIR on the predictions, per-trial scoring

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import os
import time
import warnings

import numpy as np
import scipy.linalg as sla

from . import matio
from .mcompat import (EPS, fold_ids, mround, rand_twister, mstd, nanmean_cols, nanstd_cols, nanmean_vec, safe_corr,
                      first_nanargmin, first_nanargmax)

JOINTS_DEFAULT = ['LHipW X', 'LKneW X', 'LAnkW X', 'RHipW X', 'RKneW X', 'RAnkW X']
SOLVER_LOG = {'cholesky': 0, 'fallback_sym': 0}


def _find_export_dict(d):
    if 'U_raw' in d:
        return d
    for v in d.values():
        if isinstance(v, dict) and 'U_raw' in v:
            return v
        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1 and isinstance(v.ravel()[0], dict) \
                and 'U_raw' in v.ravel()[0]:
            return v.ravel()[0]
    raise KeyError('export file has no U_raw; top-level keys: %s' % sorted(d.keys()))


class Session:
    """One exported session: un-lagged band feature window per trial + full-window target + fold assignment."""

    def __init__(self, label, fs, T_total, U, Y_full, et0, Lk, fa_fold, fa_seed, joint_labels, meta):
        self.label = label
        self.fs = float(fs)
        self.T_total = int(T_total)
        self.U = U
        self.Y_full = Y_full
        self.et0 = np.asarray(et0, dtype=np.int64)
        self.Lk = np.asarray(Lk, dtype=np.int64)
        self.fa_fold = None if fa_fold is None else np.asarray(fa_fold, dtype=np.int64)
        self.fa_seed = fa_seed
        self.joint_labels = joint_labels
        self.meta = meta
        self.n_ev = len(U)

    @classmethod
    def from_export(cls, path, baseline_mode='mean', label=None):
        raw = matio.loadmat_any(path)
        d = _find_export_dict(raw)
        n_ev = None
        U_raw = matio.cell_list(d['U_raw'])
        base_mu = matio.cell_list(d['base_mu'])
        Y_full = matio.cell_list(d['Y_full'])
        n_ev = len(U_raw)
        fs_e, fs_g = matio.scalar(d['fs_eeg']), matio.scalar(d['fs_goni'])
        if fs_e != fs_g:
            raise ValueError('fs_eeg %g != fs_goni %g (the readout assumes equal rates)' % (fs_e, fs_g))
        U = []
        bmu = []
        for t in range(n_ev):
            u = np.asarray(U_raw[t], dtype=float)
            bmu.append(np.asarray(base_mu[t], dtype=float).reshape(-1, 1) if u.size else None)
            if u.size == 0:
                U.append(None)
                continue
            if baseline_mode == 'mean':
                u = u - np.asarray(base_mu[t], dtype=float).reshape(-1, 1)
            elif baseline_mode != 'none':
                raise ValueError('eeg_baseline_mode %s not supported' % baseline_mode)
            U.append(np.ascontiguousarray(u))
        Yf = []
        for t in range(n_ev):
            y = np.asarray(Y_full[t], dtype=float)
            Yf.append(y if y.size else None)
        lab = label
        for key in ('label', 'session', 'session_label'):
            if lab is None and key in d:
                lab = matio.to_str(d[key])
        if lab is None:
            lab = os.path.splitext(os.path.basename(path))[0]
        joints = JOINTS_DEFAULT
        if 'target_labels' in d:
            joints = [matio.to_str(x) for x in matio.cell_list(d['target_labels'])]
        fa = matio.vec(d['fa_fold']).astype(np.int64) if 'fa_fold' in d else None
        seed = matio.scalar(d['fa_seed']) if 'fa_seed' in d else None
        meta = {}
        for k in ('bp_band_feature_mode', 'goni_qc_flags_env', 'basis_path', 'basis_md5', 'matlab_version',
                  'cp_md5'):
            if k in d:
                meta[k] = matio.to_str(d[k])
        for k in ('valid_trial', 'Tk', 'joint_idx', 'ev_trial_idx'):
            if k in d:
                meta[k] = matio.vec(d[k])
        et0 = matio.vec(d['et0'])
        Lk = matio.vec(d['Lk'])
        out = cls(lab, fs_e, matio.scalar(d['T_total']), U, Yf, et0, Lk, fa, seed, joints, meta)
        out.base_mu = bmu
        out.baseline_mode = baseline_mode
        return out

    def attach_rest(self, path, lags1):
        """U_rest[t] = the rest window the MATLAB decoder substitutes for test trial t (same geometry as U: crop = -min lag,
        Tk output columns), minus the IW trial's own base_mu (build_xy_delta0_local subtracts base_mu of ev(t) from
        the rest window too)."""
        d = matio.loadmat_any(path)
        lab = matio.to_str(d['session_label'])
        if lab != self.label:
            raise ValueError('rest export %s is for %s, session is %s' % (path, lab, self.label))
        if not np.array_equal(matio.vec(d['lag_samples']).astype(np.int64), np.asarray(lags1, dtype=np.int64)):
            raise ValueError('rest export lag set differs from the run lags (rest windows depend on the lag set)')
        n = int(matio.scalar(d['n_ev']))
        if n != self.n_ev:
            raise ValueError('rest export n_ev %d != %d' % (n, self.n_ev))
        ok = matio.vec(d['rest_ok']).astype(bool)
        Ur = matio.cell_list(d['U_rest'])
        bm = matio.cell_list(d['base_mu'])
        et0 = matio.vec(d['et0'])
        Tk_r = matio.vec(d['Tk']).astype(np.int64)
        _, Tk = self.geometry(np.asarray(lags1))
        bmode = matio.to_str(d['eeg_baseline_mode'])
        if bmode != self.baseline_mode:
            raise ValueError('rest export baseline %s != %s' % (bmode, self.baseline_mode))
        self.U_rest = [None] * n
        for t in range(n):
            if not ok[t]:
                continue
            if Tk[t] <= 0 or Tk_r[t] != Tk[t] or et0[t] != self.et0[t]:
                raise ValueError('rest export trial %d geometry differs from the H1 export' % (t + 1))
            b = np.asarray(bm[t], dtype=float).reshape(-1, 1)
            if not np.array_equal(b, self.base_mu[t]):
                raise ValueError('rest export trial %d base_mu differs from the H1 export' % (t + 1))
            u = np.asarray(Ur[t], dtype=float)
            self.U_rest[t] = np.ascontiguousarray(u - b if bmode == 'mean' else u)
        self.rest_meta = {'rest_export': os.path.abspath(path), 'rest_seed': int(matio.scalar(d['input_control_seed'])),
                          'rest_cp_md5': matio.to_str(d['cp_md5']), 'n_rest_ok': int(ok.sum())}

    def geometry(self, lags1):
        crop = max(1, 1 - int(lags1.min())) - 1
        max_lag = int(lags1.max())
        Tk = np.zeros(self.n_ev, dtype=np.int64)
        for t in range(self.n_ev):
            if self.U[t] is None or self.Lk[t] <= 0:
                continue
            vs = crop + 1 + self.et0[t] - 1
            ve = min(self.T_total - max_lag, self.et0[t] + self.Lk[t] - 1)
            Tk[t] = max(0, ve - vs + 1)
        return crop, Tk


def load_basis(path, K):
    """MATLAB basis(:, 1:K) from a -v7.3 basis file (h5py transposes: basis(:,1:K) = h5['basis'][:K, :].T)."""
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            B = np.asarray(f['basis'][()], dtype=float).T
    else:
        B = np.asarray(matio.loadmat_any(path)['basis'], dtype=float)
    K = min(int(round(K)), B.shape[1])
    return np.ascontiguousarray(B[:, :K])


def chol_solve(A, B):
    """mldivide on a symmetric matrix with positive diagonal = Cholesky (upper triangle, as MATLAB chol)."""
    try:
        c = sla.cho_factor(A, lower=False, check_finite=False)
        SOLVER_LOG['cholesky'] += 1
        return sla.cho_solve(c, B, check_finite=False)
    except np.linalg.LinAlgError:
        SOLVER_LOG['fallback_sym'] += 1
        return sla.solve(A, B, assume_a='sym', check_finite=False)


def fold_scaler(sess, tr, lags1, crop, Tk, mode='exact', perms=None):
    """fast: the same multiplicity-weighted sums as one matrix-vector product per trial (integer weights = number of
    lag blocks that contain each window sample; only the floating-point summation order differs)."""
    n_ch = sess.U[tr[0]].shape[0]
    sum_x = np.zeros(n_ch)
    sum_x2 = np.zeros(n_ch)
    count_x = np.zeros(n_ch)
    for t in tr:
        U = sess.U[t]
        ok_full = np.isfinite(U)
        U0 = np.where(ok_full, U, 0.0)
        U2 = U0 * U0
        n = Tk[t]
        if mode == 'exact':
            pt = None if perms is None else perms[t]
            for lag in lags1:
                a = crop + lag
                cols = slice(a, a + n) if pt is None else a + pt
                sum_x += U0[:, cols].sum(axis=1)
                sum_x2 += U2[:, cols].sum(axis=1)
                count_x += ok_full[:, cols].sum(axis=1)
        else:
            w = np.zeros(U.shape[1] + 1)
            for lag in lags1:
                w[crop + lag] += 1
                w[crop + lag + n] -= 1
            w = np.cumsum(w)[:-1]
            sum_x += U0 @ w
            sum_x2 += U2 @ w
            count_x += ok_full.astype(float) @ w
    count_safe = np.maximum(count_x, 1)
    mu = sum_x / count_safe
    var_x = (sum_x2 / count_safe) - mu ** 2
    sd = np.sqrt(np.maximum(var_x, 0))
    mu[~np.isfinite(mu) | (count_x == 0)] = 0
    sd[~np.isfinite(sd) | (sd <= EPS) | (count_x == 0)] = 1
    return mu, sd


def transform_trial(U, mu, sd, basisT, lags1, crop, n, mode, out, perm=None):
    """Write the projected lag embedding of one trial, sample-major, into out [n x K*L] (= MATLAB Z')."""
    K = basisT.shape[0]
    if mode == 'exact':
        for li, lag in enumerate(lags1):
            a = crop + lag
            blk = U[:, a:a + n] if perm is None else U[:, a + perm]
            blk = (blk - mu[:, None]) / sd[:, None]
            blk[~np.isfinite(blk)] = 0
            out[:, li * K:(li + 1) * K] = (basisT @ blk).T
    else:
        z = (U - mu[:, None]) / sd[:, None]
        z[~np.isfinite(z)] = 0
        P = (basisT @ z).T
        for li, lag in enumerate(lags1):
            a = crop + lag
            out[:, li * K:(li + 1) * K] = P[a:a + n] if perm is None else P[a + perm]


def standardize_cells(Ys):
    """As in core/run_decoder.m (N-1 std, omitnan; sd non-finite or <= 0 -> 1)."""
    Y_all = np.vstack(Ys)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = nanmean_cols(Y_all)
        sd = nanstd_cols(Y_all)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1
    return mu, sd


def train_wiener(Zall, cols, Yz, lambda_grid, inner_k, mode, want_cond=False):
    """the last column). Grams are formed as Xt' * Xt on the sample-major copy (= MATLAB Xt * Xt')."""
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    N = cols.size
    J = Yz.shape[1]
    P1 = Zall.shape[1]
    joint_active = ~np.all(np.isnan(Yz), axis=0)
    inner_K = min(max(1, int(round(inner_k))), N)
    fold = fold_ids(20260505, N, inner_K)
    ridge = np.ones(P1)
    ridge[-1] = 0
    di = np.diag_indices(P1)
    sse = np.zeros(lambda_grid.size)
    nn = np.zeros(lambda_grid.size)
    act = np.flatnonzero(joint_active)

    def accumulate(A0, XY, Xe, Ye):
        Ws = []
        for lam in lambda_grid:
            A = A0.copy()
            A[di] += lam * ridge
            Ws.append(chol_solve(A, XY))
        Yp_all = Xe @ np.hstack(Ws)
        for li in range(lambda_grid.size):
            Yp = Yp_all[:, li * J:(li + 1) * J]
            for j in act:
                r = Yp[:, j] - Ye[:, j]
                sse[li] += np.sum(r * r)
                nn[li] += r.size

    if mode == 'exact':
        X_aug = Zall[cols]
        for k in range(1, inner_K + 1):
            tr = fold != k
            te = ~tr
            Xt = X_aug[tr]
            A0 = Xt.T @ Xt
            XY = Xt.T @ Yz[tr]
            del Xt
            accumulate(A0, XY, X_aug[te], Yz[te])
        A = X_aug.T @ X_aug
        B = X_aug.T @ Yz
        del X_aug
    else:
        G, XYs = [], []
        for k in range(1, inner_K + 1):
            m = fold == k
            Xk = Zall[cols[m]]
            G.append(Xk.T @ Xk)
            XYs.append(Xk.T @ Yz[m])
        for k in range(1, inner_K + 1):
            A0 = None
            XY = None
            for m in range(inner_K):
                if m == k - 1:
                    continue
                A0 = G[m].copy() if A0 is None else A0 + G[m]
                XY = XYs[m].copy() if XY is None else XY + XYs[m]
            if A0 is None:
                A0 = np.zeros((P1, P1))
                XY = np.zeros((P1, J))
            m = fold == k
            accumulate(A0, XY, Zall[cols[m]], Yz[m])
        A = G[0].copy()
        B = XYs[0].copy()
        for m in range(1, inner_K):
            A += G[m]
            B += XYs[m]
    with np.errstate(invalid='ignore', divide='ignore'):
        cv_mse = sse / nn
    best = first_nanargmin(cv_mse)
    lam = float(lambda_grid[best])
    A[di] += lam * ridge
    W = chol_solve(A, B)
    out = {'W': W, 'lambda': lam, 'cv_mse': cv_mse, 'joint_active': joint_active, 'cv_inner_k': inner_K}
    if want_cond:
        ev = np.linalg.eigvalsh(A)
        out['cond_refit'] = float(ev[-1] / ev[0]) if ev[0] > 0 else float('inf')
    return out


STAGE1_LPF_FS = 250.0


def stage1_lowpass(x, hz, fs=STAGE1_LPF_FS):
    """first finite sample, initialised at that sample's value (lfilter on x - x0, + x0); all-NaN series unchanged."""
    from scipy.signal import butter, lfilter
    x = np.asarray(x, dtype=float)
    b, a = butter(2, hz / (fs / 2))
    xm = np.moveaxis(x, 0, 0)
    out = xm.copy()
    flat = out.reshape(out.shape[0], -1)
    src = xm.reshape(xm.shape[0], -1)
    for c in range(flat.shape[1]):
        ok = np.isfinite(src[:, c])
        if not ok.any():
            continue
        i = np.flatnonzero(ok)
        i0, i1 = i[0], i[-1] + 1
        if not ok[i0:i1].all():
            raise ValueError('stage1_lowpass: non-contiguous finite run')
        flat[i0:i1, c] = lfilter(b, a, src[i0:i1, c] - src[i0, c]) + src[i0, c]
    return np.ascontiguousarray(np.moveaxis(flat.reshape(xm.shape), 0, 0))


def fit_predict_angle(Zall, offs, Tk, Y, fit_tr, fit_te, cfg, mode, want_cond=False):
    """As in core/run_decoder.m. fit_tr / fit_te: 0-based trial ids (ascending for fit_tr). Returns A_pred list, model, mu, sd."""
    mu, sd = standardize_cells([Y[t] for t in fit_tr])
    cols = np.concatenate([np.arange(offs[t], offs[t] + Tk[t]) for t in fit_tr])
    Yz = np.vstack([(Y[t] - mu) / sd for t in fit_tr])
    model = train_wiener(Zall, cols, Yz, cfg.lambda_grid, cfg.inner_k, mode, want_cond)
    A_pred = []
    for t in fit_te:
        Xa = Zall[offs[t]:offs[t] + Tk[t]]
        A_pred.append((Xa @ model['W']) * sd + mu)
    if getattr(cfg, 'stage1_lpf_hz', 0) > 0:
        A_pred = [stage1_lowpass(x, cfg.stage1_lpf_hz) for x in A_pred]
    return A_pred, model, mu, sd


def angle_oof_predictions(Zall, offs, Tk, Y, tr, outer_fold, cfg, mode, trace=None):
    """As in core/run_decoder.m."""
    n = len(tr)
    if n < 2:
        raise ValueError('angle_oof_predictions requires at least two training trials')
    K = min(max(2, int(round(cfg.inner_k))), n)
    fid = fold_ids(20260518 + int(outer_fold), n, K)
    A_oof = [None] * n
    for k in range(1, K + 1):
        inner_tr = [tr[i] for i in range(n) if fid[i] != k]
        pos = [i for i in range(n) if fid[i] == k]
        A_val, model, _, _ = fit_predict_angle(Zall, offs, Tk, Y, inner_tr, [tr[i] for i in pos], cfg, mode)
        for ii, p in enumerate(pos):
            A_oof[p] = A_val[ii]
        if trace is not None:
            trace.setdefault('oof_lambda', []).append(model['lambda'])
            trace.setdefault('oof_cv_mse', []).append(model['cv_mse'])
    if trace is not None:
        trace['oof_fold_id'] = fid
    return A_oof


def fir_lag_matrix(x, lags):
    """As in core/run_decoder.m; returns X [rows x L] and 0-based row indices idx (MATLAB idx - 1)."""
    n = x.size
    lo = max(1, 1 - int(lags.min()))
    hi = min(n, n - int(lags.max()))
    if hi < lo:
        return None, None
    idx = np.arange(lo - 1, hi)
    return x[idx[:, None] + lags[None, :]], idx


def fir_has_rows(n, lags):
    if lags.size == 0 or n < 1:
        return False
    return min(n, n - int(lags.max())) >= max(1, 1 - int(lags.min()))


def fir_lag_design(F, Y, j, lags):
    """As in core/run_decoder.m (stack over trials in order)."""
    Xs, ys = [], []
    for Fk, Yk in zip(F, Y):
        if Fk is None or Yk is None or Fk.size == 0 or Yk.size == 0:
            continue
        n = min(Fk.shape[0], Yk.shape[0])
        if not fir_has_rows(n, lags):
            continue
        X, idx = fir_lag_matrix(Fk[:n, j], lags)
        if X is None:
            continue
        Xs.append(X)
        ys.append(Yk[idx, j])
    if not Xs:
        return np.zeros((0, lags.size)), np.zeros(0)
    return np.vstack(Xs), np.concatenate(ys)


def _first_J(Y):
    for y in Y:
        if y is not None and y.size:
            return y.shape[1]
    return 0


def fir_prepare(F, Y, j, lags):
    """Standardised design of one joint (as in core/run_decoder.m, up to the Gram); None when the model is skipped."""
    X, y = fir_lag_design(F, Y, j, lags)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]
    nv = y.size
    if X.shape[0] < lags.size + 10 or mstd(y) <= EPS:
        return None, nv
    x_mu = X.mean(axis=0)
    x_sd = mstd(X, axis=0)
    x_mu[~np.isfinite(x_mu)] = 0
    x_sd[~np.isfinite(x_sd) | (x_sd <= EPS)] = 1
    Xz = (X - x_mu) / x_sd
    y_mu = float(np.mean(y))
    y_sd = float(mstd(y))
    if not np.isfinite(y_sd) or y_sd <= EPS:
        y_sd = 1.0
    yz = (y - y_mu) / y_sd
    Xa = np.hstack([Xz, np.ones((Xz.shape[0], 1))])
    A = Xa.T @ Xa
    b = Xa.T @ yz
    return {'A': A, 'b': b, 'x_mu': x_mu, 'x_sd': x_sd, 'y_mu': y_mu, 'y_sd': y_sd}, nv


def fir_solve(prep, lam):
    ridge = max(float(lam), 0.0)
    A = prep['A'].copy()
    d = np.arange(A.shape[0] - 1)
    A[d, d] += ridge
    return chol_solve(A, prep['b'])


def fir_predict(models, F_te, lags, rowwise=False):
    """rowwise (zero_eeg only): each output row as an elementwise product summed along the row, so bitwise identical
    design rows give bitwise identical predictions (BLAS gemv may round its tail rows differently by 1 ulp, which turns
    a constant prediction into a spurious |r| up to ~0.1); default = the gated BLAS product."""
    C = []
    for Fk in F_te:
        Ck = np.full(Fk.shape, np.nan)
        for j, m in enumerate(models):
            if m is None:
                continue
            X, idx = fir_lag_matrix(Fk[:, j], lags)
            if X is None:
                continue
            Xz = (X - m['x_mu']) / m['x_sd']
            Xz[~np.isfinite(Xz)] = 0
            Xa = np.hstack([Xz, np.ones((Xz.shape[0], 1))])
            Ck[idx, j] = ((Xa * m['w']).sum(axis=1) if rowwise else Xa @ m['w']) * m['y_sd'] + m['y_mu']
        C.append(Ck)
    return C


def post_prediction_fir(F, Y, F_te, lam, lags, preps=None, rowwise=False):
    """As in core/run_decoder.m. preps: optional cached per-joint fir_prepare output for (F, Y) (lambda-independent)."""
    J = _first_J(Y)
    models, n_valid, fit_ok = [None] * J, np.zeros(J), np.zeros(J, dtype=bool)
    for j in range(J):
        prep, nv = preps[j] if preps is not None else fir_prepare(F, Y, j, lags)
        n_valid[j] = nv
        if prep is None:
            continue
        w = fir_solve(prep, lam)
        models[j] = {'w': w, 'x_mu': prep['x_mu'], 'x_sd': prep['x_sd'], 'y_mu': prep['y_mu'], 'y_sd': prep['y_sd']}
        fit_ok[j] = True
    return fir_predict(models, F_te, lags, rowwise), {'models': models, 'n_valid': n_valid, 'fit_ok': fit_ok}


def safe_rpt_cells(Y_true, Y_pred, j):
    """As in core/run_decoder.m."""
    r = []
    for k, Yt in enumerate(Y_true):
        if k >= len(Y_pred) or Yt is None or Y_pred[k] is None or Yt.size == 0 or Y_pred[k].size == 0:
            r.append(np.nan)
            continue
        n = min(Yt.shape[0], Y_pred[k].shape[0])
        if n < 2:
            r.append(np.nan)
            continue
        r.append(safe_corr(Yt[:n, j], Y_pred[k][:n, j]))
    return nanmean_vec(r)


def mean_rpt_all_joints(Y_true, Y_pred):
    """As in core/run_decoder.m."""
    J = _first_J(Y_true)
    if J == 0:
        return float('nan')
    return nanmean_vec([safe_rpt_cells(Y_true, Y_pred, j) for j in range(J)])


def post_prediction_fir_cv_lambda(F_oof, Y_train, F_te, lambda_grid, lags, inner_k, mode, rowwise=False):
    """lambda-independent standardised Gram of every (fold, joint) is built once and reused (same operations)."""
    grid = np.unique(np.asarray(lambda_grid, dtype=float))
    grid = grid[np.isfinite(grid) & (grid >= 0)]
    if grid.size == 0:
        grid = np.array([0.01])
    n = len(Y_train)
    K = min(max(1, int(round(inner_k))), n)
    cv_score = np.full(grid.size, np.nan)
    if K >= 2:
        fid = fold_ids(20260528, n, K)
        J = _first_J(Y_train)
        fold_preps = {}
        if mode == 'fast':
            for k in range(1, K + 1):
                tr = fid != k
                if tr.sum() < 3 or (~tr).sum() < 1:
                    continue
                Ftr = [F_oof[i] for i in np.flatnonzero(tr)]
                Ytr = [Y_train[i] for i in np.flatnonzero(tr)]
                fold_preps[k] = [fir_prepare(Ftr, Ytr, j, lags) for j in range(_first_J(Ytr))]
        for li, lam in enumerate(grid):
            scores = np.full(K, np.nan)
            for k in range(1, K + 1):
                tr = fid != k
                va = ~tr
                if tr.sum() < 3 or va.sum() < 1:
                    continue
                Ftr = [F_oof[i] for i in np.flatnonzero(tr)]
                Ytr = [Y_train[i] for i in np.flatnonzero(tr)]
                Fva = [F_oof[i] for i in np.flatnonzero(va)]
                C_val, _ = post_prediction_fir(Ftr, Ytr, Fva, lam, lags, preps=fold_preps.get(k), rowwise=rowwise)
                scores[k - 1] = mean_rpt_all_joints([Y_train[i] for i in np.flatnonzero(va)], C_val)
            cv_score[li] = nanmean_vec(scores)
    if np.all(~np.isfinite(cv_score)):
        best = int(np.argmin(np.abs(grid - 0.01)))
    else:
        best = first_nanargmax(cv_score)
    sel = float(grid[best])
    C_pred, info = post_prediction_fir(F_oof, Y_train, F_te, sel, lags, rowwise=rowwise)
    info.update({'lambda_grid': grid, 'cv_score': cv_score, 'selected_lambda': sel, 'cv_inner_k': K,
                 'lags': lags})
    return C_pred, info


def single_trial_r(yt_full, yh_full, j, time_sec):
    """As in core/run_decoder.m, single_trial_model_r with rpt_pred_smooth_sec = 0 -> (r, rpt_r, n_samples, first_t)."""
    if yh_full is None or yt_full is None or yh_full.size == 0 or yt_full.size == 0:
        return np.nan, np.nan, 0, np.nan
    n = min(yt_full.shape[0], yh_full.shape[0])
    if n < 5:
        return np.nan, np.nan, 0, np.nan
    yt = yt_full[:n, j]
    yh = yh_full[:n, j]
    ok = np.isfinite(yt) & np.isfinite(yh)
    ns = int(ok.sum())
    r = np.nan
    if ns >= 5:
        r = safe_corr(yt[ok], yh[ok])
    first_t = np.nan
    nn = min(time_sec.size, ok.size)
    ii = np.flatnonzero(ok[:nn])
    if ii.size:
        first_t = float(time_sec[ii[0]])
    return r, r, ns, first_t


def circular_shift_perm(n, lags1, seed, et0, crop, T_total):
    """output column c reads out_t(mod(c + shift, n)), shift = choices(randi(numel(choices))) after rng(seed,'twister');
    MATLAB randi(m) = floor(rand*m) + 1 (verified against R2023a, tests/mref_randi.mat)."""
    if n <= 1:
        return None
    min_lag, max_lag = int(np.min(lags1)), int(np.max(lags1))
    out1 = et0 + crop
    lo = max(1 - min_lag, out1)
    hi = min(T_total - max_lag, out1 + n - 1)
    if hi - lo + 1 != n:
        return None
    min_shift = min(int(np.max(np.abs(lags1))) + 1, max(1, n - 1))
    if min_shift >= n:
        min_shift = max(1, n // 3)
    ch = np.array(list(range(min_shift, n)) + list(range(1, max(1, n - min_shift) + 1)), dtype=np.int64)
    ch = np.unique((ch - 1) % n + 1)
    shift = int(ch[int(np.floor(rand_twister(seed, 1)[0] * ch.size))])
    return (np.arange(n) + shift) % n


def block_folds(n, outer_k):
    """edges = round(linspace(0, n, K+1)) with MATLAB round (linspace with d1 = 0: (0:K) .* n ./ K)."""
    K = min(int(mround(outer_k)), n)
    edges = mround(np.arange(K + 1) * float(n) / K).astype(np.int64)
    edges[0], edges[-1] = 0, n
    fold = np.zeros(n, dtype=np.int64)
    for k in range(1, K + 1):
        fold[edges[k - 1]:edges[k]] = k
    return fold


def fold_sets(fa, fold, Tk, split, n_real=None):
    """0-based (train, test) trial lists of one outer fold. random / block: train = fa != fold, test = fa == fold;
    invalid (empty) trials are dropped."""
    n = len(fa)
    if split == 'chrono_expand':
        tr = [t for t in range(n) if fa[t] < fold and Tk[t] > 0]
    else:
        tr = [t for t in range(n) if fa[t] != fold and Tk[t] > 0]
    te = [t for t in range(n) if fa[t] == fold and Tk[t] > 0]
    if n_real is not None:
        tr = [t for t in tr if t < n_real]
        te = [t for t in te if t >= n_real]
    return tr, te


def default_folds(fa, cfg):
    K = int(fa.max())
    if cfg.split == 'chrono_expand':
        return list(range(int(cfg.chrono_calib_blocks) + 1, K + 1))
    return list(range(1, K + 1))


def outer_folds(sess, cfg):
    """exported fa_fold must match). block / chrono_expand: contiguous blocks (block_folds)."""
    if cfg.split in ('block', 'chrono_expand'):
        return block_folds(sess.n_ev, cfg.outer_k)
    if cfg.split != 'random':
        raise ValueError('split must be random | block | chrono_expand, got %s' % cfg.split)
    if sess.fa_seed is None:
        raise ValueError('export has no fa_seed')
    fold = fold_ids(int(sess.fa_seed), sess.n_ev, min(int(round(cfg.outer_k)), sess.n_ev))
    if sess.fa_fold is not None and int(round(cfg.outer_k)) == 10 and not np.array_equal(fold, sess.fa_fold):
        raise AssertionError('recomputed outer folds differ from exported fa_fold')
    return fold


def run_fold(sess, basisK, cfg, fold, fa, lags1, lags2, crop, Tk, trace=False, keep=False):
    mode = cfg.mode
    fs = sess.fs
    tr, te = fold_sets(fa, fold, Tk, cfg.split, getattr(sess, 'n_real', None))
    rest = cfg.input_control == 'rest_test_only'
    if rest:
        te = [t for t in te if sess.U_rest[t] is not None]
    if not tr or not te:
        return None
    perms = None
    if cfg.input_control == 'circular_shift':
        perms = {t: circular_shift_perm(Tk[t], lags1, int(cfg.input_control_seed) + 1009 * (t + 1),
                                        int(sess.et0[t]), crop, sess.T_total) for t in tr + te}
    elif cfg.input_control == 'zero_eeg':
        import copy
        sess = copy.copy(sess)
        sess.U = [None if u is None else np.zeros_like(u) for u in sess.U]
    T = {'fold': fold, 'tr_idx': np.array(tr) + 1, 'te_idx': np.array(te) + 1} if trace else None
    Y = [None] * sess.n_ev
    for t in tr + te:
        Y[t] = sess.Y_full[t][crop:crop + Tk[t], :]
    t0 = time.time()
    mu, sd = fold_scaler(sess, tr, lags1, crop, Tk, mode, perms)
    K = basisK.shape[1]
    L = lags1.size
    P1 = K * L + 1
    allv = sorted(tr + te)
    offs = np.zeros(sess.n_ev, dtype=np.int64)
    pos = 0
    for t in allv:
        offs[t] = pos
        pos += Tk[t]
    Zall = np.empty((pos, P1))
    Zall[:, -1] = 1.0
    basisT = np.ascontiguousarray(basisK.T)
    te_set = set(te)
    for t in allv:
        U = sess.U_rest[t] if (rest and t in te_set) else sess.U[t]
        transform_trial(U, mu, sd, basisT, lags1, crop, Tk[t], mode, Zall[offs[t]:offs[t] + Tk[t], :K * L],
                        None if perms is None else perms[t])
    t1 = time.time()
    A_te, model, a_mu, a_sd = fit_predict_angle(Zall, offs, Tk, Y, tr, te, cfg, mode, want_cond=trace)
    t2 = time.time()
    A_oof = angle_oof_predictions(Zall, offs, Tk, Y, tr, fold, cfg, mode, trace=T)
    del Zall
    t3 = time.time()
    Y_train = [Y[t] for t in tr]
    C_te, pinfo = post_prediction_fir_cv_lambda(A_oof, Y_train, A_te, cfg.post_lambda_grid, lags2,
                                                cfg.post_inner_k, mode, rowwise=cfg.input_control == 'zero_eeg')
    t4 = time.time()
    tsec = [(crop + np.arange(Tk[t])) / fs for t in te]
    rows = []
    J = len(sess.joint_labels)
    for ti, t in enumerate(te):
        for j in range(J):
            r, rpt, ns, ft = single_trial_r(Y[t], C_te[ti], j, tsec[ti])
            rows.append((sess.label, fold, t + 1, ti + 1, sess.joint_labels[j], 'calib', r, rpt, ns, ft))
    info = {'fold': int(fold), 'n_train': len(tr), 'n_test': len(te), 'lambda_stage1': model['lambda'],
            'stage1_cv_mse': [float(x) for x in model['cv_mse']],
            'stage2_selected_lambda': pinfo['selected_lambda'],
            'stage2_cv_score': [float(x) for x in pinfo['cv_score']],
            'stage2_top2_gap': _top2_gap(pinfo['cv_score']),
            'seconds': {'transform': t1 - t0, 'stage1_main': t2 - t1, 'stage1_oof': t3 - t2, 'stage2': t4 - t3}}
    if trace:
        T.update({'Y_te': [Y[t] for t in te], 'Y_tr': Y_train, 'feature_mu': mu, 'feature_sd': sd, 'basis': basisK,
                  'cv_mse': model['cv_mse'], 'lambda1': model['lambda'], 'W1': model['W'],
                  'joint_active': model['joint_active'], 'target_mu': a_mu, 'target_sd': a_sd,
                  'cond_refit': model.get('cond_refit', np.nan),
                  'lambda_grid1': np.asarray(cfg.lambda_grid, dtype=float),
                  'A_te': A_te, 'F_oof': A_oof, 'cv_score': pinfo['cv_score'],
                  'selected_lambda': pinfo['selected_lambda'], 'n_valid': pinfo['n_valid'],
                  'fit_ok': pinfo['fit_ok'], 'models': pinfo['models'], 'C_te': C_te, 'rows': rows,
                  'lags2': lags2, 'time_sec': tsec})
    elif keep:
        T = {'fold': fold, 'tr_idx': np.array(tr) + 1, 'te_idx': np.array(te) + 1, 'Y_te': [Y[t] for t in te],
             'A_te': A_te, 'C_te': C_te, 'time_sec': tsec, 'cv_mse': model['cv_mse'], 'lambda1': model['lambda'],
             'cv_score': pinfo['cv_score'], 'selected_lambda': pinfo['selected_lambda'],
             'post_lambda_grid': pinfo['lambda_grid'], 'n_valid': pinfo['n_valid'], 'fit_ok': pinfo['fit_ok']}
    return rows, info, T


def _top2_gap(cv):
    v = np.sort(np.asarray(cv, dtype=float)[np.isfinite(cv)])
    return float(v[-1] - v[-2]) if v.size >= 2 else float('nan')


def combine_cross(sess, xs):
    """xs = Session of the cross-condition export built with the TRAINING condition's band CSV."""
    import copy
    if xs.label != sess.label or xs.T_total != sess.T_total or xs.fs != sess.fs:
        raise ValueError('cross export does not belong to session %s' % sess.label)
    c = copy.copy(sess)
    c.U = list(sess.U) + list(xs.U)
    c.Y_full = list(sess.Y_full) + list(xs.Y_full)
    c.et0 = np.concatenate([sess.et0, xs.et0])
    c.Lk = np.concatenate([sess.Lk, xs.Lk])
    c.base_mu = list(sess.base_mu) + list(xs.base_mu)
    c.n_ev = sess.n_ev + xs.n_ev
    c.n_real = sess.n_ev
    return c


def run_session(sess, basisK, cfg, folds=None, trace=False, log=None, keep=False):
    fs = sess.fs
    if getattr(cfg, 'stage1_lpf_hz', 0) > 0 and fs != STAGE1_LPF_FS:
        raise ValueError('stage1_lpf_hz is defined for fs %g, session fs %g' % (STAGE1_LPF_FS, fs))
    lags1 = cfg.stage1_lags(fs)
    lags2 = cfg.stage2_lags(fs)
    fa = outer_folds(sess, cfg)
    fa_main = fa
    if cfg.input_control == 'cross_task':
        xs = sess.cross
        if cfg.cross_scope == 'all':
            if cfg.split == 'chrono_expand':
                raise ValueError('cross_scope all has no outer split; use split random')
            fa_main = np.full(sess.n_ev, 2, dtype=np.int64)
            fa = np.concatenate([fa_main, np.ones(xs.n_ev, dtype=np.int64)])
            folds = [1]
        else:
            fa = np.concatenate([fa_main, outer_folds(xs, cfg)])
        sess = combine_cross(sess, xs)
    elif cfg.cross_scope != 'folds':
        raise ValueError('cross_scope only applies to input_control = cross_task')
    crop, Tk = sess.geometry(lags1)
    Kmax = int(fa_main.max())
    if folds is None:
        folds = cfg.outer_folds
    if folds is None:
        folds = default_folds(fa_main, cfg)
    folds = [f for f in sorted(set(int(f) for f in folds)) if 1 <= f <= Kmax]
    all_rows, infos, traces = [], [], []
    for f in folds:
        out = run_fold(sess, basisK, cfg, f, fa, lags1, lags2, crop, Tk, trace=trace, keep=keep)
        if out is None:
            continue
        rows, info, T = out
        all_rows.extend(rows)
        infos.append(info)
        if trace or keep:
            traces.append(T)
        if log:
            log('fold %d done: stage1 lambda %g, stage2 lambda %g (top-2 gap %.3g), %.1fs' % (
                f, info['lambda_stage1'], info['stage2_selected_lambda'], info['stage2_top2_gap'],
                sum(info['seconds'].values())))
    return all_rows, infos, traces, {'fa_fold': fa, 'crop': crop, 'Tk': Tk, 'lags1': lags1, 'lags2': lags2}
