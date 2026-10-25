"""Decoder — Python readout (02_decoding/pyport)

Fold-by-fold comparison of the Python readout with MATLAB reference dumps of the same session and configuration

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyport import decoder, matio
from pyport.config import Config
from pyport.run_session import resolve_basis


class GateFail(Exception):
    pass


def _cells(v):
    return [None if (x is None or np.asarray(x).size == 0) else np.asarray(x, dtype=float)
            for x in matio.cell_list(v)]


def _nan_pattern_equal(a, b):
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))


def cmp_cells(py, ref, scale='abs'):
    """max deviation over a list of arrays; NaN pattern must match. scale='abs' | 'rel_std' (per column std of ref)."""
    if len(py) != len(ref):
        raise GateFail('cell count %d vs %d' % (len(py), len(ref)))
    worst = 0.0
    for a, b in zip(py, ref):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        if not _nan_pattern_equal(a, b):
            raise GateFail('shape/NaN pattern differs: %s vs %s, NaN %d vs %d' % (
                a.shape, b.shape, np.isnan(a).sum(), np.isnan(b).sum()))
        d = np.where(np.isnan(a), 0.0, np.abs(a - b))
        if scale == 'rel_std':
            with np.errstate(invalid='ignore'):
                s = np.nanstd(b, axis=0, ddof=1)
            s[~np.isfinite(s) | (s == 0)] = 1.0
            d = d / s
        worst = max(worst, float(d.max()) if d.size else 0.0)
    return worst


class Reporter:
    def __init__(self, keep_going=False):
        self.lines = []
        self.keep_going = keep_going
        self.failed = []
        self.roundoff = []
        self.gaps = []

    def step(self, fold, name, dev, ok, note=''):
        status = ok if isinstance(ok, str) else ('PASS' if ok else 'FAIL')
        s = 'fold %2d  %-34s  max dev %-10s %-8s %s' % (fold, name, '%.3g' % dev if dev is not None else '-',
                                                        status, note)
        print(s, flush=True)
        self.lines.append((fold, name, dev, status, note))
        if status == 'ROUNDOFF':
            self.roundoff.append((fold, name.split()[0]))
        if status == 'FAIL':
            self.failed.append((fold, name))
            if not self.keep_going:
                raise GateFail('%s fold %d' % (name, fold))


def cfg_grid(T):
    return np.asarray(T['lambda_grid1'], dtype=float)


def gate_fold(R, f, T, fa_py, sess, dump, s2, split='random'):
    post = dump['post_info']
    st1 = dump['stage1_info']
    ok = sess.fa_fold is None or split != 'random' or np.array_equal(fa_py, sess.fa_fold)
    tr_ref = matio.vec(matio.get_field(post, 'train_trial_indices')).astype(int)
    te_ref = matio.vec(matio.get_field(post, 'test_trial_indices')).astype(int)
    ok = ok and np.array_equal(tr_ref, T['tr_idx']) and np.array_equal(te_ref, T['te_idx'])
    R.step(f, '1 fa.fold / tr_idx / te_idx', 0.0 if ok else None, ok,
           '(n_tr %d n_te %d)' % (len(T['tr_idx']), len(T['te_idx'])))
    d = cmp_cells(T['Y_te'], _cells(dump['Y_true']))
    note = ''
    if s2 is not None:
        d = max(d, cmp_cells(T['Y_tr'], _cells(s2['Y_train'])))
    else:
        note = '(Y_train not checked: no stage-2 dump)'
    R.step(f, '2 Y_delta (test + train)', d, d == 0, note)
    fi = matio.get_field(st1, 'feature_info')
    mu_ref = matio.vec(matio.get_field(fi, 'feature_mu'))
    sd_ref = matio.vec(matio.get_field(fi, 'feature_sd'))
    dmu = np.max(np.abs(T['feature_mu'] - mu_ref) / np.maximum(np.abs(mu_ref), sd_ref))
    dsd = np.max(np.abs(T['feature_sd'] - sd_ref) / sd_ref)
    B_ref = np.asarray(matio.get_field(fi, 'basis'), dtype=float)
    bas_ok = B_ref.shape == T['basis'].shape and np.array_equal(B_ref, T['basis'])
    R.step(f, '3 feature_mu / feature_sd (rel)', max(dmu, dsd), max(dmu, dsd) <= 1e-12 and bas_ok,
           'mu %.3g sd %.3g; basis(:,1:K) exact %s' % (dmu, dsd, bas_ok))
    am = matio.get_field(st1, 'angle_model')
    cv_ref = matio.vec(matio.get_field(am, 'cv_mse'))
    lam_ref = matio.scalar(matio.get_field(am, 'lambda'))
    fin = np.isfinite(cv_ref)
    nan_ok = np.array_equal(np.isnan(cv_ref), np.isnan(T['cv_mse']))
    dcv = float(np.max(np.abs(T['cv_mse'][fin] - cv_ref[fin]) / np.abs(cv_ref[fin]))) if fin.any() else 0.0
    W_ref = np.asarray(matio.get_field(am, 'W'), dtype=float)
    fw = np.isfinite(W_ref)
    dW = float(np.max(np.abs(T['W1'][fw] - W_ref[fw])) / np.max(np.abs(W_ref[fw]))) if fw.any() else 0.0
    lam_ok = T['lambda1'] == lam_ref and nan_ok
    rel = np.abs(T['cv_mse'] - cv_ref) / np.abs(cv_ref)
    print('          stage-1 cv_mse rel dev per lambda %s: %s; cond(refit A) %.2g' % (
        list(cfg_grid(T)), ' '.join('%.1e' % x for x in rel), T.get('cond_refit', np.nan)), flush=True)
    lo, hi = rel[np.argmin(cfg_grid(T))], rel[np.argmax(cfg_grid(T))]
    roundoff = bool(lam_ok and np.isfinite(lo) and (hi <= 1e-10 or lo / max(hi, 1e-300) >= 1e3))
    R.roundoff4 = roundoff or bool(lam_ok and dcv <= 1e-10)
    note = 'lambda %g; W rel %.2g (loose)' % (lam_ref, dW)
    if dcv <= 1e-10 and lam_ok:
        st = 'PASS'
    elif roundoff and dcv < 1e-5:
        st = 'ROUNDOFF'
        note += '; dev falls %.1g-fold from lambda min to max (conditioning)' % (lo / max(hi, 1e-300))
    else:
        st = 'FAIL'
        note += '; %s; lambda py %g ref %g' % ('perm/design error' if dcv >= 1e-6 and not roundoff else 'unexplained',
                                               T['lambda1'], lam_ref)
    R.step(f, '4 stage-1 cv_mse (rel) + lambda', dcv, st, note)
    dA = cmp_cells(T['A_te'], _cells(dump['A_pred']), 'rel_std')
    note = ''
    dF = 0.0
    if s2 is not None:
        dF = cmp_cells(T['F_oof'], _cells(s2['F_oof']), 'rel_std')
    else:
        note = '(F_oof not checked: no stage-2 dump)'
    m5 = max(dA, dF)
    st = 'PASS' if m5 <= 1e-9 else ('ROUNDOFF' if R.roundoff4 and m5 < 1e-3 else 'FAIL')
    R.step(f, '5 A_te / F_oof (rel per-joint std)', m5, st, 'A_te %.3g F_oof %.3g %s' % (dA, dF, note))
    cs_ref = matio.vec(matio.get_field(post, 'cv_score'))
    sl_ref = matio.scalar(matio.get_field(post, 'selected_lambda'))
    nv_ref = matio.vec(matio.get_field(post, 'n_valid'))
    fo_ref = matio.vec(matio.get_field(post, 'fit_ok')).astype(bool)
    fin = np.isfinite(cs_ref)
    dcs = float(np.max(np.abs(T['cv_score'][fin] - cs_ref[fin]))) if fin.any() else 0.0
    struct_ok = (np.array_equal(np.isnan(cs_ref), np.isnan(T['cv_score'])) and T['selected_lambda'] == sl_ref
                 and np.array_equal(nv_ref, T['n_valid']) and np.array_equal(fo_ref, T['fit_ok']))
    v = np.sort(cs_ref[fin])
    gap = v[-1] - v[-2] if v.size > 1 else np.nan
    R.gaps.append(gap)
    if struct_ok and dcs <= 1e-9:
        st = 'PASS'
    elif struct_ok and R.roundoff4 and dcs < 0.1 * gap:
        st = 'ROUNDOFF'
    else:
        st = 'FAIL'
    R.step(f, '6 stage-2 cv_score (abs) + lambda', dcs, st, 'selected %g (ref %g); top-2 gap %.3g; n_valid %s' % (
        T['selected_lambda'], sl_ref, gap, list(nv_ref.astype(int))))
    mref = matio.cell_list(matio.get_field(post, 'models'))
    dw = 0.0
    for j, m in enumerate(T['models']):
        if m is None:
            continue
        wr = matio.vec(matio.get_field(mref[j], 'w'))
        dw = max(dw, float(np.max(np.abs(m['w'] - wr)) / np.max(np.abs(wr))))
    dC = cmp_cells(T['C_te'], _cells(dump['C_pred']), 'rel_std')
    st = 'PASS' if dC <= 1e-9 else ('ROUNDOFF' if R.roundoff4 and dC < 1e-3 else 'FAIL')
    R.step(f, '7 C_te (rel per-joint std)', dC, st, 'FIR w rel %.2g (loose)' % dw)
    tr_rows = dump['trial_rows']
    ref = {}
    fields = {k: matio.cell_list(tr_rows[k]) for k in ('trial_idx', 'joint', 'model', 'rpt_r')}
    for ti, jn, mo, r in zip(fields['trial_idx'], fields['joint'], fields['model'], fields['rpt_r']):
        if matio.to_str(mo) == 'calib':
            ref[(int(matio.scalar(ti)), matio.to_str(jn))] = matio.scalar(r)
    dmax = 0.0
    nan_mismatch = 0
    per_trial_py, per_trial_ref = {}, {}
    for row in T['rows']:
        k = (row[2], row[4])
        a, b = row[7], ref[k]
        if np.isnan(a) != np.isnan(b):
            nan_mismatch += 1
        elif np.isfinite(a):
            dmax = max(dmax, abs(a - b))
        per_trial_py.setdefault(row[2], []).append(a)
        per_trial_ref.setdefault(row[2], []).append(b)
    with np.errstate(all='ignore'):
        dm = max(abs(np.nanmean(per_trial_py[t]) - np.nanmean(per_trial_ref[t])) for t in per_trial_py
                 if np.isfinite(per_trial_ref[t]).any())
    R.step(f, '8 per-trial calib rpt_r (abs)', dmax, dmax <= 1e-6 and nan_mismatch == 0 and len(ref) == len(T['rows']),
           'per-trial nanmean dev %.3g; NaN mismatches %d; rows %d/%d' % (dm, nan_mismatch, len(T['rows']), len(ref)))
    return dmax


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('export')
    ap.add_argument('basis')
    ap.add_argument('trial_dump_dir')
    ap.add_argument('stage2_dir')
    ap.add_argument('--mode', default='exact', choices=['exact', 'fast'])
    ap.add_argument('--config', default=None)
    ap.add_argument('--folds', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--keep-going', action='store_true', help='report every step instead of stopping at the first FAIL')
    ap.add_argument('--write-rows', default=None, help='dir: also write the Python per-trial rows (<label>_trial_r.csv)')
    ap.add_argument('--rest-export', default=None, help='input_control=rest_test_only: <label>_rest.mat (rest-window export)')
    ap.add_argument('--cross-export', default=None, help='input_control=cross_task: cross-condition export')
    a = ap.parse_args(argv)
    cfg = Config.from_json(a.config) if a.config else Config()
    cfg.mode = a.mode
    sess = decoder.Session.from_export(a.export, baseline_mode=cfg.eeg_baseline_mode, label=a.label)
    basisK = decoder.load_basis(resolve_basis(a.basis, sess), cfg.pca_k)
    if cfg.input_control == 'rest_test_only':
        sess.attach_rest(a.rest_export, cfg.stage1_lags(sess.fs))
        print('rest export %s' % sess.rest_meta, flush=True)
    if cfg.input_control == 'cross_task':
        sess.cross = decoder.Session.from_export(a.cross_export, baseline_mode=cfg.eeg_baseline_mode, label=a.label)
    lab = re.sub('[^A-Za-z0-9_]+', '_', sess.label)
    avail = sorted(int(re.search(r'_fold(\d+)_trial_dump\.mat$', p).group(1))
                   for p in glob.glob(os.path.join(a.trial_dump_dir, '%s_fold*_trial_dump.mat' % lab)))
    folds = [int(x) for x in a.folds.split(',')] if a.folds else avail
    print('gate %s mode=%s folds=%s dump=%s' % (sess.label, cfg.mode, folds, a.trial_dump_dir), flush=True)
    fs = sess.fs
    lags1, lags2 = cfg.stage1_lags(fs), cfg.stage2_lags(fs)
    R = Reporter(a.keep_going)
    try:
        fa = decoder.outer_folds(sess, cfg)
        fa_main = fa
        if cfg.input_control == 'cross_task':
            fa = np.concatenate([fa_main, decoder.outer_folds(sess.cross, cfg)])
    except AssertionError as e:
        print('fold  -  1 fa.fold   FAIL', e)
        return 1
    sess_run = decoder.combine_cross(sess, sess.cross) if cfg.input_control == 'cross_task' else sess
    crop, Tk = sess_run.geometry(lags1)
    worst = 0.0
    all_rows = []
    try:
        for f in folds:
            out = decoder.run_fold(sess_run, basisK, cfg, f, fa, lags1, lags2, crop, Tk, trace=True)
            T = out[2]
            dump = matio.loadmat_any(os.path.join(a.trial_dump_dir, '%s_fold%d_trial_dump.mat' % (lab, f)))
            s2 = None
            if a.stage2_dir != '-':
                p2 = os.path.join(a.stage2_dir, '%s_fold%d_stage2.mat' % (lab, f))
                if os.path.exists(p2):
                    s2 = matio.loadmat_any(p2)
            all_rows.extend(T['rows'])
            worst = max(worst, gate_fold(R, f, T, fa_main, sess, dump, s2, cfg.split))
    except GateFail as e:
        print('GATE FAIL at %s (mode %s)' % (e, cfg.mode))
        return 1
    from pyport.run_session import peak_rss_mb
    if a.write_rows:
        import csv
        os.makedirs(a.write_rows, exist_ok=True)
        with open(os.path.join(a.write_rows, '%s_trial_r.csv' % lab), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['session', 'fold', 'trial_idx', 'test_ordinal', 'joint', 'model', 'r', 'rpt_r', 'n_samples',
                        'first_finite_time_sec'])
            for r in all_rows:
                w.writerow(list(r[:6]) + ['%.17g' % r[6], '%.17g' % r[7], r[8], '%.10g' % r[9]])
    if R.failed:
        print('min stage-2 top-2 gap over gated folds %.3g' % np.nanmin(R.gaps) if R.gaps else '')
        print('GATE FAIL (keep-going) %s mode=%s: failed steps %s; max per-trial |d rpt_r| %.3g; peak RSS %.0f MB' % (
            sess.label, cfg.mode, R.failed, worst, peak_rss_mb()))
        return 1
    print('min stage-2 top-2 gap over gated folds %.3g; ROUNDOFF-classified steps %s' % (
        np.nanmin(R.gaps) if R.gaps else np.nan, R.roundoff))
    print('GATE PASS%s %s mode=%s: %d folds, max per-trial |d rpt_r| %.3g; solver %s; peak RSS %.0f MB' % (
        ' (steps 4-7 roundoff-limited)' if R.roundoff else '', sess.label, cfg.mode, len(folds), worst,
        decoder.SOLVER_LOG, peak_rss_mb()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
