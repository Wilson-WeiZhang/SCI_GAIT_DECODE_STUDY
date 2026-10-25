"""Decoder — Python readout (02_decoding/pyport)

Target-label-shuffle null computed from the per-session HDF5 dumps

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

MIN_VALID = 10


def safe_corr(y, yp):
    v = np.isfinite(y) & np.isfinite(yp)
    if v.sum() < MIN_VALID:
        return np.nan
    a, b = y[v], yp[v]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cross_r_matrix(y_cells, c_cells, n_joints):
    n = len(y_cells)
    R = np.full((n_joints, n, n), np.nan)
    for b in range(n):
        cb = c_cells[b]
        if cb.size == 0:
            continue
        for j in range(n_joints):
            if cb.shape[1] <= j:
                continue
            for a in range(n):
                ya = y_cells[a]
                m = min(ya.shape[0], cb.shape[0])
                if m < 2 or ya.shape[1] <= j:
                    continue
                R[j, a, b] = safe_corr(ya[:m, j], cb[:m, j])
    return R


def permutations_for(n, base_seed, group_id, n_perm):
    if n < 2:
        return [np.arange(n)] * n_perm
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(group_id), 20260905]))
    ident = np.arange(n)
    out = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        if np.array_equal(p, ident):
            p = np.roll(ident, -1)
        out.append(p)
    return out


def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def read_groups(path, scope):
    """-> session, joints, list of (group_id, n_trials, y_cells (T,J), c_cells (T,J)) in dump trial order."""
    with h5py.File(path, 'r') as f:
        sess = f.attrs['session']
        joints = f.attrs['joints'].split(',')
        fold = f['trials/fold'][()]
        off = f['trials/row_offset'][()]
        nr = f['trials/n_rows'][()]
        Y = f['rows/Y_true'][()]
        C = f['rows/C_pred'][()]
    cells = [(int(fold[i]), Y[off[i]:off[i] + nr[i]], C[off[i]:off[i] + nr[i]]) for i in range(fold.size)]
    if scope == 'session':
        return sess, joints, [(0, [y for _, y, _ in cells], [c for _, _, c in cells])]
    groups = []
    for g in sorted(set(x[0] for x in cells)):
        groups.append((g, [y for f_, y, _ in cells if f_ == g], [c for f_, _, c in cells if f_ == g]))
    return sess, joints, groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--scope', default='fold', choices=['fold', 'session'])
    ap.add_argument('--seed', type=int, default=20260905)
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--sessions', default=None)
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(a.dump_dir, '*_dump.h5')))
    if a.sessions:
        keep = set(a.sessions.split(','))
        paths = [p for p in paths if os.path.basename(p)[:-len('_dump.h5')] in keep]
    os.makedirs(a.out_dir, exist_ok=True)
    fold_rows, sess_rows = [], []
    t0 = time.time()
    for p in paths:
        session, joints, groups = read_groups(p, a.scope)
        cells_true, num, den, n_tr = [], np.zeros(a.n_perm), np.zeros(a.n_perm), 0
        for gid, y, c in groups:
            nt = len(y)
            n_tr += nt
            R = cross_r_matrix(y, c, len(joints))
            ident = np.arange(nt)
            true_j = np.array([nanmean(R[j, ident, ident]) for j in range(len(joints))])
            shuf_j = np.full((a.n_perm, len(joints)), np.nan)
            for pi, perm in enumerate(permutations_for(nt, a.seed, gid, a.n_perm)):
                for j in range(len(joints)):
                    shuf_j[pi, j] = nanmean(R[j, perm, ident])
            t_g = nanmean(true_j)
            s_g = np.array([nanmean(shuf_j[pi]) for pi in range(a.n_perm)])
            fin = np.isfinite(s_g)
            p_emp = ((1.0 + int(np.sum(s_g[fin] >= t_g))) / (1.0 + int(fin.sum()))
                     if fin.any() and np.isfinite(t_g) else np.nan)
            fold_rows.append({'session': session, 'fold': gid if a.scope == 'fold' else 'SESSION', 'n_trials': nt,
                              'rpt_calib_true': t_g, 'rpt_calib_shuf_mean': nanmean(s_g),
                              'rpt_calib_shuf_sd': float(np.nanstd(s_g, ddof=1)) if fin.sum() > 1 else np.nan,
                              'rpt_calib_shuf_p95': float(np.nanpercentile(s_g, 95)) if fin.any() else np.nan,
                              'p_empirical': p_emp})
            cells_true.extend([v for v in true_j if np.isfinite(v)])
            fc = np.isfinite(shuf_j)
            num += np.nansum(np.where(fc, shuf_j, 0.0), axis=1)
            den += fc.sum(axis=1)
        t_s = float(np.mean(cells_true)) if cells_true else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            dist = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        fin = np.isfinite(dist)
        sess_rows.append({'session': session, 'fold': 'ALL', 'n_trials': n_tr, 'rpt_calib_true': t_s,
                          'rpt_calib_shuf_mean': nanmean(dist),
                          'rpt_calib_shuf_sd': float(np.nanstd(dist, ddof=1)) if fin.sum() > 1 else np.nan,
                          'rpt_calib_shuf_p95': float(np.nanpercentile(dist, 95)) if fin.any() else np.nan,
                          'p_empirical': ((1.0 + int(np.sum(dist[fin] >= t_s))) / (1.0 + int(fin.sum()))
                                          if fin.any() and np.isfinite(t_s) else np.nan)})
    flds = ['session', 'fold', 'n_trials', 'rpt_calib_true', 'rpt_calib_shuf_mean', 'rpt_calib_shuf_sd',
            'rpt_calib_shuf_p95', 'p_empirical']
    for name, rows in (('per_fold.csv', fold_rows), ('per_session.csv', sess_rows)):
        tmp = os.path.join(a.out_dir, name + '.tmp')
        with open(tmp, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=flds)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, os.path.join(a.out_dir, name))
    meta = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script': os.path.abspath(__file__),
            'dump_dir': os.path.abspath(a.dump_dir), 'scope': a.scope, 'n_perm': a.n_perm, 'seed': a.seed,
            'perm_seed_rule': 'np.random.SeedSequence([seed, group_id, 20260905]); group_id = fold (scope fold) or 0',
            'n_sessions': len(sess_rows),
            'cohort_mean_rpt_calib_true': nanmean([r['rpt_calib_true'] for r in sess_rows]),
            'cohort_mean_rpt_calib_shuffled': nanmean([r['rpt_calib_shuf_mean'] for r in sess_rows])}
    with open(os.path.join(a.out_dir, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('%d sessions, %d groups, cohort true %.6f shuffled %.6f, %.0fs' % (
        len(sess_rows), len(fold_rows), meta['cohort_mean_rpt_calib_true'], meta['cohort_mean_rpt_calib_shuffled'],
        time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
