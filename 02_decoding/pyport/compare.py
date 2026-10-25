"""Decoder — Python readout (02_decoding/pyport)

Per-trial comparison of two sets of per-trial tables (match, paired differences, selected lambdas)

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np


def load_rows(dirs, sessions=None):
    """(session, trial_idx, joint) -> calib rpt_r (float, NaN allowed). Reads <dir>/<session>[_fold*]_trial_r.csv."""
    out = {}
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, '*_trial_r.csv'))):
            with open(fn) as fh:
                for r in csv.DictReader(fh):
                    if r['model'] != 'calib':
                        continue
                    if sessions and r['session'] not in sessions:
                        continue
                    v = r['rpt_r']
                    x = float('nan') if v in ('', 'NaN', 'nan') else float(v)
                    out[(r['session'], int(r['trial_idx']), r['joint'])] = x
    return out


def per_trial(rows):
    acc = defaultdict(list)
    for (s, t, _), v in rows.items():
        if np.isfinite(v):
            acc[(s, t)].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def cmd_match(a):
    ses = a.sessions.split(',') if a.sessions else None
    py = load_rows(a.py.split(','), ses)
    ref = load_rows(a.ref.split(','), ses)
    keys = sorted(set(py) & set(ref))
    d_row, nan_mis = 0.0, 0
    for k in keys:
        x, y = py[k], ref[k]
        if np.isnan(x) != np.isnan(y):
            nan_mis += 1
        elif np.isfinite(x):
            d_row = max(d_row, abs(x - y))
    tp, tr = per_trial(py), per_trial(ref)
    tk = sorted(set(tp) & set(tr))
    d_tr = max((abs(tp[k] - tr[k]) for k in tk), default=float('nan'))
    sess = sorted({k[0] for k in tk})
    print('match: rows py %d ref %d common %d; NaN-pattern mismatches %d; max |d rpt_r| per row %.3g' % (
        len(py), len(ref), len(keys), nan_mis, d_row))
    print('       trials py %d ref %d common %d (only py %d, only ref %d); max |d| per-trial mean %.3g' % (
        len(tp), len(tr), len(tk), len(set(tp) - set(tr)), len(set(tr) - set(tp)), d_tr))
    print('       pooled median py %.4f ref %.4f over %d sessions' % (
        np.median([tp[k] for k in tk]), np.median([tr[k] for k in tk]), len(sess)))
    for s in sess:
        ks = [k for k in tk if k[0] == s]
        print('   %-14s n=%3d  py %.4f  ref %.4f  max|d| %.3g' % (s, len(ks), np.median([tp[k] for k in ks]),
                                                              np.median([tr[k] for k in ks]),
                                                              max(abs(tp[k] - tr[k]) for k in ks)))
    ok = nan_mis == 0 and d_row <= 1e-6 and len(set(tp) ^ set(tr)) == 0
    print('MATCH', 'PASS' if ok else 'FAIL', '(per-row |d| <= 1e-6, identical trial set)')
    return 0 if ok else 1


def cmd_paired(a):
    ses = a.sessions.split(',') if a.sessions else None
    A = per_trial(load_rows(a.a.split(','), ses))
    B = per_trial(load_rows(a.b.split(','), ses))
    k = sorted(set(A) & set(B))
    xa = np.array([A[i] for i in k])
    xb = np.array([B[i] for i in k])
    print('%s: n=%d median %.4f | %s: n=%d median %.4f | paired n=%d, median(b - a) %+.4f, b >= a %.1f%%' % (
        a.name_a, len(A), np.median(list(A.values())), a.name_b, len(B), np.median(list(B.values())), len(k),
        np.median(xb - xa), 100 * np.mean(xb >= xa)))
    return 0


def cmd_lambdas(a):
    """Selected stage-1 / stage-2 lambdas per fold: Python <out>/<session>_run.json vs MATLAB <dump>/<s>_fold<f>_trial_dump.mat."""
    import json
    import h5py
    ses = a.sessions.split(',')
    n = bad = 0
    for s in ses:
        jp = os.path.join(a.py, '%s_run.json' % s)
        if not os.path.exists(jp):
            print('%s: no python run json' % s)
            continue
        for fi in json.load(open(jp))['folds']:
            f = fi['fold']
            dp = None
            for d in a.ref.split(','):
                p = os.path.join(d, '%s_fold%d_trial_dump.mat' % (s, f))
                if os.path.exists(p):
                    dp = p
            if dp is None:
                continue
            with h5py.File(dp, 'r') as h:
                l1 = float(np.array(h['post_info/lambda_angle']).ravel()[0])
                l2 = float(np.array(h['post_info/selected_lambda']).ravel()[0])
            n += 1
            if l1 != fi['lambda_stage1'] or l2 != fi['stage2_selected_lambda']:
                bad += 1
                print('  MISMATCH %s fold %d: py (%g, %g) matlab (%g, %g), py top-2 gap %.3g' % (
                    s, f, fi['lambda_stage1'], fi['stage2_selected_lambda'], l1, l2, fi['stage2_top2_gap']))
    print('lambdas: %d folds compared, %d mismatches' % (n, bad))
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    lm = sub.add_parser('lambdas')
    lm.add_argument('--py', required=True)
    lm.add_argument('--ref', required=True)
    lm.add_argument('--sessions', required=True)
    m = sub.add_parser('match')
    m.add_argument('--py', required=True)
    m.add_argument('--ref', required=True)
    m.add_argument('--sessions', default=None)
    p = sub.add_parser('paired')
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--sessions', default=None)
    p.add_argument('--name-a', default='a')
    p.add_argument('--name-b', default='b')
    a = ap.parse_args(argv)
    return {'match': cmd_match, 'paired': cmd_paired, 'lambdas': cmd_lambdas}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
