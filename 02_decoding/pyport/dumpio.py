"""Decoder — Python readout (02_decoding/pyport)

Readers for the per-session HDF5 dumps written by run_session.py --dump

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import csv
import os

import h5py
import numpy as np


def load_session(path):
    """dump.h5 -> dict(session, fs, joints, folds={fold: dict(trials={trial_idx: dict(Y, C, A, time_sec)},
    train_idx=[...], test_idx=[...])}); arrays (T x J) as stored."""
    with h5py.File(path, 'r') as f:
        sess = f.attrs['session']
        fs = float(f.attrs['fs'])
        joints = f.attrs['joints'].split(',')
        tr = {k: f['trials/' + k][()] for k in f['trials']}
        Y, C, A, T = (f['rows/' + k][()] for k in ('Y_true', 'C_pred', 'A_pred', 'time_sec'))
        fo = {k: f['folds/' + k][()] for k in f['folds']}
    folds = {}
    for i, fd in enumerate(fo['fold']):
        folds[int(fd)] = dict(trials={},
                              train_idx=fo['train_idx'][fo['train_offset'][i]:fo['train_offset'][i + 1]].tolist(),
                              test_idx=fo['test_idx'][fo['test_offset'][i]:fo['test_offset'][i + 1]].tolist())
    for i in range(tr['fold'].size):
        a, n = tr['row_offset'][i], tr['n_rows'][i]
        folds[int(tr['fold'][i])]['trials'][int(tr['trial_idx'][i])] = dict(
            Y=Y[a:a + n], C=C[a:a + n], A=A[a:a + n], time_sec=T[a:a + n])
    return dict(session=sess, fs=fs, joints=joints, folds=folds)


def calib_rpt(trial_r_csv, labels):
    """{(trial_idx, joint index in labels): calib rpt_r} from a pyport <label>_trial_r.csv."""
    out = {}
    with open(trial_r_csv) as fh:
        for r in csv.DictReader(fh):
            if r['model'] != 'calib':
                continue
            v = r['rpt_r']
            out[(int(r['trial_idx']), labels.index(r['joint']))] = float('nan') if v in ('', 'nan', 'NaN') else float(v)
    return out


def read_fold_like_matlab(dump_h5, fold, labels, _cache={}):
    """One fold of a pyport dump as the dict of the MATLAB per-fold trial dump:
    dict(session, fold, fs, trials={trial_idx: dict(Y (J x T), C (J x T))}, train_idx, rpt)."""
    if dump_h5 not in _cache:
        S = load_session(dump_h5)
        if S['joints'] != list(labels):
            raise RuntimeError('%s: joints %s differ from %s' % (dump_h5, S['joints'], labels))
        S['rpt'] = calib_rpt(dump_h5[:-len('_dump.h5')] + '_trial_r.csv', list(labels))
        _cache.clear()
        _cache[dump_h5] = S
    S = _cache[dump_h5]
    fd = S['folds'][int(fold)]
    trials = {ti: dict(Y=np.ascontiguousarray(d['Y'].T), C=np.ascontiguousarray(d['C'].T))
              for ti, d in fd['trials'].items()}
    rpt = {k: v for k, v in S['rpt'].items() if k[0] in trials}
    return dict(session=S['session'], fold=int(fold), fs=S['fs'], trials=trials, train_idx=list(fd['train_idx']),
                rpt=rpt)


def pseudo_fold_paths(dump_dir):
    """One pseudo per-fold path '<dump_dir>/<session>_fold<f>_trial_dump.mat' per (session, fold) of every
    <session>_dump.h5 in dump_dir (what the MATLAB-dump readers glob for)."""
    out = []
    for fn in sorted(os.listdir(dump_dir)):
        if not fn.endswith('_dump.h5'):
            continue
        with h5py.File(os.path.join(dump_dir, fn), 'r') as f:
            folds = sorted(set(int(x) for x in f['folds/fold'][()]))
        sess = fn[:-len('_dump.h5')]
        out += [os.path.join(dump_dir, '%s_fold%d_trial_dump.mat' % (sess, k)) for k in folds]
    return out
