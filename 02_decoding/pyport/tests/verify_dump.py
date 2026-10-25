"""Decoder — Python readout (02_decoding/pyport)

Recompute every per-trial r of a run from its HDF5 dump and compare it with the run's per-trial table

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import csv
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pyport import decoder

out, lab = sys.argv[1], sys.argv[2]
ref = {}
with open(os.path.join(out, '%s_trial_r.csv' % lab)) as fh:
    for r in csv.DictReader(fh):
        ref[(int(r['fold']), int(r['trial_idx']), r['joint'])] = (float(r['rpt_r']), int(r['n_samples']),
                                                                   float(r['first_finite_time_sec']))
with h5py.File(os.path.join(out, '%s_dump.h5' % lab), 'r') as f:
    joints = f.attrs['joints'].split(',')
    tr = {k: f['trials/' + k][()] for k in f['trials']}
    T, Y, C = f['rows/time_sec'][()], f['rows/Y_true'][()], f['rows/C_pred'][()]
    folds = {k: f['folds/' + k][()] for k in f['folds']}
    print('attrs:', {k: (v if len(str(v)) < 90 else str(v)[:90] + '...') for k, v in f.attrs.items()})
dmax, nan_mis, n = 0.0, 0, 0
for i in range(tr['fold'].size):
    a, m = tr['row_offset'][i], tr['n_rows'][i]
    for j, jn in enumerate(joints):
        r, _, ns, ft = decoder.single_trial_r(Y[a:a + m], C[a:a + m], j, T[a:a + m])
        rr, rns, rft = ref[(int(tr['fold'][i]), int(tr['trial_idx'][i]), jn)]
        n += 1
        if np.isnan(r) != np.isnan(rr) or ns != rns:
            nan_mis += 1
        elif np.isfinite(r):
            dmax = max(dmax, abs(r - rr))
run = json.load(open(os.path.join(out, '%s_run.json' % lab)))
fold_ok = (list(folds['fold']) == [x['fold'] for x in run['folds']]
           and list(folds['lambda_stage2']) == [x['stage2_selected_lambda'] for x in run['folds']]
           and list(folds['n_train']) == [x['n_train'] for x in run['folds']])
print('rows %d/%d; NaN / n_samples mismatches %d; max |rpt_r(dump) - rpt_r(csv)| %.3g; fold table matches run.json %s'
      % (n, len(ref), nan_mis, dmax, fold_ok))
print('VERIFY', 'PASS' if (n == len(ref) and nan_mis == 0 and dmax <= 1e-12 and fold_ok) else 'FAIL')
