"""Decoder — Python readout (02_decoding/pyport)

Decode one exported session: python -m pyport.run_session <export.mat> <basis.mat|auto> <configs/....json> <out_dir> [--mode fast] [--folds k,...] [--dump] [--rest-export F] [--cross-export F]; writes <label>_trial_r.csv and <label>_run.json (--folds: those outer folds only, as the second stage runs it)

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyport import decoder
from pyport.config import Config

DECODER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2 ** 20
        except Exception:
            return float('nan')


def md5_file(path, chunk=1 << 24):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def write_dump(path, sess, cfg, kept, infos, export_path, basis_path):
    """Per-session dump mirroring the MATLAB trial dump. HDF5 layout (all numeric float64 unless noted):
    joints (comma-separated), fs
    /trials: fold, trial_idx (1-based position in ev), test_ordinal, row_offset, n_rows  [n_test_total]  (int64)
    /rows:   time_sec [N], Y_true [N x 6], A_pred [N x 6] (stage 1), C_pred [N x 6] (final calib)  - trials stacked
    in /trials order; trial i = rows row_offset[i] : row_offset[i] + n_rows[i]; NaN columns kept
    /folds:  fold, n_train, n_test, lambda_stage1, lambda_stage2 [F]; cv_mse [F x n_lambda1] (stage-1 inner-CV MSE,
    one value per lambda pooled over active joints, as MATLAB model.cv_mse); lambda_grid1 [n_lambda1];
    cv_score [F x n_lambda2], lambda_grid2 [n_lambda2]; n_valid [F x 6]; fit_ok [F x 6] (int8);
    train_idx / test_idx (int64, stacked) with train_offset / test_offset [F+1]
    Written to a temp file and renamed."""
    import h5py
    import numpy as np
    tri, rows = [], {'time_sec': [], 'Y_true': [], 'A_pred': [], 'C_pred': []}
    off = 0
    for T in kept:
        for i, t in enumerate(T['te_idx']):
            n = T['Y_te'][i].shape[0]
            tri.append((T['fold'], int(t), i + 1, off, n))
            off += n
            rows['time_sec'].append(T['time_sec'][i])
            rows['Y_true'].append(T['Y_te'][i])
            rows['A_pred'].append(T['A_te'][i])
            rows['C_pred'].append(T['C_te'][i])
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        at = f.attrs
        at['session'] = sess.label
        at['split'] = cfg.split
        at['config_json'] = json.dumps(cfg.to_dict())
        at['decoder_md5'] = md5_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decoder.py'))
        at['export_path'] = os.path.abspath(export_path)
        at['export_md5'] = md5_file(export_path)
        at['export_cp_md5'] = sess.meta.get('cp_md5', '')
        at['basis_path'] = os.path.abspath(basis_path)
        at['basis_md5'] = md5_file(basis_path)
        at['joints'] = ','.join(sess.joint_labels)
        at['input_control'] = cfg.input_control
        rm = getattr(sess, 'rest_meta', None)
        if rm:
            at['rest_export'] = rm['rest_export']
            at['rest_export_md5'] = md5_file(rm['rest_export'])
            at['rest_seed'] = rm['rest_seed']
        cm = getattr(sess, 'cross_meta', None)
        if cm:
            at['cross_export'] = cm['cross_export']
            at['cross_export_md5'] = md5_file(cm['cross_export'])
        at['fs'] = sess.fs
        g = f.create_group('trials')
        a = np.array(tri, dtype=np.int64).reshape(-1, 5)
        for j, k in enumerate(('fold', 'trial_idx', 'test_ordinal', 'row_offset', 'n_rows')):
            g[k] = a[:, j]
        g = f.create_group('rows')
        g['time_sec'] = np.concatenate(rows['time_sec']).astype(float)
        for k in ('Y_true', 'A_pred', 'C_pred'):
            g[k] = np.vstack(rows[k]).astype(float)
        g = f.create_group('folds')
        g['fold'] = np.array([T['fold'] for T in kept], dtype=np.int64)
        g['n_train'] = np.array([len(T['tr_idx']) for T in kept], dtype=np.int64)
        g['n_test'] = np.array([len(T['te_idx']) for T in kept], dtype=np.int64)
        g['lambda_stage1'] = np.array([T['lambda1'] for T in kept], dtype=float)
        g['lambda_stage2'] = np.array([T['selected_lambda'] for T in kept], dtype=float)
        g['cv_mse'] = np.vstack([T['cv_mse'] for T in kept]).astype(float)
        g['lambda_grid1'] = np.asarray(cfg.lambda_grid, dtype=float)
        g['cv_score'] = np.vstack([T['cv_score'] for T in kept]).astype(float)
        g['lambda_grid2'] = np.asarray(kept[0]['post_lambda_grid'], dtype=float)
        g['n_valid'] = np.vstack([T['n_valid'] for T in kept]).astype(float)
        g['fit_ok'] = np.vstack([T['fit_ok'] for T in kept]).astype(np.int8)
        for k in ('tr_idx', 'te_idx'):
            name = 'train' if k == 'tr_idx' else 'test'
            g[name + '_idx'] = np.concatenate([T[k] for T in kept]).astype(np.int64)
            g[name + '_offset'] = np.r_[0, np.cumsum([len(T[k]) for T in kept])].astype(np.int64)
    os.replace(tmp, path)


def resolve_basis(arg, sess):
    if arg != 'auto':
        return arg
    p = sess.meta.get('basis_path', '')
    if not p:
        raise ValueError('basis auto: export has no basis_path')
    return p if os.path.isabs(p) else os.path.join(DECODER, p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('export')
    ap.add_argument('basis')
    ap.add_argument('config')
    ap.add_argument('out_dir')
    ap.add_argument('--mode', default=None, choices=['exact', 'fast'])
    ap.add_argument('--folds', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--rest-export', default=None,
                    help='input_control=rest_test_only: <label>_rest.mat written by core/run_decoder.m (CP_EXPORT_REST_DIR)')
    ap.add_argument('--cross-export', default=None,
                    help='input_control=cross_task: export of the OTHER condition built with the training condition bands ')
    ap.add_argument('--dump', action='store_true', help='also write <out_dir>/<label>_dump.h5 (predictions + fold info)')
    a = ap.parse_args(argv)
    cfg = Config() if a.config == '-' else Config.from_json(a.config)
    if a.mode:
        cfg.mode = a.mode
    folds = [int(x) for x in a.folds.split(',')] if a.folds else None
    t0 = time.time()
    sess = decoder.Session.from_export(a.export, baseline_mode=cfg.eeg_baseline_mode, label=a.label)
    basis_path = resolve_basis(a.basis, sess)
    if cfg.input_control == 'rest_test_only':
        if not a.rest_export:
            raise SystemExit('input_control=rest_test_only needs --rest-export <label>_rest.mat')
        sess.attach_rest(a.rest_export, cfg.stage1_lags(sess.fs))
    elif cfg.input_control == 'cross_task':
        if not a.cross_export:
            raise SystemExit('input_control=cross_task needs --cross-export <session>_export.mat')
        sess.cross = decoder.Session.from_export(a.cross_export, baseline_mode=cfg.eeg_baseline_mode, label=a.label)
        sess.cross_meta = {'cross_export': os.path.abspath(a.cross_export),
                           'cross_condition_n_ev': sess.cross.n_ev}
    elif cfg.input_control not in ('full', 'circular_shift', 'zero_eeg'):
        raise SystemExit('input_control must be full | rest_test_only | circular_shift | zero_eeg | cross_task')
    if a.rest_export and cfg.input_control != 'rest_test_only':
        raise SystemExit('--rest-export given but input_control is %s' % cfg.input_control)
    if a.cross_export and cfg.input_control != 'cross_task':
        raise SystemExit('--cross-export given but input_control is %s' % cfg.input_control)
    basisK = decoder.load_basis(basis_path, cfg.pca_k)
    t_load = time.time() - t0
    log = lambda s: print('[%s %s] %s' % (sess.label, cfg.mode, s), flush=True)
    rows, infos, kept, geo = decoder.run_session(sess, basisK, cfg, folds=folds, log=log, keep=a.dump)
    t_run = time.time() - t0 - t_load
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, '%s_trial_r.csv' % sess.label)
    tmp = csv_path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['session', 'fold', 'trial_idx', 'test_ordinal', 'joint', 'model', 'r', 'rpt_r', 'n_samples',
                    'first_finite_time_sec'])
        for r in rows:
            w.writerow(list(r[:6]) + ['%.17g' % r[6], '%.17g' % r[7], r[8], '%.10g' % r[9]])
    os.replace(tmp, csv_path)
    if a.dump:
        dump_path = os.path.join(a.out_dir, '%s_dump.h5' % sess.label)
        write_dump(dump_path, sess, cfg, kept, infos, a.export, basis_path)
        print('wrote %s (%.1f MB)' % (dump_path, os.path.getsize(dump_path) / 2 ** 20), flush=True)
    out = {'session': sess.label, 'export': os.path.abspath(a.export), 'basis': os.path.abspath(basis_path),
           'config': cfg.to_dict(), 'mode': cfg.mode, 'n_ev': sess.n_ev,
           'n_valid_trials': int((geo['Tk'] > 0).sum()), 'crop': int(geo['crop']),
           'lags1': [int(x) for x in geo['lags1']], 'lags2': [int(x) for x in geo['lags2']],
           'folds': infos, 'seconds_load': t_load, 'seconds_run': t_run, 'peak_rss_mb': peak_rss_mb(),
           'solver_counts': dict(decoder.SOLVER_LOG),
           'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
           'export_meta': {k: v for k, v in sess.meta.items() if isinstance(v, str)},
           'rest_meta': getattr(sess, 'rest_meta', None), 'cross_meta': getattr(sess, 'cross_meta', None)}
    jp = os.path.join(a.out_dir, '%s_run.json' % sess.label)
    with open(jp + '.tmp', 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(jp + '.tmp', jp)
    print('wrote %s (%d rows); load %.1fs run %.1fs peak RSS %.0f MB' % (csv_path, len(rows), t_load, t_run,
                                                                          out['peak_rss_mb']), flush=True)


if __name__ == '__main__':
    main()
