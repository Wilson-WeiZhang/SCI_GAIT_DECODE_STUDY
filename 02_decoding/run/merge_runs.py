"""Decoder — run drivers (02_decoding/run)

Merged decoder output of one run: every fold of every session comes from its own decode job (one row of
decodes.csv; job outputs <session>_trial_r.csv, _dump.h5, _run.json), assembled with the functions of
fold_tables/merge_dumps.py. Gates per session, every one required: the rows cover exactly the folds of the
trial layout; per fold the trial set equals the layout; the fold of every tested trial equals the export's
fold map; one decoder, configuration and basis per session; the written files equal the job outputs; for
every trial and joint the correlation of the merged C_pred with Y_true equals the merged per-trial r to 1e-6;
rows, sample counts and the NaN pattern of r equal the layout (zero-input control: a 0/0 correlation may be
NaN on one side). Usage: python merge_runs.py <run|all|link> [--sessions s1,s2] [--force]

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import csv
import json
import os
import shutil
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fold_tables"))
sys.path.insert(0, os.path.dirname(HERE))
import run_paths as P
from merge_dumps import read_trial_r, load_dump, write_dump, same_fold, recon_gate
from pyport.decoder import block_folds

DECODES = os.environ.get("MAIN_DECODES_CSV", os.path.join(HERE, "decodes.csv"))
EXPORTS_CSV = os.environ.get("MAIN_EXPORTS_CSV", os.path.join(HERE, "exports.csv"))
RUNS = ("iw", "post_operative", "pre_operative", "rw", "block", "cad", "resth", "pcirc", "pzero", "prest", "xme", "xem")
IC = {"cad": "circular_shift", "pcirc": "circular_shift", "pzero": "zero_eeg", "resth": "rest_test_only",
      "prest": "rest_test_only", "xme": "cross_task", "xem": "cross_task"}


def h5str(ds):
    return "".join(chr(c) for c in np.asarray(ds[()]).ravel() if c)


def export_map(path):
    with h5py.File(path, "r") as h:
        return (np.asarray(h["fa_fold"][()]).ravel().astype(int), np.asarray(h["Tk"][()]).ravel().astype(int),
                int(np.asarray(h["n_ev"][()]).ravel()[0]))


def merge_session(run, s, rows, ref_dir, tmp, E):
    rt, rd_, rj = (os.path.join(ref_dir, s + x) for x in ("_trial_r.csv", "_dump.h5", "_run.json"))
    ot, od, oj = (os.path.join(tmp, s + x) for x in ("_trial_r.csv", "_dump.h5", "_run.json"))
    head, tb = read_trial_r(rt)
    info, fails = dict(session=s), []
    if not tb:
        if rows:
            return info, [" has no rows but %d decode rows exist" % len(rows)]
        shutil.copyfile(rt, ot)
        info["note"] = "header-only as in the zero-phase run (no rest window: events, lags and seed unchanged)"
        return info, []
    folds = sorted(tb)
    byfold = {int(r["fold"]): r for r in rows}
    if sorted(byfold) != folds or len(byfold) != len(rows):
        return info, ["rows cover folds %s, layout folds %s" % (sorted(byfold), folds)]
    at14, per14, grids14 = load_dump(rd_)
    lines, per, src, at0, basis_md5 = [], {}, [], None, set()
    for fd in folds:
        r = byfold[fd]
        jd = os.path.join(P.DMI, P.JOBS, r["claim_key"], r["decode_id"])
        h2, tj = read_trial_r(os.path.join(jd, s + "_trial_r.csv"))
        if h2 != head or list(tj) != [fd]:
            fails.append("fold %d: header / folds of the job %s" % (fd, list(tj)))
            continue
        lines += tj[fd]
        jat, jper, jgrids = load_dump(os.path.join(jd, s + "_dump.h5"))
        if list(jper) != [fd]:
            fails.append("fold %d: dump folds %s" % (fd, list(jper)))
            continue
        if jper[fd]["trials"] != per14[fd]["trials"]:
            fails.append("G-a fold %d trial set" % fd)
        for k in grids14:
            if not np.array_equal(grids14[k], jgrids[k]):
                fails.append("fold %d grid %s" % (fd, k))
        for k in ("session", "joints", "fs", "split", "input_control"):
            if str(jat[k]) != str(at14[k]):
                fails.append("fold %d attr %s %s != zero-phase run %s" % (fd, k, jat[k], at14[k]))
        cfg = json.loads(str(jat["config_json"]))
        if not (str(jat["decoder_md5"]) == P.DECODER_MD5 and cfg.get("pca_k") == 30 and cfg.get("stage1_lpf_hz") == 6.0
                and cfg.get("input_control") == IC.get(run, "full")
                and cfg.get("split") == ("block" if run == "block" else "random")):
            fails.append("G-cfg fold %d: decoder %s cfg %s" % (fd, jat["decoder_md5"], cfg))
        rdy = open(os.path.join(P.DMI, P.EXPORTS, r["export_id"], "EXPORT_READY")).read().split()[1]
        if str(jat["export_md5"]) != rdy:
            fails.append("G-cfg fold %d export md5 %s != EXPORT_READY %s" % (fd, jat["export_md5"], rdy))
        basis_md5.add(str(jat["basis_md5"]))
        per[fd] = jper[fd]
        at0 = at0 or jat
        src.append(dict(fold=fd, decode_id=r["decode_id"], claim_key=r["claim_key"], export_id=r["export_id"],
                        table_id=r["table_id"], export_path=str(jat["export_path"]), export_md5=str(jat["export_md5"]),
                        basis_md5=str(jat["basis_md5"]), decoder_md5=str(jat["decoder_md5"]),
                        extra=str(jat.get("rest_export", "") or jat.get("cross_export", ""))))
    if len(basis_md5) > 1:
        fails.append("G-cfg several basis md5 %s" % basis_md5)
    if fails:
        return info, fails
    fa, Tk, n_ev = export_map(os.path.join(P.DMI, P.EXPORTS, byfold[folds[0]]["export_id"],
                                           E[byfold[folds[0]]["export_id"]]["out_file"]))
    tested = {}
    for fd in folds:
        for ti, _, _ in per[fd]["trials"]:
            tested[ti] = fd
    if run in ("xme", "xem"):
        ok_map = set(tested.values()) == {1}
    else:
        fm = block_folds(n_ev, 10) if run == "block" else fa
        ok_map = all(fm[ti - 1] == fd for ti, fd in tested.items())
        if run not in ("resth", "prest"):
            ok_map = ok_map and set(tested) == {t + 1 for t in range(n_ev) if Tk[t] > 0}
    if not ok_map:
        fails.append("G-map")
    at2 = dict(at0)
    first = src[0]
    ws = [x for x in src if x["table_id"].endswith("__ws")]
    at2["export_path"], at2["export_md5"] = (ws or [first])[0]["export_path"], (ws or [first])[0]["export_md5"]
    at2["fold_sources_all"] = json.dumps(src)
    at2["_build_id"] = P.TAG
    write_dump(od, at2, per, grids14, folds)
    _, wper, _ = load_dump(od)
    if not all(same_fold(wper[fd], per[fd]) for fd in folds):
        fails.append("G-c dump arrays")
    with open(ot + ".tmp", "w", newline="") as fh:
        fh.write(head + "\n" + "\n".join(lines) + "\n")
    os.replace(ot + ".tmp", ot)
    _, wt = read_trial_r(ot)
    for fd in folds:
        jd = os.path.join(P.DMI, P.JOBS, byfold[fd]["claim_key"], byfold[fd]["decode_id"])
        if wt[fd] != read_trial_r(os.path.join(jd, s + "_trial_r.csv"))[1][fd]:
            fails.append("G-c trial_r fold %d" % fd)
    rjs = [json.load(open(os.path.join(P.DMI, P.JOBS, byfold[fd]["claim_key"], byfold[fd]["decode_id"], s + "_run.json")))
           for fd in folds]
    rj = dict(rjs[0])
    rj["folds"] = [f for j in rjs for f in j["folds"]]
    rj["export"] = "per fold: fold_sources_all"
    rj["fold_sources_all"] = src
    rj["_build_id"] = P.TAG
    with open(oj + ".tmp", "w") as fh:
        json.dump(rj, fh, indent=1)
    os.replace(oj + ".tmp", oj)
    ok, rec = recon_gate(od, ot)
    info["G-d"] = rec
    if not ok:
        fails.append("G-d %s" % rec)
    a, b = pd.read_csv(ot), pd.read_csv(rt)
    key = ["fold", "trial_idx", "test_ordinal", "joint", "model"]
    m = a.merge(b, on=key, suffixes=("", "_14"), how="outer", indicator=True)
    same = bool((m._merge == "both").all()) and len(a) == len(b)
    struct_same = same and bool((m.n_samples == m.n_samples_14).all()
                                and np.allclose(m.first_finite_time_sec, m.first_finite_time_sec_14, equal_nan=True))
    nan_diff = m.rpt_r.isna() != m.rpt_r_14.isna()
    if run == "pzero" and nan_diff.any():
        fin = np.where(m.rpt_r.isna(), m.rpt_r_14, m.rpt_r)[nan_diff.to_numpy()]
        degenerate = bool(np.all(np.abs(fin) <= 1e-12))
        info["nan_pattern_degenerate_cells"] = int(nan_diff.sum())
        info["nan_pattern_degenerate_max_abs_r"] = float(np.max(np.abs(fin)))
        nan_same = struct_same and degenerate
    else:
        nan_same = struct_same and not nan_diff.any()
    if not nan_same:
        fails.append("G-nan rows / n_samples / NaN pattern differ from the zero-phase run")
    d = (m.rpt_r - m.rpt_r_14).abs()
    info.update(rows=len(a), folds=len(folds), max_abs_d_rpt_r_vs_current=float(np.nanmax(d)) if d.notna().any() else 0.0,
                median_abs_d_rpt_r_vs_current=float(np.nanmedian(d)) if d.notna().any() else 0.0,
                max_abs_d_r_vs_current=float(np.nanmax((m.r - m.r_14).abs())) if d.notna().any() else 0.0)
    return info, fails


def link_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        a, b = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isfile(a) and not f.startswith("_"):
            if os.path.exists(b):
                os.remove(b)
            os.link(a, b)
            n += 1
    return n


def main():
    os.chdir(P.DMI)
    what = sys.argv[1]
    only = sys.argv[sys.argv.index("--sessions") + 1].split(",") if "--sessions" in sys.argv else None
    if what == "link":
        print("h_iw_10fold_pca30 = links of h_iw_dump: %d files" % link_dir(os.path.join(P.R, "h_iw_dump"),
                                                                             os.path.join(P.R, "h_iw_10fold_pca30")))
        d = os.path.join(P.R, "basis_hglobal_cdm_pca30")
        os.makedirs(d, exist_ok=True)
        f = "shared_pca_basis.mat"
        if not os.path.exists(os.path.join(d, f)):
            os.link(os.path.join(P.BASIS_HG, f), os.path.join(d, f))
        print("basis_hglobal_cdm_pca30 = link of", os.path.join(P.BASIS_HG, f))
        return
    runs = RUNS if what == "all" else (what,)
    D = pd.read_csv(DECODES, keep_default_na=False, dtype=str)
    E = {r["export_id"]: r for r in csv.DictReader(open(EXPORTS_CSV, newline=""))}
    rc = 0
    for run in runs:
        g = D[D.run == run]
        target = g.target.iloc[0]
        if only:
            g = g[g.session.isin(only)]
        notdone = sorted(k for k in set(g.claim_key) if not os.path.exists(os.path.join(P.Q, "done", k)))
        if notdone:
            print("%s: %d claim keys not done yet (e.g. %s) -> not merged" % (run, len(notdone), notdone[:3]))
            rc |= 3
            continue
        out = os.path.join(P.R, target)
        if os.path.exists(out):
            if "--force" not in sys.argv:
                print("%s: %s exists -> skipped (--force to rebuild)" % (run, out))
                continue
        ref = os.path.join(P.R14, target)
        sessions = sorted(f[:-len("_trial_r.csv")] for f in os.listdir(ref) if f.endswith("_trial_r.csv"))
        if only:
            sessions = [s for s in sessions if s in only]
        tmp = out + ".tmp_merge"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        report, failed = [], []
        for s in sessions:
            info, f = merge_session(run, s, g[g.session == s].to_dict("records"), ref, tmp, E)
            report.append(info)
            failed += [(s, x) for x in f]
            print("  %s %s %s" % (s, "ok" if not f else "FAIL %s" % f[:2],
                                  "" if "G-d" not in info else "G-d max %.1e | rows %d | max|d rpt_r| vs the zero-phase run %.4f" % (
                                      info["G-d"]["max_abs_dev"], info["rows"], info["max_abs_d_rpt_r_vs_zero_phase_prep"])), flush=True)
        dmax = max((r.get("max_abs_d_rpt_r_vs_zero_phase_prep", 0.0) for r in report), default=0.0)
        summ = dict(run=run, target=target, build_id=P.TAG, sessions=len(sessions), rows=sum(r.get("rows", 0) for r in report),
                    folds=sum(r.get("folds", 0) for r in report), failed=[list(map(str, x)) for x in failed],
                    max_abs_d_rpt_r_vs_current=dmax, decodes_csv=os.path.abspath(DECODES), report=report,
                    date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(os.path.join(tmp, "_merge.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        print("MERGE %s %s -> %s: sessions %d, folds %d, rows %d, failures %d, max |d rpt_r| vs the zero-phase run %.3g" % (
            "PASS" if not failed else "FAIL", run, target, len(sessions), summ["folds"], summ["rows"], len(failed), dmax))
        if failed:
            print(failed[:10])
            rc |= 2
            continue
        if os.path.exists(out):
            os.rename(out, out + ".old_" + time.strftime("%Y%m%d_%H%M%S"))
        os.rename(tmp, out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
