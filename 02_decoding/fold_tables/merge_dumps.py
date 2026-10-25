"""Decoder — per-fold harmonic band tables (02_decoding/fold_tables)

Merged decoder output of one run: the first-stage output with the re-decoded folds put in.

Only the folds whose training-trial band table differs from the whole-session table are decoded
again (pyport/run_session.py --folds k --dump on that fold's export, one job per fold). For every
session of <base_dir>: without a re-decoded fold its three first-stage files are linked unchanged;
otherwise the session is rebuilt fold by fold, unchanged folds from the first stage and re-decoded
folds from their job (per-trial table lines copied as text, HDF5 dump groups restacked). Gates,
every one required: the trial set of each re-decoded fold equals the first-stage fold; unchanged
folds are bit-identical to the first stage and re-decoded folds to their job (checked on the
written files); for every trial and joint the correlation of the merged C_pred with Y_true equals
the merged per-trial r to 1e-6.
Usage: python merge_dumps.py <base_dir> <out_dir> <fold_map.csv> [--trial-r-only]
  fold_map.csv    columns session, fold, job, job_dir (job_dir holds that job's
                  <session>_trial_r.csv, _dump.h5 and _run.json)
  --trial-r-only  the first-stage run wrote no dumps: per-trial tables only, no correlation gate

Part of the decoder's second stage, the per-fold harmonic band tables (README.md,
Decoding, Step 09); it fits nothing.
"""

import csv
import json
import os
import sys

import h5py
import numpy as np

TOL = 1e-6
BUILD_ID = "zero_phase_prep"


def read_trial_r(path):
    with open(path, newline="") as fh:
        lines = fh.read().split("\n")
    head, body = lines[0], [l for l in lines[1:] if l]
    byfold = {}
    for l in body:
        byfold.setdefault(int(l.split(",")[1]), []).append(l)
    return head, byfold


def load_dump(path):
    with h5py.File(path, "r") as f:
        at = {k: f.attrs[k] for k in f.attrs}
        tr = {k: f["trials/" + k][()] for k in f["trials"]}
        rows = {k: f["rows/" + k][()] for k in f["rows"]}
        fo = {k: f["folds/" + k][()] for k in f["folds"]}
    per = {}
    for i, fd in enumerate(fo["fold"]):
        fd = int(fd)
        sel = np.where(tr["fold"] == fd)[0]
        trials = [(int(tr["trial_idx"][j]), int(tr["test_ordinal"][j]), int(tr["n_rows"][j])) for j in sel]
        sl = {k: [v[tr["row_offset"][j]:tr["row_offset"][j] + tr["n_rows"][j]] for j in sel] for k, v in rows.items()}
        scal = {k: fo[k][i] for k in ("n_train", "n_test", "lambda_stage1", "lambda_stage2")}
        mat = {k: fo[k][i] for k in ("cv_mse", "cv_score", "n_valid", "fit_ok")}
        idx = {k: fo[k + "_idx"][fo[k + "_offset"][i]:fo[k + "_offset"][i + 1]] for k in ("train", "test")}
        per[fd] = dict(trials=trials, rows=sl, scal=scal, mat=mat, idx=idx)
    grids = {k: fo[k] for k in ("lambda_grid1", "lambda_grid2")}
    return at, per, grids


def write_dump(path, at, per, grids, folds):
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k, v in at.items():
            f.attrs[k] = v
        tri, off = [], 0
        rows = {k: [] for k in ("time_sec", "Y_true", "A_pred", "C_pred")}
        for fd in folds:
            P = per[fd]
            for (ti, to, n), j in zip(P["trials"], range(len(P["trials"]))):
                tri.append((fd, ti, to, off, n))
                off += n
                for k in rows:
                    rows[k].append(P["rows"][k][j])
        g = f.create_group("trials")
        a = np.array(tri, dtype=np.int64).reshape(-1, 5)
        for j, k in enumerate(("fold", "trial_idx", "test_ordinal", "row_offset", "n_rows")):
            g[k] = a[:, j]
        g = f.create_group("rows")
        g["time_sec"] = np.concatenate(rows["time_sec"]).astype(float)
        for k in ("Y_true", "A_pred", "C_pred"):
            g[k] = np.vstack(rows[k]).astype(float)
        g = f.create_group("folds")
        g["fold"] = np.array(folds, dtype=np.int64)
        for k, dt in (("n_train", np.int64), ("n_test", np.int64), ("lambda_stage1", float), ("lambda_stage2", float)):
            g[k] = np.array([per[fd]["scal"][k] for fd in folds], dtype=dt)
        for k in ("cv_mse", "cv_score", "n_valid"):
            g[k] = np.vstack([per[fd]["mat"][k] for fd in folds]).astype(float)
        g["fit_ok"] = np.vstack([per[fd]["mat"]["fit_ok"] for fd in folds]).astype(np.int8)
        for k, v in grids.items():
            g[k] = v
        for k in ("train", "test"):
            g[k + "_idx"] = np.concatenate([per[fd]["idx"][k] for fd in folds]).astype(np.int64)
            g[k + "_offset"] = np.r_[0, np.cumsum([len(per[fd]["idx"][k]) for fd in folds])].astype(np.int64)
    os.replace(tmp, path)


def same_fold(A, B):
    if A["trials"] != B["trials"]:
        return False
    for k in A["rows"]:
        if not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(A["rows"][k], B["rows"][k])):
            return False
    for part in ("scal", "mat", "idx"):
        for k in A[part]:
            if not np.array_equal(np.asarray(A[part][k]), np.asarray(B[part][k]), equal_nan=True):
                return False
    return True


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def recon_gate(dump_path, trial_r_path):
    at, per, _ = load_dump(dump_path)
    joints = str(at["joints"]).split(",")
    rpt = {}
    with open(trial_r_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "calib":
                v = r["rpt_r"]
                rpt[(int(r["fold"]), int(r["trial_idx"]), joints.index(r["joint"]))] = \
                    np.nan if v in ("", "nan", "NaN") else float(v)
    worst, n, bad = 0.0, 0, []
    for fd, P in per.items():
        for (ti, _, _), Y, C in zip(P["trials"], P["rows"]["Y_true"], P["rows"]["C_pred"]):
            for j in range(len(joints)):
                m = np.isfinite(Y[:, j]) & np.isfinite(C[:, j])
                r = pearson(Y[m, j], C[m, j]) if m.sum() > 2 else np.nan
                ref = rpt[(fd, ti, j)]
                n += 1
                if np.isnan(ref) or np.isnan(r):
                    if not (np.isnan(ref) and np.isnan(r)):
                        bad.append((fd, ti, j, r, ref))
                else:
                    worst = max(worst, abs(r - ref))
    return n == len(rpt) and not bad and worst < TOL, dict(cells=n, rpt_cells=len(rpt), nan_mismatch=len(bad),
                                                           max_abs_dev=worst)


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    base, out, fmap = sys.argv[1], sys.argv[2], sys.argv[3]
    trial_r_only = "--trial-r-only" in sys.argv
    os.makedirs(out, exist_ok=True)
    jobs = {}
    for r in csv.DictReader(open(fmap, newline="")):
        jobs.setdefault(r["session"], {})[int(r["fold"])] = (r["job"], r["job_dir"])
    sessions = sorted(f[:-len("_trial_r.csv")] for f in os.listdir(base) if f.endswith("_trial_r.csv"))
    missing = sorted(set(jobs) - set(sessions))
    assert not missing, "fold map sessions not in base: %s" % missing
    report, failed = [], []
    for s in sessions:
        bt, bd, bj = (os.path.join(base, s + x) for x in ("_trial_r.csv", "_dump.h5", "_run.json"))
        ot, od, oj = (os.path.join(out, s + x) for x in ("_trial_r.csv", "_dump.h5", "_run.json"))
        if s not in jobs:
            for a, b in ((bt, ot), (bd, od), (bj, oj)):
                if os.path.exists(a):
                    link(a, b)
            report.append(dict(session=s, rebuilt=False))
            continue
        head, tb = read_trial_r(bt)
        if not tb:
            jt = [read_trial_r(os.path.join(jd, s + "_trial_r.csv"))[1] for _, jd in jobs[s].values()]
            if any(jt):
                failed.append((s, None, "the first stage has no rows but a re-decoded fold has rows"))
                continue
            for a, b in ((bt, ot), (bd, od), (bj, oj)):
                if os.path.exists(a):
                    link(a, b)
            report.append(dict(session=s, rebuilt=False, note="no tested trial in the first stage nor in the re-decoded folds"))
            print("linked %s: no tested trial in the first stage nor in the %d re-decoded folds" % (s, len(jobs[s])), flush=True)
            continue
        folds = sorted(tb)
        src, gates = {}, {}
        lines = []
        for fd in folds:
            if fd in jobs[s]:
                job, jd = jobs[s][fd]
                h2, tj = read_trial_r(os.path.join(jd, s + "_trial_r.csv"))
                assert h2 == head and list(tj) == [fd], (s, fd, list(tj))
                lines += tj[fd]
                src[fd] = job
            else:
                lines += tb[fd]
                src[fd] = "session_band_table"
        assert set(jobs[s]) <= set(folds), (s, sorted(jobs[s]), folds)
        info = dict(session=s, rebuilt=True, folds_redecoded=sorted(jobs[s]))
        if not trial_r_only:
            at, per, grids = load_dump(bd)
            assert sorted(per) == folds, (s, sorted(per), folds)
            fsrc = []
            new = dict(per)
            for fd in folds:
                if src[fd] == "session_band_table":
                    fsrc.append(dict(fold=fd, source="session_band_table", export_md5=str(at["export_md5"]),
                                     export_path=str(at["export_path"]), decoder_md5=str(at["decoder_md5"])))
                    continue
                jat, jper, jgrids = load_dump(os.path.join(jobs[s][fd][1], s + "_dump.h5"))
                assert list(jper) == [fd], (s, fd, list(jper))
                if jper[fd]["trials"] != per[fd]["trials"]:
                    failed.append((s, fd, "G-a trial set"))
                for k in grids:
                    assert np.array_equal(grids[k], jgrids[k]), (s, k)
                for k in ("session", "joints", "fs", "split", "input_control"):
                    assert str(jat[k]) == str(at[k]), (s, k, jat[k], at[k])
                new[fd] = jper[fd]
                fsrc.append(dict(fold=fd, source=src[fd], export_md5=str(jat["export_md5"]),
                                 export_path=str(jat["export_path"]), decoder_md5=str(jat["decoder_md5"])))
            at2 = dict(at)
            at2["fold_sources"] = json.dumps(fsrc)
            at2["merged_build"] = BUILD_ID
            write_dump(od, at2, new, grids, folds)
            _, wper, _ = load_dump(od)
            for fd in folds:
                ref = per[fd] if src[fd] == "session_band_table" else new[fd]
                if not same_fold(wper[fd], ref):
                    failed.append((s, fd, "G-b/G-c dump arrays"))
        tmp = ot + ".tmp"
        with open(tmp, "w", newline="") as fh:
            fh.write(head + "\n" + "\n".join(lines) + "\n")
        os.replace(tmp, ot)
        _, wt = read_trial_r(ot)
        for fd in folds:
            ref = tb[fd] if src[fd] == "session_band_table" else read_trial_r(os.path.join(jobs[s][fd][1], s + "_trial_r.csv"))[1][fd]
            if wt[fd] != ref:
                failed.append((s, fd, "G-b/G-c trial_r lines"))
        rj = json.load(open(bj))
        jf = {i["fold"]: i for i in rj["folds"]}
        for fd in jobs[s]:
            jj = json.load(open(os.path.join(jobs[s][fd][1], s + "_run.json")))
            assert [i["fold"] for i in jj["folds"]] == [fd]
            jf[fd] = jj["folds"][0]
        rj["folds"] = [jf[fd] for fd in folds if fd in jf]
        rj["fold_sources"] = [dict(fold=fd, source=src[fd]) for fd in folds]
        rj["merged_build"] = BUILD_ID
        with open(oj + ".tmp", "w") as fh:
            json.dump(rj, fh, indent=1)
        os.replace(oj + ".tmp", oj)
        if not trial_r_only:
            ok, rec = recon_gate(od, ot)
            info["G-d"] = rec
            if not ok:
                failed.append((s, None, "G-d reconstruction %s" % rec))
        report.append(info)
        print("merged %s: folds %s re-decoded%s" % (s, sorted(jobs[s]), "" if trial_r_only else
                                                    " | G-d max|dev| %.2e cells %d" % (info["G-d"]["max_abs_dev"],
                                                                                        info["G-d"]["cells"])), flush=True)
    summ = dict(base=os.path.abspath(base), out=os.path.abspath(out), fold_map=os.path.abspath(fmap),
                sessions=len(sessions), rebuilt=sum(r["rebuilt"] for r in report),
                folds_redecoded=sum(len(v) for v in jobs.values()), failed=[list(map(str, f)) for f in failed],
                report=report)
    with open(os.path.join(out, "_fold_merge.json.tmp"), "w") as fh:
        json.dump(summ, fh, indent=1)
    os.replace(os.path.join(out, "_fold_merge.json.tmp"), os.path.join(out, "_fold_merge.json"))
    print("MERGE %s: sessions %d, rebuilt %d, folds re-decoded %d, failures %d" % (
        "PASS" if not failed else "FAIL", summ["sessions"], summ["rebuilt"], summ["folds_redecoded"], len(failed)))
    if failed:
        print(failed[:20])
        sys.exit(2)


if __name__ == "__main__":
    main()
