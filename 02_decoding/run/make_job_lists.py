"""Decoder — run drivers (02_decoding/run)

Job lists of the decode chain: exports.csv (one MATLAB feature export per row, run by export_features.sh) and
decodes.csv (one decoded fold per row), with every count asserted.

Band tables: the whole-session table of each session and the per-fold tables of fold_tables/ (imagined
walking and patients, real walking, contiguous folds); a fold whose training-trial table equals the
whole-session table is decoded from the whole-session export. Exports: healthy imagined walking 140, patients
35, real walking 133, contiguous folds 42, rest windows (healthy, patients), cross-task 80 + 80. Decode rows:
imagined walking 800, post-operative 110, pre-operative 40, real walking 800, contiguous folds 800, cadence
shuffle 800, resting input 800, patient circular shift 150, patient zero input 150, patient resting input,
cross-task 80 + 80. The fold sets per session are those of the run on the zero-phase preprocessing (they
depend on the goniometer and the trial windows only). Usage: python make_job_lists.py

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import glob
import os
import sys
from collections import OrderedDict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths as P

D = os.environ.get("MAIN_LIST_DMI", P.DMI)
CFG = {"iw": "tenfold_pca30_lpf6.json", "post_operative": "tenfold_pca30_lpf6.json", "pre_operative": "tenfold_pca30_lpf6.json",
       "rw": "tenfold_pca30_lpf6.json", "block": "block_pca30_lpf6.json", "cad": "tenfold_pca30_lpf6_cadence_shuffle.json",
       "resth": "tenfold_pca30_lpf6_rest_input.json", "pcirc": "tenfold_pca30_lpf6_circular_shift_patient.json", "pzero": "tenfold_pca30_lpf6_zero_eeg_patient.json",
       "prest": "tenfold_pca30_lpf6_rest_input.json", "xme": "cross_task_pca30_lpf6.json", "xem": "cross_task_pca30_lpf6.json"}
TARGET = {"iw": "h_iw_dump", "post_operative": "pat_post_operative_iw", "pre_operative": "pat_pre_operative_iw", "rw": "h_rw_dump",
          "block": "h_blockcv_dump", "cad": "h_cadshuf_dump", "resth": "h_rest_testonly_dump",
          "pcirc": "pat_circshift_null", "pzero": "pat_zero_null", "prest": "pat_rest_testonly",
          "xme": "h_xtask_mi2exec_dump", "xem": "h_xtask_exec2mi_dump"}
EXPECT_FOLDS = {"iw": 800, "post_operative": 110, "pre_operative": 40, "rw": 800, "block": 800, "cad": 800, "resth": 800,
                "pcirc": 150, "pzero": 150, "xme": 80, "xem": 80}
EXPECT_EXPORTS = {"iw": 140, "pat": 35, "rw": 133, "bk": 42, "xme": 80, "xem": 80}
R14 = os.path.join(D, P.R14)


def rd(rel, **kw):
    return pd.read_csv(os.path.join(D, rel), **kw)


def current_folds(run):
    """session -> sorted folds with rows in the zero-phase run's trial_r of <run>; header-only sessions -> []"""
    out = {}
    for f in sorted(glob.glob(os.path.join(R14, TARGET[run], "*_trial_r.csv"))):
        s = os.path.basename(f)[:-len("_trial_r.csv")]
        out[s] = sorted(set(int(x) for x in pd.read_csv(f, usecols=["fold"]).fold))
    return out


def folds_of(x):
    return [int(k) for k in str(x).split(";")]


def main():
    H = list(rd(P.H_MANIFEST).session)
    assert len(H) == 80 and list(rd(P.H_MANIFEST_RW).session) == H
    PAT = list(P.PRE_OPERATIVE) + list(P.POST_OPERATIVE)
    f0 = rd(P.F0F + "/tables.csv")
    f0 = f0[~f0.table_id.str.startswith("G1_")].copy()
    rwT = rd(P.FT14 + "/rw/tables.csv", keep_default_na=False)
    bkT = rd(P.FT14 + "/block/tables.csv", keep_default_na=False)
    fH, fP = f0[f0.cohort == "healthy"], f0[f0.cohort != "healthy"]
    assert (len(fH), fH.n_folds.sum(), fH.session.nunique()) == (60, 146, 45), (len(fH), fH.n_folds.sum())
    assert (len(fP), fP.n_folds.sum()) == (20, 43), (len(fP), fP.n_folds.sum())
    assert (len(rwT), rwT.n_folds.sum()) == (53, 116) and (len(bkT), bkT.n_folds.sum()) == (96, 222)
    assert ((bkT.reuse_export != "").sum(), (bkT.reuse_export == "").sum()) == (54, 42)
    V = {r: current_folds(r) for r in TARGET}
    assert sorted(V["iw"]) == sorted(H) and sorted(V["post_operative"]) == sorted(P.POST_OPERATIVE) \
        and sorted(V["pre_operative"]) == sorted(P.PRE_OPERATIVE), " session sets"

    ex = OrderedDict()

    def add_export(eid, kind, env, s, tid, band, out, ref, pair="", order=0):
        if eid in ex:
            return eid
        ex[eid] = dict(export_id=eid, kind=kind, env=env, session=s, table_id=tid, band_csv=band, out_file=out,
                       ref_export=ref, pair_export=pair, order=order)
        return eid

    for s in H:
        add_export("iw__%s__ws" % s, "iw", "h_iw", s, s + "__ws", P.H_IW_BANDS, s + "_export.mat",
                   "%s/h_cdm_o1_c0p18_causal_ownlso/%s_export.mat" % (P.PX, s), order=1)
        for r in fH[fH.session == s].itertuples():
            add_export("iw_" + r.table_id, "iw", "h_iw", s, r.table_id, r.band_csv, s + "_export.mat",
                       "%s/%s_export.mat" % (r.export_dir, s), order=1)
    for s in PAT:
        g = P.p_group(s)
        add_export("pat__%s__ws" % s, "pat", "p_iw", s, s + "__ws", P.P_GROUP_BANDS[g], s + "_export.mat",
                   "%s/cdm_o1_c0p18_causal_ownhglobal/%s_export.mat" % (P.PX, s), order=2)
        for r in fP[fP.session == s].itertuples():
            add_export("pat_" + r.table_id, "pat", "p_iw", s, r.table_id, r.band_csv, s + "_export.mat",
                       "%s/%s_export.mat" % (r.export_dir, s), order=2)
    for s in H:
        add_export("rw__%s__ws" % s, "rw", "h_rw", s, s + "__ws", P.H_EXEC_BANDS, s + "_export.mat",
                   "%s/h_rw_cdm_o1_c0p18_causal_ownlso_exec/%s_export.mat" % (P.PX, s), order=3)
        for r in rwT[rwT.session == s].itertuples():
            add_export("rw_" + r.table_id, "rw", "h_rw", s, r.table_id, r.band_csv, s + "_export.mat",
                       "%s/%s_export.mat" % (r.export_dir, s), order=3)
    for r in bkT[bkT.reuse_export == ""].itertuples():
        add_export("bk_" + r.table_id, "bk", "h_iw", r.session, r.table_id, r.band_csv, r.session + "_export.mat",
                   "%s/%s_export.mat" % (r.export_dir, r.session), order=4)

    rows = []

    def table_map(s, tables, ws_eid, prefix, all_folds, reuse=None):
        """fold -> (export_id, table_id) for one session: fold tables first, the whole-session table for the rest"""
        m = {}
        for r in tables[tables.session == s].itertuples():
            if reuse is not None and r.reuse_export:
                eid = "iw_" + os.path.basename(r.reuse_export)
                assert eid in ex, ("block reuse of a non-exported iw table", r.table_id, eid)
            else:
                eid = prefix + r.table_id
            for k in folds_of(r.folds):
                assert k not in m, (s, k)
                m[k] = (eid, r.table_id)
        for k in all_folds:
            m.setdefault(k, (ws_eid, s + "__ws"))
        assert sorted(m) == sorted(all_folds), (s, sorted(m), all_folds)
        return m

    def add_rows(run, s, fmap, basis, extra_of=None, folds=None):
        for k in sorted(fmap) if folds is None else folds:
            eid, tid = fmap[k]
            xid = extra_of(eid, tid) if extra_of else ""
            rows.append(dict(decode_id="%s__%s__f%d" % (run, tid, k), run=run, target=TARGET[run], session=s, fold=k,
                             table_id=tid, config=CFG[run], export_id=eid, extra_id=xid,
                             extra_flag={"resth": "--rest-export", "prest": "--rest-export"}.get(run, ""),
                             basis=basis, claim_key=eid + ("+" + xid if xid else "")))

    def resth_of(eid, tid):
        s = ex[eid]["session"]
        return add_export("resth_" + tid, "resth", "h_rest", s, tid, ex[eid]["band_csv"], s + "_rest.mat",
                          ("%s/rest_cdm_o1_c0p18_causal_ownlso/%s_rest.mat" % (P.PX, s)) if tid.endswith("__ws") else
                          "%s/rest_h/export/%s/%s_rest.mat" % (P.FT14, tid, s), pair=eid, order=5)

    def restp_of(eid, tid):
        s = ex[eid]["session"]
        g = P.P_REST_GROUP[s]
        band = P.P_REST_BANDS[g] if tid.endswith("__ws") else ex[eid]["band_csv"]
        return add_export("restp_" + tid, "restp", "p_rest", s, tid, band, s + "_rest.mat",
                          ("%s/rest_cdm_o1_c0p18_causal_ownhglobal/%s_rest.mat" % (P.PX, s)) if tid.endswith("__ws")
                          else "%s/rest_p/export/%s/%s_rest.mat" % (P.FT14, tid, s), pair=eid, order=6)

    empty_prest = []
    for s in H:
        m_iw = table_map(s, fH, "iw__%s__ws" % s, "iw_", V["iw"][s])
        add_rows("iw", s, m_iw, "iw")
        add_rows("cad", s, m_iw, "iw")
        add_rows("resth", s, m_iw, "iw", extra_of=resth_of)
        assert V["cad"][s] == V["iw"][s] == V["resth"][s], s
        add_rows("block", s, table_map(s, bkT, "iw__%s__ws" % s, "bk_", V["block"][s], reuse=True), "iw")
        add_rows("rw", s, table_map(s, rwT, "rw__%s__ws" % s, "rw_", V["rw"][s]), "exec")
    for s in PAT:
        m_p = table_map(s, fP, "pat__%s__ws" % s, "pat_", V["post_operative" if s in P.POST_OPERATIVE else "pre_operative"][s])
        add_rows("post_operative" if s in P.POST_OPERATIVE else "pre_operative", s, m_p, "hglobal")
        assert V["pcirc"][s] == V["pzero"][s] == sorted(m_p), s
        add_rows("pcirc", s, m_p, "hglobal")
        add_rows("pzero", s, m_p, "hglobal")
        if s in P.POST_OPERATIVE:
            keep = V["prest"].get(s, [])
            if not keep:
                empty_prest.append(s)
                continue
            assert set(keep) <= set(m_p), s
            add_rows("prest", s, m_p, "hglobal", extra_of=restp_of, folds=keep)
    for s in H:
        x = add_export("xme_" + s, "xme", "xt_mi2exec", s, s + "__ws", P.H_IW_BANDS, s + "_export.mat",
                       "%s/xt_EXECtrials_MIbands_cdm_o1_c0p18_causal/%s_export.mat" % (P.PX, s), order=7)
        y = add_export("xem_" + s, "xem", "xt_exec2mi", s, s + "__ws", P.H_EXEC_BANDS, s + "_export.mat",
                       "%s/xt_MItrials_EXECbands_cdm_o1_c0p18_causal/%s_export.mat" % (P.PX, s), order=7)
        for run, eid, xid, basis in (("xme", "iw__%s__ws" % s, x, "iw"), ("xem", "rw__%s__ws" % s, y, "exec")):
            assert V[run][s] == [1], (run, s, V[run][s])
            rows.append(dict(decode_id="%s__%s__all" % (run, s), run=run, target=TARGET[run], session=s, fold=1,
                             table_id=s + "__ws", config=CFG[run], export_id=eid, extra_id=xid,
                             extra_flag="--cross-export", basis=basis, claim_key=eid + "+" + xid))
    for r in rows:
        if r["run"] in ("xme", "xem"):
            r["folds_arg"] = "-"
        else:
            r["folds_arg"] = str(r["fold"])

    E = pd.DataFrame(list(ex.values()))
    Dd = pd.DataFrame(rows)
    used = set(Dd.export_id) | set(Dd.extra_id[Dd.extra_id != ""])
    assert used == set(E.export_id), ("exports without a decode row", sorted(set(E.export_id) - used)[:5],
                                      "decode rows without an export", sorted(used - set(E.export_id))[:5])
    assert Dd.decode_id.is_unique and E.export_id.is_unique
    for k, n in EXPECT_EXPORTS.items():
        assert (E.kind == k).sum() == n, (k, (E.kind == k).sum(), n)
    n_prest_current = sum(len(v) for v in V["prest"].values())
    for run in TARGET:
        g = Dd[Dd.run == run]
        want = EXPECT_FOLDS.get(run, n_prest_current)
        assert len(g) == want, (run, len(g), want)
        cov = {s: sorted(gg.fold) for s, gg in g.groupby("session")}
        assert all(cov.get(s, []) == v for s, v in V[run].items()), ("fold coverage != ", run)
    assert sorted(empty_prest) == sorted(s for s, v in V["prest"].items() if not v)
    E = E.sort_values(["order"], kind="stable")
    run_order = {r: i for i, r in enumerate(["iw", "post_operative", "pre_operative", "rw", "block", "cad", "pcirc", "pzero", "resth",
                                             "prest", "xme", "xem"])}
    Dd["_o"] = Dd.run.map(run_order)
    Dd = Dd.sort_values(["_o", "session", "fold"], kind="stable").drop(columns="_o")
    for name, df in (("exports.csv", E), ("decodes.csv", Dd)):
        p = os.path.join(HERE, name)
        df.to_csv(p + ".tmp", index=False, lineterminator="\n")
        assert len(pd.read_csv(p + ".tmp", keep_default_na=False)) == len(df)
        os.replace(p + ".tmp", p)
    print("exports.csv: %d rows" % len(E))
    print(E.groupby("kind", sort=False).agg(exports=("export_id", "size"), sessions=("session", "nunique")).to_string())
    print("decodes.csv: %d rows (folds), %d claim units" % (len(Dd), Dd.claim_key.nunique()))
    print(Dd.groupby("run", sort=False).agg(target=("target", "first"), folds=("fold", "size"),
                                           sessions=("session", "nunique"), claim_units=("claim_key", "nunique"),
                                           exports=("export_id", "nunique")).to_string())
    print("prest: non-empty folds %d of the zero-phase run; sessions without any rest window there (no export, no decode, merged as "
          "header-only): %s" % (n_prest_current, sorted(empty_prest)))
    nb = Dd[Dd.run == "block"]
    print("block folds by export kind: ws %d, reused iw fold-table exports %d, bk exports %d" % (
        nb.export_id.str.endswith("__ws").sum(), (nb.export_id.str.startswith("iw_") & ~nb.export_id.str.endswith("__ws")).sum(),
        nb.export_id.str.startswith("bk_").sum()))
    print("ALL COUNT ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
