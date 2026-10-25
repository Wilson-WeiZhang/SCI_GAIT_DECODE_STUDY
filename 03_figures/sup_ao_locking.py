"""Supplementary item

Gait-phase locking |c| (panel c) and k-map similarity to the healthy cohort map (panel d) at k1 / k2 / k3
under imagined walking (MI+AO) vs observation only (AO), six supplementary participants

Reads:
  04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_config as ec
import panel_style as PS
import si_panel_style as SI
import project_palette as PAL
import sup_panel_export as _spe
from sup_panel_export import export_legend

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = _spe.PANEL_DIR = ROOT / "figures_out/supplementary/_panels"
L6 = ROOT / "gait_prep_supplementary/result/lock_k2_6p"
LP = ROOT / "gait_prep_supplementary/result/lock_k2_perm"
HC = ROOT / ("data/decoder_output/diag_harmonic_phase_responder_split/"
             "persession_harmonic_phase_cycle9_complex.csv")
F6 = ROOT / "04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv"
SD = ROOT / "04_source_data/supplementary/sup_ao_locking_source_data.csv"
META = ROOT / "04_source_data/supplementary/sup_ao_locking_metadata.csv"
SLUG = "ao_locking"
SLUG_D = "ao_similarity"
ITEM14 = ROOT / ""
P_ORDER = ["SUB_09", "SUB_01", "SUB_12", "SUB_23", "SUB_29", "SUB_35"]
N_SHUF, SHUF_SEED = 1000, 20261014
NULL_FILL, NULL_EDGE = "#9aa0a6", "#5f666d"

CONDITIONS = {
    "SUB_01": ("subj01_sess05miao", "subj01_sess05ao", "subj01_", "SUB_01_sess05", "server_1"),
    "SUB_09": ("subj09_sess03miao", "subj09_sess03ao", "subj09_", "SUB_09_sess03", "server_1"),
    "SUB_12": ("subj12_sess03_abmiao", "subj12_sess03_abao", "subj12_", "SUB_12_sess03", "server_1"),
    "SUB_23": ("subj23_sess03_abmiao", "subj23_sess03_abao", "subj23_", "SUB_23_sess03", "server_1"),
    "SUB_29": ("subj29_sess03_abcmiao", "subj29_sess03_abcao", "subj29_", "SUB_29_sess03", "server_1"),
    "SUB_35": ("subj35_sess03_abmiao", "subj35_sess03_abao", "subj35_", "SUB_35_sess03", "server_2"),
}
PARTS = list(CONDITIONS)
KS = (1, 2, 3)
COL = {"MI+AO": PAL.IW_HUE, "AO": PAL.MID}


def gmean(A, mask=None):
    sub = A if mask is None else A[mask]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def coh(P, Hh):
    p = P.ravel(); h = Hh.ravel()
    ok = np.isfinite(p.real) & np.isfinite(p.imag) & np.isfinite(h.real) & np.isfinite(h.imag)
    p, h = p[ok], h[ok]
    if p.size < 5:
        return np.nan
    denom = np.linalg.norm(p) * np.linalg.norm(h)
    return float(np.abs(np.vdot(h, p)) / denom) if denom > 0 else np.nan


def load_maps(path, channels=None):
    d = pd.read_csv(path)
    sess = list(dict.fromkeys(d.session))
    chans = channels or list(dict.fromkeys(d.channel))
    fr = sorted(d.cycle_fraction.round(6).unique())
    A = np.full((len(sess), len(chans), len(fr), 3), np.nan, complex)
    si = {s: i for i, s in enumerate(sess)}; ci = {c: i for i, c in enumerate(chans)}
    fi = {f: i for i, f in enumerate(fr)}
    d = d[d.channel.isin(chans)]
    A[d.session.map(si).values, d.channel.map(ci).values, d.cycle_fraction.round(6).map(fi).values,
      d.harmonic_k.values - 1] = d.real.values + 1j * d.imag.values
    return sess, chans, A


def gates():
    b = pd.read_csv(L6 / "s5m_lock_k2_6p_b1b2_persession.csv")
    cs = pd.read_csv(L6 / "supplementary_persession_harmonic_contrast_vs_r.csv").set_index(
        ["session", "harmonic_k"]).mean_abs
    nl = pd.read_csv(L6 / "supplementary_persession_rest_null.csv").set_index(["session", "harmonic_k"])

    g1 = 0.0
    for _, r in b.iterrows():
        key = (r.session, r.harmonic_k)
        c, n = float(cs.loc[key]), nl.loc[key]
        rebuilt = (c, n.rest_null_mean, n.rest_null_p95, (c - n.rest_null_mean) / n.rest_null_sd)
        recorded = (r.abs_c, r.rest_null_mean, r.rest_null_p95, r.z_vs_rest_null)
        g1 = max(g1, max(abs(x - y) for x, y in zip(rebuilt, recorded)))
        assert bool(c > n.rest_null_p95) == bool(r.above_rest_p95) and int(n.n_trials) == int(r.n_trials)
    assert len(b) == len(PARTS) * 2 * 3 and g1 < 1e-12, g1
    print(f"gate 1 B1/B2 read-out rebuilt from the dump tables: max |diff| {g1:.2e}")

    C6 = pd.read_csv(L6 / "supplementary_persession_harmonic_phase_cycle9_complex.csv")
    C6["a"] = np.hypot(C6.real, C6.imag)
    ma = C6.groupby(["session", "harmonic_k"]).a.mean()
    bi = b.set_index(["session", "harmonic_k"])
    g2 = float(np.max(np.abs(bi.abs_c - ma.loc[bi.index])))
    assert g2 < 1e-9, g2
    print(f"gate 2 |c| from the complex dump vs b1b2 abs_c: max |diff| {g2:.2e}")

    sess6, ch6, A6 = load_maps(L6 / "supplementary_persession_harmonic_phase_cycle9_complex.csv")
    hs, _, HA = load_maps(HC, channels=ch6)
    assert len(hs) == 80
    f6 = pd.read_csv(F6)
    f6 = f6[f6.row_type == "healthy_lpo"].set_index(["session", "harmonic_k"]).value
    subj = np.array([x.split("_sess")[0] for x in hs])
    g3 = 0.0
    HL = {k: np.full(len(hs), np.nan) for k in KS}
    for i, s in enumerate(hs):
        mask = subj != subj[i]
        G = gmean(HA, mask)
        for k in KS:
            HL[k][i] = coh(HA[i, :, :, k - 1], G[:, :, k - 1])
            g3 = max(g3, abs(HL[k][i] - f6.loc[(s, k)]))
    assert g3 < 1e-5, g3
    print(f"gate 3 Fig. 6c healthy LPO similarity reproduced: max |diff| {g3:.2e} (Source Data at 6 dp)")

    b3 = pd.read_csv(L6 / "s5m_lock_k2_6p_b3_topography.csv")
    idx6 = {s: i for i, s in enumerate(sess6)}
    g4 = 0.0
    for p in PARTS:
        mi, ao, pref = CONDITIONS[p][:3]
        G = gmean(HA, np.array([not x.startswith(pref) for x in hs]))[:, :, 1]
        Mm, Ma = A6[idx6[mi], :, :, 1], A6[idx6[ao], :, :, 1]
        for pair, v in (("MIAO_vs_cohortLOO", coh(Mm, G)), ("AO_vs_cohortLOO", coh(Ma, G)),
                        ("AO_vs_own_MIAO", coh(Ma, Mm))):
            g4 = max(g4, abs(v - b3[(b3.participant == p) & (b3.pair == pair)].coherence.iloc[0]))
    assert g4 < 1e-9, g4
    print(f"gate 4 k2 similarity vs the recorded B3 table: max |diff| {g4:.2e}")
    return b, dict(gate1=g1, gate2=g2, gate3=g3, gate4=g4), dict(sess6=sess6, A6=A6, hs=hs, HA=HA, HL=HL)


def similarity(ctx, gv):
    """Similarity section (b), same code and the same rng stream (participants in P_ORDER; per k the MI+AO,
    AO and AO-vs-own-MI+AO shuffles, then the group-map shuffles)."""
    sess6, A6, hs, HA, HL = (ctx[x] for x in ("sess6", "A6", "hs", "HA", "HL"))
    idx6 = {s: i for i, s in enumerate(sess6)}
    H80 = gmean(HA)
    rng = np.random.default_rng(SHUF_SEED)

    def shuffled(M, Gk):
        null = np.array([coh(M[rng.permutation(M.shape[0])], Gk) for _ in range(N_SHUF)])
        return null

    S, NULL = {}, {k: [] for k in KS}
    for p in P_ORDER:
        mi, ao, pref = CONDITIONS[p][:3]
        mask = np.array([not x.startswith(pref) for x in hs])
        G = gmean(HA, mask)
        for k in KS:
            Gk = G[:, :, k - 1]
            rec = dict(n_ref=int(mask.sum()))
            for c, sname in (("MI+AO", mi), ("AO", ao)):
                M = A6[idx6[sname], :, :, k - 1]
                v = coh(M, Gk)
                null = shuffled(M, Gk)
                NULL[k].append(null)
                rec[c] = dict(sim=v, full80=coh(M, H80[:, :, k - 1]), p95=float(np.quantile(null, 0.95)),
                              p=float((1 + (null >= v).sum()) / (N_SHUF + 1)),
                              pct=float(100 * (HL[k] < v).mean()))
            Mm, Ma = A6[idx6[mi], :, :, k - 1], A6[idx6[ao], :, :, k - 1]
            v = coh(Ma, Mm)
            null = shuffled(Ma, Mm)
            rec["own"] = dict(sim=v, p95=float(np.quantile(null, 0.95)), p=float((1 + (null >= v).sum()) / (N_SHUF + 1)))
            S[(p, k)] = rec

    Gc = {"MI+AO": gmean(A6[[idx6[CONDITIONS[p][0]] for p in P_ORDER]]),
          "AO": gmean(A6[[idx6[CONDITIONS[p][1]] for p in P_ORDER]])}
    mask6 = np.array([not any(x.startswith(CONDITIONS[p][2]) for p in P_ORDER) for x in hs])
    G68 = gmean(HA, mask6)
    GR = {}
    for k in KS:
        for c in ("MI+AO", "AO"):
            M = Gc[c][:, :, k - 1]
            v = coh(M, G68[:, :, k - 1])
            null = shuffled(M, G68[:, :, k - 1])
            GR[(c, k)] = dict(sim=v, p95=float(np.quantile(null, 0.95)), p=float((1 + (null >= v).sum()) / (N_SHUF + 1)),
                              abs_c=float(np.nanmean(np.abs(M))), n_ref=int(mask6.sum()))
        GR[("AO vs MI+AO", k)] = dict(sim=coh(Gc["AO"][:, :, k - 1], Gc["MI+AO"][:, :, k - 1]))

    it = pd.read_csv(ITEM14 / "similarity_per_participant.csv").set_index(["participant", "harmonic_k"])
    ig = pd.read_csv(ITEM14 / "groupmap.csv").set_index("harmonic_k")
    g5 = 0.0
    for (p, k), r in S.items():
        q = it.loc[(p, k)]
        assert int(q.n_healthy_sessions_lpo) == r["n_ref"]
        for c, a in (("MI+AO", "miao"), ("AO", "ao")):
            for key, col in (("sim", f"sim_{a}_lpo"), ("full80", f"sim_{a}_full80"), ("p95", f"shuffle_p95_{a}"),
                             ("p", f"p_shuffle_{a}"), ("pct", f"pct_in_healthy_lpo_{a}")):
                g5 = max(g5, abs(r[c][key] - q[col]))
        for key, col in (("sim", "sim_ao_vs_own_miao"), ("p95", "shuffle_p95_ao_vs_own"), ("p", "p_shuffle_ao_vs_own")):
            g5 = max(g5, abs(r["own"][key] - q[col]))
    for k in KS:
        q = ig.loc[k]
        assert int(q.n_healthy_sessions_without_six) == GR[("MI+AO", k)]["n_ref"]
        for c, a in (("MI+AO", "miao"), ("AO", "ao")):
            for key, col in (("sim", f"groupmap_sim_{a}"), ("p95", f"groupmap_shuffle_p95_{a}"),
                             ("p", f"groupmap_p_shuffle_{a}"), ("abs_c", f"groupmap_abs_c_{a}")):
                g5 = max(g5, abs(GR[(c, k)][key] - q[col]))
        g5 = max(g5, abs(GR[("AO vs MI+AO", k)]["sim"] - q.groupmap_sim_ao_vs_miao))
    assert len(S) == 18 and g5 <= 1e-6, g5
    print(f"gate 5 similarity, shuffle p95 / p, healthy percentile and group maps vs the recorded similarity tables: max |diff| {g5:.2e}")
    gv["gate5"] = g5

    band = {}
    for k in KS:
        v = np.concatenate(NULL[k])
        band[k] = dict(zip(("p10", "q25", "median", "q75", "p90", "p95"),
                           (float(np.quantile(v, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95))), n=v.size)

    T = {}
    for k in KS:
        m = np.array([S[(p, k)]["MI+AO"]["sim"] for p in PARTS]); a = np.array([S[(p, k)]["AO"]["sim"] for p in PARTS])
        d = m - a
        t = stats.ttest_rel(m, a)
        ci = t.confidence_interval(0.95)
        T[k] = dict(mean_delta=d.mean(), ci_lo=ci.low, ci_hi=ci.high, t=t.statistic, df=len(d) - 1, p_two=t.pvalue,
                    n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n=len(d))
    for k, q in zip(KS, bh([T[k]["p_two"] for k in KS])):
        T[k]["pfdr_two"] = q
    return S, GR, band, T


def bh(p):
    """Benjamini-Hochberg adjusted p over one family (here the three harmonics)."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1.0)
    return out


def build(b):
    pp = pd.read_csv(LP / "s5m_lock_b1_perm_persession.csv")
    gr = pd.read_csv(LP / "s5m_lock_b1_perm_group.csv").set_index("harmonic_k")
    condition = {"MIAO": "MI+AO", "AO": "AO"}
    L = {}
    for _, r in b.iterrows():
        L[(r.participant, condition[r.condition], int(r.harmonic_k))] = r
    P = {(r.participant, int(r.harmonic_k)): r for _, r in pp.iterrows()}
    assert set(p for p, _ in P) == set(PARTS) and len(gr) == 3

    def paired(m, a):
        m, a = np.asarray(m, float), np.asarray(a, float)
        d = m - a
        t = stats.ttest_rel(m, a)
        ci = t.confidence_interval(0.95)
        return dict(mean_delta=d.mean(), ci_lo=ci.low, ci_hi=ci.high, t=t.statistic, df=len(d) - 1,
                    p_two=t.pvalue, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n=len(d))

    G = {}
    for k in KS:
        G[("asrun", k)] = paired([L[(p, "MI+AO", k)].abs_c for p in PARTS],
                                 [L[(p, "AO", k)].abs_c for p in PARTS])
        G[("bbpooled", k)] = paired([P[(p, k)].abs_c_miao_pooled_band for p in PARTS],
                                    [P[(p, k)].abs_c_ao_pooled_band for p in PARTS])
        g = gr.loc[k]
        assert int(g.n_participants) == 6 and int(g.B_group) == 20000
        assert abs(g.mean_delta_pooled_band - G[("bbpooled", k)]["mean_delta"]) < 1e-12
        G[("perm", k)] = dict(role=g.role, p_one=g.p_one_sided, p_two=g.p_two_sided,
                              B=int(g.B_group), floor=g.p_floor, n_pos=int(g.n_positive))
    for rec, key, out in (("perm", "p_one", "pfdr_one"), ("perm", "p_two", "pfdr_two"),
                          ("bbpooled", "p_two", "pfdr_two"), ("asrun", "p_two", "pfdr_two")):
        for k, q in zip(KS, bh([G[(rec, k)][key] for k in KS])):
            G[(rec, k)][out] = q
    return L, P, G


def write_source_data(L, P, G, SM):
    rows = []
    for p in PARTS:
        mi, ao, _, sess, host = CONDITIONS[p]
        for k in KS:
            q = P[(p, k)]
            for c, vs in (("MI+AO", mi), ("AO", ao)):
                r = L[(p, c, k)]
                rows.append(dict(
                    row_type="participant_condition", participant=p, session=sess, condition=c,
                    virtual_session=vs, harmonic_k=k, input="re-prep, pooled band", prep_host=q.host,
                    abs_c=(q.abs_c_miao_pooled_band if c == "MI+AO" else q.abs_c_ao_pooled_band),
                    n_trials=int(q.n_trials_miao if c == "MI+AO" else q.n_trials_ao),
                    abs_c_asrun=r.abs_c, prep_host_asrun=host,
                    asrun_rest_null_mean=r.rest_null_mean, asrun_rest_null_p95=r.rest_null_p95,
                    asrun_rest_null_draws=int(r.n_resamples), asrun_above_rest_p95=bool(r.above_rest_p95),
                    asrun_z_vs_rest_null=r.z_vs_rest_null))
            rows.append(dict(
                row_type="participant_difference", participant=p, session=sess,
                condition="MI+AO minus AO", harmonic_k=k, input="re-prep, pooled band", prep_host=q.host,
                delta_abs_c=q.delta_pooled_band,
                ci95_lo=q.ci95_lo_boot_pairs, ci95_hi=q.ci95_hi_boot_pairs, ci_method="percentile bootstrap over the 15 cycle pairs, B 2000",
                perm_p_one_sided=q.p_primary_one_sided, perm_p_two_sided=q.p_primary_two_sided,
                perm_B=int(q.B_primary),
                delta_abs_c_asrun=L[(p, "MI+AO", k)].abs_c - L[(p, "AO", k)].abs_c, prep_host_asrun=host))
    for k in KS:
        g, t = G[("perm", k)], G[("bbpooled", k)]
        rows.append(dict(row_type="cohort_cycle_label_permutation", test_role="main", condition="MI+AO minus AO",
                         harmonic_k=k, input="re-prep, pooled band", harmonic_role=g["role"],
                         delta_abs_c=t["mean_delta"], ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"],
                         ci_method="t interval over the six participant differences",
                         perm_p_one_sided=g["p_one"], perm_pfdr_one_sided=g["pfdr_one"],
                         perm_p_two_sided=g["p_two"], perm_pfdr_two_sided=g["pfdr_two"],
                         perm_B=g["B"], p_floor=g["floor"], n_positive=t["n_pos"], n_negative=t["n_neg"], n=6))
        rows.append(dict(row_type="cohort_paired_t", test_role="companion", condition="MI+AO minus AO",
                         harmonic_k=k, input="re-prep, pooled band", delta_abs_c=t["mean_delta"],
                         ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"], t=t["t"], df=t["df"],
                         p_two_sided=t["p_two"], pfdr_two_sided=t["pfdr_two"],
                         n_positive=t["n_pos"], n_negative=t["n_neg"], n=t["n"]))
        a = G[("asrun", k)]
        rows.append(dict(row_type="cohort_paired_t_asrun", test_role="extra",
                         condition="MI+AO minus AO", harmonic_k=k,
                         input="as-run |c|, frozen per-condition band", delta_abs_c=a["mean_delta"],
                         ci95_lo=a["ci_lo"], ci95_hi=a["ci_hi"], t=a["t"], df=a["df"],
                         p_two_sided=a["p_two"], pfdr_two_sided=a["pfdr_two"],
                         n_positive=a["n_pos"], n_negative=a["n_neg"], n=a["n"]))
    S, GR, band, T = SM
    inp = "as-run complex maps, frozen per-condition band"
    ref = "healthy cohort map without the participant's own sessions (leave-participant-out)"
    for p in PARTS:
        mi, ao, _, sess, host = CONDITIONS[p]
        for k in KS:
            r = S[(p, k)]
            for c, vs in (("MI+AO", mi), ("AO", ao)):
                q = r[c]
                rows.append(dict(
                    row_type="participant_similarity", participant=p, session=sess, condition=c, virtual_session=vs,
                    harmonic_k=k, input=inp, prep_host=host, reference=ref, n_healthy_sessions_ref=r["n_ref"],
                    similarity=q["sim"], similarity_full80=q["full80"], shuffle_p95=q["p95"], shuffle_p=q["p"],
                    above_shuffle_p95=bool(q["sim"] > q["p95"]), n_shuffle=N_SHUF, shuffle_seed=SHUF_SEED,
                    healthy_percentile=q["pct"]))
            rows.append(dict(
                row_type="participant_similarity_ao_vs_own_miao", participant=p, session=sess,
                condition="AO vs own MI+AO", harmonic_k=k, input=inp, prep_host=host,
                reference="the same participant's MI+AO map", similarity=r["own"]["sim"],
                shuffle_p95=r["own"]["p95"], shuffle_p=r["own"]["p"], above_shuffle_p95=bool(r["own"]["sim"] > r["own"]["p95"]),
                n_shuffle=N_SHUF, shuffle_seed=SHUF_SEED))
            rows.append(dict(
                row_type="participant_similarity_difference", participant=p, session=sess, condition="MI+AO minus AO",
                harmonic_k=k, input=inp, prep_host=host, reference=ref,
                delta_similarity=r["MI+AO"]["sim"] - r["AO"]["sim"]))
    for k in KS:
        t = T[k]
        rows.append(dict(row_type="cohort_paired_t_similarity", test_role="panel d brackets", condition="MI+AO minus AO",
                         harmonic_k=k, input=inp, reference=ref, delta_similarity=t["mean_delta"],
                         ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"], ci_method="t interval over the six participant differences",
                         t=t["t"], df=t["df"], p_two_sided=t["p_two"], pfdr_two_sided=t["pfdr_two"],
                         n_positive=t["n_pos"], n_negative=t["n_neg"], n=t["n"]))
        b = band[k]
        rows.append(dict(row_type="similarity_null_band", condition="pooled MI+AO and AO", harmonic_k=k, input=inp,
                         reference=ref, null_p10=b["p10"], null_q25=b["q25"], null_median=b["median"], null_q75=b["q75"],
                         null_p90=b["p90"], null_p95=b["p95"], n_shuffle=b["n"], shuffle_seed=SHUF_SEED))
        for c in ("MI+AO", "AO"):
            g = GR[(c, k)]
            rows.append(dict(row_type="group_map_similarity", condition=c, harmonic_k=k, input=inp,
                             reference="healthy cohort map without all six participants' sessions",
                             n_healthy_sessions_ref=g["n_ref"], similarity=g["sim"], shuffle_p95=g["p95"],
                             shuffle_p=g["p"], above_shuffle_p95=bool(g["sim"] > g["p95"]), n_shuffle=N_SHUF,
                             shuffle_seed=SHUF_SEED, groupmap_abs_c=g["abs_c"]))
        rows.append(dict(row_type="group_map_similarity", condition="AO group map vs MI+AO group map", harmonic_k=k,
                         input=inp, reference="the six-session MI+AO group map", similarity=GR[("AO vs MI+AO", k)]["sim"]))
    df = pd.DataFrame(rows)
    tmp = SD.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g", lineterminator="\n")
    os.replace(tmp, SD)
    print("wrote", SD, f"({len(df)} rows)")
    return df


def write_metadata(G, gv, SM):
    t2, p2 = G[("bbpooled", 2)], G[("perm", 2)]
    S, GR, band, T = SM
    meta = [
        ("paper_location", "Supplementary Fig. 11 panels c and d; Main text Results (one sentence: six-participant "
                           "cohort test, cycle-label permutation pFDR); Supplementary Note 2 "
                           "(paired t as companion; per-participant breakdown of c; one sentence on the k2 map similarity of d: group-vs-group "
                           "values, AO above its null in every participant, paired difference not statistically "
                           "significant); panel files supplementary/_panels/S11c_ao_locking.png and "
                           "figures_out/supplementary/_panels/S11d_ao_similarity.png"),
        ("renderer", "03_figures/sup_ao_locking.py"),
        ("recomputes_nothing", "reads result CSVs only; no EEG is read"),
        ("metric", "|c| = mean over 60 channels x 9 gait-cycle points of |phase_mod| at the walker hip phase (LHipW X), "
                   "lag 0; 30 trials per condition in every session"),
        ("rest_null", "per-condition yoked-REST null (dump_persession_rest_null), 200 draws; above_rest_p95 = |c| > p95"),
        ("correction", "Benjamini-Hochberg FDR across k1 / k2 / k3, separately for each test and sidedness (pFDR); raw p kept"),
        ("test_cohort_companion", "paired t over the six participant differences MI+AO - AO, two-sided, same input; "
                                  "companion only, no significance claim rests on it"),
        ("k2_mean_delta", "%.6f" % t2["mean_delta"]),
        ("k2_perm_p_one_sided", "%.6g" % p2["p_one"]),
        ("k2_perm_pfdr_one_sided", "%.6g" % p2["pfdr_one"]),
        ("k2_paired_t5", "%.6f" % t2["t"]),
        ("k2_paired_t_p_two_sided", "%.6f" % t2["p_two"]),
        ("k2_paired_t_pfdr_two_sided", "%.6f" % t2["pfdr_two"]),
        ("scope", "case level: one session per participant, n = 6, experimenter-supplied cycle labels; both "
                  "conditions with the walker visible and audible"),
        ("d_metric", "Fig. 6 similarity: |sum P conj(H)| / (||P|| ||H||) over 60 channels x 9 gait-cycle points of the "
                     "complex harmonic phase map (invariant to a global phase), code copied from Fig. 6"),
        ("d_reference", "healthy cohort imagined-walking map without the participant's own sessions (leave-participant-out; "
                        "78 sessions for each of the six; each of the six is a healthy cohort participant); full-80 values kept "
                        "as similarity_full80"),
        ("d_healthy_percentile", "percent of the 80 healthy sessions' own leave-participant-out similarities (Fig. 6c "
                                 "healthy_lpo, full precision) below the value"),
        ("d_pfdr", "; ".join("k%d t(5) = %.6f, p %.6f, pFDR %.6f, mean delta %+.6f, %d / 6 positive"
                             % (k, T[k]["t"], T[k]["p_two"], T[k]["pfdr_two"], T[k]["mean_delta"], T[k]["n_pos"]) for k in KS)),
        ("d_k2_group_map", "six-session group map vs the healthy map of the sessions of the other 36 participants (n = 68): "
                           "MI+AO %.6f (shuffle p95 %.6f), AO %.6f (shuffle p95 %.6f); AO group vs MI+AO group %.6f"
                           % (GR[("MI+AO", 2)]["sim"], GR[("MI+AO", 2)]["p95"], GR[("AO", 2)]["sim"],
                              GR[("AO", 2)]["p95"], GR[("AO vs MI+AO", 2)]["sim"])),
        ("d_k2_ao_above_null", "%d / 6 participants' AO k2 maps above their own shuffle p95"
                               % sum(S[(p, 2)]["AO"]["sim"] > S[(p, 2)]["AO"]["p95"] for p in PARTS)),
        ("d_panel_mark", "points all filled; values inside their own null (shuffle p >= 0.05): "
                         + ", ".join(f"{p} k{k} {c}" for p in PARTS for k in KS for c in ("MI+AO", "AO")
                                     if S[(p, k)][c]["p"] >= 0.05)),
        ("d_scope", "case level, n = 6; both conditions with the walker visible and audible"),
        ("canvas", "16 cm SI contract (03_figures/si_panel_style.py), half tier, 600 dpi; legend as sidecar PNG"),
    ]
    tmp = META.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["key", "value"])
        w.writerows(meta)
    os.replace(tmp, META)
    print("wrote", META)


def panel(P, G):
    """Per-participant paired AO -> MI+AO |c| per harmonic (re-prep, pooled band = the tested input)."""
    fig, ax = SI.panel(plt, "half", 55.0, left_mm=12.5)
    order = ("AO", "MI+AO")
    xs = {}
    for gi, k in enumerate(KS):
        base = 2.5 * gi
        if gi % 2 == 0:
            ax.axvspan(base - 0.62, base + 1.62, facecolor="#F4F4F4", edgecolor="none", zorder=0)
        for j, c in enumerate(order):
            xs[(k, c)] = base + j
        for i, p in enumerate(PARTS):
            off = (i - 2.5) * 0.07
            y = [P[(p, k)].abs_c_ao_pooled_band, P[(p, k)].abs_c_miao_pooled_band]
            x = [xs[(k, c)] + off for c in order]
            ax.plot(x, y, color=PAL.PALE, lw=SI.LW_MINOR, zorder=2, solid_capstyle="round")
            for c, xi, yi in zip(order, x, y):
                ax.plot([xi], [yi], marker="o", ms=3.0, ls="none", zorder=4,
                        mfc=COL[c], mec=COL[c], mew=SI.LW_AXIS)
    ax.set_xticks([2.5 * gi + 0.5 for gi in range(len(KS))])
    ax.set_xticklabels([f"$\\mathrm{{k_{k}}}$" for k in KS], fontsize=SI.TITLE_PT)
    ax.set_ylabel("Gait-phase locking, $|c|$")
    ax.set_xlim(-0.75, 2.5 * (len(KS) - 1) + 1.75)
    SI.finish(ax)
    hi = max(max(r.abs_c_miao_pooled_band, r.abs_c_ao_pooled_band) for r in P.values())
    ax.set_ylim(0, hi * 1.16)
    for k in KS:
        PS.sig_bracket(ax, xs[(k, "AO")] - 0.20, xs[(k, "MI+AO")] + 0.20, G[("perm", k)]["pfdr_two"],
                       hi * 1.035, tick=hi * 0.022, lw=SI.LW_REF,
                       star_pt=SI.TITLE_PT, ns_pt=SI.TICK_PT)
    return fig


def panel_d(S, band, T):
    """Per-participant paired AO -> MI+AO similarity to the healthy LPO map per harmonic, on the channel-shuffle
    null band (10th-90th shaded, 95th dashed; Fig. 6c idiom)."""
    from matplotlib.colors import to_rgba
    fig, ax = SI.panel(plt, "half", 55.0, left_mm=9.0)
    order = ("AO", "MI+AO")
    xs = {}
    for gi, k in enumerate(KS):
        base = 2.5 * gi
        b = band[k]
        ax.fill_between([base - 0.62, base + 1.62], b["p10"], b["p90"], facecolor=to_rgba(NULL_FILL, 0.35),
                        edgecolor="none", zorder=0)
        ax.plot([base - 0.62, base + 1.62], [b["p95"]] * 2, color=NULL_EDGE, lw=SI.LW_REF * 1.4, ls=(0, (4, 2)), zorder=1)
        for j, c in enumerate(order):
            xs[(k, c)] = base + j
        for i, p in enumerate(PARTS):
            off = (i - 2.5) * 0.07
            y = [S[(p, k)]["AO"]["sim"], S[(p, k)]["MI+AO"]["sim"]]
            x = [xs[(k, c)] + off for c in order]
            ax.plot(x, y, color=PAL.PALE, lw=SI.LW_MINOR, zorder=2, solid_capstyle="round")
            for c, xi, yi in zip(order, x, y):
                ax.plot([xi], [yi], marker="o", ms=3.0, ls="none", zorder=4,
                        mfc=COL[c], mec=COL[c], mew=SI.LW_AXIS)
    ax.set_xticks([2.5 * gi + 0.5 for gi in range(len(KS))])
    ax.set_xticklabels([f"$\\mathrm{{k_{k}}}$" for k in KS], fontsize=SI.TITLE_PT)
    ax.set_ylabel("Similarity to healthy map")
    ax.set_xlim(-0.75, 2.5 * (len(KS) - 1) + 1.75)
    SI.finish(ax)
    hi = max(S[(p, k)][c]["sim"] for p in PARTS for k in KS for c in order)
    ax.set_ylim(0, hi * 1.16)
    for k in KS:
        PS.sig_bracket(ax, xs[(k, "AO")] - 0.20, xs[(k, "MI+AO")] + 0.20, T[k]["pfdr_two"],
                       hi * 1.035, tick=hi * 0.022, lw=SI.LW_REF,
                       star_pt=SI.TITLE_PT, ns_pt=SI.TICK_PT)
    return fig


def write_legend_d():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    from matplotlib.legend_handler import HandlerTuple
    key = lambda col: Line2D([], [], marker="o", ls="none", ms=4, mfc=col, mec=col, mew=0.6)
    band = (Patch(facecolor=to_rgba(NULL_FILL, 0.35), edgecolor="none"),
            Line2D([], [], color=NULL_EDGE, lw=0.8, ls=(0, (4, 2))))
    import matplotlib.pyplot as _plt
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    f = _plt.figure(figsize=(2.6, 0.8))
    f.legend(handles=[key(COL["MI+AO"]), key(COL["AO"]), band],
             labels=["Imagined walking", "Observation only", "Channel-shuffle null (10th–90th; dashed, 95th)"],
             handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, loc="center", frameon=False, fontsize=7, ncol=1)
    p = PANEL_DIR / f"S11d_{SLUG_D}_legend.png"
    f.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    _plt.close(f)
    print("wrote", p)


def write_stats_d(S, GR, band, T):
    lines = ["Supplementary Fig. 11d - similarity of each condition's k-map to the healthy cohort map "
             "(leave-participant-out), imagined walking vs observation only (n = 6)",
             "input: as-run complex maps; null: channel shuffle, %d per map, seed %d; brackets: paired t, two-sided, "
             "BH-FDR across k1-k3" % (N_SHUF, SHUF_SEED), ""]
    for k in KS:
        t, b = T[k], band[k]
        lines.append(f"k{k}: mean delta {t['mean_delta']:+.6f} [t-CI {t['ci_lo']:+.6f}, {t['ci_hi']:+.6f}], positive "
                     f"{t['n_pos']}/6; paired t({t['df']}) = {t['t']:.6f}, p {t['p_two']:.6f}, pFDR {t['pfdr_two']:.6f}; "
                     f"null band p10 {b['p10']:.6f} p90 {b['p90']:.6f} p95 {b['p95']:.6f} (n {b['n']})")
        for c in ("MI+AO", "AO"):
            lines.append(f"     {c:5s}: " + ", ".join(
                f"{p} {S[(p, k)][c]['sim']:.4f} (p95 {S[(p, k)][c]['p95']:.3f}, pct {S[(p, k)][c]['pct']:.1f})" for p in PARTS))
        lines.append(f"     group maps vs healthy (other 36 participants, n = 68): MI+AO {GR[('MI+AO', k)]['sim']:.6f} (p95 "
                     f"{GR[('MI+AO', k)]['p95']:.6f}), AO {GR[('AO', k)]['sim']:.6f} (p95 {GR[('AO', k)]['p95']:.6f}); "
                     f"AO vs MI+AO {GR[('AO vs MI+AO', k)]['sim']:.6f}")
    p = PANEL_DIR / f"S11d_{SLUG_D}_stats.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", p)
    print("\n".join(lines))


def write_legend():
    from matplotlib.lines import Line2D
    key = lambda col, fill: Line2D([], [], marker="o", ls="none", ms=4,
                                   mfc=col if fill else "white", mec=col, mew=0.6)
    export_legend([key(COL["MI+AO"], True), key(COL["AO"], True)],
                  "S11", "c", SLUG, ncol=1, fontsize=7, figsize=(2.2, 0.7),
                  labels=["Imagined walking", "Observation only"])


def write_stats(P, G):
    lines = ["Supplementary Fig. 11c - gait-phase locking, imagined walking vs observation only (n = 6)",
             "input: re-prep |c|, pooled band (both tests); BH-FDR across k1-k3", ""]
    for k in KS:
        t, q = G[("bbpooled", k)], G[("perm", k)]
        lines.append(f"k{k}: mean delta {t['mean_delta']:+.6f} [t-CI {t['ci_lo']:+.6f}, {t['ci_hi']:+.6f}], "
                     f"positive {t['n_pos']}/6; MAIN permutation p two-sided {q['p_two']:.6g} pFDR {q['pfdr_two']:.6g} "
                     f"(one-sided {q['p_one']:.6g} pFDR {q['pfdr_one']:.6g}, not quoted); companion paired t({t['df']}) = "
                     f"{t['t']:.6f}, p {t['p_two']:.6f}, pFDR {t['pfdr_two']:.6f}")
        lines.append("     per participant delta: " + ", ".join(
            f"{p} {P[(p, k)].delta_pooled_band:+.4f}" for p in PARTS))
    p = PANEL_DIR / f"S11c_{SLUG}_stats.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", p)
    print("\n".join(lines))


def main():
    b, gv, ctx = gates()
    SM = similarity(ctx, gv)
    L, P, G = build(b)
    write_source_data(L, P, G, SM)
    write_metadata(G, gv, SM)
    SI.apply(plt)
    fig = panel(P, G)
    SI.save(fig, PANEL_DIR / f"S11c_{SLUG}.png")
    plt.close(fig)
    fig = panel_d(SM[0], SM[2], SM[3])
    SI.save(fig, PANEL_DIR / f"S11d_{SLUG_D}.png")
    plt.close(fig)
    write_legend()
    write_legend_d()
    write_stats(P, G)
    write_stats_d(*SM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
