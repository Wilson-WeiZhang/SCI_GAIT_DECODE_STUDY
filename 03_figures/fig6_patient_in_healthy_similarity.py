"""Figure 6 — Representational geometry of patient imagined walking

panel c — patient sessions placed IN the healthy session-level similarity distribution

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import fig3_harmonic_phase_responder_split as H

HEAL = (ROOT / "data/decoder_output/diag_harmonic_phase_responder_split/"
               "persession_harmonic_phase_cycle9_complex.csv")
PATIENT_SD = ROOT / "fig6_patient_healthy_harmonic_similarity_source_data.csv (not distributed)"
OUT_PNG = ROOT / "figures_out/fig6/fig6_patient_in_healthy_similarity.png"
OUT_LEG = ROOT / "figures_out/fig6/fig6_patient_in_healthy_similarity_legend.png"
SRC = ROOT / "04_source_data/fig6/fig6_patient_in_healthy_similarity_source_data.csv"
META = ROOT / "04_source_data/fig6/fig6_patient_in_healthy_similarity_metadata.csv"


def _ms_mm(v):
    return (round(v / 25.4 * 600) + 0.25) / 600 * 25.4


BUILD_ID = "main"
SEED = 20260824
N_SHUF_PER_SESSION = 5

FW, AX_L = 6.6, 0.88
YLAB_X_IN = 0.42
AX_H, AX_B = 2.40, 0.46
AX_T = 0.10
LEG_PT = 11.3

RED = "#D3B8A2"
RED2 = "#E5E7CD"
BLUE = "#2E6E5E"
GREY = "#9aa0a6"
GREY_DK = "#5f666d"


LEG_W_IN = 1.45


def pad_strip(png, width_in=LEG_W_IN):
    """Pad a tight-cropped legend PNG out to the shared canvas width, ink left-aligned."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    W = int(round(width_in * 300))
    assert im.width <= W, (f"{png.name}: legend ink is {im.width / 300:.2f} in wide, past the "
                           f"shared {width_in} in strip — re-wrap a label, do not widen the strip")
    out = Image.new("RGB", (W, im.height), "white")
    out.paste(im, (0, 0))
    out.save(png, dpi=(300, 300))
    return out.size


def assert_left_clear(png, cols=4, thr=200):
    """The saved PNG must have a blank left gutter — nothing may touch or cross x = 0."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png).convert("L"))
    hit = [int((a[:, c] < thr).sum()) for c in range(cols)]
    assert not any(hit), (f"{png.name}: ink in the first {cols} px columns ({hit}) — the y label is "
                          f"clipped; raise AX_L / YLAB_X_IN, and raise it in BOTH panels so their "
                          f"axes stay aligned")
    return True


def gmean(A, mask=None):
    sub = A if mask is None else A[mask]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def coh(P, Hh):
    """|complex spatial coherence| over finite ch x point (verbatim from the similarity panel)."""
    p = P.ravel(); h = Hh.ravel()
    ok = np.isfinite(p.real) & np.isfinite(p.imag) & np.isfinite(h.real) & np.isfinite(h.imag)
    p, h = p[ok], h[ok]
    if p.size < 5:
        return np.nan
    denom = np.linalg.norm(p) * np.linalg.norm(h)
    return float(np.abs(np.vdot(h, p)) / denom) if denom > 0 else np.nan


def crawford_howell(case, norm):
    """Single-case t vs a control sample (Crawford & Howell 1998). Returns (mean, sd, n, t, df, p)."""
    from scipy import stats as sps
    v = np.asarray(norm, float)
    n = int(v.size)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    t = (float(case) - m) / (sd * np.sqrt(1.0 + 1.0 / n))
    df = n - 1
    return m, sd, n, float(t), df, float(2 * sps.t.sf(abs(t), df))


def rank_percentile(case, norm):
    """Rank percentile of the case among the controls (share strictly below) with its exact
    binomial (Clopper-Pearson) 95% limits."""
    from scipy import stats as sps
    v = np.asarray(norm, float)
    n, r = int(v.size), int(np.sum(v < case))
    lo = sps.beta.ppf(0.025, r, n - r + 1) if r > 0 else 0.0
    hi = sps.beta.ppf(0.975, r + 1, n - r) if r < n else 1.0
    return r, 100.0 * r / n, 100.0 * float(lo), 100.0 * float(hi)


def pbare(p):
    """Project style P value, no leading zero, 3 decimals, '<.001' floor."""
    if p < 0.001:
        return "<.001"
    s = f"{p:.3f}"
    return s[1:] if s.startswith("0.") else s


def pstr(p):
    """Metadata / console form, e.g. 'P=.462' / 'P<.001'."""
    b = pbare(p)
    return f"P{b}" if b.startswith("<") else f"P={b}"


def box_stats(vals):
    """The five drawn lines of a 10-90 box (idiom of 03_figures/fig5b_session_box.py)."""
    v = np.asarray(vals, float)
    p10, q25, med, q75, p90 = (float(np.quantile(v, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90))
    return {"p10": p10, "q25": q25, "median": med, "q75": q75, "p90": p90, "n": v.size}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "axes.linewidth": 1.9,
                         "xtick.major.width": 1.9, "ytick.major.width": 1.9,
                         "xtick.major.size": 3.0, "ytick.major.size": 3.0,
                         "font.size": 12.9, "xtick.labelsize": 11.3, "ytick.labelsize": 11.3,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "pdf.fonttype": 42})
    rng = np.random.default_rng(SEED)

    H.DUMP = HEAL
    h_sess, h_ch, h_fr, harms, HA = H.load_dump()
    nS = len(h_sess)
    subj_arr = np.array([s.split("_sess")[0] for s in h_sess])
    assert len(set(subj_arr)) == 42, f"expected 42 healthy participants, got {len(set(subj_arr))}"
    loo = {k: [] for k in harms}
    null = {k: [] for k in harms}
    for hidx, k in enumerate(harms):
        Ak = HA[:, :, :, hidx]
        for si in range(nS):
            mask = subj_arr != subj_arr[si]
            G = gmean(Ak, mask)
            loo[k].append(coh(Ak[si], G))
            for _ in range(N_SHUF_PER_SESSION):
                perm = rng.permutation(Ak.shape[1])
                null[k].append(coh(Ak[si][perm, :], G))
        loo[k] = np.array(loo[k]); null[k] = np.array(null[k])

    pat = {k: [] for k in harms}
    with open(PATIENT_SD, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["row_type"] != "session":
                continue
            k = int(r["harmonic_k"])
            pat[k].append((r["session"], r["patient"], r["cohort"], float(r["similarity"])))
    n_post = sum(1 for x in pat[harms[0]] if x[2] == "post_operative")
    n_pre = sum(1 for x in pat[harms[0]] if x[2] == "pre_operative")
    assert n_post == 11 and n_pre == 4, f"expected 11 post_operative + 4 pre_operative, got {n_post}+{n_pre}"
    post = {k: [(s, p, c, v) for s, p, c, v in sorted(pat[k]) if c == "post_operative"] for k in harms}
    n_p1 = sum(1 for x in post[harms[0]] if x[1] == "P1")
    n_p2 = sum(1 for x in post[harms[0]] if x[1] == "P2")
    assert n_p1 == 6 and n_p2 == 5, f"expected post_operative = 6 P1 + 5 P2, got {n_p1}+{n_p2}"

    def vals_of(k, who):
        if who == "healthy":
            return loo[k]
        if who == "null":
            return null[k]
        return np.array([v for s, p, c, v in post[k] if p == who])

    boxes = {k: {who: box_stats(vals_of(k, who)) for who in ("healthy", "P1", "P2", "null")}
             for k in harms}

    subj_of = [s.split("_sess")[0] for s in h_sess]
    subj_ids = sorted(set(subj_of))
    assert len(subj_ids) == 42, f"expected 42 healthy subjects, got {len(subj_ids)}"
    tests = {}
    ranks = {}
    for k in harms:
        norm = np.array([np.mean([loo[k][i] for i, sj in enumerate(subj_of) if sj == u])
                         for u in subj_ids])
        assert norm.size == 42
        for who in ("P1", "P2"):
            case = float(np.mean(vals_of(k, who)))
            m, sd, n, t, df, p = crawford_howell(case, norm)
            tests[(k, who)] = (case, m, sd, n, t, df, p)
            ranks[(k, who)] = rank_percentile(case, norm)

    FH = AX_B + AX_H + AX_T
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.subplots_adjust(left=AX_L / FW, right=0.990, bottom=AX_B / FH, top=(AX_B + AX_H) / FH)
    W = 2.30
    LANES = [("healthy", 0.00, "#1A1A1A", BLUE,      0.30),
             ("P1",      0.62, "#1A1A1A", RED,  1.0),
             ("P2",      1.24, "#1A1A1A", RED2, 1.0)]
    BAND_PAD = 0.40
    half, cap, elw = 0.22, 0.10, 1.8
    _lw_whi, _lw_med, _lw_n95, _ls_n95, _lw_bar = ((1.6, 2.4, 1.6, (0, (4, 2)), elw))
    for j, k in enumerate(harms):
        x0 = j * W
        nb = boxes[k]["null"]
        ax.add_patch(Rectangle((x0 - BAND_PAD, nb["p10"]),
                               LANES[-1][1] + 2 * BAND_PAD, nb["p90"] - nb["p10"],
                               facecolor=to_rgba(GREY, 0.35), edgecolor="none", zorder=0))
        ax.plot([x0 - BAND_PAD, x0 + LANES[-1][1] + BAND_PAD],
                [np.quantile(null[k], 0.95)] * 2, color=GREY_DK, lw=_lw_n95, ls=_ls_n95, zorder=1)
        for who, dx, edge, fill, alpha in LANES:
            xi = x0 + dx
            b = boxes[k][who]
            for a, z in ((b["p10"], b["q25"]), (b["q75"], b["p90"])):
                ax.plot([xi, xi], [a, z], color=edge, lw=_lw_whi, zorder=3, solid_capstyle="butt")
            for c in (b["p10"], b["p90"]):
                ax.plot([xi - cap, xi + cap], [c, c], color=edge, lw=_lw_whi, zorder=3)
            ax.add_patch(Rectangle((xi - half, b["q25"]), 2 * half, b["q75"] - b["q25"],
                                   facecolor=to_rgba(fill, alpha), edgecolor=edge, lw=elw,
                                   zorder=4))
            ax.plot([xi - half, xi + half], [b["median"]] * 2, color=edge, lw=_lw_med,
                    zorder=5, solid_capstyle="butt")

    ax.set_xticks([j * W + LANES[-1][1] / 2.0 for j in range(len(harms))])
    K_TICK_SCALE = 1.4
    ax.set_xticklabels([f"$\\mathrm{{k_{k}}}$" for k in harms],
                       fontsize=plt.rcParams["xtick.labelsize"] * K_TICK_SCALE)
    ax.set_ylabel("Phase-map similarity to healthy")
    ax.yaxis.label.set_va("center")
    ax.yaxis.set_label_coords((YLAB_X_IN - AX_L) / (FW * 0.990 - AX_L), 0.5)
    ax.set_xlim(-BAND_PAD - 0.16, (len(harms) - 1) * W + LANES[-1][1] + BAND_PAD + 0.16)
    ax.spines[["top", "right"]].set_visible(False)

    ns_fs = ax.yaxis.label.get_fontsize()
    _ns_off = 2
    ink = [boxes[k][w][q] for k in harms for w, *_ in LANES for q in ("p10", "p90")]
    ink += [boxes[k]["null"][q] for k in harms for q in ("p10", "p90")]
    ink += [float(np.quantile(null[k], 0.95)) for k in harms]
    pad = 0.05 * (max(ink) - min(ink))
    ybar = []
    for j, k in enumerate(harms):
        xa, xb = j * W + LANES[0][1], j * W + LANES[-1][1]
        y = max(boxes[k][w]["p90"] for w, *_ in LANES) + pad
        ybar.append(y)
        ax.plot([xa, xb], [y, y], color="black", lw=_lw_bar, zorder=7, solid_capstyle="butt")
        ax.annotate("ns", ((xa + xb) / 2, y), textcoords="offset points", xytext=(0, _ns_off),
                    ha="center", va="bottom", fontsize=ns_fs, color="0.10",
                    annotation_clip=False, zorder=7)
    f = 0.12
    bot = min(ink) - 0.05 * (max(ink) - min(ink))
    ax.set_ylim(bot, max(ybar) + f * (max(ybar) - bot) / (1 - f))
    fig.set_dpi(300)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)
    assert_left_clear(OUT_PNG)

    LEG = [f"healthy sessions ($n$ = 80)",
           f"patient 2, post-operative ($n$ = {n_p2})",
           f"patient 1, post-operative ($n$ = {n_p1})",
           "channel-shuffle null (10th–90th; dashed, 95th)"]
    lane = {who: (edge, fill, alpha) for (who, dx, edge, fill, alpha) in LANES}
    handles = [Patch(facecolor=to_rgba(lane["healthy"][1], lane["healthy"][2]), edgecolor=lane["healthy"][0], lw=elw),
               Patch(facecolor=to_rgba(lane["P2"][1], lane["P2"][2]), edgecolor=lane["P2"][0], lw=elw),
               Patch(facecolor=to_rgba(lane["P1"][1], lane["P1"][2]), edgecolor=lane["P1"][0], lw=elw),
               Patch(facecolor=to_rgba(GREY, 0.35), edgecolor="none")]
    lfig = plt.figure(figsize=(6.6, 1.0))
    lfig.legend(handles=handles, labels=LEG, loc="center", ncol=2, frameon=False, fontsize=LEG_PT,
                handletextpad=0.7, borderpad=0.0, handlelength=1.4, labelspacing=0.7, columnspacing=1.3)
    lfig.savefig(OUT_LEG, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(lfig)
    lw_px, lh_px = pad_strip(OUT_LEG, width_in=6.6)

    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_type", "harmonic_k", "session", "patient", "cohort", "value",
                    "p10", "q25", "median", "q75", "p90", "p95", "n",
                    "norm_mean", "norm_sd", "t", "df", "p",
                    "n_norm_below", "rank_percentile", "rank_ci95_lo", "rank_ci95_hi"])
        blank = [""] * 7
        tail = [""] * 9
        for k in harms:
            for si, s in enumerate(h_sess):
                w.writerow(["healthy_lpo", k, s, "healthy", "healthy_cohort",
                            f"{loo[k][si]:.6f}"] + blank + tail)
            for s, p, cohort, val in post[k]:
                w.writerow(["patient_session", k, s, p, cohort, f"{val:.6f}"] + blank + tail)
            for who, cohort in (("healthy", "healthy_cohort"), ("P1", "post_operative"), ("P2", "post_operative")):
                b = boxes[k][who]
                w.writerow(["box_summary", k, "", who, cohort, ""] +
                           [f"{b[q]:.6f}" for q in ("p10", "q25", "median", "q75", "p90")] +
                           ["", b["n"]] + tail)
            b = boxes[k]["null"]
            w.writerow(["null_summary", k, "", "", "pooled", ""] +
                       [f"{b[q]:.6f}" for q in ("p10", "q25", "median", "q75", "p90")] +
                       [f"{np.quantile(null[k], 0.95):.6f}", b["n"]] + tail)
            for who in ("P1", "P2"):
                case, m, sd, n, t, df, p = tests[(k, who)]
                r, pct, lo, hi = ranks[(k, who)]
                w.writerow(["case_norm_test", k, "", who, "post_operative", f"{case:.6f}"] +
                           [""] * 6 + [n, f"{m:.6f}", f"{sd:.6f}", f"{t:.4f}", df, f"{p:.6f}",
                                       r, f"{pct:.2f}", f"{lo:.2f}", f"{hi:.2f}"])

    stats = {}
    for k in harms:
        hv = loo[k]
        post_vals = [v for s, p, c, v in post[k]]
        pct = [float(np.mean(hv <= v)) * 100 for v in post_vals]
        above_null = sum(v > np.quantile(null[k], 0.95) for v in post_vals)
        stats[k] = (np.median(hv), min(pct), max(pct), above_null, len(post_vals))

    meta = [
        ("metric", "|complex spatial coherence|, verbatim from fig6_patient_healthy_harmonic_similarity.py"),
        ("patient_values_provenance",
         "READ from fig6_patient_healthy_harmonic_similarity_source_data.csv row_type=session - "
         "not recomputed, so this panel cannot drift from the similarity panel"),
    ]
    for k in harms:
        med, lo, hi, ab, npost = stats[k]
        meta.append((f"k{k}_healthy_lpo_median", f"{med:.6f}"))
        meta.append((f"k{k}_post_operative_percentile_range",
                     f"{lo:.0f}-{hi:.0f} pct of the healthy LPO distribution"))
        meta.append((f"k{k}_post_operative_above_null_p95", f"{ab}/{npost}"))
        for who in ("P1", "P2"):
            case, m, sd, n, t, df, p = tests[(k, who)]
            meta.append((f"k{k}_{who}_case_vs_healthy",
                         f"case {case:.3f} vs norm {m:.3f}+/-{sd:.3f}, t({df})={t:.2f}, "
                         f"{pstr(p)}"))
            r, pct, lo, hi = ranks[(k, who)]
            meta.append((f"k{k}_{who}_rank_percentile",
                         f"{r}/42 below = {pct:.1f} pct, exact binomial 95% CI {lo:.1f}-{hi:.1f}"))
    rv = list(ranks.values())
    meta.append(("rank_percentile_range_all",
                 f"{min(x[1] for x in rv):.0f}-{max(x[1] for x in rv):.0f} pct, 95% CI "
                 f"{min(x[2] for x in rv):.0f}-{max(x[3] for x in rv):.0f} "
                 f"(3 harmonics x 2 patients; quoted in Supplementary Note 6)"))
    with open(META, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"]); w.writerows(meta)

    print(f"wrote {OUT_PNG}  (figsize {FW}x{FH:.2f} in, axes {AX_H} in high)")
    print(f"wrote {OUT_LEG}  ({lw_px}x{lh_px} px = {lw_px/300:.2f}x{lh_px/300:.2f} in, "
          f"standalone tall-narrow legend, w:h = 1:{lh_px/lw_px:.2f})")
    for k in harms:
        med, lo, hi, ab, npost = stats[k]
        print(f"  k{k}: healthy LPO median {med:.3f}; post_operative sessions span the "
              f"{lo:.0f}-{hi:.0f} pctile of healthy; {ab}/{npost} above the null p95")
    print("Crawford-Howell single-case t vs 42 subject-level healthy means (two-sided, df=41):")
    for k in harms:
        for who in ("P1", "P2"):
            case, m, sd, n, t, df, p = tests[(k, who)]
            print(f"  k{k} {who}: case {case:.3f} vs norm {m:.3f}+/-{sd:.3f}, "
                  f"t({df}) = {t:+.2f}, {pstr(p)}")


if __name__ == "__main__":
    main()
