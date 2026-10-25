"""Figure 6 — Representational geometry of patient imagined walking

patient<->healthy harmonic gait-phase TOPOGRAPHY similarity

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import numpy as np

import fig3_harmonic_phase_responder_split as H

ROOT = Path(__file__).resolve().parents[1]
PAT = (ROOT / "data/decoder_output/diag_patient_harmonic_phase/"
              "persession_harmonic_phase_cycle9_complex.csv")
HEAL = (ROOT / "data/decoder_output/diag_harmonic_phase_responder_split/"
               "persession_harmonic_phase_cycle9_complex.csv")
OUT_PNG = ROOT / "figures_out" / "fig6" / "fig6_patient_healthy_harmonic_similarity.png"
SRC = ROOT / "04_source_data" / "fig6" / "fig6_patient_healthy_harmonic_similarity_source_data.csv"
META = ROOT / "04_source_data" / "fig6" / "fig6_patient_healthy_harmonic_similarity_metadata.csv"

BUILD_ID = "main"
PAT_TAG = "diag_patient_harmonic_phase"
HEAL_TAG = "diag_harmonic_phase_responder_split"

BLUE, GREY = "#0072B2", "#9aa0a6"
NULLGREY = "#cfd2d6"
PATCOL = "#962014"
N_PERM = 1000
SEED = 20260613

P1_SESSIONS = ["pat01_sess04", "pat01_sess05", "pat01_sess06",
               "pat01_sess07", "pat01_sess08", "pat01_sess09"]
P2_SESSIONS = ["pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
               "pat02_sess09"]
POST_OPERATIVE = P1_SESSIONS + P2_SESSIONS
PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]
KDISP = {1: r"$\mathrm{k}_1$", 2: r"$\mathrm{k}_2$", 3: r"$\mathrm{k}_3$"}


def load(path):
    H.DUMP = path
    sessions, channels, fracs, harms, A = H.load_dump()
    return sessions, channels, fracs, harms, A


def gmean(A, mask=None):
    sub = A if mask is None else A[mask]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def coh(P, Hh):
    """|complex spatial coherence| over finite ch x point; P,Hh complex 2D [ch,point]."""
    p = P.ravel(); h = Hh.ravel()
    ok = np.isfinite(p.real) & np.isfinite(p.imag) & np.isfinite(h.real) & np.isfinite(h.imag)
    p, h = p[ok], h[ok]
    if p.size < 5:
        return np.nan
    denom = np.linalg.norm(p) * np.linalg.norm(h)
    return float(np.abs(np.vdot(h, p)) / denom) if denom > 0 else np.nan


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 2.2,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True); SRC.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    p_sess, p_ch, p_fr, p_h, PA = load(PAT)
    h_sess, h_ch, h_fr, h_h, HA = load(HEAL)
    assert p_h == h_h, "harmonic mismatch"
    common = [c for c in p_ch if c in h_ch]
    pi = [p_ch.index(c) for c in common]; hi = [h_ch.index(c) for c in common]
    PA = PA[:, pi, :, :]; HA = HA[:, hi, :, :]
    assert len(p_fr) == len(h_fr), "cycle-point mismatch"

    idx = {s: i for i, s in enumerate(p_sess)}
    missing = [s for s in POST_OPERATIVE + PRE_OPERATIVE if s not in idx]
    if missing:
        raise SystemExit(f"cohort sessions absent from {PAT.name}: {missing}")
    mask_post = np.zeros(len(p_sess), bool)
    for s in POST_OPERATIVE:
        mask_post[idx[s]] = True
    cohort_of = {**{s: "post_operative" for s in POST_OPERATIVE}, **{s: "pre_operative" for s in PRE_OPERATIVE}}
    dropped = [s for s in p_sess if s not in cohort_of]

    P_grp = gmean(PA, mask_post)
    H_grp = gmean(HA)
    harms = p_h

    rows, persess_rows = [], []
    group_sim, null95, null_mean, null_p = {}, {}, {}, {}
    for hidx, k in enumerate(harms):
        Pk, Hk = P_grp[:, :, hidx], H_grp[:, :, hidx]
        group_sim[k] = coh(Pk, Hk)
        for s in POST_OPERATIVE + PRE_OPERATIVE:
            persess_rows.append((k, cohort_of[s], "P" + s[3:5].lstrip("0"), s,
                                 coh(PA[idx[s], :, :, hidx], Hk)))
        nulls = []
        nch = Pk.shape[0]
        for _ in range(N_PERM):
            perm = rng.permutation(nch)
            nulls.append(coh(Pk[perm, :], Hk))
        nulls = np.array([x for x in nulls if np.isfinite(x)])
        null95[k] = float(np.quantile(nulls, 0.95)); null_mean[k] = float(np.mean(nulls))
        null_p[k] = float((nulls >= group_sim[k]).mean())
        rows.append((k, group_sim[k], null_mean[k], null95[k], null_p[k]))

    post_by_k = {k: [v for kk, coh_, pat, s, v in persess_rows if kk == k and coh_ == "post_operative"]
                 for k in harms}
    pre_by_k = {k: [v for kk, coh_, pat, s, v in persess_rows if kk == k and coh_ == "pre_operative"]
                for k in harms}

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    xs = np.arange(len(harms), dtype=float)
    above = {k: group_sim[k] > null95[k] for k in harms}
    ax.bar(xs, [group_sim[k] for k in harms], width=0.58,
           color=[BLUE if above[k] else GREY for k in harms], zorder=2)
    for j, k in enumerate(harms):
        ax.add_patch(plt.Rectangle((xs[j] - 0.34, null_mean[k]), 0.68, null95[k] - null_mean[k],
                                   facecolor=NULLGREY, edgecolor="none", zorder=3))
        ax.plot([xs[j] - 0.34, xs[j] + 0.34], [null95[k]] * 2, color=GREY, lw=2.0, zorder=4)
    for j, k in enumerate(harms):
        v = post_by_k[k]
        ax.scatter(np.full(len(v), xs[j]) + rng.uniform(-0.15, 0.15, len(v)), v,
                   s=26, color=PATCOL, edgecolor="white", linewidth=0.6, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels([KDISP[k] for k in harms], fontsize=16)
    ax.set_ylabel("Phase-map similarity", fontsize=15, labelpad=6)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", direction="out", length=5, width=2.2, labelsize=13)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="none", label="Above null"),
               Line2D([0], [0], marker="o", color="none", mfc=PATCOL, mec="white", mew=0.6, ms=7,
                      label="Session"),
               Line2D([0], [0], color=GREY, lw=2.0, label="Shuffle null (95%)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11,
              handlelength=1.1, labelspacing=0.35, handletextpad=0.6, borderaxespad=0.2)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.10)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.close(fig)

    with SRC.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "cohort", "patient", "session", "harmonic_k", "similarity",
                    "null_mean", "null_p95", "empirical_p", "above_null_p95"])
        for k, gs, nm, n95, pv in rows:
            w.writerow(["group", "post_operative", "P1+P2", "", k, f"{gs:.6f}", f"{nm:.6f}",
                        f"{n95:.6f}", f"{pv:.4f}", str(bool(gs > n95))])
        for k, coh_, pat, s, v in persess_rows:
            w.writerow(["session", coh_, pat, s, k, f"{v:.6f}", "", "", "", ""])

    with META.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        for k, v in [
            ("figure", "Fig. 6 supporting analysis - patient<->healthy harmonic phase-map "
                       "similarity (group bars = post_operative); NOT one of the lettered panels a-h, "
                       "the plotted panel c/d counterparts are fig6_cross_harmonic_specificity "
                       "and fig6_patient_in_healthy_similarity"),
            ("patient_dump", PAT_TAG),
            ("healthy_dump", HEAL_TAG),
            ("readout", "patient-trained 10-fold CV (representation transfer, not zero-shot)"),
            ("metric", "|complex spatial coherence| over channels x 9 cycle points, per harmonic; "
                       "invariant to the arbitrary global reference phase"),
            ("group_map_definition", "patient group map = complex mean over the 11 post_operative sessions ONLY; "
                                     "healthy group map = complex mean over all 80 sessions"),
            ("post_operative_set", "pat01 sess04-09 + pat02 sess03/05/06/08/09 (11 sessions) - group map + drawn bars"),
            ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions) - per-session Source Data rows "
                              "ONLY, cohort='pre_operative', DESCRIPTIVE; excluded from the group map and not drawn"),
            ("n_common_channels", len(common)),
            ("n_cycle_points", len(p_fr)),
            ("n_patient_sessions_group", int(mask_post.sum())),
            ("n_healthy_sessions", len(h_sess)),
            ("bar_colour_rule", f"blue {BLUE} = group similarity above the channel-shuffle null 95th "
                                f"percentile; grey {GREY} = not. Data-driven, nothing hardcoded."),
            ("group_similarity_k1", f"{group_sim[1]:.6f}"),
            ("group_similarity_k2", f"{group_sim[2]:.6f}"),
            ("group_similarity_k3", f"{group_sim[3]:.6f}"),
            ("pre_operative_mean_similarity", "; ".join(f"k{k} {np.mean(pre_by_k[k]):.4f}" for k in harms)),
            ("claim", "channel-space diagnostic, not source localization: patient gait-cycle harmonic "
                      "topography matches healthy above a channel-shuffle null, strongest at the step "
                      "harmonic k2; case-level, no population claim"),
            ("output_png", "figures_out/fig6/fig6_patient_healthy_harmonic_similarity.png"),
        ]:
            w.writerow([k, v])

    print(f"common_channels={len(common)}  patient_group={int(mask_post.sum())} (post_operative)  "
          f"healthy_sessions={len(h_sess)}")
    for k in harms:
        print(f"  k{k}: group_sim={group_sim[k]:.6f}  null_mean={null_mean[k]:.6f}  "
              f"null_p95={null95[k]:.6f}  p={null_p[k]:.4f}  above={group_sim[k] > null95[k]}")
        print(f"        post_operative per-session={[round(v, 3) for v in post_by_k[k]]}")
        print(f"        pre_operative  per-session={[round(v, 3) for v in pre_by_k[k]]} (descriptive only)")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {SRC}\nwrote {META}")


if __name__ == "__main__":
    main()
