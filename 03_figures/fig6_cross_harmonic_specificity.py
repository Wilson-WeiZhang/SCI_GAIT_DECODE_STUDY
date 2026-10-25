"""Figure 6 — Representational geometry of patient imagined walking

panel d — cross-harmonic specificity matrix (patient <-> healthy, first-order maps)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Writes:
  figures_out/fig6/fig6_cross_harmonic_specificity.png
  04_source_data/fig6_cross_harmonic_specificity_{source_data,metadata}.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fig6_panel_style as S
from matplotlib.colors import LinearSegmentedColormap

SEQ_CEIL = '#333333'
TEXT_FLIP = 0.72
SEQ_LIGHT = LinearSegmentedColormap.from_list('greys_capped', ['#FFFFFF', SEQ_CEIL])

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import fig3_harmonic_phase_responder_split as H

PAT_TAG = "diag_patient_harmonic_phase"
HEAL_TAG = "diag_harmonic_phase_responder_split"
PAT_CSV = ROOT / "data/decoder_output" / PAT_TAG / "persession_harmonic_phase_cycle9_complex.csv"
HEAL_CSV = ROOT / "data/decoder_output" / HEAL_TAG / "persession_harmonic_phase_cycle9_complex.csv"
SIM_SD = ROOT / "fig6_patient_healthy_harmonic_similarity_source_data.csv (not distributed)"
OUT_PNG = ROOT / "figures_out/fig6/fig6_cross_harmonic_specificity.png"
SRC = ROOT / "04_source_data/fig6/fig6_cross_harmonic_specificity_source_data.csv"
META = ROOT / "04_source_data/fig6/fig6_cross_harmonic_specificity_metadata.csv"

POST_OPERATIVE = ["pat01_sess04", "pat01_sess05", "pat01_sess06", "pat01_sess07", "pat01_sess08",
            "pat01_sess09", "pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
            "pat02_sess09"]
KD = {1: r"$\mathrm{k}_1$", 2: r"$\mathrm{k}_2$", 3: r"$\mathrm{k}_3$"}

S.apply(plt)
PANEL_W_MM, BOX_MM_C = 40.0, 22.0
TITLE_PT, TICK_PT, CELL_PT = 5.6, 4.8, 4.8
AXIS_LW, DIAG_LW = 0.5, 1.0
plt.rcParams.update({"font.size": TITLE_PT, "axes.linewidth": AXIS_LW,
                     "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0})

def load(path):
    H.DUMP = path
    return H.load_dump()


def gmean(A, idx):
    sub = A[idx]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def coh(P, Hh):
    p = np.asarray(P).ravel()
    h = np.asarray(Hh).ravel()
    ok = np.isfinite(p.real) & np.isfinite(p.imag) & np.isfinite(h.real) & np.isfinite(h.imag)
    p, h = p[ok], h[ok]
    denom = np.linalg.norm(p) * np.linalg.norm(h)
    return float(np.abs(np.vdot(h, p)) / denom) if denom > 0 else np.nan


def main():
    ps, pch, _, _, PA = load(PAT_CSV)
    hs, hch, _, _, HA = load(HEAL_CSV)
    assert list(pch) == list(hch), "channel order mismatch between dumps"
    pidx = [i for i, s in enumerate(ps) if s in POST_OPERATIVE]
    assert len(pidx) == len(POST_OPERATIVE) == 11
    Pg = gmean(PA, pidx)
    Hg = gmean(HA, list(range(len(hs))))

    frozen = {int(r["harmonic_k"]): float(r["similarity"])
              for r in csv.DictReader(open(SIM_SD, encoding="utf-8"))
              if r["row_type"] == "group"}

    M = np.zeros((3, 3))
    for j in range(3):
        for k in range(3):
            M[j, k] = coh(Pg[:, :, j], Hg[:, :, k])
    for k in (1, 2, 3):
        assert abs(M[k - 1, k - 1] - frozen[k]) < 1e-6, \
            f"diagonal k{k} {M[k-1,k-1]:.6f} != frozen {frozen[k]:.6f}"

    fig, ax, cax = S.panel_axes(plt, cbar=True, box_mm=BOX_MM_C, left_mm=7.0, bottom_mm=8.0,
                                top_mm=1.5, right_mm=1.5, cbar_mm=2.0, cbar_gap_mm=1.5,
                                cbar_lab_mm=6.0)
    _PT_T, _PT_K, _PT_CELL, _PT_CB, _PT_CBT, _LW_AX, _LW_D, _TL, _PAD = (
        TITLE_PT, TICK_PT, CELL_PT, CELL_PT, CELL_PT, AXIS_LW, DIAG_LW, 2.0, None)
    im = ax.imshow(M, cmap=SEQ_LIGHT, vmin=0, vmax=1)
    for j in range(3):
        for k in range(3):
            ax.text(k, j, f"{M[j,k]:.2f}", ha="center", va="center", fontsize=_PT_CELL,
                    color="white" if M[j, k] > TEXT_FLIP else S.INK)
    for d in range(3):
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                                   edgecolor=S.INK, lw=_LW_D, clip_on=False))
    ax.set_xticks(range(3), [KD[k] for k in (1, 2, 3)], fontsize=_PT_K)
    ax.set_yticks(range(3), [KD[k] for k in (1, 2, 3)], fontsize=_PT_K)
    ax.set_xlabel("Healthy harmonic", fontsize=_PT_T, labelpad=2)
    ax.set_ylabel("Patient harmonic", fontsize=_PT_T, labelpad=2)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_label("|Coherence|", fontsize=_PT_CB, labelpad=2 if _PAD is None else _PAD)
    cb.ax.tick_params(labelsize=_PT_CBT, width=_LW_AX, length=_TL, **({} if _PAD is None else {"pad": _PAD}))
    cb.outline.set_linewidth(_LW_AX)
    fig.savefig(OUT_PNG, facecolor="white")

    with open(SRC, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_harmonic", "healthy_harmonic", "coherence"])
        for j in range(3):
            for k in range(3):
                w.writerow([j + 1, k + 1, f"{M[j,k]:.6f}"])
    with open(META, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows([
            ["key", "value"],
            ["build_id", "main"],
            ["cohort", "post_operative"],
            ["patient_diag_tag", PAT_TAG],
            ["healthy_diag_tag", HEAL_TAG],
            ["metric", "abs complex spatial coherence over ch x cyclepoint maps; identical to fig6_patient_healthy_harmonic_similarity.py"],
            ["diagonal_frozen_check", "matches fig6_patient_healthy_harmonic_similarity_source_data.csv group rows to 1e-6 (asserted at render)"],
            ["scope", "n=2 SCI patients, case-level, descriptive; decode-independent"],
            ["claim_note", "k2 match harmonic-specific in both directions; off-diagonal max "
             f"{np.max(M - np.diag(np.diag(M))):.6f}"],
        ])
    print(f"wrote {OUT_PNG.name} + Source Data; diag "
          f"{M[0,0]:.3f}/{M[1,1]:.3f}/{M[2,2]:.3f}, offdiag max "
          f"{np.max(M - np.diag(np.diag(M))):.3f}")


if __name__ == "__main__":
    main()
