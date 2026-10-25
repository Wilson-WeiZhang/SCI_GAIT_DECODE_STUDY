"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

Causal-FIR calibration node (boxed tile + transparent icon)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import h5py

for _c in ("/System/Library/Fonts/Supplemental/Arial.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if Path(_c).exists():
        fm.fontManager.addfont(_c)
plt.rcParams["font.family"] = "Arial"

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "data/decoder_output/trial_dump_imagined_walking/subj11_sess01_fold6_trial_dump.mat"
EXAMPLE_TRIAL = 4
OUTDIR = ROOT / "figures_out" / "fig2" / "fig2b_elements"
NODE_OUT = OUTDIR / "fig2b_node_causal_fir.png"
ICON_OUT = OUTDIR / "fig2b_el_fir_fixed.png"
import sys as _sys
_sys.path.insert(0, str(ROOT / "code"))
import analysis_config as _ec
_ec.require("main", "main")
NODE_OUT, ICON_OUT = _ec.scoped(NODE_OUT), _ec.scoped(ICON_OUT)
GREY = "#9a9a9a"
JOINT_C = "#f34142"


def _load_current(joint_name):
    """: the Fig. 2d/e example trial (same ids as fig2b_schematic_elements), before = A_pred (stage 1)."""
    import dump_h5
    from fig2b_schematic_elements import MAIN_SESSION, MAIN_TRIAL_IDX
    tr = dump_h5.load_trial(_ec.path("h_iw_dump_dir") / f"{MAIN_SESSION}_dump.h5", MAIN_TRIAL_IDX, ("A_pred", "C_pred"))
    j = tr["joints"].index(joint_name)
    return tr["fs"], tr["time_sec"], tr["A_pred"][j], tr["C_pred"][j]


def load(trial=EXAMPLE_TRIAL, joint_name="LHipW X"):
    if not _ec.IS_DEFAULT:
        fs, t, before, after = _load_current(joint_name)
        return _window(fs, t, before, after)
    with h5py.File(DUMP, "r") as f:
        def deref(name, i):
            d = f[name]
            return np.array(f[d[i, 0]]) if isinstance(d[i, 0], h5py.Reference) else np.array(d[i, 0])

        def labels():
            g = f[f["meta/target_labels"][0, 0]]
            out = []
            for j in range(g.shape[0]):
                s = f[g[j, 0]] if isinstance(g[j, 0], h5py.Reference) else g[j, 0]
                out.append("".join(chr(int(c)) for c in np.array(s).ravel()))
            return out

        fs = float(np.array(f["meta/fs"]).squeeze())
        joint = labels().index(joint_name)
        t = deref("time_sec", trial).ravel()
        before = deref("P_pred", trial)[joint]
        after = deref("C_pred", trial)[joint]
    return _window(fs, t, before, after)


def _window(fs, t, before, after):
    m = np.isfinite(before) & np.isfinite(after)
    w = np.where(m)[0]
    w = w[: int(6 * fs)] if w.size > int(6 * fs) else w
    tt = t[w] - t[w][0]

    def z(x):
        return (x[w] - np.mean(x[w])) / (np.std(x[w]) + 1e-12)

    return tt, z(before), z(after)


def _traces(ax, tt, zb, za, lw_scale=1.0):
    ax.plot(tt, zb, color=GREY, lw=1.6 * lw_scale, ls=(0, (4, 3)))
    ax.plot(tt, za, color=JOINT_C, lw=2.2 * lw_scale)
    ax.set_xlim(tt[0], tt[-1])
    lo, hi = float(min(zb.min(), za.min())), float(max(zb.max(), za.max()))
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.60 * (hi - lo))
    ax.axis("off")
    ax.plot([], [], color=GREY, lw=1.6 * lw_scale, ls=(0, (4, 3)), label="before")
    ax.plot([], [], color=JOINT_C, lw=2.2 * lw_scale, label="after")


def render_icon(tt, zb, za):
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    _traces(ax, tt, zb, za)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=8, handlelength=1.7, columnspacing=1.2)
    fig.savefig(ICON_OUT, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", ICON_OUT)


def render_node(tt, zb, za):
    fig = plt.figure(figsize=(3.8, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "Causal FIR calibration", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#141414")
    ax.add_patch(FancyBboxPatch((0.05, 0.07), 0.90, 0.75,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                transform=ax.transAxes, fill=True, facecolor="white",
                                edgecolor="#141414", linewidth=3.0))
    pax = fig.add_axes([0.16, 0.16, 0.68, 0.45])
    _traces(pax, tt, zb, za)
    pax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False,
               fontsize=10, handlelength=1.9, columnspacing=1.5)
    fig.savefig(NODE_OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", NODE_OUT)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tt, zb, za = load()
    render_icon(tt, zb, za)
    render_node(tt, zb, za)
    print("font:", fm.findfont("Arial"))


if __name__ == "__main__":
    main()
