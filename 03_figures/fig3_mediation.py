"""Figure 3 — Gait-phase locking of the imagined-walking representation

panel j — mediation: volitional control acts on decoding THROUGH neural locking

Renders one panel or table; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import analysis_config as ec

ec.require("main", "main")
plt.rcParams.update({"font.family": "Arial", "font.size": 8})
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "04_source_data" / "fig3"
OUT = ec.scoped(ROOT / "figures_out" / "fig3" / "fig3_mediation.png")
C_X = "#6f6f6f"; C_M = "#0072B2"; C_Y = "#3A3A3A"

m = pd.read_csv(ec.sd("fig3_master_source_data.csv")).copy()
m["subj"] = m.session.str.split("_sess").str[0]; m["lock"] = m.mi_k1 - m.rest_k1
d = m[["Q6_avoid", "lock", "r_reference", "subj"]].dropna().rename(columns={"Q6_avoid": "X", "lock": "M", "r_reference": "Y"})
for c in ["X", "M", "Y"]:
    d[c] = (d[c] - d[c].mean()) / d[c].std()
a = sm.OLS(d.M, sm.add_constant(d.X)).fit()
bf = sm.OLS(d.Y, sm.add_constant(d[["X", "M"]])).fit()
tot = sm.OLS(d.Y, sm.add_constant(d.X)).fit()
aA, pa = a.params["X"], a.pvalues["X"]
bB, pb = bf.params["M"], bf.pvalues["M"]
cp, pcp = bf.params["X"], bf.pvalues["X"]
cT, pcT = tot.params["X"], tot.pvalues["X"]
rng = np.random.default_rng(0); n_s = len(d); inds = []; cps = []
for _ in range(1000):
    dd = d.iloc[rng.integers(0, n_s, n_s)]
    try:
        a_b = sm.OLS(dd.M, sm.add_constant(dd.X)).fit().params["X"]
        b_f = sm.OLS(dd.Y, sm.add_constant(dd[["X", "M"]])).fit()
        inds.append(a_b * b_f.params["M"]); cps.append(b_f.params["X"])
    except Exception:
        pass
lo, hi = np.percentile(inds, [2.5, 97.5]); ind = aA * bB
cp_lo, cp_hi = np.percentile(cps, [2.5, 97.5])
inds = np.asarray(inds); n_boot = len(inds)
p_boot = 2 * min((inds <= 0).mean(), (inds >= 0).mean())
p_boot_str = f"< {1/n_boot:.0e}" if p_boot == 0 else f"= {p_boot:.3g}"

def star(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

FS   = 1.2*1.3
INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"
plt.rcParams.update({"font.family": "Arial", "font.size": 7*FS, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.6*1.8, 2.9), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
_LW_BOX, _LW_P, _LW_PNS, _FS_T = ((2.1, 2.4, 2.1, 7*FS))

BW, BH = 0.235/1.8*1.5, 0.150*1.5
nodes = {"X": (0.135, 0.175, "Volitional\ncontrol"),
         "M": (0.500, 0.635, "Gait-phase\nlocking ($\\mathrm{k}_1$)"),
         "Y": (0.865, 0.175, "Decoding\naccuracy")}
for k, (x, y, lab) in nodes.items():
    ax.add_patch(FancyBboxPatch((x-BW/2, y-BH/2), BW, BH, boxstyle="round,pad=0.012",
                                fc="white", ec=INK, lw=_LW_BOX, zorder=3))
    ax.text(x, y, lab, ha="center", va="center", fontsize=_FS_T, color=INK, zorder=4)

def arrow(p1, p2, col, lw, ls="-", rad=None):
    cs = f"arc3,rad={rad}" if rad is not None else "arc3,rad=0"
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, connectionstyle=cs,
                                shrinkA=2, shrinkB=2, mutation_scale=13), zorder=2)

arrow((0.233, 0.2875), (0.402, 0.5225), INK, _LW_P)
arrow((0.598, 0.5225), (0.767, 0.2875), INK, _LW_P)
arrow((0.238, 0.175), (0.762, 0.175), MID, _LW_PNS, ls=(0, (4, 3)))
arrow((0.135, 0.520), (0.865, 0.520), INK, _LW_P, rad=-0.40)

fmt = lambda v: f"{v:.2f}".replace("-", "\u2212")
ax.text(0.338, 0.318, f"a = {fmt(aA)} {star(pa)}", ha="center", va="center",
        fontsize=7*FS, color=INK, zorder=5)
ax.text(0.662, 0.318, f"b = {fmt(bB)} {star(pb)}", ha="center", va="center",
        fontsize=7*FS, color=INK, zorder=5)
ax.text(0.500, 0.826, f"Indirect a\u00d7b = {fmt(ind)} {star(p_boot if p_boot > 0 else 1e-9)}",
        ha="center", va="center", fontsize=7*FS, color=INK)
ax.text(0.500, 0.243, f"Direct c\u2032 = {fmt(cp)} {star(pcp)}", ha="center", va="center",
        fontsize=7*FS, color=MID)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
from safe_write import safe_write
safe_write(ec.sd("fig3_mediation_source_data.csv"), pd.DataFrame(
    [("a_X_to_M", aA, pa), ("b_M_to_Y_given_X", bB, pb), ("c_total_X_to_Y", cT, pcT),
     ("cprime_direct_X_to_Y_given_M", cp, pcp), ("indirect_axb", ind, p_boot),
     ("indirect_CI_lo", lo, np.nan), ("indirect_CI_hi", hi, np.nan),
     ("indirect_boot_p_two_sided", p_boot, np.nan), ("indirect_n_boot", n_boot, np.nan),
     ("cprime_CI_lo", cp_lo, np.nan), ("cprime_CI_hi", cp_hi, np.nan)],
    columns=["path", "beta", "p"]).to_csv(index=False, lineterminator="\n"))
print(f"wrote {OUT}")
print(f"  a={aA:+.2f}({star(pa)}) b={bB:+.2f}({star(pb)}) c={cT:+.2f} c'={cp:+.2f}[{cp_lo:+.2f},{cp_hi:+.2f}]({star(pcp)}) indirect={ind:+.2f}[{lo:+.2f},{hi:+.2f}] boot p {p_boot_str} (n_boot={n_boot}, n<=0: {(inds<=0).sum()})")


import csv as _csv
import os as _osw
from datetime import date as _date


def _write_meta(_path, _rows):
    _tmp = _path.parent / (_path.name + ".tmp")
    with _tmp.open("w", newline="", encoding="utf-8") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["key", "value"])
        _w.writerows([(k, v) for k, v in _rows])
    _osw.replace(_tmp, _path)
    print("wrote", _path)


def _libvers(_mods):
    """Runtime library versions, so a re-run that moves the last digits is explainable."""
    import importlib as _il
    import sys as _sys
    _out = ["python " + _sys.version.split()[0]]
    for _m in _mods:
        try:
            _out.append(_m + " " + _il.import_module(_m).__version__)
        except Exception:
            _out.append(_m + " unavailable")
    return "; ".join(_out)


def _era_of_master(_builder):
    """Decoder build named by the build_id row of the master metadata (written by fig3_build_master.py)."""
    _meta = ec.sd("fig3_master_main_metadata.csv")
    _build_id = pd.read_csv(_meta).set_index("key")["value"].get("build_id")
    if not isinstance(_build_id, str):
        raise SystemExit(f"INPUT CHECK: no `build_id` row in {_meta} (written by {_builder.name}); "
                         "cannot date the master. Refusing to stamp an build_id.")
    ec.check_tag(_build_id, "decoder", f"build_id row of {_meta.name}")
    return _build_id

_BUILDER = ROOT / "03_figures" / "fig3_build_master.py"
_ERA = _era_of_master(_BUILDER)
_ERA_H = "main"
_write_meta(ec.sd("fig3_mediation_metadata.csv"), [
    ("figure", "Fig. 3 panel j - statistical mediation: volitional imagery control (Q6) acts on "
               "decoding THROUGH task-evoked gait-phase locking (k1)"),
    ("producer", "03_figures/fig3_mediation.py"),
    ("generated", _date.today().isoformat()),
    ("input", ec.sd("fig3_master_source_data.csv").relative_to(ROOT).as_posix()),
    ("input_producer", "03_figures/fig3_build_master.py"),
    ("n_cohort_sessions", int(m.session.nunique())),
    ("n_analysed", int(len(d))),
    ("standardisation", "X, M, Y z-scored (pandas std, ddof=1) before every OLS"),
    ("bootstrap_p", "two-sided percentile bootstrap p = 2*min(P(a*b<=0), P(a*b>=0)); reported "
                    "as p < 1/B when no resample crosses zero"),
    ("library_versions", _libvers(("numpy", "scipy", "pandas", "matplotlib", "statsmodels"))),
    ("outputs", f"{OUT.relative_to(ROOT).as_posix()}; "
                f"{ec.sd('fig3_mediation_source_data.csv').relative_to(ROOT).as_posix()}"),
])
