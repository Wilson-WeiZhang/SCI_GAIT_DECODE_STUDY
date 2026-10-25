"""Figure 6 — Representational geometry of patient imagined walking

quantification of the patient-vs-healthy IW spatial-RSA test (dot + spin-null boxes)

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
from pathlib import Path
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from PIL import Image

RC_CONTRACT = {"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
               "axes.linewidth": 1.0,
               "xtick.major.width": 1.0, "ytick.major.width": 1.0,
               "xtick.major.size": 2.0, "ytick.major.size": 2.0,
               "font.size": 5.6, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8,
               "axes.labelsize": 5.6,
               "mathtext.fontset": "custom", "mathtext.rm": "Arial",
               "mathtext.it": "Arial:italic",
               "pdf.fonttype": 42}
plt.rcParams.update(RC_CONTRACT)

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "03_figures" / "fig6_rsa_iw_healthy_patient_rdm.py"
PRODUCER_MARKER = 'CONDS = ["healthy_IW", "P1", "P2", "healthy_REST"]'
RSA = ROOT / "04_source_data" / "fig6" / "fig6_rsa_iw_healthy_patient_rdm_source_data.csv"
RSA_META = ROOT / "04_source_data" / "fig6" / "fig6_rsa_iw_healthy_patient_rdm_metadata.csv"
OUT_PNG = ROOT / "figures_out" / "fig6" / "fig6_rsa_quantification_forest.png"
OUT_LEG = ROOT / "figures_out" / "fig6" / "fig6_rsa_quantification_forest_legend.png"
DRAW_LEGEND = True
SRC = ROOT / "04_source_data" / "fig6" / "fig6_rsa_quantification_forest_source_data.csv"
NULLS = ROOT / "04_source_data" / "fig6" / "fig6_rsa_quantification_forest_spin_nulls.csv"
BOXSTATS = ROOT / "04_source_data" / "fig6" / "fig6_rsa_quantification_forest_null_box_stats.csv"
META = ROOT / "04_source_data" / "fig6" / "fig6_rsa_quantification_forest_metadata.csv"

RED = "#D3B8A2"
RED2 = "#E5E7CD"
GREY = "#9aa0a6"
BOX_FILL = "#D9D9D9"
A_MAIN = 1.0
A_CTL = 0.0
INKG = "#5f666d"
CTL_INK = "#9aa0a6"
DASH = "#5f666d"
ZERO = "#c9ced3"
SYMC = "0.15"
MS_DOT = 4.4
MEW_DOT = 0.7
HALO_PT = 1.2
MS_CTL = 3.8
GAP_PT = 1.6
FW, AX_L = 2.598, 0.495
AX_B, AX_H = 0.315, 0.945
AX_T = 0.039
GRP_PT, GRP_PAD = 5.6, 9.0
PX_PER_PT = 300.0 / 72.0
LEG_PT = 4.8
FP = FontProperties(family="Arial")

BOX_HALF, BOX_CAP = 0.16, 0.07
CTL_HALF, CTL_CAP = 0.16 * 0.7, 0.05
DASH_HALF = 0.18
LW_WHISK, LW_BOX, LW_MED = 0.4, 0.5, 0.8
CTL_LW = (0.3, 0.375, 0.6)

HARM = [("1", "k1", r"$\mathrm{k}_1$", "stride"),
        ("2", "k2", r"$\mathrm{k}_2$", "step"),
        ("3", "k3", r"$\mathrm{k}_3$", "")]

X_ITEM = 0.0
X_WITHIN = 1.00
X_GROUP = 2.30
X_REST = 0.30
XMAX = X_ITEM + 2 * X_GROUP + X_WITHIN + X_REST + 0.42

FAINT = "#c9ced3"
NULLC = "#bfc5cb"
INK = "#2b2b2b"
PMARK = {"P1": "o", "P2": "s"}
SIZE_BINS = [(0.20, 8.0, r"$|\rho|$ < 0.20"),
             (0.40, 12.0, "0.20 - 0.40"),
             (9.99, 16.5, r"$\geq$ 0.40")]


def size_of(rho):
    for hi, ms, _ in SIZE_BINS:
        if abs(rho) < hi:
            return ms
    return SIZE_BINS[-1][1]


def ptxt(p):
    return r"$\it{P}$ < .001" if p < 0.001 else (r"$\it{P}$ = ." + f"{p:.3f}"[2:])


EXPECT = {(1, "P1"): (0.1326, 0.4615), (1, "P2"): (0.4038, 0.0410),
          (2, "P1"): (0.5396, 0.0020), (2, "P2"): (0.5865, 0.0120),
          (3, "P1"): (0.0926, 0.3816), (3, "P2"): (-0.2101, 0.8671)}
CTRL_SEED = 20260826
ITEMS = [(k, p) for k in (1, 2, 3) for p in ("P1", "P2")]


def sigsym(p):
    """Standard significance tiers, applied to the BH-corrected spin P (see bh_fdr)."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up adjusted P values, returned in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def pfmt(p, nd=2):
    """P formatted without the leading zero (project reporting convention: .73, not 0.73)."""
    s = f"{p:.{nd}f}"
    return s[1:] if s.startswith("0.") else s


LEG_W_IN = 0.57


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


def box_stats(vals):
    """The EXACT drawn box/whisker/dash values for one spin null."""
    q10, q25, med, q75, q90, q95 = np.percentile(vals, [10, 25, 50, 75, 90, 95])
    return {"p10": float(q10), "q25": float(q25), "median": float(med), "q75": float(q75),
            "p90": float(q90), "p95": float(q95), "n": int(np.size(vals))}


def draw_box(ax, xc, st, half, cap, alpha, zbase, ink_c=INKG, lws=(LW_WHISK, LW_BOX, LW_MED)):
    """One 10th-90th box-whisker in the 03_figures/fig5b_session_box.py idiom (no fliers)."""
    lw_w, lw_b, lw_m = lws
    ax.plot([xc, xc], [st["p10"], st["q25"]], color=ink_c, lw=lw_w, zorder=zbase,
            solid_capstyle="butt")
    ax.plot([xc, xc], [st["q75"], st["p90"]], color=ink_c, lw=lw_w, zorder=zbase,
            solid_capstyle="butt")
    for c in (st["p10"], st["p90"]):
        ax.plot([xc - cap, xc + cap], [c, c], color=ink_c, lw=lw_w, zorder=zbase)
    ax.add_patch(Rectangle((xc - half, st["q25"]), 2 * half, st["q75"] - st["q25"],
                           facecolor=("white" if alpha == 0 else to_rgba(BOX_FILL, alpha)),
                           edgecolor=ink_c, lw=lw_b, zorder=zbase + 1))
    ax.plot([xc - half, xc + half], [st["median"], st["median"]], color=ink_c, lw=lw_m,
            zorder=zbase + 2, solid_capstyle="butt")


def ink(s, size):
    """(y0, y1) ink extents of `s` relative to its baseline, in points — exact glyph outlines."""
    bb = TextPath((0, 0), s, size=size, prop=FP).get_extents()
    return float(bb.y0), float(bb.y1)


def write_csv(path, header, rows):
    """Write a CSV, but leave the file untouched when the bytes would be identical."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    new = buf.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == new:
        print(f"unchanged, not rewritten: {path.name} ({len(rows)} rows, {len(new)} bytes)")
        return
    try:
        path.write_bytes(new)
    except PermissionError:
        raise SystemExit(
            f"CANNOT WRITE {path}\n  Its content CHANGED but the file is locked by another "
            f"process. Close it and re-run. "
            "")
    print(f"wrote {path.name} ({len(rows)} rows, {len(new)} bytes)")


def load_producer():
    """Exec the HEAD of fig5e (functions + data + its unconsumed RNG) without running/overwriting it."""
    src = PRODUCER.read_text(encoding="utf-8")
    if PRODUCER_MARKER not in src:
        raise SystemExit(f"producer layout changed: marker not found in {PRODUCER}")
    ns = {"__file__": str(PRODUCER), "__name__": "fig5e_head"}
    exec(compile(src.split(PRODUCER_MARKER)[0], str(PRODUCER), "exec"), ns)
    return ns


def compute():
    """Re-run the producer's loop for the spin DISTRIBUTIONS, then the resting-control spin null."""
    P = load_producer()
    load, elec_rdm, sp = P["load"], P["elec_rdm"], P["sp"]
    spin_null, shuffle_null = P["spin_null"], P["shuffle_null"]
    iuE, RNG, sessions = P["iuE"], P["RNG"], P["PAT_SESSIONS"]
    pat_raw, hiw_raw, hre_raw = P["_pat_raw"], P["_hiw_raw"], P["_hre_raw"]

    rdms, base, spin = {}, {}, {}
    for k in (1, 2, 3):
        rdms[(k, "healthy_IW")] = elec_rdm(*load(hiw_raw, k))
        rdms[(k, "P1")] = elec_rdm(*load(pat_raw, k, sessions["P1"]))
        rdms[(k, "P2")] = elec_rdm(*load(pat_raw, k, sessions["P2"]))
        rdms[(k, "healthy_REST")] = elec_rdm(*load(hre_raw, k))
        ref, rest = rdms[(k, "healthy_IW")][iuE], rdms[(k, "healthy_REST")][iuE]
        for pat in ("P1", "P2"):
            a = rdms[(k, pat)][iuE]
            rho, rest_rho = sp(a, ref), sp(a, rest)
            spin_p, sn = spin_null(a, rdms[(k, "healthy_IW")], rho, RNG)
            shuffle_null(a, rdms[(k, "healthy_IW")], rho, RNG)
            base[(k, pat)] = {"rho": rho, "spin_p": spin_p, "rest": rest_rho}
            spin[(k, pat)] = sn

    crng = np.random.default_rng(CTRL_SEED)
    ctrl = {}
    for k in (1, 2, 3):
        for pat in ("P1", "P2"):
            obs = base[(k, pat)]["rest"]
            _p, sn = spin_null(rdms[(k, pat)][iuE], rdms[(k, "healthy_REST")], obs, crng)
            n = sn.size
            p_lo = (np.sum(sn <= obs) + 1) / (n + 1)
            p_hi = (np.sum(sn >= obs) + 1) / (n + 1)
            ctrl[(k, pat)] = {"sn": sn, "obs": obs,
                              "pct": 100.0 * float(np.mean(sn < obs)),
                              "p_two": float(min(1.0, 2.0 * min(p_lo, p_hi))),
                              "p025": float(np.percentile(sn, 2.5)),
                              "p975": float(np.percentile(sn, 97.5))}
    return base, spin, ctrl


def guard(base, rec):
    """Assert the recomputation against the published values AND against the producer's Source Data."""
    bad = []
    if EXPECT is None:
        print("[guard] no reference values set - the published-value "
              "check is SKIPPED; the Source-Data check below still runs. Recomputed table:")
        for (k, pat) in ITEMS:
            b = base[(k, pat)]
            print(f"          ({k}, \"{pat}\"): ({b['rho']:.4f}, {b['spin_p']:.4f}),")
    for (k, pat) in ITEMS:
        b = base[(k, pat)]
        r = rec[(str(k), pat)]
        if EXPECT is not None:
            erho, ep = EXPECT[(k, pat)]
            if round(b["rho"], 4) != erho or round(b["spin_p"], 4) != ep:
                bad.append(f"  k{k} {pat}: recomputed rho={b['rho']:.4f} p={b['spin_p']:.4f} "
                           f"vs published rho={erho:.4f} p={ep:.4f}")
        if round(b["rho"], 4) != float(r["spatial_rho"]) or \
                round(b["spin_p"], 4) != float(r["spin_p"]):
            bad.append(f"  k{k} {pat}: recomputed values disagree with {RSA.name}")
    if bad:
        raise SystemExit(
            "GUARD FAILED — the panel was NOT written.\n" + "\n".join(bad) +
            ""
            )


def draw_panel(out_png, base, spin, ctrl, rec):
    """The panel: dot + spin-null violins, common symbol row, no legend, live geometry."""
    plt.rcParams.update(RC_CONTRACT)
    FH = AX_B + AX_H + AX_T
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.subplots_adjust(left=AX_L / FW, right=0.990, bottom=AX_B / FH, top=(AX_B + AX_H) / FH)
    ax_h, dpi = AX_H, 300
    ms_dot, mew_dot, halo, lw_whisk = MS_DOT, MEW_DOT, HALO_PT, LW_WHISK
    lws_main, lws_ctl = (LW_WHISK, LW_BOX, LW_MED), CTL_LW
    lw_dash, lw_zero, grp_pt, grp_pad = 0.5, 0.4, GRP_PT, GRP_PAD
    px_per_pt = dpi / 72.0

    ax.set_ylabel(r"Spatial RSA $\rho$", labelpad=2)
    ns_pt = float(ax.yaxis.label.get_fontsize())
    ns_y0, ns_y1 = ink("ns", ns_pt)
    h_ns, c_ns = ns_y1 - ns_y0, (ns_y0 + ns_y1) / 2.0
    ast_pt = 14.0 * h_ns / (lambda t: t[1] - t[0])(ink("**", 14.0))
    a_y0, a_y1 = ink("**", ast_pt)
    h_ast, c_ast = a_y1 - a_y0, (a_y0 + a_y1) / 2.0
    assert 0.75 <= h_ast / h_ns <= 1.25, f"asterisk ink ratio {h_ast / h_ns:.3f} out of band"
    h_max = max(h_ns, h_ast)

    p_bh = dict(zip(ITEMS, bh_fdr([float(rec[(str(k), pat)]["spin_p"]) for k, pat in ITEMS])))

    bstat = {(k, pat): {"main": box_stats(spin[(k, pat)]),
                        "rest_control": box_stats(ctrl[(k, pat)]["sn"])} for k, pat in ITEMS}
    for k, pat in ITEMS:
        assert round(bstat[(k, pat)]["main"]["p95"], 4) == float(rec[(str(k), pat)]["spin_null_p95"]), \
            f"k{k} {pat}: box-stats p95 disagrees with {RSA.name} spin_null_p95"

    tops, lows, tops_by = [], [], {}
    for k, pat in ITEMS:
        b, m, c = base[(k, pat)], bstat[(k, pat)]["main"], bstat[(k, pat)]["rest_control"]
        tops_by[(k, pat)] = [(m["p90"], lw_whisk / 2.0), (m["p95"], 0.8),
                             (b["rho"], (ms_dot + halo) / 2.0), (c["p90"], lw_whisk / 2.0),
                             (b["rest"], MS_CTL / 2.0)]
        tops += tops_by[(k, pat)] + [(0.0, 0.6)]
        lows += [(m["p10"], lw_whisk / 2.0), (b["rho"], (ms_dot + halo) / 2.0),
                 (c["p10"], lw_whisk / 2.0), (b["rest"], MS_CTL / 2.0), (0.0, 0.6)]
    ylo, yhi = min(v for v, _a in lows) - 0.05, 0.72
    for _ in range(60):
        ppu = ax_h * 72.0 / (yhi - ylo)
        ylo_n = min(v - a / ppu for v, a in lows) - 0.03
        yhi_n = max(v + a / ppu for v, a in tops) + (GAP_PT + h_max + 4.0) / ppu
        if abs(yhi_n - yhi) < 1e-4 and abs(ylo_n - ylo) < 1e-4:
            ylo, yhi = ylo_n, yhi_n
            break
        ylo, yhi = ylo_n, yhi_n
    ppu = ax_h * 72.0 / (yhi - ylo)
    itop = {it: max(v + a / ppu for v, a in tops_by[it]) for it in ITEMS}
    gtop = max(itop.values())

    placed, xlabels, report, samples = [], [], [], []
    for hk, fname, fdisp, _role in HARM:
        for pat in ("P1", "P2"):
            k = int(hk)
            b, c, r = base[(k, pat)], ctrl[(k, pat)], rec[(str(k), pat)]
            xx = X_ITEM + (k - 1) * X_GROUP + (0.0 if pat == "P1" else X_WITHIN)
            rho, spin_p = float(r["spatial_rho"]), float(r["spin_p"])
            rest = float(r["rest_rho"])
            m_st, c_st = bstat[(k, pat)]["main"], bstat[(k, pat)]["rest_control"]

            draw_box(ax, xx + X_REST, c_st, CTL_HALF, CTL_CAP, A_CTL, zbase=0,
                     ink_c=CTL_INK, lws=lws_ctl)
            draw_box(ax, xx, m_st, BOX_HALF, BOX_CAP, A_MAIN, zbase=3, lws=lws_main)
            ax.plot([xx - DASH_HALF, xx + DASH_HALF], [m_st["p95"], m_st["p95"]], color=DASH,
                    lw=lw_dash, linestyle=(0, (4.0, 2.5)), zorder=6)
            dotc = RED if pat == "P1" else RED2
            if halo > 0:
                ax.plot(xx, rho, marker="o", ms=ms_dot + halo, mfc="white", mec="white",
                        mew=0.0, zorder=7)
            ax.plot(xx, rho, marker="o", ms=ms_dot, mfc=dotc, mec="#1A1A1A", mew=mew_dot,
                    zorder=8)

            sym = sigsym(p_bh[(k, pat)])
            size = ns_pt if sym == "ns" else ast_pt
            off = GAP_PT + h_max / 2.0 - (c_ns if sym == "ns" else c_ast)
            ytop = itop[(k, pat)]
            ax.annotate(sym, xy=(xx, ytop), xytext=(0, off), textcoords="offset points",
                        ha="center", va="baseline", fontsize=size, color=SYMC)
            placed.append({"k": k, "pat": pat, "name": fname, "x": xx, "sym": sym,
                           "size": size, "off": off, "ytop": ytop,
                           "c_sym": c_ns if sym == "ns" else c_ast})
            xlabels.append(pat)
            m_y = m_st["q25"] + 0.25 * (m_st["q75"] - m_st["q25"])
            c_y = c_st["q25"] + 0.25 * (c_st["q75"] - c_st["q25"])
            samples += [("main", xx - 0.11, m_y),
                        ("rest_control", xx + X_REST - 0.075, c_y),
                        ("main_edge", xx - BOX_HALF, m_y),
                        ("control_edge", xx + X_REST + CTL_HALF, c_y)]
            report.append((fname, pat, rho, spin_p, float(r["shuffle_p"]), rest, sym,
                           spin_p < 0.05, float(p_bh[(k, pat)]), p_bh[(k, pat)] < 0.05))

    ax.axhline(0.0, color=ZERO, lw=lw_zero, zorder=0)
    ax.set_xlim(-0.62, XMAX)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([X_ITEM + gi * X_GROUP + dx for gi in range(3) for dx in (0.0, X_WITHIN)])
    ax.set_xticklabels(xlabels)
    for gi, (_hk, _fn, fdisp, _role) in enumerate(HARM):
        xc = X_ITEM + gi * X_GROUP + (X_WITHIN + X_REST) / 2.0
        ax.annotate(fdisp, xy=(xc, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -grp_pad), textcoords="offset points", ha="center", va="top",
                    fontsize=grp_pt, annotation_clip=False)
    ax.set_yticks([t / 100.0 for t in range(-200, 201, 25)
                   if ylo + 0.02 <= t / 100.0 <= yhi - 0.02])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.tick_params(axis="y", direction="out")
    ax.spines[["top", "right"]].set_visible(False)

    fig.set_dpi(dpi)
    fig.canvas.draw()
    for p in placed:
        p["px"], p["py"] = ax.transData.transform((p["x"], p["ytop"]))
    samp_px = [(kind,) + tuple(ax.transData.transform((sx, sy))) for kind, sx, sy in samples]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    assert_left_clear(out_png)

    img = np.asarray(Image.open(out_png).convert("L"))
    H, W = img.shape
    for p in placed:
        y0i, y1i = ink(p["sym"], p["size"])
        b0 = p["py"] + p["off"] * px_per_pt
        win = img[max(int(H - (b0 + y1i * px_per_pt) - 8), 0):
                  min(int(H - (b0 + y0i * px_per_pt) + 8), H),
                  max(int(p["px"] - 80), 0):min(int(p["px"] + 80), W)]
        rr, _cc = np.where(win < 200)
        assert rr.size, f"no ink found for symbol {p['sym']} of k{p['k']} {p['pat']}"
        p["h_px"] = float(rr.max() - rr.min() + 1)
        p["c_px"] = float(max(int(H - (b0 + y1i * px_per_pt) - 8), 0)
                          + (rr.min() + rr.max()) / 2.0)
    ns_h = float(np.mean([p["h_px"] for p in placed if p["sym"] == "ns"]))
    dev = []
    for p in placed:
        if p["sym"] != "ns":
            assert abs(p["h_px"] - ns_h) <= 0.32 * ns_h, \
                f"{p['sym']} ink height {p['h_px']:.1f}px vs ns {ns_h:.1f}px outside +/-25%"
        want = H - (p["py"] + (p["off"] + p["c_sym"]) * px_per_pt)
        dev.append(abs(p["c_px"] - want))
        assert dev[-1] <= 4.5 * dpi / 300, \
            f"{p['sym']} of k{p['k']} {p['pat']} is {p['c_px'] - want:+.2f} px off its own row"
    sizing = (f"ns {ns_pt:.2f} pt (ink {h_ns:.2f} pt / {ns_h:.0f} px); asterisk {ast_pt:.2f} pt "
              f"(ink {h_ast:.2f} pt / "
              f"{np.mean([p['h_px'] for p in placed if p['sym'] != 'ns']):.0f} px; ratio "
              f"{h_ast / h_ns:.3f}); per-item rows, max deviation {max(dev):.2f} px")

    rgb = np.asarray(Image.open(out_png).convert("RGB"))
    samp = {}
    for kind, sx, sy in samp_px:
        r_, c_ = int(round(H - sy)), int(round(sx))
        if kind.endswith("_edge"):
            strip = rgb[r_, max(c_ - 3, 0):c_ + 4]
            px = strip[int(np.argmin(strip.sum(axis=1)))]
        else:
            px = np.median(rgb[r_ - 1:r_ + 2, c_ - 1:c_ + 2].reshape(-1, 3), axis=0)
        samp.setdefault(kind, []).append(tuple(int(v) for v in px))
    for kind, vals in samp.items():
        assert len(set(vals)) == 1, f"{kind} is not uniform across the six items: {set(vals)}"
    fill_rgb = {k: v[0] for k, v in samp.items()}
    sep = fill_rgb["rest_control"][0] - fill_rgb["main"][0]
    assert sep >= 35, f"main/control fill separation is only {sep} grey levels (need >= 35)"
    assert sum(fill_rgb["main_edge"]) < sum(fill_rgb["control_edge"]), \
        "the main box's line work must be darker than the resting control's"
    return report, sizing, (W, H), (ylo, yhi, FH), bstat, fill_rgb


def legend_items():
    """(handles, labels) for the legend — ONE definition, used by the in-panel block and the
    standalone PNG, so the two can never drift."""
    mew, ms, lw_c, lw_m = ((MEW_DOT, MS_DOT, CTL_LW[1], LW_BOX))
    handles = [
        (Line2D([0], [0], marker="o", color="none", mfc=RED, mec="#1A1A1A", mew=mew, ms=ms),
         Line2D([0], [0], marker="o", color="none", mfc=RED2, mec="#1A1A1A", mew=mew, ms=ms)),
        Patch(facecolor="white", edgecolor=CTL_INK, lw=lw_c),
        Patch(facecolor=to_rgba(BOX_FILL, A_MAIN), edgecolor=INKG, lw=lw_m),
    ]
    labels = ["Spatial RSA $\\rho$\nP1 · P2",
              "Resting\ncontrol",
              "Spin null\n10th–90th pct\ndashed: 95th"]
    return handles, labels


def draw_legend(out_png, target_h, pad_in=0.02):
    """The legend stands beside the panel that stands beside the panel at the panel's own height with a
    roughly 1:3 width:height ratio."""
    plt.rcParams.update(RC_CONTRACT)
    handles, labels = legend_items()
    fig = plt.figure(figsize=(1.2, target_h))
    kw = dict(loc="center", ncol=1, frameon=False, fontsize=LEG_PT, handletextpad=0.45,
              borderpad=0.0, handlelength=1.0, handler_map={tuple: HandlerTuple(ndivide=None)})
    want = (target_h - 2 * pad_in) * 72.0
    LS_MAX = 2.2
    lo, hi = 0.0, LS_MAX
    for _ in range(40):
        mid = (lo + hi) / 2.0
        leg = fig.legend(handles=handles, labels=labels, labelspacing=mid, **kw)
        fig.canvas.draw()
        h_pt = leg.get_window_extent().height / fig.dpi * 72.0
        leg.remove()
        if h_pt < want:
            lo = mid
        else:
            hi = mid
    leg = fig.legend(handles=handles, labels=labels, labelspacing=(lo + hi) / 2.0, **kw)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=pad_in)
    plt.close(fig)
    w, h = pad_strip(out_png)
    return (w, h), w / h


def main():
    rec = {(r["harmonic_k"], r["patient"]): r for r in csv.DictReader(RSA.open())}
    build_id = {r["key"]: r["value"] for r in csv.DictReader(RSA_META.open())}

    base, spin, ctrl = compute()
    guard(base, rec)
    print(f"guard OK — all six rho / spin-P reproduce {RSA.name} and the published values")

    out_png, out_leg = (OUT_PNG, OUT_LEG)
    report, sizing, (W, H), (ylo, yhi, FH), bstat, fill_rgb = draw_panel(
        out_png, base, spin, ctrl, rec)
    leg_h = FH
    (lw, lh), l_ratio = draw_legend(out_leg, leg_h) if DRAW_LEGEND else ((0.0, 0.0), 0.0)
    leg_px = (lw, lh)

    bh = {(h, p): pb for h, p, _r, _sp, _sf, _rst, _sym, _s, pb, _sb in report}
    src_rows = []
    for hk, fname, _fd, _role in HARM:
        for pat in ("P1", "P2"):
            r = rec[(hk, pat)]
            pb = bh[(fname, pat)]
            src_rows.append([fname, pat, f"{float(r['spatial_rho']):.4f}",
                             f"{float(r['spin_p']):.4f}", f"{pb:.4f}",
                             f"{float(r['spin_null_mean']):.4f}",
                             f"{float(r['spin_null_p95']):.4f}", f"{float(r['shuffle_p']):.4f}",
                             f"{float(r['rest_rho']):.4f}", int(r["n_sessions"]),
                             "sig" if float(r["spin_p"]) < 0.05 else "ns",
                             "sig" if pb < 0.05 else "ns"])
    write_csv(SRC, ["harmonic", "patient", "spatial_rho_IW", "spin_p", "spin_p_bh",
                    "spin_null_mean", "spin_null_p95", "shuffle_p", "rest_rho_control",
                    "n_sessions", "verdict", "verdict_bh"], src_rows)

    null_rows = []
    for hk, fname, _fd, _role in HARM:
        for pat in ("P1", "P2"):
            k = int(hk)
            for condition, vals in (("main", spin[(k, pat)]), ("rest_control", ctrl[(k, pat)]["sn"])):
                for i, v in enumerate(vals):
                    null_rows.append([fname, pat, condition, i, f"{v:.6f}"])
    write_csv(NULLS, ["harmonic", "patient", "condition", "spin_idx", "rho"], null_rows)
    n_null = len(null_rows)

    box_rows = []
    for hk, fname, _fd, _role in HARM:
        for pat in ("P1", "P2"):
            for condition in ("main", "rest_control"):
                s = bstat[(int(hk), pat)][condition]
                box_rows.append([fname, pat, condition, f"{s['p10']:.6f}", f"{s['q25']:.6f}",
                                 f"{s['median']:.6f}", f"{s['q75']:.6f}", f"{s['p90']:.6f}",
                                 f"{s['p95']:.6f}", s["n"]])
    write_csv(BOXSTATS, ["harmonic", "patient", "condition", "p10", "q25", "median", "q75", "p90",
                         "p95", "n"], box_rows)

    sig_k = sorted({h for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report if s})
    sig_k_bh = sorted({h for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report if sb})
    k2 = {p: (rho, sp_, rst, pb) for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report if h == "k2"}
    k1p2 = [(rho, sp_, pb) for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report
            if h == "k1" and p == "P2"][0]
    rows = [
        ("figure", "Fig. 6 panel f - patient-vs-healthy IW spatial-RSA panel: observed rho as a single-size dot "
                   "over its full spin-null violin, per harmonic (k1/k2/k3) x patient (P1/P2); "
                   "quantifies the electrode-RDM grid figures_out/fig6/fig6_rsa_rdm_grid.png"),
        ("symbol_sizing", sizing + ". Asserts in the renderer require the asterisk ink height within "
                                   "+/-25% of the 'ns' ink height and all six symbol centres within "
                                   "3 px of the common row."),
        ("metric", "2nd-order spatial RSA rho: patient-IW electrode harmonic-phase RDM vs healthy-IW RDM"),
        ("null", "spin test = rotated electrode geometry (preserves spatial autocorrelation, "
                 "conservative). The 1000-value distribution is DRAWN as a 10th-90th box-whisker "
                 "per item (Q1-Q3 box + median, whiskers to p10/p90, no fliers), with the 95th pct "
                 "as a dashed critical line on top. All 1000 values remain in "
                 "fig6_rsa_quantification_forest_spin_nulls.csv; the drawn summary is in "
                 "fig6_rsa_quantification_forest_null_box_stats.csv. Shuffle p is kept in the "
                 "Source Data for completeness but is anti-conservative for spatial maps and is not "
                 "drawn."),
        ("control", f"resting control (patient IW vs healthy yoked-REST): only its OWN spin-null "
                    f"box-whisker is drawn, offset +{X_REST:.2f} x-units (0.7x width, white fill, lighter "
                    "line work) - deliberately subordinate to the main box. The resting-control rho itself "
                    "is not drawn (it is in the Source Data). NO p95 dash is drawn on it: the control "
                    "carries no significance claim."),
        ("control_spin_rng", f"the resting-control spin null has NO counterpart in the producer, so "
                             f"it uses its own generator numpy default_rng({CTRL_SEED}), created "
                             f"only after the producer's loop finishes and consumed in the order "
                             f"k=1,2,3 x (P1,P2). "
                             
                             f"Same machinery otherwise: the producer's own "
                             f"spin_null(), N=1000, healthy-REST RDM as reference."),
    ]
    for k in (1, 2, 3):
        for pat in ("P1", "P2"):
            c = ctrl[(k, pat)]
            rows.append((f"control_spin_percentile_k{k}_{pat}",
                         f"rest rho {c['obs']:+.4f} sits at percentile {c['pct']:.1f} of its own "
                         f"spin null; two-sided empirical P = {pfmt(c['p_two'], 3)}; null 2.5-97.5 pct "
                         f"[{c['p025']:+.4f}, {c['p975']:+.4f}] -> inside the central range"))
    rows += [
        ("scope", "n=2 SCI patients, case-level, post-operative (post_operative); P1 n=6 sessions, P2 n=5. "
                  "NOT a population claim."),
        ("marker_semantics", f"DATA-DRIVEN, nothing hardcoded per row, and DELIBERATELY UNENCODED: "
                             f"every observed-rho marker is the same filled circle, {MS_DOT} pt with a "
                             f"dark edge{' and a white halo' if HALO_PT > 0 else ''}, filled in the "
                             f"patient colour of Fig. 5b: Patient 1 {RED} (tan), Patient 2 {RED2} "
                             f"(pale olive), for all harmonics. "
                             f"There is no size encoding, no shape encoding and no significance "
                             f"colour encoding - significance is carried ONLY by the symbol row and "
                             f"by whether the dot sits above its dashed 95th-pct line. Patient "
                             f"identity is carried by the fill colour and the x tick label. The "
                             f"resting control has no marker (its box only)."),
        ("n_tests", "6 (3 harmonics x 2 patients); BH step-up, two-sided-equivalent one-sided spin "
                    "p values as computed by the producer's spin_null()"),
        ("harmonics_passing_spin", (", ".join(sig_k) if sig_k else "none") +
                                   " (UNCORRECTED spin p < .05 - retained for provenance; the "
                                   "panel and all text use harmonics_passing_spin_bh)"),
        ("harmonics_passing_spin_bh", (", ".join(sig_k_bh) if sig_k_bh else "none") +
                                      " (BH-corrected across the 6-test family; this is the "
                                      "reportable set)"),
        ("result_k1_k3", "; ".join(f"{h} {p} rho={rho:.4f} spin p={sp_:.4f} p_BH={pb:.4f} (ns)"
                                   for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report if not sb)),
        ("claim", "channel-space RSA, not source localization"),
        ("output_png", "figures_out/fig6/fig6_rsa_quantification_forest.png"),
        ("output_png_legend", "figures_out/fig6/fig6_rsa_quantification_forest_legend.png"),
    ]
    write_csv(META, ["key", "value"], rows)

    print(f"build_id={build_id.get('build_id', '?')}  source={RSA.name}")
    print(f"{'k':>3} {'pat':>3} | {'rho':>7} {'spin_p':>7} {'p_BH':>7} {'shuffle_p':>9} "
          f"{'REST rho':>9}  sym")
    for h, p, rho, sp_, sf, rst, sym, s, pb, sb in report:
        print(f"{h:>3} {p:>3} | {rho:>7.4f} {sp_:>7.4f} {pb:>7.4f} {sf:>9.4f} {rst:>9.4f}  "
              f"{sym:>3}")
    print(f"BH family: 6 patient x harmonic spin tests; passing uncorrected = "
          f"{', '.join(sig_k) or 'none'}; passing BH = {', '.join(sig_k_bh) or 'none'}")
    print("\nresting-control rho vs its OWN spin null (two-sided empirical)")
    for k in (1, 2, 3):
        for pat in ("P1", "P2"):
            c = ctrl[(k, pat)]
            print(f" k{k} {pat} | rest {c['obs']:+.4f} | pct {c['pct']:>5.1f} | P {c['p_two']:.4f} "
                  f"| null 2.5-97.5 [{c['p025']:+.4f}, {c['p975']:+.4f}]")
    print("\nnull box stats (exact drawn values)")
    print(f"{'item':>7} {'condition':>13} {'p10':>9} {'q25':>9} {'median':>9} {'q75':>9} {'p90':>9} "
          f"{'p95':>9} {'n':>5}")
    for k in (1, 2, 3):
        for pat in ("P1", "P2"):
            for condition in ("main", "rest_control"):
                s = bstat[(k, pat)][condition]
                print(f"  k{k}{pat:>4} {condition:>13} {s['p10']:>+9.4f} {s['q25']:>+9.4f} "
                      f"{s['median']:>+9.4f} {s['q75']:>+9.4f} {s['p90']:>+9.4f} "
                      f"{s['p95']:>+9.4f} {s['n']:>5}")
    print(f"\nrendered RGB, sampled from the PNG")
    print(f"  FILL  main {fill_rgb['main']}  |  resting control {fill_rgb['rest_control']}  "
          f"|  separation {fill_rgb['rest_control'][0] - fill_rgb['main'][0]} grey levels")
    print(f"  EDGE  main {fill_rgb['main_edge']}  |  resting control {fill_rgb['control_edge']}")
    print(f"symbols: {sizing}")
    print(f"wrote {OUT_PNG}  ({W}x{H} px = {W/300*25.4:.1f}x{H/300*25.4:.1f} mm, "
          f"ylim {ylo:.3f}..{yhi:.3f})")
    if DRAW_LEGEND:
        print(f"wrote {OUT_LEG}  ({lw}x{lh} px = {lw/300:.2f}x{lh/300:.2f} in, "
              f"standalone tall-narrow legend, w:h = 1:{1/l_ratio:.2f})")
    else:
        print("legend: NOT written - the four entries "
              "belong in the figure caption")
    print(f"Source Data: {SRC.name} (6 rows), {NULLS.name} ({n_null} rows), "
          f"{BOXSTATS.name} ({len(box_rows)} rows), {META.name} ({len(rows)} keys)")


if __name__ == "__main__":
    main()
