"""Figure 3 — Gait-phase locking of the imagined-walking representation

harmonic phase maps split by decoding responder status

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Writes:
  figures_out/fig3/fig3_harmonic_phase_cohort_sharedscale.png              (panel c, whole cohort)
  figures_out/fig3/fig3_harmonic_phase_responder_sharedscale.png           (panel h, top block)
  figures_out/fig3/fig3_harmonic_phase_nonresponder_sharedscale.png        (panel h, bottom block)
  figures_out/fig3/fig3_harmonic_phase_responder_split_combined.png        (panel h, assembled)
  fig3_harmonic_phase_responder_split_metrics.csv (not distributed)   (panels c+h source data)
  04_source_data/fig3/fig3_harmonic_phase_responder_split_metadata.csv  (cohort/version provenance)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import numpy as np
from scipy.stats import t as tdist
import analysis_config as ec

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / ("data/decoder_output/"
               "diag_harmonic_phase_responder_split/"
               "persession_harmonic_phase_cycle9_complex.csv")
DEF = ec.sd("fig3_def_resnonres_source_data.csv")
EEG_SET = ROOT / "data/prep_data/healthy/eeg/SUB_37_sess02_step2.set"
OUT_DIR = ROOT / "figures_out" / "fig3"


MASK_PARAMS = dict(marker="o", markerfacecolor="k", markeredgecolor="white",
                   markersize=4.2, markeredgewidth=0.6)


def _bh(p, alpha=0.05):
    p = np.asarray(p, float); out = np.zeros(p.shape, bool); ok = np.isfinite(p)
    pv = p[ok]; m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv); passed = pv[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any():
        keep = np.zeros(m, bool); keep[order[:np.where(passed)[0].max() + 1]] = True
        o = out[ok]; o[:] = keep; out[ok] = o
    return out


def sig_mask(signed_sess, mask):
    """[ch, point, h] bool: BH-FDR (per harmonic) channels where signed locking != 0 across masked sessions."""
    s = signed_sess[mask]
    mu = np.nanmean(s, 0); sd = np.nanstd(s, 0, ddof=1)
    n = np.sum(np.isfinite(s), 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tval = mu / (sd / np.sqrt(n))
    p = 2 * tdist.sf(np.abs(tval), np.clip(n - 1, 1, None))
    sig = np.zeros(p.shape, bool)
    for ri in range(p.shape[-1]):
        sig[..., ri] = _bh(p[..., ri].ravel()).reshape(p[..., ri].shape)
    return sig


def eeglab_classic_redblue_cmap(n: int = 256):
    from matplotlib.colors import ListedColormap
    x = np.linspace(-1.0, 1.0, n)
    cmap = np.ones((n, 3))
    neg, pos = x < 0, x > 0
    cmap[neg, 0] = 1 + x[neg] * 0.95
    cmap[neg, 1] = 1 + x[neg] * 0.80
    cmap[neg, 2] = 1 + x[neg] * 0.45
    cmap[pos, 0] = 1 - x[pos] * 0.30
    cmap[pos, 1] = 1 - x[pos] * 0.95
    cmap[pos, 2] = 1 - x[pos] * 0.95
    return ListedColormap(np.clip(cmap, 0, 1), name="eeglab_classic_redblue")


def _b(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def load_groups() -> dict:
    """session -> (responder bool, ironclad bool, session_r) from the per-session-null definition."""
    out = {}
    with DEF.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rv = float(row["r_reference"])
            except (KeyError, ValueError):
                rv = float("nan")
            out[row["session"]] = (_b(row["resp"]), _b(row.get("ironclad", "")), rv)
    return out


def load_dump():
    """Return sessions, channels, cycle_fractions, harmonics, A[sess,ch,point,h] complex."""
    sessions, channels, fracs, harms = [], [], [], []
    recs = []
    with DUMP.open(newline="") as f:
        for row in csv.DictReader(f):
            s, h, cf, ch = row["session"], int(row["harmonic_k"]), float(row["cycle_fraction"]), row["channel"]
            recs.append((s, ch, cf, h, float(row["real"]), float(row["imag"])))
            if s not in sessions: sessions.append(s)
            if ch not in channels: channels.append(ch)
            if cf not in fracs: fracs.append(cf)
            if h not in harms: harms.append(h)
    fracs = sorted(fracs)
    harms = sorted(harms)
    si = {s: i for i, s in enumerate(sessions)}
    ci = {c: i for i, c in enumerate(channels)}
    fi = {round(v, 6): i for i, v in enumerate(fracs)}
    hi = {h: i for i, h in enumerate(harms)}
    A = np.full((len(sessions), len(channels), len(fracs), len(harms)), np.nan, complex)
    for s, ch, cf, h, re_, im_ in recs:
        A[si[s], ci[ch], fi[round(cf, 6)], hi[h]] = complex(re_, im_)
    return sessions, channels, fracs, harms, A


def group_mean(A, mask):
    sub = A[mask]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def draw_cycle_curve(fig, top_axes, fracs, label="Hip cycle", amp=1.0, max_height=0.12,
                     gap=0.045, ylim=(-1.6, 1.2), show_label=False, lw=2.6, vline_mult=1.2):
    """Overlay a one-cycle phase-reference curve above the topomap strip, with one
    marker per gait-phase column aligned to that column's actual center (via each
    top-row axis bbox), so alignment holds for any column spacing."""
    pos = [a.get_position() for a in top_axes]
    xs = [(p.x0 + p.x1) / 2 for p in pos]
    hw = (pos[0].x1 - pos[0].x0) / 2.0
    left_edge, right_edge = pos[0].x0, pos[-1].x1
    span = xs[-1] - xs[0]
    plo, phi = -hw / span, 1 + hw / span
    top_y = max(p.y1 for p in pos)
    bottom = top_y + gap
    height = min(max_height, 0.985 - bottom)
    ax = fig.add_axes([left_edge, bottom, right_edge - left_edge, height])
    ph = np.linspace(plo, phi, 500)
    ax.plot(ph, amp * np.cos(2 * np.pi * ph), color="#333333", lw=lw, solid_capstyle="round")
    fy = amp * np.cos(2 * np.pi * np.array(fracs))
    ax.vlines(fracs, -amp * vline_mult, fy, color="0.72", lw=lw * 0.45, zorder=1)
    ax.plot(fracs, fy, "o", color="#c0392b", ms=5.5, mec="white", mew=0.7, zorder=3)
    ax.set_xlim(plo, phi)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if show_label:
        ax.text(-0.015, 0.5, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=8, fontweight="bold")


HEAD_RING_PLOTRAD = 1.02
HEAD_RING_EAR_EXT = 1.094
HEAD_RING_MAIN = dict(head_lw=0.3, contour_lw=0.3, dot_pt=1.25, rim_pt=0.2, sensor_f=1.0)


def eeglab_rim_layout(channels, plotrad=HEAD_RING_PLOTRAD):
    """(pos, R) for `channels` (labels of standard_1005, in the order the values are passed): pos = EEGLAB 2-D layout
    (n x 2, nose up, left on the left), R = plotrad x max(Rd), the head radius = clip radius = sphere radius to pass."""
    import mne
    elc = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([elc[c] for c in channels], float)
    Rd = 0.5 - np.degrees(np.arcsin(P[:, 2] / np.linalg.norm(P, axis=1))) / 180
    az = np.arctan2(P[:, 1], P[:, 0])
    return np.column_stack([Rd * np.cos(az), Rd * np.sin(az)]), plotrad * float(Rd.max())


def head_ring_fit_plotrad(channels, head_r_mm, head_lw=0.3, dot_pt=1.25, sensor_f=1.0, plotrad=HEAD_RING_PLOTRAD):
    """Smallest plotrad (>= `plotrad`) at which the LARGEST mark - the significance dot incl. its rim (dot_pt) or the
    hollow sensor dot (0.75 pt radius x sensor_f) - placed on the OUTERMOST electrode ends inside the outer edge of the
    head line (R + head_lw / 2) for a head of radius head_r_mm on the page."""
    pos, _ = eeglab_rim_layout(channels, 1.0)
    rd_max = float(np.linalg.norm(pos, axis=1).max())
    mm = 25.4 / 72
    hw = head_lw / 2 * mm - 0.01
    mark = max(dot_pt / 2, 0.75 * sensor_f) * mm
    if mark - hw >= head_r_mm:
        raise SystemExit(f"head ring: a {2 * mark / mm:.2f} pt mark cannot fit a {head_r_mm:.2f} mm head")
    p = max(plotrad, head_r_mm / (head_r_mm + hw - mark))
    return p, p * rd_max


def head_ring_topomap(ax, vals, pos, R, cmap, vlim, mask=None, head_lw=0.3, contour_lw=0.3, dot_pt=1.25, rim_pt=0.2,
                      sensor_f=1.0):
    """One map on the EEGLAB layout: the engine's mne call (contours 6, sensors, res 256, cubic, extrapolate 'head') with
    sphere = (0, 0, 0, R); significance dots = MASK_PARAMS at dot_pt (incl."""
    import mne
    from matplotlib.collections import PathCollection
    mkw = {}
    if mask is not None:
        mkw = dict(mask=np.asarray(mask, bool),
                   mask_params=dict(MASK_PARAMS, markersize=dot_pt - rim_pt, markeredgewidth=rim_pt))
    im, cn = mne.viz.plot_topomap(vals, pos, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=6, sensors=True,
                                  sphere=(0.0, 0.0, 0.0, R), res=256, image_interp="cubic", extrapolate="head", **mkw)
    outl = [ln for ln in ax.lines if ln.get_marker() in ("None", "", None) and ln.get_linestyle() not in ("None", "")]
    for ln in outl:
        ln.set_linewidth(head_lw)
    if cn is not None:
        cn.set_linewidth(contour_lw)
    for col in ax.collections:
        if isinstance(col, PathCollection):
            col.set_sizes([0.25 * sensor_f ** 2])
            col.set_linewidths([1.0 * sensor_f])
    ear = max(float(np.max(np.abs(ln.get_xdata()))) for ln in outl) / R
    if abs(ear - HEAD_RING_EAR_EXT) > 1e-3:
        raise SystemExit(f"head ring: ear tips at {ear:.4f} R, expected {HEAD_RING_EAR_EXT} (mne outline changed?)")
    return im, cn


def head_ring_place(fig, ax, im, cx_mm, cy_mm, ear_r_mm):
    """Move and scale `ax` (about the map) so the head centre lands at (cx_mm, cy_mm) from the figure's TOP-LEFT and
    the ear tips at ear_r_mm from it."""
    W, Hh = fig.bbox.width, fig.bbox.height
    ax.apply_aspect()
    e = im.get_clip_path().get_fully_transformed_path().get_extents()
    px, py, pr = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2, (e.x1 - e.x0) / 2
    s = ear_r_mm / HEAD_RING_EAR_EXT / 25.4 * fig.dpi / pr
    tx, ty = cx_mm / 25.4 * fig.dpi, Hh - cy_mm / 25.4 * fig.dpi
    b = ax.get_position(original=True)
    ax.set_position([(tx + (b.x0 * W - px) * s) / W, (ty + (b.y0 * Hh - py) * s) / Hh, b.width * s, b.height * s])


def head_ring_gate(fig, ax, im, R, head_lw, label=""):
    """Gate, call once the axes position is final: the map's clip circle must be the head circle (radius R) and no
    sensor dot (PathCollection) or significance dot (marker Line2D) may reach past the OUTER edge of the head line
    (centre radius R + head_lw / 2)."""
    from matplotlib.collections import PathCollection
    ax.apply_aspect()
    c0 = ax.transData.transform([[0.0, 0.0]])[0]
    r_px = ax.transData.transform([[R, 0.0]])[0, 0] - c0[0]
    e = im.get_clip_path().get_fully_transformed_path().get_extents()
    if abs((e.x1 - e.x0) / 2 / r_px - 1) > 2e-3:
        raise SystemExit(f"head ring {label}: clip radius {(e.x1 - e.x0) / 2 / r_px:.4f} R, expected 1")
    pt = fig.dpi / 72
    worst = -np.inf
    for col in ax.collections:
        if isinstance(col, PathCollection) and len(col.get_offsets()):
            rd = (np.sqrt(col.get_sizes()[0]) / 2 + col.get_linewidths()[0] / 2) * pt
            worst = max(worst, np.linalg.norm(ax.transData.transform(col.get_offsets()) - c0, axis=1).max() + rd)
    for ln in ax.lines:
        if ln.get_marker() not in ("None", "", None) and len(ln.get_xdata()):
            rd = (ln.get_markersize() / 2 + ln.get_markeredgewidth() / 2) * pt
            xy = np.column_stack([ln.get_xdata(), ln.get_ydata()])
            worst = max(worst, np.linalg.norm(ax.transData.transform(xy) - c0, axis=1).max() + rd)
    margin = (worst - r_px) / fig.dpi * 25.4
    if worst > r_px + head_lw / 2 * pt:
        raise SystemExit(f"head ring {label}: a dot crosses the head line (outermost dot edge {margin:+.3f} mm from "
                         f"the line centre, line half-width {head_lw / 2 * 25.4 / 72:.3f} mm)")
    return margin


def _head_ring_maps(head_ring, channels, ch_index):
    """render_group / render_blocks head_ring= keyword -> (pos, R, weights); None when the keyword is not used."""
    if head_ring is None:
        return None
    w = dict(HEAD_RING_MAIN, **head_ring)
    plotrad = w.pop("plotrad", HEAD_RING_PLOTRAD)
    pos, R = eeglab_rim_layout([channels[i] for i in ch_index], plotrad)
    return pos, R, w


def _head_ring_gate_all(fig, drawn, R, w, label):
    margins = [head_ring_gate(fig, ax, im, R, w["head_lw"], label) for ax, im in drawn]
    print(f"  [head ring] {label}: {len(margins)} maps, outermost dot edge {max(margins):+.3f} mm from the head-line "
          f"centre (min {min(margins):+.3f}; line half-width {w['head_lw'] / 2 * 25.4 / 72:.3f} mm)")
    return margins


ENG_B_PX = (3210, 1174)
ENG_B_COL_PX = [208.05 + 325.8588 * i for i in range(9)]
ENG_B_ROW_PX = [364.12, 693.01, 1021.90]
ENG_B_DISC_R_PX = 127.962
ENG_B_CURVE_PX = (80.089, 14.400, 2943.176, 175.680)
ENG_B_CURVE_XLIM = (-0.049143, 1.049143)


CB_H_FRAC = 0.806
CB_DX     = 0.014
CB_DY     = 0.007


def render_group(signed, channels, fracs, harms, clim, gname, cmp_label, info, ch_index,
                 nrow_meta, out_png, with_cycle=False, cycle_label="Hip cycle", sig=None,
                 row_labels=None, head_ring=None):
    """drawn to the left of the first column (default None: no labels)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mne

    n_h, n_p = len(harms), len(fracs)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig = plt.figure(figsize=(1.55 * n_p + 1.9, 1.5 * n_h + (1.1 if with_cycle else 0.5)), dpi=240)
    gs = fig.add_gridspec(n_h, n_p + 1, width_ratios=[1] * n_p + [0.06], wspace=0.30, hspace=0.22,
                          top=(0.80 if with_cycle else 0.95))
    cmap = eeglab_classic_redblue_cmap()
    image = None
    top_axes = []
    _hr, _hr_drawn = _head_ring_maps(head_ring, channels, ch_index), []
    for ri, h in enumerate(harms):
        for ci, cf in enumerate(fracs):
            ax = fig.add_subplot(gs[ri, ci])
            if ri == 0:
                top_axes.append(ax)
            vals = signed[ch_index, ci, ri]
            if _hr is not None:
                image, _ = head_ring_topomap(ax, vals, _hr[0], _hr[1], cmap, clim,
                                             mask=sig[ch_index, ci, ri] if sig is not None else None, **_hr[2])
                _hr_drawn.append((ax, image))
            else:
                mkw = dict(mask=sig[ch_index, ci, ri].astype(bool), mask_params=MASK_PARAMS) if sig is not None else {}
                image, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap,
                                                vlim=clim, contours=6, sensors=True, sphere="eeglab",
                                                res=256, image_interp="cubic", extrapolate="head", **mkw)
            if ri == 0:
                ax.set_title(f"{100*cf:.0f}%", fontsize=13, pad=5)
            if ci == 0 and row_labels is not None:
                ax.text(-0.10, 0.5, row_labels[ri], transform=ax.transAxes, fontsize=13,
                        ha="right", va="center")
    cax = fig.add_subplot(gs[:, -1])
    _p = cax.get_position()
    _h = _p.height * CB_H_FRAC
    cax.set_position([_p.x0 + CB_DX, _p.y0 + (_p.height - _h) / 2 + CB_DY, _p.width, _h])
    if image is None:
        image = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(image, cax=cax)
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("Phase modulation", fontsize=13, labelpad=10)
    cb.ax.tick_params(labelsize=11, length=3)
    if with_cycle:
        draw_cycle_curve(fig, top_axes, fracs, cycle_label)
    if _hr is not None:
        _head_ring_gate_all(fig, _hr_drawn, _hr[1], _hr[2], Path(out_png).name)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def render_blocks(blocks, channels, fracs, harms, clim, info, ch_index, out_png,
                  block_labels=None, row_labels=None, block_gap=0.28,
                  with_cycle=False, cycle_label="Gait cycle reference", sigs=None,
                  font_scale=1.0, head_ring=None):
    """Multi-BLOCK variant of `render_group`: several groups stacked in ONE image under ONE
    shared colorbar."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mne

    n_b, n_h, n_p = len(blocks), len(harms), len(fracs)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    head_in = 1.12 if with_cycle else 0.25
    body_in = 1.5 * n_h * n_b + block_gap * (n_b - 1) * 1.5
    fig_h = body_in + head_in + 0.25
    fig = plt.figure(figsize=(1.55 * n_p + 1.9, fig_h), dpi=240)
    hr, rows_of = [], []
    for bi in range(n_b):
        if bi:
            hr.append(block_gap)
            rows_of.append(None)
        for ri in range(n_h):
            hr.append(1.0)
            rows_of.append((bi, ri))
    spacer = 0.28 * (font_scale - 1.0)
    gs = fig.add_gridspec(len(hr), n_p + 2, width_ratios=[1] * n_p + [spacer, 0.06],
                          height_ratios=hr, wspace=0.30, hspace=0.22,
                          top=1.0 - head_in / fig_h)
    cmap = eeglab_classic_redblue_cmap()
    image = None
    first_ax_of_block = {}
    top_axes = []
    _hr, _hr_drawn = _head_ring_maps(head_ring, channels, ch_index), []
    for gi, slot in enumerate(rows_of):
        if slot is None:
            continue
        bi, ri = slot
        for ci in range(n_p):
            ax = fig.add_subplot(gs[gi, ci])
            if gi == 0:
                top_axes.append(ax)
            vals = blocks[bi][ch_index, ci, ri]
            if _hr is not None:
                image, _ = head_ring_topomap(ax, vals, _hr[0], _hr[1], cmap, clim,
                                             mask=sigs[bi][ch_index, ci, ri] if sigs is not None else None, **_hr[2])
                _hr_drawn.append((ax, image))
            else:
                mkw = (dict(mask=sigs[bi][ch_index, ci, ri].astype(bool), mask_params=MASK_PARAMS)
                       if sigs is not None else {})
                image, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap,
                                                vlim=clim, contours=6, sensors=True, sphere="eeglab",
                                                res=256, image_interp="cubic", extrapolate="head", **mkw)
            if gi == 0:
                ax.set_title(f"{100*fracs[ci]:.0f}%", fontsize=13 * font_scale, pad=5)
            if ci == 0:
                if row_labels is not None:
                    ax.text(-0.10, 0.5, row_labels[ri], transform=ax.transAxes,
                            fontsize=13 * font_scale, ha="right", va="center")
                first_ax_of_block.setdefault(bi, []).append(ax)
    if block_labels is not None:
        for bi, axes_ in first_ax_of_block.items():
            top = axes_[0].get_position().y1
            bot = axes_[-1].get_position().y0
            x = min(a.get_position().x0 for a in axes_) - 0.022
            fig.text(x, (top + bot) / 2, block_labels[bi], fontsize=15 * font_scale, rotation=90,
                     ha="right", va="center")
    cax = fig.add_subplot(gs[:, -1])
    if image is None:
        image = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(image, cax=cax)
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("Phase modulation", fontsize=13 * font_scale, labelpad=10 * font_scale)
    cb.ax.tick_params(labelsize=11 * font_scale, length=3)
    if with_cycle:
        draw_cycle_curve(fig, top_axes, fracs, cycle_label)
    if _hr is not None:
        _head_ring_gate_all(fig, _hr_drawn, _hr[1], _hr[2], Path(out_png).name)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def stack_preview(top_png, bot_png, out_png, top_label, bot_label, width=2100):
    from PIL import Image, ImageDraw, ImageFont
    def font(sz):
        for p in ("C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if Path(p).exists():
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()
    def rs(im):
        return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
    top = rs(Image.open(top_png).convert("RGB"))
    bot = rs(Image.open(bot_png).convert("RGB"))
    band = 64
    out = Image.new("RGB", (width, band + top.height + 12 + band + bot.height + 12), "white")
    dr = ImageDraw.Draw(out)
    dr.text((20, 14), top_label, fill=(150, 20, 20), font=font(38))
    out.paste(top, (0, band))
    y = band + top.height + 12
    dr.text((20, y + 12), bot_label, fill=(20, 40, 150), font=font(38))
    out.paste(bot, (0, y + band))
    out.save(out_png, dpi=(150, 150))


def render_grid(rows, fracs, clim, info, out_png, cluster_size=4, cycle_label="Hip cycle"):
    """Multi-row scalp grid: one row per (harmonic, group) strip, columns = gait-cycle
    points."""
    import math
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mne

    n_data, n_p = len(rows), len(fracs)
    height_ratios, data_to_gs, gs_idx, count = [], [], 0, 0
    for i in range(n_data):
        data_to_gs.append(gs_idx)
        height_ratios.append(1.0)
        gs_idx += 1
        count += 1
        if count == cluster_size and i != n_data - 1:
            height_ratios.append(0.34)
            gs_idx += 1
            count = 0
    n_clusters = math.ceil(n_data / cluster_size)

    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig = plt.figure(figsize=(1.5 * n_p + 2.4, 1.12 * n_data + 0.34 * (n_clusters - 1) + 1.5), dpi=200)
    gs = fig.add_gridspec(gs_idx, n_p + 1, width_ratios=[1] * n_p + [0.05], height_ratios=height_ratios,
                          wspace=0.18, hspace=0.12, top=0.85, bottom=0.02, left=0.02, right=0.965)
    cmap = eeglab_classic_redblue_cmap()
    image, top_axes = None, []
    for ri, row in enumerate(rows):
        for ci, cf in enumerate(fracs):
            ax = fig.add_subplot(gs[data_to_gs[ri], ci])
            if ri == 0:
                top_axes.append(ax)
            image, _ = mne.viz.plot_topomap(row["signed"][:, ci], info, axes=ax, show=False, cmap=cmap,
                                            vlim=clim, contours=6, sensors=True, sphere="eeglab",
                                            res=256, image_interp="cubic", extrapolate="head")
    cax = fig.add_subplot(gs[:, -1])
    if image is None:
        image = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*clim), cmap=cmap)
    cb = fig.colorbar(image, cax=cax)
    cb.set_ticks([])
    draw_cycle_curve(fig, top_axes, fracs, amp=0.5, max_height=0.11, gap=0.02,
                     ylim=(-1.5, 1.15), show_label=False, lw=2.4, vline_mult=1.7)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def main():
    import mne
    ec.require("main", "main")
    ec.check_tag(DUMP.parent.name, "neural", "harmonic-phase |c| dump")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grp = load_groups()
    sessions, channels, fracs, harms, A = load_dump()

    resp = np.array([grp.get(s, (False, False, np.nan))[0] for s in sessions])
    r = np.array([grp.get(s, (False, False, np.nan))[2] for s in sessions])
    mask_full = np.array([s in grp for s in sessions])
    mask_resp = mask_full & resp
    mask_non = mask_full & ~resp

    full_c = group_mean(A, mask_full)
    flat = full_c.ravel()
    k = np.nanargmax(np.abs(flat))
    ref_phase = np.angle(flat[k])
    full_signed = np.real(full_c * np.exp(-1j * ref_phase))
    signed_sess = np.real(A * np.exp(-1j * ref_phase))
    clim_abs = float(np.nanquantile(np.abs(full_signed), 0.98))
    clim = (-clim_abs, clim_abs)
    print(f"per-session-null split: responder n={mask_resp.sum()} non-responder n={mask_non.sum()} "
          f"(total {mask_full.sum()})  ref_phase={ref_phase:.6f}  shared_clim=+/-{clim_abs:.6f}")

    raw = mne.io.read_raw_eeglab(str(EEG_SET), preload=False, verbose="ERROR")
    keep = [c for c in raw.ch_names if c in channels]
    info = raw.copy().pick(keep).info
    ch_index = [channels.index(c) for c in keep]

    cohort_meta = {h: (h, int(mask_full.sum()), 0, float(np.nanmean(np.abs(full_c[:, :, hi_]))))
                   for hi_, h in enumerate(harms)}
    cohort_png = ec.scoped(OUT_DIR / "fig3_harmonic_phase_cohort_sharedscale.png")
    k_labels = [rf"$\mathregular{{k_{int(h)}}}$" for h in harms]
    render_group(full_signed, channels, fracs, harms, clim, "whole cohort",
                 f"n={int(mask_full.sum())}", info, ch_index, cohort_meta, cohort_png,
                 with_cycle=True, cycle_label="Hip cycle", sig=sig_mask(signed_sess, mask_full),
                 row_labels=k_labels)
    print(f"wrote {cohort_png}")

    groups = [("responders", mask_resp, f"MI |c| > own rest, n={int(mask_resp.sum())}", "fig3_harmonic_phase_responder_sharedscale.png"),
              ("non-responders", mask_non, f"not above own rest, n={int(mask_non.sum())}", "fig3_harmonic_phase_nonresponder_sharedscale.png")]

    metrics = [("cohort", h, int(mask_full.sum()), cohort_meta[h][3], float(np.nanmean(r[mask_full])))
               for h in harms]
    pngs = {}
    for gname, mask, cmp_label, fname in groups:
        gc = group_mean(A, mask)
        gsigned = np.real(gc * np.exp(-1j * ref_phase))
        nrow_meta = {}
        for hi_, h in enumerate(harms):
            mc = float(np.nanmean(np.abs(gc[:, :, hi_])))
            nrow_meta[h] = (h, int(mask.sum()), 0, mc)
            metrics.append((gname, h, int(mask.sum()), mc, float(np.nanmean(r[mask]))))
        out_png = ec.scoped(OUT_DIR / fname)
        render_group(gsigned, channels, fracs, harms, clim, gname, cmp_label, info, ch_index, nrow_meta, out_png,
                     sig=sig_mask(signed_sess, mask))
        pngs[gname] = out_png
        print(f"wrote {out_png}")

    stack_preview(pngs["responders"], pngs["non-responders"],
                  ec.scoped(OUT_DIR / "fig3_harmonic_phase_responder_split_combined.png"),
                  f"Responders  (n={int(mask_resp.sum())}, MI locking > own resting null)",
                  f"Non-responders  (n={int(mask_non.sum())}, not above own rest)")
    print(f"wrote {ec.scoped(OUT_DIR / 'fig3_harmonic_phase_responder_split_combined.png')}")

    mcsv = ec.sd("fig3_harmonic_phase_responder_split_metrics.csv")
    with mcsv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "harmonic_k", "n_sessions", "group_map_mean_abs", "mean_r"])
        w.writerows([(g, h, n, f"{mc:.10f}", f"{mr:.6f}") for g, h, n, mc, mr in metrics])
    print(f"wrote {mcsv}")

    used = [s for s in sessions if s in grp]
    meta = [
        ("panel", "Fig. 3 panel b - cohort gait-phase locking topographies, k1-k3 x 9 "
                  "gait-cycle points (the responder/non-responder split render is SI)"),
        ("prep", "prep (data/healthy_prep)"),
        ("n_sessions", int(mask_full.sum())),
        ("n_subjects", len({s.split("_sess")[0] for s in used})),
        ("n_responder", int(mask_resp.sum())),
        ("n_nonresponder", int(mask_non.sum())),
        ("split_definition", "responder = MI |c|(k1) > own resting null 95th pct "
                             "(200-resample); within-session one-sided p<0.05"),
        ("split_source", DEF.relative_to(ROOT).as_posix()),
        ("harmonic_dump", DUMP.relative_to(ROOT).as_posix()),
        ("n_channels", len(ch_index)),
        ("harmonics", " ".join(f"k{h}" for h in harms)),
        ("ref_phase_rad", f"{ref_phase:.6f}"),
        ("shared_clim_abs", f"{clim_abs:.6f}"),
        ("sig_marker_def", "one-sample t across the group's sessions that signed locking != 0, "
                           "BH-FDR alpha=0.05 per harmonic row"),
    ]
    dcsv = ec.sd("fig3_harmonic_phase_responder_split_metadata.csv")
    with dcsv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(sorted(meta, key=lambda kv: kv[0]))
    print(f"wrote {dcsv}")


if __name__ == "__main__":
    main()
