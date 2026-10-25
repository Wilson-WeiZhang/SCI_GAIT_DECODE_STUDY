"""Shared figure settings

Panel grammar for the 16 cm canvas — a panel is RENDERED AT THE SIZE IT IS PLACED AT

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations

MM = 1 / 25.4
DPI = 600

LETTER_PT = 12.0
TITLE_PT = 7.0
TICK_PT = 6.0
SECOND_PT = 5.5
FLOOR_PT = 5.25

LW_AXIS = 0.5
LW_MAIN = 1.0
LW_REF = 0.5
LW_MINOR = 0.4
LW_MIN = 0.25
LW_AXIS_R2 = 0.65

INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"

CANVAS_MM = 160.0
GUTTER_MM = 3.0
TIERS = {"full": CANVAS_MM,
         "half": (CANVAS_MM - GUTTER_MM) / 2,
         "third": (CANVAS_MM - 2 * GUTTER_MM) / 3,
         "quarter": (CANVAS_MM - 3 * GUTTER_MM) / 4}

RC = {
    "font.family": "Arial",
    "font.size": TITLE_PT,
    "axes.titlesize": TITLE_PT,
    "axes.labelsize": TITLE_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "axes.linewidth": LW_AXIS,
    "xtick.major.width": LW_AXIS,
    "ytick.major.width": LW_AXIS,
    "xtick.minor.width": LW_MINOR,
    "ytick.minor.width": LW_MINOR,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "xtick.minor.size": 1.0,
    "ytick.minor.size": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": LW_MAIN,
    "patch.linewidth": LW_AXIS,
    "pdf.fonttype": 42,
    "savefig.dpi": DPI,
    "figure.dpi": DPI,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
}

LEFT_MM, BOTTOM_MM, TOP_MM, RIGHT_MM = 9.5, 8.5, 1.5, 1.5


def apply(plt):
    plt.rcParams.update(RC)


def set_line_weight(plt, lw):
    """Per-figure axis weight: LW_AXIS and LW_REF of THIS process (every helper here reads them at call time) and the
    matching rcParams (axes, major ticks, patches; minor ticks at 0.8 x)."""
    global LW_AXIS, LW_REF
    LW_AXIS = LW_REF = lw
    rc = {"axes.linewidth": lw, "xtick.major.width": lw, "ytick.major.width": lw, "patch.linewidth": lw,
          "xtick.minor.width": 0.8 * lw, "ytick.minor.width": 0.8 * lw}
    RC.update(rc)
    plt.rcParams.update(rc)


def figsize(w_mm, h_mm):
    return (w_mm * MM, h_mm * MM)


def panel(plt, tier_or_w, h_mm, left_mm=None, bottom_mm=None, top_mm=None, right_mm=None,
          cbar=False, cbar_mm=2.5, cbar_gap_mm=1.5, cbar_lab_mm=7.0):
    """Figure at the panel's FINAL PLACEMENT SIZE, with the data box pinned in mm."""
    from mpl_toolkits.axes_grid1 import Divider, Size
    w_mm = TIERS[tier_or_w] if isinstance(tier_or_w, str) else float(tier_or_w)
    left = LEFT_MM if left_mm is None else left_mm
    bottom = BOTTOM_MM if bottom_mm is None else bottom_mm
    top = TOP_MM if top_mm is None else top_mm
    right = RIGHT_MM if right_mm is None else right_mm
    extra = (cbar_gap_mm + cbar_mm + cbar_lab_mm) if cbar else 0.0
    box_w = w_mm - left - right - extra
    box_h = h_mm - bottom - top
    if box_w <= 0 or box_h <= 0:
        raise SystemExit(f"SI panel {w_mm} x {h_mm} mm leaves no data box "
                         f"({box_w:.1f} x {box_h:.1f} mm) — widen the panel or the tier")
    frac = (box_w * box_h) / (w_mm * h_mm)
    if frac < 0.55:
        print(f"  [si_panel] note: data frame is {frac*100:.0f}% of the panel "
              f"({box_w:.1f} x {box_h:.1f} of {w_mm:.1f} x {h_mm:.1f} mm)")
    fig = plt.figure(figsize=figsize(w_mm, h_mm), dpi=DPI)
    hs = [Size.Fixed(left * MM), Size.Fixed(box_w * MM)]
    if cbar:
        hs += [Size.Fixed(cbar_gap_mm * MM), Size.Fixed(cbar_mm * MM)]
    vs = [Size.Fixed(bottom * MM), Size.Fixed(box_h * MM)]
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    ax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))
    if not cbar:
        return fig, ax
    cax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=1))
    return fig, ax, cax


def panel_row(plt, n, tier_or_w, h_mm, gap_mm=4.0, left_mm=None, bottom_mm=None,
              top_mm=None, right_mm=None):
    """`n` sub-axes side by side inside ONE panel of the given placement width."""
    from mpl_toolkits.axes_grid1 import Divider, Size
    w_mm = TIERS[tier_or_w] if isinstance(tier_or_w, str) else float(tier_or_w)
    lefts = ([LEFT_MM] * n if left_mm is None
             else [float(left_mm)] * n if not isinstance(left_mm, (list, tuple))
             else [float(v) for v in left_mm])
    if len(lefts) != n:
        raise SystemExit(f"SI panel_row: left_mm has {len(lefts)} entries for {n} columns")
    bottom = BOTTOM_MM if bottom_mm is None else bottom_mm
    top = TOP_MM if top_mm is None else top_mm
    right = RIGHT_MM if right_mm is None else right_mm
    box_w = (w_mm - sum(lefts) - right - (n - 1) * gap_mm) / n
    box_h = h_mm - bottom - top
    if box_w <= 0 or box_h <= 0:
        raise SystemExit(f"SI panel_row: {n} boxes do not fit in {w_mm} x {h_mm} mm "
                         f"(box {box_w:.1f} x {box_h:.1f} mm)")
    fig = plt.figure(figsize=figsize(w_mm, h_mm), dpi=DPI)
    hs, cols = [], []
    for i in range(n):
        if i:
            hs.append(Size.Fixed(gap_mm * MM))
        hs.append(Size.Fixed(lefts[i] * MM))
        cols.append(len(hs))
        hs.append(Size.Fixed(box_w * MM))
    vs = [Size.Fixed(bottom * MM), Size.Fixed(box_h * MM)]
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    axes = [fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=c, ny=1))
            for c in cols]
    return fig, axes


def panel_col(plt, n, tier_or_w, h_mm, gap_mm=1.5, left_mm=None, bottom_mm=None,
              top_mm=None, right_mm=None):
    """`n` sub-axes stacked top to bottom inside ONE panel (the vertical twin of `panel_row`)."""
    from mpl_toolkits.axes_grid1 import Divider, Size
    w_mm = TIERS[tier_or_w] if isinstance(tier_or_w, str) else float(tier_or_w)
    left = LEFT_MM if left_mm is None else left_mm
    bottom = BOTTOM_MM if bottom_mm is None else bottom_mm
    top = TOP_MM if top_mm is None else top_mm
    right = RIGHT_MM if right_mm is None else right_mm
    box_w = w_mm - left - right
    box_h = (h_mm - bottom - top - (n - 1) * gap_mm) / n
    if box_w <= 0 or box_h <= 0:
        raise SystemExit(f"SI panel_col: {n} rows do not fit in {w_mm} x {h_mm} mm "
                         f"(box {box_w:.1f} x {box_h:.1f} mm)")
    fig = plt.figure(figsize=figsize(w_mm, h_mm), dpi=DPI)
    hs = [Size.Fixed(left * MM), Size.Fixed(box_w * MM)]
    vs, rows = [Size.Fixed(bottom * MM)], []
    for i in range(n):
        if i:
            vs.append(Size.Fixed(gap_mm * MM))
        rows.append(len(vs))
        vs.append(Size.Fixed(box_h * MM))
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    axes = [fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=r))
            for r in rows]
    return fig, axes[::-1]


def save_legend(handles, path, ncol=1, labels=None, **kw):
    """A standalone legend key at the contract's type size, cropped tight and transparent, so
    placed at 40-80 %, which is how S2's keys printed at about 3 pt)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    f = plt.figure(figsize=(4, 1), dpi=DPI)
    f.legend(handles=handles, labels=labels, loc="center", frameon=False, fontsize=TICK_PT,
             ncol=ncol, handlelength=kw.pop("handlelength", 1.8), **kw)
    verify_artists(f, label=Path(path).name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(f)
    print("wrote", path)
    return path


def finish(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_PT)


def group_bands(ax, n_groups, color="#F2F2F2", zorder=0):
    """Alternating background bands, one per group, so that a grouped panel separates its groups without a second hue or a divider line."""
    for i in range(n_groups):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, facecolor=color, edgecolor="none", zorder=zorder)


def raincloud(ax, x, data, color, light=None, width=0.42, seed=0, zorder=3,
              box_w=0.10, point_spread=0.14, edge=None, lw=None, log=False, violin=None,
              point_s=3.2):
    """One raincloud at position `x`: half-violin right, box-and-whisker and points left."""
    import numpy as np
    from matplotlib.patches import Rectangle
    d = np.asarray(data, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    edge = INK if edge is None else edge
    lw = LW_AXIS if lw is None else lw
    if light is None:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(color)
        light = mcolors.to_hex((1 - .30 + .30 * r, 1 - .30 + .30 * g, 1 - .30 + .30 * b))

    _want_violin = (d.size >= 8) if violin is None else bool(violin)
    if _want_violin and d.size >= 3 and np.ptp(d) > 0:
        from scipy import stats as _st
        v = np.log10(d) if log else d
        gv = np.linspace(v.min(), v.max(), 200)
        dens = _st.gaussian_kde(v)(gv)
        grid = 10 ** gv if log else gv
        dens = dens / dens.max() * width
        ax.fill_betweenx(grid, x + 0.04, x + 0.04 + dens, facecolor=light,
                         edgecolor=color, lw=LW_MINOR, zorder=zorder)

    rng = np.random.default_rng(seed)
    jx = x - 0.30 - (rng.random(d.size) - 0.5) * point_spread
    ax.scatter(jx, d, s=point_s, color=color, alpha=0.55, lw=0, zorder=zorder + 1)

    p10, p25, p50, p75, p90 = np.percentile(d, [10, 25, 50, 75, 90])
    bx = x - 0.12
    ax.plot([bx, bx], [p10, p25], color=edge, lw=lw, solid_capstyle="butt", zorder=zorder + 2)
    ax.plot([bx, bx], [p75, p90], color=edge, lw=lw, solid_capstyle="butt", zorder=zorder + 2)
    for y in (p10, p90):
        ax.plot([bx - box_w / 4, bx + box_w / 4], [y, y], color=edge, lw=lw,
                solid_capstyle="butt", zorder=zorder + 2)
    ax.add_patch(Rectangle((bx - box_w / 2, p25), box_w, p75 - p25, facecolor=light,
                           edgecolor=edge, linewidth=lw, zorder=zorder + 3))
    ax.plot([bx - box_w / 2, bx + box_w / 2], [p50, p50], color=edge, lw=lw * 1.6,
            solid_capstyle="butt", zorder=zorder + 4)
    return (float(p10), float(p25), float(p50), float(p75), float(p90))


def marginal_density(fig, ax, x, y, pad_mm=0.6, size_mm=4.0, color=PALE):
    """Thin marginal densities above and to the right of a scatter."""
    import numpy as np
    from scipy import stats as _st
    p = ax.get_position()
    fw, fh = fig.get_size_inches()
    pad, size = pad_mm * MM / fh, size_mm * MM / fh
    padx, sizex = pad_mm * MM / fw, size_mm * MM / fw
    axt = fig.add_axes([p.x0, p.y1 + pad, p.width, size])
    axr = fig.add_axes([p.x1 + padx, p.y0, sizex, p.height])
    for a in (axt, axr):
        a.set_axis_off()
    for a, v, lim, horiz in ((axt, np.asarray(x, float), ax.get_xlim(), True),
                             (axr, np.asarray(y, float), ax.get_ylim(), False)):
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        g = np.linspace(lim[0], lim[1], 200)
        d = _st.gaussian_kde(v)(g)
        d = d / d.max()
        if horiz:
            a.fill_between(g, 0, d, color=color, lw=0)
            a.set_xlim(lim); a.set_ylim(0, 1.05)
        else:
            a.fill_betweenx(g, 0, d, color=color, lw=0)
            a.set_ylim(lim); a.set_xlim(0, 1.05)
    return axt, axr


def verify_artists(fig, min_pt=FLOOR_PT, label=""):
    """EXACT type-size check on a figure we rendered: ask every Text artist its size."""
    import numpy as np
    import matplotlib.text as mtext
    fig.canvas.draw()
    sizes, small = [], []
    for t in fig.findobj(mtext.Text):
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible():
            continue
        pt = float(t.get_fontsize())
        sizes.append(pt)
        if pt < min_pt:
            small.append((pt, s[:28]))
    if not sizes:
        return None
    sizes = np.array(sizes)
    print(f"  [type] {label or 'figure'}: {len(sizes)} text artists, "
          f"min {sizes.min():.1f} pt | median {np.median(sizes):.1f} pt | max {sizes.max():.1f} pt")
    if small:
        raise SystemExit(f"TYPE-SIZE FLOOR: {len(small)} text artists below {min_pt:.0f} pt: "
                         f"{small[:6]}. Raise the size or drop the text — do not ship it.")
    return sizes.min(), float(np.median(sizes)), sizes.max()


def verify(path, min_pt=FLOOR_PT, quantile=5.0):
    """Re-measure a rendered PNG with the contract's ruler: pt = glyph height / 0.716 / (dpi/72)."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    im = Image.open(path).convert("L")
    dpi = im.info.get("dpi", (DPI, DPI))[0] or DPI
    d = np.array(im) < 128
    lab, _ = ndimage.label(d, structure=np.ones((3, 3)))
    hs = []
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (3 < max(h, w) < 80 and min(h, w) > 2):
            continue
        if max(h, w) / 0.716 / (dpi / 72.0) < 2.5:
            continue
        if (lab[sl] == i + 1).sum() / float(h * w) > 0.70:
            continue
        hs.append(max(h, w))
    if not hs:
        print(f"  [verify] {path}: no glyph-like blobs found, skipped")
        return None
    hs = np.array(hs)
    hs = hs[hs >= 0.55 * np.percentile(hs, 90)]
    if len(hs) < 5:
        print(f"  [verify] {path}: too few full-height glyphs ({len(hs)}), skipped")
        return None
    pts = hs / 0.716 / (dpi / 72.0)
    q = np.percentile(pts, [quantile, 50, 95])
    below = (pts < min_pt).mean() * 100
    print(f"  [verify] {path.name if hasattr(path,'name') else path}: "
          f"p{quantile:.0f} {q[0]:.1f} pt | median {q[1]:.1f} pt | p95 {q[2]:.1f} pt | "
          f"{below:.0f}% below {min_pt:.0f} pt")
    if q[0] < min_pt:
        print(f"  [verify] note: {below:.0f}% of glyphs read under {min_pt:.0f} pt. Screening "
              f"only — check for prose (x-height reads ~27 % low), subscripts, or a genuinely "
              f"mis-sized render before calling it a fault.")
    return q


def check_edges(path, border=3, thresh=250):
    """Ink on the canvas border means something was cut off. Returns the offending sides."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    sides = {"left": a[:, :border], "right": a[:, -border:],
             "top": a[:border, :], "bottom": a[-border:, :]}
    return [k for k, v in sides.items() if v.min() < thresh]


def drawn_outside(fig, tol_px=2.0):
    """Text actually drawn (recorded during a draw, so out-of-view tick labels do not count) whose box runs past the
    canvas by more than tol_px."""
    import matplotlib.text as mtext
    drawn, orig = [], mtext.Text.draw

    def record(self, renderer, *a, **k):
        if self.get_visible() and (self.get_text() or "").strip():
            drawn.append(self)
        return orig(self, renderer, *a, **k)

    mtext.Text.draw = record
    try:
        fig.canvas.draw()
    finally:
        mtext.Text.draw = orig
    r, w, h, out = fig.canvas.get_renderer(), fig.bbox.width, fig.bbox.height, {}
    for t in drawn:
        b = t.get_window_extent(r)
        for side, v in (("left", -b.x0), ("bottom", -b.y0), ("right", b.x1 - w), ("top", b.y1 - h)):
            if v > tol_px and v > out.get(side, (0, ""))[0]:
                out[side] = (v, t.get_text()[:30])
    return [(s, round(v, 1), txt) for s, (v, txt) in out.items()]


def save(fig, path, verify_type=True, check_clip=True, **kw):
    """Save at the contract dpi, gated on the EXACT artist-level type-size check."""
    from pathlib import Path
    path = Path(path)
    if verify_type:
        verify_artists(fig, label=path.name)
    if check_clip and kw.get("bbox_inches") is None:
        out = drawn_outside(fig)
        if out:
            raise SystemExit(f"{path.name}: drawn content runs past the canvas {out} (px) — text there is cut off "
                             f"without touching the border. Widen the matching gutter or wrap the label (same words).")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white", **kw)
    print("wrote", path)
    if check_clip:
        cut = check_edges(path)
        if cut:
            raise SystemExit(
                f"{path.name}: ink touches the canvas {', '.join(cut)} edge — a label is being "
                f"cut off. Widen the matching gutter in panel(): left_mm for a y label, "
                f"bottom_mm for an x label, top_mm for a significance mark. The file was written "
                f"so you can look at it; do not ship it.")
    return path
