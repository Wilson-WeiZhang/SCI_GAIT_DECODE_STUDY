"""Shared figure settings

Shared panel grammar — ONE render size, ONE type size, ONE axis weight

Imported by the figure scripts; it renders no panel of its own.
"""


MM = 1 / 25.4
FS = 1.2
PANEL_MM = 52.0
DPI = 400

INK, MID, PALE = "#1A1A1A", "#666666", "#B3B3B3"

BASE_PT = 7 * FS
TICK_PT = 8 * FS
NOTE_PT = 6.8 * FS
STAR_PT = 9 * FS

RC = {
    "font.family": "Arial",
    "font.size": BASE_PT,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
}


BOX_MM = 36.0
LEFT_MM = 13.0
BOTTOM_MM = 11.0
TOP_MM = 2.5
RIGHT_MM = 2.5
CBAR_MM = 3.0
CBAR_GAP_MM = 2.0
CBAR_LAB_MM = 9.0

COLUMN_MM = 183.0
GUTTER_MM = 3.0
TIERS = {"full": COLUMN_MM, "half": (COLUMN_MM - GUTTER_MM) / 2,
         "third": (COLUMN_MM - 2 * GUTTER_MM) / 3, "quarter": (COLUMN_MM - 3 * GUTTER_MM) / 4}


def figsize(w_mm=PANEL_MM, h_mm=PANEL_MM):
    """Panel size in inches. Default = Fig.3's 52 x 52 mm square."""
    return (w_mm * MM, h_mm * MM)


def panel_axes(plt, cbar=False, box_mm=BOX_MM, left_mm=None, bottom_mm=None,
               top_mm=None, tier=None, aspect=1.0, right_mm=None,
               cbar_mm=None, cbar_gap_mm=None, cbar_lab_mm=None):
    """Figure + axes whose DATA AREA is exactly `box_mm` x `box_mm`, regardless of labels or
    of whether a colorbar is present."""
    from mpl_toolkits.axes_grid1 import Divider, Size
    left_mm = LEFT_MM if left_mm is None else left_mm
    bottom_mm = BOTTOM_MM if bottom_mm is None else bottom_mm
    top_mm = TOP_MM if top_mm is None else top_mm
    right_mm = RIGHT_MM if right_mm is None else right_mm
    cbar_mm = CBAR_MM if cbar_mm is None else cbar_mm
    cbar_gap_mm = CBAR_GAP_MM if cbar_gap_mm is None else cbar_gap_mm
    cbar_lab_mm = CBAR_LAB_MM if cbar_lab_mm is None else cbar_lab_mm
    extra = (cbar_gap_mm + cbar_mm + cbar_lab_mm) if cbar else 0.0
    if tier is not None:
        box_w = TIERS[tier] - left_mm - extra - right_mm
        box_h = box_w * aspect
    else:
        box_w = box_h = box_mm
    w_mm = left_mm + box_w + extra + right_mm
    h_mm = bottom_mm + box_h + top_mm
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=DPI)
    hs = [Size.Fixed(left_mm * MM), Size.Fixed(box_w * MM)]
    if cbar:
        hs += [Size.Fixed(cbar_gap_mm * MM), Size.Fixed(cbar_mm * MM)]
    vs = [Size.Fixed(bottom_mm * MM), Size.Fixed(box_h * MM)]
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    ax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))
    if not cbar:
        return fig, ax
    cax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=1))
    return fig, ax, cax


def grid_axes(plt, nrow, ncol, cell_mm=18.0, gap_mm=1.2, cbar=False,
              left_mm=None, bottom_mm=None, top_mm=None, tier=None, cell_aspect=1.0):
    """Grid of `nrow` x `ncol` axes whose CELLS are each exactly `cell_mm` square, with fixed
    `gap_mm` gutters and an optional colorbar channel on the right."""
    from mpl_toolkits.axes_grid1 import Divider, Size
    left_mm = LEFT_MM if left_mm is None else left_mm
    bottom_mm = BOTTOM_MM if bottom_mm is None else bottom_mm
    top_mm = TOP_MM if top_mm is None else top_mm
    if tier is not None:
        extra = (CBAR_GAP_MM + CBAR_MM + CBAR_LAB_MM) if cbar else 0.0
        cell_mm = (TIERS[tier] - left_mm - RIGHT_MM - extra - (ncol - 1) * gap_mm) / ncol
    cell_h = cell_mm * cell_aspect

    hs, cols = [Size.Fixed(left_mm * MM)], []
    for c in range(ncol):
        if c:
            hs.append(Size.Fixed(gap_mm * MM))
        cols.append(len(hs))
        hs.append(Size.Fixed(cell_mm * MM))
    cax_nx = None
    if cbar:
        hs.append(Size.Fixed(CBAR_GAP_MM * MM))
        cax_nx = len(hs)
        hs.append(Size.Fixed(CBAR_MM * MM))

    vs, rows = [Size.Fixed(bottom_mm * MM)], []
    for r in range(nrow):
        if r:
            vs.append(Size.Fixed(gap_mm * MM))
        rows.append(len(vs))
        vs.append(Size.Fixed(cell_h * MM))

    w_mm = left_mm + ncol * cell_mm + (ncol - 1) * gap_mm + RIGHT_MM + (
        CBAR_GAP_MM + CBAR_MM + CBAR_LAB_MM if cbar else 0.0)
    h_mm = bottom_mm + nrow * cell_h + (nrow - 1) * gap_mm + top_mm
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=DPI)
    div = Divider(fig, (0, 0, 1, 1), hs, vs, aspect=False)
    axes = [[fig.add_axes(div.get_position(),
                          axes_locator=div.new_locator(nx=cols[c], ny=rows[nrow - 1 - r]))
             for c in range(ncol)] for r in range(nrow)]
    if not cbar:
        return fig, axes
    cax = fig.add_axes(div.get_position(),
                       axes_locator=div.new_locator(nx=cax_nx, ny=rows[0], ny1=rows[-1] + 1))
    return fig, axes, cax


def apply(plt):
    """Set the shared grammar on a matplotlib pyplot module."""
    plt.rcParams.update(RC)


def finish(ax, square=False):
    """Common axes treatment: despine top/right, outward ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", labelsize=TICK_PT)
    if square:
        ax.set_box_aspect(1)


JOINT_DISP = {"LHip": "L hip", "LKne": "L knee", "LKnee": "L knee", "LAnk": "L ankle",
              "RHip": "R hip", "RKne": "R knee", "RKnee": "R knee", "RAnk": "R ankle"}


def jlab(key):
    """Display label for a compact joint key; anything unknown passes through unchanged."""
    return JOINT_DISP.get(key, key)


def pstr(p, sym="p"):
    """House-style p as mathtext: `$\\it{p}$ < 0.001` or `$\\it{p}$ = 0.012`."""
    s = f"$\\it{{{sym}}}$"
    return f"{s} < 0.001" if p < 0.001 else f"{s} = {p:.3f}"


def num(v, fmt="+.2f"):
    """A number for a panel, with a typographic minus (U+2212), not a hyphen."""
    return format(v, fmt).replace("-", "−")


def stars(p):
    """Significance mark for a bracket. Conventional thresholds; n.s. spelled out."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


def _yspace(ax):
    """Bracket geometry in the y axis' OWN scale: `(to_pos, to_data, lo, hi)`."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        from math import log10
        return log10, lambda v: 10.0 ** v, log10(lo), log10(hi)
    return (lambda v: v), (lambda v: v), lo, hi


def sig_bracket(ax, x1, x2, p, y, tick=None, color=None, lw=1.1, text=None, star_pt=None, ns_pt=None):
    """One significance bracket from x1 to x2 with its mark, in DATA coordinates."""
    to_pos, to_data, lo, hi = _yspace(ax)
    yp = to_pos(y)
    tick = .018 * (hi - lo) if tick is None else tick
    color = INK if color is None else color
    star_pt = STAR_PT if star_pt is None else star_pt
    ax.plot([x1, x1, x2, x2],
            [to_data(yp), to_data(yp + tick), to_data(yp + tick), to_data(yp)],
            color=color, lw=lw, solid_capstyle="butt", clip_on=False, zorder=6)
    mk = stars(p) if text is None else text
    ax.text((x1 + x2) / 2, to_data(yp + tick + .008 * (hi - lo)), mk, ha="center", va="bottom",
            fontsize=star_pt if mk != "n.s." else (star_pt * .65 if ns_pt is None else ns_pt), color=color,
            clip_on=False, zorder=6)
    return to_data(yp + tick + .055 * (hi - lo))


def _pt_in_data(ax, pt):
    """`pt` points expressed in the axes' y AXIS-SCALE units (decades on a log axis)."""
    _, _, lo, hi = _yspace(ax)
    fig = ax.figure
    try:
        r = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    try:
        ax_h_pt = ax.get_window_extent(renderer=r).height * 72.0 / fig.dpi
    except Exception:
        ax_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
    return (hi - lo) * pt / ax_h_pt if ax_h_pt > 0 else 0.0


def sig_brackets(ax, pairs, headroom=0.30, y0=None, step=None, star_pt=None, lw=1.1, ns_pt=None):
    """Stack several brackets above the data and make room for them."""
    to_pos, to_data, lo, hi = _yspace(ax)
    rng = hi - lo
    tick, gap = .016 * rng, .008 * rng
    mark_h = _pt_in_data(ax, STAR_PT if star_pt is None else star_pt) or .062 * rng
    step = tick + gap + mark_h + .02 * rng if step is None else step
    y0 = hi - .04 * rng if y0 is None else to_pos(y0)
    for i, (x1, x2, p) in enumerate(pairs):
        sig_bracket(ax, x1, x2, p, to_data(y0 + i * step), tick=tick, lw=lw, star_pt=star_pt, ns_pt=ns_pt)
    top = y0 + (len(pairs) - 1) * step + tick + gap + mark_h + .02 * rng
    ax.set_ylim(to_data(lo), to_data(max(hi + headroom * rng, top)))


def stat_note(ax, lines, loc="upper left", pad=0.035, fontsize=None, color=None):
    """The one short statistics line a scatter panel carries, in an empty corner."""
    if isinstance(lines, str):
        lines = [lines]
    assert len(lines) <= 2, f"stat_note takes at most 2 lines, got {len(lines)}: {lines}"
    base = INK if color is None else color
    items = [(ln, base) if isinstance(ln, str) else (ln[0], ln[1]) for ln in lines]
    fs = NOTE_PT if fontsize is None else fontsize
    va, ha = ("top", "left")
    x, y = pad, 1 - pad
    if "lower" in loc:
        va, y = "bottom", pad
    if "right" in loc:
        ha, x = "right", 1 - pad
    if len({c for _, c in items}) == 1:
        ax.text(x, y, "\n".join(t for t, _ in items), transform=ax.transAxes, ha=ha, va=va,
                fontsize=fs, color=items[0][1], linespacing=1.35, zorder=7)
        return
    step = 1.35 * fs / (ax.get_window_extent().height * 72.0 / ax.figure.dpi) \
        if ax.get_window_extent().height else 0.09
    order = items if va == "top" else list(reversed(items))
    for i, (t, c) in enumerate(order):
        yy = y - i * step if va == "top" else y + i * step
        ax.text(x, yy, t, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=c, zorder=7)


def write_stats(path, lines):
    """Write the statistics of a panel to a text file."""
    from pathlib import Path
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {Path(path).name} ")
