"""Figure 6 — Representational geometry of patient imagined walking

panel a — SCI-patient gait-cycle harmonic phase scalp maps (patient analogue of
Fig. 3 panel b)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import csv
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import fig3_harmonic_phase_responder_split as H

plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "mathtext.default": "regular"})

ROOT = Path(__file__).resolve().parents[1]
DUMP = (ROOT / "data/decoder_output/diag_patient_harmonic_phase/"
               "persession_harmonic_phase_cycle9_complex.csv")
OUT_PNG = ROOT / "figures_out" / "fig6" / "fig6_patient_harmonic_phase.png"
SI_DIR = ROOT / "figures_out" / "supplementary"
SRC = ROOT / "04_source_data" / "fig6" / "fig6_patient_harmonic_phase_source_data.csv"
META = ROOT / "04_source_data" / "fig6" / "fig6_patient_harmonic_phase_metadata.csv"

BUILD_ID = "main"
DUMP_TAG = "diag_patient_harmonic_phase"
TRIAL_DUMP_TAG = "diag_pertrial_phasemap_post_operative"
TRIAL_DUMP = (ROOT / "data/decoder_output" / TRIAL_DUMP_TAG /
              "pertrial_harmonic_phase_cycle9_complex.csv")
LME_ALPHA = 0.05
MAX_P_NAN_FRAC = 0.10
PANEL_A_FONT_SCALE = 1.5

ROW_LABEL = {1: "$k_1$", 2: "$k_2$", 3: "$k_3$"}

P1_SESSIONS = ["pat01_sess04", "pat01_sess05", "pat01_sess06",
               "pat01_sess07", "pat01_sess08", "pat01_sess09"]
P2_SESSIONS = ["pat02_sess03", "pat02_sess05", "pat02_sess06", "pat02_sess08",
               "pat02_sess09"]
POST_OPERATIVE = P1_SESSIONS + P2_SESSIONS
PRE_OPERATIVE = ["pat01_sess01", "pat01_sess02", "pat02_sess01", "pat02_sess02"]
NOT_IN_COHORT = ["pat01_sess01iw", "pat02_sess04", "pat02_sess07"]

PATIENTS = [("Patient 1", P1_SESSIONS, "P1"),
            ("Patient 2", P2_SESSIONS, "P2")]

def load_frozen_record(channels, fracs_plot, harms):
    """The DRAWN significance mask + signed values from the frozen Source Data CSV."""
    import pandas as pd
    if not SRC.exists():
        raise SystemExit(f"Missing the frozen Source Data needed for the SI panels: {SRC}")
    df = pd.read_csv(SRC)
    ci = {c: i for i, c in enumerate(channels)}
    pi = {round(float(f), 6): i for i, f in enumerate(fracs_plot)}
    hi = {h: i for i, h in enumerate(harms)}
    shape = (len(channels), len(fracs_plot), len(harms))
    sig, signed = {}, {}
    for tag, d in df.groupby("patient"):
        s = np.zeros(shape, bool)
        v = np.full(shape, np.nan)
        a = d.channel.map(ci).to_numpy()
        b = d.cycle_fraction.round(6).map(pi).to_numpy()
        c = d.harmonic_k.map(hi).to_numpy()
        ok = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
        a, b, c = a[ok].astype(int), b[ok].astype(int), c[ok].astype(int)
        s[a, b, c] = d.sig_bh.to_numpy()[ok].astype(bool)
        v[a, b, c] = d.signed_phase_modulation.to_numpy()[ok]
        sig[tag] = s
        signed[tag] = v
    trials = {}
    if META.exists():
        for r in csv.reader(META.open(newline="", encoding="utf-8")):
            if len(r) == 2 and r[0].endswith("_n_trials"):
                trials[r[0][:-len("_n_trials")]] = r[1]
    return sig, signed, trials


def render_si_panel(gsigned, channels, fracs_plot, harms, clim, info, ch_index,
                    nrow_meta, out_png, row_labels, sig):
    """One per-patient strip for the SI panel spec, through the SHARED Fig.3 engine."""
    import matplotlib.figure as mfig
    ok_titles = {f"{100 * cf:.0f}%" for cf in fracs_plot}
    ok_texts = set(row_labels or [])
    orig = mfig.Figure.savefig

    def hooked(self, *a, **kw):
        bad = []
        for i, ax in enumerate(self.axes):
            t = ax.get_title()
            if t and t not in ok_titles:
                bad.append(f"axes[{i}] title {t!r}")
            if ax.get_legend() is not None:
                bad.append(f"axes[{i}] legend present")
            extra = [x.get_text() for x in ax.texts if x.get_text() not in ok_texts]
            if extra:
                bad.append(f"axes[{i}] texts {extra}")
        if self.texts:
            bad.append(f"fig.texts {[x.get_text() for x in self.texts]}")
        if bad:
            raise SystemExit("PANEL SPEC VIOLATION: " + "; ".join(bad))
        print(f"panel spec OK: {len(self.axes)} axes; only the gait-cycle % column labels "
              f"(as Axes titles) and the {sorted(ok_texts)} row labels are text")
        return orig(self, *a, **kw)

    mfig.Figure.savefig = hooked
    try:
        _si_render_strip(gsigned, fracs_plot, harms, clim, info, ch_index, out_png, row_labels, sig)
    finally:
        mfig.Figure.savefig = orig
    print("wrote", out_png)


SI_PX = (3780, 1276)
SI_COL_MM = [10.384 + 16.244 * i for i in range(9)]
SI_ROW_MM = [10.573, 28.394, 46.211]
SI_HEAD_R_MM = 5.603
SI_F = 240 / 600 * 3780 / 3210
SI_MASK_PARAMS = dict(H.MASK_PARAMS, markersize=H.MASK_PARAMS["markersize"] * SI_F,
                      markeredgewidth=H.MASK_PARAMS["markeredgewidth"] * SI_F)


def _si_topo(fig, ax, cn, cx_mm, cy_mm, r_mm, f=SI_F, head_lw=0.5, contour_lw=0.3):
    """SI panel only: contract line weights on one mne topomap (mne 1.10 draws head / nose / ears at a fixed 1 pt and the
    contours at mask markeredgewidth / 2, so both are set after the call: 0.5 / 0.3 pt; the sensor
    dot is scaled by SI_F), then move the axes so the head circle is centred at (cx, cy) mm with radius r mm."""
    from matplotlib.collections import PathCollection
    for ln in ax.lines:
        if ln.get_marker() in ("None", "", None) and ln.get_linestyle() not in ("None", ""):
            ln.set_linewidth(head_lw)
    if cn is not None:
        cn.set_linewidth(contour_lw)
    for col in ax.collections:
        if isinstance(col, PathCollection):
            col.set_sizes([0.25 * f ** 2])
            col.set_linewidths([1.0 * f])
    W, H_ = fig.bbox.width, fig.bbox.height
    ax.apply_aspect()
    ln = max(ax.lines, key=lambda l: len(l.get_xdata()))
    x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
    hx, hy, hr = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (x.max() - x.min()) / 2
    (px, py), (qx, _) = ax.transData.transform([(hx, hy), (hx + hr, hy)])
    s = r_mm / 25.4 * fig.dpi / (qx - px)
    tx, ty = cx_mm / 25.4 * fig.dpi, H_ - cy_mm / 25.4 * fig.dpi
    b = ax.get_position(original=True)
    ax.set_position([(tx + (b.x0 * W - px) * s) / W, (ty + (b.y0 * H_ - py) * s) / H_, b.width * s, b.height * s])


SI_EEGLAB_PLOTRAD_MIN = 1.02
SI_EEGLAB_EAR_EXT = 0.547 * 2


def _si_eeglab_frame(info):
    """SI panel only: EEGLAB layout of `info`'s channels (n x 2, in `info` order) and its largest radius, plus the frame
    radius and axes limits, all in its data units."""
    import mne
    P = np.array([ch["loc"][:3] for ch in info["chs"]])
    Rd = 0.5 - np.degrees(np.arcsin(P[:, 2] / np.linalg.norm(P, axis=1))) / 180
    az = np.arctan2(P[:, 1], P[:, 0])
    pos = np.column_stack([Rd * np.cos(az), Rd * np.sin(az)])
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    im, _ = mne.viz.plot_topomap(np.zeros(len(P)), info, axes=ax, show=False, vlim=(-1, 1), contours=6, sensors=True,
                                 sphere="eeglab", res=64, image_interp="cubic", extrapolate="head")
    ax.apply_aspect()
    x, y = (np.asarray(v) for v in max(ax.lines, key=lambda l: len(l.get_xdata())).get_data())
    hx, hy, R0 = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (x.max() - x.min()) / 2
    e = im.get_clip_path().get_fully_transformed_path().get_extents()
    (a, _), (b, _) = ax.transData.inverted().transform([[e.x0, e.y0], [e.x1, e.y0]])
    frame = (hx, hy, R0, (b - a) / 2, ax.get_xlim(), ax.get_ylim())
    plt.close(f)
    return pos, float(Rd.max()), frame


def _si_eeglab_gate(fig, ax, im, R, head_lw, label):
    """SI panel only: the map is clipped at the head circle (radius R), the ears reach SI_EEGLAB_EAR_EXT, and no
    electrode mark (sensor scatter or significance-mask marker) reaches past the outer edge of the head line."""
    from matplotlib.collections import PathCollection
    ax.apply_aspect()
    c0 = ax.transData.transform([[0.0, 0.0]])[0]
    r_px = ax.transData.transform([[R, 0.0]])[0, 0] - c0[0]
    e = im.get_clip_path().get_fully_transformed_path().get_extents()
    outl = [ln for ln in ax.lines if ln.get_marker() in ("None", "", None) and ln.get_linestyle() not in ("None", "")]
    ear = max(float(np.max(np.abs(ln.get_xdata()))) for ln in outl) / R
    if abs((e.x1 - e.x0) / 2 / r_px - 1) > 1e-3 or abs(ear - SI_EEGLAB_EAR_EXT) > 1e-3:
        raise SystemExit(f"{label}: clip {(e.x1 - e.x0) / 2 / r_px:.4f} R, ear {ear:.4f} R "
                         f"(expected 1, {SI_EEGLAB_EAR_EXT})")
    pt = fig.dpi / 72
    marks = [(col.get_offsets(), np.sqrt(col.get_sizes()[0]) / 2 + col.get_linewidths()[0] / 2)
             for col in ax.collections if isinstance(col, PathCollection)]
    marks += [(ln.get_xydata(), ln.get_markersize() / 2 + ln.get_markeredgewidth() / 2)
              for ln in ax.lines if ln.get_marker() not in ("None", "", None) and len(ln.get_xdata())]
    clear = min(r_px + head_lw / 2 * pt - (np.linalg.norm(ax.transData.transform(xy) - c0, axis=1).max() + rad * pt)
                for xy, rad in marks)
    if clear < 0:
        raise SystemExit(f"{label}: an electrode mark crosses the head outline by {-clear / fig.dpi * 25.4:.3f} mm")
    return clear / fig.dpi * 25.4


def _si_render_strip(gsigned, fracs_plot, harms, clim, info, ch_index, out_png, row_labels, sig):
    """colormap, shared clim, LME significance dots, % column labels, k row labels, colour bar titled on its left) drawn at
    160.0 x 54.0 mm on the 16 cm contract: 6 pt column labels and ticks, 7 pt row labels and colour-bar title."""
    import mne
    import si_panel_style as SI
    with plt.rc_context():
        SI.apply(plt)
        fig = plt.figure(figsize=((SI_PX[0] + 0.25) / SI.DPI, (SI_PX[1] + 0.25) / SI.DPI), dpi=SI.DPI)
        W_MM, H_MM = SI_PX[0] / SI.DPI * 25.4, SI_PX[1] / SI.DPI * 25.4
        if len(fracs_plot) != len(SI_COL_MM) or len(harms) != len(SI_ROW_MM):
            raise SystemExit(f"SI strip layout is fixed at {len(SI_ROW_MM)} x {len(SI_COL_MM)} maps, got "
                             f"{len(harms)} x {len(fracs_plot)}")
        cmap = H.eeglab_classic_redblue_cmap()
        image = None
        pos, rd_max, (hx, hy, R0, disc0, xl0, yl0) = _si_eeglab_frame(info)
        r_mm = SI_HEAD_R_MM * disc0 / R0 / SI_EEGLAB_EAR_EXT
        dot_mm = (SI_MASK_PARAMS["markersize"] + SI_MASK_PARAMS["markeredgewidth"]) / 2 * 25.4 / 72
        plotrad = max(SI_EEGLAB_PLOTRAD_MIN, np.ceil(1000 / (1 - (dot_mm - 0.25 * 25.4 / 72) / r_mm)) / 1000)
        R = plotrad * rd_max
        k = SI_EEGLAB_EAR_EXT * R / disc0
        xl, yl = [(v - hx) * k for v in xl0], [(v - hy) * k for v in yl0]
        clear = []
        for ri, h in enumerate(harms):
            for ci, cf in enumerate(fracs_plot):
                ax = fig.add_axes([0.1, 0.1, 0.1, 0.1])
                mkw = dict(mask=sig[ch_index, ci, ri].astype(bool), mask_params=SI_MASK_PARAMS) if sig is not None else {}
                image, cn = mne.viz.plot_topomap(gsigned[ch_index, ci, ri], pos, axes=ax, show=False, cmap=cmap,
                                                 vlim=clim, contours=6, sensors=True, sphere=(0.0, 0.0, 0.0, R),
                                                 res=256, image_interp="cubic", extrapolate="head", **mkw)
                ax.set(xlim=xl, ylim=yl)
                _si_topo(fig, ax, cn, SI_COL_MM[ci], SI_ROW_MM[ri], r_mm)
                clear.append(_si_eeglab_gate(fig, ax, image, R, 0.5, f"{out_png.name} k{h} {100 * cf:.0f}%"))
                if ri == 0:
                    ax.set_title(f"{100 * cf:.0f}%", fontsize=SI.TICK_PT, pad=1.5)
                if ci == 0 and row_labels is not None:
                    ax.annotate(row_labels[ri], xy=(0, 0.5), xycoords="axes fraction", xytext=(-3.5, 0),
                                textcoords="offset points", ha="right", va="center", fontsize=SI.TITLE_PT)
        print(f"  {out_png.name}: EEGLAB layout, plotrad {plotrad:.3f} x max Rd ({rd_max:.4f}), head radius "
              f"{r_mm:.3f} mm (ears at {SI_HEAD_R_MM * disc0 / R0:.3f} mm); every sensor / "
              f"significance dot inside the head line, smallest clearance {min(clear):.3f} mm over {len(clear)} maps")
        r = SI_HEAD_R_MM
        x0 = SI_COL_MM[-1] + 1.14 * r + 0.45 + 2.29 + 0.56
        yt, yb = SI_ROW_MM[0] - 1.05 * r, SI_ROW_MM[-1] + 1.05 * r
        cax = fig.add_axes([x0 / W_MM, 1 - yb / H_MM, 1.1 / W_MM, (yb - yt) / H_MM])
        cb = fig.colorbar(image, cax=cax)
        cb.ax.yaxis.set_label_position("left")
        cb.set_label("Phase modulation", fontsize=SI.TITLE_PT, labelpad=1.6)
        cb.ax.tick_params(labelsize=SI.TICK_PT, length=1.4, width=SI.LW_AXIS, pad=1.0)
        cb.outline.set_linewidth(SI.LW_AXIS)
        SI.save(fig, out_png)
        plt.close(fig)


SI_CYCLE_PX_H = 142
SI_CYCLE_BAND_MM = (0.3, 5.756)
SI_CYCLE_HW_MM = 5.9
SI_CYCLE_DOT_F = 240 / 600 * 3496 / 3201


def _si_cycle_strip(fracs, out_png):
    """SI panel only: the gait-cycle reference curve with one dot per S6 column (fracs = the nine column phases)."""
    import si_panel_style as SI
    with plt.rc_context():
        SI.apply(plt)
        fig = plt.figure(figsize=((SI_PX[0] + 0.25) / SI.DPI, (SI_CYCLE_PX_H + 0.25) / SI.DPI), dpi=SI.DPI)
        W_MM, H_MM = SI_PX[0] / SI.DPI * 25.4, SI_CYCLE_PX_H / SI.DPI * 25.4
        span, hw = SI_COL_MM[-1] - SI_COL_MM[0], SI_CYCLE_HW_MM
        cy0, cy1 = SI_CYCLE_BAND_MM
        axc = fig.add_axes([(SI_COL_MM[0] - hw) / W_MM, 1 - cy1 / H_MM, (span + 2 * hw) / W_MM, (cy1 - cy0) / H_MM])
        plo, phi = -hw / span, 1 + hw / span
        ph = np.linspace(plo, phi, 500)
        axc.plot(ph, np.cos(2 * np.pi * ph), color="#333333", lw=SI.LW_MAIN, solid_capstyle="round")
        fy = np.cos(2 * np.pi * np.asarray(fracs, float))
        axc.vlines(fracs, -1.2, fy, color="0.72", lw=SI.LW_REF, zorder=1)
        axc.plot(fracs, fy, "o", color="#c0392b", ms=5.5 * SI_CYCLE_DOT_F, mec="white", mew=0.7 * SI_CYCLE_DOT_F,
                 zorder=3)
        axc.set_xlim(plo, phi)
        axc.set_ylim(-1.6, 1.2)
        axc.axis("off")
        SI.save(fig, out_png)
        plt.close(fig)
    print("wrote", out_png)


ENG_A_PX = (3290, 2294)
ENG_A_COL_PX = [293.09 + 314.9525 * i for i in range(9)]
ENG_A_ROW_PX = [384.44, 711.03, 1037.63, 1494.43, 1821.03, 2147.62]
ENG_A_HEAD_R_PX = 110.447
ENG_A_CURVE_PX = (14.4, 133.569)
ENG_A_CURVE_HW = 0.049964
ENG_A_EAR = 1.094
ENG_A_DISC = 1.1385


def _reml_intercept(y, g):
    """Exact REML fit of y ~ 1 + (1|g): (beta, se, tau2 / scale, REML log-likelihood)."""
    from scipy.optimize import minimize_scalar
    labels, gi, n_g = np.unique(g, return_inverse=True, return_counts=True)
    N = y.size
    ybar = np.bincount(gi, weights=y) / n_g

    def fit(gamma):
        w = n_g / (1.0 + n_g * gamma)
        beta = np.sum(w * ybar) / np.sum(w)
        Q = np.sum((y - beta) ** 2) - np.sum(gamma / (1.0 + n_g * gamma) * (n_g * (ybar - beta)) ** 2)
        scale = Q / (N - 1)
        llf = -0.5 * ((N - 1) * np.log(2 * np.pi * scale) + (N - 1)
                      + np.sum(np.log1p(n_g * gamma)) + np.log(np.sum(w)))
        return beta, np.sqrt(scale / np.sum(w)), llf

    r = minimize_scalar(lambda lg: -fit(10.0 ** lg)[2], bounds=(-12.0, 6.0), method="bounded",
                        options={"xatol": 1e-10})
    gamma = 10.0 ** r.x
    if fit(0.0)[2] >= fit(gamma)[2]:
        gamma = 0.0
    beta, se, llf = fit(gamma)
    return float(beta), float(se), float(gamma), float(llf)


def _lme_cell(args):
    """Wald z P for the intercept of y ~ 1 + (1|session); (P, n_trials, n_sessions, tau2 / scale)."""
    from scipy.stats import norm
    y, g = args
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    if y.size < 10 or len(set(g)) < 2:
        return np.nan, int(y.size), len(set(g)), np.nan
    beta, se, gamma, _ = _reml_intercept(y, g)
    return float(2 * norm.sf(abs(beta) / se)), int(y.size), len(set(g)), gamma


def load_trial_maps(channels, fracs_plot, harms, ref_phase):
    """Per-trial signed projection T[trial, ch, point, harm] + session label per trial."""
    import pandas as pd
    if not TRIAL_DUMP.exists():
        raise SystemExit(f"Missing per-trial phase-map dump: {TRIAL_DUMP}\n"
                         f"Run run_pertrial_phasemap_dump_post_operative.sh (not distributed) on the compute server.")
    df = pd.read_csv(TRIAL_DUMP)
    df = df[df.cycle_fraction < 0.9999]
    ch_i = {c: i for i, c in enumerate(channels)}
    pt_i = {round(float(f), 6): i for i, f in enumerate(fracs_plot)}
    h_i = {h: i for i, h in enumerate(harms)}
    df["tkey"] = df.session + "#" + df.trial_ord.astype(str)
    trials = sorted(df.tkey.unique(), key=lambda k: (k.split("#")[0], int(k.split("#")[1])))
    t_i = {k: i for i, k in enumerate(trials)}
    T = np.full((len(trials), len(channels), len(fracs_plot), len(harms)), np.nan)
    ti = df.tkey.map(t_i).values
    ci = df.channel.map(ch_i).values
    pi = df.cycle_fraction.round(6).map(pt_i).values
    hi = df.harmonic_k.map(h_i).values
    ok = np.isfinite(ci) & np.isfinite(pi)
    val = np.real((df.real.values + 1j * df.imag.values) * np.exp(-1j * ref_phase))
    T[ti[ok], ci[ok].astype(int), pi[ok].astype(int), hi[ok]] = val[ok]
    return T, np.array([k.split("#")[0] for k in trials])


def lme_sig(T, sess_of_trial, sess_list):
    """(sig_z [ch,pt,h] bool, p_z, sig_tdf, n_trials, tau2/scale [ch,pt,h]) for one group."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import norm as _norm, t as _t
    tmask = np.isin(sess_of_trial, sess_list)
    X, G = T[tmask], sess_of_trial[tmask]
    shape = X.shape[1:]
    cells = [(X[:, c, p, k], G) for c in range(shape[0]) for p in range(shape[1]) for k in range(shape[2])]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_lme_cell, cells, chunksize=40))
    p = np.array([r[0] for r in res]).reshape(shape)
    tau = np.array([r[3] for r in res]).reshape(shape)
    sig = np.zeros(shape, bool)
    for k in range(shape[2]):
        sig[..., k] = H._bh(p[..., k].ravel(), LME_ALPHA).reshape(shape[:2])
    z = _norm.isf(np.clip(p, 1e-300, 1) / 2)
    p_t = 2 * _t.sf(z, len(sess_list) - 1)
    p_t[~np.isfinite(p)] = np.nan
    sig_t = np.zeros(shape, bool)
    for k in range(shape[2]):
        sig_t[..., k] = H._bh(p_t[..., k].ravel(), LME_ALPHA).reshape(shape[:2])
    return sig, p, sig_t, int(X.shape[0]), tau


def stack_patients(pngs, labels, out_png, width=1900):
    """Vertical stack of the per-patient strips with a label band each."""
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for p in ("C:/Windows/Fonts/arialbd.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if Path(p).exists():
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    def rs(im):
        return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)

    ims = [rs(Image.open(p).convert("RGB")) for p in pngs]
    band = 60
    out = Image.new("RGB", (width, sum(band + im.height + 12 for im in ims)), "white")
    dr = ImageDraw.Draw(out)
    y = 0
    for im, lab in zip(ims, labels):
        dr.text((20, y + 12), lab, fill=(20, 20, 20), font=font(36))
        out.paste(im, (0, y + band))
        y += band + im.height + 12
    out.save(out_png, dpi=(150, 150))


def main():
    import mne
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    SRC.parent.mkdir(parents=True, exist_ok=True)
    SI_DIR.mkdir(parents=True, exist_ok=True)
    if not DUMP.exists():
        raise SystemExit(f"Missing  patient dump: {DUMP}\n"
                         f"Run run_patient_harmonic_maps.sh (not distributed) on the compute server and copy it back.")

    H.DUMP = DUMP
    all_sessions, channels, fracs, harms, A = H.load_dump()

    missing = [s for s in POST_OPERATIVE if s not in all_sessions]
    if missing:
        raise SystemExit(f"post_operative sessions absent from {DUMP.name}: {missing}")
    idx = {s: i for i, s in enumerate(all_sessions)}
    mask_post = np.zeros(len(all_sessions), bool)
    for s in POST_OPERATIVE:
        mask_post[idx[s]] = True
    dropped = [s for s in all_sessions if s not in POST_OPERATIVE]
    n_sess = int(mask_post.sum())

    full_c = H.group_mean(A, mask_post)
    flat = full_c.ravel()
    ref_phase = np.angle(flat[np.nanargmax(np.abs(flat))])
    clim_abs = float(np.nanquantile(np.abs(np.real(full_c * np.exp(-1j * ref_phase))), 0.98))
    clim = (-clim_abs, clim_abs)
    signed_sess = np.real(A * np.exp(-1j * ref_phase))

    raw = mne.io.read_raw_eeglab(str(H.EEG_SET), preload=False, verbose="ERROR")
    keep = [c for c in raw.ch_names if c in channels]
    info = raw.copy().pick(keep).info
    ch_index = [channels.index(c) for c in keep]

    groups = [("patient", mask_post, POST_OPERATIVE, f"n={n_sess} post-operative sessions", "P_all", OUT_PNG)]
    for disp, sess_list, tag in PATIENTS:
        m = np.zeros(len(all_sessions), bool)
        for s in sess_list:
            m[idx[s]] = True
        png = OUT_PNG.with_name(f"fig6_patient_harmonic_phase_{tag}.png")
        groups.append((disp, m, sess_list, f"{tag}, {len(sess_list)} post-operative sessions", tag, png))

    wrap_maxdiff = None
    keep_pts = list(range(len(fracs)))
    if float(fracs[-1]) == 1.0:
        d = A[:, :, -1, :] - A[:, :, 0, :]
        wrap_maxdiff = float(np.nanmax(np.abs(d)))
        if wrap_maxdiff > 1e-9:
            raise SystemExit(f"cycle_fraction 1.000 is NOT the wrap of 0.000 "
                             f"(max|diff| = {wrap_maxdiff:.3e}); do not drop it blindly.")
        keep_pts = list(range(len(fracs) - 1))
    fracs_plot = [fracs[i] for i in keep_pts]
    row_labels = [ROW_LABEL[h] for h in harms]

    if PANEL_MODE:
        frozen_sig, frozen_signed, frozen_trials = load_frozen_record(channels, fracs_plot, harms)
        T = sess_of_trial = None
    else:
        T, sess_of_trial = load_trial_maps(channels, fracs_plot, harms, ref_phase)
        missing_t = [s for s in POST_OPERATIVE if s not in set(sess_of_trial)]
        if missing_t:
            raise SystemExit(f"post_operative sessions absent from the per-trial dump: {missing_t}")
        for s in POST_OPERATIVE:
            a = np.nanmean(T[sess_of_trial == s], 0).ravel()
            b = np.real(A[idx[s]][:, keep_pts, :] * np.exp(-1j * ref_phase)).ravel()
            okk = np.isfinite(a) & np.isfinite(b)
            rr = np.corrcoef(a[okk], b[okk])[0, 1]
            assert rr > 0.99, f"trial dump does not reproduce the session dump for {s}: r={rr:.4f}"

    lme_by_tag = {}
    if not PANEL_MODE:
        for _disp, _mask, _sess, _cmp, _tag, _png in groups:
            lme_by_tag[_tag] = lme_sig(T, sess_of_trial, _sess)
        n_p = sum(r[1].size for r in lme_by_tag.values())
        n_nan = sum(int((~np.isfinite(r[1])).sum()) for r in lme_by_tag.values())
        print(f"LME Wald p: {n_nan} of {n_p} cells NaN ({n_nan / n_p:.1%}); committed build 0 of 4320")
        if n_nan > MAX_P_NAN_FRAC * n_p:
            raise SystemExit(
                f"NaN-SE GUARD: p_lme_wald_z would be NaN in {n_nan} of {n_p} cells "
                f"({n_nan / n_p:.1%} > {MAX_P_NAN_FRAC:.0%}; the committed build has 0 of 4320), "
                f"which would empty the Fig. 6a significance dots. Nothing was written; the committed "
                f"PNGs and Source Data are untouched. Check the per-trial dump and _reml_intercept.")

    src_rows, meta_rows, patient_pngs, patient_labels = [], [], [], []
    c_report = {}
    combined_blocks, combined_sigs = {}, {}
    sig_report, sig_report_tdf, sig_report_sess, n_trials_report = {}, {}, {}, {}
    for disp, mask, sess_list, cmp_label, tag, png in groups:
        gc = H.group_mean(A, mask)
        gsigned = np.real(gc * np.exp(-1j * ref_phase))[:, keep_pts, :]
        if PANEL_MODE:
            sig = frozen_sig[tag]
            dev = np.nanmax(np.abs(gsigned - frozen_signed[tag]))
            if not (dev < 1e-7):
                raise SystemExit(f"FROZEN-RECORD GUARD [{tag}]: the map computed in this run "
                                 f"differs from the Source Data by {dev:.3e}; the SI panel may "
                                 f"not reuse a mask that belongs to different values.")
            print(f"  {tag}: frozen mask reused, map matches the Source Data "
                  f"(max |diff| = {dev:.1e})")
            n_tr = frozen_trials.get(tag, "")
            p_lme = np.full(sig.shape, np.nan)
            tau_lme = np.full(sig.shape, np.nan)
            sig_tdf = sig_sess = np.zeros(sig.shape, bool)
        else:
            sig, p_lme, sig_tdf, n_tr, tau_lme = lme_by_tag[tag]
            sig_sess = H.sig_mask(signed_sess[:, :, keep_pts, :], mask)
        sig_report[tag] = {h: int(sig[:, :, hi].sum()) for hi, h in enumerate(harms)}
        sig_report_tdf[tag] = {h: int(sig_tdf[:, :, hi].sum()) for hi, h in enumerate(harms)}
        sig_report_sess[tag] = {h: int(sig_sess[:, :, hi].sum()) for hi, h in enumerate(harms)}
        n_trials_report[tag] = n_tr
        ns = int(mask.sum())
        nrow_meta, c_by_k = {}, {}
        for hi, h in enumerate(harms):
            mc = float(np.nanmean(np.abs(gc[:, keep_pts, hi])))
            nrow_meta[h], c_by_k[h] = (h, ns, 0, mc), mc
        c_report[tag] = c_by_k

        if PANEL_MODE:
            _tgt = {"P1": PANEL_DIR / "S6a_patient_harmonic_P1.png",
                    "P2": PANEL_DIR / "S6b_patient_harmonic_P2.png"}
            if tag in _tgt:
                PANEL_DIR.mkdir(parents=True, exist_ok=True)
                wrap = (wrap_maxdiff is not None)
                gsig_w = np.concatenate([gsigned, gsigned[:, :1, :]], axis=1) if wrap else gsigned
                sig_w = np.concatenate([sig, sig[:, :1, :]], axis=1) if wrap else sig
                fracs_w = list(fracs_plot) + [1.0] if wrap else list(fracs_plot)
                render_si_panel(gsig_w, channels, fracs_w, harms, clim, info, ch_index,
                                nrow_meta, _tgt[tag], row_labels, sig_w)
                if tag == "P1":
                    _si_cycle_strip(fracs_w, PANEL_DIR / "S6_gait_cycle_strip.png")
        else:
            H.render_group(gsigned, channels, fracs_plot, harms, clim,
                           disp, cmp_label, info, ch_index, nrow_meta, png, row_labels=row_labels,
                           sig=sig)
            H.render_group(gsigned, channels, fracs_plot, harms, clim, disp, cmp_label,
                           info, ch_index, nrow_meta, SI_DIR / f"sup_fig6_patient_harmonic_allk_{tag}.png",
                           row_labels=row_labels, sig=sig)
            print(f"wrote {png.name} (k1/k2/k3) + SI all-k (redundant copy)  (n={ns} sess, {n_tr} trials)  "
                  + "  ".join(f"k{h} |c|={c_by_k[h]:.4f}" for h in harms)
                  + "  LME-z sig: " + "/".join(str(sig_report[tag][h]) for h in harms)
                  + "  [LME t-df " + "/".join(str(sig_report_tdf[tag][h]) for h in harms)
                  + " | session-t " + "/".join(str(sig_report_sess[tag][h]) for h in harms) + "]")
        if tag in ("P1", "P2"):
            combined_blocks[tag] = gsigned
            combined_sigs[tag] = sig
        for hi, h in enumerate(harms):
            for pi, cf in enumerate(fracs_plot):
                for ci, ch in enumerate(channels):
                    src_rows.append((tag, h, f"{cf:.6f}", ch, f"{gsigned[ci, pi, hi]:.8f}",
                                     int(sig[ci, pi, hi]),
                                     "" if not np.isfinite(p_lme[ci, pi, hi]) else f"{p_lme[ci, pi, hi]:.4e}",
                                     "" if not np.isfinite(tau_lme[ci, pi, hi]) else f"{tau_lme[ci, pi, hi]:.4e}"))
            meta_rows.append((f"{tag}_mean_abs_k{h}", f"{c_by_k[h]:.6f}"))
            meta_rows.append((f"{tag}_n_sig_bh_k{h}",
                              f"{sig_report[tag][h]} of {sig.shape[0] * sig.shape[1]} channel x cycle-point cells "
                              f"(DRAWN: trial-level LME, Wald z, BH-FDR per harmonic row)"))
            meta_rows.append((f"{tag}_n_sig_comparator_lme_tdf_k{h}",
                              f"{sig_report_tdf[tag][h]} (same LME, Wald statistic on t with n_sessions-1 df; NOT drawn)"))
            meta_rows.append((f"{tag}_n_sig_comparator_session_t_k{h}",
                              f"{sig_report_sess[tag][h]} (Fig.3b test: one-sample t across sessions; NOT drawn)"))
        meta_rows.append((f"{tag}_n_trials", n_trials_report[tag]))
        meta_rows.append((f"{tag}_n_sessions", ns))
        meta_rows.append((f"{tag}_sessions", ";".join(sess_list)))
        if tag in ("P1", "P2"):
            patient_pngs.append(png)
            patient_labels.append(f"{disp}  ({cmp_label})")

    if not PANEL_MODE:
        combined = OUT_PNG.with_name("fig6_patient_harmonic_phase_P1P2.png")
        stack_patients(patient_pngs, patient_labels, combined)
        print(f"wrote {combined}")

    panel_a = OUT_PNG.with_name("fig6_patient_harmonic_phase_panel.png")
    if {"P1", "P2"} <= set(combined_blocks) and not PANEL_MODE:
        wrap = (wrap_maxdiff is not None)
        blocks_a, sigs_a, fracs_a = [], [], list(fracs_plot)
        for tag in ("P1", "P2"):
            g, sg = combined_blocks[tag], combined_sigs[tag]
            blocks_a.append(np.concatenate([g, g[:, :1, :]], axis=1) if wrap else g)
            sigs_a.append(np.concatenate([sg, sg[:, :1, :]], axis=1) if wrap else sg)
        if wrap:
            fracs_a = fracs_a + [1.0]
        H.render_blocks(blocks_a, channels, fracs_a, harms, clim, info, ch_index, panel_a,
                        block_labels=["Patient 1", "Patient 2"],
                        row_labels=row_labels, with_cycle=True, sigs=sigs_a,
                        font_scale=PANEL_A_FONT_SCALE)
        print(f"wrote {panel_a.name}  (panel a: 2 blocks x {len(harms)} harmonics x "
              f"{len(fracs_a)} cycle points, one shared colorbar)")

    if PANEL_MODE:
        n_cells = len(channels) * len(fracs_plot)
        for tag, lab, letter in (("P1", "Patient 1", "a"), ("P2", "Patient 2", "b")):
            lines = [f"{lab}, gait-cycle harmonic phase locking "
                     f"({len(dict(P1=P1_SESSIONS, P2=P2_SESSIONS)[tag])} post-operative "
                     f"sessions, {n_trials_report[tag]} trials):"]
            for h in harms:
                lines.append(f"  k{h}: mean |c| = {c_report[tag][h]:.4f}; "
                             f"{sig_report[tag][h]} of {n_cells} electrode x cycle-point cells "
                             f"differ from zero")
            lines.append("Cells are tested with a trial-level mixed model (session as a random "
                         "intercept) and corrected within each harmonic row by the "
                         "Benjamini-Hochberg procedure at 0.05; a black dot marks such a cell.")
            lines.append("All maps share one reference phase and one colour limit, taken from "
                         "the pooled 11-session post-operative mean, so a weaker patient or "
                         "harmonic washes out instead of being stretched.")
            lines.append("Scope: 2 participants, case level, descriptive; channel space, not "
                         "source localisation.")
            export_stats(lines, "S6", letter, f"patient_harmonic_{tag}", width=6.4)
            (PANEL_DIR / f"S6{letter}_patient_harmonic_{tag}_stats.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            print("wrote", PANEL_DIR / f"S6{letter}_patient_harmonic_{tag}_stats.txt")

    if PANEL_MODE:
        print("panel mode: Source Data and metadata left untouched")
        return

    from safe_write import safe_write_csv
    safe_write_csv(SRC, ["patient", "harmonic_k", "cycle_fraction", "channel", "signed_phase_modulation",
                         "sig_bh", "p_lme_wald_z", "tau2_over_scale"], src_rows)
    meta_out = []
    for k, v in [
            ("figure", "Fig. 6 panel a - patient harmonic gait-phase scalp maps, split by patient; "
                       "MAIN = ALL harmonics (rows k1/k2/k3 x 8 gait-cycle phase points, "
                       "one PNG per patient)"),
            ("harmonic_dump", DUMP_TAG),
            ("readout", "patient-trained 10-fold CV (representation transfer, not zero-shot)"),
            ("scope", "n=2 SCI patients, case-level, descriptive; 11 post-operative sessions "
                      "drawn per patient (P1=6 sess, P2=5 sess) + pooled"),
            ("post_operative_set", "pat01 sess04-09 + pat02 sess03/05/06/08/09 (11 sessions) - PLOTTED"),
            ("pre_operative_set", "pat01 sess01/02 + pat02 sess01/02 (4 sessions) - present in the dump, NOT plotted here"),
            ("cohort_selection", "EXPLICIT session lists, not an id-prefix match"),
            ("excluded_not_plotted", "; ".join(dropped)),
            ("n_sessions_total", n_sess),
            ("sessions", ";".join(POST_OPERATIVE)),
            ("split", "P1=pat01 sess04-09; P2=pat02 sess03/05/06/08/09; P_all=pooled post_operative"),
            ("harmonics", ";".join(str(h) for h in harms)),
            ("main_panel_harmonics", ";".join(str(h) for h in harms)),
            ("main_panel_layout", f"per patient: {len(harms)} rows (k1/k2/k3) x {len(fracs_plot)} "
                                  "gait-cycle phase columns; row labels drawn, no panel letter, no title"),
            ("si_harmonics", ";".join(str(h) for h in harms)),
            ("colour_scale", "ONE reference phase + ONE symmetric colour limit shared across BOTH "
                             "patients AND all three harmonics, computed from the pooled post_operative group "
                             "mean over channels x cycle points x harmonics. Not per-row, not per-patient: "
                             "a weak patient and a weak harmonic must visibly wash out rather than be "
                             "stretched to look as strong as a strong one."),
            ("trial_dump", TRIAL_DUMP_TAG),
            ("shared_reference_phase_and_clim_across_main_and_si", "true (both rendered in one run)"),
            ("n_channels", len(ch_index)),
            ("n_cycle_points", len(fracs_plot)),
            ("cycle_points_plotted", ";".join(f"{c:.3f}" for c in fracs_plot)),
            ("wrap_column_removed", "cycle_fraction 1.000 is the wrap of 0.000; verified in this run "
                                    f"max|diff| = {wrap_maxdiff:.3e} over "
                                    f"{A.shape[0] * A.shape[1] * A.shape[3]} session x channel x harmonic cells; "
                                    f"the Source Data and the statistics hold the {len(fracs_plot)} unique points; "
                                    f"panel a (fig6_patient_harmonic_phase_panel.png) draws "
                                    f"{len(fracs_plot) + (wrap_maxdiff is not None)} columns, the last (1.000) "
                                    "redrawing the 0.000 map and its significance mask"),
            ("reference_phase_rad", f"{ref_phase:.6f}"),
            ("shared_clim_abs", f"{clim_abs:.6f}"),
            ("montage_source", "electrode positions read from the project EEGLAB donor .set "
                               f"({H.EEG_SET.relative_to(ROOT).as_posix()}), sphere='eeglab'; "
                               "NOT mne make_standard_montage"),
            ("claim", "channel-space diagnostic, not source localization: gait-cycle-organised phase "
                      "structure is present in EACH patient's imagined-walking EEG; no claim about "
                      "cortical generators or causal neural contribution"),
            ("output_png_main", "figures_out/fig6/fig6_patient_harmonic_phase{,_P1,_P2}.png"),
    ]:
        meta_out.append([k, v])
    meta_out.extend(meta_rows)
    safe_write_csv(META, ["key", "value"], meta_out)

    print(f"post_operative sessions ({n_sess}) = {POST_OPERATIVE}")
    print(f"dropped from dump ({len(dropped)}) = {dropped}")
    print(f"ref_phase={ref_phase:.4f} clim=+/-{clim_abs:.6f} wrap_maxdiff={wrap_maxdiff:.3e}")
    for tag in ("P1", "P2", "P_all"):
        print(f"  {tag}: " + "  ".join(f"k{h} |c|={c_report[tag][h]:.4f}" for h in harms))
    print(f"wrote {SRC}")
    print(f"wrote {META}")

if __name__ == "__main__":
    main()
