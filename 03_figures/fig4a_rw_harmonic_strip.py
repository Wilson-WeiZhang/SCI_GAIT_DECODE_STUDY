"""Figure 4 — Shared locomotor representation across imagined and real walking

real-walking (EXEC) gait-cycle harmonic phase strip

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "data" / "decoder_output"
EEG_SET = ROOT / "data/prep_data/healthy/eeg/SUB_37_sess02_step2.set"
FIGDIR = ROOT / "figures_out" / "fig4"
SDDIR = ROOT / "04_source_data" / "fig4"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_config as ec
BUILD_ID = ec.BUILD_IDS[ec.NEURAL_BUILD_ID]["gait_build_id"]
EXEC_DIR = RESULT / "diag_real_walking_cohort"
EXEC_CSV = EXEC_DIR / "cohort_alltrial_harmonic_phase_cycle9_complex_session_equal.csv"
COUNTS_CSV = EXEC_DIR / "cohort_alltrial_harmonic_phase_cycle9_session_counts.csv"
PERSESSION_DIR = RESULT / "diag_real_walking_persession"
PERSESSION_CSV = PERSESSION_DIR / "persession_harmonic_phase_cycle9_complex.csv"
EXPECT_SESSIONS = 80

MASK_PARAMS = dict(marker="o", markerfacecolor="k", markeredgecolor="white",
                   markersize=4.2, markeredgewidth=0.6)


def assert_cohort() -> int:
    """Cohort identity is read off the session roster."""
    if not COUNTS_CSV.exists():
        raise SystemExit(f"missing session-count file, cannot verify cohort: {COUNTS_CSV}")
    n = pd.read_csv(COUNTS_CSV)["session"].nunique()
    if n != EXPECT_SESSIONS:
        raise SystemExit(
            f"cohort mismatch: {COUNTS_CSV.parent.name} has {n} sessions, expected {EXPECT_SESSIONS}. "
            "Refusing to emit a mixed-cohort Fig.4 panel."
        )
    return n


def _bh(p, alpha=0.05):
    """Benjamini-Hochberg step-up. Same implementation as 03_figures/fig3_harmonic_phase_responder_split.py."""
    p = np.asarray(p, float); out = np.zeros(p.shape, bool); ok = np.isfinite(p)
    pv = p[ok]; m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv); passed = pv[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any():
        keep = np.zeros(m, bool); keep[order[:np.where(passed)[0].max() + 1]] = True
        o = out[ok]; o[:] = keep; out[ok] = o
    return out


def sig_mask(signed_sess):
    """([ch, point, h] bool, [ch, point, h] p): BH-FDR (per harmonic) cells where signed locking != 0."""
    mu = np.nanmean(signed_sess, 0); sd = np.nanstd(signed_sess, 0, ddof=1)
    n = np.sum(np.isfinite(signed_sess), 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tval = mu / (sd / np.sqrt(n))
    p = 2 * tdist.sf(np.abs(tval), np.clip(n - 1, 1, None))
    sig = np.zeros(p.shape, bool)
    for ri in range(p.shape[-1]):
        sig[..., ri] = _bh(p[..., ri].ravel()).reshape(p[..., ri].shape)
    return sig, p


def load_persession(csv_path: Path, channels, fracs, harms):
    """[session, ch, point, h] complex, on the SAME axis order as the cohort map."""
    df = pd.read_csv(csv_path)
    sessions = sorted(pd.unique(df["session"]))
    ci = {c: i for i, c in enumerate(channels)}
    fi = {round(v, 6): i for i, v in enumerate(fracs)}
    hi = {h: i for i, h in enumerate(harms)}
    si = {ss: i for i, ss in enumerate(sessions)}
    S = np.full((len(sessions), len(channels), len(fracs), len(harms)), np.nan, complex)
    for r in df.itertuples():
        S[si[r.session], ci[r.channel], fi[round(float(r.cycle_fraction), 6)], hi[int(r.harmonic_k)]] = complex(r.real, r.imag)
    return sessions, S


def eeglab_classic_redblue_cmap(n: int = 256):
    """Same as 03_figures/fig3_harmonic_phase_responder_split.py for identical look."""
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


def load_cohort(csv_path: Path):
    df = pd.read_csv(csv_path)
    channels = list(pd.unique(df["channel"]))
    fracs = sorted(df["cycle_fraction"].unique())
    harms = sorted(df["harmonic_k"].unique())
    ci = {c: i for i, c in enumerate(channels)}
    fi = {round(v, 6): i for i, v in enumerate(fracs)}
    hi = {h: i for i, h in enumerate(harms)}
    A = np.full((len(channels), len(fracs), len(harms)), np.nan, complex)
    for r in df.itertuples():
        A[ci[r.channel], fi[round(float(r.cycle_fraction), 6)], hi[int(r.harmonic_k)]] = complex(r.real, r.imag)
    return channels, fracs, harms, A


def main():
    import matplotlib
    matplotlib.use("Agg")
    import mne
    if not EXEC_CSV.exists():
        print(f"real-walking complex map missing -> {EXEC_CSV}")
        sys.exit(0)
    n_sessions = assert_cohort()
    print(f"build_id={BUILD_ID}  input={EXEC_DIR.name}  sessions={n_sessions}")
    FIGDIR.mkdir(parents=True, exist_ok=True); SDDIR.mkdir(parents=True, exist_ok=True)

    channels, fracs, harms, A = load_cohort(EXEC_CSV)
    flat = A.ravel()
    k = np.nanargmax(np.abs(flat))
    ref_phase = float(np.angle(flat[k]))
    signed = np.real(A * np.exp(-1j * ref_phase))
    clim_abs = float(np.nanquantile(np.abs(signed), 0.98))
    clim = (-clim_abs, clim_abs)
    ki, pi, hi = np.unravel_index(k, A.shape)
    print(f"RW unified ref: k{harms[hi]} {channels[ki]} @ {100*fracs[pi]:.0f}%  phase={ref_phase:+.4f}  clim=+/-{clim_abs:.5f}")

    if not PERSESSION_CSV.exists():
        raise SystemExit(f"missing per-session dump, cannot mark significant channels: {PERSESSION_CSV}")
    sessions, S = load_persession(PERSESSION_CSV, channels, fracs, harms)
    if len(sessions) != EXPECT_SESSIONS:
        raise SystemExit(f"per-session dump has {len(sessions)} sessions, expected {EXPECT_SESSIONS}: {PERSESSION_CSV}")
    dev = float(np.nanmax(np.abs(np.nanmean(S, 0) - A)))
    if dev > 1e-6:
        raise SystemExit(f"per-session dump does not reproduce the cohort map (max dev {dev:.3e}); "
                         "refusing to mark significance from a different build.")
    sig, pvals = sig_mask(np.real(S * np.exp(-1j * ref_phase)))
    n_cells = int(sig[..., 0].size)
    n_sig = {int(h): int(sig[..., ri].sum()) for ri, h in enumerate(harms)}
    print(f"sig dots (one-sample t vs 0 across {len(sessions)} sessions, BH-FDR per harmonic row, "
          f"{n_cells} cells/row): " +
          "  ".join(f"k{int(h)}={n_sig[int(h)]} ({100*n_sig[int(h)]/n_cells:.1f}%)" for h in harms))

    raw = mne.io.read_raw_eeglab(str(EEG_SET), preload=False, verbose="ERROR")
    keep = [c for c in raw.ch_names if c in channels]
    info = raw.copy().pick(keep).info
    ch_index = [channels.index(c) for c in keep]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fig3_harmonic_phase_responder_split import render_group

    n_h, n_p = len(harms), len(fracs)
    out_png = ec.scoped(FIGDIR / "fig4a_rw_harmonic_strip.png")
    render_group(signed, channels, fracs, harms, clim,
                 gname="RW", cmp_label="real walking", info=info, ch_index=ch_index,
                 nrow_meta=None, out_png=out_png, with_cycle=True,
                 cycle_label="Gait cycle reference", sig=sig,
                 row_labels=[r"$\mathrm{k}_%d$" % int(h) for h in harms])

    rows = [dict(harmonic_k=int(h), cycle_percent=100 * cf, channel=ch, signed_value=signed[ci_, pi_, ri],
                 p_value=pvals[ci_, pi_, ri], significant_bh_fdr=bool(sig[ci_, pi_, ri]))
            for ri, h in enumerate(harms) for pi_, cf in enumerate(fracs) for ci_, ch in enumerate(channels)]
    pd.DataFrame(rows).to_csv(ec.scoped(SDDIR / "fig4a_rw_harmonic_strip_source_data.csv"), index=False)

    meta = [
        ("figure", "Fig. 4 panel a - real-walking (EXEC) gait-cycle harmonic phase strip"),
        ("panel_letter", "a"),
        ("n_sessions", n_sessions),
        ("exec_source", EXEC_DIR.name),
        ("chanlocs_source", EEG_SET.name),
        ("n_channels_plotted", len(keep)),
        ("unified_ref_phase_rad", f"{ref_phase:.6f}"),
        ("clim_abs", f"{clim_abs:.6f}"),
        ("aggregation", "session-equal complex average; |c| shown per harmonic is the mean magnitude of that complex average"),
        ("persession_source", PERSESSION_DIR.name),
        ("sig_test", "two-sided one-sample t across sessions that the signed locking (projection on "
                     "unified_ref_phase_rad) != 0; per channel x cycle-point x harmonic"),
        ("sig_correction", "Benjamini-Hochberg FDR, alpha=0.05, within each harmonic row"),
        ("sig_n_sessions", len(sessions)),
        ("sig_n_tests_per_harmonic", n_cells),
        ("sig_marker", "filled black dot, white edge - same convention as the Fig. 3 harmonic-phase maps"),
        ("persession_vs_cohort_max_dev", f"{dev:.3e}"),
    ] + [(f"n_sig_k{int(h)}", n_sig[int(h)]) for h in harms] + [(f"mean_abs_c_k{h}", f"{float(np.nanmean(np.abs(A[:, :, ri]))):.6f}") for ri, h in enumerate(harms)]
    pd.DataFrame(meta, columns=["key", "value"]).to_csv(
        ec.scoped(SDDIR / "fig4a_rw_harmonic_strip_metadata.csv"), index=False)
    print(f"wrote {out_png}  channels={len(keep)}  rows={len(rows)}  build_id={BUILD_ID}  sessions={n_sessions}  "
          f"sig={sum(n_sig.values())}/{len(harms) * n_cells}")


if __name__ == "__main__":
    main()
