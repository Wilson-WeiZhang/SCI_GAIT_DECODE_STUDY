#!/usr/bin/env python3
"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

Draw the Fig. 2 goniometer power-spectrum panel

Reads:
  preprocessing output in $GAIT_PREP_OUTPUT (not distributed)
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""


from __future__ import annotations
PREP_OUTPUT = __import__("pathlib").Path(__import__("os").environ.get("GAIT_PREP_OUTPUT", "preprocessing_output"))
_release_rel = lambda p, r: p.relative_to(r) if p.is_relative_to(r) else p

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from project_palette import ANKLE, HIP, KNEE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PREP_OUTPUT / "healthy" / "note" / "session_manifest.csv"
DEFAULT_QC_FLAGS: Path | None = None
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures_out" / "fig2"
SD_DIR = PROJECT_ROOT / "04_source_data" / "fig2"
DEFAULT_BASENAME = "fig2a_goni_power"
JOINTS = ("hip", "knee", "ankle")
JOINT_STEMS = {"hip": "Hip", "knee": "Kne", "ankle": "Ank"}
COLORS = {"hip": HIP, "knee": KNEE, "ankle": ANKLE}
PEAK_BAND = (0.45, 1.15)

def matlab_char(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset).squeeze()
    return "".join(chr(int(x)) for x in arr.ravel() if int(x))


def default_data_roots() -> list[Path]:
    roots = []
    env_root = os.environ.get("GAIT_TRIPLE_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path.home() / "gait" / "data" / "healthy_prep")
    return roots


def resolve_data_root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    for root in default_data_roots():
        if root.exists():
            return root
    tried = "\n".join(str(root) for root in default_data_roots())
    raise FileNotFoundError(f"No healthy export root found. Tried:\n{tried}")


def read_manifest(manifest: Path, max_sessions: int | None) -> list[str]:
    if not manifest.exists():
        raise SystemExit(
            f"\n[fig2a_goni_power] SESSION MANIFEST NOT FOUND:\n  {manifest}\n\n"
            "The healthy session manifest is session_manifest.csv (not distributed)\n"
            "(80 sessions, 42 participants). Nothing was written.\n"
        )
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sessions = [row["session"].strip() for row in reader if row.get("session", "").strip()]
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    if not sessions:
        raise ValueError(f"No sessions found in {manifest}")
    return sessions


def read_qc_status(qc_flags: Path | None, condition: str) -> dict[tuple[str, int], str]:
    if qc_flags is None or not qc_flags.exists():
        return {}
    status_by_trial: dict[tuple[str, int], str] = {}
    with qc_flags.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type", "").lower() != condition.lower():
                continue
            try:
                trial = int(row["trial"])
            except (KeyError, ValueError):
                continue
            status_by_trial[(row["label"], trial)] = row.get("status", "")
    return status_by_trial


def load_goni(goni_path: Path) -> tuple[np.ndarray, float, list[str]]:
    with h5py.File(goni_path, "r") as h5:
        x = np.asarray(h5["goni/x"], dtype=float)
        srate = float(np.asarray(h5["goni/s"]).squeeze())
        labels = [matlab_char(h5[ref]) for ref in np.asarray(h5["goni/c"]).ravel()]
    if x.shape[0] == len(labels) and x.shape[1] != len(labels):
        x = x.T
    if x.shape[1] != len(labels):
        raise ValueError(f"goni.x shape {x.shape} does not match {len(labels)} labels in {goni_path}")
    return x, srate, labels


def load_events(event_path: Path) -> list[object]:
    events = loadmat(event_path, squeeze_me=True, struct_as_record=False)["event"]
    return list(np.atleast_1d(events))


def as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).ravel())
        if value.size == 1:
            return as_text(value.item())
    return str(value)


def event_code(event: object) -> str:
    return as_text(getattr(event, "code", "")).lower()


def event_qc(event: object) -> str:
    return as_text(getattr(event, "qc", "good")).lower()


def event_slice(event: object, n_time: int) -> slice | None:
    start = int(getattr(event, "gtstart")) - 1
    stop = int(getattr(event, "gtstop"))
    start = max(start, 0)
    stop = min(stop, n_time)
    if stop <= start:
        return None
    return slice(start, stop)


def labels_for_joint(joint: str, role: str, axis: str) -> list[str]:
    stem = JOINT_STEMS[joint]
    return [f"L{stem}{role} {axis}", f"R{stem}{role} {axis}"]


def channel_index(labels: list[str], wanted: str) -> int | None:
    try:
        return labels.index(wanted)
    except ValueError:
        return None


def normalized_psd(y: np.ndarray, fs: float, freq_grid: np.ndarray, fmax: float) -> np.ndarray | None:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < round(2 * fs):
        return None
    if np.nanstd(y) <= 0:
        return None
    y = y - np.nanmean(y)
    win = min(round(4 * fs), max(round(2 * fs), y.size // 2))
    freqs, pxx = welch(
        y,
        fs=fs,
        window="hamming",
        nperseg=int(win),
        noverlap=int(win // 2),
        nfft=8192,
        detrend=False,
        scaling="density",
    )
    keep = (freqs >= 0.05) & (freqs <= fmax)
    denom = np.trapezoid(pxx[keep], freqs[keep])
    if not np.isfinite(denom) or denom <= 0:
        return None
    return np.interp(freq_grid, freqs, pxx / denom, left=0.0, right=0.0)


def collect_session_psd(
    session: str,
    data_root: Path,
    condition: str,
    role: str,
    axis: str,
    qc_status: dict[tuple[str, int], str],
    freq_grid: np.ndarray,
    fmax: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], float]:
    base = data_root / session
    goni_x, fs, labels = load_goni(base / f"{session}_goni.mat")
    events = load_events(base / f"{session}_event.mat")
    traces: dict[str, list[np.ndarray]] = {joint: [] for joint in JOINTS}

    for trial_idx, event in enumerate(events, start=1):
        if event_code(event) != condition.lower():
            continue
        qc = qc_status.get((session, trial_idx), event_qc(event))
        if qc not in {"good", "ok"}:
            continue
        sl = event_slice(event, goni_x.shape[0])
        if sl is None:
            continue
        for joint in JOINTS:
            for label in labels_for_joint(joint, role, axis):
                ch = channel_index(labels, label)
                if ch is None:
                    continue
                psd = normalized_psd(goni_x[sl, ch], fs, freq_grid, fmax)
                if psd is not None:
                    traces[joint].append(psd)

    session_psd = {}
    n_traces = {}
    for joint, values in traces.items():
        n_traces[joint] = len(values)
        if values:
            session_psd[joint] = np.mean(np.vstack(values), axis=0)
    return session_psd, n_traces, fs


def summarize_group(session_curves: dict[str, list[np.ndarray]]) -> dict[str, dict[str, np.ndarray | float]]:
    summary = {}
    for joint, curves in session_curves.items():
        arr = np.vstack(curves)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        ci95 = 1.96 * sem
        summary[joint] = {"mean": mean, "sem": sem, "ci95": ci95, "n_sessions": float(arr.shape[0])}
    return summary


def compute_peak_hz(summary: dict[str, dict[str, np.ndarray | float]], freq_grid: np.ndarray) -> dict[str, float]:
    peak_mask = (freq_grid >= PEAK_BAND[0]) & (freq_grid <= PEAK_BAND[1])
    peak_hz = {}
    for joint in JOINTS:
        mean = summary[joint]["mean"]
        peak_hz[joint] = float(freq_grid[peak_mask][np.argmax(mean[peak_mask])])
    return peak_hz


def write_summary_csv(
    csv_path: Path,
    freq_grid: np.ndarray,
    summary: dict[str, dict[str, np.ndarray | float]],
    trace_counts: dict[str, int],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "freq_hz",
                "hip_mean",
                "hip_sem",
                "hip_ci95_low",
                "hip_ci95_high",
                "knee_mean",
                "knee_sem",
                "knee_ci95_low",
                "knee_ci95_high",
                "ankle_mean",
                "ankle_sem",
                "ankle_ci95_low",
                "ankle_ci95_high",
                "hip_n_sessions",
                "knee_n_sessions",
                "ankle_n_sessions",
                "hip_n_traces",
                "knee_n_traces",
                "ankle_n_traces",
            ]
        )
        for i, freq in enumerate(freq_grid):
            row = [f"{freq:.5f}"]
            for joint in JOINTS:
                mean = summary[joint]["mean"][i]
                ci95 = summary[joint]["ci95"][i]
                row.append(f"{mean:.10g}")
                row.append(f"{summary[joint]['sem'][i]:.10g}")
                row.append(f"{mean - ci95:.10g}")
                row.append(f"{mean + ci95:.10g}")
            for joint in JOINTS:
                row.append(f"{int(summary[joint]['n_sessions'])}")
            for joint in JOINTS:
                row.append(f"{trace_counts[joint]}")
            writer.writerow(row)


def read_summary_csv(csv_path: Path) -> tuple[np.ndarray, dict, dict, int]:
    """Re-load the panel's own Source Data so the PNG can be re-rendered without the raw prep."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty Source Data: {csv_path}")
    freq_grid = np.array([float(r["freq_hz"]) for r in rows], dtype=float)
    summary: dict[str, dict[str, np.ndarray | float]] = {}
    for joint in JOINTS:
        mean = np.array([float(r[f"{joint}_mean"]) for r in rows], dtype=float)
        sem = np.array([float(r[f"{joint}_sem"]) for r in rows], dtype=float)
        hi = np.array([float(r[f"{joint}_ci95_high"]) for r in rows], dtype=float)
        summary[joint] = {
            "mean": mean,
            "sem": sem,
            "ci95": hi - mean,
            "n_sessions": int(float(rows[0][f"{joint}_n_sessions"])),
        }
    trace_counts = {joint: int(float(rows[0][f"{joint}_n_traces"])) for joint in JOINTS}
    n_sessions = int(float(rows[0]["hip_n_sessions"]))
    return freq_grid, summary, trace_counts, n_sessions


def draw_figure(
    freq_grid: np.ndarray,
    summary: dict[str, dict[str, np.ndarray | float]],
    output_dir: Path,
    basename: str,
    condition: str,
    role: str,
    axis: str,
    font_scale: float = 1.0,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10 * font_scale,
            "axes.linewidth": 1.7,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
        }
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    peak_hz = compute_peak_hz(summary, freq_grid)
    mean_peak_hz = float(np.mean([peak_hz[joint] for joint in JOINTS]))
    for harmonic in (1, 2, 3):
        x = harmonic * mean_peak_hz
        if x <= float(freq_grid[-1]):
            ax.axvline(x, color="black", linestyle=(0, (3, 2)), linewidth=0.8, alpha=0.75, zorder=2)

    for joint in JOINTS:
        mean = summary[joint]["mean"]
        ci95 = summary[joint]["ci95"]
        color = COLORS[joint]
        ax.fill_between(freq_grid, mean - ci95, mean + ci95, color=color, alpha=0.20, linewidth=0, zorder=1)
        ax.plot(freq_grid, mean, color=color, linewidth=2.2, label=joint.capitalize(), zorder=3)

    ax.set_xlim(0, float(freq_grid[-1]))
    ymax = max(float(np.nanmax(summary[joint]["mean"] + summary[joint]["ci95"])) for joint in JOINTS)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10 * font_scale, length=3.5, width=1.7)

    fig.tight_layout(pad=0.45)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qc-flags", type=Path, default=DEFAULT_QC_FLAGS)
    parser.add_argument("--condition", default="gaitimagery")
    parser.add_argument("--role", choices=("W", "S"), default="W")
    parser.add_argument("--axis", choices=("X", "Y"), default="X")
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--freq-step", type=float, default=0.01)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--font-scales", default=None,
                        help="comma-separated scales -> one PNG each")
    parser.add_argument("--from-source-data", nargs="?", const="auto", default=None,
                        help="re-render the PNG from this panel's own Source Data CSV instead of "
                             "the raw prep exports (style-only changes; omit the value to use "
                             "04_source_data/<basename>_summary.csv). Does not rewrite Source Data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_source_data is not None:
        csv_in = (SD_DIR / f"{args.basename}_summary.csv"
                  if args.from_source_data == "auto" else Path(args.from_source_data))
        freq_grid, summary, trace_counts, n_sessions = read_summary_csv(csv_in)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scales = [float(v) for v in args.font_scales.split(",")] if args.font_scales else [args.font_scale]
        for _fs in scales:
            _bn = args.basename if (len(scales) == 1 and _fs == 1.0) else f"{args.basename}_fs{int(round(_fs * 100))}"
            print(f"PNG fs={_fs}: "
                  f"{draw_figure(freq_grid, summary, args.output_dir, _bn, args.condition, args.role, args.axis, _fs)}")
        peak_hz = compute_peak_hz(summary, freq_grid)
        mean_peak_hz = float(np.mean([peak_hz[joint] for joint in JOINTS]))
        print(f"Source Data: {csv_in}")
        print(f"Sessions in Source Data: {n_sessions}")
        print("Trace counts: " + ", ".join(f"{joint}={trace_counts[joint]}" for joint in JOINTS))
        print("Peak 0.45-1.15 Hz: " + ", ".join(f"{joint}={peak_hz[joint]:.2f}Hz" for joint in JOINTS))
        print("Harmonics: " + ", ".join(f"{k}x={k * mean_peak_hz:.2f}Hz" for k in (1, 2, 3)))
        print("Source Data NOT rewritten (style-only re-render).")
        return

    data_root = resolve_data_root(args.data_root)
    sessions = read_manifest(args.manifest, args.max_sessions)
    qc_status = read_qc_status(args.qc_flags, args.condition)
    freq_grid = np.arange(0.0, args.fmax + args.freq_step / 2, args.freq_step)
    session_curves: dict[str, list[np.ndarray]] = {joint: [] for joint in JOINTS}
    trace_counts: dict[str, int] = defaultdict(int)
    fs_values = []

    for i, session in enumerate(sessions, start=1):
        curves, counts, fs = collect_session_psd(
            session,
            data_root,
            args.condition,
            args.role,
            args.axis,
            qc_status,
            freq_grid,
            args.fmax,
        )
        fs_values.append(fs)
        for joint in JOINTS:
            trace_counts[joint] += counts[joint]
            if joint in curves:
                session_curves[joint].append(curves[joint])
        print(
            f"[{i:02d}/{len(sessions):02d}] {session}: "
            f"hip={counts['hip']} knee={counts['knee']} ankle={counts['ankle']}"
        )

    missing = [joint for joint, curves in session_curves.items() if not curves]
    if missing:
        raise RuntimeError(f"No valid PSD curves for: {', '.join(missing)}")

    summary = summarize_group(session_curves)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    SD_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SD_DIR / f"{args.basename}_summary.csv"
    write_summary_csv(csv_path, freq_grid, summary, trace_counts)
    scales = [float(s) for s in args.font_scales.split(",")] if args.font_scales else [args.font_scale]
    png_path = None
    for _fs in scales:
        _bn = args.basename if (len(scales) == 1 and _fs == 1.0) else f"{args.basename}_fs{int(round(_fs * 100))}"
        png_path = draw_figure(freq_grid, summary, args.output_dir, _bn, args.condition, args.role, args.axis, _fs)
        print(f"PNG fs={_fs}: {png_path}")

    peak_text = []
    peak_hz = compute_peak_hz(summary, freq_grid)
    mean_peak_hz = float(np.mean([peak_hz[joint] for joint in JOINTS]))
    for joint in JOINTS:
        peak_text.append(f"{joint}={peak_hz[joint]:.2f}Hz")
    harmonic_text = ", ".join(f"{k}x={k * mean_peak_hz:.2f}Hz" for k in (1, 2, 3))

    print(f"Data root: {data_root}")
    print(f"Sessions processed: {len(sessions)}")
    print(f"Sampling rates: {sorted(set(fs_values))}")
    print("Trace counts: " + ", ".join(f"{joint}={trace_counts[joint]}" for joint in JOINTS))
    print("Peak 0.45-1.15 Hz: " + ", ".join(peak_text))
    print(f"Mean peak harmonics: {harmonic_text}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
