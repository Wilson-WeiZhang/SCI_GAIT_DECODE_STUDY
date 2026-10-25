#!/usr/bin/env python3
"""Figure 1 — Study design and reference-walker kinematics

Draw Fig. 1c: walker goniometer example traces

Reads:
  the aligned per-session recordings (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""


from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_LABEL = "subj30_sess01"
DATA_ROOT_CANDIDATES = [
    (__import__("pathlib").Path(__import__("os").environ.get("GAIT_DATA_ROOT", "")) / "healthy_prep"),
    (__import__("pathlib").Path(__import__("os").environ.get("GAIT_DATA_ROOT", "")) / "healthy_prep"),
    Path.home() / "gait" / "data" / "healthy_prep",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures_out" / "fig1"
SOURCE_DATA_DIR = PROJECT_ROOT / "04_source_data" / "fig1"
DEFAULT_BASENAME = "fig1c_goni_eeg_draw_example"


def resolve_session_dir(label: str) -> Path:
    for root in DATA_ROOT_CANDIDATES:
        cand = root / label
        if (cand / f"{label}_goni.mat").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find {label}_goni.mat under any known data root; pass "
        f"--session-dir explicitly. Tried: "
        + "; ".join(str(r / label) for r in DATA_ROOT_CANDIDATES)
    )


FONT_SIZE_SCALE = 2.70
FIGSIZE = (18.0, 7.6)
TRACE_COLOR = "#8f8f8f"
TRACE_ALPHA = 0.15
TRACE_LW = 1.0
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_palette import HIP as _HIP, KNEE as _KNEE, ANKLE as _ANKLE
JOINT_COLOR = {"hip": _HIP, "knee": _KNEE, "ankle": _ANKLE}
HIGHLIGHT_COLOR = "#1f1f1f"
HIGHLIGHT_LW = 2.4

WALKER_CHANNELS = [
    ("Left hip", "LHipW X"),
    ("Right hip", "RHipW X"),
    ("Left knee", "LKneW X"),
    ("Right knee", "RKneW X"),
    ("Left ankle", "LAnkW X"),
    ("Right ankle", "RAnkW X"),
]

DISPLAY_POLARITY = {"RHipW X": -1.0, "LKneW X": -1.0, "LAnkW X": -1.0}


def matlab_char(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset).squeeze()
    return "".join(chr(int(x)) for x in arr.ravel() if int(x))


def load_goni(goni_path: Path) -> tuple[np.ndarray, float, list[str]]:
    with h5py.File(goni_path, "r") as h5:
        x = np.asarray(h5["goni/x"], dtype=float)
        srate = float(np.asarray(h5["goni/s"]).squeeze())
        labels = [matlab_char(h5[ref]) for ref in np.asarray(h5["goni/c"]).ravel()]
    if x.shape[0] != len(labels) and x.shape[1] == len(labels):
        x = x.T
    return x, srate, labels


def load_events(event_path: Path) -> list[object]:
    events = loadmat(event_path, squeeze_me=True, struct_as_record=False)["event"]
    return list(np.atleast_1d(events))


def as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).ravel())
        if value.size == 1:
            return as_text(value.item())
    return str(value)


def event_code(event: object) -> str:
    return as_text(getattr(event, "code", "")).lower()


def event_slice(event: object, n_time: int) -> slice | None:
    start = int(getattr(event, "gtstart")) - 1
    stop = int(getattr(event, "gtstop"))
    start = max(start, 0)
    stop = min(stop, n_time)
    if stop <= start:
        return None
    return slice(start, stop)


def channel_index(labels: list[str], wanted: str) -> int:
    try:
        return labels.index(wanted)
    except ValueError as exc:
        available = ", ".join(labels)
        raise ValueError(f"Missing channel {wanted!r}; available: {available}") from exc


def collect_trial_segments(
    goni_x: np.ndarray,
    srate: float,
    events: list[object],
    labels: list[str],
) -> tuple[dict[str, list[tuple[int, np.ndarray, np.ndarray]]], int, float]:
    """Per joint, list of (trial_id, time, angle) for every imagined-walking trial."""
    mi_events = [event for event in events if event_code(event) == "gaitimagery"]
    segments: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    max_time = 0.0

    for joint_name, channel_name in WALKER_CHANNELS:
        ch = channel_index(labels, channel_name)
        joint_segments = []
        for trial_id, event in enumerate(mi_events, start=1):
            sl = event_slice(event, goni_x.shape[1])
            if sl is None:
                continue
            y = goni_x[ch, sl] * DISPLAY_POLARITY.get(channel_name, 1.0)
            if y.size == 0 or np.all(~np.isfinite(y)):
                continue
            t = np.arange(y.size, dtype=float) / srate
            max_time = max(max_time, float(t[-1]) if t.size else 0.0)
            joint_segments.append((trial_id, t, y))
        segments[joint_name] = joint_segments

    return segments, len(mi_events), max_time


def representative_trial_id(
    segments: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
) -> int | None:
    """Trial whose Left-knee peak-to-peak is closest to the median (typical)."""
    items = segments.get("Left knee") or []
    if not items:
        return None
    p2p = [(tid, float(np.nanmax(y) - np.nanmin(y))) for tid, _, y in items]
    med = float(np.median([v for _, v in p2p]))
    return min(p2p, key=lambda kv: abs(kv[1] - med))[0]


def draw_figure(
    segments: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
    max_time: float,
    highlight_id: int | None,
    output_dir: Path,
    basename: str,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8 * FONT_SIZE_SCALE,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True)
    row_pairs = [
        ("Left hip", "Right hip"),
        ("Left knee", "Right knee"),
        ("Left ankle", "Right ankle"),
    ]

    for row, pair in enumerate(row_pairs):
        row_values = []
        for joint_name in pair:
            for _tid, _t, y in segments[joint_name]:
                row_values.append(y[np.isfinite(y)])
        finite = np.concatenate(row_values) if row_values else np.array([])
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            pad = max((hi - lo) * 0.12, 1.0)
            ylim = (lo - pad, hi + pad)
        else:
            ylim = (-1.0, 1.0)

        for col, joint_name in enumerate(pair):
            ax = axes[row, col]
            for tid, t, y in segments[joint_name]:
                if tid == highlight_id:
                    continue
                ax.plot(t, y, color=TRACE_COLOR, linewidth=TRACE_LW, alpha=TRACE_ALPHA)
            joint_key = joint_name.split(" ", 1)[1].lower()
            hl_color = JOINT_COLOR.get(joint_key, HIGHLIGHT_COLOR)
            if highlight_id is not None:
                for tid, t, y in segments[joint_name]:
                    if tid == highlight_id:
                        ax.plot(t, y, color=hl_color, linewidth=HIGHLIGHT_LW, alpha=0.95)
            short = joint_name.split(" ", 1)[1].capitalize()
            ax.set_ylabel(f"{short} (deg)")
            ax.set_ylim(*ylim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7 * FONT_SIZE_SCALE, length=3.5, width=1.1)

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    legend_handles = [
        Line2D([0], [0], color=TRACE_COLOR, lw=2.6, alpha=0.6),
        tuple(Line2D([0], [0], color=c, lw=HIGHLIGHT_LW)
              for c in (_HIP, _KNEE, _ANKLE)),
    ]
    legend_labels = ["All trials (overlaid)", "Representative trial"]
    _leg_kw = dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})
    fig_leg = plt.figure(figsize=(2.4, 0.6))
    fig_leg.legend(
        legend_handles, legend_labels, loc="center", frameon=False,
        fontsize=7 * FONT_SIZE_SCALE, handlelength=1.6,
        labelspacing=0.4, handletextpad=0.6, borderpad=0.2, **_leg_kw,
    )
    legend_path = output_dir / f"{basename}_legend.png"
    fig_leg.savefig(legend_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig_leg)
    print(f"Legend (separate): {legend_path}")

    xlim_hi = np.ceil(max_time * 2) / 2 if max_time > 0 else 8.0
    for ax in axes[-1, :]:
        ax.set_xlabel("Time from trial onset (s)")
    for ax in axes.ravel():
        ax.set_xlim(0, xlim_hi)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.tight_layout(pad=0.9, w_pad=5.4, h_pad=1.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    fig.savefig(png_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png_path


def write_source_data(
    segments: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
    n_mi_events: int,
    highlight_id: int | None,
    srate: float,
    session_label: str,
    source_dir: Path,
    basename: str,
) -> tuple[Path, Path]:
    """Write the exact plotted traces + metadata as Fig. 1c Source Data."""
    source_dir.mkdir(parents=True, exist_ok=True)
    traces_path = source_dir / f"{basename}_source_data.csv"
    with traces_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint", "trial", "time_s", "angle_deg"])
        for joint_name, joint_segments in segments.items():
            for trial_id, t, y in joint_segments:
                for tt, yy in zip(t, y):
                    writer.writerow([joint_name, trial_id, f"{tt:.6f}", f"{yy:.6f}"])
    meta_path = source_dir / f"{basename}_metadata.csv"
    with meta_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerow(["figure", "Fig. 1 panel c"])
        writer.writerow(["build_id", "main"])
        writer.writerow(["build_id_note", "goniometer traces from the preprocessing "
                                     "export; no decoder output enters this panel"
                                     ""])
        writer.writerow(["session", session_label])
        writer.writerow(["n_imagined_walking_trials", n_mi_events])
        writer.writerow(["highlighted_trial", "" if highlight_id is None else highlight_id])
        writer.writerow(["goni_srate_hz", f"{srate:g}"])
        writer.writerow(["channels", ";".join(ch for _, ch in WALKER_CHANNELS)])
        writer.writerow(["display_polarity", "RHipW X,LKneW X,LAnkW X x(-1): L/R sign harmonization for plotting only; decoding z-scores (Y_delta), unaffected"])
    return traces_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir or resolve_session_dir(SESSION_LABEL)
    label = session_dir.name
    goni_path = session_dir / f"{label}_goni.mat"
    event_path = session_dir / f"{label}_event.mat"
    goni_x, srate, labels = load_goni(goni_path)
    events = load_events(event_path)
    segments, n_mi_events, max_time = collect_trial_segments(goni_x, srate, events, labels)
    highlight_id = representative_trial_id(segments)
    png_path = draw_figure(
        segments, max_time, highlight_id, args.output_dir, args.basename
    )
    traces_path, meta_path = write_source_data(
        segments, n_mi_events, highlight_id, srate, label, args.source_data_dir, args.basename
    )
    print(f"Session: {label}  (dir: {session_dir})")
    print(f"Imagined-walking trials plotted: {n_mi_events}")
    print(f"Highlighted representative trial: {highlight_id}")
    print(f"Goniometer sampling rate: {srate:g} Hz")
    print(f"PNG: {png_path}")
    print(f"Source data: {traces_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
