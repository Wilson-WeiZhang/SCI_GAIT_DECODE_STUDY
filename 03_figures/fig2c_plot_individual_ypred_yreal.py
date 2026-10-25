"""Figure 2 — Reconstruction of continuous gait kinematics from imagined walking

fig2c_plot_individual_ypred_yreal.py

Reads:
  04_source_data/fig2/fig2c_individual_ypred_yreal_metadata.csv
  04_source_data/fig2/fig2c_individual_ypred_yreal_source_data.csv

Renders one panel or table from the inputs above; it performs no
preprocessing and fits no model.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import panel_style as PS
from project_palette import JOINT_COLORS, TARGET_GONI
from sup_panel_export import PANEL_MODE, PANEL_DIR, export_stats
import analysis_config as _ec
_ec.require("main", "main")


ROOT = Path(__file__).resolve().parents[1]
PANEL_ALL_JOINTS = ["LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"]
PANEL_LEGEND_OWNER = "S2f_trial_good_best"
DEFAULT_MAT = (
    _ec.path("h_iw_dump_dir")
    / "subj11_sess01_fold6_trial_dump.mat"
)
MAIN_EXAMPLE = ("subj42_sess01", 1)
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLE = ("subj06_sess02", 31)
if _ec.BUILD_ID in _ec.DECODER_BUILDS:
    MAIN_EXAMPLE = ("subj17_sess01", 51)
if _ec.BUILD_ID == _ec.MAIN:
    MAIN_EXAMPLE = ("subj27_sess01", 45)
DEFAULT_OUT = ROOT / "figures_out" / "fig2" / "fig2c_individual_ypred_yreal.png"
SD_DIR = ROOT / "04_source_data" / "fig2"
DEFAULT_META_CSV = SD_DIR / "fig2c_individual_ypred_yreal_metadata.csv"
DEFAULT_SOURCE_CSV = SD_DIR / "fig2c_individual_ypred_yreal_source_data.csv"
START_SEC = 4.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--trial-idx", type=int, default=None, help="trial_idx in <session>_dump.h5")
    parser.add_argument("--trial-ordinal", type=int, default=5)
    parser.add_argument("--joints", nargs="+", default=["LHipW X", "LKneW X", "LAnkW X"])
    parser.add_argument("--pred-field", choices=["A_pred", "P_pred", "F_pred", "C_pred"], default="C_pred")
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-csv", type=Path, default=DEFAULT_META_CSV)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--normalize", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--fig-width", type=float, default=5.67)
    parser.add_argument("--fig-height", type=float, default=2.75)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--meta-extra", action="append", default=[], metavar="KEY=VALUE",
                        help="extra metadata column, repeatable (e.g. --meta-extra key=value)")
    parser.add_argument("--panel-name", default=None, metavar="S2f_trial_good_best",
                        help="SI panel file stem; with SI_PANEL_SPEC=panel writes "
                             "figures_out/supplementary/_panels/<name>.png plus its sidecars and nothing else")
    return parser.parse_args()


def decode_matlab_string(value: np.ndarray) -> str:
    arr = np.asarray(value).reshape(-1)
    if arr.dtype.kind in {"U", "S", "O"}:
        return "".join(str(x) for x in arr).strip()
    return "".join(chr(int(x)) for x in arr if int(x) > 0).strip()


def read_cell(mat: h5py.File, name: str, index_1based: int) -> np.ndarray:
    if name not in mat:
        raise KeyError(f"{name} not found in {mat.filename}")
    cell = mat[name]
    if index_1based < 1 or index_1based > cell.shape[0]:
        raise IndexError(f"{name} has {cell.shape[0]} cells; requested {index_1based}")
    ref = cell[index_1based - 1, 0]
    return np.asarray(mat[ref])


def target_labels(mat: h5py.File) -> list[str]:
    labels_ref = mat["meta/target_labels"][0, 0]
    labels_cell = mat[labels_ref]
    labels = []
    for idx in range(labels_cell.shape[0]):
        ref = labels_cell[idx, 0]
        labels.append(decode_matlab_string(np.asarray(mat[ref])))
    return labels


def original_trial_index(mat: h5py.File, trial_ordinal: int) -> int | None:
    if "meta/original_trial_indices" not in mat:
        return None
    values = np.asarray(mat["meta/original_trial_indices"]).reshape(-1)
    if 1 <= trial_ordinal <= values.size:
        return int(values[trial_ordinal - 1])
    return None


def session_label(mat: h5py.File) -> str | None:
    if "meta/session" not in mat:
        return None
    data = np.asarray(mat["meta/session"])
    return decode_matlab_string(data)


def joint_color(joint: str) -> str:
    low = joint.lower()
    for key, color in JOINT_COLORS.items():
        if key in low:
            return color
    return TARGET_GONI


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    ok = np.isfinite(a[:n]) & np.isfinite(b[:n])
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[:n][ok], b[:n][ok])[0, 1])


def _extract_trace_h5(h5_path: Path, trial_idx: int, joint: str, pred_field: str) -> dict[str, object]:
    """same window rule (time_sec >= START_SEC) and the same returned fields."""
    import dump_h5
    if pred_field not in ("A_pred", "C_pred"):
        raise SystemExit(f"the dumps carry Y_true / A_pred / C_pred only, not {pred_field}")
    t = dump_h5.load_trial(h5_path, trial_idx, ("Y_true", pred_field))
    if joint not in t["joints"]:
        raise ValueError(f"Joint {joint!r} not found. Available labels: {t['joints']}")
    j = t["joints"].index(joint)
    time = np.asarray(t["time_sec"], dtype=float).reshape(-1)
    y_true = np.asarray(t["Y_true"][j], dtype=float).reshape(-1)
    y_pred = np.asarray(t[pred_field][j], dtype=float).reshape(-1)
    keep = time >= START_SEC
    if keep.any():
        time, y_true, y_pred = time[keep], y_true[keep], y_pred[keep]
    return {"time": time, "y_true": y_true, "y_pred": y_pred, "x_label": "Time (s)", "joint": joint,
            "pred_field": pred_field, "session": t["session"], "trial_idx": t["trial_idx"],
            "trial_ordinal": t["test_ordinal"], "fold": t["fold"], "r": safe_corr(y_true, y_pred)}


def extract_trace(mat_path: Path, trial_ordinal: int, joint: str, pred_field: str) -> dict[str, object]:
    if _ec.DUMP_FORMAT == "pyport_session_h5":
        return _extract_trace_h5(mat_path, trial_ordinal, joint, pred_field)
    _ec.require_dump_format("matlab_fold_mat")
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)
    with h5py.File(mat_path, "r") as mat:
        labels = target_labels(mat)
        if joint not in labels:
            raise ValueError(f"Joint {joint!r} not found. Available labels: {labels}")
        joint_idx = labels.index(joint)

        y_true_cell = read_cell(mat, "Y_true", trial_ordinal)
        y_pred_cell = read_cell(mat, pred_field, trial_ordinal)
        time_cell = read_cell(mat, "time_sec", trial_ordinal) if "time_sec" in mat else None

        y_true = np.asarray(y_true_cell[joint_idx, :], dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred_cell[joint_idx, :], dtype=float).reshape(-1)
        n = min(y_true.size, y_pred.size)
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        if time_cell is None:
            time = np.arange(n, dtype=float)
            x_label = "Sample"
        else:
            time = np.asarray(time_cell, dtype=float).reshape(-1)[:n]
            x_label = "Time (s)"

        keep = time >= START_SEC
        if keep.any():
            time = time[keep]
            y_true = y_true[keep]
            y_pred = y_pred[keep]

        return {
            "time": time,
            "y_true": y_true,
            "y_pred": y_pred,
            "x_label": x_label,
            "joint": joint,
            "pred_field": pred_field,
            "session": session_label(mat),
            "trial_idx": original_trial_index(mat, trial_ordinal),
            "trial_ordinal": trial_ordinal,
            "r": safe_corr(y_true, y_pred),
        }


def rnote(v: float) -> str:
    """The one statistic a trace axes carries: its own per-trial r (italic symbol)."""
    return "$\\it{r}$ = " + f"{v:.3f}".replace("-", "−")


def assert_panel_clean(fig, axes, name, allow_texts=()) -> None:
    """SI panel spec: data + axes + axis titles only (colour-bar tick text exempted; none here)."""
    bad = []
    for i, ax in enumerate(axes):
        if any(ax.get_title(loc) for loc in ("center", "left", "right")):
            bad.append(f"axes[{i}] carries a title")
        if ax.get_legend() is not None:
            bad.append(f"axes[{i}] carries a legend")
        stray = [t.get_text() for t in ax.texts if t.get_text() not in allow_texts]
        if stray:
            bad.append(f"axes[{i}] carries text {stray}")
    if list(fig.texts):
        bad.append(f"figure carries text {[t.get_text() for t in fig.texts]}")
    if bad:
        raise SystemExit(f"PANEL SPEC VIOLATION [{name}]: " + "; ".join(bad))
    shown = " | ".join(sorted(allow_texts)).replace("−", "-")
    print(f"panel spec OK [{name}]: no title, no legend, no figure text; in-axes text limited to "
          f"the whitelisted statistic{'s' if len(allow_texts) != 1 else ''} [{shown}]")


def draw_traces(traces: list[dict[str, object]], out_png: Path, dpi: int,
                fig_width: float = 4.86, fig_height: float = 2.75,
                annotate_r: bool = False, show_legend: bool = False,
                panel_name: str | None = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 7.7,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
        }
    )
    if panel_name is not None:
        import si_panel_style as SI
        SI.apply(plt)
        fig, axes = SI.panel_col(plt, len(traces), 72.01, 34.93, gap_mm=1.2, left_mm=9.6, bottom_mm=8.0,
                                 top_mm=0.5, right_mm=1.6)
        for ax in axes[:-1]:
            ax.sharex(axes[-1])
            ax.tick_params(labelbottom=False)
    else:
        fig, axes = plt.subplots(len(traces), 1, figsize=(fig_width, fig_height), sharex=True, constrained_layout=True)
    if len(traces) == 1:
        axes = [axes]
    lw_trace = 1.0 if panel_name is not None else 1.2

    notes: list[str] = []
    for ax, trace in zip(axes, traces):
        joint = str(trace["joint"])
        color = joint_color(joint)
        time = np.asarray(trace["time"], dtype=float)
        y_true = np.asarray(trace["y_true"], dtype=float)
        y_pred = np.asarray(trace["y_pred"], dtype=float)
        if trace.get("normalize") == "zscore":
            for arr in (y_true, y_pred):
                ok = np.isfinite(arr)
                if ok.sum() > 1:
                    arr -= np.nanmean(arr[ok])
                    sd = np.nanstd(arr[ok])
                    if sd > 0:
                        arr /= sd

        ax.plot(time, y_true, color="#1a1a1a", linewidth=lw_trace, label="Reference", zorder=3)
        ax.plot(time, y_pred, color=color, linewidth=lw_trace, label="Decoded", zorder=2)
        finite = np.concatenate([y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_pred)]])
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            rng = max(hi - lo, 1e-6)
            ax.set_ylim(lo - 0.12 * rng, hi + 0.5 * rng)
        _s = joint.replace("W X", "")
        _n = {"Hip": "hip", "Kne": "knee", "Ank": "ankle"}.get(_s[1:], _s[1:].lower())
        ax.set_ylabel(f"{_s[0]} {_n}", labelpad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if annotate_r and trace.get("r") is not None and np.isfinite(float(trace["r"])):
            ax.text(0.015, 0.96, f"r = {float(trace['r']):.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.8, color="#1a1a1a")
        elif panel_name is not None and np.isfinite(float(trace["r"])):
            notes.append(rnote(float(trace["r"])))
            PS.stat_note(ax, notes[-1], loc="upper left", fontsize=SI.TICK_PT)
        if panel_name is not None:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(3))
            SI.finish(ax)

    if show_legend:
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                       frameon=False, fontsize=7.7, handletextpad=0.6, columnspacing=1.6)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel(str(traces[0]["x_label"]))
    ylabel = "Normalized angle" if traces[0].get("normalize") == "zscore" else "Goni angle (a.u.)"
    if panel_name is None:
        fig.supylabel(ylabel, x=-0.03, fontsize=8.8)
    elif panel_name is not None:
        assert_panel_clean(fig, list(axes), panel_name, allow_texts=set(notes))

    if panel_name is not None:
        SI.save(fig, out_png)
    else:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def panel_export(args: argparse.Namespace, traces: list[dict[str, object]]) -> None:
    """SI panel spec: write supplementary/_panels/<panel-name>.png + legend / statistics sidecars."""
    from matplotlib.lines import Line2D
    name = args.panel_name
    if not name or not name.startswith("S") or "_" not in name:
        raise SystemExit("SI_PANEL_SPEC=panel needs --panel-name like S2f_trial_good_best")
    figno, letter, slug = name[:2], name[2], name.split("_", 1)[1]

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    png = PANEL_DIR / f"{name}.png"
    draw_traces(traces, png, args.dpi, args.fig_width, args.fig_height, panel_name=name)
    print("wrote", png)

    if name == PANEL_LEGEND_OWNER:
        import si_panel_style as SI
        handles = [Line2D([0], [0], color="#1a1a1a", lw=1.0, label="Reference")]
        handles += [Line2D([0], [0], color=joint_color(str(t["joint"])), lw=1.0,
                           label=f"Decoded, {PS.jlab(str(t['joint']).replace('W X', ''))}")
                    for t in traces]
        SI.save_legend(handles, PANEL_DIR / f"{name}_legend.png", ncol=len(handles), columnspacing=1.5)

    six = [extract_trace(args.mat, args.trial_ordinal, j, args.pred_field)["r"]
           for j in PANEL_ALL_JOINTS]
    lines = [f"{PS.jlab(str(t['joint']).replace('W X', ''))}  $r$ = {float(t['r']):.3f}"
             for t in traces]
    lines.append(f"three-joint mean  $r$ = {np.nanmean([float(t['r']) for t in traces]):.3f}")
    lines.append(f"six-joint mean  $r$ = {np.nanmean(six):.3f}")
    export_stats(lines, figno, letter, slug, width=2.6)
    (PANEL_DIR / f"{name}_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", PANEL_DIR / f"{name}_stats.txt")
    for ln in lines:
        print("   ", ln)
    print("    six-joint r, full precision:", ", ".join(f"{v:.6f}" for v in six))


def main() -> None:
    args = parse_args()
    if not _ec.IS_DEFAULT:
        if args.mat == DEFAULT_MAT:
            args.mat = _ec.path("h_iw_dump_dir") / f"{MAIN_EXAMPLE[0]}_dump.h5"
            if args.trial_idx is None:
                args.trial_idx = MAIN_EXAMPLE[1]
        if args.trial_idx is None:
            raise SystemExit(f"name the trial with --trial-idx (the per-session dump is addressed by trial index)")
        args.trial_ordinal = args.trial_idx
    traces = [extract_trace(args.mat, args.trial_ordinal, joint, args.pred_field) for joint in args.joints]
    for trace in traces:
        trace["normalize"] = args.normalize
    if PANEL_MODE:
        panel_export(args, traces)
        return
    args.out_png = _ec.scoped(args.out_png)
    args.meta_csv = _ec.scoped(args.meta_csv)
    args.source_csv = _ec.scoped(args.source_csv)
    draw_traces(traces, args.out_png, args.dpi, args.fig_width, args.fig_height)

    args.meta_csv.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    for item in args.meta_extra:
        if "=" not in item:
            raise SystemExit(f"--meta-extra expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra[key.strip()] = value
    try:
        src = args.mat.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        src = args.mat.as_posix()
    build_id = _ec.TAG if _ec.TAG in src else "undetermined_from_inputs"
    if _ec.BUILD_ID == _ec.MAIN and _ec.MAIN in src:
        build_id = _ec.TAG
    rows = []
    for trace in traces:
        row = {
            "source": src,
            "session": trace["session"],
            "trial_ordinal": trace["trial_ordinal"],
            "trial_idx": trace["trial_idx"],
            "joint": trace["joint"],
            "pred_field": trace["pred_field"],
            "r": float(trace["r"]),
            "build_id": build_id,
            "panel": "Fig. 2d" if args.panel_name is None else f"Supplementary Fig. {args.panel_name.split('_')[0]}",
            "paired_panel": "Fig. 2e (same trial, right limb)",
        }
        row.update(extra)
        rows.append(row)
    import csv
    with args.meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    args.source_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.source_csv.open("w", newline="", encoding="utf-8") as f:
        sw = csv.writer(f)
        sw.writerow(["joint", "sample", "time_s", "reference", "decoded"])
        for trace in traces:
            t = np.asarray(trace["time"], float).ravel()
            cols = {}
            for key in ("y_true", "y_pred"):
                a = np.asarray(trace[key], float).ravel().copy()
                if trace.get("normalize") == "zscore":
                    ok = np.isfinite(a)
                    if ok.sum() > 1:
                        a -= np.nanmean(a[ok])
                        s = np.nanstd(a[ok])
                        if s > 0:
                            a /= s
                cols[key] = a
            jn = str(trace["joint"]).replace("W X", "")
            for i in range(len(t)):
                sw.writerow([jn, i, f"{t[i]:.4f}", f"{cols['y_true'][i]:.6f}", f"{cols['y_pred'][i]:.6f}"])

    print(f"source={args.mat}")
    print(f"session={traces[0]['session']}")
    print(f"trial_ordinal={traces[0]['trial_ordinal']}")
    print(f"trial_idx={traces[0]['trial_idx']}")
    for trace in traces:
        print(f"{trace['joint']} r={float(trace['r']):.6f}")
    print(f"wrote={args.out_png}")
    print(f"wrote={args.meta_csv}")


if __name__ == "__main__":
    main()
