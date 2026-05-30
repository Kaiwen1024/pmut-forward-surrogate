#!/usr/bin/env python3
"""Compute application-level test metrics and draw a four-panel performance summary.

All numerical content is regenerated from the current fixed held-out test
artifacts. The older graduation-project figures are used only as style and
metric-definition references.
"""

from __future__ import annotations

import csv
import json
import math
import re
import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "export"

PROJECT_ROOT = Path(".")
EVAL_DIR = PROJECT_ROOT / "results/fullrange_10k_retrain/eval_test"
SOURCE_DATA_DIR = Path("data/raw_mat")

TRUE_PATH = EVAL_DIR / "y_test_true.npy"
PRED_PATH = EVAL_DIR / "y_test_pred.npy"
TEST_INDICES_PATH = EVAL_DIR / "test_global_indices.npy"
SUMMARY_PATH = EVAL_DIR / "summary.json"

ROI_ROWS = slice(20, 30)
ROI_COLS = slice(20, 30)
FIELD_UNIT = "Pa"
COUNT_RE = re.compile(r"_50x50-(\d+)_")

COUNT_INTERVALS = [
    ("0-100", 0, 101),
    ("101-159", 101, 160),
    ("160-279", 160, 280),
    ("280-427", 280, 428),
    ("428-848", 428, 849),
    ("849-1267", 849, 1268),
    ("1268-1678", 1268, 1679),
    ("1679-2092", 1679, 2093),
    ("2093-2500", 2093, 2501),
]

INTERVAL_COLORS = {
    "0-100": "#8B9AAD",
    "101-159": "#D9A441",
    "160-279": "#74A86F",
    "280-427": "#4DA1B8",
    "428-848": "#4F78B8",
    "849-1267": "#8664B8",
    "1268-1678": "#B05EBB",
    "1679-2092": "#5D6170",
    "2093-2500": "#D86759",
}

HIST_FACE = "#A7C9D6"
HIST_EDGE = "#5D91A2"
MEAN_COLOR = "#D16B5F"
MEDIAN_COLOR = "#20252B"
P95_COLOR = "#6E7681"


def load_frequencies_mhz() -> np.ndarray:
    files = sorted(SOURCE_DATA_DIR.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {SOURCE_DATA_DIR}")
    mat = sio.loadmat(files[0])
    if "fHz" not in mat:
        raise KeyError(f"{files[0]} does not contain fHz")
    return np.asarray(mat["fHz"], dtype=np.float64).reshape(-1) / 1e6


def parse_active_count(path: Path) -> int:
    match = COUNT_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse active count from {path.name}")
    return int(match.group(1))


def count_interval(active_count: int) -> str:
    for label, lo, hi in COUNT_INTERVALS:
        if lo <= active_count < hi:
            return label
    return "other"


def scalar_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def response_features(curve: np.ndarray, freq_mhz: np.ndarray) -> dict[str, float]:
    curve = np.asarray(curve, dtype=np.float64)
    peak_idx = int(np.argmax(curve))
    peak = float(curve[peak_idx])
    peak_freq = float(freq_mhz[peak_idx])
    above_6db = np.where(curve >= 0.5 * peak)[0]
    if above_6db.size:
        left = float(freq_mhz[above_6db[0]])
        right = float(freq_mhz[above_6db[-1]])
    else:
        left = peak_freq
        right = peak_freq
    center = 0.5 * (left + right)
    bw = right - left
    fbw = bw / max(center, 1e-12)
    return {
        "peak_pa": peak,
        "peak_freq_mhz": peak_freq,
        "center_6db_mhz": center,
        "bandwidth_6db_mhz": bw,
        "fbw_6db": fbw,
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ["mean", "median", "p95", "max"]}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def empty_accumulator() -> dict[str, float]:
    return {"sse": 0.0, "sae": 0.0, "sum": 0.0, "sum_sq": 0.0, "n": 0.0}


def update_accumulator(acc: dict[str, float], truth: np.ndarray, pred: np.ndarray) -> None:
    yt = np.asarray(truth, dtype=np.float64)
    yp = np.asarray(pred, dtype=np.float64)
    err = yp - yt
    acc["sse"] += float(np.sum(err**2))
    acc["sae"] += float(np.sum(np.abs(err)))
    acc["sum"] += float(np.sum(yt))
    acc["sum_sq"] += float(np.sum(yt**2))
    acc["n"] += float(yt.size)


def finalize_accumulator(acc: dict[str, float]) -> dict[str, float]:
    n = max(float(acc["n"]), 1.0)
    sst = float(acc["sum_sq"] - (acc["sum"] ** 2) / n)
    return {
        "rmse_pa": float(np.sqrt(acc["sse"] / n)),
        "mae_pa": float(acc["sae"] / n),
        "r2": float(1.0 - acc["sse"] / sst) if sst > 1e-12 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def fmt_value(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "--"
    if unit == "Pa":
        if abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.1f}"
    if unit == "MHz":
        return f"{value:.3f}"
    if unit == "fraction":
        return f"{value:.3f}"
    return f"{value:.3g}"


def latex_escape(text: str) -> str:
    return text.replace("%", r"\%").replace("_", r"\_")


def build_latex_table(summary_rows: list[dict[str, object]]) -> str:
    keep = [
        "ROI curve RMSE",
        "Peak response error",
        "Peak-frequency error",
        "6-dB center-frequency error",
        "6-dB bandwidth error",
        "High-response RMSE",
    ]
    rows = [r for r in summary_rows if r["quantity"] in keep]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Application-level errors on the fixed held-out test set.}",
        r"\label{tab:application_metrics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Quantity & Mean & Median & 95th \\",
        r"\midrule",
    ]
    for row in rows:
        unit = str(row["unit"])
        label = latex_escape(str(row["latex_label"]))
        vals = [fmt_value(float(row[key]), unit) for key in ["mean", "median", "p95"]]
        if unit == "Pa":
            vals = [f"{v}" for v in vals]
        lines.append(f"{label} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Values are in Pa except frequency errors in MHz. High-response RMSE is computed on the top 10\% true-field amplitudes of each test sample.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def compute_metrics() -> tuple[dict, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    y_true = np.load(TRUE_PATH, mmap_mode="r")
    y_pred = np.load(PRED_PATH, mmap_mode="r")
    test_indices = np.load(TEST_INDICES_PATH, mmap_mode="r")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")
    if y_true.shape[0] != test_indices.shape[0]:
        raise ValueError("test_global_indices does not match prediction arrays")

    freq_mhz = load_frequencies_mhz()
    if y_true.shape[1] != freq_mhz.shape[0]:
        raise ValueError(f"Frequency count {freq_mhz.shape[0]} does not match channels {y_true.shape[1]}")

    source_files = sorted(SOURCE_DATA_DIR.glob("*.mat"))
    active_counts = np.asarray([parse_active_count(source_files[int(i)]) for i in test_indices], dtype=np.int64)
    intervals = [count_interval(int(v)) for v in active_counts]

    roi_true = np.asarray(y_true[:, :, ROI_ROWS, ROI_COLS].mean(axis=(2, 3)), dtype=np.float64)
    roi_pred = np.asarray(y_pred[:, :, ROI_ROWS, ROI_COLS].mean(axis=(2, 3)), dtype=np.float64)
    roi_err = roi_pred - roi_true

    aggregate_acc = {
        "All test": empty_accumulator(),
        "n(M)>100": empty_accumulator(),
    }
    for label, _, _ in COUNT_INTERVALS:
        aggregate_acc[label] = empty_accumulator()

    sample_rows: list[dict[str, object]] = []
    high_rmse_values: list[float] = []
    for i in range(y_true.shape[0]):
        truth = np.asarray(y_true[i], dtype=np.float64)
        pred = np.asarray(y_pred[i], dtype=np.float64)
        err = pred - truth
        mse = float(np.mean(err**2))
        mae = float(np.mean(np.abs(err)))
        field_rmse = math.sqrt(mse)
        field_r2 = scalar_r2(truth, pred)

        threshold = float(np.percentile(truth, 90.0))
        high_mask = truth >= threshold
        high_rmse = float(np.sqrt(np.mean(err[high_mask] ** 2))) if np.any(high_mask) else float("nan")
        high_rmse_values.append(high_rmse)

        update_accumulator(aggregate_acc["All test"], truth, pred)
        if int(active_counts[i]) > 100:
            update_accumulator(aggregate_acc["n(M)>100"], truth, pred)
        update_accumulator(aggregate_acc[intervals[i]], truth, pred)

        true_feat = response_features(roi_true[i], freq_mhz)
        pred_feat = response_features(roi_pred[i], freq_mhz)
        row: dict[str, object] = {
            "sample_index_in_test": int(i),
            "global_index": int(test_indices[i]),
            "active_count": int(active_counts[i]),
            "active_count_interval": intervals[i],
            "field_rmse_pa": field_rmse,
            "field_mae_pa": mae,
            "field_r2": field_r2,
            "roi_curve_rmse_pa": float(np.sqrt(np.mean(roi_err[i] ** 2))),
            "roi_curve_mae_pa": float(np.mean(np.abs(roi_err[i]))),
            "high_response_rmse_pa": high_rmse,
        }
        for key, value in true_feat.items():
            row[f"true_{key}"] = value
        for key, value in pred_feat.items():
            row[f"pred_{key}"] = value
            row[f"abs_err_{key}"] = abs(value - true_feat[key])
        sample_rows.append(row)

    per_freq_rows: list[dict[str, object]] = []
    for ch, freq in enumerate(freq_mhz):
        truth = np.asarray(y_true[:, ch], dtype=np.float64)
        pred = np.asarray(y_pred[:, ch], dtype=np.float64)
        err = pred - truth
        per_freq_rows.append(
            {
                "channel": int(ch),
                "frequency_mhz": float(freq),
                "rmse_pa": float(np.sqrt(np.mean(err**2))),
                "mae_pa": float(np.mean(np.abs(err))),
                "r2": scalar_r2(truth, pred),
            }
        )

    group_rows: list[dict[str, object]] = []
    sample_rmse = np.asarray([float(r["field_rmse_pa"]) for r in sample_rows], dtype=np.float64)
    sample_mae = np.asarray([float(r["field_mae_pa"]) for r in sample_rows], dtype=np.float64)
    for label, _, _ in COUNT_INTERVALS:
        mask = np.asarray([iv == label for iv in intervals], dtype=bool)
        if not np.any(mask):
            continue
        rmse = sample_rmse[mask]
        mae = sample_mae[mask]
        aggregate = finalize_accumulator(aggregate_acc[label])
        group_rows.append(
            {
                "active_count_interval": label,
                "n_samples": int(np.sum(mask)),
                "pointwise_rmse_pa": aggregate["rmse_pa"],
                "pointwise_mae_pa": aggregate["mae_pa"],
                "pointwise_r2": aggregate["r2"],
                "rmse_mean_pa": float(np.mean(rmse)),
                "rmse_median_pa": float(np.median(rmse)),
                "rmse_p95_pa": float(np.percentile(rmse, 95.0)),
                "rmse_max_pa": float(np.max(rmse)),
                "mae_mean_pa": float(np.mean(mae)),
                "mae_median_pa": float(np.median(mae)),
            }
        )

    summary_specs = [
        ("ROI curve RMSE", r"ROI curve RMSE", "Pa", np.asarray([r["roi_curve_rmse_pa"] for r in sample_rows])),
        ("ROI curve MAE", r"ROI curve MAE", "Pa", np.asarray([r["roi_curve_mae_pa"] for r in sample_rows])),
        ("Peak response error", r"Peak $|P|$ err.", "Pa", np.asarray([r["abs_err_peak_pa"] for r in sample_rows])),
        ("Peak-frequency error", r"Peak $f$ err.", "MHz", np.asarray([r["abs_err_peak_freq_mhz"] for r in sample_rows])),
        ("6-dB center-frequency error", r"6-dB center $f$ err.", "MHz", np.asarray([r["abs_err_center_6db_mhz"] for r in sample_rows])),
        ("6-dB bandwidth error", r"6-dB BW err.", "MHz", np.asarray([r["abs_err_bandwidth_6db_mhz"] for r in sample_rows])),
        ("6-dB fractional-bandwidth error", r"6-dB FBW err.", "fraction", np.asarray([r["abs_err_fbw_6db"] for r in sample_rows])),
        ("High-response RMSE", r"High-response RMSE", "Pa", np.asarray(high_rmse_values, dtype=np.float64)),
    ]
    summary_rows: list[dict[str, object]] = []
    for quantity, label, unit, values in summary_specs:
        stats = summarize(values)
        summary_rows.append({"quantity": quantity, "latex_label": label, "unit": unit, **stats})

    summary_json = {
        "figure": "Application-level metrics and error breakdown",
        "data_sources": {
            "y_test_true": str(TRUE_PATH),
            "y_test_pred": str(PRED_PATH),
            "test_global_indices": str(TEST_INDICES_PATH),
            "summary": str(SUMMARY_PATH),
            "source_data_dir": str(SOURCE_DATA_DIR),
        },
        "n_test_samples": int(y_true.shape[0]),
        "field_unit": FIELD_UNIT,
        "roi_definition": {"rows": [20, 30], "cols": [20, 30], "shape": "10x10 center ROI"},
        "frequency_mhz": [float(v) for v in freq_mhz],
        "active_count_intervals": [{"label": label, "min_inclusive": lo, "max_exclusive": hi} for label, lo, hi in COUNT_INTERVALS],
        "application_metric_summary": summary_rows,
        "overall_field_metrics": [
            {"scope": "All test", "n_samples": int(y_true.shape[0]), **finalize_accumulator(aggregate_acc["All test"])},
            {"scope": "n(M)>100", "n_samples": int(np.sum(active_counts > 100)), **finalize_accumulator(aggregate_acc["n(M)>100"])},
        ],
        "group_summary": group_rows,
        "per_frequency_summary": per_freq_rows,
        "notes": [
            "ROI response metrics are computed from the mean acoustic magnitude over the central 10x10 ROI.",
            "Peak, center-frequency, and bandwidth metrics are derived from the ROI mean frequency response.",
            "High-response RMSE is computed over the top 10% true-field amplitudes for each test sample.",
        ],
    }
    return summary_json, summary_rows, per_freq_rows, group_rows, sample_rows


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="major", length=2.4, width=0.55, pad=1.6)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color="#DEE5EE", linewidth=0.42, alpha=0.78)
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color="#E8ECF2", linewidth=0.42, alpha=0.78)
    ax.set_axisbelow(True)


def compact_number(value: float) -> str:
    value = float(value)
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def panel_label(ax: plt.Axes, label: str, fontsize: float = 7.0) -> None:
    ax.text(
        -0.14,
        1.07,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )


def mean_axis_label(
    ax: plt.Axes,
    value: float,
    text: str,
    y_offset_points: float = 0.0,
    labelsize: float = 5.0,
) -> None:
    label_value = value
    if y_offset_points:
        y0, y1 = ax.get_ylim()
        label_value += y_offset_points / 72.0 * (y1 - y0) / max(ax.bbox.height / ax.figure.dpi, 1e-9)
    ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([label_value]))
    ax.yaxis.set_minor_formatter(mpl.ticker.FixedFormatter([text]))
    ax.tick_params(
        axis="y",
        which="minor",
        length=0,
        pad=1.6,
        labelsize=labelsize,
        labelcolor="#D58B5C",
    )
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_horizontalalignment("right")
        tick.label1.set_verticalalignment("center")


def draw_breakdown(
    summary_json: dict,
    per_freq_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Tinos",
            "mathtext.it": "Tinos:italic",
            "mathtext.bf": "Tinos:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.8,
            "axes.labelsize": 6.9,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    labels = [str(r["active_count_interval"]) for r in group_rows]
    tick_labels = [label.replace("-", "-\n") for label in labels]
    point_rmse = np.asarray([r["pointwise_rmse_pa"] for r in group_rows], dtype=np.float64)
    point_mae = np.asarray([r["pointwise_mae_pa"] for r in group_rows], dtype=np.float64)
    point_r2 = np.asarray([r["pointwise_r2"] for r in group_rows], dtype=np.float64)
    group_counts = np.asarray([r["n_samples"] for r in group_rows], dtype=np.int64)

    overall_metrics = {str(r["scope"]): r for r in summary_json["overall_field_metrics"]}
    all_metric = overall_metrics["All test"]
    main_metric = overall_metrics["n(M)>100"]
    all_n = int(all_metric["n_samples"])
    main_n = int(main_metric["n_samples"])
    all_rmse = float(all_metric["rmse_pa"])
    all_mae = float(all_metric["mae_pa"])
    all_r2 = float(all_metric["r2"])
    main_rmse = float(main_metric["rmse_pa"])
    main_mae = float(main_metric["mae_pa"])
    main_r2 = float(main_metric["r2"])

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.15), dpi=240)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.925, bottom=0.185, wspace=0.42, hspace=0.54)

    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.25
    rmse_bars = ax.bar(x - width / 2, [all_rmse / 1000, main_rmse / 1000], width=width, color="#5D91BF", label="RMSE / 1000")
    mae_bars = ax.bar(x + width / 2, [all_mae / 1000, main_mae / 1000], width=width, color="#7CB19F", label="MAE / 1000")
    ax2 = ax.twinx()
    ax2.plot(x, [all_r2, main_r2], color="#D58B5C", marker="o", markersize=3.4, linewidth=1.1, label=r"$R^2$")
    ax.set_title("Overall pointwise metrics", pad=2.5)
    panel_label(ax, "a")
    ax.set_ylabel("Pointwise error / 1000")
    ax2.set_ylabel(r"$R^2$", labelpad=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["All test", r"$n(M)>100$"])
    ax.text(x[0], -0.12, f"n={all_n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.5, color="#626A73")
    ax.text(x[1], -0.12, f"n={main_n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.5, color="#626A73")
    ax.set_ylim(0, max(all_rmse, main_rmse) / 1000 * 1.34)
    r2_pad = max((max(all_r2, main_r2) - min(all_r2, main_r2)) * 0.7, 0.00002)
    ax2.set_ylim(min(all_r2, main_r2) - r2_pad, max(all_r2, main_r2) + r2_pad * 2.0)
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.5f"))
    for bars, values, scale in [(rmse_bars, [all_rmse, main_rmse], 1.0), (mae_bars, [all_mae, main_mae], 1.0)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{value:.0f}", ha="center", va="bottom", fontsize=5.5, color="#38434D")
    for xi, rv in zip(x, [all_r2, main_r2]):
        ax2.text(xi, rv - r2_pad * 0.38, f"{rv:.5f}", ha="center", va="top", fontsize=5.2, color="#D58B5C")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=5.7,
        loc="upper center",
        bbox_to_anchor=(0.46, 1.01),
        ncol=3,
        handlelength=1.15,
        columnspacing=0.75,
    )
    clean_axis(ax, "y")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=5.8, length=2.2, width=0.55, pad=1.5)

    ax = axes[0, 1]
    xg = np.arange(len(labels))
    ax.bar(xg, point_rmse, width=0.72, color="#80A9C9", edgecolor="white", linewidth=0.35)
    ax.axhline(all_rmse, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.9, label="Mean")
    for xi, yi, n in zip(xg, point_rmse, group_counts):
        ax.text(xi, yi + max(point_rmse) * 0.035, f"n={n}", ha="center", va="bottom", fontsize=4.9, color="#626A73")
    ax.set_title("RMSE by active-count interval", pad=2.5)
    panel_label(ax, "b")
    ax.set_ylabel("RMSE (Pa)")
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylim(0, max(np.max(point_rmse), all_rmse) * 1.25)
    mean_axis_label(ax, all_rmse, f"{all_rmse:.1f}", y_offset_points=0.8)
    ax.legend(fontsize=5.7, loc="upper left", handlelength=1.6)
    clean_axis(ax, "y")

    ax = axes[1, 0]
    ax.bar(xg, point_mae, width=0.72, color="#90B9A7", edgecolor="white", linewidth=0.35)
    ax.axhline(all_mae, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.9, label="Mean")
    ax.set_title("MAE by active-count interval", pad=2.5)
    panel_label(ax, "c")
    ax.set_ylabel("MAE (Pa)")
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylim(0, max(np.max(point_mae), all_mae) * 1.25)
    mean_axis_label(ax, all_mae, f"{all_mae:.1f}")
    ax.legend(fontsize=5.7, loc="upper left", handlelength=1.6)
    clean_axis(ax, "y")

    ax = axes[1, 1]
    ax.plot(xg, point_r2, color="#2F3D49", marker="o", markersize=2.9, linewidth=1.1)
    ax.fill_between(xg, point_r2, np.min(point_r2) - 0.00035, color="#AFC6D9", alpha=0.28)
    ax.axhline(all_r2, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.9, label="Mean")
    ax.set_title(r"$R^2$ by active-count interval", pad=2.5)
    panel_label(ax, "d")
    ax.set_ylabel(r"$R^2$")
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    r2_low = min(np.min(point_r2), all_r2)
    ax.set_ylim(r2_low - 0.0012, 1.002)
    mean_axis_label(ax, all_r2, f"{all_r2:.5f}", y_offset_points=-2.2)
    ax.legend(fontsize=5.7, loc="lower right", handlelength=1.6)
    clean_axis(ax, "both")

    for ext in ["pdf", "svg", "png"]:
        path = EXPORT_DIR / f"fig5_field_error_breakdown_wide.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
    plt.close(fig)


def draw_breakdown_singlecol(
    summary_json: dict,
    per_freq_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    suffix: str = "",
    large: bool = False,
) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "font.size": 7.4,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.65,
    }
    if large:
        params = {
            "font.size": 8.45,
            "axes.labelsize": 8.55,
            "axes.titlesize": 8.85,
            "xtick.labelsize": 7.75,
            "ytick.labelsize": 7.75,
            "axes.linewidth": 0.72,
        }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Tinos",
            "mathtext.it": "Tinos:italic",
            "mathtext.bf": "Tinos:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            **params,
            "legend.frameon": False,
        }
    )

    labels = [str(r["active_count_interval"]) for r in group_rows]
    tick_labels = [label.replace("-", "-\n") for label in labels]
    point_rmse = np.asarray([r["pointwise_rmse_pa"] for r in group_rows], dtype=np.float64)
    point_mae = np.asarray([r["pointwise_mae_pa"] for r in group_rows], dtype=np.float64)
    point_r2 = np.asarray([r["pointwise_r2"] for r in group_rows], dtype=np.float64)
    group_counts = np.asarray([r["n_samples"] for r in group_rows], dtype=np.int64)

    overall_metrics = {str(r["scope"]): r for r in summary_json["overall_field_metrics"]}
    all_metric = overall_metrics["All test"]
    main_metric = overall_metrics["n(M)>100"]
    all_n = int(all_metric["n_samples"])
    main_n = int(main_metric["n_samples"])
    all_rmse = float(all_metric["rmse_pa"])
    all_mae = float(all_metric["mae_pa"])
    all_r2 = float(all_metric["r2"])
    main_rmse = float(main_metric["rmse_pa"])
    main_mae = float(main_metric["mae_pa"])
    main_r2 = float(main_metric["r2"])

    figsize = (7.05, 4.95)
    adjust = {"left": 0.08, "right": 0.975, "top": 0.93, "bottom": 0.20, "wspace": 0.34, "hspace": 0.68}
    panel_fs = 7.6
    small_fs = {"n": 6.1, "bar": 6.0, "r2": 5.8, "legend": 6.2, "mean": 5.7, "marker": 3.8, "line_marker": 3.2}
    if large:
        figsize = (5.72, 4.88)
        adjust = {"left": 0.074, "right": 0.995, "top": 0.943, "bottom": 0.184, "wspace": 0.30, "hspace": 0.64}
        panel_fs = 8.6
        small_fs = {"n": 6.9, "bar": 6.9, "r2": 5.8, "legend": 7.05, "mean": 6.5, "marker": 4.45, "line_marker": 3.9}

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=240)
    fig.subplots_adjust(**adjust)

    ax = axes[0, 0]
    x = np.asarray([0.0, 0.88] if large else [0.0, 1.0], dtype=np.float64)
    width = 0.25
    rmse_bars = ax.bar(x - width / 2, [all_rmse / 1000, main_rmse / 1000], width=width, color="#5D91BF", label="RMSE / 1000")
    mae_bars = ax.bar(x + width / 2, [all_mae / 1000, main_mae / 1000], width=width, color="#7CB19F", label="MAE / 1000")
    ax2 = ax.twinx()
    ax2.plot(x, [all_r2, main_r2], color="#D58B5C", marker="o", markersize=small_fs["marker"], linewidth=1.2, label=r"$R^2$")
    ax.set_title("Overall pointwise metrics", pad=2.7)
    panel_label(ax, "a", fontsize=panel_fs)
    ax.set_ylabel("Pointwise error / 1000")
    ax2.set_ylabel(r"$R^2$", labelpad=(-8.0 if large else 1.0))
    if large:
        ax2.yaxis.set_label_coords(0.965, 0.585)
    ax.set_xticks(x)
    ax.set_xticklabels(["All test", r"$n(M)>100$"])
    if large:
        ax.set_xlim(-0.32, 1.24)
    ax.text(x[0], -0.12, f"n={all_n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=small_fs["n"], color="#626A73")
    ax.text(x[1], -0.12, f"n={main_n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=small_fs["n"], color="#626A73")
    ax.set_ylim(0, max(all_rmse, main_rmse) / 1000 * 1.34)
    r2_pad = max((max(all_r2, main_r2) - min(all_r2, main_r2)) * 0.7, 0.00002)
    ax2.set_ylim(min(all_r2, main_r2) - r2_pad, max(all_r2, main_r2) + r2_pad * 2.0)
    if large:
        ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.5f"))
    for bars, values in [(rmse_bars, [all_rmse, main_rmse]), (mae_bars, [all_mae, main_mae])]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{value:.0f}", ha="center", va="bottom", fontsize=small_fs["bar"], color="#38434D")
    for xi, rv in zip(x, [all_r2, main_r2]):
        ax2.text(xi, rv - r2_pad * 0.38, f"{rv:.5f}", ha="center", va="top", fontsize=small_fs["r2"], color="#D58B5C")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=small_fs["legend"],
        loc="upper center",
        bbox_to_anchor=(0.46, 1.015),
        ncol=3,
        handlelength=1.15,
        columnspacing=0.75,
    )
    clean_axis(ax, "y")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(
        labelsize=(small_fs["r2"] if large else small_fs["n"]),
        length=2.2,
        width=0.55,
        pad=(-4.0 if large else 1.5),
        direction=("in" if large else "out"),
    )
    if large:
        for tick in ax2.yaxis.get_major_ticks():
            tick.label2.set_horizontalalignment("right")

    ax = axes[0, 1]
    xg = np.arange(len(labels))
    ax.bar(xg, point_rmse, width=0.72, color="#80A9C9", edgecolor="white", linewidth=0.35)
    ax.axhline(all_rmse, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.95, label="Mean")
    for xi, yi, n in zip(xg, point_rmse, group_counts):
        ax.text(xi, yi + max(point_rmse) * 0.035, f"n={n}", ha="center", va="bottom", fontsize=small_fs["r2"], color="#626A73")
    ax.set_title("RMSE by active-count interval", pad=2.7)
    panel_label(ax, "b", fontsize=panel_fs)
    ax.set_ylabel("RMSE (Pa)", labelpad=(0.8 if large else None))
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylim(0, max(np.max(point_rmse), all_rmse) * 1.25)
    mean_axis_label(ax, all_rmse, f"{all_rmse:.1f}", y_offset_points=0.8, labelsize=small_fs["mean"])
    ax.legend(fontsize=small_fs["legend"], loc="upper left", handlelength=1.6)
    clean_axis(ax, "y")

    ax = axes[1, 0]
    ax.bar(xg, point_mae, width=0.72, color="#90B9A7", edgecolor="white", linewidth=0.35)
    ax.axhline(all_mae, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.95, label="Mean")
    ax.set_title("MAE by active-count interval", pad=2.7)
    panel_label(ax, "c", fontsize=panel_fs)
    ax.set_ylabel("MAE (Pa)")
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylim(0, max(np.max(point_mae), all_mae) * 1.25)
    mean_axis_label(ax, all_mae, f"{all_mae:.1f}", labelsize=small_fs["mean"])
    ax.legend(fontsize=small_fs["legend"], loc="upper left", handlelength=1.6)
    clean_axis(ax, "y")

    ax = axes[1, 1]
    ax.plot(xg, point_r2, color="#2F3D49", marker="o", markersize=small_fs["line_marker"], linewidth=1.2)
    ax.fill_between(xg, point_r2, np.min(point_r2) - 0.00035, color="#AFC6D9", alpha=0.28)
    ax.axhline(all_r2, color="#D58B5C", linestyle=(0, (4, 2)), linewidth=0.95, label="Mean")
    ax.set_title(r"$R^2$ by active-count interval", pad=2.7)
    panel_label(ax, "d", fontsize=panel_fs)
    ax.set_ylabel(r"$R^2$")
    ax.set_xticks(xg)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    r2_low = min(np.min(point_r2), all_r2)
    ax.set_ylim(r2_low - 0.0012, 1.002)
    mean_axis_label(ax, all_r2, f"{all_r2:.5f}", y_offset_points=-2.2, labelsize=small_fs["mean"])
    ax.legend(fontsize=small_fs["legend"], loc="lower right", handlelength=1.6)
    clean_axis(ax, "both")

    for ext in ["pdf", "svg", "png"]:
        name = "fig5_field_error_breakdown" + (f"_{suffix}" if suffix else "")
        path = EXPORT_DIR / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true", help="Write the wider exploratory export.")
    parser.add_argument("--singlecol-only", action="store_true", help="Alias for the manuscript export.")
    parser.add_argument("--singlecol-large-only", action="store_true", help="Alias for the manuscript export.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json, summary_rows, per_freq_rows, group_rows, sample_rows = compute_metrics()

    if args.wide:
        draw_breakdown(summary_json, per_freq_rows, group_rows, sample_rows)
    elif args.singlecol_only or args.singlecol_large_only:
        draw_breakdown_singlecol(summary_json, per_freq_rows, group_rows, sample_rows, large=True)
    else:
        sample_fields = list(sample_rows[0].keys())
        write_csv(DATA_DIR / "per_sample_application_metrics.csv", sample_rows, sample_fields)
        write_csv(DATA_DIR / "application_metric_summary.csv", summary_rows, ["quantity", "latex_label", "unit", "mean", "median", "p95", "max"])
        write_csv(DATA_DIR / "per_frequency_error.csv", per_freq_rows, ["channel", "frequency_mhz", "rmse_pa", "mae_pa", "r2"])
        write_csv(
            DATA_DIR / "active_count_group_error.csv",
            group_rows,
            [
                "active_count_interval",
                "n_samples",
                "pointwise_rmse_pa",
                "pointwise_mae_pa",
                "pointwise_r2",
                "rmse_mean_pa",
                "rmse_median_pa",
                "rmse_p95_pa",
                "rmse_max_pa",
                "mae_mean_pa",
                "mae_median_pa",
            ],
        )
        (DATA_DIR / "application_metrics_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

        table_tex = build_latex_table(summary_rows)
        (DATA_DIR / "table_application_metrics.tex").write_text(table_tex, encoding="utf-8")

        draw_breakdown_singlecol(summary_json, per_freq_rows, group_rows, sample_rows, large=True)


if __name__ == "__main__":
    main()
