#!/usr/bin/env python3
"""Draw a compact application-metric consistency figure.

The layout follows the graduation-project Fig. 4-4 style: left panels show
true-vs-predicted metric consistency, while right panels show grouped error
behavior. All numbers are regenerated from the latest fixed held-out test
artifacts used by the IEEE SENSORS manuscript.
"""

from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "export"

SOURCE_METRICS = Path(
    "figures/paper/fig5_field_error_breakdown/data/per_sample_application_metrics.csv"
)
SOURCE_SUMMARY = Path(
    "figures/paper/fig5_field_error_breakdown/data/application_metrics_summary.json"
)

COUNT_INTERVALS = [
    "101-159",
    "160-279",
    "280-427",
    "428-848",
    "849-1267",
    "1268-1678",
    "1679-2092",
    "2093-2500",
]

INTERVAL_COLORS = {
    "101-159": "#D9A441",
    "160-279": "#74A86F",
    "280-427": "#4DA1B8",
    "428-848": "#4F78B8",
    "849-1267": "#8664B8",
    "1268-1678": "#B05EBB",
    "1679-2092": "#5D6170",
    "2093-2500": "#D86759",
}

CAT_COLORS = {
    "Exact": "#79A889",
    "One bin": "#E8C778",
    "Two+ bins": "#D77C73",
}


def load_rows() -> list[dict[str, object]]:
    with SOURCE_METRICS.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            converted: dict[str, object] = {}
            for key, value in row.items():
                if key in {"active_count_interval"}:
                    converted[key] = value
                elif key in {"sample_index_in_test", "global_index", "active_count"}:
                    converted[key] = int(value)
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return [r for r in rows if int(r["active_count"]) > 100]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def relative_error_pct(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    out = np.full(truth.shape, np.nan, dtype=np.float64)
    mask = np.abs(truth) > 1e-12
    out[mask] = np.abs(pred[mask] - truth[mask]) / np.abs(truth[mask]) * 100.0
    return out


def format_metric_tick(value: float, key: str) -> str:
    if key == "center":
        return f"{value:.3g}"
    if key == "fbw":
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def format_pct(value: float) -> str:
    if abs(value - 100.0) < 1e-8:
        return "100%"
    if value >= 99.9:
        return "99.9%"
    return f"{value:.1f}%"


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.16,
        1.075,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="major", length=2.4, width=0.55, pad=1.7)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color="#D8DEE8", linewidth=0.42, alpha=0.72)
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color="#E8ECF2", linewidth=0.40, alpha=0.68)
    ax.set_axisbelow(True)


def discrete_values(rows: list[dict[str, object]], true_col: str, pred_col: str) -> list[float]:
    vals = {round(float(r[true_col]), 12) for r in rows}
    vals.update(round(float(r[pred_col]), 12) for r in rows)
    return sorted(vals)


def confusion_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    true_col: str,
    pred_col: str,
    key: str,
    xlabel: str,
    ylabel: str,
    stat_loc: str = "upper_left",
) -> dict[str, float]:
    values = discrete_values(rows, true_col, pred_col)
    rank = {v: i for i, v in enumerate(values)}
    matrix = np.zeros((len(values), len(values)), dtype=int)
    diffs = []
    for row in rows:
        t = round(float(row[true_col]), 12)
        p = round(float(row[pred_col]), 12)
        matrix[rank[p], rank[t]] += 1
        diffs.append(abs(rank[p] - rank[t]))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sensors_confusion_bluegreen",
        ["#F8FBFC", "#D8EAF0", "#96C5CF", "#4D91A4", "#216A7D"],
    )
    vmax = max(int(matrix.max()), 1)
    ax.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
    n = len(values)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([format_metric_tick(v, key) for v in values], fontsize=5.8)
    ax.set_yticklabels([format_metric_tick(v, key) for v in values], fontsize=5.8)
    ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        ax.add_patch(
            mpl.patches.Rectangle(
                (i - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#30343B",
                linewidth=0.7,
                zorder=3,
            )
        )

    for y in range(n):
        for x in range(n):
            count = int(matrix[y, x])
            if count <= 0:
                continue
            color = "white" if count >= vmax * 0.52 else "#1F252B"
            ax.text(x, y, f"{count}", ha="center", va="center", fontsize=5.7, color=color, zorder=4)

    diffs_arr = np.asarray(diffs, dtype=np.float64)
    exact = float(np.mean(diffs_arr == 0) * 100.0)
    within_one = float(np.mean(diffs_arr <= 1) * 100.0)
    mean_step = float(np.mean(diffs_arr))
    if stat_loc == "lower_right":
        stat_x, stat_y = 0.965, 0.035
        stat_ha, stat_va = "right", "bottom"
    else:
        stat_x, stat_y = 0.035, 0.965
        stat_ha, stat_va = "left", "top"
    ax.text(
        stat_x,
        stat_y,
        f"Exact={exact:.1f}%\nWithin 1 bin={within_one:.1f}%\nMean bin err.={mean_step:.3f}",
        transform=ax.transAxes,
        ha=stat_ha,
        va=stat_va,
        fontsize=5.4,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.42,
            "alpha": 0.90,
        },
        zorder=5,
    )
    ax.set_xlabel(xlabel, labelpad=1.4)
    ax.set_ylabel(ylabel, labelpad=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#30343B")
    return {"exact_pct": exact, "within_one_pct": within_one, "mean_bin_error": mean_step}


def discrete_error_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    true_col: str,
    pred_col: str,
) -> list[dict[str, object]]:
    values = discrete_values(rows, true_col, pred_col)
    rank = {v: i for i, v in enumerate(values)}
    categories = ["Exact", "One bin", "Two+ bins"]
    positions = np.arange(len(COUNT_INTERVALS), 0, -1)
    stats_rows = []
    percentages = np.zeros((len(COUNT_INTERVALS), len(categories)), dtype=np.float64)
    for i, interval in enumerate(COUNT_INTERVALS):
        part = [r for r in rows if r["active_count_interval"] == interval]
        diffs = []
        for row in part:
            t = round(float(row[true_col]), 12)
            p = round(float(row[pred_col]), 12)
            diffs.append(abs(rank[p] - rank[t]))
        total = max(len(diffs), 1)
        percentages[i, 0] = sum(v == 0 for v in diffs) / total * 100.0
        percentages[i, 1] = sum(v == 1 for v in diffs) / total * 100.0
        percentages[i, 2] = sum(v >= 2 for v in diffs) / total * 100.0
        stats_rows.append(
            {
                "active_count_interval": interval,
                "n_samples": len(diffs),
                "exact_pct": percentages[i, 0],
                "one_bin_pct": percentages[i, 1],
                "two_plus_bins_pct": percentages[i, 2],
            }
        )

    left = np.zeros(len(COUNT_INTERVALS), dtype=np.float64)
    for j, cat in enumerate(categories):
        widths = percentages[:, j]
        ax.barh(
            positions,
            widths,
            left=left,
            height=0.58,
            color=CAT_COLORS[cat],
            edgecolor="white",
            linewidth=0.55,
            label=cat,
            zorder=3,
        )
        for pos, x0, width in zip(positions, left, widths):
            if width >= 9.0:
                ax.text(x0 + width / 2, pos, format_pct(float(width)), ha="center", va="center", fontsize=5.4, color="#242424")
        left += widths

    ax.set_xlim(0, 100)
    ax.set_ylim(0.25, len(COUNT_INTERVALS) + 0.95)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(positions)
    ax.set_yticklabels(COUNT_INTERVALS, fontsize=5.8)
    ax.set_xlabel("Sample share (%)", labelpad=1.4)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, 0.998),
        ncol=3,
        columnspacing=0.65,
        handlelength=1.05,
        handletextpad=0.24,
        borderaxespad=0.0,
        fontsize=5.3,
    )
    for txt in leg.get_texts():
        txt.set_fontsize(5.3)
    clean_axis(ax, "x")
    ax.grid(False, axis="y")
    return stats_rows


def scatter_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    x_fn: Callable[[dict[str, object]], float],
    y_fn: Callable[[dict[str, object]], float],
    xlabel: str,
    ylabel: str,
    stat_unit: str,
) -> dict[str, float]:
    all_x = np.asarray([x_fn(r) for r in rows], dtype=np.float64)
    all_y = np.asarray([y_fn(r) for r in rows], dtype=np.float64)
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    all_x = all_x[finite]
    all_y = all_y[finite]
    low = float(min(all_x.min(), all_y.min()))
    high = float(max(all_x.max(), all_y.max()))
    pad = max((high - low) * 0.065, 1e-6)

    for interval in COUNT_INTERVALS:
        part = [r for r in rows if r["active_count_interval"] == interval]
        x = np.asarray([x_fn(r) for r in part], dtype=np.float64)
        y = np.asarray([y_fn(r) for r in part], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[mask],
            y[mask],
            s=8.0,
            color=INTERVAL_COLORS[interval],
            alpha=0.56,
            edgecolors="white",
            linewidths=0.12,
            rasterized=True,
        )

    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#575757", linestyle=(0, (3.0, 2.0)), linewidth=0.78)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, labelpad=1.4)
    ax.set_ylabel(ylabel, labelpad=1.4)

    mae = float(np.mean(np.abs(all_y - all_x)))
    r_value = pearson_r(all_x, all_y)
    if stat_unit == "log":
        stat_text = f"MAE={mae:.4f}\nr={r_value:.5f}"
    else:
        stat_text = f"MAE={mae:.1f} Pa\nr={r_value:.5f}"
    ax.text(
        0.045,
        0.955,
        stat_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.42,
            "alpha": 0.88,
        },
    )
    clean_axis(ax, "both")
    return {"mae": mae, "pearson_r": r_value}


def relative_distribution_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    true_col: str,
    pred_col: str,
    cap: float,
    rng_seed: int,
) -> list[dict[str, object]]:
    raw_values = []
    stats_rows = []
    for interval in COUNT_INTERVALS:
        part = [r for r in rows if r["active_count_interval"] == interval]
        truth = np.asarray([float(r[true_col]) for r in part], dtype=np.float64)
        pred = np.asarray([float(r[pred_col]) for r in part], dtype=np.float64)
        vals = relative_error_pct(truth, pred)
        vals = vals[np.isfinite(vals)]
        raw_values.append(vals)
        stats_rows.append(
            {
                "active_count_interval": interval,
                "n_samples": int(vals.size),
                "mean_rel_err_pct": float(np.mean(vals)) if vals.size else float("nan"),
                "median_rel_err_pct": float(np.median(vals)) if vals.size else float("nan"),
                "p95_rel_err_pct": float(np.percentile(vals, 95.0)) if vals.size else float("nan"),
                "max_rel_err_pct": float(np.max(vals)) if vals.size else float("nan"),
            }
        )

    positions = np.arange(len(COUNT_INTERVALS), 0, -1)
    box_values = []
    for vals in raw_values:
        clipped = vals[vals <= cap]
        box_values.append(clipped if clipped.size else np.array([np.nan]))
    box = ax.boxplot(
        box_values,
        vert=False,
        positions=positions,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#2E2E2E", "linewidth": 0.82},
        whiskerprops={"color": "#6E6E6E", "linewidth": 0.62},
        capprops={"color": "#6E6E6E", "linewidth": 0.62},
        boxprops={"linewidth": 0.62},
    )
    for patch, interval in zip(box["boxes"], COUNT_INTERVALS):
        patch.set_facecolor(INTERVAL_COLORS[interval])
        patch.set_alpha(0.24)
        patch.set_edgecolor(INTERVAL_COLORS[interval])

    rng = np.random.default_rng(rng_seed)
    has_clipped = False
    for pos, vals, interval in zip(positions, raw_values, COUNT_INTERVALS):
        clipped_count = int(np.sum(vals > cap))
        vals = vals[vals <= cap]
        if vals.size > 120:
            vals = rng.choice(vals, size=120, replace=False)
        jitter = rng.normal(0.0, 0.052, size=vals.size)
        ax.scatter(vals, np.full(vals.size, pos) + jitter, s=4.8, color=INTERVAL_COLORS[interval], alpha=0.34, edgecolors="none", rasterized=True)
        if clipped_count:
            has_clipped = True
            ax.scatter([cap * 1.018], [pos], marker=">", s=13, color=INTERVAL_COLORS[interval], alpha=0.82, edgecolors="none", clip_on=False, zorder=4)

    ax.set_xlim(-0.02 * cap, cap * 1.08)
    ax.set_yticks(positions)
    ax.set_yticklabels(COUNT_INTERVALS, fontsize=5.8)
    ax.set_xlabel("Relative error (%)", labelpad=1.4)
    if has_clipped:
        ax.text(0.995, 0.035, "triangles: beyond axis", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.1, color="#5C5C5C")
    clean_axis(ax, "x")
    ax.grid(False, axis="y")
    return stats_rows


def draw(rows: list[dict[str, object]]) -> dict[str, object]:
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
            "font.size": 6.7,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.9,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "axes.linewidth": 0.62,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.05, 6.65), dpi=240)
    fig.subplots_adjust(left=0.082, right=0.985, top=0.94, bottom=0.132, wspace=0.25, hspace=0.56)

    summary: dict[str, object] = {
        "figure": "Application-metric consistency",
        "scope": "n(M)>100 held-out test samples",
        "n_samples": len(rows),
        "data_sources": {
            "per_sample_application_metrics": str(SOURCE_METRICS),
            "application_metrics_summary": str(SOURCE_SUMMARY),
        },
    }

    ax = axes[0, 0]
    center_stats = confusion_panel(ax, rows, "true_center_6db_mhz", "pred_center_6db_mhz", "center", "True (MHz)", "Predicted (MHz)")
    panel_label(ax, "a", r"6-dB center frequency")
    ax = axes[0, 1]
    center_err_rows = discrete_error_panel(ax, rows, "true_center_6db_mhz", "pred_center_6db_mhz")
    panel_label(ax, "b", r"Center-frequency error")

    ax = axes[1, 0]
    fbw_stats = confusion_panel(ax, rows, "true_fbw_6db", "pred_fbw_6db", "fbw", "True", "Predicted")
    panel_label(ax, "c", r"6-dB fractional bandwidth")
    ax = axes[1, 1]
    fbw_err_rows = discrete_error_panel(ax, rows, "true_fbw_6db", "pred_fbw_6db")
    panel_label(ax, "d", r"Fractional-bandwidth error")

    ax = axes[2, 0]
    peak_stats = scatter_panel(
        ax,
        rows,
        lambda r: math.log10(max(float(r["true_peak_pa"]), 1e-12)),
        lambda r: math.log10(max(float(r["pred_peak_pa"]), 1e-12)),
        r"True $\log_{10}P_{\mathrm{peak}}$",
        r"Pred. $\log_{10}P_{\mathrm{peak}}$",
        "log",
    )
    panel_label(ax, "e", r"Peak response")
    ax = axes[2, 1]
    peak_err_rows = relative_distribution_panel(ax, rows, "true_peak_pa", "pred_peak_pa", cap=12.0, rng_seed=20260528)
    panel_label(ax, "f", r"Peak-response relative error")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=3.8,
            markerfacecolor=INTERVAL_COLORS[interval],
            markeredgecolor="white",
            markeredgewidth=0.25,
            alpha=0.88,
            label=interval,
        )
        for interval in COUNT_INTERVALS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.018),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.32,
        fontsize=5.8,
        title="Active-count interval",
        title_fontsize=6.2,
        frameon=False,
    )

    summary["center_6db"] = center_stats
    summary["fbw_6db"] = fbw_stats
    summary["peak_log"] = peak_stats

    for ext in ["pdf", "svg", "png"]:
        path = EXPORT_DIR / f"fig4_application_metric_consistency.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if ext == "png":
            kwargs["dpi"] = 500
        fig.savefig(path, **kwargs)
    plt.close(fig)

    write_csv(
        DATA_DIR / "center_frequency_error_categories.csv",
        center_err_rows,
        ["active_count_interval", "n_samples", "exact_pct", "one_bin_pct", "two_plus_bins_pct"],
    )
    write_csv(
        DATA_DIR / "fractional_bandwidth_error_categories.csv",
        fbw_err_rows,
        ["active_count_interval", "n_samples", "exact_pct", "one_bin_pct", "two_plus_bins_pct"],
    )
    write_csv(
        DATA_DIR / "peak_response_relative_error_by_interval.csv",
        peak_err_rows,
        ["active_count_interval", "n_samples", "mean_rel_err_pct", "median_rel_err_pct", "p95_rel_err_pct", "max_rel_err_pct"],
    )
    with (DATA_DIR / "metric_consistency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (DATA_DIR / "DATA_SOURCE.md").write_text(
        "\n".join(
            [
                "# Data Source",
                "",
                "This figure is regenerated from the latest fixed held-out test artifacts and contains no illustrative or synthetic values.",
                "",
                f"- Per-sample metrics: `{SOURCE_METRICS}`",
                f"- Upstream summary: `{SOURCE_SUMMARY}`",
                "- Scope: samples with active count n(M)>100, matching the physical-metric consistency analysis where weak-response sparse masks are excluded.",
                "",
                "Panels (a,c) are true-vs-predicted confusion matrices for discrete 6-dB metrics.",
                "Panels (b,d) show active-count grouped error categories for the same discrete metrics.",
                "Panel (e) shows peak-response consistency on a log10 scale.",
                "Panel (f) shows active-count grouped peak-response relative-error distributions.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def draw_single_column(rows: list[dict[str, object]], suffix: str = "", large: bool = False) -> None:
    params = {
        "font.size": 6.7,
        "axes.labelsize": 6.5,
        "axes.titlesize": 6.9,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "axes.linewidth": 0.62,
    }
    if large:
        params = {
            "font.size": 7.45,
            "axes.labelsize": 7.25,
            "axes.titlesize": 7.65,
            "xtick.labelsize": 6.55,
            "ytick.labelsize": 6.55,
            "axes.linewidth": 0.68,
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

    figsize = (7.05, 6.65)
    adjust = {"left": 0.058, "right": 0.996, "top": 0.965, "bottom": 0.104, "wspace": 0.10, "hspace": 0.36}
    legend = {"markersize": 4.4, "y": 0.008, "fontsize": 6.4, "title_fontsize": 6.8}
    if large:
        figsize = (5.62, 6.10)
        adjust = {"left": 0.052, "right": 0.998, "top": 0.964, "bottom": 0.137, "wspace": 0.082, "hspace": 0.43}
        legend = {"markersize": 4.95, "y": 0.014, "fontsize": 7.1, "title_fontsize": 7.5}

    fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=240)
    fig.subplots_adjust(**adjust)

    ax = axes[0, 0]
    confusion_panel(
        ax,
        rows,
        "true_center_6db_mhz",
        "pred_center_6db_mhz",
        "center",
        "True (MHz)",
        "Predicted (MHz)",
        stat_loc=("lower_right" if large else "upper_left"),
    )
    panel_label(ax, "a", r"6-dB center frequency")
    ax = axes[0, 1]
    discrete_error_panel(ax, rows, "true_center_6db_mhz", "pred_center_6db_mhz")
    panel_label(ax, "b", r"Center-frequency error")

    ax = axes[1, 0]
    confusion_panel(
        ax,
        rows,
        "true_fbw_6db",
        "pred_fbw_6db",
        "fbw",
        "True",
        "Predicted",
        stat_loc=("lower_right" if large else "upper_left"),
    )
    panel_label(ax, "c", r"6-dB fractional bandwidth")
    ax = axes[1, 1]
    discrete_error_panel(ax, rows, "true_fbw_6db", "pred_fbw_6db")
    panel_label(ax, "d", r"Fractional-bandwidth error")

    ax = axes[2, 0]
    scatter_panel(
        ax,
        rows,
        lambda r: math.log10(max(float(r["true_peak_pa"]), 1e-12)),
        lambda r: math.log10(max(float(r["pred_peak_pa"]), 1e-12)),
        r"True $\log_{10}P_{\mathrm{peak}}$",
        r"Pred. $\log_{10}P_{\mathrm{peak}}$",
        "log",
    )
    panel_label(ax, "e", r"Peak response")

    ax = axes[2, 1]
    relative_distribution_panel(ax, rows, "true_peak_pa", "pred_peak_pa", cap=12.0, rng_seed=20260528)
    panel_label(ax, "f", r"Peak-response relative error")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=legend["markersize"],
            markerfacecolor=INTERVAL_COLORS[interval],
            markeredgecolor="white",
            markeredgewidth=0.25,
            alpha=0.88,
            label=interval,
        )
        for interval in COUNT_INTERVALS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, legend["y"]),
        ncol=4,
        columnspacing=1.08,
        handletextpad=0.36,
        fontsize=legend["fontsize"],
        title="Active-count interval",
        title_fontsize=legend["title_fontsize"],
        frameon=False,
    )

    for ext in ["pdf", "svg", "png"]:
        name = "fig4_application_metric_consistency" + (f"_{suffix}" if suffix else "")
        path = EXPORT_DIR / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 500
        fig.savefig(path, **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true", help="Write the wider exploratory export as fig4_application_metric_consistency.*.")
    parser.add_argument("--singlecol-only", action="store_true", help="Alias for the manuscript export.")
    parser.add_argument("--singlecol-large-only", action="store_true", help="Alias for the manuscript export.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if args.wide:
        summary = draw(rows)
    elif args.singlecol_only:
        draw_single_column(rows, large=True)
        summary = {"figure": "Application-metric consistency manuscript export", "n_samples": len(rows)}
    elif args.singlecol_large_only:
        draw_single_column(rows, large=True)
        summary = {"figure": "Application-metric consistency manuscript export", "n_samples": len(rows)}
    else:
        draw_single_column(rows, large=True)
        summary = {"figure": "Application-metric consistency manuscript export", "n_samples": len(rows)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
