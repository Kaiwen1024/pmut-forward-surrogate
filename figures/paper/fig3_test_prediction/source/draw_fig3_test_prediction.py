#!/usr/bin/env python3
"""Create Fig. 3 from the final held-out test predictions.

The figure is designed as a single-column grouped comparison. Each active-count
group is a compact 2 x 3 grid: two frequencies by ground truth, prediction, and
normalized absolute error.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source"
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "export"

EVAL_DIR = Path("./results/fullrange_10k_retrain/eval_test")
TRUE_PATH = EVAL_DIR / "y_test_true.npy"
PRED_PATH = EVAL_DIR / "y_test_pred.npy"
INDICES_PATH = EVAL_DIR / "test_global_indices.npy"
SUMMARY_PATH = EVAL_DIR / "summary.json"
X_PATH = Path("./data/fullrange_cache/x.npy")

FREQ_INDICES = [11, 22]
FREQ_LABELS = ["3.55 MHz", "12.50 MHz"]
FIELD_UNIT = "Pa"
COUNT_TARGETS = [
    {"name": "Low", "target": 150, "lower": None},
    {"name": "Medium", "target": 1020, "lower": 1001},
    {"name": "High", "target": 2050, "lower": 2001},
]


def compute_per_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    se = (y_pred.astype(np.float64) - y_true.astype(np.float64)) ** 2
    per_sample_rmse = np.sqrt(se.mean(axis=(1, 2, 3)))
    per_sample_r2 = np.empty(y_true.shape[0], dtype=np.float64)
    for i in range(y_true.shape[0]):
        ss_res = float(np.sum((y_pred[i] - y_true[i]) ** 2))
        ss_tot = float(np.sum((y_true[i] - np.mean(y_true[i])) ** 2))
        per_sample_r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return per_sample_rmse, per_sample_r2


def select_density_samples(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[dict], dict]:
    if not X_PATH.exists():
        raise FileNotFoundError(f"Missing cached input layouts: {X_PATH}")
    if not INDICES_PATH.exists():
        raise FileNotFoundError(f"Missing test global indices: {INDICES_PATH}")

    x_all = np.load(X_PATH, mmap_mode="r")
    test_indices = np.load(INDICES_PATH)
    active_counts = x_all[test_indices].reshape(len(test_indices), -1).sum(axis=1).astype(int)
    per_sample_rmse, per_sample_r2 = compute_per_sample_metrics(y_true, y_pred)

    selected = []
    used = set()
    for spec in COUNT_TARGETS:
        valid = np.ones_like(active_counts, dtype=bool)
        if spec["lower"] is not None:
            valid &= active_counts >= int(spec["lower"])
        else:
            valid &= active_counts != int(spec["target"])
        for sample_idx in np.argsort(np.where(valid, np.abs(active_counts - int(spec["target"])), np.inf)):
            sample_idx = int(sample_idx)
            if sample_idx in used:
                continue
            selected.append(
                {
                    "density": spec["name"],
                    "sample_index_in_test": sample_idx,
                    "global_index": int(test_indices[sample_idx]),
                    "active_count": int(active_counts[sample_idx]),
                    "field_rmse": float(per_sample_rmse[sample_idx]),
                    "field_r2": float(per_sample_r2[sample_idx]),
                    "target_active_count": int(spec["target"]),
                }
            )
            used.add(sample_idx)
            break

    metrics_summary = {
        "all_sample_rmse_summary": {
            "min": float(np.min(per_sample_rmse)),
            "q25": float(np.quantile(per_sample_rmse, 0.25)),
            "median": float(np.median(per_sample_rmse)),
            "q75": float(np.quantile(per_sample_rmse, 0.75)),
            "max": float(np.max(per_sample_rmse)),
        },
        "active_count_summary": {
            "min": int(np.min(active_counts)),
            "median": float(np.median(active_counts)),
            "max": int(np.max(active_counts)),
        },
    }
    return selected, metrics_summary


def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


def save_metadata(selected: list[dict], metrics_summary: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    metadata = {
        "figure": "Fig. 3 Test-Set Prediction Example",
        "selection_rule": "held-out test samples closest to active-count targets: low around 150 but not exactly 150, medium slightly above 1000, and high above 2000",
        "data_sources": {
            "y_test_true": str(TRUE_PATH),
            "y_test_pred": str(PRED_PATH),
            "x": str(X_PATH),
            "summary": str(SUMMARY_PATH),
            "test_global_indices": str(INDICES_PATH),
        },
        "samples": selected,
        "frequencies": {
            "indices": FREQ_INDICES,
            "labels": FREQ_LABELS,
        },
        "unit": {
            "field_magnitude": FIELD_UNIT,
            "absolute_error": FIELD_UNIT,
            "displayed_error": "normalized absolute error",
            "provenance": "Confirmed by the project advisor and reported by the user on 2026-05-26.",
        },
        "selection_metrics_summary": metrics_summary,
        "test_metrics_snapshot": {
            "evaluated_samples": summary.get("evaluated_samples"),
            "pointwise_metrics_denorm": summary.get("pointwise_metrics_denorm"),
            "pointwise_metrics_standardized": summary.get("pointwise_metrics_standardized"),
            "speed_metrics": summary.get("speed_metrics"),
        },
        "note": "Each active-count group is a 2 x 3 grid: two frequencies by ground truth, prediction, and normalized absolute error. Colors are normalized independently within each frequency row for visualization; numbers in error panels are RMSE values normalized by the field dynamic range of the corresponding frequency row and shown as percentages.",
    }
    (DATA_DIR / "fig3_selected_samples.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def draw_figure(y_true: np.ndarray, y_pred: np.ndarray, selected: list[dict]):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
            "axes.linewidth": 0.6,
        }
    )

    def normalize_panel(arr: np.ndarray, low: float, high: float) -> np.ndarray:
        high = max(high, low + 1e-12)
        return np.clip((arr.astype(np.float64) - low) / (high - low), 0.0, 1.0)

    fig_w, fig_h = 3.55, 5.18
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=240)
    col_titles = ["Ground Truth", "Prediction", "Normalized\nAbsolute Error"]

    image_s = 0.680
    col_gap = 0.045
    row_gap = 0.050
    group_title_h = 0.125
    group_gap = 0.120
    top_margin = 0.135
    bottom_margin = 0.050
    label_x = 0.032
    grid_x = 0.405
    cbar_gap = 0.065
    cbar_w = 0.032
    cbar_x = grid_x + 3 * image_s + 2 * col_gap + cbar_gap
    grid_w = 3 * image_s + 2 * col_gap
    n_freq = len(FREQ_INDICES)
    grid_h = n_freq * image_s + (n_freq - 1) * row_gap
    group_h = group_title_h + grid_h

    for c, title in enumerate(col_titles):
        x = grid_x + c * (image_s + col_gap) + image_s / 2
        fig.text(x / fig_w, 1.0 - 0.052 / fig_h, title, ha="center", va="center", fontsize=6.5, fontweight="bold")

    for group_i, sample in enumerate(selected):
        group_top = fig_h - top_margin - group_i * (group_h + group_gap)
        title_y = group_top - group_title_h / 2
        grid_top = group_top - group_title_h
        group_tag = ["(a)", "(b)", "(c)"][group_i] if group_i < 3 else f"({chr(ord('a') + group_i)})"
        fig.text(
            grid_x / fig_w,
            title_y / fig_h,
            f"{group_tag} {sample['density']} density, n={sample['active_count']}",
            ha="left",
            va="center",
            fontsize=5.2,
            fontweight="normal",
        )

        for row_i, (ch, freq_label) in enumerate(zip(FREQ_INDICES, FREQ_LABELS)):
            idx = int(sample["sample_index_in_test"])
            true_field = np.asarray(y_true[idx, ch], dtype=np.float64)
            pred_field = np.asarray(y_pred[idx, ch], dtype=np.float64)
            err = np.abs(pred_field - true_field)
            field_low = float(np.percentile(np.r_[true_field.ravel(), pred_field.ravel()], 0.5))
            field_high = float(np.percentile(np.r_[true_field.ravel(), pred_field.ravel()], 99.5))
            field_range = max(field_high - field_low, 1e-12)
            err_high = max(float(np.percentile(err, 99.0)), 1e-12)
            display_panels = [
                normalize_panel(true_field, field_low, field_high),
                normalize_panel(pred_field, field_low, field_high),
                np.clip(err / err_high, 0.0, 1.0),
            ]
            row_bottom = grid_top - (row_i + 1) * image_s - row_i * row_gap
            row_center = row_bottom + image_s / 2
            fig.text(label_x / fig_w, row_center / fig_h, freq_label.replace(" ", "\n"), ha="left", va="center", fontsize=5.3)

            for col_i, arr in enumerate(display_panels):
                col_left = grid_x + col_i * (image_s + col_gap)
                ax = fig.add_axes([col_left / fig_w, row_bottom / fig_h, image_s / fig_w, image_s / fig_h])
                if col_i < 2:
                    ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
                else:
                    ax.imshow(arr, cmap="Blues", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
                    row_rmse = float(np.sqrt(np.mean((pred_field - true_field) ** 2)))
                    row_nrmse_pct = 100.0 * row_rmse / field_range
                    rmse_label = f"{row_nrmse_pct:.2f}%"
                    ax.text(
                        0.97,
                        0.04,
                        rmse_label,
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=4.7,
                        color="#1F2933",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.45},
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.32)
                    spine.set_edgecolor("#8B98A5")

        field_sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=1))
        err_sm = mpl.cm.ScalarMappable(cmap="Blues", norm=mpl.colors.Normalize(vmin=0, vmax=1))
        field_cax = fig.add_axes(
            [
                cbar_x / fig_w,
                (grid_top - image_s) / fig_h,
                cbar_w / fig_w,
                image_s / fig_h,
            ]
        )
        err_cax = fig.add_axes(
            [
                cbar_x / fig_w,
                (grid_top - n_freq * image_s - (n_freq - 1) * row_gap) / fig_h,
                cbar_w / fig_w,
                image_s / fig_h,
            ]
        )
        cb1 = fig.colorbar(field_sm, cax=field_cax)
        cb2 = fig.colorbar(err_sm, cax=err_cax)
        for cb in [cb1, cb2]:
            cb.set_ticks([0.0, 0.5, 1.0])
            cb.ax.tick_params(labelsize=3.9, width=0.35, length=1.2)
        cb1.set_label("Field", fontsize=4.1)
        cb2.set_label("Error", fontsize=4.1)

    fig.savefig(EXPORT_DIR / "fig3_test_prediction.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(EXPORT_DIR / "fig3_test_prediction.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(EXPORT_DIR / "fig3_test_prediction.png", dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    missing = [str(p) for p in [TRUE_PATH, PRED_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required test prediction files: " + ", ".join(missing))
    y_true = np.load(TRUE_PATH, mmap_mode="r")
    y_pred = np.load(PRED_PATH, mmap_mode="r")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true {y_true.shape}, pred {y_pred.shape}")

    selected, metrics_summary = select_density_samples(y_true, y_pred)
    save_metadata(selected, metrics_summary)
    draw_figure(y_true, y_pred, selected)


if __name__ == "__main__":
    main()
