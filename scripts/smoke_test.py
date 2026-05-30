#!/usr/bin/env python
"""Quick dataset and split sanity checks for the full-range PMUT forward task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pmut_forward import train_2d_unet as base


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-test PMUT forward data loading and deterministic split.")
    p.add_argument("--data_dir", type=Path, default=base.DEFAULT_DATA_DIR)
    p.add_argument("--cache_path", type=Path, default=base.DEFAULT_CACHE_DIR)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    x, y, meta = base.load_dataset(
        data_dir=args.data_dir,
        input_key="mask_rand",
        target_key="Ptot",
        complex_mode="magnitude",
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        max_samples=None,
        min_active_count=0,
        max_active_count=None,
    )
    train_idx, val_idx, test_idx = base.split_indices(
        n=x.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    summary = {
        "n_samples": int(x.shape[0]),
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_seed": int(args.split_seed),
        "x_finite": bool(np.isfinite(x).all()),
        "y_finite": bool(np.isfinite(y).all()),
        "cache_meta": {
            "input_key": meta.get("input_key"),
            "target_key": meta.get("target_key"),
            "complex_mode": meta.get("complex_mode"),
            "n_samples": meta.get("n_samples"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["n_samples"] != 10000:
        raise SystemExit(f"Expected 10000 samples, got {summary['n_samples']}")
    if summary["input_shape"][1:] != [1, 50, 50]:
        raise SystemExit(f"Unexpected input shape: {summary['input_shape']}")
    if summary["target_shape"][1:] != [23, 50, 50]:
        raise SystemExit(f"Unexpected target shape: {summary['target_shape']}")


if __name__ == "__main__":
    main()
