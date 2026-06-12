#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D U-Net surrogate model for PMUT acoustic field prediction.

Task:
    mask_rand (50x50) -> |Ptot| (50x50x23)

This script is intentionally self-contained so it can be smoke-tested locally
and then run on a GPU server for full training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = Path("data") / "raw_mat"
DEFAULT_CACHE_DIR = Path("data") / "fullrange_cache"


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def configure_torch(fast_math: bool = False, cudnn_benchmark: bool = False) -> None:
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if not fast_math:
        return
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _is_numeric_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _mat_payload(mat_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if _is_numeric_array(v):
            payload[k] = v
    return payload


def inspect_first_sample(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    sample = sio.loadmat(files[0])
    payload = _mat_payload(sample)
    print(f"[Inspect] file: {files[0].name}")
    for key, value in payload.items():
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, size={value.size}")


def _complex_to_real(arr: np.ndarray, complex_mode: str) -> np.ndarray:
    arr_np = np.asarray(arr)
    if np.iscomplexobj(arr_np):
        if complex_mode == "magnitude":
            arr_np = np.abs(arr_np)
        elif complex_mode == "real":
            arr_np = np.real(arr_np)
        elif complex_mode == "imag":
            arr_np = np.imag(arr_np)
        else:
            raise ValueError(f"Unsupported complex_mode: {complex_mode}")
    return np.asarray(arr_np, dtype=np.float32)


def _prepare_input_tensor(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {x.shape}")
    return x[None, :, :]


def _prepare_target_tensor(arr: np.ndarray, complex_mode: str) -> np.ndarray:
    y = _complex_to_real(arr, complex_mode=complex_mode)
    if y.ndim == 2:
        return y[None, :, :]
    if y.ndim != 3:
        raise ValueError(f"Expected 3D target array, got shape {y.shape}")

    # Prefer channels-last MATLAB field layout: H x W x C.
    if y.shape[0] == y.shape[1]:
        return np.transpose(y, (2, 0, 1)).astype(np.float32)

    # If channels-first already, keep it.
    if y.shape[1] == y.shape[2]:
        return y.astype(np.float32)

    raise ValueError(f"Unable to infer target channel layout from shape {y.shape}")


def _cache_meta_matches(
    meta: Dict[str, Any],
    input_key: str,
    target_key: str,
    complex_mode: str,
    max_samples: Optional[int],
    min_active_count: int,
    max_active_count: Optional[int],
) -> bool:
    return (
        meta.get("input_key") == input_key
        and meta.get("target_key") == target_key
        and meta.get("complex_mode") == complex_mode
        and meta.get("max_samples") == max_samples
        and int(meta.get("min_active_count", 1)) == int(min_active_count)
        and meta.get("max_active_count") == max_active_count
    )


def _load_split_cache(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    x = np.load(cache_dir / "x.npy", mmap_mode=None).astype(np.float32)
    y_parts = [
        np.load(path, mmap_mode=None).astype(np.float32)
        for path in sorted(cache_dir.glob("y_part_*.npy"))
    ]
    if not y_parts:
        raise FileNotFoundError(f"No y_part_*.npy files found in {cache_dir}")
    y = np.concatenate(y_parts, axis=0)
    return x, y, meta


def _save_split_cache(cache_dir: Path, x: np.ndarray, y: np.ndarray, meta: Dict[str, Any], max_part_bytes: int) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "x.npy", x.astype(np.float32))
    sample_bytes = int(np.prod(y.shape[1:]) * np.dtype(np.float32).itemsize)
    samples_per_part = max(1, max_part_bytes // max(sample_bytes, 1))
    part_files: List[str] = []
    for part_id, start in enumerate(range(0, y.shape[0], samples_per_part)):
        end = min(start + samples_per_part, y.shape[0])
        name = f"y_part_{part_id:03d}.npy"
        np.save(cache_dir / name, y[start:end].astype(np.float32))
        part_files.append(name)
    meta_out = dict(meta)
    meta_out["cache_format"] = "split-npy-v1"
    meta_out["x_file"] = "x.npy"
    meta_out["y_part_files"] = part_files
    meta_out["max_part_bytes"] = int(max_part_bytes)
    (cache_dir / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(
    data_dir: Path,
    input_key: str,
    target_key: str,
    complex_mode: str = "magnitude",
    cache_path: Optional[Path] = None,
    rebuild_cache: bool = False,
    max_samples: Optional[int] = None,
    min_active_count: int = 1,
    max_active_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        if cache_path.is_dir():
            x_cached, y_cached, meta = _load_split_cache(cache_path)
            if _cache_meta_matches(meta, input_key, target_key, complex_mode, max_samples, min_active_count, max_active_count):
                print(f"[Data] Loaded split cached dataset: {cache_path}")
                return x_cached, y_cached, meta
        else:
            cached = np.load(cache_path, allow_pickle=False)
            meta = json.loads(str(cached["meta"].item()))
            if _cache_meta_matches(meta, input_key, target_key, complex_mode, max_samples, min_active_count, max_active_count):
                print(f"[Data] Loaded cached dataset: {cache_path}")
                return cached["x"].astype(np.float32), cached["y"].astype(np.float32), meta
        print(f"[Data] Cache metadata mismatch, rebuilding: {cache_path}")

    files_all = sorted(data_dir.glob("*.mat"))
    if not files_all:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    file_entries = list(enumerate(files_all))
    if max_samples is not None:
        file_entries = file_entries[:max_samples]

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    first_meta: Optional[Dict[str, Any]] = None
    source_indices: List[int] = []

    for idx, (source_idx, file_path) in enumerate(file_entries, start=1):
        payload = _mat_payload(sio.loadmat(file_path))
        if input_key not in payload:
            raise KeyError(f"{file_path.name}: missing input_key '{input_key}'")
        if target_key not in payload:
            raise KeyError(f"{file_path.name}: missing target_key '{target_key}'")

        x = _prepare_input_tensor(payload[input_key])
        y = _prepare_target_tensor(payload[target_key], complex_mode=complex_mode)
        xs.append(x)
        ys.append(y)
        source_indices.append(int(source_idx))

        if first_meta is None:
            first_meta = {
                "input_shape": list(x.shape),
                "target_shape": list(y.shape),
            }
        if idx % 100 == 0 or idx == len(file_entries):
            print(f"[Data] loaded {idx}/{len(file_entries)}")

    x_arr = np.stack(xs).astype(np.float32)
    y_arr = np.stack(ys).astype(np.float32)
    source_idx_arr = np.asarray(source_indices, dtype=np.int64)
    active_counts = np.rint(x_arr.reshape(x_arr.shape[0], -1).sum(axis=1)).astype(np.int64)
    keep = active_counts >= int(min_active_count)
    if max_active_count is not None:
        keep &= active_counts <= int(max_active_count)
    if not np.any(keep):
        raise ValueError(
            f"No samples remain after filtering active_count in "
            f"[{min_active_count}, {max_active_count if max_active_count is not None else 'inf'}]."
        )
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    source_idx_arr = source_idx_arr[keep]
    active_counts = active_counts[keep]
    meta = {
        "input_key": input_key,
        "target_key": target_key,
        "complex_mode": complex_mode,
        "max_samples": max_samples,
        "min_active_count": int(min_active_count),
        "max_active_count": None if max_active_count is None else int(max_active_count),
        "n_samples": int(x_arr.shape[0]),
        "n_samples_before_filter": int(len(file_entries)),
        "input_shape": list(x_arr.shape[1:]),
        "target_shape": list(y_arr.shape[1:]),
        "field_hw": list(y_arr.shape[-2:]),
        "n_output_channels": int(y_arr.shape[1]),
        "active_count_min_retained": int(active_counts.min()),
        "active_count_max_retained": int(active_counts.max()),
        "source_indices": [int(v) for v in source_idx_arr.tolist()],
    }
    if first_meta is not None:
        meta.update(first_meta)

    if cache_path is not None:
        if cache_path.suffix:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, x=x_arr, y=y_arr, meta=json.dumps(meta, ensure_ascii=False))
        else:
            _save_split_cache(cache_path, x_arr, y_arr, meta, max_part_bytes=1_800_000_000)
        print(f"[Data] Saved cached dataset: {cache_path}")

    if not np.isfinite(x_arr).all():
        raise ValueError("Input tensor contains non-finite values.")
    if not np.isfinite(y_arr).all():
        raise ValueError("Target tensor contains non-finite values.")

    return x_arr, y_arr, meta


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Require 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


@dataclass
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> "ChannelStandardizer":
        mean = y.mean(axis=(0, 2, 3), keepdims=True)
        std = y.std(axis=(0, 2, 3), keepdims=True)
        std[std < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


class ArrayFieldDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 23,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels: List[int] = []

        in_ch = in_channels
        out_ch = base_channels
        for _ in range(depth):
            self.encoders.append(DoubleConv(in_ch, out_ch, dropout=dropout))
            channels.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)

        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(depth - 1)])
        self.up_blocks = nn.ModuleList()
        self.up_reduce = nn.ModuleList()

        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.up_reduce.append(nn.Conv2d(decoder_in, skip_ch, kernel_size=1))
            self.up_blocks.append(DoubleConv(skip_ch * 2, skip_ch, dropout=dropout))
            decoder_in = skip_ch

        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for idx, encoder in enumerate(self.encoders):
            h = encoder(h)
            if idx < len(self.encoders) - 1:
                skips.append(h)
                h = self.pools[idx](h)

        for reduce_conv, up_block, skip in zip(self.up_reduce, self.up_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = up_block(h)

        return self.head(h)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    ds = ArrayFieldDataset(x, y)
    kwargs: Dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def parse_device_ids(device_ids: Optional[str]) -> Optional[List[int]]:
    if device_ids is None or device_ids.strip() == "":
        return None
    return [int(part.strip()) for part in device_ids.split(",") if part.strip()]


def maybe_parallel(model: nn.Module, use_multi_gpu: bool, device_ids: Optional[List[int]]) -> nn.Module:
    if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))
        if len(ids) > 1:
            print(f"[Device] DataParallel enabled on CUDA devices: {ids}")
            return nn.DataParallel(model, device_ids=ids)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {name}")


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    use_fused: bool = False,
) -> torch.optim.Optimizer:
    kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if use_fused:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def maybe_compile_model(
    model: nn.Module,
    enable_compile: bool,
    compile_mode: str,
    dynamic: bool = False,
) -> nn.Module:
    if not enable_compile:
        return model
    if isinstance(model, nn.DataParallel):
        print("[Compile] skip torch.compile because DataParallel is enabled")
        return model
    if not hasattr(torch, "compile"):
        print("[Compile] skip torch.compile because current torch has no compile()")
        return model
    print(f"[Compile] enabling torch.compile(mode={compile_mode}, dynamic={dynamic})")
    return torch.compile(model, mode=compile_mode, dynamic=dynamic)


def maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        xb = maybe_channels_last(xb, channels_last)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = criterion(pred.float(), yb.float())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss detected.")
        bs = xb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    mse = total_loss / max(n_samples, 1)
    return {"loss": mse, "mse": mse, "rmse": math.sqrt(max(mse, 0.0))}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    amp: bool = False,
    amp_dtype: str = "bfloat16",
    channels_last: bool = False,
    use_fused_optimizer: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    criterion = nn.MSELoss()
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay, use_fused=use_fused_optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 2)
    )
    use_amp = amp and device.type == "cuda"
    autocast_dtype = resolve_amp_dtype(amp_dtype)
    use_grad_scaler = use_amp and autocast_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "lr": [],
        "train_sec": [],
        "val_sec": [],
        "train_samples_per_sec": [],
        "val_samples_per_sec": [],
    }
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        train_start = time.perf_counter()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            xb = maybe_channels_last(xb, channels_last)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
            with autocast_ctx:
                pred = model(xb)
            loss = criterion(pred.float(), yb.float())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. "
                    f"pred_abs_max={pred.detach().abs().max().item():.6e}, "
                    f"target_abs_max={yb.detach().abs().max().item():.6e}"
                )
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            seen += bs

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_sec = time.perf_counter() - train_start
        train_loss = total_loss / max(seen, 1)
        val_start = time.perf_counter()
        val_stats = evaluate(model, val_loader, device, criterion, channels_last=channels_last)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_sec = time.perf_counter() - val_start
        val_loss = val_stats["loss"]
        val_rmse = val_stats["rmse"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_sec"].append(train_sec)
        history["val_sec"].append(val_sec)
        history["train_samples_per_sec"].append(seen / max(train_sec, 1e-12))
        history["val_samples_per_sec"].append(len(val_loader.dataset) / max(val_sec, 1e-12))

        print(
            f"[Epoch {epoch:03d}] train={train_loss:.6f} "
            f"val={val_loss:.6f} val_rmse={val_rmse:.6f} "
            f"train_sec={train_sec:.2f} val_sec={val_sec:.2f} "
            f"train_sps={history['train_samples_per_sec'][-1]:.1f} "
            f"val_sps={history['val_samples_per_sec'][-1]:.1f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[Train] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def predict_numpy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool = False,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb = maybe_channels_last(xb, channels_last)
        pred = model(xb).cpu().numpy()
        outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def benchmark_training(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    amp: bool = False,
    amp_dtype: str = "bfloat16",
    channels_last: bool = False,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
    use_fused_optimizer: bool = False,
) -> Dict[str, Any]:
    criterion = nn.MSELoss()
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay, use_fused=use_fused_optimizer)
    use_amp = amp and device.type == "cuda"
    autocast_dtype = resolve_amp_dtype(amp_dtype)
    use_grad_scaler = use_amp and autocast_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    model.train()
    iterator = iter(loader)
    total_compute_sec = 0.0
    total_fetch_sec = 0.0
    total_samples = 0
    measured_losses: List[float] = []
    measured_steps = warmup_steps + benchmark_steps

    if measured_steps <= 0:
        raise ValueError("warmup_steps + benchmark_steps must be > 0")

    for step in range(measured_steps):
        fetch_start = time.perf_counter()
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        fetch_sec = time.perf_counter() - fetch_start

        xb = xb.to(device, non_blocking=True)
        xb = maybe_channels_last(xb, channels_last)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
        with autocast_ctx:
            pred = model(xb)
        loss = criterion(pred.float(), yb.float())
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite benchmark loss detected. "
                f"pred_abs_max={pred.detach().abs().max().item():.6e}, "
                f"target_abs_max={yb.detach().abs().max().item():.6e}"
            )
        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_sec = time.perf_counter() - compute_start

        if step >= warmup_steps:
            total_fetch_sec += fetch_sec
            total_compute_sec += compute_sec
            total_samples += int(xb.size(0))
            measured_losses.append(float(loss.item()))

    gpu_count = len(getattr(model, "device_ids", [])) if isinstance(model, nn.DataParallel) else (1 if device.type == "cuda" else 0)
    batch_size = loader.batch_size or 0
    return {
        "device": str(device),
        "gpu_count": int(gpu_count),
        "batch_size": int(batch_size),
        "per_gpu_batch_estimate": int(math.ceil(batch_size / max(gpu_count, 1))) if gpu_count > 0 else int(batch_size),
        "warmup_steps": int(warmup_steps),
        "benchmark_steps": int(benchmark_steps),
        "mean_fetch_sec": total_fetch_sec / max(benchmark_steps, 1),
        "mean_step_sec": total_compute_sec / max(benchmark_steps, 1),
        "samples_per_sec": total_samples / max(total_compute_sec, 1e-12),
        "mean_loss": float(np.mean(measured_losses)) if measured_losses else float("nan"),
    }


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def plot_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_rmse"], label="val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_rmse_curve.png", dpi=180)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train 2D U-Net surrogate for PMUT acoustic field prediction.")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--input_key", type=str, default="mask_rand")
    p.add_argument("--target_key", type=str, default="Ptot")
    p.add_argument("--complex_mode", type=str, default="magnitude", choices=["magnitude", "real", "imag"])
    p.add_argument("--cache_path", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--min_active_count", type=int, default=0)
    p.add_argument("--max_active_count", type=int, default=None)
    p.add_argument("--inspect_only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--fast_math", action="store_true")
    p.add_argument("--cudnn_benchmark", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile_dynamic", action="store_true")
    p.add_argument("--fused_optimizer", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--device_ids", type=str, default=None)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--benchmark_warmup_steps", type=int, default=10)
    p.add_argument("--benchmark_steps", type=int, default=50)
    p.add_argument("--out_dir", type=Path, default=Path("forward") / "outputs" / "legacy_train")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    configure_torch(args.fast_math, cudnn_benchmark=args.cudnn_benchmark and not args.deterministic)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        inspect_first_sample(args.data_dir)
        return

    x, y, data_meta = load_dataset(
        data_dir=args.data_dir,
        input_key=args.input_key,
        target_key=args.target_key,
        complex_mode=args.complex_mode,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        max_samples=args.max_samples,
        min_active_count=args.min_active_count,
        max_active_count=args.max_active_count,
    )

    split_seed = args.seed if args.split_seed is None else args.split_seed

    train_idx, val_idx, test_idx = split_indices(
        n=x.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=split_seed,
    )
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    source_indices = np.asarray(data_meta.get("source_indices", list(range(x.shape[0]))), dtype=np.int64)
    train_global_indices = source_indices[train_idx]
    val_global_indices = source_indices[val_idx]
    test_global_indices = source_indices[test_idx]

    y_std = ChannelStandardizer.fit(y_train)
    y_train_norm = y_std.transform(y_train)
    y_val_norm = y_std.transform(y_val)
    y_test_norm = y_std.transform(y_test)

    train_loader = make_loader(
        x_train,
        y_train_norm,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = make_loader(
        x_val,
        y_val_norm,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = make_loader(
        x_test,
        y_test_norm,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] using {device}")

    model = UNet2D(
        in_channels=int(x.shape[1]),
        out_channels=int(y.shape[1]),
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = maybe_parallel(model, args.multi_gpu, parse_device_ids(args.device_ids))
    model = maybe_compile_model(
        model,
        enable_compile=args.compile,
        compile_mode=args.compile_mode,
        dynamic=args.compile_dynamic,
    )

    if args.benchmark_only:
        bench = benchmark_training(
            model=model,
            loader=train_loader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
            warmup_steps=args.benchmark_warmup_steps,
            benchmark_steps=args.benchmark_steps,
            use_fused_optimizer=args.fused_optimizer,
        )
        bench["n_samples"] = int(x.shape[0])
        bench["split"] = {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        }
        bench["data_meta"] = data_meta
        bench["config"] = {
            "base_channels": args.base_channels,
            "depth": args.depth,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory),
            "amp": bool(args.amp),
            "amp_dtype": args.amp_dtype,
            "multi_gpu": bool(args.multi_gpu),
            "channels_last": bool(args.channels_last),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "fast_math": bool(args.fast_math),
            "compile": bool(args.compile),
            "compile_mode": args.compile_mode,
            "compile_dynamic": bool(args.compile_dynamic),
            "fused_optimizer": bool(args.fused_optimizer),
        }
        with open(args.out_dir / "benchmark.json", "w", encoding="utf-8") as f:
            json.dump(bench, f, ensure_ascii=False, indent=2)
        print(json.dumps(bench, ensure_ascii=False, indent=2))
        print(f"[Done] saved benchmark outputs to {args.out_dir}")
        return

    model, history, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        use_fused_optimizer=args.fused_optimizer,
    )

    criterion = nn.MSELoss()
    test_stats_norm = evaluate(model, test_loader, device, criterion, channels_last=args.channels_last)
    y_pred_norm = predict_numpy(model, test_loader, device, channels_last=args.channels_last)
    y_pred = y_std.inverse_transform(y_pred_norm)

    mse = float(np.mean((y_pred - y_test) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    r2 = r2_score_np(y_test.reshape(-1), y_pred.reshape(-1))

    np.save(args.out_dir / "y_test_true.npy", y_test)
    np.save(args.out_dir / "y_test_pred.npy", y_pred)
    np.save(args.out_dir / "filtered_global_indices.npy", source_indices)
    np.save(args.out_dir / "train_global_indices.npy", train_global_indices)
    np.save(args.out_dir / "val_global_indices.npy", val_global_indices)
    np.save(args.out_dir / "test_global_indices.npy", test_global_indices)

    checkpoint = {
        "state_dict": unwrap_model(model).state_dict(),
        "model_config": {
            "in_channels": int(x.shape[1]),
            "out_channels": int(y.shape[1]),
            "base_channels": args.base_channels,
            "depth": args.depth,
            "dropout": args.dropout,
        },
        "data_meta": data_meta,
        "standardizer": {"mean": y_std.mean, "std": y_std.std},
        "args": vars(args),
    }
    torch.save(checkpoint, args.out_dir / "best_2d_unet.pt")

    metrics = {
        "n_samples": int(x.shape[0]),
        "active_count_filter": {
            "min_active_count": int(args.min_active_count),
            "max_active_count": None if args.max_active_count is None else int(args.max_active_count),
        },
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "seed": int(args.seed),
        "split_seed": int(split_seed),
        "best_val_mse_norm": best_val,
        "test_metrics_denorm": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "test_metrics_norm": test_stats_norm,
        "timing": {
            "train_epoch_sec_mean": float(np.mean(history["train_sec"])) if history["train_sec"] else 0.0,
            "train_epoch_sec_last": float(history["train_sec"][-1]) if history["train_sec"] else 0.0,
            "val_epoch_sec_mean": float(np.mean(history["val_sec"])) if history["val_sec"] else 0.0,
            "val_epoch_sec_last": float(history["val_sec"][-1]) if history["val_sec"] else 0.0,
            "train_samples_per_sec_mean": float(np.mean(history["train_samples_per_sec"])) if history["train_samples_per_sec"] else 0.0,
            "train_samples_per_sec_last": float(history["train_samples_per_sec"][-1]) if history["train_samples_per_sec"] else 0.0,
            "val_samples_per_sec_mean": float(np.mean(history["val_samples_per_sec"])) if history["val_samples_per_sec"] else 0.0,
            "val_samples_per_sec_last": float(history["val_samples_per_sec"][-1]) if history["val_samples_per_sec"] else 0.0,
        },
        "data_meta": data_meta,
    }
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_curves(history, args.out_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Done] saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
