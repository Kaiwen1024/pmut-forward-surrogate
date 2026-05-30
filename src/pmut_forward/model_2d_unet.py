#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configurable 2D U-Net model for PMUT forward acoustic-field prediction."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import train_2d_unet as base
except ImportError:
    import train_2d_unet as base


def make_pool(pool_type: str) -> nn.Module:
    if pool_type == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if pool_type == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    raise ValueError(f"Unsupported pool_type: {pool_type}")


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding behavior.")
        pad = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            base.make_group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            base.make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PMUTUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 23,
        base_channels: int = 56,
        depth: int = 6,
        kernel_size: int = 5,
        pool_type: str = "avg",
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.encoders = nn.ModuleList()
        channels: List[int] = []

        in_ch = in_channels
        out_ch = base_channels
        for _ in range(depth):
            self.encoders.append(DoubleConv2D(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout))
            channels.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)

        self.pools = nn.ModuleList([make_pool(pool_type) for _ in range(depth - 1)])
        self.up_blocks = nn.ModuleList()
        self.up_reduce = nn.ModuleList()

        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.up_reduce.append(nn.Conv2d(decoder_in, skip_ch, kernel_size=1))
            self.up_blocks.append(DoubleConv2D(skip_ch * 2, skip_ch, kernel_size=kernel_size, dropout=dropout))
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
