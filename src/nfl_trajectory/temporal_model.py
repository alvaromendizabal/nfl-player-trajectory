"""Temporal convolutions, geometry-biased player attention, and continuous-time decoding.

PyTorch is an optional, independently locked research runtime. No Kaggle deployment
uses this challenger until it passes model selection and organizer gateway checks.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nfl_trajectory.runtime import atomic_bytes, atomic_json, sha256
from nfl_trajectory.temporal_data import CHANNELS, EDGE_NAMES, HISTORY, STATIC_NAMES, edge_features


def collate(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad players/history/requests independently; padding never contributes to loss."""
    batch = len(samples)
    players = max(len(s["ids"]) for s in samples)
    requests = max(len(s["time"]) for s in samples)
    shapes = {
        "history": (batch, players, HISTORY, len(CHANNELS)),
        "observed": (batch, players, HISTORY),
        "static": (batch, players, len(STATIC_NAMES)),
        "role": (batch, players),
        "side": (batch, players),
        "valid": (batch, players),
        "edges": (batch, players, players, len(EDGE_NAMES)),
        "player": (batch, requests),
        "time": (batch, requests),
        "baseline": (batch, requests, 2),
        "truth": (batch, requests, 2),
        "scored": (batch, requests),
    }
    arrays = {
        name: np.zeros(
            shape,
            dtype=np.int64
            if name in {"role", "side", "player"}
            else bool
            if name in {"observed", "valid", "scored"}
            else np.float32,
        )
        for name, shape in shapes.items()
    }
    for i, sample in enumerate(samples):
        p, q = len(sample["ids"]), len(sample["time"])
        for name in ("history", "observed", "static", "role", "side"):
            arrays[name][i, :p] = sample[name]
        arrays["valid"][i, :p] = True
        arrays["edges"][i, :p, :p] = edge_features(sample)
        for name in ("player", "time", "baseline", "truth"):
            if name in sample:
                arrays[name][i, :q] = sample[name]
        arrays["scored"][i, :q] = True
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


class Interaction(nn.Module):
    """Permutation-equivariant attention with learned biases from directed pair geometry."""

    def __init__(self, width: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.edge = nn.Sequential(nn.Linear(len(EDGE_NAMES), 32), nn.SiLU(), nn.Linear(32, heads))
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, p, _ = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(b, p, 3, self.heads, self.dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        logits = q @ k.transpose(-1, -2) / self.dim**0.5
        logits = logits + self.edge(edges).permute(0, 3, 1, 2)
        weights = logits.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(-1)
        value = (weights @ v).transpose(1, 2).reshape(b, p, -1)
        x = x + self.out(value)
        return (x + self.ff(x)) * valid[..., None]


class TemporalModel(nn.Module):
    """Predict canonical residuals above the training-only role ridge baseline."""

    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(len(CHANNELS) + 1, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=1),
            nn.GELU(),
        )
        self.role = nn.Embedding(5, 8)
        self.side = nn.Embedding(2, 4)
        self.project = nn.Sequential(
            nn.Linear(HISTORY * 48 + len(STATIC_NAMES) + 12, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.interactions = nn.ModuleList([Interaction(width), Interaction(width)])
        self.final_norm = nn.LayerNorm(width)
        output = nn.Linear(64, 2)
        self.decode = nn.Sequential(
            nn.Linear(width + len(STATIC_NAMES) + 10, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            output,
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h, observed = batch["history"], batch["observed"]
        b, p, t, _ = h.shape
        h = torch.cat([h * observed[..., None], observed[..., None].to(h.dtype)], dim=-1)
        temporal = self.temporal(h.reshape(b * p, t, -1).transpose(1, 2)).reshape(b, p, -1)
        static = batch["static"]
        hidden = (
            self.project(
                torch.cat(
                    [
                        temporal,
                        static,
                        self.role(batch["role"]),
                        self.side(batch["side"]),
                    ],
                    dim=-1,
                )
            )
            * batch["valid"][..., None]
        )
        for layer in self.interactions:
            hidden = layer(hidden, batch["edges"], batch["valid"])
        hidden = self.final_norm(hidden) * batch["valid"][..., None]
        index = batch["player"]
        rows = torch.arange(b, device=index.device)[:, None]
        selected = static[rows, index]
        time = batch["time"]
        horizon = selected[..., 6].clamp_min(0.1)
        fraction = time / horizon
        clock = torch.stack(
            [
                time / 5,
                time.square() / 25,
                fraction,
                fraction.square(),
                torch.sin(torch.pi * fraction),
                torch.cos(torch.pi * fraction),
                (horizon - time) / 5,
                torch.log1p(time),
            ],
            dim=-1,
        )
        inputs = torch.cat([hidden[rows, index], selected, clock, batch["baseline"] / 10], dim=-1)
        # A velocity correction goes to zero at the observed endpoint by construction.
        return self.decode(inputs) * time[..., None] * batch["scored"][..., None]


def coordinate_loss(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    error = (prediction - batch["truth"]).square().sum(-1)
    return (error * batch["scored"]).sum() / (2 * batch["scored"].sum().clamp_min(1))


def save_checkpoint(path: Path, payload: dict[str, Any], signature: str) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())
    atomic_json(path.with_suffix(".json"), {"signature": signature, "sha256": sha256(path)})


def load_checkpoint(path: Path, signature: str) -> dict[str, Any]:
    """Only load our signature/hash-verified tensor checkpoint with weights_only enabled."""
    import json

    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt != {"signature": signature, "sha256": sha256(path)}:
        raise ValueError("Temporal checkpoint differs from its code/data signature or hash.")
    return torch.load(path, map_location="cpu", weights_only=True)
