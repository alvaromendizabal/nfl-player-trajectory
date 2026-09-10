"""New matched velocity-supervision primitives, not the lost historical model.

The forward boundary accepts observed/request tensors only. Future coordinates
and finite-difference motion labels are held separately. The two arms use the
same model; only the auxiliary loss weight differs. No experiment launches here.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from nfl_trajectory.motion_targets import motion_targets
from nfl_trajectory.temporal_data import CHANNELS
from nfl_trajectory.temporal_model import TemporalModel, collate

EXPERIMENT_ID = "velocity-isolation-20260910-v1"
FEATURE_KEYS = frozenset(
    (
        "history",
        "observed",
        "static",
        "role",
        "side",
        "valid",
        "edges",
        "player",
        "time",
        "baseline",
        "scored",
    )
)
TensorBatch = dict[str, torch.Tensor]


class GroupedMotionModel(nn.Module):
    """Per-signal temporal filters followed by the maintained attention decoder.

    Thirty observed channels plus their presence mask each receive four filters.
    Dilations 1/2/4 use only the already observed 20-frame window. A pointwise
    projection returns 48 channels, retaining the parent's decoder dimensions.
    The learned velocity head shares the decoder representation but is not an
    input to coordinate prediction. There is no 48-frame forecast ceiling.
    """

    def __init__(self, width: int = 96) -> None:
        if width < 8 or width % 4:
            raise ValueError("Width must be at least eight and divisible by four.")
        super().__init__()
        parent = TemporalModel(width)
        self.role = parent.role
        self.side = parent.side
        self.project = parent.project
        self.interactions = parent.interactions
        self.final_norm = parent.final_norm
        self.decode = parent.decode
        signals = len(CHANNELS) + 1
        channels = signals * 4
        self.temporal = nn.Sequential(
            nn.Conv1d(signals, channels, 3, padding=1, groups=signals),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2, groups=signals),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4, groups=signals),
            nn.GELU(),
            nn.Conv1d(channels, 48, 1),
            nn.GELU(),
        )
        self.velocity_head = nn.Linear(64, 2)
        nn.init.normal_(self.velocity_head.weight, std=0.01)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, batch: TensorBatch) -> TensorBatch:
        if set(batch) != FEATURE_KEYS:
            raise ValueError("Forward accepts only observed/request tensors; labels are forbidden.")
        h, observed = batch["history"], batch["observed"]
        b, p, t, _ = h.shape
        h = torch.where(observed[..., None], h, 0.0)
        h = torch.cat([h, observed[..., None].to(h.dtype)], dim=-1)
        temporal = self.temporal(h.reshape(b * p, t, -1).transpose(1, 2)).reshape(b, p, -1)
        static = batch["static"]
        hidden = (
            self.project(
                torch.cat(
                    [temporal, static, self.role(batch["role"]), self.side(batch["side"])], dim=-1
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
        shared = self.decode[:-1](inputs)
        mask = batch["scored"][..., None]
        return {
            "coordinate": self.decode[-1](shared) * time[..., None] * mask,
            "velocity": self.velocity_head(shared) * mask,
        }


def observed_batch(samples: list[dict[str, np.ndarray]]) -> TensorBatch:
    """Remove every future-label field before calling the maintained collator."""
    if not samples:
        raise ValueError("A batch must contain at least one play.")
    clean = [{key: value for key, value in s.items() if key != "truth"} for s in samples]
    batch = collate(clean)
    return {key: value for key, value in batch.items() if key in FEATURE_KEYS}


def supervised_batch(
    samples: list[dict[str, np.ndarray]],
) -> tuple[TensorBatch, TensorBatch]:
    """Return disjoint inputs and labels, preserving the maintained derivative contract."""
    features = observed_batch(samples)
    shape = (*features["time"].shape, 2)
    coordinate = np.zeros(shape, dtype=np.float32)
    velocity = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape[:-1], dtype=bool)
    for i, sample in enumerate(samples):
        labels = motion_targets(sample)
        count = len(sample["keys"])
        coordinate[i, :count] = sample["truth"]
        velocity[i, :count] = labels["velocity"]
        mask[i, :count] = labels["velocity_mask"]
    return features, {
        "coordinate": torch.from_numpy(coordinate),
        "velocity": torch.from_numpy(velocity),
        "velocity_mask": torch.from_numpy(mask),
    }


def fit_velocity_scale(samples: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Train-only shared-axis RMS; no acceleration value enters normalization."""
    square = 0.0
    coordinates = 0
    for sample in samples:
        if str(sample["split"]) != "train":
            raise ValueError("Velocity normalization accepts training samples only.")
        labels = motion_targets(sample)
        values = labels["velocity"][labels["velocity_mask"]]
        square += float(np.square(values).sum())
        coordinates += int(values.size)
    if coordinates == 0 or not math.isfinite(square):
        raise ValueError("Finite supported training velocity labels are required.")
    return {"rms": max(0.1, math.sqrt(square / coordinates)), "coordinates": coordinates}


def supervised_loss(
    output: TensorBatch,
    labels: TensorBatch,
    scored: torch.Tensor,
    velocity_weight: float,
    velocity_scale: float,
    coordinate_denominator: float | None = None,
    velocity_denominator: float | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Uniform coordinate SSE; auxiliary velocity is normalized in physical units.

    Scientific batching must pass fixed fold totals / number of batches as each
    denominator. This avoids implicitly giving short plays or the final batch
    more weight. Defaults are suitable for isolated synthetic checks only.
    """
    if velocity_weight not in (0.0, 0.1):
        raise ValueError("This protocol permits only coordinate (0) or velocity (0.1).")
    if not math.isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("Velocity scale must be finite and positive.")
    valid = scored.bool()
    if not bool(valid.any()):
        raise ValueError("Coordinate loss requires requested rows.")
    residual = output["coordinate"][valid] - labels["coordinate"][valid]
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("Active coordinate errors must be finite.")
    sse = residual.square().sum()
    count = residual.numel()
    denominator = float(count) if coordinate_denominator is None else coordinate_denominator
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Coordinate denominator must be finite and positive.")
    total = sse / denominator
    report: dict[str, float | int] = {
        "coordinate_sse": float(sse.detach()),
        "coordinates": count,
        "velocity_sse_normalized": 0.0,
        "velocity_coordinates": 0,
    }
    # In the control, future velocity labels and the velocity head cannot affect gradients.
    if velocity_weight:
        mask = valid & labels["velocity_mask"].bool()
        error = (output["velocity"][mask] - labels["velocity"][mask]) / velocity_scale
        if not bool(torch.isfinite(error).all()):
            raise ValueError("Supported velocity errors must be finite.")
        velocity_sse = error.square().sum()
        vcount = error.numel()
        vdenom = float(max(1, vcount)) if velocity_denominator is None else velocity_denominator
        if not math.isfinite(vdenom) or vdenom <= 0:
            raise ValueError("Velocity denominator must be finite and positive.")
        total = total + velocity_weight * velocity_sse / vdenom
        report.update(
            velocity_sse_normalized=float(velocity_sse.detach()), velocity_coordinates=vcount
        )
    return total, report


class MatchedState:
    """CPU training-step state with model, EMA, optimizer, dropout RNG and cursor.

    This is an engineering primitive, not an authorized scientific fit runner.
    A future runner must freeze data-order/reflection manifests, exposure, source,
    learning-rate schedule, fold denominators and durable per-arm publication.
    """

    def __init__(self, width: int, seed: int, weight: float, scale: float) -> None:
        if weight not in (0.0, 0.1) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid matched arm weight or normalization scale.")
        torch.manual_seed(seed)
        self.width, self.seed, self.weight, self.scale = width, seed, weight, scale
        self.model = GroupedMotionModel(width)
        self.ema = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.rng = torch.get_rng_state().clone()
        self.steps = 0
        self.coordinate_sse = 0.0
        self.coordinates = 0

    def step(
        self,
        features: TensorBatch,
        labels: TensorBatch,
        coordinate_denominator: float | None = None,
        velocity_denominator: float | None = None,
    ) -> dict[str, float | int]:
        if any(value.device.type != "cpu" for value in features.values()):
            raise ValueError("This tested state contract is CPU only.")
        torch.set_rng_state(self.rng)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, stats = supervised_loss(
            self.model(features),
            labels,
            features["scored"],
            self.weight,
            self.scale,
            coordinate_denominator,
            velocity_denominator,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for target, current in zip(self.ema.parameters(), self.model.parameters(), strict=True):
                target.mul_(0.98).add_(current, alpha=0.02)
        self.rng = torch.get_rng_state().clone()
        self.steps += 1
        self.coordinate_sse += float(stats["coordinate_sse"])
        self.coordinates += int(stats["coordinates"])
        return {**stats, "loss": float(loss.detach()), "steps": self.steps}

    def payload(self) -> dict[str, Any]:
        """Only tensors and simple Python values; load with weights_only=True."""
        return {
            "width": self.width,
            "seed": self.seed,
            "weight": self.weight,
            "scale": self.scale,
            "steps": self.steps,
            "coordinate_sse": self.coordinate_sse,
            "coordinates": self.coordinates,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.clone(),
        }

    @classmethod
    def restore(cls, payload: dict[str, Any]) -> MatchedState:
        state = cls(payload["width"], payload["seed"], payload["weight"], payload["scale"])
        state.model.load_state_dict(payload["model"], strict=True)
        state.ema.load_state_dict(payload["ema"], strict=True)
        state.optimizer.load_state_dict(payload["optimizer"])
        state.rng = payload["rng"].clone()
        state.steps = int(payload["steps"])
        state.coordinate_sse = float(payload["coordinate_sse"])
        state.coordinates = int(payload["coordinates"])
        return state
