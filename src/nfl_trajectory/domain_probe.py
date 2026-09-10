"""Fixed-capacity, resumable residual probes for controlled feature attribution."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nfl_trajectory.domain_features import FAMILIES
from nfl_trajectory.runtime import Run
from nfl_trajectory.temporal_model import load_checkpoint, save_checkpoint

SETTINGS: dict[str, Any] = {
    "epochs": 12,
    "seed": 2026,
    "batch_rows": 4096,
    "widths": [64, 32],
    "learning_rate": 0.002,
    "minimum_learning_rate": 0.0001,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "threads": 4,
    "standardized_clip": 10.0,
    "selection": "final_epoch; no validation selection",
}
ARMS = ("control", *FAMILIES, "all", *("without_" + f for f in FAMILIES))


def screen(training: np.ndarray, names: list[str], families: list[str]) -> dict[str, Any]:
    """Unsupervised training-only screen; never accept evaluation rows or outcomes."""
    mean = training.mean(axis=0, dtype=np.float64)
    scale = training.std(axis=0, dtype=np.float64)
    retained = np.ones(training.shape[1], dtype=bool)
    rejected: list[dict[str, str]] = []
    fingerprints: dict[str, str] = {}
    for i, name in enumerate(names):
        if families[i] == "control":
            continue
        if scale[i] < 1e-7:
            retained[i] = False
            rejected.append({"feature": name, "reason": "constant_on_training"})
            continue
        digest = hashlib.sha256(np.ascontiguousarray(training[:, i]).tobytes()).hexdigest()
        if digest in fingerprints:
            retained[i] = False
            rejected.append(
                {"feature": name, "reason": "exact_duplicate", "of": fingerprints[digest]}
            )
        else:
            fingerprints[digest] = name
    return {
        "mean": mean.tolist(),
        "scale": np.maximum(scale, 1e-4).tolist(),
        "retained": retained.tolist(),
        "rejected": rejected,
        "candidate_count": sum(f != "control" for f in families),
        "retained_count": sum(
            bool(r) for r, f in zip(retained, families, strict=True) if f != "control"
        ),
    }


def arm_mask(families: list[str], retained: list[bool], arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError("Unknown domain-feature arm.")
    allowed = {"control"}
    if arm == "all":
        allowed.update(FAMILIES)
    elif arm.startswith("without_"):
        allowed.update(set(FAMILIES) - {arm.removeprefix("without_")})
    elif arm in FAMILIES:
        allowed.add(arm)
    return np.array([f in allowed and r for f, r in zip(families, retained, strict=True)])


def transform(values: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    standardized = (values - np.asarray(fitted["mean"], dtype=np.float32)) / np.asarray(
        fitted["scale"], dtype=np.float32
    )
    return np.clip(standardized, -SETTINGS["standardized_clip"], SETTINGS["standardized_clip"])


class Probe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor, seconds: torch.Tensor) -> torch.Tensor:
        return self.layers(x) * seconds[:, None]


def train_probe(
    x: np.ndarray,
    truth: np.ndarray,
    seconds: np.ndarray,
    folder: Path,
    signature: str,
    run: Run,
    arm: str,
    fold: str,
) -> tuple[Probe, dict[str, Any]]:
    """Fit on training rows only, preserving optimizer and RNG at every epoch."""
    torch.set_num_threads(SETTINGS["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SETTINGS["seed"])
    model = Probe(x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    x_tensor, y_tensor, t_tensor = map(torch.from_numpy, (x, truth, seconds))
    state: dict[str, Any] = {"epoch": 0, "curve": [], "seconds": 0.0}
    checkpoint = folder / "checkpoint.pt"
    if checkpoint.exists():
        restored = load_checkpoint(checkpoint, signature)
        model.load_state_dict(restored.pop("model"))
        optimizer.load_state_dict(restored.pop("optimizer"))
        torch.set_rng_state(restored.pop("rng"))
        state = restored
        run.event("probe_resumed", fold=fold, arm=arm, epoch=state["epoch"])
    start, previous = time.monotonic(), state["seconds"]
    while state["epoch"] < SETTINGS["epochs"]:
        epoch = state["epoch"]
        cosine = (1 + math.cos(math.pi * epoch / (SETTINGS["epochs"] - 1))) / 2
        lr = (
            SETTINGS["minimum_learning_rate"]
            + (SETTINGS["learning_rate"] - SETTINGS["minimum_learning_rate"]) * cosine
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = np.random.default_rng(SETTINGS["seed"] + epoch).permutation(len(x))
        total = 0.0
        model.train()
        for begin in range(0, len(x), SETTINGS["batch_rows"]):
            index = torch.from_numpy(order[begin : begin + SETTINGS["batch_rows"]])
            optimizer.zero_grad(set_to_none=True)
            output = model(x_tensor[index], t_tensor[index])
            loss = (output - y_tensor[index]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Domain probe objective became nonfinite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(index)
        state["epoch"] += 1
        state["seconds"] = previous + time.monotonic() - start
        state["curve"].append(
            {
                "epoch": state["epoch"],
                "training_rmse": math.sqrt(total / len(x)),
                "learning_rate": lr,
            }
        )
        save_checkpoint(
            checkpoint,
            {
                **state,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            signature,
        )
        run.event(
            "probe_epoch",
            fold=fold,
            arm=arm,
            **state["curve"][-1],
            training_seconds=state["seconds"],
        )
    return model.eval(), state


@torch.inference_mode()
def predict(model: Probe, x: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            model(
                torch.from_numpy(x[i : i + 8192]), torch.from_numpy(seconds[i : i + 8192])
            ).numpy()
            for i in range(0, len(x), 8192)
        ]
    )
