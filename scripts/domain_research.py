# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "plotly==7.0.0", "matplotlib==3.10.8",
#   "filelock==3.32.5", "torch==2.8.0", "pytest==9.1.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Execute the preregistered domain-feature screen on saved chronological models."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.domain_features import FAMILIES, candidates  # noqa: E402
from nfl_trajectory.domain_probe import (  # noqa: E402
    ARMS,
    SETTINGS,
    arm_mask,
    predict,
    screen,
    train_probe,
    transform,
)
from nfl_trajectory.motion import KEYS  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage  # noqa: E402
from nfl_trajectory.temporal_model import TemporalModel, collate, load_checkpoint  # noqa: E402
from nfl_trajectory.temporal_research import validate_fold  # noqa: E402


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def specification() -> dict[str, Any]:
    reference = json.loads((ROOT / "docs/results/temporal_research.json").read_text())
    parent_path = ROOT / "artifacts/temporal/research/plan.json"
    parent = json.loads(parent_path.read_text())
    signature = parent.pop("signature")
    if fingerprint(parent) != signature:
        raise ValueError("Parent chronological plan fingerprint changed.")
    inputs = {
        "docs/results/temporal_research.json": sha256(ROOT / "docs/results/temporal_research.json"),
        "artifacts/temporal/research/plan.json": sha256(parent_path),
    }
    for name, digest in {**parent["inputs"], **parent["sources"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Parent numerical source or input failed verification: " + name)
        inputs[name] = digest
    original = json.loads((ROOT / "artifacts/temporal/development/plan.json").read_text())
    for name, digest in original["inputs"].items():
        if name.startswith(("src/", "scripts/")):
            if sha256(ROOT / name) != digest:
                raise ValueError("Original temporal source changed: " + name)
            inputs[name] = digest
    splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
    for fold in parent["folds"]:
        validate_fold(fold, splits)
        name = fold["name"]
        path = ROOT / f"artifacts/temporal/research/{name}/samples.pkl"
        receipt = json.loads((ROOT / f".state/{name}-temporal-samples.json").read_text())
        relative = str(path.relative_to(ROOT))
        if receipt["outputs"].get(relative) != sha256(path):
            raise ValueError("Fold-local sample hash failed verification.")
        inputs[relative] = sha256(path)
        for path, digest in reference["artifacts"].items():
            if f"/{name}/full/" in path or (f"/{name}/" in path and "without_metadata" in path):
                if sha256(ROOT / path) != digest:
                    raise ValueError("Saved temporal or tree reference changed.")
                inputs[path] = digest
        summary = f"artifacts/temporal/research/{name}/full/summary.json"
        inputs[summary] = sha256(ROOT / summary)
    paths = [
        "scripts/domain_research.py",
        "scripts/domain_research.py.lock",
        "src/nfl_trajectory/domain_features.py",
        "src/nfl_trajectory/domain_probe.py",
        "docs/DOMAIN_EXPERIMENT.md",
    ]
    return {
        "version": 1,
        "folds": parent["folds"],
        "settings": SETTINGS,
        "families": FAMILIES,
        "arms": ARMS,
        "inputs": inputs,
        "sources": {p: sha256(ROOT / p) for p in paths},
        "selection": "predeclared all-families vs matched control; never minimum of arms",
        "scope": "original training partition only; no development/holdout/Kaggle evaluation",
    }


@torch.inference_mode()
def extract(
    samples: list[dict[str, np.ndarray]], model: TemporalModel, run: Run
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("x", "y", "time", "keys", "train", "sign", "role", "cold")
    }
    schema: dict[str, Any] = {}
    captured: list[torch.Tensor] = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0].detach())

    hook = model.decode[0].register_forward_pre_hook(capture)
    try:
        batches: list[list[dict[str, np.ndarray]]] = []
        for split in ("train", "validation"):
            partition = [s for s in samples if s["split"] == split]
            batches.extend(partition[i : i + 64] for i in range(0, len(partition), 64))
        if sum(map(len, batches)) != len(samples):
            raise ValueError("Unexpected split in domain feature extraction.")
        completed = 0
        for batch_index, subset in enumerate(batches):
            batch = collate(subset)
            captured.clear()
            prediction = model(batch).numpy()
            original = captured[0].numpy()
            for i, sample in enumerate(subset):
                n = len(sample["time"])
                engineered, names, families = candidates(sample)
                role = sample["role"][sample["player"]]
                control = np.concatenate(
                    [original[i, :n], prediction[i, :n] / 10, np.eye(5, dtype=np.float32)[role]],
                    axis=1,
                )
                current = {
                    "names": [f"control__{j:03d}" for j in range(control.shape[1])] + names,
                    "families": ["control"] * control.shape[1] + families,
                }
                if schema and current != schema:
                    raise ValueError("Feature schema changed between plays.")
                schema = current
                parts["x"].append(np.concatenate([control, engineered], axis=1))
                parts["y"].append(sample["truth"] - prediction[i, :n])
                parts["time"].append(sample["time"])
                parts["keys"].append(sample["keys"])
                parts["train"].append(np.full(n, sample["split"] == "train"))
                parts["sign"].append(np.full(n, sample["sign"], dtype=np.float32))
                parts["role"].append(role)
                parts["cold"].append(sample["static"][sample["player"], 11] > 0)
            completed += len(subset)
            if batch_index % 16 == 0:
                run.event(
                    "domain_features_prepared",
                    plays=completed,
                    total_plays=len(samples),
                )
    finally:
        hook.remove()
    return {k: np.concatenate(v) for k, v in parts.items()}, schema


def errors(data: dict[str, np.ndarray], mask: np.ndarray, correction: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(data["keys"][mask], columns=KEYS)
    result[["dx", "dy"]] = (correction - data["y"][mask]) * data["sign"][mask, None]
    return result


def prepare(
    fold: dict[str, Any], folder: Path, signature: str, run: Run
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parent = ROOT / "artifacts/temporal/research" / fold["name"]
    cache, schema_path = folder / "features.npz", folder / "schema.json"

    def build() -> None:
        summary = json.loads((parent / "full/summary.json").read_text())
        model = TemporalModel().eval()
        checkpoint = load_checkpoint(parent / "full/checkpoint.pt", summary["signature"])
        model.load_state_dict(checkpoint["ema"])
        samples = pickle.loads((parent / "samples.pkl").read_bytes())
        data, schema = extract(samples, model, run)
        old, replayed = aligned_errors(
            pd.read_csv(parent / "full/errors.csv"),
            errors(data, ~data["train"], np.zeros_like(data["y"][~data["train"]])),
        )
        delta = np.abs(old[["dx", "dy"]].to_numpy() - replayed[["dx", "dy"]].to_numpy()).max()
        if delta > 2e-6:
            raise ValueError("Frozen network predictions differ from the saved reference.")
        schema["reference_replay_max_absolute_difference"] = float(delta)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        atomic_bytes(cache, buffer.getvalue())
        atomic_json(schema_path, schema)

    stage(ROOT, fold["name"] + "-domain-features", signature, [cache, schema_path], build, run)
    with np.load(cache, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    schema = json.loads(schema_path.read_text())
    return data, schema


def run_fold(fold: dict[str, Any], folder: Path, signature: str, run: Run) -> None:
    data, schema = prepare(fold, folder, signature, run)
    train, evaluation = data["train"], ~data["train"]
    fitted_path = folder / "screen.json"

    def fit_screen() -> None:
        fitted = screen(data["x"][train], schema["names"], schema["families"])
        z = (data["x"][evaluation] - np.array(fitted["mean"])) / np.array(fitted["scale"])
        fitted["evaluation_clipped_fraction"] = (
            (np.abs(z) > SETTINGS["standardized_clip"]).mean(axis=0).tolist()
        )
        atomic_json(fitted_path, fitted)

    stage(ROOT, fold["name"] + "-domain-screen", signature, [fitted_path], fit_screen, run)
    fitted = json.loads(fitted_path.read_text())
    x = transform(data.pop("x"), fitted)
    for arm in ARMS:
        destination = folder / arm
        destination.mkdir(parents=True, exist_ok=True)
        arm_signature = fingerprint({"signature": signature, "fold": fold["name"], "arm": arm})
        summary_path = destination / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["signature"] != arm_signature or not all(
                sha256(destination / p) == h for p, h in summary["artifacts"].items()
            ):
                raise ValueError("Completed domain arm no longer matches its lineage.")
            run.event("domain_arm_reused", fold=fold["name"], arm=arm)
            continue
        keep = arm_mask(schema["families"], fitted["retained"], arm)
        model, state = train_probe(
            np.ascontiguousarray(x[train] * keep),
            data["y"][train],
            data["time"][train],
            destination,
            arm_signature,
            run,
            arm,
            fold["name"],
        )
        correction = predict(
            model, np.ascontiguousarray(x[evaluation] * keep), data["time"][evaluation]
        )
        frame = errors(data, evaluation, correction)
        atomic_bytes(destination / "errors.csv", frame.to_csv(index=False).encode())
        summary = {
            "status": "completed",
            "signature": arm_signature,
            "fold": fold["name"],
            "arm": arm,
            "settings": SETTINGS,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "features_active": int(keep.sum()),
            "candidate_count": fitted["candidate_count"],
            "training_rows": int(train.sum()),
            "evaluation_rows": int(evaluation.sum()),
            "training_seconds": state["seconds"],
            "epochs": state["epoch"],
            "curve": state["curve"],
            "run_id": run.run_id,
            **error_metrics(frame),
            "artifacts": {
                p: sha256(destination / p)
                for p in ("checkpoint.pt", "checkpoint.json", "errors.csv")
            },
        }
        atomic_json(summary_path, summary)
        run.event("domain_arm_completed", fold=fold["name"], arm=arm, **error_metrics(frame))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["all", "inner_1", "inner_2", "inner_3"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(SETTINGS["threads"])
    torch.use_deterministic_algorithms(True)
    folder = ROOT / "artifacts/domain_research"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(ROOT, "domain-research") as run:
        plan = specification()
        signature = fingerprint(plan)
        plan_path = folder / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text())["signature"] != signature:
            raise ValueError(
                "Domain study specification changed; preserve the existing experiment."
            )
        atomic_json(plan_path, {"signature": signature, **plan})
        for fold in plan["folds"]:
            if args.fold != "all" and args.fold != fold["name"]:
                continue
            destination = folder / fold["name"]
            destination.mkdir(parents=True, exist_ok=True)
            run_fold(fold, destination, signature, run)


if __name__ == "__main__":
    main()
