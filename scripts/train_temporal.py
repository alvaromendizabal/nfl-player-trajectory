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
"""One bounded, resumable temporal attention experiment on the fixed development split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.feature_experiment import load_week  # noqa: E402
from nfl_trajectory.feature_research import fold_history, verify_inputs  # noqa: E402
from nfl_trajectory.model_capacity import validate_partitions  # noqa: E402
from nfl_trajectory.motion import KEYS  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage  # noqa: E402
from nfl_trajectory.temporal_data import (  # noqa: E402
    CHANNELS,
    EDGE_NAMES,
    STATIC_NAMES,
    attach_targets,
    encode_play,
    observed_history,
    reflect,
)
from nfl_trajectory.temporal_model import (  # noqa: E402
    TemporalModel,
    collate,
    coordinate_loss,
    load_checkpoint,
    save_checkpoint,
)

SETTINGS: dict[str, Any] = {
    "epochs": 40,
    "batch_plays": 64,
    "width": 96,
    "learning_rate": 0.002,
    "minimum_learning_rate": 0.00005,
    "weight_decay": 0.01,
    "ema_decay": 0.98,
    "gradient_clip": 1.0,
    "seed": 2026,
    "threads": 4,
    # The stopped stability diagnostic used 319 seconds; keep both attempts below 20 minutes.
    "training_seconds_limit": 850,
    "reflection_probability": 0.5,
    "loss": "unweighted_coordinate_mse",
    "selection": "final_epoch_ema; no best-development checkpoint selection",
}


def prepare(folder: Path, run: Run) -> tuple[list[dict[str, np.ndarray]], str, dict[str, Any]]:
    caches, inputs = verify_inputs(ROOT)
    splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
    raw_paths = [ROOT / "data/raw/train" / (p.parent.name + ".csv") for p in caches]
    manifest = json.loads((ROOT / "artifacts/research/input_manifest.json").read_text())
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    if any(expected.get(str(p.relative_to(ROOT))) != sha256(p) for p in raw_paths):
        raise ValueError("Raw observed histories differ from the verified restore manifest.")
    sources = [Path(__file__), Path(__file__).with_suffix(".py.lock")]
    sources += [
        ROOT / "src/nfl_trajectory" / (name + ".py")
        for name in (
            "temporal_data",
            "temporal_model",
            "feature_candidates",
            "feature_research",
            "runtime",
        )
    ]
    inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in [*raw_paths, *sources]})
    parameters: dict[str, Any] = {
        "settings": SETTINGS,
        "inputs": inputs,
        "history_channels": CHANNELS,
        "static_channels": STATIC_NAMES,
        "edge_channels": EDGE_NAMES,
        "environment": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "split": {
            name: sorted(splits.loc[splits.split.eq(name), "game_id"].astype(int))
            for name in ("train", "validation")
        },
        "holdout_evaluation": "not_run",
        "kaggle_submission": "not_run",
    }
    signature = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    atomic_json(folder / "plan.json", {"signature": signature, **parameters})
    prior_path, encoder_path = folder / "history.csv", folder / "history_model.json"

    def history() -> None:
        table, fitted = fold_history(
            ROOT, caches, set(parameters["split"]["train"]), set(parameters["split"]["validation"])
        )
        atomic_bytes(prior_path, table.to_csv(index=False).encode())
        atomic_json(encoder_path, fitted)

    stage(ROOT, "temporal-history", signature, [prior_path, encoder_path], history, run)
    priors = pd.read_csv(prior_path)
    prior_groups = dict(tuple(priors.groupby(KEYS[:2], sort=False)))
    samples = []
    for week, (cache, raw_path) in enumerate(zip(caches, raw_paths, strict=True), 1):
        destination = folder / "weeks" / (cache.parent.name + ".pkl")

        def materialize(
            cache: Path = cache, raw_path: Path = raw_path, destination: Path = destination
        ) -> None:
            _, targets, arrays = load_week(cache)
            validate_partitions(targets, arrays["labels"], splits)
            raw = pd.read_csv(raw_path)
            raw = raw[raw.game_id.isin(targets.game_id.unique())]
            h = observed_history(raw)
            grouped = targets.groupby(KEYS[:2], sort=True).indices
            output = []
            for key, play in h.groupby(KEYS[:2], sort=True):
                if key not in grouped:
                    continue
                indices = grouped[key]
                sample = attach_targets(
                    encode_play(play, prior_groups[key]),
                    targets.iloc[indices],
                    arrays["ridge"][indices],
                    arrays["truth"][indices],
                )
                label = np.unique(arrays["labels"][indices])
                if len(label) != 1:
                    raise ValueError("A play crosses partition boundaries.")
                sample["split"] = np.array(label[0])
                output.append(sample)
            if sum(len(s["time"]) for s in output) != len(targets):
                raise ValueError("Temporal cache lost requested rows.")
            atomic_bytes(destination, pickle.dumps(output, protocol=5))

        stage(ROOT, "temporal-" + cache.parent.name, signature, [destination], materialize, run)
        # Own pickle: written above or hash-verified by stage, never an untrusted model download.
        samples.extend(pickle.loads(destination.read_bytes()))
        run.event("week_prepared", completed=week, total=len(caches), plays=len(samples))
    return samples, signature, parameters


@torch.inference_mode()
def evaluate(model: TemporalModel, samples: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    model.eval()
    parts = []
    for start in range(0, len(samples), SETTINGS["batch_plays"]):
        subset = samples[start : start + SETTINGS["batch_plays"]]
        prediction = model(collate(subset)).numpy()
        for i, sample in enumerate(subset):
            errors = pd.DataFrame(sample["keys"], columns=KEYS)
            errors[["dx", "dy"]] = (prediction[i, : len(errors)] - sample["truth"]) * sample["sign"]
            parts.append(errors)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(KEYS).any() or not np.isfinite(result[["dx", "dy"]]).all().all():
        raise ValueError("Temporal evaluation must preserve unique, finite requested rows.")
    return result


def train(
    folder: Path,
    samples: list[dict[str, np.ndarray]],
    signature: str,
    run: Run,
) -> tuple[TemporalModel, dict[str, Any]]:
    torch.set_num_threads(SETTINGS["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SETTINGS["seed"])
    model = TemporalModel(SETTINGS["width"])
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    training = [s for s in samples if s["split"] == "train"]
    development = [s for s in samples if s["split"] == "validation"]
    checkpoint = folder / "checkpoint.pt"
    state: dict[str, Any] = {
        "epoch": 0,
        "next_batch": 0,
        "curve": [],
        "seconds": 0.0,
        "sse": 0.0,
        "coordinates": 0,
    }
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, signature)
        model.load_state_dict(state.pop("model"))
        ema.load_state_dict(state.pop("ema"))
        optimizer.load_state_dict(state.pop("optimizer"))
        torch.set_rng_state(state.pop("rng"))
        run.event("training_resumed", epoch=state["epoch"], next_batch=state["next_batch"])
    started = time.monotonic()
    previous_seconds = float(state["seconds"])

    def checkpoint_now() -> None:
        state["seconds"] = previous_seconds + time.monotonic() - started
        save_checkpoint(
            checkpoint,
            {
                **state,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            signature,
        )

    mean_rows = sum(len(s["time"]) for s in training) / len(training)
    while state["epoch"] < SETTINGS["epochs"]:
        epoch = state["epoch"]
        order = np.random.default_rng(SETTINGS["seed"] + epoch).permutation(len(training))
        batches = math.ceil(len(order) / SETTINGS["batch_plays"])
        cosine = (1 + math.cos(math.pi * epoch / (SETTINGS["epochs"] - 1))) / 2
        lr = (
            SETTINGS["minimum_learning_rate"]
            + (SETTINGS["learning_rate"] - SETTINGS["minimum_learning_rate"]) * cosine
        )
        if epoch < 3:
            lr *= (epoch + 1) / 3
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        for batch_index in range(state["next_batch"], batches):
            if previous_seconds + time.monotonic() - started >= SETTINGS["training_seconds_limit"]:
                checkpoint_now()
                run.event("training_budget_reached", epoch=epoch, next_batch=state["next_batch"])
                return ema, state
            indices = order[
                batch_index * SETTINGS["batch_plays"] : (batch_index + 1) * SETTINGS["batch_plays"]
            ]
            rng = np.random.default_rng(SETTINGS["seed"] * 100000 + epoch * 1000 + batch_index)
            subset = [
                reflect(training[i])
                if rng.random() < SETTINGS["reflection_probability"]
                else training[i]
                for i in indices
            ]
            batch = collate(subset)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            mse = coordinate_loss(prediction, batch)
            # Fixed expected row denominator gives equal expected weight to each coordinate,
            # including when plays have different numbers of players or future frames.
            loss = mse * batch["scored"].sum() / (len(indices) * mean_rows)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters(), strict=True):
                    a.lerp_(b, 1 - SETTINGS["ema_decay"])
            count = 2 * int(batch["scored"].sum())
            state["sse"] += float(mse.detach()) * count
            state["coordinates"] += count
            state["next_batch"] = batch_index + 1
        row = {
            "epoch": epoch + 1,
            "training_rmse_augmented": math.sqrt(state["sse"] / state["coordinates"]),
            "learning_rate": lr,
        }
        if epoch == 0 or (epoch + 1) % 5 == 0:
            row["development_rmse"] = error_metrics(evaluate(ema, development))[
                "coordinate_rmse_yards"
            ]
        state["curve"].append(row)
        state.update(epoch=epoch + 1, next_batch=0, sse=0.0, coordinates=0)
        checkpoint_now()
        run.event("epoch_completed", **row, training_seconds=state["seconds"])
    return ema, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import pytest

        raise SystemExit(pytest.main([str(ROOT / "tests/test_temporal_model.py"), "-q"]))
    folder = ROOT / "artifacts/temporal/development"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(ROOT, "temporal-training") as run:
        samples, signature, parameters = prepare(folder, run)
        model, state = train(folder, samples, signature, run)
        development = [s for s in samples if s["split"] == "validation"]
        errors = evaluate(model, development)
        atomic_bytes(folder / "errors.csv", errors.to_csv(index=False).encode())
        metrics = error_metrics(errors)
        report = {
            "status": "completed" if state["epoch"] == SETTINGS["epochs"] else "budget_stopped",
            "source_signature": signature,
            "run_id": run.run_id,
            "settings": SETTINGS,
            "epochs_completed": state["epoch"],
            "training_seconds": state["seconds"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "curve": state["curve"],
            "training_games": len(parameters["split"]["train"]),
            "development_games": len(parameters["split"]["validation"]),
            "training_plays": int(sum(s["split"] == "train" for s in samples)),
            "development_plays": len(development),
            "training_rows": sum(len(s["time"]) for s in samples if s["split"] == "train"),
            "development_rows": len(errors),
            "model": "temporal_player_attention",
            **metrics,
            "frozen_full_feature_development_rmse": 0.6880521879583761,
            "comparison": (
                "Identical 192 training games and 32 later development games; "
                "development already inspected."
            ),
            "checkpoint_selection": SETTINGS["selection"],
            "target_encoding": (
                "Strictly earlier-date player/role motion residual priors; "
                "smoothing=20; frozen at evaluation."
            ),
            "holdout_evaluation": "not_run",
            "kaggle_submission": "not_run",
            "promotion": (
                "Requires chronological inner-fold confirmation and organizer gateway validation."
            ),
            "artifacts": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted(folder.rglob("*"))
                if p.is_file()
                and p.suffix in {".pt", ".json", ".pkl", ".csv"}
                and p.name != "summary.json"
            },
        }
        atomic_json(folder / "summary.json", report)
        if args.publish:
            atomic_json(ROOT / "docs/results/temporal_model.json", report)
        run.event("development_evaluated", status=report["status"], **metrics)


if __name__ == "__main__":
    main()
