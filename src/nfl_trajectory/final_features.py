"""Materialize the frozen final schema with the independently refitted preprocessing."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.context_experiment import context_weeks
from nfl_trajectory.context_features import build_context_features, context_catalog
from nfl_trajectory.feature_candidates import candidate_matrix, research_catalog
from nfl_trajectory.feature_experiment import target_state
from nfl_trajectory.feature_research import BATCH, baseline_prediction, history_rows
from nfl_trajectory.features import build_player_features
from nfl_trajectory.final_protocol import digest, load_protocol
from nfl_trajectory.final_training import validate_preprocessing
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.representation_features import build_representation, representation_catalog
from nfl_trajectory.research_evidence import verified_checkpoint
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage


def load_preprocessing(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Check implementation, protocol, coverage, and each independent stage receipt."""
    folder = root / "artifacts/final/preprocessing"
    report = json.loads((folder / "summary.json").read_text())
    expected = {
        "protocol_source": protocol["source_signature"],
        "protocol_sha256": sha256(root / "artifacts/final/protocol.json"),
        "implementation_sha256": sha256(Path(__file__).with_name("final_training.py")),
        "script_sha256": sha256(root / "scripts/refit_final.py"),
        "training_games": protocol["contract"]["training_games"],
    }
    if (
        report.get("status") != "passed"
        or report.get("provenance") != expected
        or report.get("source_signature") != digest(expected)
    ):
        raise ValueError("Final preprocessing provenance no longer matches the frozen protocol.")
    for name, component in (
        ("baseline.json", "baseline"),
        ("history.csv", "history"),
        ("history_model.json", "history"),
        ("routes.json", "routes"),
    ):
        if report["outputs"].get(name) != sha256(folder / name):
            raise ValueError("Final preprocessing output changed: " + name)
        verified_checkpoint(root, "final-" + component, digest(expected), folder / name)
    inventory = protocol["training_inventory"]
    measured = validate_preprocessing(
        folder,
        expected["training_games"],
        inventory["training_rows"],
        inventory["training_trajectories"],
    )
    if any(report.get(key) != value for key, value in measured.items()):
        raise ValueError("Final preprocessing receipt has incorrect measured coverage.")
    return dict(report)


def feature_groups(names: list[str]) -> dict[str, list[str]]:
    catalogs = {
        "research": set(research_catalog().feature),
        "context": set(context_catalog().feature),
        "representation": set(representation_catalog().feature),
    }
    groups = {kind: [name for name in names if name in known] for kind, known in catalogs.items()}
    if not names or len(set(names)) != len(names) or sum(map(len, groups.values())) != len(names):
        raise ValueError("Final columns must be unique, nonempty, known feature names.")
    return groups


def feature_matrix(
    bank: Any,
    targets: pd.DataFrame,
    history: np.ndarray,
    context: Any,
    representation: Any,
    names: list[str],
) -> np.ndarray:
    """Retain frozen interleaved column order across all three candidate banks."""
    groups = feature_groups(names)
    positions = {name: index for index, name in enumerate(names)}
    matrix = np.empty((len(targets), len(names)), dtype=np.float32)
    for kind, columns in groups.items():
        if columns:
            values = (
                candidate_matrix(bank, targets, history, columns)
                if kind == "research"
                else (context if kind == "context" else representation).matrix(targets, columns)
            )
            matrix[:, [positions[name] for name in columns]] = values
    if not np.isfinite(matrix).all():
        raise ValueError("Final features contain nonfinite values.")
    return matrix


def materialize(
    root: Path,
    caches: list[Path],
    protocol: dict[str, Any],
    source: str,
    run: Run,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Write one primary matrix; fallback fitting takes its frozen ordered subset."""
    folder = root / "artifacts/final/matrix"
    preprocessing = root / "artifacts/final/preprocessing"
    names = protocol["contract"]["availability_profile_features"]["without_metadata"]
    games = protocol["contract"]["training_games"]
    inventory = protocol["training_inventory"]
    rows = inventory["training_rows"]
    shape = (rows, len(names))
    outputs = [folder / name for name in ("features.npy", "targets.npz", "summary.json")]

    def action() -> None:
        folder.mkdir(parents=True, exist_ok=True)
        required = rows * (len(names) * 4 + 128) + 1024**3
        reusable = outputs[0].stat().st_size if outputs[0].exists() else 0
        if shutil.disk_usage(folder).free + reusable < required:
            raise ValueError(f"Final materialization needs {required / 1024**3:.2f} GiB free.")
        history = pd.read_csv(preprocessing / "history.csv")
        baseline = json.loads((preprocessing / "baseline.json").read_text())
        routes = json.loads((preprocessing / "routes.json").read_text())
        matrix = np.lib.format.open_memmap(outputs[0], mode="w+", dtype=np.float32, shape=shape)
        frames, truth, predictions, signs = [], [], [], []
        cursor = 0
        for cache in caches:
            for bank, targets, arrays, state, basis, context in context_weeks(
                root, [cache], set(games)
            ):
                require_keys(targets)
                weekly = next(row for row in inventory["weeks"] if row["week"] == cache.parent.name)
                if len(targets) != weekly["rows"] or cursor + len(targets) > rows:
                    raise ValueError("Final matrix row count differs from the verified inventory.")
                keys = targets[KEYS].to_numpy(dtype="<i8")
                if (
                    hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()
                    != weekly["ordered_keys_sha256"]
                ):
                    raise ValueError("Final matrix row order differs from the verified inventory.")
                representation = build_representation(bank, routes)
                h = history_rows(history, targets)
                for start in range(0, len(targets), BATCH):
                    end = min(start + BATCH, len(targets))
                    matrix[cursor + start : cursor + end] = feature_matrix(
                        bank, targets.iloc[start:end], h[start:end], context, representation, names
                    )
                frames.append(targets[KEYS])
                truth.append(arrays["truth"])
                predictions.append(baseline_prediction(state, basis, baseline))
                signs.append(target_state(bank, targets).sign.to_numpy()[:, None])
                cursor += len(targets)
                run.event(
                    "final_matrix_progress",
                    week=cache.parent.name,
                    completed_rows=cursor,
                    total_rows=rows,
                )
        combined = pd.concat(frames, ignore_index=True)
        require_keys(combined)
        if (
            cursor != rows
            or set(combined.game_id) != set(games)
            or len(combined[ENTITY].drop_duplicates()) != inventory["training_trajectories"]
        ):
            raise ValueError("Final matrix does not cover the complete frozen training partition.")
        actual, base, sign = (
            np.concatenate(truth),
            np.concatenate(predictions),
            np.concatenate(signs),
        )
        residual = (actual - base) * sign
        if not np.isfinite(residual).all() or not np.isin(sign, [-1, 1]).all():
            raise ValueError("Final residual targets or coordinate signs are invalid.")
        matrix.flush()
        del matrix
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            keys=combined.to_numpy(np.int64),
            residual=residual,
            baseline=base,
            truth=actual,
            sign=sign,
        )
        atomic_bytes(outputs[1], payload.getvalue())
        atomic_json(
            outputs[2],
            {
                "rows": rows,
                "features": names,
                "source_signature": source,
                "holdout_evaluation": "not_run",
            },
        )

    stage(root, "final-matrix", source, outputs, action, run)
    matrix = np.load(outputs[0], mmap_mode="r", allow_pickle=False)
    with np.load(outputs[1], allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if matrix.shape != shape or matrix.dtype != np.float32 or arrays["residual"].shape != (rows, 2):
        raise ValueError("The saved final training matrix violates its frozen shape contract.")
    return matrix, arrays


def validate_observed_features(root: Path, caches: list[Path], run: Run) -> dict[str, Any]:
    """Compare full frozen features against raw input for one complete play in every week."""
    protocol = load_protocol(root)
    preprocessing = load_preprocessing(root, protocol)
    folder = root / "artifacts/final/preprocessing"
    history = pd.read_csv(folder / "history.csv")
    routes = json.loads((folder / "routes.json").read_text())
    games = protocol["contract"]["training_games"]
    profiles = protocol["contract"]["availability_profile_features"]
    names = profiles["without_metadata"]
    lookup = {name: i for i, name in enumerate(names)}
    results: list[dict[str, Any]] = []
    for cache in caches:
        for bank, targets, _, _, _, context in context_weeks(root, [cache], set(games)):
            first = targets.sort_values(KEYS).iloc[0]
            targets = targets.loc[
                targets.game_id.eq(first.game_id) & targets.play_id.eq(first.play_id)
            ]
            raw_path = root / "data/raw/train" / (cache.parent.name + ".csv")
            observed = pd.read_csv(raw_path)
            observed = observed.loc[
                observed.game_id.eq(first.game_id) & observed.play_id.eq(first.play_id)
            ]
            raw_bank = build_player_features(observed, targets[ENTITY])
            raw_context = build_context_features(observed, raw_bank.state)
            raw_representation = build_representation(raw_bank, routes)
            representation = build_representation(bank, routes)
            differences = {name: 0.0 for name in profiles}
            h = history_rows(history, targets)
            for start in range(0, len(targets), BATCH):
                end = start + BATCH
                part = targets.iloc[start:end]
                expected = feature_matrix(bank, part, h[start:end], context, representation, names)
                actual = feature_matrix(
                    raw_bank, part, h[start:end], raw_context, raw_representation, names
                )
                for profile, columns in profiles.items():
                    indices = [lookup[name] for name in columns]
                    np.testing.assert_allclose(
                        actual[:, indices], expected[:, indices], rtol=1e-6, atol=1e-5
                    )
                    differences[profile] = max(
                        differences[profile],
                        float(np.max(np.abs(actual[:, indices] - expected[:, indices]))),
                    )
            results.append(
                {
                    "week": cache.parent.name,
                    "rows": len(targets),
                    "max_absolute_feature_difference": differences,
                }
            )
            run.event("final_raw_feature_check", **results[-1])
    if len(results) != len(protocol["training_inventory"]["weeks"]):
        raise ValueError("Final raw feature validation did not cover every training week.")
    if load_protocol(root) != protocol or load_preprocessing(root, protocol) != preprocessing:
        raise ValueError("Final feature validation inputs changed during execution.")
    report = {
        "status": "passed",
        "scope": "One complete training play per week; full frozen schemas.",
        "protocol_source": protocol["source_signature"],
        "preprocessing_source": preprocessing["source_signature"],
        "implementation_sha256": sha256(Path(__file__)),
        "profile_columns": {name: len(columns) for name, columns in profiles.items()},
        "weeks": results,
        "rows": sum(row["rows"] for row in results),
        "holdout_evaluation": "not_run",
        "full_training_run": False,
    }
    atomic_json(root / "artifacts/final/feature_validation.json", report)
    return report
