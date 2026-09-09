"""Seal keyed predictions before opening the reserved, chronological holdout outcomes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
from nfl_trajectory.final_inference import final_bundle, predict_final
from nfl_trajectory.final_protocol import EVALUATION, digest, load_protocol
from nfl_trajectory.models import predict
from nfl_trajectory.motion import ENTITY, KEYS, constant_velocity, require_keys
from nfl_trajectory.research_evidence import verified_checkpoint
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

SOURCE_SNAPSHOT = "c83a828f42f13af498443574284db8d6e014b44967bc4ee09ab7a49f4b362817"
MAX_FORECAST_FRAMES = 1000
SCENARIOS = (
    "complete",
    "without_metadata",
    "without_telemetry",
    "cold_history",
    "constant_velocity",
    "role_ridge",
)


def prediction_request(observed: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Use the supplied forecast horizon, never keys or coordinates from an outcome file."""
    require_keys(observed)
    required = {"player_to_predict", "num_frames_output", "player_role"}
    if not required.issubset(observed.columns) or observed[list(required)].isna().any().any():
        raise ValueError("Forecast availability and horizon fields must be complete.")
    if not observed.player_to_predict.isin([True, False]).all():
        raise ValueError("player_to_predict must be Boolean.")
    entities = (
        observed.loc[observed.player_to_predict]
        .sort_values(KEYS)
        .groupby(ENTITY, sort=False)
        .tail(1)
    )
    horizons = entities.num_frames_output.to_numpy(float)
    if (
        not len(entities)
        or not np.isfinite(horizons).all()
        or (horizons < 1).any()
        or (horizons > MAX_FORECAST_FRAMES).any()
        or not np.equal(horizons, np.floor(horizons)).all()
    ):
        raise ValueError("Prediction horizons must be positive integer frame counts at most 1000.")
    index = np.repeat(np.arange(len(entities)), horizons.astype(int))
    request = entities.iloc[index][ENTITY].reset_index(drop=True)
    request["frame_id"] = np.concatenate([np.arange(1, int(n) + 1) for n in horizons])
    require_keys(request)
    return request, entities.player_role.to_numpy(dtype=str)[index]


def validate_forecast_partition(keys: pd.DataFrame, games: list[int]) -> None:
    require_keys(keys)
    if set(keys.game_id) != set(games):
        raise ValueError("Predictions must cover exactly the frozen holdout games.")


def seal_predictions(root: Path, run: Run) -> dict[str, Any]:
    """Checkpoint all prespecified scenarios and references, then publish an immutable seal."""
    folder = root / "artifacts/final"
    with FileLock(str(folder / "pipeline.lock"), timeout=1):
        if (folder / "model_seal.json").exists():
            return load_seal(root)
        protocol, bundle = load_protocol(root), final_bundle(root)
        if protocol["contract"]["evaluation"] != EVALUATION:
            raise ValueError("Final evaluation rules differ from the frozen protocol.")
        gateway = json.loads((folder / "inference/summary.json").read_text())
        if gateway.get("status") != "passed" or gateway["provenance"]["bundle_sha256"] != digest(
            bundle
        ):
            raise ValueError("Final gateway validation is missing or belongs to another model.")
        for name in ("summary.json", "bundle.json", "standalone.py", "submission.parquet"):
            verified_checkpoint(
                root, "final-gateway", gateway["source_signature"], folder / "inference" / name
            )
        manifest = folder / "source_manifest.json"
        if sha256(manifest) != SOURCE_SNAPSHOT:
            raise ValueError("The original competition snapshot manifest is required.")
        entries = json.loads(manifest.read_text())["files"]
        inputs: dict[str, str] = {}
        outcomes: dict[str, str] = {}
        for week in (16, 17, 18):
            for kind, destination in (("input", inputs), ("output", outcomes)):
                name = f"data/raw/train/{kind}_2023_w{week:02d}.csv"
                entry = next(item for item in entries if item["path"] == name)
                destination[name] = entry["sha256"]
                if kind == "input" and sha256(root / name) != entry["sha256"]:
                    raise ValueError("Holdout observed input differs from the original snapshot.")
        for name in ("src/nfl_trajectory/final_evaluation.py", "scripts/evaluate_final.py"):
            inputs[name] = sha256(root / name)
        inputs["artifacts/final/inference/summary.json"] = sha256(folder / "inference/summary.json")
        provenance = {
            "protocol_sha256": sha256(folder / "protocol.json"),
            "bundle_sha256": digest(bundle),
            "inputs": inputs,
            "outcome_hashes": outcomes,
            "rules": EVALUATION,
        }
        source = digest(provenance)
        outputs, keys = [], []
        for week in (16, 17, 18):
            path = root / f"data/raw/train/input_2023_w{week:02d}.csv"
            output = folder / "predictions" / f"week_{week:02d}.npz"

            def action(path: Path = path, output: Path = output, week: int = week) -> None:
                observed = pd.read_csv(path)
                if not set(observed.game_id).issubset(protocol["contract"]["holdout_games"]):
                    raise ValueError("Holdout input contains an unexpected game.")
                request, roles = prediction_request(observed)
                arrays: dict[str, Any] = {"keys": request.to_numpy(np.int64), "roles": roles}
                for scenario in SCENARIOS:
                    if scenario == "constant_velocity":
                        prediction = constant_velocity(observed, request)
                    elif scenario == "role_ridge":
                        prediction = predict(observed, request, "role_ridge", bundle["baseline"])
                    else:
                        columns = (
                            ["s", "a", "dir", "o"]
                            if scenario == "without_telemetry"
                            else [
                                "player_height",
                                "player_weight",
                                "player_birth_date",
                                "player_position",
                            ]
                            if scenario == "without_metadata"
                            else []
                        )
                        prediction = predict_final(
                            observed.drop(columns=columns, errors="ignore"),
                            request,
                            bundle,
                            cold_history=scenario == "cold_history",
                        )
                    pd.testing.assert_frame_equal(prediction[KEYS], request)
                    arrays[scenario] = prediction[["x", "y"]].to_numpy(float)
                    if not np.isfinite(arrays[scenario]).all():
                        raise ValueError("Nonfinite sealed predictions.")
                    run.event(
                        "holdout_predictions_generated",
                        week=week,
                        scenario=scenario,
                        rows=len(request),
                        outcomes_opened=False,
                    )
                np.testing.assert_array_equal(arrays["complete"], arrays["without_metadata"])
                payload = io.BytesIO()
                np.savez_compressed(payload, **arrays)
                atomic_bytes(output, payload.getvalue())

            stage(root, f"final-predict-week-{week}", source, [output], action, run)
            outputs.append(output)
            with np.load(output, allow_pickle=False) as saved:
                keys.append(pd.DataFrame(saved["keys"], columns=KEYS))
        combined = pd.concat(keys, ignore_index=True)
        validate_forecast_partition(combined, protocol["contract"]["holdout_games"])
        if digest(final_bundle(root)) != digest(bundle):
            raise ValueError("Final model changed during prediction generation.")
        for name, expected in inputs.items():
            if sha256(root / name) != expected:
                raise ValueError("Final prediction input or implementation changed.")
        payload = {
            "provenance": provenance,
            "predictions": {str(path.relative_to(root)): sha256(path) for path in outputs},
            "rows": len(combined),
            "games": sorted(combined.game_id.unique().tolist()),
            "scenarios": list(SCENARIOS),
            "outcomes_opened": False,
        }
        seal: dict[str, Any] = {**payload, "source_signature": digest(payload)}
        atomic_json(folder / "model_seal.json", seal)
        run.event(
            "final_predictions_sealed",
            rows=len(combined),
            games=len(seal["games"]),
            source_signature=seal["source_signature"],
        )
        return seal


def load_seal(root: Path) -> dict[str, Any]:
    folder = root / "artifacts/final"
    seal = json.loads((folder / "model_seal.json").read_text())
    if seal["source_signature"] != digest(
        {k: v for k, v in seal.items() if k != "source_signature"}
    ):
        raise ValueError("Final prediction seal has inconsistent provenance.")
    protocol = load_protocol(root)
    if seal["games"] != protocol["contract"]["holdout_games"] or seal["scenarios"] != list(
        SCENARIOS
    ):
        raise ValueError("The sealed forecast partition or scenarios changed.")
    provenance = seal["provenance"]
    if (
        provenance["bundle_sha256"] != digest(final_bundle(root))
        or provenance["protocol_sha256"] != sha256(folder / "protocol.json")
        or provenance["rules"] != EVALUATION
    ):
        raise ValueError("The final model or evaluation protocol changed after sealing.")
    for name, expected in {**provenance["inputs"], **seal["predictions"]}.items():
        if sha256(root / name) != expected:
            raise ValueError("A sealed input, prediction, or implementation changed: " + name)
    return dict(seal)


def score_predictions(
    truth: pd.DataFrame, keys: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    """An exact outer key join prevents row loss or positional target misalignment."""
    require_keys(truth)
    require_keys(keys)
    if predictions.shape != (len(keys), 2) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be aligned finite coordinates.")
    joined = keys.assign(
        px=predictions[:, 0], py=predictions[:, 1], request_order=np.arange(len(keys))
    ).merge(
        truth[KEYS + ["x", "y"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if (
        not joined._merge.eq("both").all()
        or not np.isfinite(joined[["x", "y"]].to_numpy(float)).all()
    ):
        raise ValueError("Every sealed prediction must match exactly one finite outcome.")
    joined = joined.sort_values("request_order").reset_index(drop=True)
    return joined[KEYS].assign(dx=joined.px - joined.x, dy=joined.py - joined.y)


def evaluate_sealed(root: Path, run: Run) -> dict[str, Any]:
    """Read outcomes only after seal verification; reuse only the exact same evaluation."""
    folder = root / "artifacts/final"
    with FileLock(str(folder / "pipeline.lock"), timeout=1):
        seal = load_seal(root)
        paths = [folder / "evaluation" / name for name in ("summary.json", "errors.csv")]

        def action() -> None:
            opening = folder / "outcome_access.json"
            receipt = {"seal": seal["source_signature"], "run_id": run.run_id}
            if opening.exists():
                if json.loads(opening.read_text())["seal"] != seal["source_signature"]:
                    raise ValueError("Outcomes were already opened under a different seal.")
            else:
                atomic_json(opening, receipt)
            run.event("holdout_outcomes_opened", seal=seal["source_signature"])
            errors = []
            for week, prediction_path in zip(
                (16, 17, 18), sorted(seal["predictions"]), strict=True
            ):
                label_path = f"data/raw/train/output_2023_w{week:02d}.csv"
                if sha256(root / label_path) != seal["provenance"]["outcome_hashes"][label_path]:
                    raise ValueError("Holdout outcome checksum differs from the original snapshot.")
                truth = pd.read_csv(root / label_path)
                with np.load(root / prediction_path, allow_pickle=False) as saved:
                    keys = pd.DataFrame(saved["keys"], columns=KEYS)
                    for scenario in SCENARIOS:
                        error = score_predictions(truth, keys, saved[scenario])
                        error["player_role"] = saved["roles"]
                        errors.append(
                            error.assign(
                                scenario=scenario,
                                week=week,
                                forecast_second=np.ceil(error.frame_id / 10).astype(int),
                            )
                        )
            frame = pd.concat(errors, ignore_index=True)
            primary = frame.loc[frame.scenario.eq("complete")]
            validate_forecast_partition(primary[KEYS], seal["games"])
            if len(primary) != seal["rows"]:
                raise ValueError("Final evaluation row count differs from the sealed predictions.")
            metrics = []
            for scenario in SCENARIOS:
                part = frame.loc[frame.scenario.eq(scenario)]
                ci: Any = EVALUATION["confidence_interval"]
                scores = bootstrap_scores(part, ci["resamples"], ci["seed"])
                metrics.append(
                    {
                        "scenario": scenario,
                        **error_metrics(part),
                        "coordinate_rmse_95_interval": np.quantile(
                            scores, ci["quantiles"]
                        ).tolist(),
                    }
                )
            slices = [
                {"group": group, "value": str(value), "rows": len(part), **error_metrics(part)}
                for group in EVALUATION["descriptive_slices"]
                for value, part in primary.groupby(group, sort=True)
            ]
            atomic_bytes(paths[1], frame.to_csv(index=False).encode())
            atomic_json(
                paths[0],
                {
                    "status": "passed",
                    "seal": seal["source_signature"],
                    "fit_source": final_bundle(root)["source_signature"],
                    "protocol_source": load_protocol(root)["source_signature"],
                    "holdout_evaluation": "completed",
                    "primary_scenario": "complete",
                    "rows": len(primary),
                    "games": len(seal["games"]),
                    "metrics": metrics,
                    "slices": slices,
                    "rules": EVALUATION,
                    "outcome_hashes": seal["provenance"]["outcome_hashes"],
                    "errors_sha256": sha256(paths[1]),
                    "limitations": (
                        "One labelled 2023 season; the bootstrap measures game sampling "
                        "uncertainty, not cross-season generalization. Availability scenarios "
                        "are descriptive, not model selection."
                    ),
                },
            )

        stage(root, "final-holdout-evaluation", seal["source_signature"], paths, action, run)
        return dict(json.loads(paths[0].read_text()))
