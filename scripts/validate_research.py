"""Verify selected research inference on every development frame and input stress scenarios."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from nfl_trajectory.feature_experiment import load_week
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.research_inference import input_variant, predict_research, research_bundle
    from nfl_trajectory.runtime import Run, atomic_json, sha256, stage

    root = ROOT
    with Run(root, "research-inference-validation") as run:
        caches, inputs = verify_inputs(root)
        bundle = research_bundle(root)
        evaluation = set(bundle["evaluation_games"])
        folder = root / "artifacts/research/inference"
        before = root / "artifacts/research/inference_before_fallback.json"
        previous = folder / "summary.json"
        if previous.exists() and not before.exists():
            atomic_json(before, json.loads(previous.read_text()))
        raw_paths = [root / "data/raw/train" / (c.parent.name + ".csv") for c in caches]
        manifest = json.loads((root / "artifacts/research/input_manifest.json").read_text())
        expected = {item["path"]: item["sha256"] for item in manifest["files"]}
        for path in raw_paths:
            if sha256(path) != expected.get(str(path.relative_to(root))):
                raise ValueError("Raw inference data differs from the verified source snapshot.")
        inputs.update({str(p.relative_to(root)): sha256(p) for p in raw_paths})
        source = hashlib.sha256(
            json.dumps(
                {
                    "bundle": bundle,
                    "inputs": inputs,
                    "validator": sha256(Path(__file__)),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        results = []
        first_play = None
        for cache in caches:
            _, targets, arrays = load_week(cache)
            choose = targets.game_id.isin(evaluation).to_numpy()
            if not choose.any():
                continue
            targets = targets.loc[choose, KEYS].reset_index(drop=True)
            truth = arrays["truth"][choose]
            observed = pd.read_csv(root / "data/raw/train" / (cache.parent.name + ".csv"))
            observed = observed[observed.game_id.isin(evaluation)].reset_index(drop=True)
            destination = folder / (cache.parent.name + ".json")

            def action(
                data: pd.DataFrame = observed,
                request: pd.DataFrame = targets,
                actual: np.ndarray = truth,
                output: Path = destination,
            ) -> None:
                scenarios = {}
                for scenario in (
                    "complete_inputs",
                    "missing_metadata",
                    "missing_telemetry",
                    "cold_player_history",
                ):
                    current, model = data, bundle
                    if scenario == "missing_metadata":
                        current = data.drop(
                            columns=[
                                "player_height",
                                "player_weight",
                                "player_birth_date",
                                "player_position",
                            ],
                            errors="ignore",
                        )
                    elif scenario == "missing_telemetry":
                        current = data.drop(columns=["s", "a", "dir", "o"], errors="ignore")
                    elif scenario == "cold_player_history":
                        model = copy.deepcopy(bundle)
                        model["history"]["tables"]["player"] = {}
                    started = time.monotonic()
                    predicted = predict_research(current, request, model)[["x", "y"]].to_numpy()
                    error = predicted - actual
                    scenarios[scenario] = {
                        "variant": input_variant(current, model),
                        "squared_coordinate_error": float(np.sum(error**2)),
                        "coordinate_count": int(error.size),
                        "elapsed_seconds": time.monotonic() - started,
                        "rows": len(request),
                    }
                    run.event("inference_scenario_completed", scenario=scenario, rows=len(request))
                atomic_json(output, scenarios)

            stage(
                root, "research-inference-" + cache.parent.name, source, [destination], action, run
            )
            results.append(json.loads(destination.read_text()))
            if first_play is None:
                game, play = targets.iloc[0][["game_id", "play_id"]]
                first_play = (
                    observed[observed.game_id.eq(game) & observed.play_id.eq(play)],
                    targets[targets.game_id.eq(game) & targets.play_id.eq(play)],
                )
        pooled = []
        for scenario in results[0]:
            count = sum(r[scenario]["coordinate_count"] for r in results)
            sse = sum(r[scenario]["squared_coordinate_error"] for r in results)
            pooled.append(
                {
                    "scenario": scenario,
                    "variant": results[0][scenario]["variant"],
                    "coordinate_rmse_yards": float(np.sqrt(sse / count)),
                    "rows": count // 2,
                    "elapsed_seconds": sum(r[scenario]["elapsed_seconds"] for r in results),
                }
            )
        measured = pooled[0]["coordinate_rmse_yards"]
        if not np.isclose(
            measured, bundle["validation_coordinate_rmse_yards"], rtol=1e-8, atol=1e-9
        ):
            raise ValueError(
                "Research inference does not reproduce its measured development score."
            )
        spec = importlib.util.spec_from_file_location(
            "research_export_validation", root / "kaggle/export.py"
        )
        if spec is None or spec.loader is None or first_play is None:
            raise ValueError("Research export validation requires code and an observed play.")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        code, _ = exporter.model_source(root, "research", root / "artifacts/benchmark/model.json")
        namespace: dict[str, Any] = {}
        exec(compile(code, "standalone-research.py", "exec"), namespace)
        reference = predict_research(*first_play, bundle)
        latencies = []
        for _ in range(3):
            started = time.monotonic()
            actual = namespace["research_predict"](*first_play, namespace["RESEARCH_MODEL"])
            latencies.append(time.monotonic() - started)
            pd.testing.assert_frame_equal(actual, reference)
        receipt = {
            "status": "passed",
            "validation_signature": source,
            "validator_sha256": sha256(Path(__file__)),
            "bundle_sha256": hashlib.sha256(
                json.dumps(bundle, sort_keys=True).encode()
            ).hexdigest(),
            "weekly_receipts": [
                str((folder / (c.parent.name + ".json")).relative_to(root))
                for c in caches
                if (folder / (c.parent.name + ".json")).exists()
            ],
            "selected_stage": bundle["selected_stage"],
            "selected_model": bundle["selected_model"],
            "source_signatures": bundle["source_signatures"],
            "inference_sources": bundle["inference_sources"],
            "retained_features": bundle["retained_features"],
            "scenarios": pooled,
            "standalone_export_parity": (
                "passed on one complete development play; source shared with all-frame package test"
            ),
            "single_play_seconds": latencies,
            "single_play_rows": len(first_play[1]),
            "gateway_status": "not_run",
            "holdout_evaluation": "not_run",
            "limitations": (
                "Input stress tests on previously inspected development data; "
                "cold history is not a player-disjoint refit."
            ),
        }
        atomic_json(folder / "summary.json", receipt)
        atomic_json(folder / "model.json", bundle)
        run.event(
            "research_inference_validation_completed",
            rows=pooled[0]["rows"],
            retained_features=bundle["retained_features"],
        )


if __name__ == "__main__":
    main()
