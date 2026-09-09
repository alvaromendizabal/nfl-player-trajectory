# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "filelock==3.32.5",
#   "plotly==7.0.0", "matplotlib==3.10.8",
#   "nbformat==5.11.1",
#   "polars>=1.32,<2", "pyarrow>=20,<24", "grpcio>=1.73,<2", "protobuf>=5.29,<7",
# ]
# ///
"""Validate final inference with the unchanged, unlabelled organizer gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from validate_gateway import ORGANIZER_HASHES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data/raw"))


def main(root: Path, publish: bool = False) -> dict[str, Any]:
    from nfl_trajectory.final_inference import final_bundle, predict_final, standalone_source
    from nfl_trajectory.final_protocol import digest
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    raw = root / "data/raw"
    for name, expected in ORGANIZER_HASHES.items():
        if sha256(raw / name) != expected:
            raise ValueError("Organizer source or unlabelled sample checksum changed.")
    bundle = final_bundle(root)
    code = standalone_source(root, bundle)
    inputs = {
        str((raw / name).relative_to(root)): value for name, value in ORGANIZER_HASHES.items()
    }
    for name in ("validate_final.py", "validate_final.py.lock", "validate_gateway.py"):
        inputs["scripts/" + name] = sha256(root / "scripts" / name)
    provenance = {"bundle_sha256": digest(bundle), "inputs": inputs}
    source = digest(provenance)
    folder = root / "artifacts/final/inference"
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [
        folder / name
        for name in ("summary.json", "bundle.json", "standalone.py", "submission.parquet")
    ]

    with Run(root, "final-gateway-validation") as run:

        def action() -> None:
            from kaggle_evaluation.nfl_inference_server import NFLInferenceServer

            request = pd.read_csv(raw / "test.csv")
            if {"x", "y"} & set(request.columns) or request.id.duplicated().any():
                raise ValueError("Final validation requires unique unlabelled sample rows.")
            namespace: dict[str, Any] = {}
            exec(compile(code, "final-standalone.py", "exec"), namespace)
            latencies: list[float] = []
            predictions: list[pd.DataFrame] = []
            stress_rows = 0

            def predict(test: Any, test_input: Any) -> pd.DataFrame:
                nonlocal stress_rows
                targets = test.to_pandas() if hasattr(test, "to_pandas") else test
                observed = (
                    test_input.to_pandas() if hasattr(test_input, "to_pandas") else test_input
                )
                started = time.monotonic()
                actual = namespace["final_predict"](
                    observed, targets[KEYS], namespace["FINAL_MODEL"]
                )
                latencies.append(time.monotonic() - started)
                expected = predict_final(observed, targets[KEYS], bundle)
                pd.testing.assert_frame_equal(actual, expected)
                if not predictions:
                    for scenario in ("without_metadata", "without_telemetry", "cold_history"):
                        current = observed.drop(
                            columns=(
                                [
                                    "player_height",
                                    "player_weight",
                                    "player_birth_date",
                                    "player_position",
                                ]
                                if scenario == "without_metadata"
                                else ["s", "a", "o", "dir"]
                                if scenario == "without_telemetry"
                                else []
                            ),
                            errors="ignore",
                        )
                        kwargs = {"cold_history": scenario == "cold_history"}
                        stress = predict_final(current, targets[KEYS], bundle, **kwargs)
                        pd.testing.assert_frame_equal(
                            namespace["final_predict"](
                                current, targets[KEYS], namespace["FINAL_MODEL"], **kwargs
                            ),
                            stress,
                        )
                        if scenario == "without_metadata":
                            pd.testing.assert_frame_equal(stress, expected)
                        stress_rows += len(stress)
                    shuffled = (
                        targets[KEYS].sample(frac=1, random_state=2026).assign(x=np.nan, y=np.inf)
                    )
                    reordered = predict_final(
                        observed.sample(frac=1, random_state=2026), shuffled, bundle
                    )
                    pd.testing.assert_frame_equal(
                        reordered.sort_values(KEYS).reset_index(drop=True),
                        expected.sort_values(KEYS).reset_index(drop=True),
                    )
                output = actual[["x", "y"]].reset_index(drop=True)
                predictions.append(output.assign(id=targets.id.to_numpy()))
                if len(predictions) % 25 == 0:
                    run.event(
                        "final_gateway_progress",
                        plays=len(predictions),
                        rows=sum(map(len, predictions)),
                    )
                return output

            previous = Path.cwd()
            try:
                os.chdir(folder)
                outputs[3].unlink(missing_ok=True)
                NFLInferenceServer(predict).run_local_gateway((str(raw),))
            finally:
                os.chdir(previous)
            actual = pd.read_parquet(outputs[3])
            expected = pd.concat(predictions, ignore_index=True)
            if (
                len(actual) != len(request)
                or set(actual.id) != set(request.id)
                or actual.id.duplicated().any()
            ):
                raise ValueError("Final gateway output does not cover each request exactly once.")
            pd.testing.assert_frame_equal(
                actual[["x", "y", "id"]], expected[["x", "y", "id"]], check_dtype=False
            )
            atomic_json(outputs[1], bundle)
            atomic_bytes(outputs[2], code.encode())
            atomic_json(
                outputs[0],
                {
                    "status": "passed",
                    "source_signature": source,
                    "provenance": provenance,
                    "fit_source": bundle["source_signature"],
                    "protocol_source": bundle["protocol_source"],
                    "sample_rows": len(actual),
                    "plays": len(predictions),
                    "stress_rows": stress_rows,
                    "standalone_prediction_parity": "exact on every organizer callback",
                    "stress_checks": [
                        "metadata invariant",
                        "telemetry fallback",
                        "cold history",
                        "row order",
                        "target coordinates ignored",
                    ],
                    "callback_seconds_median": float(np.median(latencies)),
                    "callback_seconds_p95": float(np.quantile(latencies, 0.95)),
                    "callback_seconds_max": float(max(latencies)),
                    "organizer_response_limit_seconds": 300,
                    "labels_available": False,
                    "evaluation_metric": None,
                    "holdout_evaluation": "not_run",
                    "competition_submission": "not_run",
                    "outputs": {path.name: sha256(path) for path in outputs[1:]},
                },
            )

        stage(root, "final-gateway", source, outputs, action, run)
        report = json.loads(outputs[0].read_text())
        if publish:
            atomic_json(root / "docs/results/final_inference.json", report)
        return dict(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    main(ROOT, parser.parse_args().publish)
