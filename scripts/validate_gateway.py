# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "filelock==3.32.5",
#   "plotly==7.0.0", "matplotlib==3.10.8",
#   "nbformat==5.11.1",
#   "polars>=1.32,<2", "pyarrow>=20,<24", "grpcio>=1.73,<2", "protobuf>=5.29,<7",
# ]
# ///
"""Run the unchanged organizer gateway against its saved, unlabelled sample inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ORGANIZER_HASHES = {
    "kaggle_evaluation/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "kaggle_evaluation/core/__init__.py": (
        "7d04526624b1f71259b958206a343546f2ef31fd0ad6f0d606c6f2c8494b8874"
    ),
    "kaggle_evaluation/core/base_gateway.py": (
        "0b9f5e8c516202659513c0adc89b51778a1a383d8247137d9bfddce038ee06a5"
    ),
    "kaggle_evaluation/core/generated/__init__.py": (
        "7d04526624b1f71259b958206a343546f2ef31fd0ad6f0d606c6f2c8494b8874"
    ),
    "kaggle_evaluation/core/generated/kaggle_evaluation_pb2.py": (
        "04108189b4ac7751901a7fbaa95231ed7335b437c33706d5850cab868bd16a11"
    ),
    "kaggle_evaluation/core/generated/kaggle_evaluation_pb2_grpc.py": (
        "620d76b7b993a3fbf7809b1e7423ed71a94fe73156918215aeae86cb6f9b8a4f"
    ),
    "kaggle_evaluation/core/kaggle_evaluation.proto": (
        "cbc742d6b9e6f9a4981d8354cee79c7136edd8fe30cb90e1f6786ae72bc57ab1"
    ),
    "kaggle_evaluation/core/relay.py": (
        "b070f6f2c154b7ae5de2081125a0f27a0a7a577b46136a39d746091d323ae85e"
    ),
    "kaggle_evaluation/core/templates.py": (
        "3aa3eaf75139cd9220652b0601c24ef5f6188fcf44d4463b9b7e290271b5fa3a"
    ),
    "kaggle_evaluation/nfl_gateway.py": (
        "29335f26a9f4e8486c99c60072d824f11e4cf581591f588e6ecaf8cd15aa91bb"
    ),
    "kaggle_evaluation/nfl_inference_server.py": (
        "37ff289a1d2ab6f49fc4c0096c866b31dae1b7f98f0c9a7f361daff104b1da8e"
    ),
    "test.csv": ("6f45e50eb79442561cdb2cc6ba2d05b05f7967fad3964a98916c9a8ffecd44b2"),
    "test_input.csv": ("f894c3380da3caba2f0dbc0618ccf8e2a256e4e0fec56760fab050b03d964fd7"),
}


def main(root: Path) -> None:
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.research_inference import predict_research, research_bundle
    from nfl_trajectory.runtime import Run, atomic_json, sha256, stage

    raw = root / "data/raw"
    inputs = {}
    for name, expected in ORGANIZER_HASHES.items():
        path = raw / name
        if sha256(path) != expected:
            raise ValueError(
                "Organizer source or sample differs from the saved competition snapshot."
            )
        inputs[str(path.relative_to(root))] = expected
    bundle = research_bundle(root)
    for name, expected in bundle["inference_sources"].items():
        inputs["src/nfl_trajectory/" + name + ".py"] = expected
    for path in (Path(__file__), Path(__file__).with_suffix(".py.lock"), root / "kaggle/export.py"):
        inputs[str(path.relative_to(root))] = sha256(path)
    provenance = {
        "inputs": inputs,
        "bundle_sha256": hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
        "source_snapshot": ("c83a828f42f13af498443574284db8d6e014b44967bc4ee09ab7a49f4b362817"),
        "purpose": "Unlabelled official sample interface and standalone predictor parity only",
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    destination = root / "artifacts/research/gateway/summary.json"
    quality = root / "artifacts/quality/official_gateway"
    quality.mkdir(parents=True, exist_ok=True)

    def action() -> None:
        import polars as pl

        request = pd.read_csv(raw / "test.csv")
        if {"x", "y"} & set(request.columns) or request.id.duplicated().any():
            raise ValueError("Gateway validation requires unlabelled, unique request rows.")
        spec = importlib.util.spec_from_file_location(
            "gateway_export_check", root / "kaggle/export.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("The canonical exporter is required.")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        source, _ = exporter.model_source(root, "research", root / "artifacts/benchmark/model.json")
        namespace: dict[str, Any] = {}
        exec(compile(source, "standalone-gateway-check.py", "exec"), namespace)
        callbacks: list[dict[str, Any]] = []
        expected_predictions = []

        def predict(test: Any, test_input: Any) -> pd.DataFrame:
            targets = test.to_pandas() if isinstance(test, pl.DataFrame) else test
            observed = (
                test_input.to_pandas() if isinstance(test_input, pl.DataFrame) else test_input
            )
            if {"x", "y"} & set(targets.columns):
                raise ValueError("The organizer request unexpectedly includes target coordinates.")
            started = time.monotonic()
            prediction = namespace["research_predict"](
                observed, targets[KEYS], namespace["RESEARCH_MODEL"]
            )
            seconds = time.monotonic() - started
            reference = predict_research(observed, targets[KEYS], bundle)
            pd.testing.assert_frame_equal(prediction, reference)
            result = prediction[["x", "y"]].reset_index(drop=True)
            if len(result) != len(targets) or not np.isfinite(result.to_numpy()).all():
                raise ValueError("Standalone predictor violated the gateway shape contract.")
            callbacks.append({"rows": len(result), "elapsed_seconds": seconds})
            expected_predictions.append(result.assign(id=targets.id.to_numpy()))
            return result

        # Import the organizer only after loading weights, so startup timing is explicit.
        from kaggle_evaluation.nfl_inference_server import NFLInferenceServer

        previous = Path.cwd()
        try:
            os.chdir(quality)
            # This directory belongs to validation; it cannot overwrite an owner's export.
            (quality / "submission.parquet").unlink(missing_ok=True)
            server = NFLInferenceServer(predict)
            server.run_local_gateway((str(raw),))
        finally:
            os.chdir(previous)
        actual = pd.read_parquet(quality / "submission.parquet")
        expected = pd.concat(expected_predictions, ignore_index=True)
        if len(actual) != len(request) or list(actual.columns) != ["x", "y", "id"]:
            if set(actual.columns) != {"x", "y", "id"} or len(actual) != len(request):
                raise ValueError("Official gateway produced an incomplete or malformed output.")
        pd.testing.assert_frame_equal(
            actual[["x", "y", "id"]].reset_index(drop=True),
            expected[["x", "y", "id"]],
            check_dtype=False,
        )
        if set(actual.id) != set(request.id) or actual.id.duplicated().any():
            raise ValueError("Official gateway output lost or duplicated request identifiers.")
        latencies = [item["elapsed_seconds"] for item in callbacks]
        atomic_json(
            destination,
            {
                "status": "passed",
                "official_gateway_status": "passed",
                "source_signature": signature,
                "source_signatures": bundle["source_signatures"],
                "provenance": provenance,
                "selected_stage": bundle["selected_stage"],
                "selected_model": bundle["selected_model"],
                "retained_features": bundle["retained_features"],
                "sample_rows": len(actual),
                "plays": len(callbacks),
                "sample_calendar_years": sorted((request.game_id // 1000000).unique().tolist()),
                "callback_seconds_median": float(np.median(latencies)),
                "callback_seconds_p95": float(np.quantile(latencies, 0.95)),
                "callback_seconds_max": float(np.max(latencies)),
                "organizer_response_limit_seconds": 300,
                "standalone_prediction_parity": "exact for every sample callback",
                "output": str((quality / "submission.parquet").relative_to(root)),
                "output_sha256": sha256(quality / "submission.parquet"),
                "evaluation_metric": None,
                "labels_available": False,
                "competition_submission": "not_run",
                "holdout_evaluation": "not_run",
            },
        )

    with Run(root, "official-sample-gateway") as run:
        stage(
            root,
            "official-gateway",
            signature,
            [destination, quality / "submission.parquet"],
            action,
            run,
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "data/raw"))
    main(root)
