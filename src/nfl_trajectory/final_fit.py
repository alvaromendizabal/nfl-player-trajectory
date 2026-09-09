"""Fit the frozen final tree schemas with independent coordinate checkpoints."""

from __future__ import annotations

import io
import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
from nfl_trajectory.feature_research import verify_inputs
from nfl_trajectory.final_features import feature_groups, load_preprocessing, materialize
from nfl_trajectory.final_protocol import PROFILES, digest, load_protocol
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage
from nfl_trajectory.tree_inference import tree_correction

Fit = Callable[[np.ndarray, np.ndarray, dict[str, Any]], Any]
Convert = Callable[[list[Any], list[str]], dict[str, Any]]


def memory_budget_gib() -> float:
    limits = [os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")]
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if path.is_file() and path.read_text().strip().isdecimal():
            limits.append(int(path.read_text().strip()))
    return min(limits) / 1024**3


def validate_profiles(profiles: dict[str, list[str]]) -> None:
    if set(profiles) != set(PROFILES):
        raise ValueError("Both frozen final availability schemas are required.")
    for profile in PROFILES:
        feature_groups(profiles[profile])
        if any(metadata_dependent(name) for name in profiles[profile]):
            raise ValueError("Final schemas cannot depend on optional metadata.")
    if profiles[PROFILES[1]] != [n for n in profiles[PROFILES[0]] if not telemetry_dependent(n)]:
        raise ValueError("Final fallback must preserve the frozen positional subset and order.")


def fit_profile(
    root: Path,
    profile: str,
    names: list[str],
    source: str,
    settings: dict[str, Any],
    data: Callable[[], tuple[np.ndarray, dict[str, np.ndarray]]],
    fit: Fit,
    convert: Convert,
    run: Run,
) -> dict[str, Any]:
    """A failed coordinate or export never discards a verified completed coordinate fit."""
    folder = root / "artifacts/final/models" / profile
    axes = [folder / (axis + ".pkl") for axis in ("x", "y")]
    for axis, path in enumerate(axes):

        def action(axis: int = axis, path: Path = path) -> None:
            matrix, arrays = data()
            if matrix.shape != (len(arrays["residual"]), len(names)):
                raise ValueError("Final estimator received the wrong schema width or target rows.")
            model = fit(matrix, arrays["residual"][:, axis], settings)
            if model.n_features_in_ != len(names):
                raise ValueError("Final estimator did not fit the complete frozen schema.")
            atomic_bytes(path, pickle.dumps(model, protocol=5))

        stage(root, f"final-fit-{profile}-{axis}", source, [path], action, run)

    outputs = [folder / name for name in ("tree.json", "training_predictions.npz", "summary.json")]
    axis_hashes = {path.name: sha256(path) for path in axes}
    export_source = digest({"fit_source": source, "axes": axis_hashes, "features": names})

    def export() -> None:
        # Each pickle was hash-verified by its completed fit stage immediately above.
        models = [pickle.loads(path.read_bytes()) for path in axes]
        portable = convert(models, names)
        lookup = {name: index for index, name in enumerate(names)}
        selected = [lookup[name] for name in portable["features"]]
        if portable["screened_feature_count"] != len(names) or len(set(selected)) != len(selected):
            raise ValueError("Portable conversion changed the fitted feature contract.")
        matrix, arrays = data()
        prediction = np.empty((len(matrix), 2))
        difference = 0.0
        for start in range(0, len(matrix), 4096):
            end = min(start + 4096, len(matrix))
            batch = matrix[start:end]
            expected = np.column_stack([model.predict(batch) for model in models])
            actual = tree_correction(batch[:, selected], portable)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            difference = max(difference, float(np.max(np.abs(actual - expected))))
            prediction[start:end] = (
                arrays["baseline"][start:end] + arrays["sign"][start:end] * actual
            )
        if not np.isfinite(prediction).all():
            raise ValueError("Final model produced nonfinite training predictions.")
        payload = io.BytesIO()
        np.savez_compressed(payload, keys=arrays["keys"], prediction=prediction)
        atomic_json(outputs[0], portable)
        atomic_bytes(outputs[1], payload.getvalue())
        atomic_json(
            outputs[2],
            {
                "status": "passed",
                "profile": profile,
                "source_signature": source,
                "export_signature": export_source,
                "training_rows": len(matrix),
                "fitted_features": len(names),
                "active_features": len(selected),
                "schema_sha256": digest(names),
                "portable_max_absolute_difference": difference,
                "training_prediction_scope": "Conversion integrity; no generalization claim.",
                "holdout_evaluation": "not_run",
                "axis_sha256": axis_hashes,
                "tree_sha256": sha256(outputs[0]),
                "predictions_sha256": sha256(outputs[1]),
            },
        )

    stage(root, "final-export-" + profile, export_source, outputs, export, run)
    return dict(json.loads(outputs[2].read_text()))


def fit_models(
    root: Path,
    run: Run,
    fit: Fit,
    convert: Convert,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Preserve the original research model; fit only the authorized final partition."""
    folder = root / "artifacts/final"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1):
        if (folder / "model_seal.json").exists():
            raise ValueError("Final model is sealed; refitting is forbidden.")
        protocol = load_protocol(root)
        preprocessing = load_preprocessing(root, protocol)
        contract = protocol["contract"]
        profiles = contract["availability_profile_features"]
        validate_profiles(profiles)
        if set(contract["training_games"]) & set(contract["holdout_games"]):
            raise ValueError("Final training cannot contain reserved holdout games.")
        if environment.get("scikit-learn") != "1.8.0":
            raise ValueError("Final fitting requires the validated scikit-learn 1.8.0 environment.")
        caches, _ = verify_inputs(root)
        files = [
            "src/nfl_trajectory/final_features.py",
            "src/nfl_trajectory/final_fit.py",
            "scripts/fit_final.py",
            "scripts/fit_final.py.lock",
            "scripts/prepare_tree.py",
        ]
        provenance = {
            "protocol_source": protocol["source_signature"],
            "protocol_sha256": sha256(folder / "protocol.json"),
            "preprocessing_source": preprocessing["source_signature"],
            "preprocessing_sha256": sha256(folder / "preprocessing/summary.json"),
            "implementation": {name: sha256(root / name) for name in files},
            "environment": environment,
        }
        source = digest(provenance)
        plan = {
            "source_signature": source,
            "provenance": provenance,
            "profiles": profiles,
            "settings": contract["estimator"]["settings"],
            "holdout_evaluation": "not_run",
        }
        plan_path = folder / "fit_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("The final fit plan changed; existing fits cannot be reused.")
        else:
            atomic_json(plan_path, plan)
        loaded: tuple[np.ndarray, dict[str, np.ndarray]] | None = None

        def training() -> tuple[np.ndarray, dict[str, np.ndarray]]:
            nonlocal loaded
            if loaded is None:
                rows = protocol["training_inventory"]["training_rows"]
                # 32 bytes/entry budgets sklearn's float64, binning, column-selection,
                # and working copies; an additional 8 GiB covers the feature banks/runtime.
                required = 8 + rows * len(profiles[PROFILES[0]]) * 32 / 1024**3
                if memory_budget_gib() < required:
                    raise ValueError(
                        f"Final fitting needs {required:.1f} GiB memory; use a 128 GiB worker."
                    )
                loaded = materialize(root, caches, protocol, source, run)
            return loaded

        results = []
        for profile in PROFILES:
            selected_data: tuple[np.ndarray, dict[str, np.ndarray]] | None = None

            def profile_data(profile: str = profile) -> tuple[np.ndarray, dict[str, np.ndarray]]:
                nonlocal selected_data
                if selected_data is None:
                    matrix, arrays = training()
                    lookup = {name: i for i, name in enumerate(profiles[PROFILES[0]])}
                    selected_data = (
                        matrix
                        if profile == PROFILES[0]
                        else matrix[:, [lookup[n] for n in profiles[profile]]],
                        arrays,
                    )
                return selected_data

            results.append(
                fit_profile(
                    root,
                    profile,
                    profiles[profile],
                    source,
                    plan["settings"],
                    profile_data,
                    fit,
                    convert,
                    run,
                )
            )
            selected_data = None
        if load_protocol(root) != protocol or load_preprocessing(root, protocol) != preprocessing:
            raise ValueError("Final fit inputs changed during fitting.")
        report = {
            "status": "passed",
            "source_signature": source,
            "provenance": provenance,
            "training_games": len(contract["training_games"]),
            "profiles": results,
            "final_fit": True,
            "model_sealed": False,
            "holdout_evaluation": "not_run",
            "remaining": "Validate inference, seal predictions, evaluate reserved outcomes.",
        }
        atomic_json(folder / "models/summary.json", report)
        run.event("final_fits_completed", source_signature=source, profiles=len(results))
        return report
