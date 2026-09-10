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
"""Run the frozen matched experiment of learned motion supervision."""

from __future__ import annotations

import argparse
import hashlib
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

from domain_research import fingerprint  # noqa: E402
from domain_research import specification as parent_specification  # noqa: E402
from evaluate_domain_research import comparison, decision  # noqa: E402
from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.motion_supervision import (  # noqa: E402
    ARMS,
    SETTINGS,
    MotionSupervisionModel,
    evaluate,
    fit_scales,
    train,
)
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256  # noqa: E402


def specification() -> dict[str, Any]:
    parent = parent_specification()
    paths = [
        "scripts/motion_supervision.py",
        "scripts/motion_supervision.py.lock",
        "src/nfl_trajectory/motion_supervision.py",
        "src/nfl_trajectory/benchmark.py",
        "docs/MOTION_SUPERVISION_EXPERIMENT.md",
        "scripts/evaluate_domain_research.py",
        "scripts/evaluate_temporal.py",
    ]
    return {
        "version": 1,
        "folds": parent["folds"],
        "settings": SETTINGS,
        "arms": ARMS,
        "inputs": {**parent["inputs"], **parent["sources"]},
        "sources": {p: sha256(ROOT / p) for p in paths},
        "feature_completion_gate": "open",
        "parent_signature": fingerprint(parent),
    }


def run_arm(
    name: str,
    arm: str,
    samples: list[dict[str, np.ndarray]],
    fitted: dict[str, Any],
    folder: Path,
    signature: str,
    run: Run,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    digest = fingerprint({"parent": signature, "fold": name, "arm": arm})
    path = folder / "summary.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["status"] != "completed" or old["signature"] != digest:
            raise ValueError("Completed fit has a different signature.")
        if any(sha256(folder / p) != h for p, h in old["artifacts"].items()):
            raise ValueError("Completed fitted artifact changed.")
        run.event("motion_supervision_arm_reused", fold=name, arm=arm)
        return pd.read_csv(folder / "errors.csv"), old
    torch.manual_seed(SETTINGS["seed"])
    model = MotionSupervisionModel(SETTINGS["width"])
    initial = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in model.state_dict().values())
    ).hexdigest()
    run.event("motion_supervision_arm_started", fold=name, arm=arm)
    model, state = train(model, samples, fitted, arm == "motion_supervised", folder, digest, run)
    if state["epoch"] != SETTINGS["epochs"]:
        raise ValueError("Training did not complete the declared budget.")
    errors = evaluate(model, [s for s in samples if str(s["split"]) == "validation"])
    atomic_bytes(folder / "errors.csv", errors.to_csv(index=False).encode())
    report = {
        "status": "completed",
        "signature": digest,
        "fold": name,
        "arm": arm,
        "epochs": state["epoch"],
        "training_seconds": state["seconds"],
        "curve": state["curve"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial,
        **error_metrics(errors),
        "artifacts": {
            p: sha256(folder / p) for p in ("checkpoint.pt", "checkpoint.json", "errors.csv")
        },
    }
    atomic_json(path, report)
    run.event("motion_supervision_arm_completed", fold=name, arm=arm, **error_metrics(errors))
    return errors, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import pytest

        raise SystemExit(pytest.main([str(ROOT / "tests/test_motion_supervision.py"), "-q"]))
    folder = ROOT / "artifacts/motion_supervision"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(ROOT, "motion-supervision") as run:
        plan = specification()
        signature = fingerprint(plan)
        path = folder / "plan.json"
        if path.exists() and json.loads(path.read_text())["signature"] != signature:
            raise ValueError("The numerical plan changed; existing fits cannot be overwritten.")
        atomic_json(path, {"signature": signature, **plan})
        torch.set_num_threads(SETTINGS["threads"])
        accumulated: dict[str, list[pd.DataFrame]] = {}
        folds: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        for fold in plan["folds"]:
            name = fold["name"]
            source = ROOT / f"artifacts/temporal/research/{name}"
            # Parent specification verifies these private pickle bytes before deserialization.
            samples = pickle.loads((source / "samples.pkl").read_bytes())
            destination = folder / name
            scales_path = destination / "scales.json"
            if scales_path.exists():
                fitted = json.loads(scales_path.read_text())
                if fitted["signature"] != signature:
                    raise ValueError("The training-only scale receipt changed.")
            else:
                fitted = {"signature": signature, **fit_scales(samples)}
                atomic_json(scales_path, fitted)
            values = {
                "attention": pd.read_csv(source / "full/errors.csv"),
                "tree": pd.read_csv(
                    ROOT / f"artifacts/feature_attribution/{name}/without_metadata.csv"
                ),
            }
            summaries = {}
            for arm in ARMS:
                values[arm], summaries[arm] = run_arm(
                    name, arm, samples, fitted, destination / arm, signature, run
                )
            if (
                summaries[ARMS[0]]["initial_state_sha256"]
                != summaries[ARMS[1]]["initial_state_sha256"]
            ):
                raise ValueError("Matched arms did not start with identical parameters.")
            for arm in ("attention", *ARMS):
                tree, other = aligned_errors(values["tree"], values[arm])
                blended = tree.copy()
                blended[["dx", "dy"]] = (
                    tree[["dx", "dy"]].to_numpy() + other[["dx", "dy"]].to_numpy()
                ) / 2
                values[arm + "_equal_blend"] = blended
            primary = comparison(values["motion_supervised"], values["coordinate"])
            folds.append(
                {
                    "fold": name,
                    "rows": len(values["coordinate"]),
                    "primary": primary,
                    "scales": fitted,
                    "scores": {a: error_metrics(v) for a, v in values.items()},
                    "fits": summaries,
                }
            )
            for arm, errors in values.items():
                accumulated.setdefault(arm, []).append(errors)
            for path in destination.rglob("*"):
                if path.is_file() and path.suffix != ".lock":
                    artifacts[str(path.relative_to(ROOT))] = sha256(path)
        pooled = {a: pd.concat(parts, ignore_index=True) for a, parts in accumulated.items()}
        primary = comparison(pooled["motion_supervised"], pooled["coordinate"])
        slices = []
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 100)):
            entry: dict[str, Any] = {"seconds": [lo, hi]}
            for arm in ("attention", *ARMS):
                frame = pooled[arm]
                selected = frame[frame.frame_id.gt(lo * 10) & frame.frame_id.le(hi * 10)]
                entry[arm] = {"rows": len(selected), **error_metrics(selected)}
            slices.append(entry)
        report = {
            "status": "completed",
            "signature": signature,
            "sources": plan["sources"],
            "inputs": plan["inputs"],
            "settings": SETTINGS,
            "run_id": run.run_id,
            "feature_completion_gate": "open",
            "fits": 6,
            "rows": len(pooled["coordinate"]),
            "folds": folds,
            "pooled": {a: error_metrics(v) for a, v in pooled.items()},
            "primary": primary,
            "primary_gate_passed": decision(folds, primary),
            "secondary_reference": comparison(pooled["motion_supervised"], pooled["attention"]),
            "secondary_blend": comparison(
                pooled["motion_supervised_equal_blend"], pooled["attention_equal_blend"]
            ),
            "horizon_slices": slices,
            "artifacts": artifacts,
        }
        if specification() != plan:
            raise ValueError("Numerical source or inputs changed during the study.")
        atomic_json(folder / "report.json", report)
        atomic_json(ROOT / "docs/results/motion_supervision.json", report)
        run.event(
            "motion_supervision_study_completed",
            primary=primary,
            primary_gate_passed=report["primary_gate_passed"],
        )


if __name__ == "__main__":
    main()
