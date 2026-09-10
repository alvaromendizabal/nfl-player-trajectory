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
"""Execute the fixed, matched continuation test of smoothed temporal inputs."""

from __future__ import annotations

import argparse
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

from domain_research import fingerprint  # noqa: E402
from domain_research import specification as parent_specification  # noqa: E402
from evaluate_domain_research import comparison, decision  # noqa: E402
from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.motion_representation import (  # noqa: E402
    ARMS,
    SETTINGS,
    MotionModel,
    evaluate,
    fit_motion_screen,
    motion_state,
    train,
)
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage  # noqa: E402
from nfl_trajectory.temporal_model import load_checkpoint  # noqa: E402


def specification() -> dict[str, Any]:
    parent = parent_specification()
    sources = [
        "scripts/motion_representation.py",
        "scripts/motion_representation.py.lock",
        "src/nfl_trajectory/motion_representation.py",
        "src/nfl_trajectory/motion_research.py",
        "src/nfl_trajectory/benchmark.py",
        "docs/REPRESENTATION_EXPERIMENT.md",
        "scripts/evaluate_domain_research.py",
        "scripts/evaluate_temporal.py",
    ]
    return {
        "version": 1,
        "folds": parent["folds"],
        "settings": SETTINGS,
        "arms": ARMS,
        "inputs": {**parent["inputs"], **parent["sources"]},
        "sources": {p: sha256(ROOT / p) for p in sources},
        "feature_completion_gate": "open",
        "parent_signature": fingerprint(parent),
    }


def prepare(
    samples: list[dict[str, np.ndarray]], folder: Path, signature: str, run: Run
) -> tuple[list[dict[str, np.ndarray]], list[str], dict[str, Any]]:
    paths = [folder / "motion.npz", folder / "screen.json", folder / "schema.json"]

    def build() -> None:
        matrices = []
        names: list[str] = []
        for sample in samples:
            values, names = motion_state(sample)
            matrices.append(values)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, values=np.concatenate(matrices), counts=[len(x) for x in matrices]
        )
        augmented = [{**s, "motion": m} for s, m in zip(samples, matrices, strict=True)]
        atomic_bytes(paths[0], buffer.getvalue())
        atomic_json(paths[1], fit_motion_screen(augmented, names))
        atomic_json(
            paths[2],
            {
                "names": names,
                "training_players": sum(
                    len(s["ids"]) for s in samples if str(s["split"]) == "train"
                ),
            },
        )

    stage(ROOT, folder.name + "-motion-representation", signature, paths, build, run)
    with np.load(paths[0], allow_pickle=False) as archive:
        counts = [len(s["ids"]) for s in samples]
        if not np.array_equal(archive["counts"], counts):
            raise ValueError("Motion cache player counts changed.")
        matrices = np.split(archive["values"], np.cumsum(counts)[:-1])
    augmented = [{**s, "motion": m} for s, m in zip(samples, matrices, strict=True)]
    return augmented, json.loads(paths[2].read_text())["names"], json.loads(paths[1].read_text())


def run_arm(
    name: str,
    arm: str,
    samples: list[dict[str, np.ndarray]],
    names: list[str],
    fitted: dict[str, Any],
    initial: dict[str, torch.Tensor],
    folder: Path,
    signature: str,
    run: Run,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    digest = fingerprint({"parent": signature, "fold": name, "arm": arm})
    result_path = folder / "summary.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["status"] != "completed" or old["signature"] != digest:
            raise ValueError("A completed representation fit has a different signature.")
        if any(sha256(folder / p) != h for p, h in old["artifacts"].items()):
            raise ValueError("A completed representation artifact changed.")
        run.event("representation_arm_reused", fold=name, arm=arm)
        return pd.read_csv(folder / "errors.csv"), old
    torch.manual_seed(SETTINGS["seed"])
    model = MotionModel(initial, fitted, arm == "smoothed_state", SETTINGS["width"])
    model, state = train(model, samples, names, folder, digest, run)
    if state["epoch"] != SETTINGS["epochs"]:
        raise ValueError("Representation training did not complete its declared budget.")
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
        **error_metrics(errors),
        "artifacts": {
            p: sha256(folder / p) for p in ("checkpoint.pt", "checkpoint.json", "errors.csv")
        },
    }
    atomic_json(result_path, report)
    run.event("representation_arm_completed", fold=name, arm=arm, **error_metrics(errors))
    return errors, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import pytest

        raise SystemExit(pytest.main([str(ROOT / "tests/test_motion_representation.py"), "-q"]))
    folder = ROOT / "artifacts/motion_representation"
    folder.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(folder / "pipeline.lock"), timeout=1),
        Run(ROOT, "motion-representation") as run,
    ):
        plan = specification()
        signature = fingerprint(plan)
        plan_path = folder / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text())["signature"] != signature:
            raise ValueError("The representation plan changed; do not overwrite existing fits.")
        atomic_json(plan_path, {"signature": signature, **plan})
        torch.set_num_threads(SETTINGS["threads"])
        accumulated: dict[str, list[pd.DataFrame]] = {}
        folds: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        for fold in plan["folds"]:
            name = fold["name"]
            run.event("representation_fold_started", fold=name)
            source = ROOT / f"artifacts/temporal/research/{name}"
            samples = pickle.loads((source / "samples.pkl").read_bytes())
            destination = folder / name
            samples, names, fitted = prepare(samples, destination, signature, run)
            old = json.loads((source / "full/summary.json").read_text())
            initial = load_checkpoint(source / "full/checkpoint.pt", old["signature"])["ema"]
            values = {
                "attention": pd.read_csv(source / "full/errors.csv"),
                "tree": pd.read_csv(
                    ROOT / f"artifacts/feature_attribution/{name}/without_metadata.csv"
                ),
            }
            replay_path = destination / "initial_replay.json"
            if not replay_path.exists():
                replay = evaluate(
                    MotionModel(initial, fitted, False).eval(),
                    [s for s in samples if str(s["split"]) == "validation"],
                )
                a, b = aligned_errors(values["attention"], replay)
                difference = float(
                    np.abs(a[["dx", "dy"]].to_numpy() - b[["dx", "dy"]].to_numpy()).max()
                )
                if difference > 0.00002:
                    raise ValueError(
                        "Expanded input does not reproduce the saved initial predictions."
                    )
                atomic_json(
                    replay_path, {"signature": signature, "max_absolute_difference": difference}
                )
            else:
                replay = json.loads(replay_path.read_text())
                if replay["signature"] != signature:
                    raise ValueError("Initial replay signature changed.")
            summaries = {}
            for arm in ARMS:
                values[arm], summaries[arm] = run_arm(
                    name, arm, samples, names, fitted, initial, destination / arm, signature, run
                )
            for arm in ("attention", *ARMS):
                tree, other = aligned_errors(values["tree"], values[arm])
                blended = tree.copy()
                blended[["dx", "dy"]] = (
                    tree[["dx", "dy"]].to_numpy() + other[["dx", "dy"]].to_numpy()
                ) / 2
                values[arm + "_equal_blend"] = blended
            primary = comparison(values["smoothed_state"], values["control"])
            folds.append(
                {
                    "fold": name,
                    "rows": len(values["control"]),
                    "primary": primary,
                    "screen": fitted,
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
        primary = comparison(pooled["smoothed_state"], pooled["control"])
        slices = []
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 100)):
            entry: dict[str, Any] = {"seconds": [lo, hi]}
            for arm in ARMS:
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
            "rows": len(pooled["control"]),
            "folds": folds,
            "pooled": {a: error_metrics(v) for a, v in pooled.items()},
            "primary": primary,
            "primary_gate_passed": decision(folds, primary),
            "secondary_blend": comparison(
                pooled["smoothed_state_equal_blend"], pooled["attention_equal_blend"]
            ),
            "horizon_slices": slices,
            "artifacts": artifacts,
        }
        if specification() != plan:
            raise ValueError("Numerical source or inputs changed during the study.")
        atomic_json(folder / "report.json", report)
        atomic_json(ROOT / "docs/results/motion_representation.json", report)
        run.event(
            "representation_study_completed",
            fits=6,
            primary=primary,
            primary_gate_passed=report["primary_gate_passed"],
        )


if __name__ == "__main__":
    main()
