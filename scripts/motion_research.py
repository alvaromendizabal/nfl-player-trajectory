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
"""Identify the motion signal using cached inputs and matched correction heads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain_research import errors, fingerprint  # noqa: E402
from evaluate_domain_research import comparison  # noqa: E402
from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.domain_probe import SETTINGS, predict, train_probe, transform  # noqa: E402
from nfl_trajectory.motion import KEYS, require_keys  # noqa: E402
from nfl_trajectory.motion_research import ARMS, GROUPS, feature_group, keep_columns  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256  # noqa: E402
from nfl_trajectory.temporal_research import validate_fold  # noqa: E402


def specification() -> dict[str, Any]:
    parent_path = ROOT / "docs/results/domain_research.json"
    parent = json.loads(parent_path.read_text())
    if parent["status"] != "completed" or parent["fits"] != 30:
        raise ValueError("The parent feature study must be complete and verified.")
    inputs = {**parent["artifacts"], **parent["sources"]}
    inputs[str(parent_path.relative_to(ROOT))] = sha256(parent_path)
    plan = json.loads((ROOT / "artifacts/domain_research/plan.json").read_text())
    if plan["signature"] != parent["signature"]:
        raise ValueError("Parent report and protocol differ.")
    for name, digest in inputs.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Parent experiment artifact changed: " + name)
    sources = (
        "scripts/motion_research.py",
        "scripts/motion_research.py.lock",
        "scripts/evaluate_domain_research.py",
        "src/nfl_trajectory/motion_research.py",
        "docs/MOTION_EXPERIMENT.md",
    )
    return {
        "version": 1,
        "parent_signature": parent["signature"],
        "folds": plan["folds"],
        "settings": SETTINGS,
        "groups": GROUPS,
        "arms": ARMS,
        "inputs": inputs,
        "sources": {p: sha256(ROOT / p) for p in sources},
        "scope": "adaptive subfamily attribution on reused chronological development folds",
        "selection": "all additions and removals; final epoch; no minimum-score selection",
    }


def run_fold(fold: dict[str, Any], destination: Path, signature: str, run: Run) -> dict[str, Any]:
    name = fold["name"]
    parent = ROOT / "artifacts/domain_research" / name
    with np.load(parent / "features.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    schema = json.loads((parent / "schema.json").read_text())
    fitted = json.loads((parent / "screen.json").read_text())
    groups = Counter(feature_group(n) for n in schema["names"])
    if [groups[g] for g in GROUPS] != [5, 5, 30, 9, 8]:
        raise ValueError("The frozen motion feature partition changed.")
    train, evaluation = data["train"], ~data["train"]
    x = transform(data.pop("x"), fitted)
    references = {
        "control": parent / "control/errors.csv",
        "motion_state": parent / "motion_state/errors.csv",
        "tree": ROOT / f"artifacts/feature_attribution/{name}/without_metadata.csv",
        "temporal": ROOT / f"artifacts/temporal/research/{name}/full/errors.csv",
    }
    values = {a: pd.read_csv(p) for a, p in references.items()}
    artifacts = {str(p.relative_to(ROOT)): sha256(p) for p in references.values()}
    fits = []
    for arm in ARMS:
        folder = destination / arm
        folder.mkdir(parents=True, exist_ok=True)
        arm_signature = fingerprint({"signature": signature, "fold": name, "arm": arm})
        path = folder / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            if (
                summary["signature"] != arm_signature
                or summary["epochs_completed"] != SETTINGS["epochs"]
                or not all(sha256(folder / p) == h for p, h in summary["artifacts"].items())
            ):
                raise ValueError("Completed motion arm changed.")
            run.event("motion_arm_reused", fold=name, arm=arm)
        else:
            keep = keep_columns(schema["names"], fitted["retained"], arm)
            model, state = train_probe(
                np.ascontiguousarray(x[train] * keep),
                data["y"][train],
                data["time"][train],
                folder,
                arm_signature,
                run,
                arm,
                name,
            )
            correction = predict(
                model, np.ascontiguousarray(x[evaluation] * keep), data["time"][evaluation]
            )
            frame = errors(data, evaluation, correction)
            atomic_bytes(folder / "errors.csv", frame.to_csv(index=False).encode())
            summary = {
                "status": "completed",
                "signature": arm_signature,
                "fold": name,
                "arm": arm,
                "run_id": run.run_id,
                "settings": SETTINGS,
                "epochs_completed": state["epoch"],
                "training_seconds": state["seconds"],
                "curve": state["curve"],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "evaluation_rows": len(frame),
                **error_metrics(frame),
                "artifacts": {
                    p: sha256(folder / p)
                    for p in ("checkpoint.pt", "checkpoint.json", "errors.csv")
                },
            }
            atomic_json(path, summary)
            run.event("motion_arm_completed", fold=name, arm=arm, **error_metrics(frame))
        fits.append(summary)
        artifacts[str(path.relative_to(ROOT))] = sha256(path)
        artifacts.update(
            {str((folder / p).relative_to(ROOT)): h for p, h in summary["artifacts"].items()}
        )
        values[arm] = pd.read_csv(folder / "errors.csv")
    for arm in ("motion_state", "temporal"):
        first, tree = aligned_errors(values[arm], values["tree"])
        result = first[KEYS].copy()
        result[["dx", "dy"]] = (first[["dx", "dy"]].to_numpy() + tree[["dx", "dy"]].to_numpy()) / 2
        values[arm + "_equal_blend"] = result
    expected = pd.DataFrame(data["keys"][evaluation], columns=KEYS)
    for value in values.values():
        require_keys(value)
        if (
            not value[KEYS]
            .sort_values(KEYS)
            .reset_index(drop=True)
            .equals(expected.sort_values(KEYS).reset_index(drop=True))
        ):
            raise ValueError("Motion predictions do not cover exactly the evaluation rows.")
    return {"name": name, "values": values, "fits": fits, "artifacts": artifacts}


def evaluate(results: list[dict[str, Any]], plan: dict[str, Any], run: Run) -> dict[str, Any]:
    pooled = {
        a: pd.concat([r["values"][a] for r in results], ignore_index=True)
        for a in results[0]["values"]
    }
    folds = []
    for row in results:
        folds.append(
            {"fold": row["name"], "scores": {a: error_metrics(v) for a, v in row["values"].items()}}
        )
    comparisons = []
    for group in GROUPS:
        for arm, reference, label in (
            (group, "control", "addition"),
            ("motion_state", "without_" + group, "ablation"),
        ):
            effects = [
                f["scores"][arm]["coordinate_rmse_yards"]
                - f["scores"][reference]["coordinate_rmse_yards"]
                for f in folds
            ]
            comparisons.append(
                {
                    "group": group,
                    "comparison": label,
                    "fold_effects": effects,
                    **comparison(pooled[arm], pooled[reference], 10),
                }
            )
    promising = []
    for group in GROUPS:
        rows = [r for r in comparisons if r["group"] == group]
        if all(r["rmse_difference"] < 0 and max(r["fold_effects"]) < 0 for r in rows):
            promising.append(group)
    fits = [f for r in results for f in r["fits"]]
    if len(fits) != 30 or {f["parameter_count"] for f in fits} != {31842}:
        raise ValueError("Matched-capacity motion experiment is incomplete.")
    return {
        "status": "completed",
        "signature": plan["signature"],
        "run_id": run.run_id,
        "folds": folds,
        "pooled": {a: error_metrics(v) for a, v in pooled.items()},
        "comparisons": comparisons,
        "promising_groups": promising,
        "complete_motion_vs_control": comparison(pooled["motion_state"], pooled["control"]),
        "secondary_blend_vs_previous_blend": comparison(
            pooled["motion_state_equal_blend"], pooled["temporal_equal_blend"]
        ),
        "secondary_blend_vs_tree": comparison(pooled["motion_state_equal_blend"], pooled["tree"]),
        "fits": len(fits),
        "reused_reference_fits": 6,
        "epochs_per_fit": SETTINGS["epochs"],
        "parameter_count": 31842,
        "training_seconds": sum(f["training_seconds"] for f in fits),
        "rows": len(pooled["control"]),
        "games": int(pooled["control"].game_id.nunique()),
        "learning_curves": [
            {"fold": f["fold"], "arm": f["arm"], "curve": f["curve"]} for f in fits
        ],
        "artifacts": {p: h for r in results for p, h in r["artifacts"].items()},
        "sources": plan["sources"],
        "parent_signature": plan["parent_signature"],
        "feature_completion_gate": "open",
        "limitations": [
            "Adaptive follow-up on reused folds; one seed; conditional residual features.",
            "Twelve-epoch frozen probes; no final model or Kaggle evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import pytest

        raise SystemExit(
            pytest.main(
                [
                    str(ROOT / "tests/test_domain_probe.py"),
                    str(ROOT / "tests/test_motion_research.py"),
                    "-q",
                ]
            )
        )
    destination = ROOT / "artifacts/motion_research"
    destination.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(destination / "pipeline.lock"), timeout=1),
        Run(ROOT, "motion-research") as run,
    ):
        plan = specification()
        signature = fingerprint(plan)
        path = destination / "plan.json"
        if path.exists() and json.loads(path.read_text())["signature"] != signature:
            raise ValueError("Motion protocol changed; preserve existing experiments.")
        plan = {"signature": signature, **plan}
        atomic_json(path, plan)
        splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
        results = []
        for fold in plan["folds"]:
            validate_fold(fold, splits)
            results.append(run_fold(fold, destination / fold["name"], signature, run))
        report = evaluate(results, plan, run)
        report["artifacts"][str(path.relative_to(ROOT))] = sha256(path)
        atomic_json(destination / "report.json", report)
        atomic_json(ROOT / "docs/results/motion_research.json", report)
        run.event(
            "motion_study_evaluated",
            fits=report["fits"],
            promising_groups=report["promising_groups"],
            **report["secondary_blend_vs_previous_blend"],
        )


if __name__ == "__main__":
    main()
