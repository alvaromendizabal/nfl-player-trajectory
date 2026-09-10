"""Bounded, hash-bound single-fold probe of observed soft matchup features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import pickle
import platform
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage  # noqa: E402
from nfl_trajectory.soft_coverage import candidates  # noqa: E402

ARMS = ("control_motion", "static_affinity", "temporal_affinity")
L2 = 0.01


def save_arrays(path: Path, **arrays: Any) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def fit_ridge(x: np.ndarray, y: np.ndarray, seconds: np.ndarray) -> dict[str, np.ndarray]:
    """Constant/duplicate screening and all fitted statistics use training only."""
    mean, scale = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
    retained = scale > 1e-7
    seen: set[str] = set()
    for i in np.flatnonzero(retained):
        digest = hashlib.sha256(np.ascontiguousarray(x[:, i]).tobytes()).hexdigest()
        if digest in seen:
            retained[i] = False
        seen.add(digest)
    z = np.clip((x[:, retained] - mean[retained]) / scale[retained], -10, 10)
    design = np.column_stack([z, np.ones(len(z))]) * seconds[:, None]
    gram = design.T @ design / len(design) + L2 * np.eye(design.shape[1])
    coef = solve(gram, design.T @ y / len(design), assume_a="pos")
    return {"mean": mean, "scale": scale, "retained": retained, "coef": coef}


def predict(model: Any, x: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    kept = model["retained"]
    z = np.clip((x[:, kept] - model["mean"][kept]) / model["scale"][kept], -10, 10)
    return (z @ model["coef"][:-1] + model["coef"][-1]) * seconds[:, None]


def comparison(reference: np.ndarray, treatment: np.ndarray, games: np.ndarray) -> dict[str, Any]:
    rmse = float(np.sqrt(np.mean(treatment**2)))
    control = float(np.sqrt(np.mean(reference**2)))
    groups, inverse = np.unique(games, return_inverse=True)
    count = np.bincount(inverse) * 2
    a = np.bincount(inverse, weights=(reference**2).sum(1))
    b = np.bincount(inverse, weights=(treatment**2).sum(1))
    draws = np.random.default_rng(20260910).integers(0, len(groups), (5000, len(groups)))
    delta = np.sqrt(b[draws].sum(1) / count[draws].sum(1))
    delta -= np.sqrt(a[draws].sum(1) / count[draws].sum(1))
    interval = np.quantile(delta, [0.025, 0.975]).tolist()
    return {
        "coordinate_rmse_yards": rmse,
        "reference_rmse_yards": control,
        "relative_reduction": 1 - rmse / control,
        "paired_game_delta_interval": interval,
        "continue_gate": bool(rmse <= 0.995 * control and interval[1] < 0),
    }


def execute(source_commit: str, run: Run) -> None:
    folder = ROOT / "artifacts/soft_coverage/inner_1"
    parent = ROOT / "artifacts/domain_research/inner_1"
    samples_path = ROOT / "artifacts/temporal/research/inner_1/samples.pkl"
    paths = [
        samples_path,
        parent / "features.npz",
        parent / "schema.json",
        ROOT / "artifacts/temporal/research/plan.json",
        ROOT / "artifacts/game_splits.csv",
    ]
    sources = [
        "scripts/soft_coverage.py",
        "src/nfl_trajectory/soft_coverage.py",
        "src/nfl_trajectory/runtime.py",
        "docs/SOFT_COVERAGE_EXPERIMENT.md",
    ]
    spec = {
        "source_commit": source_commit,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "sources": {p: sha256(ROOT / p) for p in sources},
        "arms": ARMS,
        "l2": L2,
        "threads": 2,
        "max_seconds": 300,
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scipy", "filelock", "threadpoolctl")
        },
    }
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    plan = folder.parent / "plan.json"
    if plan.exists() and json.loads(plan.read_text())["signature"] != signature:
        raise ValueError("Existing probe has different source, inputs or environment.")
    if not plan.exists():
        atomic_json(plan, {**spec, "signature": signature})
    data = dict(np.load(parent / "features.npz"))
    schema = json.loads((parent / "schema.json").read_text())
    control = np.array([f in ("control", "motion_state") for f in schema["families"]])
    feature_path = folder / "features.npz"

    def prepare() -> None:
        # This is the user's hash-verified private pickle, not an arbitrary upload.
        samples = pickle.loads(samples_path.read_bytes())
        parts, keys, train_flags = [], [], []
        names: list[str] = []
        for split in ("train", "validation"):
            for sample in samples:
                if sample["split"] != split:
                    continue
                values, current, _ = candidates(sample)
                if names and names != current:
                    raise ValueError("Candidate schema changed between plays.")
                names = current
                parts.append(values)
                keys.append(sample["keys"])
                train_flags.append(np.full(len(values), split == "train"))
                if len(parts) % 250 == 0:
                    run.event("features_prepared", plays=len(parts), total_plays=len(samples))
        np.testing.assert_array_equal(np.concatenate(keys), data["keys"])
        np.testing.assert_array_equal(np.concatenate(train_flags), data["train"])
        save_arrays(feature_path, x=np.concatenate(parts), names=np.array(names))

    stage(ROOT, "soft-coverage-features", signature, [feature_path], prepare, run)
    extra = dict(np.load(feature_path))
    train = data["train"]
    train_games, eval_games = np.unique(data["keys"][train, 0]), np.unique(data["keys"][~train, 0])
    fold = json.loads(paths[3].read_text())["folds"][0]
    if set(train_games) & set(eval_games) or max(train_games // 100) >= min(eval_games // 100):
        raise ValueError("Chronological game separation failed.")
    if len(train_games) != 94 or len(eval_games) != 41 or (~train).sum() != 83938:
        raise ValueError("Unexpected predefined fold membership or row count.")
    if set(map(int, fold["training_games"])) != set(train_games):
        raise ValueError("Training games differ from original fold plan.")
    results: dict[str, np.ndarray] = {}
    counts: dict[str, Any] = {}
    for arm in ARMS:
        selected = np.array(
            [
                arm == "temporal_affinity"
                or (arm == "static_affinity" and n.startswith("soft_static"))
                for n in extra["names"]
            ]
        )
        x = np.column_stack([data["x"][:, control], extra["x"][:, selected]])
        model_path, error_path = folder / (arm + ".npz"), folder / (arm + "_errors.npz")

        def fit(x: np.ndarray = x, model_path: Path = model_path) -> None:
            fitted = fit_ridge(x[train], data["y"][train], data["time"][train])
            save_arrays(model_path, **fitted)

        stage(ROOT, "soft-coverage-fit-" + arm, signature, [model_path], fit, run)
        model = np.load(model_path)

        def evaluate(model: Any = model, x: np.ndarray = x, error_path: Path = error_path) -> None:
            error = predict(model, x[~train], data["time"][~train]) - data["y"][~train]
            save_arrays(error_path, errors=error, keys=data["keys"][~train])

        stage(ROOT, "soft-coverage-evaluate-" + arm, signature, [error_path], evaluate, run)
        results[arm] = np.load(error_path)["errors"]
        counts[arm] = {"candidates": x.shape[1], "retained": int(model["retained"].sum())}
        run.event("arm_complete", arm=arm, rmse=float(np.sqrt(np.mean(results[arm] ** 2))))
    report_path = ROOT / "docs/results/soft_coverage.json"

    def report() -> None:
        scores = {a: float(np.sqrt(np.mean(e**2))) for a, e in results.items()}
        primary = comparison(results[ARMS[0]], results[ARMS[2]], data["keys"][~train, 0])
        artifacts = {str(p.relative_to(ROOT)): sha256(p) for p in folder.glob("*.npz")}
        atomic_json(
            report_path,
            {
                "status": "completed",
                "signature": signature,
                "source_commit": source_commit,
                "training_games": len(train_games),
                "evaluation_games": len(eval_games),
                "training_rows": int(train.sum()),
                "evaluation_rows": int((~train).sum()),
                "candidate_counts": counts,
                "scores": scores,
                "primary": primary,
                "temporal_vs_static": comparison(
                    results[ARMS[1]], results[ARMS[2]], data["keys"][~train, 0]
                ),
                "decision": "continue_to_folds" if primary["continue_gate"] else "stop_this_probe",
                "fit_count": 3,
                "elapsed_seconds": round(__import__("time").monotonic() - run.started, 3),
                "artifacts": artifacts,
                "kaggle_private_rmse_unchanged": 0.7009,
                "limitations": [
                    "one reused chronological fold",
                    "linear correction interface",
                    "in-sample parent training residuals",
                    "no neural feature exhaustion",
                ],
            },
        )

    stage(ROOT, "soft-coverage-report", signature, [report_path], report, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Bounded probe exceeded 300 seconds; completed stages remain resumable.")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    with FileLock(str(ROOT / ".soft-coverage.lock"), timeout=1), threadpool_limits(limits=2):
        with Run(ROOT, "soft-coverage") as run:
            execute(args.source_commit, run)


if __name__ == "__main__":
    main()
