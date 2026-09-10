"""Run independent fixed folds concurrently, then verify the original study.

This is an orchestration wrapper, not a numerical variant. It uses the already
frozen study specification and resumes its existing checkpoints unchanged.
"""

from __future__ import annotations

import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import motion_supervision as study
from filelock import FileLock

from nfl_trajectory.runtime import Run, atomic_json, sha256


def fit_fold(fold: dict[str, Any], signature: str) -> str:
    name = fold["name"]
    folder = study.ROOT / "artifacts/motion_supervision" / name
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "worker.lock"), timeout=1), Run(study.ROOT, name + "-motion") as run:
        if study.fingerprint(study.specification()) != signature:
            raise ValueError("The frozen numerical study changed before worker execution.")
        samples_path = study.ROOT / f"artifacts/temporal/research/{name}/samples.pkl"
        samples = pickle.loads(samples_path.read_bytes())
        scales_path = folder / "scales.json"
        if scales_path.exists():
            fitted = json.loads(scales_path.read_text())
            if fitted["signature"] != signature:
                raise ValueError("The scale receipt has a different study signature.")
        else:
            fitted = {"signature": signature, **study.fit_scales(samples)}
            atomic_json(scales_path, fitted)
        for arm in study.ARMS:
            study.run_arm(name, arm, samples, fitted, folder / arm, signature, run)
    return name


def main() -> None:
    root = study.ROOT
    folder = root / "artifacts/motion_supervision"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(root, "motion-workers") as run:
        plan = study.specification()
        signature = study.fingerprint(plan)
        path = folder / "plan.json"
        if path.exists() and json.loads(path.read_text())["signature"] != signature:
            raise ValueError("The existing fixed study has a different signature.")
        atomic_json(path, {"signature": signature, **plan})
        atomic_json(
            folder / "orchestration.json",
            {
                "signature": signature,
                "run_id": run.run_id,
                "source": "scripts/parallel_motion_supervision.py",
                "source_sha256": sha256(Path(__file__)),
                "workers": 3,
                "threads_per_worker": study.SETTINGS["threads"],
                "numerical_settings_changed": False,
            },
        )
        with ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(fit_fold, f, signature) for f in plan["folds"]]
            for future in as_completed(futures):
                run.event("motion_worker_completed", fold=future.result())
    # Original runner independently checks and reuses every completed fit.
    study.main()


if __name__ == "__main__":
    main()
