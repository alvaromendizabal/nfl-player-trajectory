"""Matched feature attribution using preserved Round 1 rows, controls and fits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import zipfile
from typing import Any

import numpy as np

from origin_features import ALL_NAMES, BASE_NAMES, fit_ridge, predict_ridge, rmse
from motion_features import FAMILIES, NAMES

BOOTSTRAPS = 10000
SEED = 20260911
COMPARISON_COUNT = 9  # six arrival tests plus three new motion tests, fixed in advance
ARRIVAL_GROUPS = {
    "velocity": np.array([i for i, n in enumerate(ALL_NAMES) if "__required_velocity_" in n], int),
    "acceleration": np.array([i for i, n in enumerate(ALL_NAMES) if "__required_acceleration_" in n], int),
    "receiver": np.array([i for i, n in enumerate(ALL_NAMES) if "__receiver_velocity_gap_" in n], int),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    _atomic(path, lambda f: f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    _atomic(path, lambda f: np.savez_compressed(f, **arrays))


def _atomic(path: Path, writer: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing symlink output")
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def seal_json(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"Sealed plan changed: {path.name}. Preserve earlier results; do not overwrite.")
    else:
        atomic_json(path, value)


def safe_file(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe artifact path")
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink artifact paths are not accepted")
    if not current.is_file():
        raise FileNotFoundError(current)
    return current


def read_npz(path: Path, fields: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 64 * 1024**2:
            raise ValueError("Array checkpoint exceeds 64 MiB decoded safety limit")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if fields is None else fields)}


def load_parent(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Read original-origin arrays; augmented keys are read ONLY to map saved indices.

    No augmented features/labels are loaded, transformed or fitted. Hashes verify
    old bytes. Old model forward predictions must reproduce before new fits.
    This function does not write to the parent directory or refit old models.
    """
    root = Path(root)
    manifest_path = safe_file(root, "research/dataset_manifest.json")
    if digest(manifest_path) != contract["parent_manifest_sha256"]:
        raise ValueError("Round 1 dataset manifest differs from the uploaded report")
    summary_path = safe_file(root, "research/screen_summary.json")
    if digest(summary_path) != contract["parent_summary_sha256"]:
        raise ValueError("Round 1 screen report changed; preserve it and review the new report")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_signature"] != contract["parent_source_signature"]:
        raise ValueError("Round 1 source lineage differs")
    paths = [r["path"] for r in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate parent artifact paths")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ("X", "y", "keys", "role")}
    globals_, cursor, verified, decoded = [], 0, 0, 0
    for entry in manifest["files"]:
        path = safe_file(root / "research", entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Parent feature checkpoint checksum failed")
        if int(entry["offset"]) == 0:
            z = read_npz(path)
            n = len(z["keys"])
            if z["X"].shape != (n, len(ALL_NAMES)) or z["y"].shape != (n, 2):
                raise ValueError("Parent feature/label shapes changed")
            if np.any(z["offset"] != 0) or np.any(z["prethrow"]):
                raise ValueError("Augmented rows reached original-origin dataset")
            if not np.isfinite(z["X"]).all() or not np.isfinite(z["y"]).all():
                raise ValueError("Nonfinite parent data")
            for key in arrays:
                arrays[key].append(z[key])
                decoded += z[key].nbytes
            if decoded > 128 * 1024**2:
                raise ValueError("Parent original-origin arrays exceed 128 MiB")
            globals_.append(np.arange(cursor, cursor + n, dtype=np.int64))
        else:
            # Saved fit indices refer to a global concatenation. Only key shape
            # is needed to traverse nonzero-offset entries, never their labels.
            n = len(read_npz(path, ("keys",))["keys"])
        cursor += n
        verified += 1
    if not arrays["X"]:
        raise ValueError("No original-origin parent rows")
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    data["global_indices"] = np.concatenate(globals_)
    keys = data["keys"]
    if keys.ndim != 2 or keys.shape[1] != 4 or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError("Parent join-key schema changed")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Duplicate original forecast keys")
    if not np.isin(data["role"], [0, 1, 2, 3]).all():
        raise ValueError("Parent role vocabulary changed")
    if (len(keys), len(np.unique(keys[:, 0])), len(np.unique(keys[:, :2], axis=0))) != (
        contract["original_rows"], contract["original_games"], contract["original_plays"]
    ):
        raise ValueError("Parent original-origin population differs from the report")
    protocol = json.loads(safe_file(root, "research/screen/protocol.json").read_text())
    expected_signature = hashlib.sha256((contract["parent_manifest_sha256"] + contract["parent_source_signature"] + "screen-v1").encode()).hexdigest()
    if protocol["signature"] != expected_signature or protocol["penalty"] != 0.01 or len(protocol["folds"]) != 3:
        raise ValueError("Parent fitting protocol changed")
    folds, predictions, all_eval = [], {}, []
    for fold in protocol["folds"]:
        fi = int(fold["fold"])
        train_games = np.array(fold["train_games"], dtype=np.int64)
        eval_games = np.array(fold["validation_games"], dtype=np.int64)
        if len(train_games) == 0 or len(eval_games) == 0 or train_games.max() // 100 >= eval_games.min() // 100:
            raise ValueError("Parent chronological split is invalid")
        vi = np.flatnonzero(np.isin(keys[:, 0], eval_games))
        ti = None
        for arm, width in (("control", len(BASE_NAMES)), ("arrival", len(ALL_NAMES))):
            name = f"research/screen/fold_{fi}_{arm}"
            receipt = json.loads(safe_file(root, name + ".json").read_text())
            path = safe_file(root, name + ".npz")
            if receipt["signature"] != expected_signature or digest(path) != receipt["sha256"]:
                raise ValueError("Parent model/prediction artifact differs from its receipt")
            saved = read_npz(path)
            if not np.array_equal(saved["evaluation_keys"], keys[vi]) or not np.array_equal(saved["truth"], data["y"][vi]):
                raise ValueError("Parent evaluation keys or labels are not aligned")
            index = np.searchsorted(data["global_indices"], saved["train_indices"])
            if np.any(index >= len(keys)) or not np.array_equal(data["global_indices"][index], saved["train_indices"]):
                raise ValueError("A saved control uses augmented or unknown training rows")
            if not np.isin(keys[index, 0], train_games).all() or len(np.unique(index)) != len(index):
                raise ValueError("Saved training rows violate the declared split")
            if ti is not None and not np.array_equal(ti, index):
                raise ValueError("Parent arms did not train on identical rows")
            ti = index
            model = {k.removeprefix("model_"): v for k, v in saved.items() if k.startswith("model_")}
            replay = predict_ridge(model, data["X"][vi, :width])
            if not np.array_equal(replay, saved["pred"]):
                raise ValueError("Parent numerical replay changed; do NOT refit to hide the mismatch")
            if abs(rmse(data["y"][vi], replay) - receipt["rmse"]) > 1e-10:
                raise ValueError("Parent metric receipt mismatch")
            predictions[(fi, arm)] = replay
        folds.append({"fold": fi, "train": ti, "eval": vi,
                      "train_games": train_games, "validation_games": eval_games})
        all_eval.extend(vi.tolist())
    if len(all_eval) != len(set(all_eval)):
        raise ValueError("Parent evaluation folds overlap")
    summary = json.loads(summary_path.read_text())
    if len(all_eval) != summary["evaluation_rows"]:
        raise ValueError("Parent evaluation row population changed")
    for arm in ("control", "arrival"):
        pred = np.concatenate([predictions[(f["fold"], arm)] for f in folds])
        if abs(rmse(data["y"][all_eval], pred) - summary["pooled_rmse"][arm]) > 1e-10:
            raise ValueError("Parent pooled score cannot be reproduced")
    return {"data": data, "folds": folds, "predictions": predictions,
            "verified_checkpoints": verified, "parent_summary": summary,
            "parent_models_replayed": 6, "parent_models_refitted": 0}


def attribution_arms() -> dict[str, dict[str, Any]]:
    base = np.arange(len(BASE_NAMES))
    all_ = np.arange(len(ALL_NAMES))
    result = {}
    for group, columns in ARRIVAL_GROUPS.items():
        result[f"only_{group}"] = {"columns": np.r_[base, columns], "baseline": "control",
                                   "direction": "addition", "family": group}
        result[f"without_{group}"] = {"columns": np.setdiff1d(all_, columns), "baseline": "arrival",
                                      "direction": "removal", "family": group}
    return result


def motion_arms() -> dict[str, dict[str, Any]]:
    return {f"arrival_{family}": {"columns": np.r_[np.arange(len(ALL_NAMES)), len(ALL_NAMES) + cols],
                                 "baseline": "arrival", "direction": "addition", "family": family}
            for family, cols in FAMILIES.items()}


def paired_interval(y: np.ndarray, a: np.ndarray, b: np.ndarray, games: np.ndarray) -> dict[str, Any]:
    unique, inverse = np.unique(games, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two evaluation games required")
    n = np.bincount(inverse)
    sa = np.bincount(inverse, weights=((y - a) ** 2).sum(1))
    sb = np.bincount(inverse, weights=((y - b) ** 2).sum(1))
    rng = np.random.default_rng(SEED + 2)
    indices = rng.integers(0, len(unique), size=(BOOTSTRAPS, len(unique)))
    delta = np.sqrt(sb[indices].sum(1) / (2 * n[indices].sum(1))) - np.sqrt(sa[indices].sum(1) / (2 * n[indices].sum(1)))
    lo, hi = np.quantile(delta, [0.025, 0.975])
    adjusted = 0.05 / (2 * COMPARISON_COUNT)
    alo, ahi = np.quantile(delta, [adjusted, 1 - adjusted])
    return {"delta_rmse": rmse(y, b) - rmse(y, a), "ci95_low": float(lo), "ci95_high": float(hi),
            "simultaneous_low": float(alo), "simultaneous_high": float(ahi),
            "resamples": BOOTSTRAPS, "planned_comparisons": COMPARISON_COUNT,
            "simultaneous_method": "Bonferroni-adjusted percentile paired-game bootstrap; exploratory, reused folds"}


def candidate_fit(path: Path, x: np.ndarray, y: np.ndarray, keys: np.ndarray,
                  train: np.ndarray, valid: np.ndarray, columns: np.ndarray,
                  spec: dict[str, Any], feature_names: list[str]) -> tuple[np.ndarray, dict[str, Any], bool]:
    """Atomic self-contained checkpoints, read-back forward replay, and safe reuse."""
    started = time.monotonic()
    existed = path.exists()
    receipt_path = path.with_suffix(".json")
    if existed:
        if receipt_path.exists() and digest(path) != json.loads(receipt_path.read_text())["sha256"]:
            raise ValueError("Candidate checkpoint changed after publication")
        saved = read_npz(path)
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata != spec:
            raise ValueError("Candidate checkpoint signature/protocol differs")
        if not np.array_equal(saved["evaluation_keys"], keys[valid]) or not np.array_equal(saved["train_indices"], train) or not np.array_equal(saved["truth"], y[valid]):
            raise ValueError("Candidate training/evaluation rows changed")
        model = {k.removeprefix("model_"): v for k, v in saved.items() if k.startswith("model_")}
        pred = saved["pred"]
    else:
        model = fit_ridge(x[np.ix_(train, columns)], y[train], penalty=0.01)
        pred = predict_ridge(model, x[np.ix_(valid, columns)])
        if pred.shape != (len(valid), 2) or not np.isfinite(pred).all():
            raise ValueError("Invalid candidate forecast; cannot filter rows")
        atomic_npz(path, **{f"model_{k}": v for k, v in model.items()}, pred=pred,
                   truth=y[valid], evaluation_keys=keys[valid], train_indices=train,
                   metadata=np.array(json.dumps(spec, sort_keys=True)))
    restored = read_npz(path)
    model = {k.removeprefix("model_"): v for k, v in restored.items() if k.startswith("model_")}
    replay = predict_ridge(model, x[np.ix_(valid, columns)])
    if not np.array_equal(replay, pred):
        raise ValueError("Candidate exact forward replay failed")
    receipt = {"sha256": digest(path), "rmse": rmse(y[valid], replay),
               "forward_replay_exact": True, "input_columns": len(columns),
               "retained_columns": int(model["keep"].sum()),
               "retained_feature_names": [feature_names[int(columns[i])] for i in np.flatnonzero(model["keep"])],
               "train_rows": len(train), "evaluation_rows": len(valid),
               "elapsed_seconds": round(time.monotonic() - started, 4)}
    if not receipt_path.exists():
        # Also recovers the narrow interruption window after the atomic NPZ save.
        atomic_json(receipt_path, receipt)
    return replay, receipt, existed


def screen(parent: dict[str, Any], out: Path, phase: str, signature: str,
           extra: np.ndarray | None = None, *, replay_only: bool = False,
           event: Any = lambda *a, **k: None) -> dict[str, Any]:
    data, folds = parent["data"], parent["folds"]
    arms = attribution_arms() if phase == "attribution" else motion_arms()
    if phase not in ("attribution", "motion"):
        raise ValueError("Unknown screen phase")
    if phase == "motion" and (extra is None or extra.shape != (len(data["X"]), len(NAMES))):
        raise ValueError("Aligned new feature matrix required")
    x = data["X"] if phase == "attribution" else np.column_stack([data["X"], extra])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite feature matrix")
    y, keys = data["y"].astype(float), data["keys"]
    names = list(ALL_NAMES) + ([] if extra is None else list(NAMES))
    output = out / phase
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"signature": signature, "phase": phase, "penalty": 0.01,
                "bootstrap_resamples": BOOTSTRAPS, "planned_comparisons": COMPARISON_COUNT,
                "arms": {a: {**s, "columns": s["columns"].tolist()} for a, s in arms.items()},
                "folds": [{"fold": f["fold"], "train_indices": f["train"].tolist(),
                           "eval_indices": f["eval"].tolist()} for f in folds]}
    seal_json(output / "protocol.json", protocol)
    metrics, slices, retained = [], [], []
    predictions = {a: [] for a in ["control", "arrival", *arms]}
    all_y, all_games = [], []
    new, reused = 0, 0
    for fold in folds:
        fi, ti, vi = fold["fold"], fold["train"], fold["eval"]
        pred = {arm: parent["predictions"][(fi, arm)] for arm in ("control", "arrival")}
        for arm, setup in arms.items():
            path = output / f"fold_{fi}_{arm}.npz"
            if replay_only and not path.exists():
                raise ValueError("Replay never fits a missing arm. Finish this phase before replay.")
            spec = {"signature": signature, "fold": fi, "arm": arm, "penalty": 0.01,
                    "columns": setup["columns"].tolist(), "phase": phase}
            p, receipt, reuse = candidate_fit(path, x, y, keys, ti, vi, setup["columns"], spec, names)
            pred[arm] = p
            new += int(not reuse)
            reused += int(reuse)
            retained.append({"fold": fi, "arm": arm, "input_columns": receipt["input_columns"],
                             "retained_columns": receipt["retained_columns"],
                             "retained_feature_names": receipt["retained_feature_names"]})
            event("reused_fit" if reuse else "fit_checkpoint", phase=phase, fold=fi, arm=arm,
                  rmse=receipt["rmse"], completed=new + reused, total=3 * len(arms))
        for arm, p in pred.items():
            predictions[arm].append(p)
            metrics.append({"fold": fi, "arm": arm, "rows": len(vi), "rmse": rmse(y[vi], p)})
            masks = {"first_second": keys[vi, 3] <= 10, "after_first_second": keys[vi, 3] > 10}
            masks.update({"role_" + str(i): data["role"][vi] == i for i in range(4)})
            for label, mask in masks.items():
                if mask.any():
                    slices.append({"fold": fi, "arm": arm, "slice": label, "rows": int(mask.sum()),
                                   "rmse": rmse(y[vi][mask], p[mask]),
                                   "sse": float(((y[vi][mask] - p[mask]) ** 2).sum())})
        all_y.append(y[vi]); all_games.append(keys[vi, 0])
        atomic_json(output / "progress.json", {"phase": phase, "completed_folds": fi,
                                               "new_fits": new, "reused_new_arm_fits": reused,
                                               "fold_metrics": metrics})
    yy, gg = np.concatenate(all_y), np.concatenate(all_games)
    pp = {a: np.concatenate(v) for a, v in predictions.items()}
    pooled = {a: rmse(yy, p) for a, p in pp.items()}
    decisions = []
    for arm, setup in arms.items():
        baseline = setup["baseline"]
        interval = paired_interval(yy, pp[baseline], pp[arm], gg)
        deltas = [next(m["rmse"] for m in metrics if m["fold"] == f["fold"] and m["arm"] == arm)
                  - next(m["rmse"] for m in metrics if m["fold"] == f["fold"] and m["arm"] == baseline) for f in folds]
        gain = 1 - pooled[arm] / pooled[baseline] if pooled[baseline] else 0.0
        if setup["direction"] == "addition":
            passing = gain >= 0.01 and interval["simultaneous_high"] < 0 and sum(d < 0 for d in deltas) >= 2
            status = "candidate_for_established_model_test" if passing else "not_earned_scaling_in_this_interface"
        else:
            passing = interval["simultaneous_low"] > 0 and sum(d > 0 for d in deltas) >= 2
            status = "conditional_removal_evidence" if passing else "conditional_necessity_not_established"
        decisions.append({"arm": arm, "baseline": baseline, "family": setup["family"],
                          "direction": setup["direction"], "relative_gain": gain,
                          "improving_folds": sum(d < 0 for d in deltas),
                          "worsening_folds": sum(d > 0 for d in deltas),
                          "gate_passed": bool(passing), "decision": status, **interval})
    result = {"status": phase + "_complete", "phase": phase, "signature": signature,
              "new_fits_this_invocation": new, "reused_new_arm_fits": reused,
              "new_arm_fit_count_total": 3 * len(arms), "parent_models_refitted": 0,
              "parent_models_replayed": 6, "pooled_rmse": pooled, "decisions": decisions,
              "fold_metrics": metrics, "slices": slices, "retention": retained,
              "evaluation_rows": len(yy), "evaluation_games": len(np.unique(gg)),
              "original_origin_only": True, "augmentation_rerun": False,
              "outer_validation_used": False, "kaggle_score": None,
              "scope": "Exploratory matched training-side feature study on REUSED Round 1 folds, not neural/Kaggle performance",
              "feature_research": "open", "all_forward_replays_exact": True}
    atomic_json(output / ("replay_summary.json" if replay_only else "summary.json"), result)
    return result
