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

