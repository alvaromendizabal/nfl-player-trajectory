"""Per-file downloads and memory-bounded audits of the official competition files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from nfl_trajectory.motion import KEYS, require_keys, split_games
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, fingerprint, stage

COMPETITION = "nfl-big-data-bowl-2026-prediction"


def safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("Unsafe competition file path.")
    return Path(*path.parts)


def unpack_download(folder: Path, name: str, destination: Path) -> None:
    """Accept exactly the requested file, including Kaggle's single-file ZIP response."""
    relative = safe_relative(name)
    files = [p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".kaggle-partial")]
    if len(files) != 1:
        raise ValueError("Expected exactly one response file from Kaggle.")
    source = files[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) != 1:
                raise ValueError("Single-file response contained an unexpected archive layout.")
            member = members[0]
            member_path = safe_relative(member.filename)
            if member_path not in (relative, Path(relative.name)):
                raise ValueError("Archive member did not match the requested file.")
            if member.file_size > 2 * 1024**3:
                raise ValueError("Unexpectedly large single-file archive.")
            fd, temporary = tempfile.mkstemp(dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as target, archive.open(member) as content:
                    shutil.copyfileobj(content, target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
    else:
        if source.name != relative.name:
            raise ValueError("Downloaded filename does not match request.")
        os.replace(source, destination)


def kaggle_api() -> Any:
    # Lazy import: importing kaggle can authenticate. Offline tests never import it.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download(root: Path, run: Run, api: Any = None) -> None:
    api = kaggle_api() if api is None else api
    files: list[Any] = []
    token = None
    seen = set()
    while True:
        response = api.competition_list_files(COMPETITION, page_token=token, page_size=100)
        files.extend(response.files or [])
        token = response.next_page_token
        if not token:
            break
        if token in seen:
            raise ValueError("Kaggle returned a repeated pagination token.")
        seen.add(token)
    if not files:
        raise ValueError("Kaggle returned no files; confirm account access and accepted rules.")
    names = [item.name for item in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate names in competition file inventory.")
    inventory = []
    for index, item in enumerate(files, 1):
        name = item.name
        target = root / "data" / "raw" / safe_relative(name)
        metadata = {
            "name": name,
            "size": getattr(item, "total_bytes", None),
            "creation_date": str(getattr(item, "creation_date", "")),
        }
        signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        stage_name = "download-" + hashlib.sha256(name.encode()).hexdigest()[:20]

        def action(
            name: str = name,
            target: Path = target,
            signature: str = signature,
            expected_size: int | None = metadata["size"],
        ) -> None:
            temporary_root = root / ".downloads"
            temporary_root.mkdir(parents=True, exist_ok=True)
            folder = temporary_root / signature
            folder.mkdir(exist_ok=True)
            # The pinned official client validates remote identity before byte-range resume.
            # Retain its partial file and marker across failed invocations.
            api.competition_download_file(
                COMPETITION, name, path=str(folder), force=False, quiet=True
            )
            unpack_download(folder, name, target)
            size = expected_size
            if size and target.stat().st_size != size:
                target.unlink()
                raise ValueError("Downloaded file size differs from the competition inventory.")
            shutil.rmtree(folder)

        stage(root, stage_name, signature, [target], action, run)
        inventory.append(metadata)
        run.event("download_progress", completed_files=index, total_files=len(files), file=name)
    atomic_json(root / "artifacts" / "data_inventory.json", inventory)


def audit(root: Path, run: Run) -> None:
    """Read one weekly pair at a time; completed pair checks survive interruption."""
    raw = root / "data" / "raw"
    inputs = sorted((raw / "train").glob("input_*.csv"))
    outputs = sorted((raw / "train").glob("output_*.csv"))
    if not inputs or {p.name.replace("input_", "") for p in inputs} != {
        p.name.replace("output_", "") for p in outputs
    }:
        raise ValueError("Download complete input/output training pairs before auditing.")
    summaries = []
    for i, input_path in enumerate(inputs, 1):
        output_path = input_path.with_name(input_path.name.replace("input_", "output_"))
        summary_path = root / "artifacts" / "audit" / f"{input_path.stem}.json"

        def action(
            input_path: Path = input_path,
            output_path: Path = output_path,
            summary_path: Path = summary_path,
        ) -> None:
            x = pd.read_csv(input_path)
            y = pd.read_csv(output_path)
            require_keys(x)
            require_keys(y)
            import numpy as np

            for frame in [x, y]:
                if not np.isfinite(frame[["x", "y"]].to_numpy(dtype=float)).all():
                    raise ValueError("Nonfinite tracking coordinates.")
            required = {"player_to_predict", "num_frames_output", "ball_land_x", "ball_land_y"}
            if not required.issubset(x.columns):
                raise ValueError("Input file does not match the published 2026 prediction schema.")
            flags = x["player_to_predict"].astype(str).str.lower()
            if not flags.isin(["true", "false", "1", "0"]).all():
                raise ValueError("player_to_predict contains invalid values.")
            scored = x.loc[flags.isin(["true", "1"])].sort_values(KEYS)
            last = scored.groupby(KEYS[:3], sort=False).tail(1)
            actual = y.groupby(KEYS[:3])["frame_id"].agg(["min", "max", "count"])
            expected = last.set_index(KEYS[:3])["num_frames_output"]
            joined = actual.join(expected, how="outer")
            if (
                joined.isna().any().any()
                or not (
                    (joined["min"] == 1)
                    & (joined["max"] == joined["num_frames_output"])
                    & (joined["count"] == joined["num_frames_output"])
                ).all()
            ):
                raise ValueError("Scored players or forecast horizons disagree with output rows.")
            atomic_json(
                summary_path,
                {
                    "file": input_path.name,
                    "input_rows": len(x),
                    "output_rows": len(y),
                    "games": sorted(int(v) for v in x.game_id.unique()),
                    "plays": len(x[["game_id", "play_id"]].drop_duplicates()),
                    "scored_trajectories": len(actual),
                    "status": "passed",
                },
            )

        stage(
            root,
            f"audit-{input_path.stem}",
            fingerprint(root, [input_path, output_path], {}),
            [summary_path],
            action,
            run,
        )
        summaries.append(json.loads(summary_path.read_text()))
        run.event("audit_progress", completed_pairs=i, total_pairs=len(inputs))
    all_games = [game for summary in summaries for game in summary["games"]]
    if len(all_games) != len(set(all_games)):
        raise ValueError("A game appears in multiple weekly files; investigate before splitting.")
    splits = split_games(pd.DataFrame({"game_id": all_games}))
    atomic_bytes(root / "artifacts" / "game_splits.csv", splits.to_csv(index=False).encode())
    atomic_json(
        root / "artifacts" / "audit_summary.json",
        {
            "competition": COMPETITION,
            "pairs": summaries,
            "input_rows": sum(x["input_rows"] for x in summaries),
            "output_rows": sum(x["output_rows"] for x in summaries),
            "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().items()},
            "status": "passed",
        },
    )
