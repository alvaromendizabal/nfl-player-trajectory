"""Archive safety, API pagination, download reuse, and weekly schema contracts."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nfl_trajectory.data import audit, download, safe_relative, unpack_download
from nfl_trajectory.runtime import Run


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../b", "a\\b", "C:/file", ""])
def test_unsafe_download_paths_fail(name: str) -> None:
    with pytest.raises(ValueError):
        safe_relative(name)


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "download"
    folder.mkdir()
    with zipfile.ZipFile(folder / "data.zip", "w") as archive:
        archive.writestr("../outside.csv", "bad")
    with pytest.raises(ValueError):
        unpack_download(folder, "train/input.csv", tmp_path / "output.csv")
    assert not (tmp_path / "outside.csv").exists()


def test_single_file_zip_extracts_atomically(tmp_path: Path) -> None:
    folder = tmp_path / "download"
    folder.mkdir()
    with zipfile.ZipFile(folder / "input.csv.zip", "w") as archive:
        archive.writestr("input.csv", "a,b\n1,2\n")
    destination = tmp_path / "data/input.csv"
    unpack_download(folder, "train/input.csv", destination)
    assert destination.read_text() == "a,b\n1,2\n"


class KaggleDouble:
    def __init__(self) -> None:
        self.downloads = 0

    def competition_list_files(self, competition, page_token, page_size):
        name = "test.csv" if page_token is None else "train/input.csv"
        return SimpleNamespace(
            files=[SimpleNamespace(name=name, total_bytes=8, creation_date="2026")],
            next_page_token="page2" if page_token is None else "",
        )

    def competition_download_file(self, competition, name, path, force, quiet):
        self.downloads += 1
        (Path(path) / Path(name).name).write_text("a,b\n1,2\n")


def test_paginated_download_reuses_verified_files(tmp_path: Path) -> None:
    api = KaggleDouble()
    with Run(tmp_path, "download") as run:
        download(tmp_path, run, api)
        download(tmp_path, run, api)
        assert api.downloads == 2
        (tmp_path / "data/raw/test.csv").write_text("corrupt")
        download(tmp_path, run, api)
        assert api.downloads == 3


def make_week(root: Path, corrupt: bool = False) -> None:
    folder = root / "data/raw/train"
    folder.mkdir(parents=True)
    inputs, outputs = [], []
    for day in pd.date_range("2023-09-01", periods=12):
        game = int(day.strftime("%Y%m%d") + "00")
        common = {"game_id": game, "play_id": 1, "nfl_id": 1, "frame_id": 1, "x": 30.0, "y": 20.0}
        inputs.append(
            {
                **common,
                "num_frames_output": 2 if corrupt else 1,
                "player_to_predict": True,
                "ball_land_x": 40.0,
                "ball_land_y": 20.0,
            }
        )
        outputs.append(common)
    pd.DataFrame(inputs).to_csv(folder / "input_2023_w01.csv", index=False)
    pd.DataFrame(outputs).to_csv(folder / "output_2023_w01.csv", index=False)


def test_weekly_audit_and_temporal_manifest(tmp_path: Path) -> None:
    make_week(tmp_path)
    with Run(tmp_path, "audit") as run:
        audit(tmp_path, run)
        audit(tmp_path, run)
    summary = json.loads((tmp_path / "artifacts/audit_summary.json").read_text())
    assert summary["output_rows"] == 12
    assert summary["status"] == "passed"
    splits = pd.read_csv(tmp_path / "artifacts/game_splits.csv")
    assert splits.game_id.nunique() == 12


def test_mismatched_horizon_is_rejected(tmp_path: Path) -> None:
    make_week(tmp_path, corrupt=True)
    with Run(tmp_path, "audit") as run, pytest.raises(ValueError):
        audit(tmp_path, run)
