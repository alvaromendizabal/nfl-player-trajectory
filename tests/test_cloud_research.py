"""The cloud recovery path must keep holdout tracking and unsafe archives excluded."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def cloud_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cloud_research.py"
    spec = importlib.util.spec_from_file_location("cloud_research_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data/raw/train/input_2023_w01.csv", True),
        ("data/raw/train/input_2023_w15.csv", True),
        ("data/raw/train/input_2023_w16.csv", False),
        ("data/raw/train/output_2023_w01.csv", False),
        ("artifacts/features/weeks/input_2023_w15/features.npz", True),
        ("artifacts/features/weeks/input_2023_w18/features.npz", False),
        ("artifacts/kaggle/submission.ipynb", False),
        ("artifacts/benchmark/summary.json", True),
    ],
)
def test_cloud_input_scope(path, expected):
    assert cloud_module().selected_input(path) is expected


@pytest.mark.parametrize("name", ["/tmp/escape", "../escape", "artifacts/../../escape"])
def test_cloud_rejects_unsafe_snapshot_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        cloud_module().selected_input(name)


@pytest.mark.parametrize("symlink", [False, True])
def test_seed_extraction_rejects_escape_and_link_members(tmp_path, symlink):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("artifacts/link" if symlink else "../escape")
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/escape"
        output.addfile(info)
    with pytest.raises(ValueError, match="Archive"):
        cloud_module().unpack(archive, tmp_path / "restored")


def test_seed_restores_artifacts_without_overwriting_canonical_code(tmp_path):
    archive = tmp_path / "seed.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in ("artifacts/research/plan.json", "scripts/cloud_research.py"):
            info = tarfile.TarInfo(name)
            info.size = 2
            output.addfile(info, io.BytesIO(b"{}"))
    destination = tmp_path / "restored"
    cloud_module().unpack(archive, destination)
    assert (destination / "artifacts/research/plan.json").read_text() == "{}"
    assert not (destination / "scripts/cloud_research.py").exists()


@pytest.mark.parametrize(("container_limit", "expected"), [("max", 128), (str(64 * 1024**3), 64)])
def test_memory_budget_respects_container_limit(monkeypatch, tmp_path, container_limit, expected):
    module = cloud_module()
    monkeypatch.setattr(
        module.os, "sysconf", lambda name: 1 if name == "SC_PAGE_SIZE" else 128 * 1024**3
    )
    (tmp_path / "memory.max").write_text(container_limit)
    assert module.memory_budget_gib(tmp_path) == expected
