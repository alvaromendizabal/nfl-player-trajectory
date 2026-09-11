"""Cloud motion profile restores only checksum-pinned safe inputs."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import scripts.cloud_motion_supervision as cloud


def make_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as handle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def test_selective_sample_restore_verifies_archive_and_sample(tmp_path, monkeypatch):
    archive = tmp_path / "inputs.tar.gz"
    payload = b"private-synthetic-sample"
    make_archive(archive, [(cloud.SAMPLE_PATH, payload), ("ignored.txt", b"not restored")])
    monkeypatch.setattr(cloud, "ARCHIVE_SHA256", cloud.digest(archive))
    monkeypatch.setattr(cloud, "SAMPLE_SHA256", hashlib.sha256(payload).hexdigest())
    destination = tmp_path / "project"
    cloud.extract_verified_sample(archive, destination)
    assert (destination / cloud.SAMPLE_PATH).read_bytes() == payload
    assert not (destination / "ignored.txt").exists()


def test_unsafe_member_stops_before_restoration(tmp_path, monkeypatch):
    archive = tmp_path / "inputs.tar.gz"
    make_archive(archive, [("../escape", b"bad"), (cloud.SAMPLE_PATH, b"sample")])
    monkeypatch.setattr(cloud, "ARCHIVE_SHA256", cloud.digest(archive))
    with pytest.raises(ValueError, match="Unsafe private archive"):
        cloud.extract_verified_sample(archive, tmp_path / "project")
    assert not (tmp_path / "escape").exists()


def test_archive_hash_mismatch_stops(tmp_path):
    archive = tmp_path / "inputs.tar.gz"
    make_archive(archive, [(cloud.SAMPLE_PATH, b"sample")])
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        cloud.extract_verified_sample(archive, tmp_path / "project")


def test_repository_requires_pinned_commit(tmp_path):
    with pytest.raises(ValueError, match="Pinned 40-character"):
        cloud.repository_archive("main", tmp_path / "source.tar.gz")
