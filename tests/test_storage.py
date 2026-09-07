"""Snapshots resume, restore verified bytes, and do not overwrite unrelated work."""

import io
from pathlib import Path

import pytest

from nfl_trajectory.runtime import Run
from nfl_trajectory.storage import backup, restore


class S3Double:
    def __init__(self) -> None:
        self.objects = {}
        self.uploads = 0
        self.corrupt_download = False

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def upload_file(self, path, bucket, key, ExtraArgs):
        self.uploads += 1
        self.objects[key] = Path(path).read_bytes()

    def put_object(self, Bucket, Key, Body, ServerSideEncryption):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(b"bad" if self.corrupt_download else self.objects[key])


def test_backup_reuse_and_restore(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "data/test.csv").write_text("a,b\n1,2\n")
    s3 = S3Double()
    with Run(source, "backup") as run:
        manifest = backup(source, "bucket", run, s3)
        backup(source, "bucket", run, s3)
        assert s3.uploads == 1
    target = tmp_path / "target"
    with Run(target, "restore") as run:
        restore(target, "bucket", manifest, run, s3)
        restore(target, "bucket", manifest, run, s3)
    assert (target / "data/test.csv").read_bytes() == (source / "data/test.csv").read_bytes()


def test_restore_corruption_never_commits_bad_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "data/test.csv").write_text("valid")
    s3 = S3Double()
    with Run(source, "backup") as run:
        manifest = backup(source, "bucket", run, s3)
    s3.corrupt_download = True
    target = tmp_path / "target"
    with Run(target, "restore") as run, pytest.raises(ValueError):
        restore(target, "bucket", manifest, run, s3)
    assert not (target / "data/test.csv").exists()


def test_restore_preserves_divergent_local_file(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    file = tmp_path / "data/test.csv"
    file.write_text("original")
    s3 = S3Double()
    with Run(tmp_path, "backup") as run:
        manifest = backup(tmp_path, "bucket", run, s3)
    file.write_text("new work")
    with Run(tmp_path, "restore") as run, pytest.raises(ValueError):
        restore(tmp_path, "bucket", manifest, run, s3)
    assert file.read_text() == "new work"
