"""Private-S3 scientific runner tests use an in-memory client; no AWS calls occur."""

import copy
import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

torch = pytest.importorskip("torch")

from nfl_trajectory.motion_supervision import MatchedState  # noqa: E402
from nfl_trajectory.supervision_evidence import save_generation  # noqa: E402
from scripts.run_motion_experiment import RemoteStore, write_errors  # noqa: E402


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        body = kwargs["Body"]
        if hasattr(body, "read"):
            body = body.read()
        self.objects[key] = bytes(body)
        return {"ETag": '"fake"'}

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[key])}


def store(fake):
    value = RemoteStore("bucket", "experiments/test", fake)
    value._fresh_client = lambda: fake
    return value


def checkpoint(tmp_path, signature="a" * 64):
    folder = tmp_path / "local"
    state = MatchedState(32, 2026, 0.0, 3.0)
    receipt = save_generation(folder, state.payload(), signature)
    return folder, receipt


def test_publish_and_independent_restore_exact_bytes(tmp_path):
    fake = FakeS3()
    remote = store(fake)
    folder, receipt = checkpoint(tmp_path)
    published = remote.publish("coordinate", folder, receipt)
    assert published["independent_readback_verified"]
    restored = remote.restore("coordinate", tmp_path / "readback", "a" * 64)
    assert restored is not None
    assert restored["independent_readback_verified"]
    assert restored["receipt"] == receipt
    assert restored["state"]["steps"] == 0
    assert torch.equal(restored["state"]["rng"], MatchedState(32, 2026, 0.0, 3.0).rng)


def test_corrupted_remote_blob_is_rejected(tmp_path):
    fake = FakeS3()
    remote = store(fake)
    folder, receipt = checkpoint(tmp_path)
    published = remote.publish("coordinate", folder, receipt)
    fake.objects[published["blob_key"]] = b"corrupt"
    with pytest.raises(ValueError, match="absent or corrupt"):
        remote.restore("coordinate", tmp_path / "readback", "a" * 64)


def test_newer_remote_pointer_cannot_be_replaced(tmp_path):
    fake = FakeS3()
    remote = store(fake)
    folder, receipt = checkpoint(tmp_path)
    published = remote.publish("coordinate", folder, receipt)
    pointer = json.loads(fake.objects[published["pointer_key"]])
    pointer["step"] = 99
    fake.objects[published["pointer_key"]] = (json.dumps(pointer) + "\n").encode()
    with pytest.raises(ValueError, match="newer remote"):
        remote.publish("coordinate", folder, receipt)


def test_validation_errors_are_content_addressed_and_read_back(tmp_path):
    fake = FakeS3()
    remote = store(fake)
    frame = pd.DataFrame(
        {
            "game_id": [1, 1],
            "play_id": [2, 2],
            "nfl_id": [3, 3],
            "frame_id": [1, 2],
            "dx": [0.1, 0.2],
            "dy": [0.3, 0.4],
        }
    )
    original = copy.deepcopy(frame)
    receipt = write_errors(remote, "coordinate", frame)
    assert receipt["readback_verified"]
    assert receipt["key"].startswith("experiments/test/errors/coordinate-")
    pd.testing.assert_frame_equal(frame, original)


def test_unsafe_prefix_and_arm_are_rejected(tmp_path):
    fake = FakeS3()
    with pytest.raises(ValueError):
        RemoteStore("bucket", "../bad", fake)
    remote = store(fake)
    folder, receipt = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        remote.publish("joint", folder, receipt)
