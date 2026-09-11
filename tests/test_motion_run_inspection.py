"""Read-only evidence inspection tests; these never contact AWS."""

import io

import pytest

from scripts.inspect_motion_run import JOB, Reader, digest, pair_totals, safe_key


class FakeS3:
    def __init__(self, payload):
        self.payload = payload

    def get_object(self, **kwargs):
        assert kwargs["ExpectedBucketOwner"] == "560403859723"
        return {"Body": io.BytesIO(self.payload), "ContentLength": len(self.payload)}


@pytest.mark.parametrize("key", ["/bad", "other-project/x", "experiments/../x"])
def test_unrelated_object_keys_rejected(key):
    with pytest.raises(ValueError):
        safe_key(key)


def test_verified_read_and_corruption_rejection(tmp_path):
    payload = b'{"status":"complete"}'
    reader = Reader(FakeS3(payload), tmp_path)
    assert reader.get(f"cloud-runs/{JOB}/status.json", "status.json", 100, digest(payload)) == payload
    assert reader.receipts[0]["sha256"] == digest(payload)
    with pytest.raises(ValueError, match="SHA256"):
        reader.get(f"cloud-runs/{JOB}/status.json", "other.json", 100, "0" * 64)
    assert not (tmp_path / "other.json").exists()


def test_oversize_read_rejected(tmp_path):
    with pytest.raises(ValueError, match="size limit"):
        Reader(FakeS3(b"x" * 20), tmp_path).get(f"cloud-runs/{JOB}/status.json", "a", 10)


def csv_bytes(rows):
    return ("game_id,play_id,nfl_id,frame_id,dx,dy\n" + "\n".join(rows)).encode()


def test_official_coordinate_metric_not_euclidean_distance():
    a = csv_bytes(["1,2,3,1,3,4", "1,2,3,2,0,0"])
    b = csv_bytes(["1,2,3,2,0,0", "1,2,3,1,0,0"])
    result = pair_totals(a, b)
    assert result["control_rmse"] == 2.5
    assert result["velocity_rmse"] == 0
    assert result["relative_gain"] == 1
    assert (result["rows"], result["games"]) == (2, 1)


@pytest.mark.parametrize("bad", ["1,2,3,2,0,0", "1,2,3,1,nan,0"])
def test_mismatched_or_nonfinite_errors_rejected(bad):
    with pytest.raises(ValueError):
        pair_totals(csv_bytes(["1,2,3,1,0,1"]), csv_bytes([bad]))


def test_duplicate_rows_rejected():
    data = csv_bytes(["1,2,3,1,0,1", "1,2,3,1,0,1"])
    with pytest.raises(ValueError, match="Duplicate"):
        pair_totals(data, data)
