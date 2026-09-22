"""Strict historical checkpoint provenance; never rewrites saved metadata.

The caller must verify the complete checkpoint SHA-256 before deserialization.
A shared evaluation cache is not the checkpoint's historical training cache.
"""
from collections.abc import Mapping

TRAINING = "winner_repro_training.py"


def verify_lineage(checkpoint: Mapping, record: Mapping) -> dict:
    if checkpoint.get("epoch_completed") != record["best_epoch"]:
        raise ValueError("CHECKPOINT_FAILURE: selected epoch mismatch")
    if checkpoint.get("cache_sha256") != record["checkpoint_cache_sha256"]:
        raise ValueError("CHECKPOINT_FAILURE: historical training-cache mismatch")
    actual = checkpoint.get("source_hashes")
    if not isinstance(actual, Mapping):
        raise ValueError("CHECKPOINT_FAILURE: missing source lineage")
    for name, expected in record["source_hashes"].items():
        if name == TRAINING:
            nested = actual.get(name)
            continuation = checkpoint.get("continuation_source_sha256")
            if nested is None and continuation is None:
                raise ValueError("CHECKPOINT_FAILURE: missing trainer lineage")
            if any(value != expected for value in (nested, continuation) if value is not None):
                raise ValueError("CHECKPOINT_FAILURE: conflicting or changed trainer lineage")
        elif actual.get(name) != expected:
            raise ValueError("CHECKPOINT_FAILURE: frozen source lineage mismatch " + name)
    return {
        "status": "PASS",
        "epoch": record["best_epoch"],
        "checkpoint_cache_sha256": record["checkpoint_cache_sha256"],
        "trainer_field": "continuation_source_sha256" if checkpoint.get("continuation_source_sha256") else "source_hashes",
        "checkpoint_bytes_must_be_pinned_by_caller": True,
    }
