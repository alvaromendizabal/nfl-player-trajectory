"""A figure must not silently draw newer predictions beside an older verified score."""

import hashlib
import json

import pytest

from nfl_trajectory import research_visuals


@pytest.mark.parametrize("mismatch", ["model", "bundle"])
def test_stale_validation_cannot_label_an_example_as_current(monkeypatch, tmp_path, mismatch):
    bundle = {"selected_model": "current_tree"}
    monkeypatch.setattr(research_visuals, "research_bundle", lambda root: bundle)
    report = {
        "status": "passed",
        "selected_model": "old_linear" if mismatch == "model" else "current_tree",
        "bundle_sha256": "stale"
        if mismatch == "bundle"
        else hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
    }
    path = tmp_path / "artifacts/research/inference/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="raw validation of the current predictor"):
        research_visuals.load_development_example(tmp_path)
