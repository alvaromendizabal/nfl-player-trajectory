"""Published research must preserve squared-error aggregation and source lineage."""

import copy
import json
import math

import pytest

from nfl_trajectory.domain_evidence import load_domain_research, verify_pooled


def test_pooled_rmse_uses_rows_and_squared_error_not_mean_fold_rmse():
    counts = {"a": 1, "b": 4, "c": 16}
    report = {
        "folds": [
            {"fold": f, "scores": {"model": {"coordinate_rmse_yards": v}}}
            for f, v in zip(counts, (1, 2, 4), strict=True)
        ],
        "pooled": {"model": {"coordinate_rmse_yards": math.sqrt(273 / 21)}},
    }
    verify_pooled(report, counts)
    changed = copy.deepcopy(report)
    changed["pooled"]["model"]["coordinate_rmse_yards"] = 7 / 3
    with pytest.raises(ValueError, match="row-weighted"):
        verify_pooled(changed, counts)


def test_partial_or_changed_publication_is_rejected(tmp_path):
    assert load_domain_research(tmp_path) is None
    folder = tmp_path / "docs/results"
    folder.mkdir(parents=True)
    (folder / "domain_research.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        load_domain_research(tmp_path)
    (folder / "domain_manifest.json").write_text(
        json.dumps({"files": {"docs/results/domain_research.json": "stale"}})
    )
    with pytest.raises(ValueError, match="evidence changed"):
        load_domain_research(tmp_path)
