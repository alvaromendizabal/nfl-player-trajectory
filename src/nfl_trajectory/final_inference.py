"""Load verified final fits and predict from the refitted, frozen representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_contracts import TELEMETRY_COLUMNS
from nfl_trajectory.motion import KEYS, require_keys
from nfl_trajectory.research_inference import INFERENCE_MODULES
from nfl_trajectory.runtime import sha256
from nfl_trajectory.tree_inference import predict_tree, tree_correction

FINAL_MODULES = (*INFERENCE_MODULES, "final_inference")


def final_bundle(root: Path) -> dict[str, Any]:
    """Verify protocol, preprocessing, coordinate fits, conversion, and current source."""
    from nfl_trajectory.final_features import feature_groups, load_preprocessing
    from nfl_trajectory.final_fit import validate_profiles
    from nfl_trajectory.final_protocol import PROFILES, digest, load_protocol
    from nfl_trajectory.research_evidence import verified_checkpoint

    folder = root / "artifacts/final"
    protocol = load_protocol(root)
    preprocessing = load_preprocessing(root, protocol)
    plan = json.loads((folder / "fit_plan.json").read_text())
    report = json.loads((folder / "models/summary.json").read_text())
    provenance = plan["provenance"]
    source = digest(provenance)
    profiles = protocol["contract"]["availability_profile_features"]
    validate_profiles(profiles)
    if (
        plan["source_signature"] != source
        or plan["profiles"] != profiles
        or plan["settings"] != protocol["contract"]["estimator"]["settings"]
        or provenance["protocol_source"] != protocol["source_signature"]
        or provenance["protocol_sha256"] != sha256(folder / "protocol.json")
        or provenance["preprocessing_source"] != preprocessing["source_signature"]
        or provenance["preprocessing_sha256"] != sha256(folder / "preprocessing/summary.json")
        or provenance["environment"].get("scikit-learn") != "1.8.0"
        or report.get("status") != "passed"
        or report.get("final_fit") is not True
        or report.get("source_signature") != source
        or report.get("provenance") != provenance
        or report.get("training_games") != len(protocol["contract"]["training_games"])
    ):
        raise ValueError("Final fit, frozen protocol, and preprocessing provenance disagree.")
    expected_sources = {
        "src/nfl_trajectory/final_features.py",
        "src/nfl_trajectory/final_fit.py",
        "scripts/fit_final.py",
        "scripts/fit_final.py.lock",
        "scripts/prepare_tree.py",
    }
    if set(provenance["implementation"]) != expected_sources:
        raise ValueError("The final fit implementation manifest is incomplete.")
    for name, expected in provenance["implementation"].items():
        if sha256(root / name) != expected:
            raise ValueError("Final fitting implementation changed: " + name)
    trees, summaries = {}, []
    for profile in PROFILES:
        directory = folder / "models" / profile
        summary = json.loads((directory / "summary.json").read_text())
        axes = {axis + ".pkl": sha256(directory / (axis + ".pkl")) for axis in ("x", "y")}
        export = digest({"fit_source": source, "axes": axes, "features": profiles[profile]})
        for axis, coordinate in enumerate(("x", "y")):
            verified_checkpoint(
                root, f"final-fit-{profile}-{axis}", source, directory / (coordinate + ".pkl")
            )
        for name in ("tree.json", "training_predictions.npz", "summary.json"):
            verified_checkpoint(root, "final-export-" + profile, export, directory / name)
        tree = json.loads((directory / "tree.json").read_text())
        if (
            summary.get("status") != "passed"
            or summary.get("profile") != profile
            or summary.get("source_signature") != source
            or summary.get("export_signature") != export
            or summary.get("axis_sha256") != axes
            or summary.get("schema_sha256") != digest(profiles[profile])
            or summary.get("fitted_features") != len(profiles[profile])
            or summary.get("active_features") != len(tree["features"])
            or summary.get("training_rows") != protocol["training_inventory"]["training_rows"]
            or summary.get("tree_sha256") != sha256(directory / "tree.json")
            or summary.get("predictions_sha256") != sha256(directory / "training_predictions.npz")
            or not 0 <= summary["portable_max_absolute_difference"] <= 1e-12
            or tree["screened_feature_count"] != len(profiles[profile])
            or not set(tree["features"]).issubset(profiles[profile])
        ):
            raise ValueError("Final model conversion or schema validation failed: " + profile)
        if tree["features"]:
            feature_groups(tree["features"])
        tree_correction(np.zeros((1, len(tree["features"]))), tree)
        trees[profile] = tree
        summaries.append(summary)
    if report["profiles"] != summaries:
        raise ValueError("Final model report does not match its coordinate/export receipts.")
    return {
        "format": 1,
        "kind": "final_trajectory",
        "training_games": protocol["contract"]["training_games"],
        "baseline": json.loads((folder / "preprocessing/baseline.json").read_text()),
        "history": json.loads((folder / "preprocessing/history_model.json").read_text()),
        "routes": json.loads((folder / "preprocessing/routes.json").read_text()),
        "trees": trees,
        "source_signature": source,
        "protocol_source": protocol["source_signature"],
        "preprocessing_source": preprocessing["source_signature"],
        "fitted_features": {name: len(columns) for name, columns in profiles.items()},
        "inference_sources": {
            name: sha256(Path(__file__).with_name(name + ".py")) for name in FINAL_MODULES
        },
        "input_contract": "Any absent or nonfinite telemetry field uses the positional model.",
        "final_model": True,
    }


def final_variant(observed: pd.DataFrame) -> str:
    """Missing metadata has no effect; partial telemetry requests use one fitted fallback."""
    if not set(TELEMETRY_COLUMNS).issubset(observed.columns):
        return "without_optional_inputs"
    if not np.isfinite(observed[list(TELEMETRY_COLUMNS)].to_numpy(dtype=float)).all():
        return "without_optional_inputs"
    return "without_metadata"


def predict_final(
    observed: pd.DataFrame,
    targets: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    cold_history: bool = False,
) -> pd.DataFrame:
    """Ignore target coordinates and require dates strictly after the final training cutoff."""
    require_keys(observed)
    require_keys(targets)
    if bundle.get("format") != 1 or bundle.get("kind") != "final_trajectory":
        raise ValueError("Use a verified final inference bundle.")
    cutoff = max(bundle["training_games"]) // 100
    if any(
        (frame.game_id.to_numpy(np.int64) // 100 <= cutoff).any() for frame in (observed, targets)
    ):
        raise ValueError("Final inference must strictly follow its frozen training dates.")
    variant = final_variant(observed)
    history = bundle["history"]
    if cold_history:
        history = {**history, "tables": {**history["tables"], "player": {}}}
    # Omit invalid telemetry completely so feature construction cannot propagate NaN/Inf.
    if variant == "without_optional_inputs":
        observed = observed.drop(columns=list(TELEMETRY_COLUMNS), errors="ignore")
    return predict_tree(
        observed,
        targets[KEYS],
        {**bundle, "history": history, "tree": bundle["trees"][variant]},
    )


def standalone_source(root: Path, bundle: dict[str, Any]) -> str:
    """Embed the exact tested numeric modules, with no project import at prediction time."""
    sources = {}
    for name in FINAL_MODULES:
        path = root / "src/nfl_trajectory" / (name + ".py")
        if sha256(path) != bundle["inference_sources"][name]:
            raise ValueError("Final standalone inference source changed: " + name)
        sources[name] = path.read_text().replace("nfl_trajectory.", "_nfl_final.")
    return (
        "import json, sys, types\n"
        "_package = types.ModuleType('_nfl_final')\n"
        "_package.__path__ = []\n"
        "sys.modules['_nfl_final'] = _package\n"
        f"_sources = {sources!r}\n"
        "for _name, _source in _sources.items():\n"
        "    _qualified = '_nfl_final.' + _name\n"
        "    _module = types.ModuleType(_qualified)\n"
        "    _module.__file__ = _name + '.py'\n"
        "    _module.__package__ = '_nfl_final'\n"
        "    sys.modules[_qualified] = _module\n"
        "    exec(compile(_source, _module.__file__, 'exec'), _module.__dict__)\n"
        "final_predict = sys.modules['_nfl_final.final_inference'].predict_final\n"
        f"FINAL_MODEL = json.loads({json.dumps(bundle, allow_nan=False)!r})\n"
    )
