"""Checkpoint final preprocessing on the frozen refit partition, without holdout scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.feature_candidates import HISTORY_NAMES
from nfl_trajectory.feature_experiment import load_week
from nfl_trajectory.feature_research import fit_baseline, fold_history, verify_inputs
from nfl_trajectory.final_protocol import digest, load_protocol
from nfl_trajectory.motion import ENTITY
from nfl_trajectory.representation_features import fit_routes, route_inputs
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage


def fit_route_encoder(caches: list[Path], training_games: list[int]) -> dict[str, Any]:
    """Give each scored player/play one vote, with deterministic ordering across caches."""
    entities, values = [], []
    for cache in caches:
        bank, _, _ = load_week(cache)
        selected = bank.state.game_id.isin(training_games).to_numpy()
        if selected.any():
            entities.append(bank.state.loc[selected, ENTITY])
            values.append(route_inputs(bank)[selected])
    frame = pd.concat(entities, ignore_index=True)
    if frame.duplicated(ENTITY).any() or set(frame.game_id) != set(training_games):
        raise ValueError("Route fitting requires every authorized entity exactly once.")
    order = frame.sort_values(ENTITY).index.to_numpy()
    fitted = fit_routes(np.concatenate(values)[order])
    fitted["training_games"] = training_games
    return fitted


def validate_preprocessing(
    folder: Path, training_games: list[int], rows: int, trajectories: int
) -> dict[str, Any]:
    """Reject a research-era fit or an incomplete final training partition."""
    models = {
        name: json.loads((folder / (name + ".json")).read_text())
        for name in ("baseline", "history_model", "routes")
    }
    if any(model.get("training_games") != training_games for model in models.values()):
        raise ValueError("Final preprocessing does not match the frozen training games.")
    baseline, history, routes = (models[name] for name in ("baseline", "history_model", "routes"))
    if baseline["global"]["coordinate_count"] != 2 * rows:
        raise ValueError("The final physical baseline did not fit every forecast coordinate.")
    if routes["training_entities"] != trajectories:
        raise ValueError("The final route encoder did not fit every training trajectory.")
    table = pd.read_csv(folder / "history.csv")
    if (
        len(table) != trajectories
        or table.duplicated(ENTITY).any()
        or set(table.game_id) != set(training_games)
        or not np.isfinite(table[HISTORY_NAMES].to_numpy(float)).all()
        or history["smoothing"] != 20.0
        or sum(row[0] for row in history["tables"]["role"].values()) != trajectories
        or sum(row[0] for row in history["tables"]["player"].values()) != trajectories
    ):
        raise ValueError("The final historical encodings have incomplete or invalid coverage.")
    earliest = table.game_id // 100 == min(training_games) // 100
    # The first date has no prior outcomes; counts and means must remain at their cold prior.
    history_values = table.loc[earliest, HISTORY_NAMES].to_numpy(float)
    expected = np.zeros_like(history_values)
    expected[:, [1, 6]] = 1.0
    if not np.array_equal(history_values, expected):
        raise ValueError("Final history leaked outcomes into the first training date.")
    return {
        "baseline_coordinate_count": baseline["global"]["coordinate_count"],
        "history_entities": len(table),
        "history_players": len(history["tables"]["player"]),
        "history_roles": len(history["tables"]["role"]),
        "route_training_entities": routes["training_entities"],
        "route_components": len(routes["pc_scale"]),
        "route_prototypes": len(routes["centers"]),
    }


def preprocess(root: Path, run: Run) -> dict[str, Any]:
    """Reuse each verified completed component independently after interruption."""
    folder = root / "artifacts/final/preprocessing"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "artifacts/final/pipeline.lock"), timeout=1):
        if (root / "artifacts/final/model_seal.json").exists():
            raise ValueError(
                "Final model is sealed; preprocessing cannot be refitted after sealing."
            )
        protocol = load_protocol(root)
        games = protocol["contract"]["training_games"]
        if set(games) & set(protocol["contract"]["holdout_games"]):
            raise ValueError("Final preprocessing cannot include holdout games.")
        caches, _ = verify_inputs(root)
        provenance = {
            "protocol_source": protocol["source_signature"],
            "protocol_sha256": sha256(root / "artifacts/final/protocol.json"),
            "implementation_sha256": sha256(Path(__file__)),
            "script_sha256": sha256(root / "scripts/refit_final.py"),
            "training_games": games,
        }
        source = digest(provenance)

        def baseline_action() -> None:
            atomic_json(folder / "baseline.json", fit_baseline(root, caches, set(games)))

        def history_action() -> None:
            table, fitted = fold_history(root, caches, set(games), set())
            atomic_bytes(folder / "history.csv", table.to_csv(index=False).encode())
            atomic_json(folder / "history_model.json", fitted)

        actions = [
            ("baseline", ["baseline.json"], baseline_action),
            ("history", ["history.csv", "history_model.json"], history_action),
            (
                "routes",
                ["routes.json"],
                lambda: atomic_json(folder / "routes.json", fit_route_encoder(caches, games)),
            ),
        ]
        for name, outputs, action in actions:
            stage(root, "final-" + name, source, [folder / n for n in outputs], action, run)
        measurements = validate_preprocessing(
            folder,
            games,
            protocol["training_inventory"]["training_rows"],
            protocol["training_inventory"]["training_trajectories"],
        )
        # Recheck the bound source and data after computation, before publishing the receipt.
        if load_protocol(root) != protocol:
            raise ValueError("The final protocol changed while preprocessing was running.")
        report = {
            "status": "passed",
            "stage": "final_preprocessing",
            "source_signature": source,
            "provenance": provenance,
            "training_games": len(games),
            "training_rows": protocol["training_inventory"]["training_rows"],
            **measurements,
            "outputs": {
                name: sha256(folder / name)
                for name in ("baseline.json", "history.csv", "history_model.json", "routes.json")
            },
            "final_model": False,
            "holdout_evaluation": "not_run",
            "remaining": "Fit both tree profiles, seal predictions, and evaluate the holdout.",
        }
        atomic_json(folder / "summary.json", report)
        run.event("final_preprocessing_completed", source_signature=source, **measurements)
        return report
