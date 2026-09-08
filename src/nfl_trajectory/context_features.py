"""Observed metadata, robust path distributions and matched-player histories.

These candidates use only organizer input fields. The nearest opponent and
receiver are fixed using the final observed state; all relationship histories
then use exact matching observed frame IDs. No future positions enter this bank.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_trajectory.features import REQUIRED, _history
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.runtime import atomic_bytes

POSITIONS = (
    "QB",
    "RB",
    "FB",
    "WR",
    "TE",
    "T",
    "G",
    "C",
    "DE",
    "DT",
    "NT",
    "OLB",
    "ILB",
    "MLB",
    "LB",
    "CB",
    "FS",
    "SS",
    "S",
    "Unknown",
)
ROBUST_CHANNELS = (
    "vx",
    "vy",
    "speed",
    "acceleration",
    "closing_speed",
    "lateral_speed",
    "radial_acceleration",
    "lateral_acceleration",
    "turn_rate",
    "jerk",
    "ball_distance",
    "ball_ux",
)
ROBUST_WINDOWS = (3, 8, 20, 40)
ANCHORS = ("opponent1", "opponent2", "receiver")
RELATIVE_CHANNELS = ("dx", "dy", "distance", "dvx", "dvy", "closing")


def context_catalog() -> pd.DataFrame:
    records: list[tuple[str, str]] = []
    metadata = [
        "height_inches",
        "weight_pounds",
        "age_years",
        "height_present",
        "weight_present",
        "age_present",
        "weight_per_height_squared",
    ]
    metadata += [f"position__{position}" for position in POSITIONS]
    records.extend((f"metadata__{name}", "metadata") for name in metadata)
    records.extend(
        (f"robust{w:02d}__{channel}__{stat}", "robust_history")
        for w in ROBUST_WINDOWS
        for channel in ROBUST_CHANNELS
        for stat in ("median", "iqr", "tail_span")
    )
    records.extend(
        (f"matched__{anchor}__w{w:02d}__{channel}__{stat}", "matched_history")
        for anchor in ANCHORS
        for w in (3, 8, 20)
        for channel in RELATIVE_CHANNELS
        for stat in ("mean", "std", "change")
    )
    records.extend(
        (f"matched__{anchor}__w{w:02d}__coverage", "matched_history")
        for anchor in ANCHORS
        for w in (3, 8, 20)
    )
    records.extend(
        (f"peer__{channel}__{kind}", "peer_context")
        for channel in ("speed", "ball_distance", "x", "y")
        for kind in ("side_percentile", "side_difference", "play_percentile")
    )
    records.extend(
        (f"reach__{name}", "reachability")
        for name in (
            "required_vx",
            "required_vy",
            "required_ax",
            "required_ay",
            "required_speed",
            "speed_margin",
            "heading_cross",
            "reported_vx_gap",
            "reported_vy_gap",
        )
    )
    records.extend((f"reach__within_{speed}yps", "reachability") for speed in (3, 6, 9, 12))
    base = list(records)
    records.extend(
        (f"{gate}__{name}", family)
        for gate in ("time", "fraction", "fraction_squared")
        for name, family in base
    )
    result = pd.DataFrame(records, columns=["feature", "family"])
    result["availability"] = "pre-throw input plus supplied horizon; exact observed frame matches"
    result["rationale"] = result.family.map(
        {
            "metadata": "Position, body size and game-date age describe movement capability.",
            "robust_history": "Quantiles describe observed path distributions.",
            "matched_history": "Relative motion distinguishes closing coverage from proximity.",
            "peer_context": "Within-play ranks describe spacing without a learned population.",
            "reachability": "Required arrival motion measures the gap from current motion.",
        }
    )
    return result


@dataclass
class ContextBank:
    state: pd.DataFrame
    values: np.ndarray

    def matrix(self, targets: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        require_keys(targets)
        names = context_catalog().feature.tolist()
        chosen = names if columns is None else columns
        if len(chosen) != len(set(chosen)) or not set(chosen).issubset(names):
            raise ValueError("Unknown or repeated context features.")
        if len(targets) * len(chosen) * 4 > 256 * 1024**2:
            raise MemoryError("Context expansion requires bounded batches.")
        aligned = targets[ENTITY].merge(
            self.state.assign(_row=np.arange(len(self.state))),
            on=ENTITY,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if aligned._row.isna().any():
            raise ValueError("Missing observed context entity.")
        rows = aligned._row.to_numpy(int)
        frame = targets.frame_id.to_numpy(float)
        horizon = aligned.num_frames_output.to_numpy(float)
        if (frame > horizon).any():
            raise ValueError("Context forecast exceeds the supplied horizon.")
        width = self.values.shape[1]
        if width * 4 != len(names):
            raise ValueError("Context feature schema changed; rebuild its cache.")
        lookup = {name: i for i, name in enumerate(names)}
        output = np.empty((len(targets), len(chosen)), dtype=np.float32)
        for j, name in enumerate(chosen):
            gate, index = divmod(lookup[name], width)
            factor = (1.0, frame / 10, frame / horizon, (frame / horizon) ** 2)[gate]
            output[:, j] = self.values[rows, index] * factor
        if not np.isfinite(output).all():
            raise ValueError("Context features must be finite.")
        return output


def _matched_histories(
    h: pd.DataFrame, terminal: pd.DataFrame, wanted: pd.DataFrame
) -> pd.DataFrame:
    links = []
    wanted_keys = set(map(tuple, wanted[ENTITY].to_numpy()))
    for _, play in terminal.groupby(KEYS[:2], sort=False):
        for row in play.itertuples(index=False):
            if (row.game_id, row.play_id, row.nfl_id) not in wanted_keys:
                continue
            opposite = play[play.player_side.ne(row.player_side)].copy()
            opposite["distance"] = np.hypot(opposite.x - row.x, opposite.y - row.y)
            opposite = opposite.sort_values(["distance", "x", "y", "vx", "vy", "nfl_id"])
            receiver = play[play.player_role.eq("Targeted Receiver")].sort_values("nfl_id")
            peers = [
                opposite.iloc[0] if len(opposite) else None,
                opposite.iloc[1] if len(opposite) > 1 else None,
                receiver.iloc[0] if len(receiver) else None,
            ]
            for anchor, peer in zip(ANCHORS, peers, strict=True):
                if peer is not None:
                    links.append(
                        {
                            "game_id": row.game_id,
                            "play_id": row.play_id,
                            "nfl_id": row.nfl_id,
                            "peer_id": int(peer.nfl_id),
                            "anchor": anchor,
                        }
                    )
    result = wanted[ENTITY].set_index(ENTITY)
    if not links:
        return result
    left = h[ENTITY + ["frame_id", "x", "y", "vx", "vy"]].merge(
        pd.DataFrame(links), on=ENTITY, validate="many_to_many"
    )
    peer = h[ENTITY + ["frame_id", "x", "y", "vx", "vy"]].rename(
        columns={"nfl_id": "peer_id", **{c: "peer_" + c for c in ("x", "y", "vx", "vy")}}
    )
    joined = left.merge(
        peer, on=["game_id", "play_id", "peer_id", "frame_id"], how="inner", validate="many_to_one"
    )
    joined = joined.merge(
        terminal[ENTITY + ["frame_id"]].rename(columns={"frame_id": "last_frame"}),
        on=ENTITY,
        validate="many_to_one",
    )
    joined["dx"], joined["dy"] = joined.peer_x - joined.x, joined.peer_y - joined.y
    joined["dvx"], joined["dvy"] = joined.peer_vx - joined.vx, joined.peer_vy - joined.vy
    joined["distance"] = np.hypot(joined.dx, joined.dy)
    joined["closing"] = -(joined.dx * joined.dvx + joined.dy * joined.dvy) / np.maximum(
        joined.distance, 1e-6
    )
    columns = {}
    for anchor in ANCHORS:
        for w in (3, 8, 20):
            selected = joined[
                joined.anchor.eq(anchor) & (joined.frame_id > joined.last_frame - w)
            ].sort_values(KEYS)
            groups = selected.groupby(ENTITY)[list(RELATIVE_CHANNELS)]
            aggregations = {
                "mean": groups.mean(),
                "std": groups.std(ddof=0),
                "change": groups.last() - groups.first(),
            }
            for stat, values in aggregations.items():
                for channel in RELATIVE_CHANNELS:
                    columns[f"matched__{anchor}__w{w:02d}__{channel}__{stat}"] = values[channel]
            columns[f"matched__{anchor}__w{w:02d}__coverage"] = selected.groupby(ENTITY).size() / w
    return result.join(pd.DataFrame(columns))


def build_context_features(inputs: pd.DataFrame, entities: pd.DataFrame) -> ContextBank:
    # Whitelist all model inputs. player_name and incidental outcome columns are ignored.
    metadata = [
        c
        for c in ("player_height", "player_weight", "player_birth_date", "player_position")
        if c in inputs
    ]
    allowed = (
        REQUIRED
        + [c for c in ("s", "a", "dir", "o", "absolute_yardline_number") if c in inputs]
        + metadata
    )
    raw = inputs[allowed].copy()
    if metadata and (raw.groupby(ENTITY)[metadata].nunique(dropna=False) > 1).any().any():
        raise ValueError("Player metadata changes within an observed play.")
    h = _history(raw)
    terminal = h.groupby(ENTITY, sort=False).tail(1)
    wanted = entities[ENTITY].drop_duplicates()
    last = wanted.merge(terminal, on=ENTITY, how="left", validate="one_to_one")
    if last.x.isna().any():
        raise ValueError("Context entities are missing from observed input.")
    info = raw.groupby(ENTITY, sort=False).tail(1)[ENTITY + metadata]
    last = last.merge(info, on=ENTITY, validate="one_to_one")
    indexed = last.set_index(ENTITY)
    values = pd.DataFrame(index=indexed.index)
    height = indexed.get("player_height", pd.Series(index=indexed.index, dtype="str")).astype(
        "string"
    )
    parsed = height.str.extract(r"^(\d+)-(\d+)$")
    if (height.notna() & parsed[0].isna()).any():
        raise ValueError("Supplied player height must use feet-inches format.")
    inches = pd.to_numeric(parsed[0], errors="coerce") * 12 + pd.to_numeric(
        parsed[1], errors="coerce"
    )
    weight = pd.to_numeric(
        indexed.get("player_weight", pd.Series(index=indexed.index, dtype=float)), errors="raise"
    )
    born = pd.to_datetime(
        indexed.get("player_birth_date", pd.Series(index=indexed.index, dtype="str")),
        errors="raise",
    )
    game_date = pd.Series(
        pd.to_datetime(
            indexed.index.get_level_values("game_id").astype(str).str[:8], format="%Y%m%d"
        ),
        index=indexed.index,
    )
    age = (game_date - born).dt.days / 365.2425
    for name, series in (("height", inches), ("weight", weight), ("age", age)):
        unit = {"height": "inches", "weight": "pounds", "age": "years"}[name]
        if np.isinf(series.to_numpy(float, na_value=np.nan)).any() or (series.dropna() <= 0).any():
            raise ValueError("Observed player dimensions and age must be positive when supplied.")
        values[f"metadata__{name}_{unit}"] = series.fillna(0).astype(float)
        values[f"metadata__{name}_present"] = series.notna().astype(float)
    values["metadata__weight_per_height_squared"] = weight.div(inches**2).fillna(0)
    position = indexed.get("player_position", pd.Series("Unknown", index=indexed.index)).fillna(
        "Unknown"
    )
    position = position.where(position.isin(POSITIONS), "Unknown")
    for name in POSITIONS:
        values[f"metadata__position__{name}"] = position.eq(name).astype(float)
    history = h.merge(wanted, on=ENTITY, validate="many_to_one")
    history["last_frame"] = history.groupby(ENTITY).frame_id.transform("max")
    robust = {}
    for w in ROBUST_WINDOWS:
        selected = history[history.frame_id > history.last_frame - w]
        quantiles = selected.groupby(ENTITY)[list(ROBUST_CHANNELS)].quantile(
            [0.1, 0.25, 0.5, 0.75, 0.9]
        )
        for channel in ROBUST_CHANNELS:
            for stat, series in (
                ("median", quantiles[channel].xs(0.5, level=-1)),
                (
                    "iqr",
                    quantiles[channel].xs(0.75, level=-1) - quantiles[channel].xs(0.25, level=-1),
                ),
                (
                    "tail_span",
                    quantiles[channel].xs(0.9, level=-1) - quantiles[channel].xs(0.1, level=-1),
                ),
            ):
                robust[f"robust{w:02d}__{channel}__{stat}"] = series
    values = values.join(pd.DataFrame(robust)).join(_matched_histories(h, terminal, wanted))
    peers = terminal.set_index(ENTITY).copy()
    for channel in ("speed", "ball_distance", "x", "y"):
        side = peers.groupby(["game_id", "play_id", "player_side"])[channel]
        for kind, series in (
            ("side_percentile", side.rank(pct=True)),
            ("side_difference", peers[channel] - side.transform("mean")),
            ("play_percentile", peers.groupby(["game_id", "play_id"])[channel].rank(pct=True)),
        ):
            values[f"peer__{channel}__{kind}"] = series.reindex(values.index)
    horizon = indexed.num_frames_output / 10
    reach = {
        "required_vx": indexed.ball_dx / horizon,
        "required_vy": indexed.ball_dy / horizon,
        "required_ax": 2 * (indexed.ball_dx - indexed.vx * horizon) / horizon**2,
        "required_ay": 2 * (indexed.ball_dy - indexed.vy * horizon) / horizon**2,
        "required_speed": indexed.ball_distance / horizon,
        "speed_margin": indexed.speed - indexed.ball_distance / horizon,
        "heading_cross": indexed.dir_sin * indexed.o_cos - indexed.dir_cos * indexed.o_sin,
        "reported_vx_gap": indexed.s * indexed.dir_sin - indexed.vx,
        "reported_vy_gap": indexed.s * indexed.dir_cos - indexed.vy,
    }
    reach.update(
        {
            f"within_{s}yps": (indexed.ball_distance <= s * horizon).astype(float)
            for s in (3, 6, 9, 12)
        }
    )
    values = values.join(pd.DataFrame({f"reach__{k}": v for k, v in reach.items()}))
    names = context_catalog().feature.tolist()
    static_names = names[: len(names) // 4]
    array = values.reindex(columns=static_names).fillna(0).to_numpy(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite observed context features.")
    return ContextBank(last[ENTITY + ["num_frames_output"]], array)


def save_context(path: Path, bank: ContextBank) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        keys=bank.state[ENTITY].to_numpy(np.int64),
        horizon=bank.state.num_frames_output.to_numpy(np.int64),
        values=bank.values,
        names=context_catalog().feature.to_numpy(str),
    )
    atomic_bytes(path, buffer.getvalue())


def load_context(path: Path) -> ContextBank:
    with np.load(path, allow_pickle=False) as data:
        if data["names"].tolist() != context_catalog().feature.tolist():
            raise ValueError("Context cache has a different feature schema.")
        state = pd.DataFrame(data["keys"], columns=ENTITY)
        state["num_frames_output"] = data["horizon"]
        return ContextBank(state, data["values"])
