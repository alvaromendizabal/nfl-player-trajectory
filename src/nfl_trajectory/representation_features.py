"""Train-only route representations and deterministic football interaction hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.features import PlayerFeatures
from nfl_trajectory.motion import ENTITY, require_keys

ROUTE_INPUTS = [
    f"lag{lag:02d}__{channel}"
    for channel in ("relative_x", "relative_y", "vx", "vy", "speed", "turn_rate", "present")
    for lag in range(20)
]
POOL_CHANNELS = (
    "dx",
    "dy",
    "dvx",
    "dvy",
    "closing_speed",
    "closest_distance",
    "ball_distance_advantage",
    "heading_alignment",
    "age_seconds",
    "relative_speed",
)
WIDTHS = (2, 5, 10, 20)
GATES = ("level", "time", "fraction", "fraction_squared")


def static_catalog() -> list[tuple[str, str]]:
    records = []
    for side in ("teammate", "opponent"):
        for width in WIDTHS:
            records += [
                (f"pool__{side}__{width}yd__{channel}__{stat}", "graph_pool")
                for channel in POOL_CHANNELS
                for stat in ("mean", "std")
            ]
            records += [
                (f"pool__{side}__{width}yd__{stat}", "graph_pool")
                for stat in ("weight_sum", "effective_neighbours")
            ]
    for anchor in ("ball", "receiver", "opponent1"):
        records += [
            (f"destination__{anchor}__{stat}", "role_destination")
            for stat in (
                "dx",
                "dy",
                "vx",
                "vy",
                "velocity_gap_x",
                "velocity_gap_y",
                "ax",
                "ay",
                "speed",
                "alignment",
            )
        ]
    records += [
        (f"route_pc{i:02d}__{kind}", "route_representation")
        for i in range(16)
        for kind in ("score", "square", "coverage", "receiver")
    ]
    records += [
        (f"route_cluster{i:02d}__{kind}", "route_representation")
        for i in range(8)
        for kind in ("distance", "affinity")
    ]
    return records


def representation_catalog() -> pd.DataFrame:
    records = static_catalog()
    catalog = pd.DataFrame(
        [(f"{gate}__{name}", family) for gate in GATES for name, family in records],
        columns=["feature", "family"],
    )
    catalog["rationale"] = catalog.family.map(
        {
            "graph_pool": "Distance-weighted neighbour moments summarize local traffic.",
            "role_destination": "Receiver-following destinations encode defensive coupling.",
            "route_representation": "Training-only PCA and prototypes compress observed routes.",
        }
    )
    catalog["availability"] = (
        "Observed pre-throw tracking and organizer-supplied landing point, role and horizon."
    )
    catalog["provenance"] = np.where(
        catalog.family.eq("route_representation"),
        "PCA and eight prototypes fit on unique training players/plays; no outcome labels.",
        "Deterministic transforms of verified observed neighbour and motion bank.",
    )
    return catalog


def route_inputs(bank: PlayerFeatures) -> np.ndarray:
    requests = bank.state[ENTITY].assign(frame_id=1)
    return bank.matrix(requests, ROUTE_INPUTS).astype(float)


def fit_routes(values: np.ndarray) -> dict[str, Any]:
    """Fit only caller-selected training entities; equal entity weight, no outcome inputs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ROUTE_INPUTS) or len(values) < 2:
        raise ValueError("Route fitting needs at least two training entities and the fixed schema.")
    if not np.isfinite(values).all():
        raise ValueError("Route training values must be finite.")
    mean, scale = values.mean(0), values.std(0)
    scale[scale < 1e-5] = 1.0
    z = np.clip((values - mean) / scale, -8, 8)
    eigenvalues, vectors = np.linalg.eigh(z.T @ z / len(z))
    order = np.argsort(eigenvalues)[::-1][:16]
    components = vectors[:, order]
    signs = np.sign(components[np.abs(components).argmax(0), np.arange(16)])
    components *= np.where(signs == 0, 1, signs)
    pc_scale = np.sqrt(np.maximum(eigenvalues[order], 1e-5))
    scores = np.clip((z @ components) / pc_scale, -8, 8)
    # Deterministic farthest-first initialization, then fixed-budget Lloyd iterations.
    centers = [scores[0].copy()]
    for _ in range(7):
        distance = np.min(np.sum((scores[:, None] - np.asarray(centers)) ** 2, axis=2), axis=1)
        centers.append(scores[int(distance.argmax())].copy())
    centers_array = np.asarray(centers)
    for _ in range(25):
        distance = np.sum((scores[:, None] - centers_array) ** 2, axis=2)
        assigned = distance.argmin(1)
        previous = centers_array.copy()
        for i in range(8):
            if np.any(assigned == i):
                centers_array[i] = scores[assigned == i].mean(0)
        if np.allclose(previous, centers_array, rtol=0, atol=1e-8):
            break
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "components": components.tolist(),
        "pc_scale": pc_scale.tolist(),
        "centers": centers_array.tolist(),
        "training_entities": len(values),
        "inputs": ROUTE_INPUTS,
        "fit_scope": "training observed histories only; no targets",
    }


def route_transform(values: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["inputs"] != ROUTE_INPUTS:
        raise ValueError("Route representation schema mismatch.")
    z = np.clip((values - np.asarray(fitted["mean"])) / np.asarray(fitted["scale"]), -8, 8)
    scores = np.clip((z @ np.asarray(fitted["components"])) / np.asarray(fitted["pc_scale"]), -8, 8)
    if not np.isfinite(scores).all():
        raise ValueError("Route representation must be finite.")
    return scores


@dataclass
class RepresentationBank:
    state: pd.DataFrame
    values: np.ndarray

    def matrix(self, targets: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        require_keys(targets)
        names = representation_catalog().feature.tolist()
        chosen = names if columns is None else columns
        if len(chosen) != len(set(chosen)) or not set(chosen).issubset(names):
            raise ValueError("Unknown or duplicate representation features.")
        if len(targets) * len(chosen) * 4 > 256 * 1024**2:
            raise MemoryError("Representation expansion requires bounded batches.")
        aligned = targets[ENTITY].merge(
            self.state[ENTITY + ["num_frames_output"]].assign(_row=np.arange(len(self.state))),
            on=ENTITY,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if aligned._row.isna().any():
            raise ValueError("Missing representation entity.")
        rows = aligned._row.to_numpy(int)
        frame, horizon = targets.frame_id.to_numpy(float), aligned.num_frames_output.to_numpy(float)
        if (frame > horizon).any():
            raise ValueError("Representation forecast exceeds supplied horizon.")
        factors = (np.ones(len(frame)), frame / 10, frame / horizon, (frame / horizon) ** 2)
        lookup = {name: i for i, name in enumerate(names)}
        width = len(static_catalog())
        result = np.empty((len(targets), len(chosen)), dtype=np.float32)
        for j, name in enumerate(chosen):
            gate, i = divmod(lookup[name], width)
            result[:, j] = self.values[rows, i] * factors[gate]
        if not np.isfinite(result).all():
            raise ValueError("Representation features must be finite.")
        return result


def build_representation(bank: PlayerFeatures, fitted: dict[str, Any]) -> RepresentationBank:
    requests = bank.state[ENTITY].assign(frame_id=1)
    columns = [
        "lag00__vx",
        "lag00__vy",
        "lag00__ball_dx",
        "lag00__ball_dy",
        "horizon_seconds",
        "role__Defensive Coverage",
        "role__Targeted Receiver",
        "receiver1__dx",
        "receiver1__dy",
        "receiver1__present",
    ]
    columns += [
        f"{side}{slot}__{channel}"
        for side in ("teammate", "opponent")
        for slot in range(1, 7)
        for channel in (*POOL_CHANNELS, "distance", "present")
    ]
    x = bank.matrix(requests, columns).astype(float)
    lookup = {name: x[:, i] for i, name in enumerate(columns)}
    results = []
    for side in ("teammate", "opponent"):
        distance = np.column_stack([lookup[f"{side}{i}__distance"] for i in range(1, 7)])
        present = np.column_stack([lookup[f"{side}{i}__present"] for i in range(1, 7)])
        for width in WIDTHS:
            weights = np.exp(-distance / width) * present
            total = weights.sum(1)
            normalized = np.divide(
                weights, total[:, None], out=np.zeros_like(weights), where=total[:, None] > 0
            )
            for channel in POOL_CHANNELS:
                values = np.column_stack([lookup[f"{side}{i}__{channel}"] for i in range(1, 7)])
                average = np.sum(values * normalized, axis=1)
                spread = np.sqrt(
                    np.maximum(np.sum((values - average[:, None]) ** 2 * normalized, axis=1), 0)
                )
                results += [average, spread]
            effective = np.divide(
                1.0, np.sum(normalized**2, axis=1), out=np.zeros(len(x)), where=total > 0
            )
            results += [total, effective]
    horizon = lookup["horizon_seconds"]
    vx, vy = lookup["lag00__vx"], lookup["lag00__vy"]
    ball_x, ball_y = lookup["lag00__ball_dx"], lookup["lag00__ball_dy"]
    for anchor in ("ball", "receiver", "opponent1"):
        dx, dy = ball_x.copy(), ball_y.copy()
        mask = lookup["role__Targeted Receiver"] + lookup["role__Defensive Coverage"]
        if anchor != "ball":
            prefix = "receiver1" if anchor == "receiver" else "opponent1"
            dx -= lookup[f"{prefix}__dx"]
            dy -= lookup[f"{prefix}__dy"]
            mask = lookup["role__Defensive Coverage"] * lookup[f"{prefix}__present"]
        ux, uy = dx / horizon, dy / horizon
        alignment = np.divide(
            vx * dx + vy * dy,
            np.hypot(vx, vy) * np.hypot(dx, dy),
            out=np.zeros(len(x)),
            where=np.hypot(vx, vy) * np.hypot(dx, dy) > 1e-8,
        )
        results += [
            value * mask
            for value in (
                dx,
                dy,
                ux,
                uy,
                ux - vx,
                uy - vy,
                2 * (ux - vx) / horizon,
                2 * (uy - vy) / horizon,
                np.hypot(ux, uy),
                alignment,
            )
        ]
    scores = route_transform(route_inputs(bank), fitted)
    for i in range(16):
        pc = scores[:, i]
        results += [
            pc,
            pc**2,
            pc * lookup["role__Defensive Coverage"],
            pc * lookup["role__Targeted Receiver"],
        ]
    distance = np.sqrt(np.sum((scores[:, None] - np.asarray(fitted["centers"])) ** 2, axis=2))
    for i in range(8):
        results += [distance[:, i], np.exp(-(distance[:, i] ** 2) / 16)]
    values = np.column_stack(results).astype(np.float32)
    if values.shape[1] != len(static_catalog()) or not np.isfinite(values).all():
        raise ValueError("Representation schema or finite-value contract failed.")
    return RepresentationBank(bank.state.copy(), values)
