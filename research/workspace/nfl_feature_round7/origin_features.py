"""Observed-origin feature laboratory. No cloud calls and no learned inputs.

Targets and feature construction have separate interfaces. All distances are in
canonical yards; the exact coordinate RMSE is computed over both coordinates.
This is a fixed-ridge diagnostic, not the existing production/neural model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

WIDTH = 160.0 / 3.0
ROLES = ("Targeted Receiver", "Defensive Coverage", "Passer", "Other Route Runner")
KEYS = ["game_id", "play_id", "nfl_id", "frame_id"]
REQUIRED = set(KEYS + ["x", "y", "play_direction", "player_role", "player_side",
                      "player_to_predict", "num_frames_output", "ball_land_x", "ball_land_y"])
# Terminal state is not the only control information: it already includes recent
# motion and synchronized receiver/passer histories. No ID or fitted prior enters X.
STATE_NAMES = ["x_center", "y_center", "vx", "vy", "ball_dx", "ball_dy", "age",
               "horizon", "observed_fraction", "vx_observed", "orientation_x",
               "orientation_y", "orientation_observed", "receiver_dx", "receiver_dy",
               "receiver_vx", "receiver_vy", "receiver_joint", "passer_dx", "passer_dy",
               "passer_joint", "is_offense"]
for window in (5, 10, 20):
    STATE_NAMES += [f"w{window}_{k}" for k in
                   ("displacement_x", "displacement_y", "mean_vx", "mean_vy",
                    "velocity_change_x", "velocity_change_y", "path_length",
                    "straightness", "joint_receiver_dx_change", "joint_receiver_dy_change",
                    "observed_fraction", "joint_receiver_fraction")]
STATE_NAMES += [f"role_{r}" for r in ROLES]
# Time multipliers describe displacement response to state; no general polynomial bank.
BASE_NAMES = ["intercept", "elapsed", "elapsed_squared", "remaining", "phase", "origin_offset"]
BASE_NAMES += [f"response_time__{k}" for k in STATE_NAMES]
BASE_NAMES += ["ball_remaining_x", "ball_remaining_y", "receiver_projected_x", "receiver_projected_y"]
ARRIVAL_BASE = ("required_velocity_x", "required_velocity_y", "required_acceleration_x",
                "required_acceleration_y", "receiver_velocity_gap_x", "receiver_velocity_gap_y")
ARRIVAL_NAMES = [f"{role}__{k}" for role in ROLES for k in ARRIVAL_BASE]
ALL_NAMES = BASE_NAMES + ARRIVAL_NAMES


class IneligibleOrigin(ValueError):
    """A shifted view cannot preserve the original scored-player population."""


def flags(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.lower()
    if not s.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid player_to_predict values")
    return s.isin(["true", "1"]).to_numpy()


def validate_raw(raw: pd.DataFrame) -> None:
    if not REQUIRED.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED - set(raw.columns))}")
    if raw.empty or raw[KEYS[:2]].drop_duplicates().shape[0] != 1:
        raise ValueError("Exactly one nonempty observed play is required")
    if raw.duplicated(KEYS).any():
        raise ValueError("Duplicate observed frame keys")
    if not np.isfinite(raw[["x", "y", "ball_land_x", "ball_land_y"]].to_numpy(float)).all():
        raise ValueError("Nonfinite observed/landing coordinates")
    if not raw.play_direction.isin(["left", "right"]).all() or raw.play_direction.nunique() != 1:
        raise ValueError("Inconsistent play direction")
    if not raw.player_role.isin(ROLES).all():
        raise ValueError("Unknown role: do not silently merge distinct roles")
    if not raw.player_side.isin(["Offense", "Defense"]).all():
        raise ValueError("Unknown side")
    for col in ["frame_id", "num_frames_output"]:
        v = raw[col].to_numpy(float)
        if not np.isfinite(v).all() or (v != np.floor(v)).any() or (v < 1).any():
            raise ValueError(f"Invalid {col}")
    flags(raw.player_to_predict)
    for col in ["player_role", "player_side", "player_to_predict", "num_frames_output",
                "ball_land_x", "ball_land_y"]:
        if (raw.groupby("nfl_id")[col].nunique(dropna=False) != 1).any():
            raise ValueError(f"Within-player task metadata changes: {col}")


def canonical_xy(xy: np.ndarray, left: bool) -> np.ndarray:
    v = np.asarray(xy, dtype=float)
    return np.array([120.0, WIDTH]) - v if left else v.copy()


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    y, p = np.asarray(truth, float), np.asarray(pred, float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or len(y) == 0:
        raise ValueError("Matching nonempty [rows, 2] arrays required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite metric input")
    return float(np.sqrt(np.square(y - p).sum() / (2 * len(y))))


def velocity(g: pd.DataFrame, left: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reported vectors first, adjacent observed finite differences as fallback.

    No bridge across a missing frame; no future interpolation/backward fill.
    Fallback validity is retained, rather than calling a fabricated zero measured.
    """
    xy = canonical_xy(g[["x", "y"]].to_numpy(float), left)
    f = g.frame_id.to_numpy(int)
    out = np.zeros_like(xy)
    valid = np.zeros(len(g), bool)
    adjacent = np.diff(f) == 1
    out[1:][adjacent] = np.diff(xy, axis=0)[adjacent] * 10
    valid[1:] = adjacent
    if {"s", "dir"}.issubset(g.columns):
        s, d = g.s.to_numpy(float), g.dir.to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(d) & (s >= 0)
        a = np.deg2rad(d[ok])
        out[ok] = np.column_stack([np.sin(a), np.cos(a)]) * s[ok, None] * (-1 if left else 1)
        valid[ok] = True
    return out, valid


@dataclass(frozen=True)
class ObservedView:
    origin: int
    offset: int
    cutoff: int
    left: bool
    states: dict[int, np.ndarray]
    anchors: dict[int, np.ndarray]
    velocities: dict[int, np.ndarray]
    ages: dict[int, float]
    horizons: dict[int, float]
    roles: dict[int, int]
    receiver_relative: dict[int, np.ndarray]
    receiver_velocity: dict[int, np.ndarray]
    receiver_valid: dict[int, bool]
    balls: dict[int, np.ndarray]
    scored: tuple[int, ...]


def build_view(observed: pd.DataFrame, *, origin: int, offset: int = 0) -> ObservedView:
    """Build only from the already-truncated observed prefix and known task fields.

    Input must contain NO rows after origin. Offset is a known training-task
    parameter. num_frames_output here is the original organizer horizon.
    Neither future player coordinates nor target residuals are arguments.
    """
    validate_raw(observed)
    if not isinstance(offset, int) or offset < 0 or offset > 20:
        raise ValueError("Offset must be an integer in 0..20")
    if observed.frame_id.max() > origin:
        raise ValueError("Future observed rows reached the feature interface")
    left = observed.play_direction.iloc[0] == "left"
    cutoff = origin + offset
    data: dict[int, dict[str, Any]] = {}
    for ident, all_g in observed.groupby("nfl_id", sort=True):
        all_g = all_g.sort_values("frame_id")
        g = all_g.loc[all_g.frame_id > origin - 20].copy()
        if g.empty:
            continue
        # Compute derivatives on the full prefix; retaining one prior frame is legal.
        full_v, full_ok = velocity(all_g, left)
        keep = all_g.frame_id.to_numpy(int) > origin - 20
        v, vok = full_v[keep], full_ok[keep]
        row = g.iloc[-1]
        data[int(ident)] = {"g": g, "v": v, "vok": vok,
                           "xy": canonical_xy(g[["x", "y"]].to_numpy(float), left), "row": row}
    if not data:
        raise IneligibleOrigin("No recent observed player")
    role_by_frame: dict[str, pd.DataFrame] = {}
    for role in ("Targeted Receiver", "Passer"):
        candidates = []
        for ident, d in data.items():
            if d["row"].player_role == role:
                g = pd.DataFrame({"frame_id": d["g"].frame_id.to_numpy(int),
                                  "id": ident, "x": d["xy"][:, 0], "y": d["xy"][:, 1],
                                  "vx": d["v"][:, 0], "vy": d["v"][:, 1], "vok": d["vok"]})
                candidates.append(g)
        combined = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
        if not combined.empty:
            if combined.frame_id.duplicated().any():
                raise ValueError(f"Ambiguous simultaneous {role} anchor")
            combined = combined.set_index("frame_id")
        role_by_frame[role] = combined
    states, anchors, velocities, ages, horizons, roles, rec_rel, rec_vel, rec_ok, balls = ({ } for _ in range(10))
    scored = []
    for ident, d in data.items():
        g, xy, v, row = d["g"], d["xy"], d["v"], d["row"]
        f = g.frame_id.to_numpy(int)
        last = xy[-1]
        age = (origin - f[-1]) / 10
        horizon = (int(row.num_frames_output) + offset) / 10
        ball = canonical_xy(np.array([row.ball_land_x, row.ball_land_y]), left) - last
        role = ROLES.index(row.player_role)
        ori = np.zeros(2); ook = False
        if "o" in g and np.isfinite(float(row.o)):
            a = np.deg2rad(float(row.o)); ori = np.array([np.sin(a), np.cos(a)]) * (-1 if left else 1); ook = True
        relative, anchor_v, joint = {}, {}, {}
        for label, key in [("receiver", "Targeted Receiver"), ("passer", "Passer")]:
            ref = role_by_frame[key]
            r = np.zeros((len(g), 2)); av = np.zeros_like(r); ok = np.zeros(len(g), bool)
            if not ref.empty:
                ok = np.isin(f, ref.index.to_numpy())
                if ok.any():
                    a = ref.loc[f[ok]]
                    r[ok] = a[["x", "y"]].to_numpy(float) - xy[ok]
                    av[ok] = a[["vx", "vy"]].to_numpy(float)
                # We use a position-support mask; velocity zero fallback has its own
                # underlying availability. Receiver velocity demand additionally checks vok.
            relative[label], anchor_v[label], joint[label] = r, av, ok
        rv_ok = bool(joint["receiver"][-1])
        if rv_ok:
            rv_ok = bool(role_by_frame["Targeted Receiver"].loc[f[-1], "vok"])
        vector = [(last[0]-60)/60, (last[1]-WIDTH/2)/(WIDTH/2), *list(v[-1]/10),
                  *list(ball/20), age, horizon/5, len(g)/20, float(d["vok"][-1]),
                  *ori, float(ook), *list(relative["receiver"][-1]/20),
                  *list(anchor_v["receiver"][-1]/10), float(joint["receiver"][-1]),
                  *list(relative["passer"][-1]/20), float(joint["passer"][-1]),
                  float(row.player_side == "Offense")]
        for window in (5, 10, 20):
            take = f > origin-window
            xx, vv, ff, vo = xy[take], v[take], f[take], d["vok"][take]
            rr, jj = relative["receiver"][take], joint["receiver"][take]
            if len(xx) == 0:
                vector.extend([0.0]*12); continue
            disp = xx[-1]-xx[0]
            adjacent = np.diff(ff) == 1
            path = float(np.linalg.norm(np.diff(xx, axis=0)[adjacent], axis=1).sum())
            # Straightness is defined only on an uninterrupted observed path.
            straight = float(np.linalg.norm(disp)/path) if path > 1e-8 and adjacent.all() else 0.0
            mean_v = vv[vo].mean(axis=0) if vo.any() else np.zeros(2)
            change = vv[-1]-vv[0] if vo[0] and vo[-1] else np.zeros(2)
            rchange = rr[-1]-rr[0] if jj[0] and jj[-1] else np.zeros(2)
            vector.extend([*list(disp/10), *list(mean_v/10), *list(change/10), path/10,
                           straight, *list(rchange/10), len(xx)/window, jj.sum()/window])
        vector.extend(np.eye(len(ROLES))[role].tolist())
        st = np.asarray(vector, dtype=np.float64)
        if st.shape != (len(STATE_NAMES),) or not np.isfinite(st).all():
            raise ValueError("Invalid state shape or nonfinite feature")
        states[ident], anchors[ident], velocities[ident] = st, last, v[-1]
        ages[ident], horizons[ident], roles[ident] = age, horizon, role
        rec_rel[ident], rec_vel[ident], rec_ok[ident], balls[ident] = relative["receiver"][-1], anchor_v["receiver"][-1], rv_ok, ball
        if flags(pd.Series([row.player_to_predict]))[0]: scored.append(ident)
    return ObservedView(origin, offset, cutoff, left, states, anchors, velocities, ages, horizons,
                        roles, rec_rel, rec_vel, rec_ok, balls, tuple(sorted(scored)))


def query_features(view: ObservedView, player_ids: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return X and constant-velocity reference displacement; no target argument."""
    ids, ts = np.asarray(player_ids), np.asarray(times, float)
    if ids.ndim != 1 or ts.shape != ids.shape or not np.isfinite(ts).all() or (ts <= 0).any():
        raise ValueError("Finite positive query times and matching IDs required")
    rows, ref = [], []
    for ident_raw, t in zip(ids, ts):
        ident = int(ident_raw)
        if ident not in view.states or t > view.horizons[ident]+1e-6:
            raise ValueError("Query player/horizon not in observed task")
        h, age, vel, ball = view.horizons[ident], view.ages[ident], view.velocities[ident], view.balls[ident]
        tau = t+age; total = h+age
        cv = vel*tau
        phase = t/h
        base = [1, t, t*t, h-t, phase, view.offset/10]
        base.extend((view.states[ident]*tau).tolist())
        base.extend(((ball-cv)/20).tolist())
        receiver_future = view.receiver_relative[ident] + (view.receiver_velocity[ident]-vel)*tau
        base.extend((receiver_future/20 * float(view.receiver_valid[ident])).tolist())
        arrival = np.zeros((len(ROLES), len(ARRIVAL_BASE)))
        # Constant-velocity and constant-acceleration arrival hypotheses. The target
        # need NOT reach the landing point; role gates let the learner reject them.
        demand = ball-vel*total
        row = np.r_[demand*(tau/total)/10, demand*(tau/total)**2/10,
                    (view.receiver_velocity[ident]-vel)*tau/10*float(view.receiver_valid[ident])]
        arrival[view.roles[ident]] = row
        rows.append(np.r_[base, arrival.ravel()]); ref.append(cv)
    x = np.asarray(rows, dtype=np.float64); cv = np.asarray(ref, dtype=np.float64)
    if x.shape != (len(ids), len(ALL_NAMES)) or not np.isfinite(x).all():
        raise ValueError("Invalid query feature matrix")
    return x, cv


def make_example(raw: pd.DataFrame, output: pd.DataFrame, offset: int) -> dict[str, Any]:
    """Separate target construction, preserving original rows at offset zero.

    Shifted targets combine withheld pre-pass positions and original labels.
    Missing withheld frames are not invented; no original post-pass row is dropped.
    Offset-zero evaluation never uses augmented views.
    """
    validate_raw(raw)
    if not set(KEYS+["x", "y"]).issubset(output.columns) or output.empty or output.duplicated(KEYS).any():
        raise ValueError("Invalid label keys/schema")
    if not np.isfinite(output[["x", "y"]].to_numpy(float)).all(): raise ValueError("Nonfinite labels")
    ff = output.frame_id.to_numpy(float)
    if not np.isfinite(ff).all() or (ff < 1).any() or (ff != np.floor(ff)).any():
        raise ValueError("Output frame IDs must be positive integers")
    if set(map(tuple, output[KEYS[:2]].drop_duplicates().to_numpy())) != set(map(tuple, raw[KEYS[:2]].drop_duplicates().to_numpy())):
        raise ValueError("Labels belong to another play")
    terminal = raw.sort_values("frame_id").groupby("nfl_id", sort=True).tail(1)
    scored = terminal.loc[flags(terminal.player_to_predict)]
    expected = {(int(r.nfl_id), f) for r in scored.itertuples() for f in range(1,int(r.num_frames_output)+1)}
    actual = set(zip(output.nfl_id.astype(int),output.frame_id.astype(int)))
    if actual != expected: raise ValueError("Original requested label keys/horizon mismatch")
    cutoff = int(raw.frame_id.max()); origin = cutoff-offset
    prefix = raw.loc[raw.frame_id <= origin].copy()
    if prefix.empty: raise IneligibleOrigin("No history before shifted origin")
    view = build_view(prefix, origin=origin, offset=offset)
    original_ids = tuple(sorted(scored.nfl_id.astype(int)))
    if view.scored != original_ids: raise IneligibleOrigin("Shift loses an original scored player")
    if any(origin-int(prefix.loc[prefix.nfl_id.eq(i),'frame_id'].max()) >= 20 for i in original_ids):
        raise IneligibleOrigin("Scored player's observed history is too stale")
    future = output[KEYS+["x","y"]].copy(); future["frame_id"] = future.frame_id.astype(int)+offset
    future["prethrow_target"] = False
    if offset:
        withheld = raw.loc[(raw.frame_id > origin)&raw.nfl_id.isin(original_ids), KEYS+["x","y"]].copy()
        withheld["frame_id"] = withheld.frame_id.astype(int)-origin
        withheld["prethrow_target"] = True
        future = pd.concat([withheld, future],ignore_index=True)
    future = future.sort_values(["nfl_id","frame_id"]).reset_index(drop=True)
    if future.duplicated(KEYS).any(): raise ValueError("Duplicate augmented target keys")
    x, cv = query_features(view, future.nfl_id.to_numpy(),future.frame_id.to_numpy(float)/10)
    y = canonical_xy(future[["x","y"]].to_numpy(float),view.left)-np.stack([view.anchors[int(i)] for i in future.nfl_id])
    meta = future[KEYS].to_numpy(np.int64)
    return {"X":x,"y":y-cv,"cv":cv,"keys":meta,
            "offset":np.full(len(y),offset,dtype=np.int16),
            "prethrow":future.prethrow_target.to_numpy(bool),
            "role":np.array([view.roles[int(i)] for i in future.nfl_id],np.int8),
            "state_count":len(view.states),"original_rows":len(output)}


def fit_ridge(x: np.ndarray,y: np.ndarray, *, penalty: float=0.01) -> dict[str,np.ndarray]:
    """MSE + fixed L2 penalty on TRAIN-standardized columns; intercept unpenalized.

    Scaling and near-constant screening are fitted only to each arm's training data.
    No tuning on validation, target encoding, or parent-model baseline is used.
    """
    x,y = np.asarray(x,float),np.asarray(y,float)
    if len(x)==0 or y.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid fitting arrays")
    if not np.isfinite(penalty) or penalty<=0: raise ValueError("Positive fixed regularization required")
    mean, scale = x.mean(0), x.std(0)
    keep=scale>1e-8
    z=(x[:,keep]-mean[keep])/scale[keep]
    ym=y.mean(0)
    gram=z.T@z/len(x)+penalty*np.eye(z.shape[1])
    coef=np.linalg.solve(gram,z.T@(y-ym)/len(x))
    return {"mean":mean,"scale":scale,"keep":keep,"coef":coef,"intercept":ym}


def predict_ridge(model:dict[str,np.ndarray],x:np.ndarray) -> np.ndarray:
    keep=model["keep"]
    return ((np.asarray(x,float)[:,keep]-model["mean"][keep])/model["scale"][keep])@model["coef"]+model["intercept"]


def chronological_folds(game_ids:np.ndarray) -> list[dict[str,np.ndarray]]:
    games=np.unique(game_ids).astype(np.int64)
    dates=np.unique(games//100)
    if len(dates)<8: raise ValueError("Need at least 8 distinct training dates for chronological screening")
    cuts=[int(len(dates)*v) for v in (.50,.67,.84)]+[len(dates)]
    folds=[]
    for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:]),1):
        tr=games[(games//100)<dates[a]]
        va=games[np.isin(games//100,dates[a:b])]
        if len(tr)<4 or len(va)<2: raise ValueError("Insufficient games in chronological fold")
        if np.max(tr//100)>=np.min(va//100): raise ValueError("Chronological overlap")
        folds.append({"fold":i,"train_games":tr,"validation_games":va})
    return folds


def paired_bootstrap(y:np.ndarray,a:np.ndarray,b:np.ndarray,games:np.ndarray,seed:int=20260911,repeats:int=2000)->dict[str,float]:
    unique=np.unique(games)
    if len(unique)<2: raise ValueError("Need multiple games")
    sa=np.array([np.square(y[games==g]-a[games==g]).sum() for g in unique])
    sb=np.array([np.square(y[games==g]-b[games==g]).sum() for g in unique])
    n=np.array([(games==g).sum() for g in unique])
    rng=np.random.default_rng(seed); deltas=[]
    for _ in range(repeats):
        ix=rng.integers(0,len(unique),len(unique))
        deltas.append(np.sqrt(sb[ix].sum()/(2*n[ix].sum()))-np.sqrt(sa[ix].sum()/(2*n[ix].sum())))
    lo,hi=np.quantile(deltas,[.025,.975])
    return {"delta_rmse":rmse(y,b)-rmse(y,a),"ci_low":float(lo),"ci_high":float(hi),"resamples":repeats}
