# Synchronized player-relationship histories

**11 September 2026 — opt-in feature prerequisite; predictive value is unmeasured.**
No existing model, checkpoint signature, scientific configuration, or competition
submission is modified by this module. Feature research remains open. The existing
coordinate/velocity run must be inspected before any additional scientific fit.

## Evidence and hypothesis

The official final private [leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard)
reports **0.46340** for the winner. The project's recorded **0.70090** Kaggle result
is separate from its historical **0.62708** internal blend, whose missing model
artifacts still prevent replay. A lower internal score does not establish that
we have beaten the winning submission. The target is strictly below the verified
winning score on a comparable test, not merely below 0.62 on a reused fold.

The [winning author's account](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
emphasizes compact motion inputs, temporal/player modeling, auxiliary motion
objectives, augmentation, and substantially more training exposure than this
project's bounded reconstruction. This argues against treating more columns as
a complete solution. The win is not evidence that this new module will help.

[Song et al., 2026](https://arxiv.org/abs/2603.25901) use factorized temporal/agent
attention for changing defensive responsibilities. Their task and supervision
are coverage identification, not this project's post-throw coordinate RMSE.
They motivate a relationship-history hypothesis; we do not import unavailable
coverage labels, later player positions, or the paper's accuracy claims.

**Hypothesis:** explicit, synchronized pair histories may preserve changing
separation, relative movement, and alignment better than terminal geometry alone.
This is a proposed representation for a later learned interaction encoder, not a
verified defender assignment, learned encoder, or already measured improvement.
It is deliberately different from rerunning the rejected fixed soft-affinity
ridge treatment, but its benefit still requires a new controlled experiment.

## Implemented information contract

`pair_history(history, observed, side, role)` consumes the maintained canonical
20-frame observed grid. It accepts no future labels, prediction outcomes, player
IDs, validation statistics, or fitted normalization. Stable player slots and a
shared cutoff come from `temporal_data.play_features` / `encode_play`.

For focal i and neighbour j at a jointly observed frame, geometry is directed
j relative to i. The difference of the two organizer-supplied ball-relative
offsets equals the true player displacement; the common landing point cancels.
Subtracting each player's individually centered `relative_x/y` would use different
origins and is explicitly tested against. The input adapter, not this tensor-only
function, establishes the shared landmark and pre-throw cutoff; arbitrary raw
tables must not bypass that adapter contract.

The 12 channels are grouped for later ablation, not counted as proven improvements:

| Group | Channels | Units / interpretation |
|---|---|---|
| Geometry | dx, dy, distance, separation support | Position divided by 20 yards; explicit degenerate-distance flag. |
| Relative motion | dvx, dvy, closing speed, velocity alignment, alignment support | Velocity divided by 10 yards/s; positive closing means approaching; unsupported zero-speed alignment is not treated as an angle. |
| Context | Same side, neighbour is targeted receiver, neighbour is passer | Supplied categories, not retrospective coverage labels. |

The mask requires both players at the **same original frame** and excludes
self-edges. Missing values are masked before arithmetic, never forward filled.
The returned clock preserves `[-1.9, ..., 0.0]` seconds relative to the observed
cutoff. Unknown roles keep the existing explicit category. Fixed physical scales
come from the maintained encoder; numerical support thresholds are not tuned on
validation. Inputs are not mutated, including read-only NumPy arrays.

`terminal_view()` selects the final **joint** observation for each pair without
changing the tensor shape or its time. If that frame is -0.3 seconds, it remains
-0.3, not zero. Disjoint observations never form a fictional terminal pair.
This supplies an interface for a later equal-capacity history/terminal ablation;
that training-loop integration is not implemented in this prerequisite.

## Executed evidence and limitations

[Notebook 03](../notebooks/03_player_relationships.ipynb) is an executed analytic
example with invented linear trajectories, a gap-preserving Plotly chart, and a
text-table fallback. It asserts independent distance calculations, actual joint
observation age, missing-value immunity, and absent self-edges. It is not an NFL
validation notebook and does not train or download data.

[The receipt](results/relation_history.json) records the actual local test run,
versions, module hash, synthetic maximum-player timing, returned-array size, and
repeat-output verification. Full locked-runtime GitHub CI is reported separately
in the PR. The previous notebook files and frozen experiment are unchanged.

The initial physics test failed only because its default zero absolute tolerance
was tighter than the float32 input quantization error (maximum difference
2.98e-8 in normalized units). The independent expected geometry was retained;
an explicit 1e-7 absolute test tolerance was added. This was not a changed model,
changed feature, or relaxed scientific performance gate.

At 22 players, the feature/mask/clock output is bounded and should be built per
minibatch. Returned-array bytes are **not** a claim about total training memory;
intermediate float64 arrays and any learned encoder add memory. No cloud
throughput, private-data support distribution, causal benefit, or new RMSE is
claimed by synthetic timing.

## Next experimental gates

First recover and review `nfl-motion-scientific-20260911-055510-d265d9d` without
launching a duplicate. Its existing >=1% RMSE plus negative paired-game confidence
bound rule remains unchanged. If it passes, replicate its frozen treatment before
attributing gains to another representation. If it fails, diagnose execution
versus scientific failure and stop the rejected exact treatment.

Before fitting this relationship representation: audit training-only pair support
and age distributions, freeze the learned interface and its capacity-matched
terminal control, verify masks at that interface, smoke-test real-data loss and
checkpoint replay, and freeze one chronological comparison and resource budget.
Then evaluate all requested coordinates with official RMSE and game-cluster
uncertainty. Do not choose pair scales, cutoffs, loss weights, or subgroup gates
from observed validation outcomes. Preserve negative results and long horizons.

No extra cloud job, paid compute, Kaggle submission, or claim of an AWS deployment
is part of this implementation milestone. Exact GitHub/AWS synchronization needs
a fresh workspace read and verified deployment, separately from a GitHub merge.
