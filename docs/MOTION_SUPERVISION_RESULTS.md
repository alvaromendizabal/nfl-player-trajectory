# Learned motion supervision: executed results

Six fixed-budget fits are complete, dated 10 September 2026. Motion supervision
scores **0.66029 coordinate RMSE**, compared with
**0.69635** for position-only supervision on exactly the
same **202,361 rows from 98 later games**. The pooled reduction is
**5.18%**. Improvement in every fold: **True**.
The declared primary gate **passes**. **The 0.46 objective remains unmet.**

This is a completed feature-learning experiment, not a completed project or a
new Kaggle result. Recorded Kaggle private RMSE remains **0.70090**. The previous
declared domain-correction/tree blend scored **0.64160** on these same internal
rows; retain that result when comparing the new blends below. No hidden-test
submission was made.

## What changed and what the comparison identifies

Both arms learn per-signal temporal features from the existing 30 observed
channels and their masks, then combine players with attention. Both have the
same 281,430 parameters, starting weights, preprocessing,
48 passes, optimizer, reflection schedule and fixed final-EMA selection.
The treatment additionally supervises four outputs: x/y velocity and x/y
acceleration, constructed from consecutive future coordinates **as training
labels only**. Its shared latent features must retain information useful to
these motion tasks. The control contains the same auxiliary head with zero
auxiliary-loss weight. Read the [frozen protocol](MOTION_SUPERVISION_EXPERIMENT.md).

No new independent observed fields or hand-built candidate columns are added.
The 124 internal temporal channels are learned features, not 124 new data
sources. This isolates the joint auxiliary package; velocity and acceleration
have not yet been separated. The new model also differs from the old attention
in stem, batch size, learning-rate settings and exposure. Comparisons with that
old reference combine those effects and are secondary.

## All declared models

| Model | Fold 1 | Fold 2 | Fold 3 | Pooled RMSE |
|---|---:|---:|---:|---:|
| Preserved tree | 0.69851 | 0.67209 | 0.73530 | 0.70260 |
| Previous attention | 0.69855 | 0.67041 | 0.79213 | 0.72063 |
| Matched position supervision | 0.67475 | 0.65881 | 0.75695 | 0.69635 |
| Position + motion supervision | 0.62967 | 0.63138 | 0.72456 | 0.66029 |
| Previous attention/tree 50/50 blend | 0.66669 | 0.63754 | 0.73019 | 0.67866 |
| Position-supervised/tree 50/50 blend | 0.63458 | 0.61411 | 0.69489 | 0.64787 |
| Motion-supervised/tree 50/50 blend | 0.61157 | 0.59578 | 0.67475 | 0.62708 |

The headline metric pools squared coordinate errors before taking the square
root. It is not an average of unequal fold RMSEs. The sole primary contrast is
motion-supervised minus matched position-supervised RMSE:
**-0.036055 yards**, with paired game-bootstrap 95% interval
**[-0.055866, -0.016358]**
from 10,000 resamples. The gate requires improvement in all three folds, at
least 1% pooled reduction, and a negative interval upper bound. All 50/50 blends
were declared in advance; their weights were not fitted to these outcomes.

These folds were reused during research. One seed and a conditional paired
interval do not correct for the complete history of adaptive experiments or
establish cross-season generalization. No reserved-game outcomes were used in
this study. The previously inspected reserved partition is no longer untouched.

## Forecast periods and football roles

| Forecast time | Rows | Position supervision | Motion supervision |
|---|---:|---:|---:|
| 0–0.5 s | 82,540 | 0.14011 | 0.11948 |
| 0.5–1 s | 68,620 | 0.38779 | 0.34896 |
| 1–2 s | 43,148 | 1.05907 | 0.98854 |
| Above 2 s | 8,053 | 2.16624 | 2.12976 |

Every requested row is retained, including long forecasts and incomplete player
contexts. [Role slices](results/supervision_roles.json) and notebook 02 section
14 show the roles actually scored, using the independently audited raw-category
mapping. These are descriptive diagnostics, not separate feature-selection
tests. Overall improvement does not imply equal improvement for every player.

The new motion model still places **47.79% of squared error in the 1–2 second
window** and **41.40% after two seconds**. Those windows contain 21.32% and 3.98%
of rows, respectively. As an error-budget calculation only, making every error
after two seconds zero while leaving all others unchanged would still give
**0.50545 RMSE on the complete population**. This is an oracle counterfactual,
not a model result or an attainable performance bound. It shows that fixing
only the extreme tail cannot achieve 0.46 on these internal rows; the ordinary
1–2 second movement also needs improvement. It says nothing about an unchanged
model's score on the distinct Kaggle population.

The separate [training-target audit](results/motion_conditioning.json) found
that roughly 3.8% of training rows occur after two seconds, yet account for
48.8–52.5% of ridge baseline squared error. One training request extrapolates
49–55 yards along x against about 12 yards of true displacement. This supports
testing target parameterization and realistic motion beyond short horizons;
it does not establish bad labels or justify removing plays.

## Execution, recovery and evidence

The six fits used local CPUs; no new AWS training job was launched. Their
recorded training-wall counters sum to **5620.0 seconds
(93.7 aggregate arm-minutes)**, with a maximum of
**1187.9 seconds** per fit, inside the
1,200-second per-arm cap. These counters exclude preparation, reporting and
quality checks, and are not a billing statement. Folds ran concurrently with
two threads per worker. An early serial launch was stopped; fold 1 resumed its
epoch-9 checkpoint. A partial epoch may have been replayed during that switch.

Six targeted tests passed: physical derivative construction and missing frames,
training-only scales and reflection, future/identity exclusion, player
permutation and padding, auxiliary gradients, and exact interrupted recovery.
The [real completed-fit replay](results/supervision_recovery.json) verifies all
six fits are reused, **zero extra training epochs** execute, and all **24 fitted
files keep identical content and modification times**. The
[publication manifest](results/supervision_manifest.json) checks source, inputs,
fitted artifacts and reports. The [notebook receipt](results/supervision_notebooks.json)
records the completed full quality gate and all three canonical executions.

## Feature research remains open

The [extended primary-source review](MOTION_LEARNING_RESEARCH.md) records
mechanisms, available inputs, transfer limits and unresolved feature families.
The next attribution question is whether velocity, acceleration or their joint
effect causes any supported gain. High-value unresolved work also includes
target conditioning, learned uncertain matchups, route/coverage auxiliary labels,
and eligible historical NFL data. A favorable bounded experiment cannot justify
closing these avenues or declaring state-of-the-art performance.
