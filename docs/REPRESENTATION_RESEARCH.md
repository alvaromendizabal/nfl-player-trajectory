# Smoothed motion in the temporal encoder

This experiment follows the [domain review](DOMAIN_RESEARCH.md) and motion
addition/removal study. Its purpose is to determine whether the supported
smoothed-state information improves the entire temporal model under a matched
continuation budget. It does not establish an exhaustive feature search or a
state-of-the-art predictor. The objective of 0.46 RMSE remains unmet, and the
recorded Kaggle private score remains 0.70090. No new submission was made.

## Hypothesis and information boundary

Three-, eight- and twenty-frame summaries estimate motion over different scales.
Backward position slopes reduce differencing noise; supplied and derived velocity
provide complementary state estimates. Their disagreement indicates uncertain
telemetry. Supplied velocity trends, supplied acceleration and window coverage
complete the thirty-candidate family. Every value uses the observed input window,
with actual frame spacing and masks. No forecast coordinates enter the features.
These are explicit summaries of information already available to the model,
not new external observations.

The [released winning training notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)
and the broader primary-source review motivate preserving motion in a learned
representation. The specific zero-initialized continuation experiment is our
hypothesis; the cited solution did not validate these thirty channels or this
protocol. Auxiliary motion tasks and feature-wise temporal encoders remain
separate, untested candidates in this project.

## Declared comparison

The [protocol](REPRESENTATION_EXPERIMENT.md) was frozen before fitting. The six
fits use the same three chronological folds as the preceding study: 94/41,
135/28 and 163/29 training/evaluation games. They cover 98 later evaluation games
and 202,361 scored rows, wholly within the original 192-game training partition.
All evaluation rows remain present, including long forecasts and large errors.

Each fold initializes from its saved 40-epoch attention EMA. The first temporal
convolution receives thirty extra channels with zero initial weights. The old
observation-mask weight retains its position. Initial predictions are checked
against saved reference errors before any training. Both arms have 294,202
parameters; the control zeros the new channels after normalization. All original
layers remain trainable. Both arms receive twelve additional epochs, the same
seed, deterministic play order and reflection draws, fresh AdamW state, cosine
schedule, gradient clipping and EMA. The final EMA is selected in advance;
validation does not select an epoch or a blend weight.

Screening and normalization fit only training players plus deterministic
reflections. Constant and duplicate channels are zeroed. Raw reflected motion is
normalized using the same training statistics and clipped to [-10, 10]. New state
is repeated across the observed temporal window. This is one representation
choice; it does not exhaust learnable smoothing or motion supervision.

## Executed results

| Model | Fold 1 | Fold 2 | Fold 3 | Pooled RMSE |
|---|---:|---:|---:|---:|
| Existing tree | 0.698510 | 0.672092 | 0.735295 | 0.702601 |
| Frozen attention | 0.698549 | 0.670412 | 0.792132 | 0.720631 |
| Matched continuation control | 0.695295 | 0.665584 | 0.790314 | 0.717450 |
| Motion-state inputs | 0.692959 | 0.665051 | 0.790118 | 0.716307 |
| Previous attention/tree blend | 0.666686 | 0.637543 | 0.730194 | 0.678661 |
| Continuation control/tree blend | 0.662687 | 0.633800 | 0.729092 | 0.675681 |
| Motion-state/tree blend | 0.661702 | 0.633668 | 0.729012 | 0.675219 |

The primary feature gate **fails**: the pooled feature gain is only **0.1593%**,
below the declared 1% requirement. The direction favors features in all three
folds and the conditional paired interval excludes zero, but statistical
detectability does not meet the practical improvement threshold. Both neural
arms remain worse than the tree. The measured comparison is:

```json
{
  "coordinate_rmse_yards": 0.716306841558617,
  "reference_rmse_yards": 0.7174497193528406,
  "rmse_difference": -0.001142877794223618,
  "relative_reduction": 0.0015929726688784784,
  "paired_game_delta_interval": [
    -0.0019891102607536154,
    -0.00033226444678497206
  ],
  "interval_level": 0.95,
  "simultaneous_comparisons": 1
}
```

The gate requires at least 1% pooled RMSE improvement over the matched control,
improvement in every fold, and a paired whole-game 95% difference interval
strictly below zero. Ten thousand paired game-bootstrap replicates are used.

Training-only screen counts: inner_1: 28/30 retained, inner_2: 28/30 retained, inner_3: 28/30 retained.

The new fixed motion/tree blend scores **0.675219**, versus **0.678661** for
the previous attention/tree blend and **0.675681** for the equally continued
control/tree blend. Most of that continuation gain is not attributable to the
new features. The prior correction/tree blend remains stronger at **0.641603**.

The fixed 50/50 blends are declared secondary diagnostics. They have no fitted
weight and are not an independently validated selection winner. The prior
all-feature correction/tree blend remains 0.641603 on these same rows; it is
reported in the parent review with its failed standalone consistency gate.
A result from the original 32-game development partition cannot be substituted
for any number in this table.

## What this resolves, and what it leaves open

This motion-input continuation does not earn promotion under its declared
feature gate. Extra epochs and the effect of the thirty inputs are reported
separately, so any improvement from additional training is not credited to
feature engineering. The earlier stronger correction result is preserved.

The first study established that motion summaries can improve a frozen-model
correction. This study tests whether repeating the supported state through the
existing temporal encoder provides the same benefit. A weak or adverse result
rejects promotion of this particular continuation, not the importance of motion.
The next high-value representation questions are feature-wise temporal learning,
velocity/acceleration auxiliary supervision, strictly observed forecast-origin
augmentation, and aligned older NFL tracking. Their effects need separate,
matched comparisons. Additional nearest-neighbor coverage summaries have not
earned more compute; uncertain learned assignments remain a different hypothesis.

## Role contract audit

The [reproducible audit](results/role_contract_audit.json) verifies the hashes of
all fifteen available input-week files and inspects 4,031,663 observed rows. It
finds four raw roles and no missing roles. The historical vocabulary contains
`Pass Route`, whereas the actual label is `Other Route Runner`; the latter maps
to the historical fallback slot. The four observed categories still occupy four
distinct slots. Relabeling alone therefore does not demonstrate lost categorical
information or an explanation of the score gap. Historical checkpoint encoding
is preserved in this comparison. Future presentation should use actual raw labels.

## Recovery, compute and limits

The interrupted-training test resumes midway through an epoch and reproduces
every EMA tensor and the complete training curve exactly. A separate replay of
all six completed fits verifies unchanged checkpoint, optimizer, prediction and
summary bytes and modification times with zero additional training epochs. The
[recovery receipt](results/representation_recovery.json) records that check.
All numerical sources, parent inputs and private fitted outputs are hash-bound;
the publication manifest verifies the evidence consumed by canonical notebooks.

The six new fits ran on the existing local CPU environment with a declared
450-second limit per arm. No AWS training resource was launched. Total fitted training time was **674.94 seconds** across the six models;
the complete initial run took **778.60 seconds** including preparation and
evaluation. Actual per-arm
timing and all twelve-epoch curves are in the [result](results/motion_representation.json).
No seed, epoch, feature gate or ensemble weight was changed after viewing scores.

These historical folds have already influenced feature choices. One seed and
paired game bootstraps quantify a conditional comparison, not uncertainty from
all previous adaptive selection. The data does not establish across-season
generalization. The original development and previously inspected reserved set
were not rescored. Feature research remains open; this report supports neither
final-model promotion nor a claim of reaching 0.46.
