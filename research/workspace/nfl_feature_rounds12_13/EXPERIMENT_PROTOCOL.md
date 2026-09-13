# Two independent feature studies — frozen before evaluation

## Common design

Round 12: receiver-relative history. Round 13: observed motion dynamics.
Each has 12 numeric player-by-frame channels, 12 explicit validity masks,
three equal-capacity models, two notebooks and nine Plotly views. Both retain
699 training plays / 26,039 forecast rows and 152 evaluation plays / 5,886 rows
from the same 14 evaluation games. These are reused internal fold-2 games.
No new season, unseen holdout or Kaggle result is available through this protocol.

The 62 direct-state tree treatment, goal-geometry tree treatment, passing-line
bank, earlier-origin mixture, and historical-response treatments are not silently
added. The Round 10 group-separated Cartesian architecture is the starting code,
with its player encoder widened from 10 to 22 channels. Its original six numeric
goal-pair channels stay zero for every arm. All original group/mask/age inputs
remain shared. Every new model has 19,826 parameters.

| Arm | Extra numerical values | Validity masks |
|---|---|---|
| mask | All 12 zero | All 12 actual masks |
| core | First six actual; last six zero | Identical to mask |
| full | All 12 actual | Identical to mask |

A versus B isolates the audited six numerical values conditional on their masks
and the expanded encoder; B versus C isolates the added six; A versus C evaluates
the complete numeric family. This does not separately attribute masking, capacity,
or architecture relative to old models. A new matched control is necessary: an
old 10-channel encoder is not a valid capacity-matched control for this extension.
Round 12 results never enter Round 13 selection, features, architecture, seeds,
normalization, stopping thresholds, or initialization.

## Information timing

Only observed node, pair, role and mask arrays enter feature construction. No y,
future coordinates, parent predictions, player-history targets, or evaluation
labels are accepted by the feature functions. Prediction requests and base
features are used by the downstream forecast interface, not by feature generation.
The supplied ball landing point is organizer input. It is not an unknown endpoint
estimated from future player coordinates in this experiment.

No future information fills gaps. Rates require actual consecutive 0.1-second
slots. All rolling windows trail their endpoint and require full valid contiguous
support. Self-receiver relations are masked. Multiple targeted receivers are an
error, not an arbitrary choice. The first six channels are byte-equivalent values
and masks to the corresponding Round 11 feature arrays. Existing raw CSVs are not
read; the authenticated original tensors and cached labels are reused.

## Feature formulas

Full definitions, units, fixed scales and minimum observation requirements are in
`FEATURE_DICTIONARY.csv`; executable source is `families.py`.

Round 12 core: receiver relative x/y, relative vx/vy, distance and closing speed.
Extension: signed lateral relative speed; velocity alignment; adjacent bearing
rate; adjacent separation change; relative position projected along/across the
player-to-landing direction. Projection is a coordinate reference, not a coverage
assignment or a claim of real ball flight. Several concepts overlap prior pair
features; this study tests direct player-history exposure rather than new signals.

Round 13 core: observed ax/ay, tangential and normal acceleration, velocity turn
rate and speed-change rate. Extension: 3-frame and 5-frame trailing mean ax/ay and
adjacent jerk x/y. These are finite differences of observed velocity, not future
acceleration supervision. They can amplify tracking noise; no clipping, outlier
removal, adaptive window search or validation-fit scaling is performed.

## Training and compute

Identical seeded initial weights, 24 epochs, batch size 8 plays, AdamW learning
rate .001, weight decay .0001, gradient-norm cap 5, two CPU threads and the existing
row-normalized squared-error objective. All settings are the same across arms
and rounds. Base normalization is fit only on training rows; physical feature
scales are constants. Default epoch exposure is 2,112 optimizer steps per arm.

Each round profiles all three arms with 8 disposable steps per arm on large
training-only batches. A conservative projection must be <=300 seconds per arm.
Hard stage stops: preflight 120s, smoke 120s, preparation 360s, runtime tests 180s,
profile 180s, each train arm 360s, evaluation 180s, replay 240s, report 60s.
The limits are caps, not runtime estimates. No cap can be increased via the CLI.

Completed optimizer/RNG checkpoints are immutable and saved at most every 25
optimizer steps plus epoch boundaries. An interruption can resume only with
matching source, input, environment, protocol and checkpoint identity. Do not
retry deterministic errors without diagnosis. Existing successful checkpoints
are not regenerated merely to obtain a new report.

Evaluation opens labels only after all three arms finish their fixed exposure.
No validation checkpoint selection, learning-rate tuning, best-epoch picking,
extra seeds, drop-row rules or ensemble-weight search is included. Prediction
replay performs no optimizer steps and refuses missing artifacts.

## Metrics and decisions

Coordinate RMSE is sqrt(SSE/(2N)) over all requested coordinates. Report final
training RMSE as descriptive, evaluation RMSE, horizon and role slices, and three
paired game-bootstrap contrasts per round. Bootstrap: 20,000 fixed-seed whole-game
resamples. Bonferroni-adjusted two-sided percentile tails use .05/(2*6), sharing
a family across the six predeclared comparisons in these TWO rounds.

This adjustment does not account for the many earlier adaptive experiments or
make reused games an untouched test set. Treat the intervals as an exploratory
screen conditional on one seed, this sample, and these repeated game splits.

Core needs >=1% gain and adjusted upper delta below zero versus mask. Full needs
those conditions versus both core and mask. Before spending another fold, the
candidate must also be no worse than the preserved tree on the same forecast
keys. A failed feature gate is a completed scientific result, not an execution
failure. Complete the other predeclared round unchanged after numerical replay;
do not stack favorable arms. Integrity or runtime/budget failures require a report
before further work. Neither round launches another fold or a Kaggle submission.

## Beyond these two rounds

The 20.85% observed-play coverage is a separate unresolved limitation. Next verify
unused-play labels and build a bounded data-scale-only study, preserving model
and feature definitions. Do not claim a 0.46340 comparison until an actual
comparable hidden-test result exists. No model superiority or feature maturity
is implied by counts, software tests, gradient checks or synthetic validation.
