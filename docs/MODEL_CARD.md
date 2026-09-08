# Role-conditioned motion ridge

A compact, interpretable reference for the NFL Big Data Bowl 2026 Prediction task.
This is a measured development model, not a claim of frontier leaderboard performance.

## Inputs and behavior

For each scored player, use only observed x/y history, the supplied player role,
ball landing coordinates, and number of requested future frames. Output frame 1
occurs 0.1 seconds after the final observed frame. Identity keys align rows; player
IDs are not fitted model features. No supplementary post-play outcome is used.

Let `t = output_frame / 10`, `q = output_frame / num_frames_output`, and
`b = supplied_landing_point - final_observed_position`. The six vector basis terms
are `v_last*t`, `v_recent*t`, `a_last*t²/2`, `b*q`, `b*q²`, and `b*q³`.
The prediction is final observed position plus a weighted sum of these vectors.

Recent velocity is a least-squares estimate over at most five observed frames.
Acceleration uses the interval between consecutive velocity midpoints, including
irregular sampling. A single observation falls back to zero velocity/acceleration.
Weights are shared across coordinates, preserving translation and rotation
equivariance. Input coordinates are returned in their original field orientation.
No arbitrary field clipping is applied.

## Training and selection

- Training: 192 games, 32,681 trajectories, 395,813 target positions.
- Dates: September 7–December 3, 2023.
- Objective: mean squared coordinate displacement plus ridge regularization.
- Scaling: training RMS of each vector feature; no centering or fitted coordinate intercept.
- Regularization: alpha 0.001, declared before validation.
- Parameters: six coefficients for each of two scored roles, plus a six-coefficient global fallback.
- Optimization: additive normal-equation statistics and a small deterministic linear solve.
- Selection: lowest coordinate RMSE among six predeclared methods on chronological validation.
- Holdout: 48 games, unscored by this development pipeline.

Validation contains 32 games, 5,399 player trajectories and 67,857 target positions
from December 4–18, 2023. All frames and players of each game stay together.

| Measure | Result |
| --- | ---: |
| Coordinate RMSE | 0.9895688 yd |
| RMSE 95% game-cluster interval | 0.9218024–1.0518861 yd |
| Frame-weighted ADE | 0.8847315 yd |
| Trajectory-weighted ADE | 0.7217791 yd |
| Trajectory-weighted FDE | 1.5057010 yd |
| p95 displacement | 3.0028329 yd |
| Coordinate MAE | 0.5632612 yd |
| RMSE reduction vs constant velocity | 42.5510% |
| Paired RMSE difference 95% interval | −0.7789459 to −0.6888573 yd |

Intervals use 2,000 whole-game bootstrap samples and seed 2026. Each replicate pools
squared errors and coordinate counts before taking a square root. They are not
confidence intervals over arbitrary future seasons, nor do they correct repeated
validation selection.

## Latency and operational limits

The measured median was 27.09 ms per play and p95 was 87.75 ms on the recorded Linux
x86_64 environment, over the first 32 validation plays in the first validation week.
One warmup precedes timing. Timing includes feature construction and prediction,
excluding CSV loading, server startup and networking. It is a small environment-specific
measurement, not an AWS serving SLA. The library operates one weekly pair at a time.

Long-horizon and defensive-coverage errors remain larger. The model lacks explicit
player interactions and multimodal trajectories. It was trained on one season;
cross-season performance and the final holdout are unmeasured. Supplied landing
coordinates are permitted competition inputs but would not be known to every live
football application. Reusing this model without them changes the task.

## Reproducibility and provenance

`docs/results/model.json` contains fitted coefficients and training-game identifiers.
`protocol.json` freezes date boundaries and metric settings; `summary.json` records
input checksums and numerical-code/environment fingerprints. See `docs/VALIDATION.md`
for test and execution evidence. The standalone Kaggle export embeds the coefficients;
it requires an official local-gateway test before a submission is attempted.
