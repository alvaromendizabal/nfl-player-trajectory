# Model card: landing-aware NFL trajectory prediction

## Intended use and evidence

Research and portfolio demonstration of pre-throw tracking features, temporal
validation, interpretable residual modeling, and reproducible inference. Not a
production player-assessment system or a claim of a competition-winning model.

The completed experiment selects `landing_ridge`: coordinate RMSE 0.9268683892 yards,
frame-weighted ADE 0.8253979992, trajectory-weighted FDE 1.4167735708, and p95
Euclidean displacement 2.7537849192. The reference role ridge has RMSE 0.9895687826.
See `results/feature_summary.json` for the complete precision and comparison table.

## Data and protocol

NFL Big Data Bowl 2026 Prediction competition, observed 2023 tracking. Whole games
are separated chronologically: 192 train, 32 validation, 48 reserved holdout.
Validation has 67,857 player-frames and 5,399 trajectories. Holdout is not evaluated.
The supplied landing point and forecast horizon are permitted inference inputs.
Future positions, post-play outcomes, names, and player IDs as predictors are excluded.

## Model

A role-conditioned six-vector ridge baseline is corrected by a 64-feature residual
ridge model. The bank contains 2,843 domain-informed candidates. Candidate screening,
correlation pruning, centering, scaling, and coefficient fitting use training data.
Fixed residual regularization is 0.01. Corrections are learned in the play-direction-
normalized coordinate system and transformed back for the requested output rows.

The original baseline uses landing geometry. Therefore the motion-only residual
ablation is not a no-landing-information end-to-end model. The residual correction
is not generally rotation-equivariant like the original shared-vector baseline;
field boundaries and play-direction normalization provide a football-specific frame.

## Interpretation and limitations

The selected model improves RMSE 6.34% versus role ridge. Its paired 95% game-cluster
bootstrap difference interval is [-0.0828731, -0.0433695] yards (2,000 resamples).
These development intervals are not corrected for repeated model selection. No
independent season, reserved-holdout, or Kaggle leaderboard result is claimed.

The interaction variant replaces 23 selected landing-model columns. It is not a
clean experiment that only adds interactions while keeping the useful inputs fixed.
Training residual correlations are marginal associations, not causal importance
or out-of-fold importance. Long forecasts and defensive coverage dominate remaining
squared error. The final forecast bucket has only 127 frames.

## Artifact and inference controls

The attached experiment summary matched the completed private snapshot's report
and publication receipts. `results/feature_analysis.json` records the inspected
selection overlap. Raw tracking and private checkpoints are not committed.

User-invoked export verifies source hashes, baseline/model identity, frozen split,
training-game identity, and completion receipts. It embeds the tested numerical
implementation. The organizer interface returns numeric x/y only, in incoming row
order. Per-play checkpoints use model/input signatures and output hashes.
No export or report code calls a Kaggle submission API. Official gateway execution
and submission eligibility remain to be verified in the user's own account.
