# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with controlled sequence modeling, ensemble diversity, GPU inference engineering, and reproducible AWS research.**

[Latest aggregate notebook](notebooks/05_multisplit_score_and_acceleration.ipynb) · [Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [Model card](docs/MODEL_CARD.md) · [Recent model review](research/RECENT_MODELS.ipynb) · [Run and reproduce](START_HERE.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. Competition slug: `nfl-big-data-bowl-2026-prediction`. AWS/SageMaker is the canonical research workspace; Kaggle is used only for the required submission surface.

## Current research snapshot

The strongest recorded late post-competition private submission is **0.46487 coordinate RMSE**, improving the prior **0.46547** result by **0.00060**. The published first-place private comparator is **0.46340**, leaving a **0.00147 RMSE gap**. No official competition rank is claimed for late submissions.

The strongest completed local system remains the **20-model, four-split-family equal-weight ensemble** at **0.4631723213 OOF RMSE over 561,607 rows**. Local OOF and private leaderboard measurements are intentionally kept separate.

A separate AWS inference study reduced measured median per-play latency from **0.541 s to 0.113 s — 4.784× faster** by sharing input preparation across the 20-model ensemble. Predictions were bitwise-identical on the declared **96-play / 3,723-row** parity sample. Faster vectorized alternatives that exceeded the numerical tolerance were rejected.

## Research decisions since the prior publication

The multisplit ensemble earned a stronger private score and remains the competition-facing baseline. Several subsequent feature studies were **not promoted** because their control or evaluation checks did not establish a credible improvement. Recovery work showed that evaluation fixes alone did not restore the weak control, and a source audit identified drift between the successful training contract and later high-throughput experiments.

The next accuracy experiment therefore returns to the verified training contract before testing another feature configuration. It is prepared for AWS execution but has **no published metric yet**.

## What the project demonstrates

- **Sequence modeling:** temporal convolutions, cross-player interaction, trajectory uncertainty, motion auxiliaries, moving-average inference, and controlled augmentation.
- **Validation discipline:** grouped-game folds, exact coordinate RMSE, paired-game uncertainty, predeclared gates, and explicit negative-result preservation.
- **Ensemble research:** seed, context, geometry, and multi-split diversity evaluated for complementarity rather than standalone score alone.
- **GPU engineering:** bounded single-L40S studies, measured data-loader and microbatch behavior, exact numerical parity checks, and a validated 4.784× inference speedup.
- **Production-style research engineering:** immutable receipts, hash verification, restartable checkpoints, resumable workflows, and separate AWS/GitHub/Kaggle responsibilities.

## Public reproducibility boundary

This repository is intentionally **semi-reproducible**. It publishes aggregate metrics, validation rules, selected protocols, decision history, executed aggregate notebooks, and privacy-safe source snapshots. It does **not** publish raw competition data, fitted weights, private checkpoint locations, credentials, or unreleased feature transforms that provide competition-specific edge.

That boundary keeps the project technically reviewable without turning the public repository into a one-command clone of the private competition system.

## Evidence map

- [`notebooks/05_multisplit_score_and_acceleration.ipynb`](notebooks/05_multisplit_score_and_acceleration.ipynb): executed aggregate notebook for the latest private-score and inference-engineering milestone.
- [`docs/CURRENT_RESEARCH_STATUS.md`](docs/CURRENT_RESEARCH_STATUS.md): current score state, research decisions, limitations, and next direction.
- [`docs/results/post_pr37_progress.json`](docs/results/post_pr37_progress.json): machine-readable public snapshot of the latest progress.
- [`docs/results/frontier_submission.json`](docs/results/frontier_submission.json): competition-facing score lineage.
- [`research/RECENT_MODELS.ipynb`](research/RECENT_MODELS.ipynb): earlier reproduced-model and frontier evidence.

## Validation and limitations

The official coordinate metric is `sqrt(sum(dx^2 + dy^2) / (2N))`. Pooled OOF uses row-weighted squared errors, not an unweighted average of fold RMSEs. Repeated development-fold inspection limits independence, and local OOF must never be presented as a private-leaderboard score.
