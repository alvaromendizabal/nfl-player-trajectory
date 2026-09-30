# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with controlled sequence modeling, ensemble diversity, GPU engineering, and reproducible AWS research.**

[Latest aggregate notebook](notebooks/06_architecture_research_progress.ipynb) · [Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [Model card](docs/MODEL_CARD.md) · [Recent model review](research/RECENT_MODELS.ipynb) · [Run and reproduce](START_HERE.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. Competition slug: `nfl-big-data-bowl-2026-prediction`. AWS/SageMaker is the canonical research workspace; Kaggle is used only for the required submission surface.

## Current research snapshot

The strongest recorded late post-competition private submission remains **0.46487 coordinate RMSE**. The published first-place private comparator is **0.46340**, leaving a **0.00147 RMSE gap**. No official competition rank is claimed for late submissions.

The strongest completed local system remains the **20-model, four-split-family equal-weight ensemble** at **0.4631723213 OOF RMSE over 561,607 rows**. Local OOF and private leaderboard measurements are intentionally kept separate.

Since the previous public snapshot, two source-recipe neural candidates completed controlled Fold-0 studies:

| Candidate family | Standalone RMSE | Fixed 80/20 blend RMSE | Gain vs reference | Decision |
|---|---:|---:|---:|---|
| Expanded observed-motion representation | 0.460633 | **0.452585** | +0.001140 | No promotion |
| Future-conditioned delta decoder | **0.459976** | 0.452981 | +0.000745 | No promotion |

Both candidates improved the fixed **0.453726** Fold-0 ensemble reference, but neither cleared the predeclared **0.0015 RMSE** promotion threshold and both paired-game confidence intervals crossed zero. No post-hoc blend search was used.

## What changed technically

- **Training-contract recovery:** new studies returned to the verified successful recipe instead of continuing a drifted high-throughput control.
- **Representation research:** an observed-motion expansion produced useful but insufficient ensemble complementarity.
- **Architecture research:** a future-conditioned incremental-motion decoder improved standalone quality but did not add enough robust ensemble gain.
- **GPU engineering:** on the current NVIDIA L4, loader benchmarking reduced median full training-step time from **0.1102 s to 0.0486 s** while holding the scientific training contract fixed.
- **Research discipline:** negative results are preserved and retired rather than tuned retrospectively on the inspected development fold.

The next prepared branch studies **target-specific sparse interaction** at a high level; it is intentionally labeled unmeasured until a real AWS result exists.

## Public reproducibility boundary

This repository is intentionally **semi-reproducible**. It publishes aggregate metrics, validation rules, selected protocols, decision history, executed aggregate notebooks, and privacy-safe implementation patterns.

It does **not** publish raw competition data, fitted weights, private checkpoint locations, credentials, complete private experiment runners, or unreleased feature/interaction transforms that provide competition-specific edge.

That boundary keeps the project technically reviewable without turning the public repository into a one-command clone of the private competition system.

## Evidence map

- [`notebooks/06_architecture_research_progress.ipynb`](notebooks/06_architecture_research_progress.ipynb): executed aggregate notebook for the two completed post-PR38 neural studies and current GPU benchmark.
- [`docs/CURRENT_RESEARCH_STATUS.md`](docs/CURRENT_RESEARCH_STATUS.md): current score state, controlled results, limitations, and next direction.
- [`docs/results/post_pr38_architecture_progress.json`](docs/results/post_pr38_architecture_progress.json): machine-readable aggregate snapshot.
- [`docs/results/frontier_submission.json`](docs/results/frontier_submission.json): competition-facing score lineage.
- [`research/RECENT_MODELS.ipynb`](research/RECENT_MODELS.ipynb): earlier reproduced-model and frontier evidence.

## Validation and limitations

The official coordinate metric is `sqrt(sum(dx^2 + dy^2) / (2N))`. Pooled OOF uses row-weighted squared errors, not an unweighted average of fold RMSEs. The latest architecture studies use an inspected development fold and are not private-leaderboard scores or untouched test estimates.
