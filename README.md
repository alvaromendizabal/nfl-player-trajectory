# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with temporal convolutions, cross-player interaction, controlled supervision, ensemble diversity, and reproducible AWS research.**

[Latest executed study](notebooks/04_observed_context_supervision.ipynb) · [Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [Recent model review](research/RECENT_MODELS.ipynb) · [Run and reproduce](START_HERE.md) · [Model card](docs/MODEL_CARD.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. This repository is the curated public research record; AWS/SageMaker remains the canonical workspace for private data, fitted weights, checkpoints, and full experiment state.

## Current research snapshot

The strongest recorded late post-competition private submission is **0.46547 coordinate RMSE**. The published first-place private score is **0.46340**, leaving a **0.00207 RMSE gap**. No official competition rank is claimed for late submissions.

The strongest completed local system is a **20-model, four-split-family equal-weight ensemble** at **0.4631723213 OOF RMSE over 561,607 rows**. That local OOF number is not directly comparable to the private leaderboard.

The latest controlled study tested whether real pre-pass trajectories withheld by earlier-origin augmentation could provide extra context-player supervision without adding parameters or external data. Fold 0 passed both the reference and matched-control gates, but Fold 1 failed confirmation, so the family was **not promoted**.

| Latest observed-context evidence | Fold 0 | Fold 1 |
|---|---:|---:|
| Multisplit-20 reference RMSE | 0.453726 | 0.475846 |
| Fixed 80/20 context-supervision blend | **0.452149** | 0.476089 |
| Gain vs reference | **+0.001577** | **-0.000243** |
| Matched-control incremental gain | **+0.000893** | +0.000093 |
| Promotion outcome | Discovery passed | **Confirmation failed** |

The result is scientifically useful because it separates a promising first-fold effect from a cross-fold confirmation failure instead of promoting a favorable slice retrospectively.

## What the project demonstrates

- **Sequence modeling:** temporal convolutions, player interaction, Gaussian trajectory uncertainty, motion auxiliaries, EMA inference, and controlled augmentation.
- **Validation discipline:** grouped-game folds, exact coordinate RMSE, paired-game bootstrap intervals, predeclared promotion gates, and explicit negative-result preservation.
- **Ensemble research:** seed, geometry, context, multi-split, and heterogeneous-family experiments evaluated for complementarity rather than standalone score alone.
- **GPU engineering:** measured microbatch and loader selection on a single NVIDIA L40S, BF16/TF32 execution, persistent workers, pinned transfers, and checkpointed/resumable studies.
- **Production-style research engineering:** immutable receipts, byte/hash verification, fail-closed provenance, restartable checkpoints, compact return bundles, and separate AWS/GitHub responsibilities.

The latest study completed **4 fits / 8,416 optimizer updates in 1,095.7 seconds** on `ml.g6e.2xlarge`, with **220 delivery tests passing**. The measured fastest microbatch was 256 at roughly **5.6k samples/s** in the optimizer benchmark; six workers reduced end-to-end loader/optimizer step time to about **0.086 s** in the reported Fold-0 plan.

## Leading-solution reproduction boundary

This project independently recreates mechanisms from leading 2026 public work rather than copying trained weights or feature files. Substantially covered areas include the first-place-style temporal/player-interaction backbone, grouped folds, EMA, motion objectives, earlier-origin augmentation, and multi-split ensemble diversity. Third- and fifth-place-inspired mechanisms have been tested in controlled adaptations, with several families rejected by locked gates.

Important gaps remain: full feature-configuration diversity, larger heterogeneous ensembles justified by residual complementarity, and more complete standalone reproductions of some medal-solution training recipes. The default data boundary remains competition data only.

## Evidence map

- [`notebooks/04_observed_context_supervision.ipynb`](notebooks/04_observed_context_supervision.ipynb): executed aggregate-only notebook for the latest matched study.
- [`docs/CURRENT_RESEARCH_STATUS.md`](docs/CURRENT_RESEARCH_STATUS.md): current score state, experiment ledger, GPU findings, and roadmap.
- [`docs/results/observed_context_supervision.json`](docs/results/observed_context_supervision.json): machine-readable latest result.
- [`research/RECENT_MODELS.ipynb`](research/RECENT_MODELS.ipynb): prior reproduced-model and private-score progression.
- [`research/RESEARCH_REVIEW.ipynb`](research/RESEARCH_REVIEW.ipynb): earlier feature-engineering work.

## Validation and limitations

The official coordinate metric is `sqrt(sum(dx^2 + dy^2) / (2N))`. Pooled OOF uses row-weighted squared errors, not an unweighted average of fold RMSEs. Repeated development-fold inspection limits independence, and local OOF must never be presented as a private-leaderboard score.

AWS and GitHub intentionally differ: raw competition data, private checkpoints, large caches, and credentials remain out of public Git history. Public artifacts contain aggregate evidence, selected source/research documentation, executed notebooks, and privacy-safe figures.
