# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with temporal convolutions, cross-player attention, game-grouped validation, and reproducible cloud research.**

[Latest model review](research/RECENT_MODELS.ipynb) · [Research guide](research/README.md) · [Run and reproduce](START_HERE.md) · [Model card](docs/MODEL_CARD.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. This repository focuses on the **Prediction** task.

## Latest research snapshot

The current scored system is a **seven-model equal ensemble**: five source-faithful game-grouped base models plus one independently initialized seed-1 model and one protected context-dropout model. It improved the prior five-base private score from **0.46615 to 0.46547 coordinate RMSE**. The final first-place private score is **0.46340**, leaving a verified private-score gap of **0.00207 (0.447%)**. Because these are post-competition research submissions, no official competition rank is claimed.

| Evidence | Coordinate RMSE / result | Meaning |
|---|---:|---|
| **Five-fold base OOF** | **0.468143852** | 561,607 retained rows; every row predicted only by its excluded-fold model. |
| Seed-1 Fold-0 fixed 50/50 blend | 0.458822850 → **0.453926574** | Discovery gain 0.004896276; paired-game interval entirely positive. |
| Seed-1 Fold-1 fixed confirmation | 0.481571182 → **0.478239278** | Point gain 0.003331904, but interval crossed zero; recipe not promoted by itself. |
| Protected context dropout, Fold 0 | 0.458822850 → **0.455515340** | Point gain 0.003307510, but interval crossed zero; model retained only for diversity. |
| Five-base Kaggle private | **0.46615** | Submission `56470102`. |
| **Seven-model Kaggle private** | **0.46547** | Submission `56478271`; improved 0.00068 versus the five-base system. |
| Final first-place private | **0.46340** | Comparable private-score target; current gap 0.00207. |

The [executed model review](research/RECENT_MODELS.ipynb) shows fold variability, fixed-replication uncertainty, diversity experiments, horizon error concentration, and the private-score progression with inline Plotly figures and static fallbacks.

![Preserved base-model fold variability](research/figures/winner_folds.png)

## Research progression

1. **Reproduced base family.** Five source-faithful models produced 0.468143852 pooled OOF RMSE over 561,607 excluded-fold rows.
2. **Rejected unstable variants.** ROT geometry, mirror inference, seed diversity, and context dropout each produced useful evidence, but predeclared confirmation or robustness gates prevented standalone promotion where uncertainty remained.
3. **Used complementarity scientifically.** The seed-1 and context-dropout models were retained as diverse ensemble members rather than mislabeled as standalone wins.
4. **Verified leaderboard transfer.** The fixed seven-model equal ensemble improved the private score from 0.46615 to 0.46547.
5. **Next capability: alternate CV diversity.** A second complete game-grouped split family is prepared as the next experiment. It is **pending** and no metric is claimed yet.

## What the work demonstrates

- **Modeling:** temporal convolutions, cross-player attention, relative-motion features, EMA inference, training-time context masking, and controlled model diversity.
- **Statistics:** grouped OOF evaluation, row-weighted coordinate RMSE, paired-game bootstrap intervals, fixed replication protocols, and explicit promotion gates.
- **Engineering:** byte-pinned checkpoints, schema-aware provenance, restartable S3 receipts, label-free inference, duplicate-safe Kaggle submission journals, and exact-version scoring.
- **Research judgment:** negative experiments remain visible; local metrics are never presented as leaderboard scores; private-score improvement is reported only after exact submission readback.

## Validation and limitations

The official metric is `sqrt(sum(dx² + dy²) / (2N))`, in yards. Pooled OOF is computed from row-weighted squared errors, not an unweighted mean of fold RMSEs. Checkpoint selection and repeated inspection limit independence, and the complete first-place 100+ model diversity system has not been reproduced. The seed/context studies use previously inspected folds, so their point gains are exploratory until they transfer through the full deployment path.

The seven-model private improvement is real but modest: **0.00068 RMSE**. The project therefore remains below the strongest comparable private score by **0.00207**. The next alternate-CV experiment targets the largest still-missing ensemble-diversity mechanism rather than repeating ordinary seeds.

## Reproducibility, privacy, and attribution

**AWS and GitHub have different roles.** AWS retains competition data, trained weights, private execution state, and large checkpoints. GitHub contains selected source, aggregate evidence, executed notebooks, and documentation; it is not an AWS mirror. Raw competition data, credentials, private checkpoints, and cloud-only artifacts are not redistributed.

The deep-model reproduction follows the public [chack3 training reference](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train), with provenance and limitations documented in the notebook. Competition data remains subject to its own terms. See the [data card](docs/DATA_CARD.md), [sources](docs/SOURCES.md), and [latest scored submission record](docs/results/frontier_submission.json).
