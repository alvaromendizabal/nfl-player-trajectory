# Model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026 Prediction.

**Primary metric:** coordinate RMSE; lower is better.

## Current system — October 2026

The strongest completed deployment candidate remains the **20-model multisplit ensemble** spanning four grouped-CV split families.

| Evidence | Coordinate RMSE |
|---|---:|
| Multisplit local OOF, 561,607 rows | **0.4631723213** |
| Seven-model private submission | 0.46547 |
| Multisplit-20 private submission | **0.46487** |

Local OOF and private submission measurements are not treated as interchangeable.

## Model family

The main neural family combines temporal sequence encoding over observed player motion, player-interaction modeling, static play/player context, Gaussian trajectory supervision, velocity/acceleration auxiliary supervision, exponential moving-average weights, geometric and frame-shift augmentation, and grouped-game cross-validation.

The final ensemble obtains diversity from multiple grouped split families rather than a single training seed.

## Recent controlled research

The latest research program tested fresh configurations, temporal objectives, dense auxiliary supervision, dropout/architecture changes, optimizer transfer, explicit physics integration, residual heads, and external historical context.

The most informative outcomes:

- **dense temporal correspondence** improved standalone source-family quality but remained ensemble-correlated
- **zero dropout** materially improved standalone fitting but did not pass ensemble uncertainty gates
- **direct NFL NGS priors** produced strong standalone external-data gains
- **direct ESPN prior-game PBP context** produced the strongest recent complementarity signal
- **player-specific PBP context** is the next active representation

Exact unsuccessful configurations are retired to avoid repeated search over already answered questions.

## External historical context

The project maintains reusable direct-source external-data infrastructure.

### NFL Next Gen Stats

The point-in-time NGS feature store contains historical passing, receiving, and rushing context acquired directly from NFL public endpoints.

Coverage in the current competition bridge:

- passer: **95.74%**
- targeted receiver: **85.35%**

### ESPN historical game context

The project directly acquired 36 weekly scoreboards, 544 game summaries, complete competition-game mapping, team and player historical priors, and prior-game PBP tendency summaries.

For 2023, a play in week `w` may only use historical information from weeks `< w`.

## Validation and promotion

A candidate is evaluated through early quality screening, fixed-weight blend comparison, paired-game bootstrap uncertainty, confirmation on another grouped fold, and full OOF evaluation before system promotion.

This deliberately favors reproducible, complementary signal over one-fold point improvements.

## GPU / runner engineering

The AWS research workflow uses NVIDIA L4 acceleration with mixed precision, workload-specific data-loader benchmarks, checkpoint recovery, device-safe EMA restoration, structured telemetry, cost accounting, package integrity checks, immutable run manifests, and single outer return bundles.

The previously measured shared-preparation inference path accelerated the fixed 20-model ensemble by **4.784×** with exact prediction parity on its declared benchmark.

## Public reproducibility boundary

This public repository includes selected implementation, aggregate research evidence, validation logic, notebooks, and documentation.

It excludes raw competition data, fitted private weights, large checkpoints, credentials, private cloud paths, full private runners, and unreleased feature combinations that are still active research IP.

## Limitations

- one labelled competition season limits external-validity claims
- player-level external-source coverage varies
- long-horizon errors remain harder than short-horizon errors
- component-level improvements may be redundant with the ensemble
- public artifacts do not reproduce private competition data or trained weights
