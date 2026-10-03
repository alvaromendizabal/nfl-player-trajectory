# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with sequence modeling, ensemble diversity, direct-source external data, GPU engineering, and reproducible AWS research.**

[Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [October research review](docs/OCTOBER_RESEARCH_PROGRESS.md) · [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md) · [Model card](docs/MODEL_CARD.md) · [Run and reproduce](START_HERE.md)

Predict selected NFL players' future x/y locations after a pass using observed tracking, player roles, organizer-supplied landing context, and forecast horizon. AWS/SageMaker is the canonical research workspace; Kaggle is reserved for submission delivery.

## Current research snapshot

The strongest recorded private submission is **0.46487 coordinate RMSE**. The strongest completed local system is a **20-model, four-split-family ensemble** at **0.4631723213 pooled OOF RMSE over 561,607 rows**.

Those two measurements serve different purposes: private-submission evidence tracks delivered performance, while grouped-game OOF is the primary local research instrument.

The project has evolved into a full ML research system rather than a single model:

- **20-model ensemble diversity** across repeated grouped splits
- **5-fold sequence-model research** with locked promotion gates
- **direct NFL Next Gen Stats acquisition and identity bridging**
- **direct ESPN schedule, game-summary, and play-by-play acquisition**
- **point-in-time historical feature stores** with strict prior-week leakage controls
- **resumable GPU runners** with checkpointing, telemetry, cost accounting, and fail-closed integrity checks
- **negative-result retention** so unsuccessful hypotheses reduce future search cost instead of being repeated

## Recent research program

Since the previous public milestone, the project has completed a large controlled sequence of representation, objective, architecture, optimization, physics, and external-context studies.

The most useful conclusions are:

1. **Split/model diversity remains the strongest system-level mechanism.**
2. **Several candidate families improved standalone Fold-0 quality without adding enough independent residual signal to the ensemble.**
3. **Zero dropout and dense temporal supervision improved source-family fitting but remained strongly correlated with the incumbent.**
4. **Direct NFL Next Gen Stats materially improved standalone fitting.**
5. **Direct ESPN prior-game context produced more promising ensemble complementarity than static external priors alone.**
6. **Fine-grained player-specific prior-game context is the current active research direction.**
7. **Cross-fold confirmation and paired-game uncertainty gates prevent attractive one-fold results from being promoted prematurely.**

The detailed public-safe record is in [docs/OCTOBER_RESEARCH_PROGRESS.md](docs/OCTOBER_RESEARCH_PROGRESS.md).

## External data engineering

The external-data pipeline is built from **self-pulled public sources**, not competitor-prepared datasets.

Current reusable assets include:

- **3,920 direct NFL Next Gen Stats rows**
- **691 NGS players**
- **401 competition players bridged**
- **95.74% passer prior coverage**
- **85.35% targeted-receiver prior coverage**
- **36/36 ESPN weekly scoreboards**
- **544/544 ESPN game summaries**
- **272/272 competition games mapped**
- **100% play-team mapping**
- **97.15% ESPN player-prior coverage**
- **100% dual team-PBP prior coverage**

All 2023 historical priors are point-in-time: a play in week `w` may only use historical observations from weeks `< w`.

See [docs/EXTERNAL_DATA_ENGINEERING.md](docs/EXTERNAL_DATA_ENGINEERING.md).

## GPU and systems engineering

The research stack runs on AWS SageMaker with a single NVIDIA L4.

Selected measured engineering results:

- **4.784× inference acceleration** for the fixed 20-model ensemble through shared preparation
- exact prediction parity on the declared inference benchmark
- workload-specific `DataLoader` benchmarking rather than fixed worker assumptions
- checkpointed epoch/stage recovery
- structured JSONL and human-readable logs
- CPU/RAM/GPU/disk telemetry
- run-level cost accounting
- device-safe EMA restoration
- AMP overflow handling through `GradScaler`
- collision-resistant run IDs and immutable manifests
- one-artifact / one-command / one-return-bundle execution

## Scientific operating discipline

A candidate is not promoted because it looks good on one fold.

The project uses:

- game-grouped validation
- fixed predeclared blend weights
- paired-game bootstrap uncertainty
- explicit screen / midpoint / final gates
- separate treatment of local OOF and submission evidence
- negative-result retirement
- point-in-time external-data construction
- immutable raw-source provenance and hashes

This keeps the repository focused on defensible ML engineering rather than leaderboard storytelling.

## Public reproducibility boundary

This repository is intentionally **semi-reproducible**.

It publishes aggregate metrics, validation rules, selected protocols, public-safe data-engineering patterns, decision history, notebooks, and machine-readable public evidence.

It intentionally does **not** publish raw competition data, fitted private weights, large checkpoints, private cloud object locations, credentials, complete private runners, unreleased competition-specific transforms, or the exact feature combinations that constitute active competitive IP.

## Repository tour

- [START_HERE.md](START_HERE.md) — review and reproducibility entry point
- [docs/CURRENT_RESEARCH_STATUS.md](docs/CURRENT_RESEARCH_STATUS.md) — latest measured state
- [docs/OCTOBER_RESEARCH_PROGRESS.md](docs/OCTOBER_RESEARCH_PROGRESS.md) — controlled research program since the last publication
- [docs/EXTERNAL_DATA_ENGINEERING.md](docs/EXTERNAL_DATA_ENGINEERING.md) — direct-source acquisition, identity resolution, and leakage controls
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — system evidence, limitations, and deployment boundary
- [research/README.md](research/README.md) — research-evidence hierarchy
- [src/nfl_trajectory/](src/nfl_trajectory/) — maintained public implementation
- [tests/](tests/) — software and research-contract tests

## Validation and limitations

The official coordinate metric is `sqrt(mean((prediction_xy - target_xy)^2))` over all scored x/y coordinates.

Development folds, pooled OOF, and private submission measurements are kept distinct. The current public dataset covers one competition season, so cross-season generalization remains a research limitation. External priors are historical and point-in-time, but public coverage varies by player and source.

The project is a research artifact, not a certified player-evaluation or production decision system.
