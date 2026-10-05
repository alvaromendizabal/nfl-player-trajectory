# NFL Big Data Bowl 2026 — Player Trajectory Prediction

**End-to-end player-motion forecasting research with temporal deep learning, repeated grouped-CV ensembles, direct-source historical context, GPU engineering, and reproducible AWS experimentation.**

[Current status](docs/CURRENT_RESEARCH_STATUS.md) · [Research system](docs/RESEARCH_SYSTEM.md) · [October research review](docs/OCTOBER_RESEARCH_PROGRESS.md) · [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md) · [Model card](docs/MODEL_CARD.md) · [Start here](START_HERE.md)

The task is to forecast selected NFL players' future x/y locations after a pass using observed tracking, player roles, organizer-supplied landing context, and forecast horizon. AWS SageMaker is the canonical research environment; the public repository is the versioned, employer-facing reproducibility layer.

Competition identifier: `nfl-big-data-bowl-2026-prediction`.

## Portfolio snapshot

| Evidence | Result |
|---|---:|
| Strongest recorded private submission | **0.46487 coordinate RMSE** |
| Strongest completed local ensemble | **0.4631723213 pooled OOF RMSE** |
| OOF evaluation population | **561,607 scored rows / 272 games** |
| Accepted ensemble | **20 models / 4 grouped split families** |
| Measured inference acceleration | **4.784×** with exact parity on the declared benchmark |
| Direct NFL NGS acquisition | **3,920 rows / 691 historical players** |
| Direct ESPN acquisition | **36 scoreboards / 544 game summaries / 272 of 272 games mapped** |

Local OOF and private-submission results are intentionally reported as separate evaluation settings.

## What this project demonstrates

This repository represents a full ML research system rather than a single competition notebook:

- **sequence modeling:** temporal convolution, player-interaction attention, Gaussian trajectory supervision, and auxiliary motion objectives
- **robust validation:** game-grouped cross-validation, repeated split families, paired-game bootstrap uncertainty, fixed promotion gates, and leakage-safe OOF analysis
- **ensemble research:** a 20-model multisplit system with explicit prediction/residual complementarity analysis
- **direct-source data engineering:** first-party/neutral historical acquisition, identity bridging, point-in-time feature construction, coverage audits, and immutable provenance receipts
- **GPU engineering:** NVIDIA L4 mixed-precision training, workload-specific loader/thread benchmarking, checkpoint recovery, structured telemetry, and cost accounting
- **research operations:** immutable experiment manifests, champion/challenger states, negative-result retirement, collision-resistant run IDs, and one-artifact/one-command/one-return-bundle execution

## Strongest system-level evidence

The clearest transferable result is **split/model diversity**.

A full-OOF audit of the accepted prediction bank reproduced the 20-model system at **0.4631723213 RMSE** and measured a **0.00217835 RMSE gain** when expanding from two to four split families. The paired-game 95% interval was **[0.000629, 0.003861]**; all five original folds improved, and the direction remained positive under all 272 leave-one-game-out removals.

That evidence motivated a fresh grouped-split source-family pilot rather than another post-hoc blend search.

The active pilot is deliberately labeled **experimental**: it has completed **28 of 35 prespecified epochs**. Its best standalone checkpoint is **0.46597135 RMSE** on its own split-4/fold-0 population; the corresponding fixed blend remains slightly worse than that population's incumbent, so **no promotion is claimed**.

## Recent controlled research

Recent experiments deliberately tested materially different sources of signal rather than repeatedly tuning one recipe.

| Research direction | Public-safe conclusion |
|---|---|
| dense temporal correspondence | improved source-family standalone fit; insufficient ensemble complementarity |
| zero-dropout source family | strong standalone improvement; residuals remained too correlated |
| direct NFL Next Gen Stats priors | useful standalone historical signal; exact injection recipe not promoted |
| direct ESPN prior-game PBP | stronger complementarity signal; uncertainty gate still failed |
| player-specific PBP histories | data/gradient pipeline validated; exact model failed midpoint promotion |
| early joint temporal/player history | controlled treatments underperformed the matched short-history control |
| source-anchored longer history | exact parent replay passed; longer-history treatments did not beat control |
| trajectory-memory retrieval | all three memory variants completed; gains were too small and uncertain |
| full-OOF diversity audit | robust support for a bounded fresh-split expansion pilot |

Negative results remain in the research record so future work does not repeatedly spend compute on answered questions.

## Direct-source historical data

External historical context is accepted only when it can be acquired directly from a first-party or neutral public source and made point-in-time safe.

Current reusable assets include:

- **NFL Next Gen Stats:** 3,920 rows, 691 historical players, 401 competition-player matches
- **NGS coverage:** 95.74% passer priors, 85.35% targeted-receiver priors
- **ESPN:** 36/36 weekly scoreboards and 544/544 game summaries
- **competition bridge:** 272/272 games mapped and 100% play-team mapping
- **historical coverage:** 97.15% player-prior coverage and 100% dual team-PBP coverage

For a 2023 competition play in week `w`, 2023 historical features may only use observations from weeks `< w`.

See [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md).

## Engineering highlights

The canonical research stack runs on AWS SageMaker with a single NVIDIA L4.

Measured and implemented engineering capabilities include:

- **4.784×** shared-preparation inference acceleration with exact parity on the declared benchmark
- resumable model/optimizer/EMA/scaler/RNG checkpointing
- workload-specific `DataLoader` and CPU-thread benchmarks
- FP16/BF16 training where numerically valid and FP32 evaluation where required
- CPU/RAM/GPU/disk utilization heartbeats and peak-memory tracking
- structured JSONL plus human-readable logs
- reconstructable run-cost estimates and explicit cost ceilings
- fail-closed schema, source-hash, checkpoint, metric, and packaging gates
- executed-notebook and Plotly persistence checks
- deterministic regression tests for avoidable execution failures

See [Research system and reproducibility](docs/RESEARCH_SYSTEM.md).

## Scientific operating discipline

A candidate is not promoted because one fold looks attractive.

The project uses:

- game-grouped validation
- fixed, predeclared blend weights
- paired-game bootstrap uncertainty
- explicit screen / midpoint / final gates
- standalone-versus-ensemble attribution
- full-OOF confirmation before scaling
- point-in-time external-data construction
- immutable provenance and checkpoint lineage
- negative-result retirement

This keeps the repository centered on defensible ML research rather than score-chasing.

## Public reproducibility boundary

The repository is intentionally **semi-reproducible**.

**Published:** selected implementation, aggregate metrics, validation contracts, source/provenance patterns, executed aggregate notebooks, tests, machine-readable snapshots, and research decisions.

**Kept private:** raw competition data, raw third-party response archives, fitted weights, large checkpoints, credentials, private cloud object locations, complete private runners, and unreleased active feature combinations.

The public artifacts are sufficient to review the architecture, scientific process, engineering quality, and reproducibility discipline without distributing restricted data or active competitive IP.

## Review path

For a fast technical review:

1. [Current research status](docs/CURRENT_RESEARCH_STATUS.md)
2. [Research system and reproducibility](docs/RESEARCH_SYSTEM.md)
3. [Model card](docs/MODEL_CARD.md)
4. [October research review](docs/OCTOBER_RESEARCH_PROGRESS.md)
5. [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md)
6. [Research evidence archive](research/README.md)
7. [Run and validate the public artifacts](START_HERE.md)

## Repository map

- `src/nfl_trajectory/` — maintained public implementation
- `tests/` — software and research-contract tests
- `notebooks/` — selected project notebooks
- `research/` — public research evidence and source archive
- `docs/` — model, validation, system, and data-engineering documentation

## Metric and limitations

The official metric is coordinate RMSE:

`sqrt(mean((prediction_xy - target_xy)^2))`

Lower is better.

The labeled competition data cover one competition season, so cross-season generalization remains a research limitation. External-source coverage varies by player and source. Component improvements can be redundant inside an ensemble. Development-fold evidence, pooled OOF, and private submission measurements are never treated as interchangeable.

This is a research artifact, not a certified player-evaluation or production decision system.