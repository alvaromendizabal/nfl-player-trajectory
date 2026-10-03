# Current research status — October 2026

## Current measured system

- Strongest recorded private submission: **0.46487 coordinate RMSE**
- Strongest completed local system: **0.4631723213 pooled OOF RMSE**
- Pooled OOF population: **561,607 rows**
- Ensemble size: **20 models across four grouped split families**
- Canonical research environment: **AWS SageMaker / 1× NVIDIA L4**

Local OOF and private submission evidence are intentionally treated as separate measurements.

## Research completed since the previous public milestone

The project completed a broad controlled program across representation, supervision, architecture, optimization, physics, and external historical context.

| Research family | Public-safe evidence | Decision |
|---|---|---|
| Fresh source-faithful configurations | One configuration survived early screening but failed the locked midpoint ensemble gate | Retired exact branch |
| Alternative temporal objectives | Temporal Huber rejected early; late-horizon objective failed midpoint complementarity | Retired |
| Dense temporal correspondence | Improved standalone source-family fit by ~0.00097 RMSE but did not improve the fixed ensemble | Retired exact branch |
| Zero-dropout source model | Improved standalone source-family fit by ~0.00183 RMSE; ensemble gain remained uncertain | Retired exact branch |
| Wide/shallow interaction transfer | Did not outperform the matched source-family control | Retired |
| Layer-wise AdamW transfer | Weak at early screen | Retired |
| Explicit physics integration | Survived the screen but did not pass the midpoint ensemble gate | Retired |
| Residual physics/context heads | Failed early quality screens | Retired |
| Direct NFL NGS priors | Strong standalone improvement; exact static-injection recipe remained too correlated | Retired exact recipe |
| Direct ESPN team/PBP priors | Produced the most promising recent fixed-blend gain but uncertainty still crossed zero | Retired exact recipe |
| Player-specific prior-game PBP | Packaged and tested; next active research milestone | Prepared |

## What transferred

### 1. Ensemble diversity remains the strongest system-level result

The 20-model, four-split-family ensemble remains the strongest completed local system. Repeated evidence shows that independent error diversity matters more than isolated one-fold gains.

### 2. Standalone quality is not enough

Several candidates were better than the matched source-family model on Fold 0 but still failed to add enough independent residual signal to the ensemble.

That distinction is now a central design constraint: new work is prioritized for **complementarity**, not merely component RMSE.

### 3. External historical context is useful

Direct NFL Next Gen Stats and direct ESPN historical data both added measurable signal.

The strongest external-data findings so far are:

- NGS historical priors improved standalone source-family fitting
- team-level ESPN PBP context produced more promising ensemble complementarity
- fine-grained player-specific history is the next logical representation

### 4. Point-in-time engineering is now first-class infrastructure

For a competition play in week `w`, 2023 priors only use observations from weeks `< w`.

The pipeline preserves raw-source provenance, hashes, coverage reports, identity bridges, and reusable point-in-time feature stores.

## Direct-source external-data assets

### NFL Next Gen Stats

- **3,920 rows**
- **691 NGS players**
- **401 competition players bridged**
- passer prior coverage: **95.74%**
- targeted-receiver prior coverage: **85.35%**

### ESPN historical context

- **36/36 weekly scoreboards acquired**
- **544/544 game summaries acquired**
- **272/272 competition games mapped**
- **100% play-team mapping**
- player-prior coverage: **97.15%**
- dual team-PBP prior coverage: **100%**

These assets are cached in AWS and are reused rather than redownloaded for every experiment.

## Validation discipline

Candidate promotion requires more than one attractive fold result.

The research protocol uses game-grouped validation, fixed predeclared blend weights, paired-game bootstrap uncertainty, early screen / midpoint / final gates, negative-result retirement, explicit checkpoint lineage, point-in-time external-data construction, and local OOF / private submission separation.

## Current active direction

The next prepared experiment builds **player-specific prior-game PBP tendencies** from the already acquired ESPN corpus.

Two controlled arms test each player's own prior-week tendencies and the same player-specific priors plus passer/target context broadcast to the play.

The milestone is prepared and tested but remains unmeasured until AWS execution completes.

## Engineering state

The AWS runner framework now supports bounded 20–30 minute child executions, checkpointed recovery, collision-resistant run IDs, structured JSONL + human-readable logs, CPU/RAM/GPU/disk telemetry, cost accumulation, workload-specific worker benchmarks, fail-closed package integrity, executed-notebook and Plotly persistence gates, and one artifact / one command / one outer return bundle.

## Public/private boundary

GitHub publishes aggregate metrics, validation logic, selected engineering patterns, notebooks, decision history, and machine-readable snapshots.

AWS retains raw competition data, fitted weights, large checkpoints, exact private object locations, complete private runners, and unreleased competitive feature combinations.
