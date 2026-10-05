# Current research status — October 2026

## Current measured system

- Strongest recorded private submission: **0.46487 coordinate RMSE**
- Strongest completed local system: **0.4631723213 pooled OOF RMSE**
- Pooled OOF population: **561,607 scored rows across 272 games**
- Accepted ensemble: **20 models across four grouped split families**
- Canonical research environment: **AWS SageMaker / 1× NVIDIA L4**

Local OOF, individual-fold validation, and private submission evidence are intentionally kept separate.

## Current active milestone

The current active challenger is a **fresh source-family model trained on a new grouped split family**.

The pilot uses split seed 4 / fold 0 and is predeclared for 35 epochs. It has completed **28/35 epochs**.

Current public-safe evidence:

- best standalone pilot checkpoint: **0.46597135 RMSE**
- exact-population four-family incumbent: **0.45634058 RMSE**
- fixed 80% incumbent / 20% pilot blend: **0.45655750 RMSE**
- current blend delta: **−0.00021691 RMSE** relative to the incumbent
- seven prespecified epochs remain
- **no promotion is claimed**

The pilot passed its earlier continuation gates, but its current best blend remains slightly worse than the incumbent. Finishing the prespecified program is a controlled completion step, not a post-hoc extension.

## Strongest recent system evidence

### Full-OOF diversity audit

The accepted 20-model prediction bank was replayed across all 561,607 rows.

Expanding from two to four split families produced:

- OOF gain: **0.00217835 RMSE**
- paired-game 95% interval: **[0.000629, 0.003861]**
- original folds improved: **5/5**
- leave-one-game-out direction: positive for **272/272** game removals

This is the strongest recent evidence for system-level scaling and motivated the current fresh-split pilot.

The audit also produced a conditional four-to-eight-family scenario, but that value is treated as an assumption-dependent planning calculation rather than a trained result.

## Research completed since the previous public milestone

| Research direction | Public-safe evidence | Decision |
|---|---|---|
| player-specific PBP history | direct-data coverage and model gradients validated; exact candidate failed locked midpoint promotion | Retired exact recipe |
| early joint temporal/player history | controlled joint-history treatments underperformed the short-history control | Retired |
| source-anchored longer history | frozen parent reproduced to tight numerical tolerance; longer-history treatments did not beat control | Retired |
| trajectory-memory retrieval | three retrieval memories and two controls completed; best gain was small and statistically uncertain | Retired |
| full-OOF diversity audit | 20-model OOF replay and robustness analysis strongly supported bounded expansion testing | Validated system evidence |
| fresh split-4 source-family pilot | 28/35 epochs complete; no blend improvement yet | Experimental |

Earlier October studies also covered fresh source configurations, temporal objectives, dense correspondence, dropout, architecture/optimizer transfer, explicit physics, residual heads, NFL NGS priors, and ESPN prior-game context.

## What transferred

### 1. Split/model diversity is the strongest system-level mechanism

The 20-model, four-split-family ensemble remains the strongest completed local system. The full-OOF audit shows that its diversity benefit is broad across folds and games.

### 2. Standalone quality is not sufficient

Several candidates improved a matched source-family checkpoint but remained too residual-correlated to improve the accepted ensemble.

This distinction now drives experiment design: new work is evaluated for **complementarity**, not only component RMSE.

### 3. Direct historical context is useful

Direct NFL Next Gen Stats and direct ESPN historical data both produced measurable signal.

The exact tested integrations did not earn promotion, but the reusable direct-source acquisition, identity, provenance, and point-in-time infrastructure remain valuable project assets.

### 4. Negative experiments are durable evidence

Exact rejected configurations are retired rather than rescued with small hidden-size, learning-rate, epoch, or blend-weight changes.

That reduces repeated compute and makes the research ledger cumulative.

## Direct-source external-data assets

### NFL Next Gen Stats

- **3,920 rows**
- **691 historical players**
- **401 competition players bridged**
- passer prior coverage: **95.74%**
- targeted-receiver prior coverage: **85.35%**

### ESPN historical context

- **36/36 weekly scoreboards**
- **544/544 game summaries**
- **272/272 competition games mapped**
- **100% play-team mapping**
- player-prior coverage: **97.15%**
- dual team-PBP prior coverage: **100%**

For a 2023 competition play in week `w`, 2023 historical priors may only use observations from weeks `< w`.

## Validation discipline

Candidate promotion is evidence-based.

The research protocol uses:

- game-grouped validation
- exact coordinate RMSE
- fixed, predeclared blend weights
- paired whole-game bootstrap uncertainty
- screen / midpoint / final spending gates
- standalone-versus-ensemble attribution
- full-OOF confirmation before scaling
- negative-result retirement
- point-in-time external-data construction
- immutable checkpoint/source lineage

## Engineering state

The AWS research framework supports:

- bounded, resumable executions
- model / optimizer / EMA / scaler / RNG checkpoint recovery
- collision-resistant run IDs
- structured JSONL and human-readable logs
- CPU/RAM/GPU/disk telemetry
- cost accumulation and explicit runtime ceilings
- workload-specific loader/thread benchmarks
- fail-closed package, source, schema, checkpoint, and metric integrity checks
- executed-notebook and Plotly persistence gates
- one artifact / one command / one outer return bundle

Measured fixed-ensemble inference acceleration remains **4.784×** through shared preparation with exact prediction parity on the declared benchmark.

## Public/private boundary

GitHub publishes selected implementation, aggregate metrics, validation logic, system architecture, public-safe notebooks, research decisions, provenance patterns, and machine-readable snapshots.

AWS retains raw competition data, raw third-party response archives, fitted weights, large checkpoints, exact private cloud locations, complete private runners, and unreleased active feature combinations.
