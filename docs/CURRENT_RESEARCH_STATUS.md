# Current research status - October 2026

## Current accepted system

- Strongest recorded private submission: **0.46487 coordinate RMSE**
- Strongest completed local system: **0.4631723213 pooled OOF RMSE**
- Pooled OOF population: **561,607 scored rows across 272 games**
- Accepted ensemble: **20 models across four grouped split families**
- Canonical research environment: **AWS SageMaker / 1x NVIDIA L4**

Local OOF, individual-fold validation, confirmation-stage results, and private-submission evidence are deliberately reported as different evaluation settings.

## Active milestone: separate-fold confirmation

The current challenger is a **feature-configuration portfolio** built from the same core temporal architecture but trained as multiple full models with complementary input representations.

The exact active feature recipes remain private. Public evidence focuses on the scientific design, aggregate metrics, and validation protocol.

### Development-fold result

The maturity-matched study trained one native control and four feature-configuration models through the same 35-epoch endpoint.

Across **109,144 validation rows from 55 games**:

| Comparison | RMSE |
|---|---:|
| Existing accepted ensemble | **0.4537257476** |
| Existing ensemble + native-control blend | **0.4547125988** |
| Existing ensemble + four-model feature portfolio | **0.4484976512** |

Measured improvements:

- versus existing ensemble: **0.0052280964 RMSE**
- versus native-control blend: **0.0062149476 RMSE**
- adjusted whole-game interval versus incumbent: **[0.0002693, 0.0099451]**
- adjusted whole-game interval versus control: **[0.0013023, 0.0114739]**
- games improved: **38/55**
- direction after every leave-one-game-out removal: **55/55 positive**
- displayed horizon bands improved: **all**
- scored player roles improved: **both**

The challenger therefore reached **READY_FOR_SEPARATE_FOLD_CONFIRMATION**.

It is **not promoted**. The accepted system remains the 20-model multisplit ensemble.

### Confirmation protocol

The next validation stage uses a different grouped fold:

- validation games: **54**
- scored rows: **112,694**
- fresh initialization for all five confirmation models
- no discovery-fold trained tensors reused
- same 35-epoch maturity endpoint
- same fixed portfolio construction
- same grouped uncertainty logic

A confirmation failure cannot be rescued by pooling it with the discovery fold.

## Why the protocol changed

An earlier feature-portfolio attempt was stopped at an immature checkpoint. A subsequent audit showed that a historical native model had also looked weak at the same training age before improving materially later.

Rather than retroactively changing that earlier decision, the project preserved the original rejection and created a new, explicit maturity-matched protocol.

This is an important research-systems lesson: **spending gates must be calibrated against the learning dynamics of the model family they govern**.

## Strongest validated system evidence

### Full-OOF diversity audit

The accepted 20-model prediction bank was replayed across all **561,607 rows**.

Expanding from two to four split families produced:

- OOF gain: **0.00217835 RMSE**
- paired-game 95% interval: **[0.000629, 0.003861]**
- original folds improved: **5/5**
- leave-one-game-out direction: positive for **272/272** game removals

This remains the strongest fully completed system-level evidence and is why the feature-diversity challenger is being evaluated as an additional diversity axis rather than a replacement for grouped split diversity.

## Recent controlled research

| Direction | Public-safe conclusion | State |
|---|---|---|
| additional unmodified split family | fresh family did not improve the accepted ensemble | Retired exact path |
| probabilistic trajectory mixtures | mixture treatments did not establish incremental value over matched controls | Retired exact path |
| capability/error audit | identified concentrated long-horizon defensive error and reusable analysis infrastructure | Completed |
| predicted-future interaction | joint interaction did not beat matched self-only refinements | Retired exact path |
| early feature-portfolio screen | stopped at the original gate | Historical rejection preserved |
| maturity audit | demonstrated that the old early screen could reject a later-useful control | Completed |
| maturity-matched feature portfolio | passed development-fold effect-size, uncertainty, and robustness gates | Confirmation stage |

Earlier October work also covered direct NFL/ESPN context, player history, longer histories, retrieval, dense correspondence, dropout, optimizer/architecture transfers, physics-inspired features, residual heads, and multiple objective variants.

## Direct-source historical-data assets

### NFL Next Gen Stats

- **3,920 rows**
- **691 historical players**
- **401 competition players bridged**
- passer-prior coverage: **95.74%**
- targeted-receiver prior coverage: **85.35%**

### ESPN historical context

- **36/36 weekly scoreboards**
- **544/544 game summaries**
- **272/272 competition games mapped**
- **100% play-team mapping**
- player-prior coverage: **97.15%**
- dual team-PBP prior coverage: **100%**

For a 2023 competition play in week `w`, 2023 historical priors may only use observations from weeks `< w`.

Historical frame-level tracking transfer remains a separate sourcing/provenance problem and is not claimed as implemented.

## Validation discipline

Candidate promotion uses:

- game-grouped validation
- exact coordinate RMSE
- fixed, predeclared ensemble weights
- paired whole-game bootstrap uncertainty
- matched controls
- leave-one-game-out robustness
- role/horizon accounting with row and SSE conservation
- confirmation on fresh grouped folds
- full pooled OOF before system promotion
- negative-result retirement
- point-in-time external-data construction
- immutable checkpoint/source lineage

## Engineering state

The AWS research framework supports:

- resumable model / optimizer / EMA / scaler / RNG checkpoints
- exact batch-cursor recovery where required
- collision-resistant run IDs
- structured JSONL and human-readable logs
- CPU/RAM/GPU/disk telemetry
- cost accounting and runtime estimates
- workload-specific loader/thread benchmarks
- fail-closed package, source, schema, checkpoint, and metric integrity checks
- executed-notebook and Plotly persistence gates
- one artifact / one command / one outer return bundle
- deterministic regression tests for avoidable execution failures

Measured fixed-ensemble inference acceleration remains **4.784x** through shared preparation with exact prediction parity on the declared benchmark.

## Public/private boundary

GitHub publishes selected implementation, aggregate metrics, validation logic, public-safe protocols, system architecture, notebooks, research decisions, provenance patterns, and machine-readable snapshots.

AWS retains raw competition data, raw third-party response archives, fitted weights, large checkpoints, private cloud locations, complete private runners, row-level private predictions, and the exact active feature recipes.
