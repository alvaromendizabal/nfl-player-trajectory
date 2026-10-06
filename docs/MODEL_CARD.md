# Model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026 Prediction.

**Primary metric:** coordinate RMSE; lower is better.

## Current accepted system

The strongest completed local system is the **20-model multisplit ensemble** spanning four grouped-CV split families.

| Evidence | Coordinate RMSE |
|---|---:|
| Multisplit pooled OOF, 561,607 rows | **0.4631723213** |
| Strongest recorded private submission | **0.46487** |

Local OOF and private-submission measurements are not treated as interchangeable.

## Model family

The primary neural family combines:

- temporal sequence encoding over observed player motion
- player-interaction modeling
- static play/player context
- Gaussian trajectory supervision
- velocity/acceleration auxiliary supervision
- exponential moving-average weights
- geometric and frame-shift augmentation
- grouped-game cross-validation

The accepted ensemble obtains most of its validated diversity from **multiple grouped split families**, not from post-hoc learned blend weights.

## Strongest system evidence

A full-OOF audit of the accepted prediction bank reproduced **0.4631723213 RMSE** across **561,607 rows and 272 games**.

Expanding from two to four split families yielded:

- incremental OOF gain: **0.00217835 RMSE**
- paired-game 95% interval: **[0.000629, 0.003861]**
- improved original folds: **5/5**
- positive direction under leave-one-game-out removal: **272/272**

This evidence motivates the current fresh-split pilot.

## Active challenger

The current challenger is a **four-model feature-configuration portfolio** evaluated against a maturity-matched native control.

The core architecture and model size are held constant. The exact active feature recipes remain private; public evidence records the experimental design and aggregate validation outcome.

On the development fold:

- validation population: **109,144 rows / 55 games**
- incumbent RMSE: **0.4537257476**
- native-control blend RMSE: **0.4547125988**
- feature-portfolio blend RMSE: **0.4484976512**
- improvement versus incumbent: **0.0052280964**
- improvement versus control: **0.0062149476**
- adjusted grouped interval versus incumbent: **[0.0002693, 0.0099451]**
- adjusted grouped interval versus control: **[0.0013023, 0.0114739]**
- leave-one-game-out direction: **55/55 positive**

The challenger is **READY_FOR_SEPARATE_FOLD_CONFIRMATION**, not promoted.

The next gate uses a different grouped fold, fresh initialization, and no discovery-fold weight reuse.

## Recent controlled research

The October program tested representation, supervision, architecture, optimization, external historical context, longer histories, and retrieval.

Key conclusions:

- dense correspondence and zero dropout improved source-family standalone fitting but remained ensemble-correlated
- direct NFL NGS historical priors added useful standalone information
- direct ESPN prior-game context produced stronger complementarity than static NGS-only integration
- repaired player-specific PBP history reached the model but failed its locked midpoint gate
- early joint-history and source-anchored longer-history treatments failed matched controls
- trajectory-memory retrieval produced only small, uncertain gains
- repeated grouped-split diversity remains the clearest fully validated system-level mechanism
- maturity-matched full-model feature diversity produced a strong development-fold ensemble gain and is now in separate-fold confirmation

Exact rejected recipes are retired to avoid repeated search over already answered questions.

## External historical context

The project maintains reusable direct-source historical-data infrastructure.

### NFL Next Gen Stats

- historical rows: **3,920**
- historical players: **691**
- competition-player matches: **401**
- passer-prior coverage: **95.74%**
- targeted-receiver-prior coverage: **85.35%**

### ESPN historical game context

- weekly scoreboards: **36/36**
- game summaries: **544/544**
- competition games mapped: **272/272**
- play-team mapping: **100%**
- player-prior coverage: **97.15%**
- dual team-PBP coverage: **100%**

For 2023, a competition play in week `w` may only use eligible historical information from weeks `< w`.

## Validation and promotion

A candidate is evaluated through:

1. correctness/data/integration gates
2. early quality screening
3. fixed-weight blend comparison
4. paired whole-game bootstrap uncertainty
5. confirmation on additional grouped folds where required
6. full OOF analysis before system promotion

This favors reproducible, complementary signal over attractive one-fold point estimates. A discovery-fold pass is therefore treated as a confirmation-stage result rather than a promoted system.

## GPU and research-systems engineering

The canonical AWS workflow uses an NVIDIA L4 and supports:

- mixed-precision training with explicit precision contracts
- workload-specific loader/thread benchmarks
- resumable model/optimizer/EMA/scaler/RNG checkpoints
- structured telemetry and cost accounting
- fail-closed source/schema/checkpoint/metric validation
- immutable run manifests and champion/challenger lifecycle states
- executed-notebook and Plotly persistence gates
- single outer return bundles

The fixed 20-model inference path achieved a measured **4.784× acceleration** through shared preparation with exact prediction parity on the declared benchmark.

## Public reproducibility boundary

The public repository includes selected implementation, aggregate research evidence, validation logic, notebooks, source/provenance patterns, tests, and system documentation.

It excludes raw competition data, raw third-party response archives, fitted private weights, large checkpoints, credentials, private cloud locations, complete private runners, and unreleased active feature combinations.

## Limitations

- one labeled competition season limits external-validity claims
- external-source coverage varies by player and source
- long-horizon trajectories remain harder than short-horizon trajectories
- strong component models can remain redundant inside an ensemble
- development-fold and OOF evidence are not private leaderboard evidence
- public artifacts intentionally do not reproduce private fitted models end to end
