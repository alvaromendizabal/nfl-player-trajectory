# Engineering case study: NFL player trajectory forecasting

**From multi-agent motion prediction to a validated, reproducible ML delivery system.**

I built a forecasting system that combines temporal deep learning, grouped validation, cloud GPU execution, and inference verification. Across the project’s recorded submissions, private coordinate RMSE fell **33.7%, from 0.70090 to 0.46468**. The latest result comes from a 20-model ensemble in Kaggle late evaluation. This measures progress across successive systems; controlled experiment effects are reported separately, and no official competition placement is claimed.

## The problem

Forecast selected players' future x/y coordinates from observed tracking history and supplied play context. Players move in relation to one another, forecasts have different lengths, and rows from a game share substantial context. A useful evaluation must keep games separate, align every requested prediction, and preserve difficult cases rather than silently dropping them.

Compute was another design constraint. The research workflow ran on a single NVIDIA L4 in AWS SageMaker, so redundant preparation, interrupted jobs, and weak experiment selection all had practical costs.

## What I delivered

| Workstream | Delivered capability |
|---|---|
| Modeling | Temporal/player-interaction model integration, controlled extensions, and a 20-model ensemble across four grouped split families |
| Validation | Game-held-out predictions, exact coordinate RMSE, fixed ensemble comparisons, paired whole-game uncertainty estimates, and slice/coverage accounting |
| Data engineering | Direct NFL/ESPN acquisition, provenance records, player/game identity resolution, and historical features restricted to eligible earlier observations |
| Runtime | Shared inference preparation, measured GPU-workload settings, resumable training state, and resource/throughput telemetry |
| Delivery | Integrity-checked model artifacts, packaged inference, organizer-gateway validation, and an exact recorded private submission |
| Public review | Selected source, tests, aggregate evidence, an interactive trajectory lab, a deterministic Python report, and documented reproduction scope |

I owned the modeling and evaluation workflow, data integration, system extensions, runtime engineering and delivery controls. Published architecture and training methods remain attributed in [Sources](SOURCES.md).

## Three decisions with measurable consequences

### Evaluate diversity at the ensemble level

A lower standalone model error does not establish that a candidate improves an existing ensemble. I retained out-of-fold prediction banks and compared fixed ensemble constructions on the same game-held-out rows.

An earlier expansion from two to four grouped split families improved supported-population OOF RMSE by **0.00217835**, with a paired-game 95% interval of **[0.000629, 0.003861]**. All five original folds improved. The gain remained positive after each of 272 single-game removals. This established a system-level benefit while preserving unsuccessful variants as negative evidence.

### Make evaluation coverage part of correctness

A later inference-policy study included all **562,936 forecast rows**, covering rare missing-context and unusually long-horizon cases. Output clipping and horizon extension were evaluated against the existing policy before the selected behavior was packaged for delivery.

The accepted policy produced **0.4629258204 supported-population OOF RMSE** on 561,607 rows and **0.5242026277 full-population OOF RMSE** on all rows. The larger full-population change was concentrated in two unusually long plays, so it is not evidence of a broad improvement of the same magnitude.

The subsequent private measurement moved from **0.46487 to 0.46468**. That separate measurement established transfer to the late-evaluation setting; local OOF gains alone did not establish it.

### Remove repeated work without changing predictions

The fixed 20-model inference path originally repeated preparation work across ensemble members. Sharing that preparation produced a measured **4.784× speedup**, with exact prediction parity on the declared benchmark.

Long-running training also saved more than weights: checkpoints retained optimizer state, exponential moving-average weights, gradient-scaler state, random state, progress, and source/configuration identity. Integrity checks and resumable stages let valid upstream work survive downstream failures. Runtime improvements were tied to measured workloads rather than claimed as universal speedups.

## Evidence at closeout

| Measurement | Result | Interpretation |
|---|---:|---|
| Recorded private evaluation | **0.46468** | Submission 56928100; late evaluation, no official rank claim |
| Recorded private RMSE reduction | **33.7%** | 0.70090 → 0.46468 across successive project systems |
| Supported-population OOF | **0.4629258204** | 561,607 forecast rows |
| Full-population OOF | **0.5242026277** | 562,936 forecast rows across 272 games |
| Scored ensemble | **20 models** | Four grouped split families |

These are separate evaluation scopes. OOF uses held-out models for each game; deployment uses the full ensemble. OOF results were inspected during research and should not be interpreted as a fresh untouched test set. [Results](RESULTS.md) and the [sanitized snapshot](results/project_closeout.json) contain the exact values and provenance.

## Data and reliability work

The historical-data pipeline acquired **3,920 NFL Next Gen Stats rows covering 691 historical players**, plus **36 ESPN scoreboards and 544 game summaries**. Game identity resolution mapped all 272 competition games. For within-season historical context, a play in week w could use only eligible observations from earlier weeks.

Coverage, data access, model integration, and predictive benefit remained separate decisions. A successful data pipeline did not automatically qualify its features for the accepted ensemble.

Operational failures also became reproducible checks: source schemas, prediction keys, numerical precision, artifact hashes, and external-service response handling were tested before reuse. Submission identity and recorded score were tracked separately from dataset or notebook creation, preventing an uploaded artifact from being mistaken for a completed evaluation.

## Make trajectory errors inspectable

I built [Route Lab](https://alvaro-nfl-route-lab.tartmacaw2.chatgpt.site) to expose the relationship between observed motion, a forecast rule and the resulting error. A reviewer can select a synthetic play and player, change the forecast horizon or damping, animate the trajectory and compare coordinate RMSE with displacement diagnostics. Forecasts are computed from observations; labels enter only after prediction.

The existing Python generator produces deterministic fixtures and a standalone report. The browser adds interactive forecasts and JSON export. This is a public engineering demonstration with authored data, not the private temporal ensemble or a new competition measurement. [Run and verify](REPRODUCIBILITY.md)

## Public reproduction and limitations

The public repository contains selected source and tests, aggregate evidence, and a standard-library demo on deterministic synthetic trajectories. It runs without credentials or restricted data. The reported private model scores require private fitted artifacts and cannot be regenerated by the public demo.

Private neural ensemble weights, row-level private predictions, the private ensemble configuration, and nonpublic execution bundles are withheld. A small fitted baseline and historical source archives remain public. Evidence from one labeled season limits claims about other seasons; repeated validation inspection limits claims of unbiased generalization. The system is a completed research and engineering artifact, not a certified player-evaluation product.

[Run and verify the public project](REPRODUCIBILITY.md) · [Read the final project state](PROJECT_CLOSEOUT.md) · [Return to the review guide](../START_HERE.md)
