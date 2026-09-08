# NFL Player Trajectory Lab

Predict post-throw player movement from observed tracking, the supplied landing
point, and player roles. **Measured development challenger:** 0.9269 coordinate RMSE versus 0.9896
for role ridge and 1.7225 for constant velocity on 32 later games. This is local validation, not a Kaggle score.

## Review the work

**[Data and features](notebooks/01_data_analysis.ipynb) →
[Models and evaluation](notebooks/02_motion_benchmarks.ipynb)**

Both notebooks contain rendered output. They read local results when available and
otherwise label the committed experiment snapshot; they do not silently retrain.
Publication validates result provenance before replacing canonical files.
[Readiness](notebooks/00_project_readiness.ipynb) is optional orientation.
[Run instructions](START_HERE.md), [model card](docs/MODEL_CARD.md), and
[validation protocol](docs/VALIDATION.md) provide implementation details.

![Measured motion benchmark](docs/results/benchmark.png)

## Measured result

| Model | Coordinate RMSE | Frame-weighted ADE | Trajectory-weighted FDE |
|---|---:|---:|---:|
| Landing-aware residual ridge | **0.9269** | **0.8254** | **1.4168** |
| Interaction-aware residual ridge | 0.9422 | 0.8438 | 1.4309 |
| Motion-only residual ridge | 0.9467 | 0.8545 | 1.4686 |
| Role-conditioned ridge | 0.9896 | 0.8847 | 1.5057 |
| Constant velocity | 1.7225 | 1.5142 | 2.8696 |
| Constant acceleration | 1.9965 | 1.3073 | 2.6664 |
| Smoothed velocity | 2.0051 | 1.8581 | 3.3798 |
| Ball arrival | 4.0842 | 3.8607 | 5.6192 |
| Last position | 4.4806 | 4.4882 | 6.5526 |

All values are in yards. The 67,857 validation player-frames are identical across
models. The landing challenger reduces RMSE by 6.34% versus role ridge. Its paired
95% game-cluster bootstrap delta is -0.0829 to -0.0434 yards (2,000 resamples).
This supports a validation improvement, not independent holdout confirmation. The
original six-vector role baseline and its verified numerical artifacts are preserved.

The official metric pools squared x/y errors over all coordinates and frames:
`sqrt(sum(dx² + dy²) / (2N))`. ADE, FDE, p95 displacement, role slices, and
forecast-time slices supplement it. Landing-model defensive coverage RMSE is 1.0321 versus 0.5777
for targeted receivers; error grows with forecast time.

## Feature research

**Completed: 2,843 candidates and three real-data residual-ridge ablations.**
The published feature summary exactly matches the checksummed completed S3 artifact.
The notebooks label replayed results as a published experiment snapshot; displaying
them is not a new training run or fresh error-row metric recomputation.

The bank covers twenty frame offsets, seven history windows, circular angles,
landing-relative dynamics, nearest six teammates/opponents, receiver and passer
anchors, closest-approach geometry, and forecast-time interactions. Missing history
and neighbours are masked. Coordinates outside the field are not clipped. Player
IDs, names, birth dates, and post-play outcomes are not numerical predictors.

Training-only residual screening admits at most 96 candidates per ablation;
training-correlation pruning retains at most 64. Fixed-regularization residual ridge
compares motion, landing, and interaction signals on the same validation games.
A large feature count or training association does not establish predictive value,
causal importance, or state-of-the-art performance. The landing feature set beats the richer interaction set in this fixed-budget
ridge experiment; more candidate features did not automatically yield a better model.
A winning temporal neural model has not been reproduced here.

The two review notebooks now show the completed experiment, training-only feature
associations, uncertainty, role slices, and forecast-time error. No repeat training is
needed to read them. [Continue from your saved workspace](START_HERE.md).

## Reliability and validation

Training: 192 games, September 7–December 3, 2023. Validation: 32 games,
December 4–18, 2023. Holdout: 48 games, December 21, 2023–January 7, 2024,
**not evaluated**. Whole games remain together; holdout-only files are not opened
by the feature experiment. No validation targets enter feature screening or scaling.

UTC JSONL events include stage/total elapsed time and 15-second heartbeats.
Weekly checkpoints use atomic writes, locks, input/source signatures, and output
hashes. Feature expansion is batched and guarded at 256 MiB per matrix.
`--checkpoint-s3` adds private full-workspace snapshots after each prepared week,
fitted model, and final feature report. It is phase-level, not mid-operation,
remote recovery. Completed local stages survive ordinary interruption.

CI checks formatting, types, tests, standalone export, and notebook execution;
warnings fail tests rather than being hidden. On a trusted feature-branch push,
`[publish notebooks]` in the commit message requests formatting and a post-quality
commit of the allowlisted research modules, tests, and three executed notebook paths.
The resulting PR must pass CI again.

## Kaggle and scope

**Generate and download your own output through notebook 02.** Submission generation
is off by default. Set `GENERATE_SUBMISSION = True` in its final cell to export the
selected landing model from matching local saved weights. The cell provides download
links for your generated standalone inference notebook and checksum manifest; it
never retrains or uploads to Kaggle.

Import that generated notebook into Kaggle and run the official local gateway. The
notebook validates `submission.parquet` and provides download links. Local play caches
are checksummed; hidden reruns cannot reuse preview caches. A fresh Kaggle session
needs its saved cache restored explicitly. **Actual official gateway execution and a
scored Kaggle submission remain not run in this publication.** The competition has
ended; authenticated late-submission eligibility must be checked.

The code is MIT licensed. Competition data has separate non-commercial conditions
and is not redistributed here. See [sources](docs/SOURCES.md) and the competition
rules before redistributing data or models. A Hugging Face release is not required
to review these notebooks.
