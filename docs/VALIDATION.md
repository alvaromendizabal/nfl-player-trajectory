# Validation evidence

Validated September 7, 2026, on Linux x86_64 / Python 3.11.15.

| Check | Result |
| --- | --- |
| Frozen dependency install, compilation, Ruff lint and formatting | Passed |
| mypy | Passed; 16 source files |
| Automated tests | 70 passed; warnings treated as errors |
| Complete offline quality gate | 13 checks passed in 30.084 seconds |
| Project notebooks | All three executed with captured outputs |
| Constant-velocity and fitted-model Kaggle exports | Passed schema, lint, formatting and numerical parity tests |
| Real NFL benchmark | Six models; 32 validation games; 67,857 target positions |
| Full benchmark / repeat | 33.924 seconds / 5.166 seconds |
| Repeat computation | All 33 numerical stages reused; no new numerical stage started |
| Published static figures | Rendered and visually inspected |
| Offline field animation | Frame/trace alignment, slider and embedded resources tested |

Quality run: `20260907T013100Z-1b8b6e14`.
Real benchmark run: `20260907T012026Z-03b641f0`.
Resume run: `20260907T012254Z-4c89d51e`.
No warnings were emitted by the completed quality run. GitHub Actions repeats the
quality gate on pushes and pull requests and retains its evidence for 30 days.

## Scientific evidence

The user's completed authenticated download and audit covered 18 weekly pairs:
4,880,579 observed rows and 562,936 target rows. The existing chronological split
is unchanged: 192 training, 32 validation, and 48 holdout games. This benchmark
restored only weeks needed for training and validation; holdout-only weeks were
not downloaded or evaluated in this analysis workspace.

Role-conditioned ridge achieved coordinate RMSE **0.9895688 yards**, compared with
**1.7225165** for constant velocity: **42.55% lower validation RMSE**. The whole-game
bootstrap 95% interval is **[0.9218024, 1.0518861]**. This is a local validation score,
not a Kaggle leaderboard result. All six predictors use exactly the same target rows.

The split SHA256 is
`383b3b76cd7dd0085ee5eac9e52ff49bb380fbb55e3e46e2603b570c8bf417e7`.
The numerical cache signature is
`affbaca23a9aadfd2beda75b203200566804dd6f7a51a47a9ac7fe8ab3825166`.
Public aggregate metrics, coefficients, protocol and figures are in `docs/results/`.

## Failure and correctness coverage

Tests cover the exact 2N metric denominator, unequal trajectory lengths, whole-game
bootstrap pooling, row-order preservation, missing/extra/duplicate/nonfinite
predictions, player alignment, output-clock reset, irregular input spacing,
single-frame fallback, horizon validation, grouped temporal splits and sealed holdout,
translation and rotation equivariance, additive training statistics, unseen-role
fallback, target-coordinate leakage, export equivalence, interrupted stages, changed
fingerprints, corrupt caches, process locks, timestamped heartbeats, unsafe archive
paths, download reuse, backup reuse, corrupt restoration and divergent local files.

Notebook execution uses dedicated IPython processes with actual rich output capture.
The ordinary SageMaker Jupyter kernel and interactive browser playback still need
user-side acceptance. Static visual inspection and animation structure tests passed;
a browser rendering check could not be completed in this environment.

## Cloud and submission status

The user's latest log confirms GitHub push, authenticated Kaggle download, real-data
audit and a successful private S3 snapshot. The existing SageMaker development space
is in use. No GPU training job, paid model deployment or Hugging Face publication was
started for this phase.

The standalone trained-model notebook embeds its coefficients and canonical predictor
code. Its prediction parity is tested, but the official Kaggle local gateway and any
leaderboard submission remain pending. Late-submission eligibility is unverified.
This phase establishes an interpretable benchmark; it is not a state-of-the-art claim.
