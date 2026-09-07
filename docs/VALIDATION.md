# Phase 0 validation

Validated September 7, 2026, on Linux x86_64 / Python 3.11.15.

| Check | Result |
| --- | --- |
| Dependency resolution and locked install | Passed |
| Python compilation | Passed |
| Ruff lint and formatting | Passed |
| mypy, 13 source files | Passed |
| Automated tests | 41 passed; warnings treated as test errors |
| Synthetic pipeline and checkpoint reuse | Passed |
| Notebook 00 cells | 4 executed successfully with real captured outputs |
| Baseline Kaggle notebook export | Passed; schema-valid notebook |
| Bootstrap command rerun | Passed; both existing environment and original archive + update tested |
| Final quality gate | 12 checks passed |
| Static trajectory preview | Rendered and visually inspected |

Final quality run: `20260907T003834Z-e505ce57` (7.17 seconds).

The notebook test executes code in a dedicated Python process using IPython, with
real output capture. This session cannot open kernel sockets, so a standard Jupyter
kernel execution was unavailable. Interactive SageMaker Jupyter execution remains a
user-side acceptance step. No warnings were emitted by the final quality/bootstrap run.

## Tested failure cases

The suite checks the exact 2N metric denominator, row-order invariance, missing,
extra, duplicate and nonfinite predictions, incorrect player alignment, output-clock
reset, irregular input-frame spacing, single-frame fallback, temporal grouping,
short-history rejection, interrupted stages, stale source fingerprints, corrupted
outputs, process locks, heartbeat logging, unsafe archive paths, ZIP extraction,
paginated file download, download reuse, audit reuse, inconsistent horizons, backup
reuse, checksum-corrupt recovery, and preservation of divergent local files.

## Cloud state

The new private SageMaker space is InService with a default ml.t3.large instance,
50 GB EBS, and a 60-minute idle timeout. No application or training job was started.
The dedicated S3 bucket has versioning, AES256 encryption, and all public-access
blocks enabled. IAM policy simulation allows the existing Studio execution role to
list/read/write this bucket. Simulation is not a live transfer from a Studio process.

## Explicitly not yet verified

- Authenticated Kaggle inventory/download using the user's account in this new space.
- Actual NFL CSV audit or data-dependent validation results.
- Interactive notebook execution inside SageMaker.
- Official Kaggle local gateway or leaderboard submission.
- Late-submission eligibility for the authenticated account.
- GitHub repository creation, remote push, hosted CI, or a merged PR.
- GPU training, trained models, or Hugging Face publication.

These are visible gates for the next steps, not successful results. Synthetic scores
must not be presented as NFL model performance.

## Export regression resolved

The first delivered archive failed Ruff in the user's SageMaker space because
its generated submission notebook placed imports after setup code. The original
local check had not reliably linted ignored generated artifacts.

The canonical exporter now places future and module imports in the first code
cell, loads the evaluation module after path setup using importlib, and formats
the exported notebook. The quality gate regenerates the artifact first, then
explicitly checks its lint and formatting independent of Git ignore behavior.
No lint rule was disabled.

Two new regression tests validate the generated notebook outside Git and execute
its exported predictor against the package reference, including shuffled target
row order. The exact original archive was extracted into a fresh /tmp directory,
the three code/test updates were applied, and bootstrap completed with 35 tests
passing. The existing-environment bootstrap also passed. The official Kaggle
gateway remains untested because this environment lacks the competition data.

## Browser sign-in

The canonical authentication script now uses the official Kaggle SDK OAuth flow.
Remote SageMaker terminals print a Kaggle approval link and accept the one-time
verification code returned by the browser. Existing OAuth credentials are refreshed
and validated on reuse; `--force` deliberately starts a new approval. The SDK owns
credential storage outside the repository. Project logs retain only progress and
error types, with UTC timestamps, elapsed time, and heartbeats.

Six offline tests exercise first sign-in, credential reuse, explicit reauthorization,
cancellation, end-of-input, and secret-safe failure reporting. The complete 12-check
quality gate passed with 41 tests. The installed CLI also confirms support for
`kaggle auth login --no-launch-browser`. No live OAuth session was started in this
workspace; browser approval must occur for the user's persistent SageMaker space.

The user's supplied SageMaker log records `bootstrap_completed` at
2026-09-07T00:33:39.769681+00:00 with a total time of 10.04 seconds.
The environment does not need to be rebuilt to use browser sign-in.
