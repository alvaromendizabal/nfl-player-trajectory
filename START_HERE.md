# Run and review the NFL trajectory project

## Continue in the existing SageMaker terminal

Keep `$HOME/nfl-player-trajectory`, its `.venv`, `data/`, `.state/`, and `artifacts/`.
Do not extract another bundle, reinitialize Git, or redownload successful work.
Bootstrap checks the environment and notebooks; it does **not** train the real-data
benchmark. Complete the sequence below after reviewing `git status --short` and
committing any intentional local source changes.

```bash
cd "$HOME/nfl-player-trajectory" &&
git pull --ff-only origin main &&
python3 scripts/bootstrap.py &&
.venv/bin/nfl benchmark &&
.venv/bin/python scripts/notebooks.py --publish &&
.venv/bin/python kaggle/export.py --model role_ridge &&
.venv/bin/nfl backup &&
.venv/bin/nfl status
```

This uses the existing CPU environment. It does not create a cloud instance or
launch a training job. Your running SageMaker app and storage retain their normal
AWS charges. Stop the app when finished; do not delete the space.

## What the commands produce

`nfl benchmark` verifies the frozen game split, prepares development weeks, fits
only training games, evaluates validation games, and renders the report. Its
verified numerical stages are reused on repeat; holdout evaluation stays locked.

`scripts/notebooks.py --publish` requires completed local benchmark evidence and
matching split hashes. It executes each canonical notebook in a separate Python
process, or reuses an output with the same source/input signature and verified
hash. Warnings and stderr fail the execution rather than being hidden. Failed
execution leaves the last successful notebook output in place.

All notebook candidates are validated before publication. The command refreshes
the **same** files in `notebooks/` and the eight aggregate result files in
`docs/results/`. Each replacement is atomic; the multi-file publication is not a
single filesystem transaction. A receipt at `artifacts/notebooks/publication.json`
records completion and output hashes. An interrupted publication can be retried
with the same command; do not treat a missing or failed receipt as a complete release.

Notebook execution without `--publish` remains suitable for CI and can use the
explicitly labeled published snapshot. It does not claim to fit a new model.
The supported execution method is isolated-process IPython, not a live Jupyter
kernel; the notebooks use plain Python cells and standard rich display outputs.

The exporter creates `artifacts/kaggle/submission.ipynb` with embedded ridge weights.
Export is **not** an official gateway test or a leaderboard submission.
`nfl backup` records a private S3 snapshot; `artifacts/last_backup.json` stores its
manifest identifier. Keep logs, data, and recovery state outside Git.

## Read the results

Start with [README](README.md), then open
[01 · Data analysis](notebooks/01_data_analysis.ipynb) and
[02 · Motion benchmarks](notebooks/02_motion_benchmarks.ipynb).
The canonical notebooks contain rendered outputs, so an employer does not need AWS,
a notebook kernel, or Kaggle credentials to read them. Notebook 00 is optional
orientation, not the lead portfolio demonstration.

For interactive play analysis, open `artifacts/benchmark/report.html` locally.
Use the **Python (NFL Trajectory)** kernel for interactive work in SageMaker.

## Progress and recovery

Terminal events include UTC timestamps, elapsed command time, stage or cell timings,
and a 15-second heartbeat. `stage_reused` means matching output hashes were checked;
it is not a new training run. `notebooks_published` and a passed publication receipt
mark successful local notebook publication.

If a command fails, rerun that command after addressing its reported cause. Do not
remove verified checkpoints or suppress warnings to make a run appear successful.
See [recovery instructions](docs/RECOVERY.md) for restoring a private S3 snapshot.

## Publish the refreshed outputs to GitHub

Local publication does not push or merge Git changes. Review `git diff --stat`,
then commit **only** `notebooks/` and `docs/results/` on a feature branch with a
message such as `docs: publish verified NFL benchmark notebooks`. Open a pull
request against `main`; inspect the rendered outputs and require Quality to pass
before merging. Do not stage `data/`, `artifacts/`, logs, credentials, or private
cloud configuration. Commit or otherwise preserve intentional local changes before
the next `git pull`; never discard them merely to make an update succeed.

## Remaining modeling and submission work

The learned ridge model is an interpretable baseline, not a state-of-the-art claim.
Next research is role/interaction feature ablation and stronger residual models,
then a temporal interaction model under the same game-separated development split.
Select on validation coordinate RMSE, retain ADE/FDE and role/horizon diagnostics,
and evaluate the holdout only after locking model selection.

The official Kaggle gateway and authenticated late-submission eligibility still
need verification. This historical competition cannot yield a new medal. Record a
leaderboard score only after Kaggle actually returns one. No Hugging Face release
is required to read the portfolio.
