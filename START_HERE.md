# Continue from the completed feature experiment

The baseline and all three feature ablations are complete. Landing residual ridge
is the current development choice: 0.9269 coordinate RMSE versus 0.9896 for role
ridge on the same 32 validation games. Do not restart training to refresh notebooks.

## Update the existing SageMaker workspace

Close project notebook tabs before updating, so stale tabs cannot autosave over
updated canonical sources. Run this in the **NFL** terminal, not the Jigsaw terminal:

```bash
cd "$HOME/nfl-player-trajectory" &&
test "$(git branch --show-current)" = main &&
git stash push -m "notebook-session-$(date -u +%Y%m%dT%H%M%SZ)" -- notebooks docs/results &&
git pull --ff-only origin main &&
.venv/bin/python scripts/quality.py &&
.venv/bin/python scripts/notebooks.py --publish &&
.venv/bin/nfl backup &&
.venv/bin/nfl status
```

The named stash preserves tracked local notebook/results edits. Do not pop it over
the new sources, reset hard, or clean private artifacts. Unrelated source edits are
not discarded; a conflicting pull stops. This sequence does not change dependencies,
retrain the baseline/feature models, or create an AWS instance. It verifies code,
executes the review notebooks, publishes validated local evidence into the same
canonical paths, and backs up the completed work to the existing private bucket.

Watch UTC cell/stage events, `stage_reused`, 15-second heartbeats, stage/total elapsed
time, `notebooks_published`, and `backup_completed`. Logs are under `logs/`.
A completed notebook is reused only when its inputs, source, and output hashes match.
An interrupted active notebook restarts; completed model/week checkpoints stay intact.
Do not commit raw data, private artifacts, credentials, or logs to public GitHub.

## Read the work

Open **`notebooks/01_data_analysis.ipynb`**, then
**`notebooks/02_motion_benchmarks.ipynb`**. The first explains data, football signals,
and training-only feature relationships. The second compares actual ablations and
shows uncertainty, defense/receiver errors, and forecast-time slices.

The published snapshot is sufficient for employer review without AWS credentials.
Publication in your local checkout is separate from GitHub commits; do not push
private outputs. The merged research publication already supplies rendered notebooks.

## Generate and download your own inference notebook

In the final cell of notebook 02, change only:

```python
GENERATE_SUBMISSION = True
EXPORT_MODEL = "landing_ridge"
```

Run that cell after the earlier setup cells. It requires your existing
`artifacts/benchmark/model.json` and `artifacts/features/model.json`. Provenance
checks reject weights that disagree with the baseline, frozen training split,
numerical feature code, finite coefficient shapes, or positive scaling factors.
There is no silent fallback and no automatic retraining.

Click **Download your generated submission.ipynb** in the cell output. The second
link downloads its checksum manifest. The canonical generated path is
`artifacts/kaggle/submission.ipynb`; repeated identical exports preserve its bytes.
Set `GENERATE_SUBMISSION = False` and clear the final cell's private download output
before deliberately publishing notebook changes to public GitHub.

## Run your generated notebook in Kaggle

Import the notebook you generated, attach **NFL Big Data Bowl 2026 Prediction**
(`nfl-big-data-bowl-2026-prediction`), select CPU, and disable internet. Run all cells.
The organizer's local inference gateway creates `submission.parquet`, not a CSV.
Only after gateway success, finite coordinates, row counts, unique IDs, and exact
play-by-play ordering are verified does the notebook expose download links for your
output and manifest. Look for `SUBMISSION_VALIDATED`.

You choose whether to submit a saved Kaggle version, subject to your signed-in
account's late-submission eligibility. A preview output is not a hidden-test score.
No Kaggle upload or submission is automated by the project.

Play predictions have checksummed local receipts. A failed play is recomputed while
valid completed plays can be reused. These files must be retained or restored in a
new session; a stopped/deleted Kaggle runtime does not itself provide durable cloud
storage. Hidden reruns never reuse preview caches. The original SageMaker training
artifacts and notebook receipts are backed up with `nfl backup`.

## Next modeling decision

Keep the landing challenger as the measured comparator. Next compare nonlinear
residual learning and an interaction-aware temporal encoder with controlled landing
and neighbor ablations, training-only inner tuning, and a fixed outer split. The
48-game holdout is reserved until selection is frozen. No temporal neural result,
optimizer-level recovery, official gateway pass, or leaderboard score is claimed by
this publication.
