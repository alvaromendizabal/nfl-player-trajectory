# Review the completed experiment

The feature run completed successfully: landing ridge reached 0.9268683892 RMSE;
all three notebooks were published locally and the final S3 backup completed.
**Do not recreate data, the environment, or the successful benchmark/features.**

## Update the notebook presentation without retraining

In your existing SageMaker terminal:

```bash
cd "$HOME/nfl-player-trajectory" &&
git stash push --include-untracked -m "executed feature notebooks before research update" -- notebooks/ docs/results/ &&
git pull --ff-only origin main &&
python3 scripts/bootstrap.py &&
.venv/bin/python scripts/notebooks.py --publish &&
.venv/bin/nfl backup
```

The narrowly scoped stash preserves tracked notebook edits **and newly generated
untracked result files** before pulling canonical replacements. It does not touch
`data/`, `artifacts/`, `.state/`, or other source edits. Do not pop the old render
over the new notebooks. Bootstrap checks the code/environment; the command block
does not rerun real-data feature construction or fit a model.

Open `notebooks/01_data_analysis.ipynb`, then `notebooks/02_motion_benchmarks.ipynb`
using **Python (NFL Trajectory)**. They show current real-data results, training
associations, feature-set overlap, error budgets, and the next modeling decision.
The numerical modules and dependency lock are unchanged.

## Your export and download

At the end of notebook 02, set `GENERATE_EXPORT = True` and run that cell. It
reads your current local selected model, verifies the completed model/source/split
checksums, writes `artifacts/kaggle/submission.ipynb`, and displays a download link.
No file is submitted or uploaded. Restore the switch to `False` before committing
the public research notebook.

Open the generated inference notebook in Kaggle yourself, attach the competition
input, use CPU, and disable internet. Run its organizer gateway. A local
`submission.parquet` link appears if that file is produced. Local sample output
is not a hidden-test prediction or a leaderboard score. You decide whether to make
an actual submission; this repository does not call a submission endpoint.

## Next modeling step

Do not add thousands more columns blindly. Preserve the full landing model and
compare an additive interaction correction and role-conditioned residuals. The
prior 64-column interaction experiment displaced 23 landing features, so it is
not a clean test of interaction value. Inner chronological training folds should
select feature budgets and hyperparameters; the existing validation compares the
final candidates and the holdout remains untouched. Those next fits are not yet
implemented or claimed complete in this presentation/export update.

The existing optional training cell in notebook 01 can run/resume `nfl features`
when explicitly enabled. It is off for normal review. Rerun it only when a verified
stage is missing or a deliberate numerical change warrants recomputation.

Automatic publication refuses an enabled training/export switch before any cell runs.
Restore those manual controls to `False` before publication. The exported predictor
retains verified per-play results only while its working directory is retained or
restored from saved outputs; it does not assume a fresh Kaggle session restores
earlier disk state.
