# Review the completed experiment and create your own export

The real-data feature experiment is complete: `landing_ridge` achieved 0.9268683892
coordinate RMSE. Do not recreate the data or rerun baseline training.

Preserve locally executed notebooks and published results before pulling updated sources:

```bash
cd "$HOME/nfl-player-trajectory" &&
git stash push --include-untracked -m "completed results before notebook review" -- notebooks docs/results &&
git pull --ff-only origin main &&
python3 scripts/bootstrap.py &&
.venv/bin/python scripts/notebooks.py --publish &&
.venv/bin/nfl backup
```

The scoped stash preserves notebook/result edits and newly generated, untracked result
files. Do not pop an old render over newly published notebooks. Other source edits are
not discarded; conflicting pulls stop safely. Your data, `.state/`, features, and fitted
models are not removed. Bootstrap tests the updated code without retraining the baseline,
and test exports no longer overwrite your own `artifacts/kaggle/submission.ipynb`.

## Your notebook workflow

Open `notebooks/01_data_analysis.ipynb`, then `notebooks/02_motion_benchmarks.ipynb`,
using **Python (NFL Trajectory)**. The report renders the saved real results. It checks
source/baseline provenance and reconciles error slices before presenting conclusions.

To create your own inference notebook, set `CREATE_SUBMISSION = True` in notebook 02
and run the final cell interactively. It uses your completed local model and exposes
a download link. The default is off; automatic report publication does not perform
this action even if an edited notebook has the flag enabled. No Kaggle upload or
submission is performed. Use your own Kaggle account to run the exported notebook's
local gateway and decide whether to submit. Check submission eligibility there.

The exported predictor includes the actual feature implementation and learned weights,
not a second hand-maintained approximation. Its per-play output hashes and input/model
signatures permit reuse when the working directory is retained. A fresh cloud runtime
needs saved outputs restored before those checkpoints can be reused.

## Modeling comes after the recorded decision

The next research question is a protected landing-feature block plus interactions,
role-aware residual modeling, and then a temporal attention challenger. Use forward-
chaining training folds; refit the baseline and feature selector inside each fold.
The current cached residuals were generated with the full training baseline and must
not be silently reused as out-of-fold residuals. Holdout remains locked.

Notebook 00 has an explicit run/resume control for the existing feature experiment,
not an implemented new neural experiment. Leave it off for ordinary portfolio review.
A later new-model experiment needs its own measured evidence before claiming improvement.
