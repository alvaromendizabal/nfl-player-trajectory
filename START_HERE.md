# NFL Big Data Bowl 2026 - Prediction

This is the [Prediction project](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction). The canonical notebooks are 00, 01, and 02;
the repository name describes the player's trajectory prediction target.

## Review the evidence

Open [notebook 02](notebooks/02_motion_benchmarks.ipynb) for feature attribution,
then [notebook 01](notebooks/01_data_analysis.ipynb) for data and feature rationale.
Public aggregates support review without private tracking or cloud credentials.
All 15 feature-research criteria now pass, and the feature/selection manifest is
frozen. The final-phase input protocol is implemented and verified; final
refitting and the reserved-holdout evaluation remain unfinished.

For a public software review, run the following from the repository with Python
3.11 or later and at least 5 GiB free storage:

```bash
python scripts/bootstrap.py
```

Bootstrap creates the locked Python 3.11 environment, registers the notebook
kernel, and runs the quality suite. No AWS or Kaggle credentials are needed for
that review. The tests use controlled fixtures, and notebooks can read published
aggregates. Reproducing the fitted numerical results requires the licensed
competition inputs and the experiment sequence below.

## Reproduce in the existing project

Use the locked Python 3.11 environment created by `scripts/bootstrap.py`.
Private data, trained artifacts, and stage receipts belong in the existing
workspace or its verified S3 restore. Keep the canonical repository checkout;
do not create alternate notebook copies or overwrite owner exports.

The canonical research sequence is:

```bash
.venv/bin/nfl feature-research
.venv/bin/nfl context-research
.venv/bin/nfl representation-research
.venv/bin/nfl research-report
uv run --locked scripts/nonlinear_probe.py
```

Run each of the following scripts for `inner_1`, `inner_2`, `inner_3`, and
`development` via its `--fold` argument:

The full-bank budget and omission fits require **128 GiB RAM**. The cloud runner
checks the actual container memory limit before starting this phase. Reviewing
published notebooks and predicting individual plays have much smaller requirements.

```bash
uv run --locked scripts/ablate_features.py --fold inner_1
uv run --locked scripts/joint_feature_fit.py --fold inner_1
uv run --locked scripts/feature_budget.py --fold inner_1
```

Then attribute the selected width, convert its verified trees, replay raw
inference, and check the organizer's unlabelled sample before publishing:

```bash
uv run --locked scripts/feature_attribution.py
uv run --locked scripts/prepare_tree.py --self-test
uv run --locked scripts/prepare_tree.py
.venv/bin/python scripts/validate_research.py
# Run this command for each of inner_1, inner_2, inner_3, and development.
uv run --locked scripts/ablate_wide_features.py --fold inner_1
# Also run this recorded combined omission for each of the four folds.
uv run --locked scripts/feature_simplification.py --fold inner_1
uv run --locked scripts/validate_gateway.py
.venv/bin/python scripts/review_feature_gate.py
.venv/bin/python scripts/notebooks.py --publish
.venv/bin/python scripts/quality.py
.venv/bin/nfl backup
```

These commands document reproduction of the completed feature research.
The project's bounded SageMaker runner restores
checksum-verified inputs, excludes holdout tracking, resumes completed stages,
and checkpoints each full-bank fold and major phase. S3 snapshots reference content-addressed
objects; the canonical restore command verifies those hashes. Do not use a
checkpoint from changed numerical source as if it were current.

The main dependency lock is unchanged. The nonlinear and gateway scripts use
separate PEP 723 locks, invoked with `uv run --locked`. Raw model pickle files are
private, source-verified artifacts from this run, not files to load from strangers.

## Prepare the final experiment

After restoring the verified research artifacts:

```bash
.venv/bin/python scripts/prepare_final.py --publish
```

This checks the complete feature gate, all 224 refit games, cache row alignment,
both frozen schemas, and input fingerprints. It preserves an immutable protocol
and reuses its verified input review on a repeated run. It does not fit or score
the final model. The [final protocol](docs/FINAL_PROTOCOL.md) records the exact
refit and reserved-evaluation rules before those steps are implemented.

## Export only when you choose

The final cell in notebook 02 is off by default. Enabling it checks the current
research bundle and creates `artifacts/kaggle/submission.ipynb` for your download.
The exporter resolves the latest verified inference artifact and checks its source
lineage. Automated quality exports and sample gateway output live under
`artifacts/quality/` and cannot replace the owner's generated artifact.

The generated notebook uses the organizer inference interface and never submits
to Kaggle. A local sample Parquet is an interface check, not a hidden-test score.
No public model-hosting service is required to review the project.

The [data card](docs/DATA_CARD.md) explains the exact Prediction inventory,
season boundaries, supplied task information, and limits of the evidence.
