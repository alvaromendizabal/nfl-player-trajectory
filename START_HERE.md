# NFL Big Data Bowl 2026 - Prediction

This is the Prediction project. The canonical notebooks are 00, 01, and 02;
the repository name describes the player's trajectory prediction target.

## Review the evidence

Open [notebook 02](notebooks/02_motion_benchmarks.ipynb) for feature attribution,
then [notebook 01](notebooks/01_data_analysis.ipynb) for data and feature rationale.
Public aggregates support review without private tracking or cloud credentials.
Final training is gated on the research evidence, not on a successful pipeline run.

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

```bash
uv run --locked scripts/ablate_features.py --fold inner_1
uv run --locked scripts/joint_feature_fit.py --fold inner_1
uv run --locked scripts/feature_budget.py --fold inner_1
```

Then validate raw inference, wider attribution, and the organizer's unlabelled
sample interface before publishing:

```bash
.venv/bin/python scripts/validate_research.py
uv run --locked scripts/feature_attribution.py
uv run --locked scripts/validate_gateway.py
.venv/bin/python scripts/notebooks.py --publish
.venv/bin/python scripts/quality.py
.venv/bin/nfl backup
```

These commands document reproduction; the current research run is executed and
monitored on the project's bounded SageMaker processing job. Its runner restores
checksum-verified inputs, excludes holdout tracking, resumes completed stages,
and checkpoints each major phase. S3 snapshot manifests reference content-addressed
objects; the canonical restore command verifies those hashes. Do not use a
checkpoint from changed numerical source as if it were current.

The main dependency lock is unchanged. The nonlinear and gateway scripts use
separate PEP 723 locks, invoked with `uv run --locked`. Raw model pickle files are
private, source-verified artifacts from this run, not files to load from strangers.

## Export only when you choose

The final cell in notebook 02 is off by default. Enabling it checks the current
research bundle and creates `artifacts/kaggle/submission.ipynb` for your download.
The linear research predictor and the stronger experimental tree are labelled
separately. Automated quality exports and sample gateway output live under
`artifacts/quality/` and cannot replace the owner's generated artifact.

The generated notebook uses the organizer inference interface and never submits
to Kaggle. A local sample Parquet is an interface check, not a hidden-test score.
No public model-hosting service is required to review the project.
