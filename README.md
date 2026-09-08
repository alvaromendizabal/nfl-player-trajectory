# NFL Player Trajectory Lab

Predict post-throw player movement from pre-throw tracking, the supplied landing
point, and player roles. **Landing-aware residual ridge: 0.9269 coordinate RMSE**
on 32 later games—**6.34% lower** than the learned motion baseline. These are
measured local validation results, not a Kaggle leaderboard score.

## Review the work

**[01 · Data, football hypotheses, and features](notebooks/01_data_analysis.ipynb) →
[02 · Models, error analysis, and next decisions](notebooks/02_motion_benchmarks.ipynb)**

The canonical notebooks include executed tables and embedded figures. No AWS,
Kaggle, or Hugging Face account is needed to review them. Notebook 00 is optional
orientation. See [run instructions](START_HERE.md) for your existing workspace.

| Model | Coordinate RMSE | Frame-weighted ADE | Trajectory-weighted FDE |
|---|---:|---:|---:|
| Landing-aware residual ridge | **0.9269** | **0.8254** | **1.4168** |
| Interaction-aware residual ridge | 0.9422 | 0.8438 | 1.4309 |
| Motion-only residual ridge | 0.9467 | 0.8545 | 1.4686 |
| Original role-conditioned ridge | 0.9896 | 0.8847 | 1.5057 |
| Constant velocity | 1.7225 | 1.5142 | 2.8696 |

All values are yards; all models score the same 67,857 player-frames / 5,399
trajectories. The landing model's 95% game-cluster RMSE interval is 0.8587–0.9922.
Its paired difference versus role ridge is −0.0829 to −0.0434 yards. The
[recorded feature results](docs/results/feature_summary.json) retain exact values.

## What the experiments teach

The bank contains **2,843 pre-throw candidates**, but each residual challenger
retains only 64 training-selected features. Thirty-one of the landing model's
features derive from longitudinal ball bearing (`ball_ux`). The interaction
model shares only 41 features with it, replacing 23—including a major lateral
motion term. This is **not a nested, add-only ablation**: the result cannot isolate
the value of interactions from the cost of removing useful landing features.

Defensive coverage accounts for **89.0% of remaining squared error**. Forecast
seconds two and three contribute **80.2%**; the fourth-second slice has just 127
rows. The next experiment should preserve the complete landing representation,
then test a small interaction correction and role-conditioned residuals with
chronological training-only selection. No new challenger or holdout result is
claimed before it is measured. The notebooks show the calculations and caveats.

## Validation and engineering

The official metric is `sqrt(sum(dx² + dy²) / (2N))`. ADE, FDE, p95 displacement,
role/horizon slices, and paired game-bootstrap intervals supplement it.
Training uses 192 games (September 7–December 3, 2023); validation uses 32 games
(December 4–18). The later **48-game holdout remains unscored**. All frames and
players from one game stay together. Feature screening/scaling use training only.

Canonical code has explicit numerical, leakage, artifact-integrity, recovery,
and export-parity tests. CI checks lint, formatting, types, warnings-as-errors,
and notebook execution in the locked Python 3.11 environment. UTC JSONL logs
include stage/cell and total durations plus a 15-second heartbeat. Atomic writes,
locks, input/source signatures, and output hashes protect resumable stages.
Private content-addressed S3 snapshots retain completed work. Reporting changes
do not modify numerical source or invalidate the completed feature experiment.

## Generate an artifact yourself

The final cell in notebook 02 is **off by default**. Set `GENERATE_EXPORT = True`
in your workspace to generate and download your own standalone inference notebook
from the verified, selected local model. The exporter supports the trained residual
champion; it does not silently substitute the older baseline. Tests use a separate
quality-output directory and cannot overwrite the owner's generated artifact.

The generated notebook uses the official organizer gateway and provides a local
Parquet download when that gateway creates it. **It never submits to Kaggle.**
Local sample predictions are not hidden-test results. You control gateway execution
and any subsequent submission; official gateway execution remains unverified here.

The code is MIT licensed; competition data has separate conditions and is not
redistributed. See [sources](docs/SOURCES.md). No extra model-hosting service is
required for the employer review path.
