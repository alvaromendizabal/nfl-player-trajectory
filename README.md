# NFL Player Trajectory

**Measured result: 0.9269 coordinate RMSE (yards)** on 32 later validation games.
Landing-aware residual ridge improves the established learned baseline by **6.34%**
and constant velocity by **46.19%**. The 48-game holdout remains unscored.
These are development-validation results, not Kaggle leaderboard scores.

## Review the work

**[01 · Data, physics, and features](notebooks/01_data_analysis.ipynb) →
[02 · Results, failure analysis, and your own export](notebooks/02_motion_benchmarks.ipynb)**

The notebooks contain executed tables and embedded figures. No cloud account or
rerun is needed to review them. [00 · Project readiness](notebooks/00_project_readiness.ipynb)
is optional orientation and provides an explicit run/resume control.

| Model | Coordinate RMSE / yd | Frame ADE / yd | Trajectory FDE / yd |
|---|---:|---:|---:|
| **Landing-aware residual ridge** | **0.9269** | **0.8254** | **1.4168** |
| Interaction residual ridge | 0.9422 | 0.8438 | 1.4309 |
| Motion residual ridge | 0.9467 | 0.8545 | 1.4686 |
| Role-conditioned ridge baseline | 0.9896 | 0.8847 | 1.5057 |
| Constant velocity | 1.7225 | 1.5142 | 2.8696 |

All five models were evaluated on the same recorded 67,857 player-frames / 5,399
trajectories. The winning model's paired 95% game-bootstrap RMSE difference versus
role ridge is **[-0.0829, -0.0434] yards**. Intervals are development diagnostics,
not selection-adjusted guarantees of future performance.
[Measured evidence](docs/results/feature_summary.json) · [Model card](docs/MODEL_CARD.md)

## Research conclusions

The 2,843 deterministic candidates encode observed motion, landing-relative dynamics,
player interactions, and forecast time. Screening and scaling use training games only;
three fixed-regularization residual models retain 64 features each. The original
baseline already uses the supplied landing point, including in the motion-residual ablation.

**More interaction features did not automatically win.** The interaction candidate
displaced 23 columns from the landing model. The next controlled experiment must preserve
the successful feature block and separate added information from changed capacity.

**Target the objective's largest errors.** Defensive coverage accounts for 89.0% of
remaining squared error. Forecasts beyond one second are 26.9% of frames but 83.6% of
squared error. Notebook 02 derives and reconciles these quantities from the recorded slices.
See the [next experiment](docs/RESEARCH_PLAN.md); no new temporal neural result is claimed.

## Create your own inference notebook

In notebook 02, set **`CREATE_SUBMISSION = True`** and run the final cell yourself.
It verifies your completed local model and creates a download link for
`artifacts/kaggle/submission.ipynb`. Export supports all three residual challengers
as well as the preserved physical and role-ridge references.

Automatic portfolio rendering leaves generation off. Quality checks use a separate
artifact path and cannot replace your generated notebook. No credentials are requested
and no Kaggle submission API is called. This is a code-competition inference notebook,
not a fabricated hidden-test CSV. Run its official local gateway on Kaggle, review the
result, and choose whether to submit using your own account. Official gateway execution
and leaderboard scoring are not claimed complete here.

## Reproducibility

Training: 192 games, September 7–December 3, 2023. Validation: 32 games,
December 4–18, 2023. Holdout: 48 games, December 21, 2023–January 7, 2024.
Whole games stay together. Existing numerical source and the dependency lock remain
unchanged by the report/export release, so completed feature/model work stays reusable.

UTC JSONL logs include stage/total timing and a 15-second heartbeat. Completed stages
are reused only after signature and output-hash checks. Atomic writes preserve the last
successful outputs. Weekly private S3 snapshots provide phase-level recovery; a fresh
runtime requires restoring saved artifacts rather than assuming its disk persists.
The standalone inference notebook also uses verified per-play checkpoints.

[Continue in the existing workspace](START_HERE.md) ·
[Validation protocol](docs/VALIDATION.md) · [Recovery](docs/RECOVERY.md)

Original code is MIT licensed. Competition data and third-party code have separate
terms and are not redistributed here. Published evidence contains aggregate measurements,
figures, and research interpretation. See [sources](docs/SOURCES.md).
