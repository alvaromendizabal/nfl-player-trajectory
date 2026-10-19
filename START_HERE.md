# Technical review

Start with the [case study](docs/EMPLOYER_CASE_STUDY.md), then
[reproduce the public review](docs/REPRODUCIBILITY.md).

The maintained code covers organizer-data auditing, observation-only reference
forecasting, fitted reference controls, keyed coordinate scoring, whole-game
resampling, and atomic recovery. The final neural ensemble is documented through
its [original aggregate evidence](docs/results/project_closeout.json); its private
weights and exact recipe are not shipped.

| Review question | Implementation and tests |
|---|---|
| Are forecasts aligned to every requested row? | `final_evaluation.py`, `test_final_evaluation.py` |
| Can labels leak into reference prediction? | `models.py`, `test_models.py` |
| Can a stale or partial stage be reused? | `runtime.py`, `test_runtime.py` |
| Are reported errors independently reproducible? | `tools/audit_public_forecasts.py`, `tools/test_forecast_audit.py` |

Run all package tests with `uv sync --frozen --group dev` followed by
`uv run --frozen python scripts/quality.py`. This does not call AWS or submit to Kaggle.
