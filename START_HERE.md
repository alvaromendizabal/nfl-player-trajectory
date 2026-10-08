# NFL Big Data Bowl 2026 - Prediction | Start here

[Competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

This guide is the shortest route through the completed NFL player-trajectory project. The [README](README.md) gives the headline result; the documents below explain the evidence and provide a runnable public demonstration.

## A two-minute review

Read the [engineering case study](docs/EMPLOYER_CASE_STUDY.md).

It connects the forecasting problem to the main deliverables: a 20-model ensemble, evaluation across 272 games, GPU performance work, reliable inference delivery, and a recorded private RMSE of **0.46468**. That score comes from late evaluation and is not an official competition placement.

## A ten-minute technical review

1. [Results](docs/RESULTS.md) — exact metrics, evaluation populations, recorded submission, and interpretation.
2. [Reproducibility](docs/REPRODUCIBILITY.md) — public demo, software checks, and the boundary around private data and fitted models.
3. [Project closeout](docs/PROJECT_CLOSEOUT.md) — final scope, delivered artifacts, limitations, and archived research status.
4. [Sanitized result snapshot](docs/results/project_closeout.json) — machine-readable evidence supporting the closeout.

The two OOF populations are intentionally separate: **0.4629258204** on 561,607 supported forecast rows and **0.5242026277** on all 562,936 forecast rows. Both differ from the private evaluation. The larger population includes rare cases with much longer horizons.

## Run the demo

Use Python 3.11 or newer from the repository root:

```bash
python scripts/run_portfolio_demo.py --output demo_output
```

No package installation, credentials, GPU, network access, or competition dataset is needed. The inputs are deterministic synthetic trajectories. Open demo_output/dashboard.html for the self-contained visual report. The directory also contains trajectory graphics, metrics, input/prediction CSV files, game partitions, and a reproducibility manifest. [Reproducibility](docs/REPRODUCIBILITY.md) describes these artifacts and their checks.

This demonstration makes the public workflow inspectable. Its synthetic metrics are not competition results and do not stand in for replaying the private fitted ensemble.

## Inspect implementation and tests

| Location | What to review |
|---|---|
| [Public package](src/nfl_trajectory/) | Selected reusable modeling, data, evaluation, and engineering components |
| [Tests](tests/) | Software behavior and research-contract checks |
| [Scripts](scripts/) | Public entry points, including the standalone demo |
| [Results](docs/results/) | Aggregate evidence with explicit evaluation scope |
| [Sources](docs/SOURCES.md) | Original references and attribution |

The research neural baseline reproduces chack3's public temporal/player-interaction model. The project adds controlled experiments, data integration, ensemble evaluation, runtime engineering, and deployment verification. Attribution does not imply reproduction of another team's complete system.

## Historical research is preserved

The [research archive](research/README.md) and older technical pages record earlier milestones, rejected hypotheses, and intermediate model states. Read their dates and population definitions before comparing scores. An earlier page's “current” or “next” wording describes that milestone, not a new project commitment.

The archive's publication manifest has its own integrity check:

```bash
python research/publication/validate.py
```

This verifies registered public artifact hashes. It does not retrain models, retrieve restricted inputs, or reproduce private scores. The maintained quality checks and their environments are described in [Reproducibility](docs/REPRODUCIBILITY.md).

Private neural champion weights, row-level private predictions, the latest production recipe, and nonpublic execution bundles are withheld. A small fitted baseline and selected historical implementation are public. This boundary distinguishes runnable public software, recorded research evidence, and private ensemble reproduction.
