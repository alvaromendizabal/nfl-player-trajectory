# NFL Big Data Bowl 2026 - Prediction | Start Here

**Competition ID:** `nfl-big-data-bowl-2026-prediction`

This repository is the **employer-facing, semi-reproducible record** of an AWS-first NFL player-trajectory forecasting research program.

The public surface is intentionally curated: it exposes the system design, validation discipline, selected implementation, aggregate evidence, and engineering decisions needed for technical review while keeping restricted data, fitted weights, private cloud locations, and active competitive IP out of public history.

## Choose a review path

### 60 seconds — recruiter / hiring manager

Read:

1. [README](README.md)
2. [Employer engineering case study](docs/EMPLOYER_CASE_STUDY.md)

Focus on:

- 20-model / four-split-family ensemble
- 561,607-row pooled OOF evaluation
- robust diversity evidence across folds and games
- confirmation-stage full-model feature diversity with fresh-fold validation gates
- 4.784× measured inference acceleration
- direct NFL / ESPN historical-data engineering
- resumable AWS GPU execution and research controls

### 10 minutes — ML engineer / applied scientist

Read:

1. [Current research status](docs/CURRENT_RESEARCH_STATUS.md)
2. [Research system and reproducibility](docs/RESEARCH_SYSTEM.md)
3. [Model card](docs/MODEL_CARD.md)
4. [Feature-diversity confirmation study](docs/FEATURE_DIVERSITY_STUDY.md)
5. [October research progress](docs/OCTOBER_RESEARCH_PROGRESS.md)
6. [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md)

This path shows how hypotheses move from integration checks to grouped validation, uncertainty analysis, ensemble testing, promotion, rejection, or retirement.

### Deep review — implementation and evidence

Inspect:

- `src/nfl_trajectory/` — maintained public implementation
- `tests/` — software and research-contract tests
- `research/` — public research evidence and preserved source archive
- selected notebooks under `notebooks/`
- machine-readable snapshots under `docs/results/`

## Validate the frozen public evidence

The repository includes a frozen publication manifest and validator.

From the repository root:

```bash
python research/publication/validate.py
```

The validator checks the registered public artifact hashes. It does **not** claim to retrain private models or reproduce restricted competition inputs.

## Review the aggregate notebook

Open [research/RESEARCH_REVIEW.ipynb](research/RESEARCH_REVIEW.ipynb).

Its inline Plotly figures are generated from registered public aggregate evidence rather than private row-level predictions.

The optional renderer:

```bash
python research/publication/render.py
```

requires Matplotlib in addition to the project environment. Rerendering under a different library stack can legitimately change derived notebook/HTML bytes; frozen publication hashes must not be rewritten merely to hide environment drift.

## Inspect the maintained implementation

The maintained public package lives in:

`src/nfl_trajectory/`

The package and tests demonstrate selected reusable project patterns while intentionally excluding complete private competition runners and fitted artifacts.

The project uses Python 3.11 for the maintained public environment, with pinned dependencies and quality tooling defined in `pyproject.toml`.

## What is reproducible publicly

The public repository supports review of:

- metric definitions
- grouped validation design
- selected preprocessing and modeling components
- public-safe tests
- direct-source provenance patterns
- experiment lifecycle design
- aggregate model / ensemble evidence
- system architecture
- CI and publication-integrity controls

## What remains private by design

The following are intentionally excluded:

- raw competition data
- raw third-party response archives
- fitted model weights
- large checkpoints
- credentials
- private AWS object paths
- row-level private predictions
- complete private runners
- unreleased active feature combinations

That boundary is part of the project design, not a missing-file accident.

## Canonical environment

The live research source of truth is AWS SageMaker.

GitHub is the versioned, employer-facing implementation and evidence layer derived from validated milestones.

Publishing to GitHub does not:

- mutate AWS experiment state
- retrain a model
- alter checkpoint lineage
- submit a candidate externally
- convert an experimental challenger into a promoted model

## Research archive

[research/README.md](research/README.md) explains the evidence hierarchy and the distinction between:

- software correctness
- individual-fold validation
- pooled OOF
- ensemble promotion evidence
- runtime/performance evidence
- private submission evidence
- experimental work still awaiting its required gates

Historical experiment packages under `research/workspace/` are preserved source mirrors. They are not a reorganized public training application and should not be treated as turnkey replacements for the AWS-canonical workflow.

## Review standard

A reviewer should be able to answer the following from the public repository:

- What is the current accepted system?
- Which metrics support it, and on what evaluation population?
- What did the project owner build end to end?
- Which hypotheses improved standalone quality but failed ensemble promotion?
- How are leakage and point-in-time correctness handled?
- How does long-running GPU work resume after interruption?
- Which execution failures became regression tests?
- How are public artifacts separated from private/restricted research state?
- Which results were measured versus projected or still experimental?

That reviewability is treated as a first-class engineering deliverable.
