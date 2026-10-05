# NFL Big Data Bowl 2026 — Start Here

This repository is the employer-facing, semi-reproducible record for an AWS-first player-trajectory forecasting research program.

## 10-minute technical review

Read these in order:

1. [README](README.md) — project scope and headline evidence
2. [Current research status](docs/CURRENT_RESEARCH_STATUS.md) — latest verified system and active milestone
3. [Research system and reproducibility](docs/RESEARCH_SYSTEM.md) — architecture, validation, observability, and public/private boundary
4. [Model card](docs/MODEL_CARD.md) — model family, evidence, limitations, and lifecycle
5. [October research review](docs/OCTOBER_RESEARCH_PROGRESS.md) — controlled experiment history
6. [External data engineering](docs/EXTERNAL_DATA_ENGINEERING.md) — direct-source acquisition and point-in-time controls
7. [Research evidence archive](research/README.md) — evidence hierarchy and preserved public artifacts

The strongest completed local system is a 20-model, four-split-family ensemble at **0.4631723213 pooled OOF RMSE over 561,607 rows**. The strongest recorded private submission is **0.46487 RMSE**. These are different evaluation settings and are reported separately.

## Validate the public publication

From the repository root:

```bash
python research/publication/validate.py
```

This standard-library validator checks registered public hashes, publication structure, source-archive provenance, credential-like patterns, notebook structure, and aggregate RMSE arithmetic. It does **not** retrain private models.

## Review the executed aggregate evidence

Open [research/RESEARCH_REVIEW.ipynb](research/RESEARCH_REVIEW.ipynb) for the frozen aggregate research review. Its public figures are generated from registered aggregate evidence; the notebook is an evidence view, not an inference service.

The later October narrative in `docs/` extends the public-safe research record beyond the frozen aggregate notebook without publishing private row-level predictions or active experiment recipes.

## Inspect the maintained implementation

- `src/nfl_trajectory/` — maintained public implementation
- `tests/` — software and research-contract tests
- `notebooks/` — selected project notebooks
- `research/workspace/` — preserved public source mirrors and experiment materials
- `docs/results/` — selected machine-readable public snapshots

Historical experiment packages under `research/workspace/` are preserved evidence, not a single reorganized training application. Public copies intentionally omit private input contracts, fitted objects, raw competition files, private row-level results, and private recovery receipts.

## Reproducibility contract

The public repository is intentionally semi-reproducible.

A public reviewer should be able to inspect:

- the forecasting architecture and selected source implementation
- the official metric and grouped-validation contract
- public experiment protocols and negative-result decisions
- aggregate OOF/private evidence without conflating them
- direct-source acquisition architecture and leakage controls
- test, CI, checkpoint-lineage, and publication-validation patterns
- research-system engineering and measured performance outcomes

A public clone is **not** expected to reproduce private competition scores without the authorized data, private fitted weights, exact active feature combinations, and corresponding environment.

## Canonical environment

AWS SageMaker is the live source of truth for research, data preparation, training, validation, checkpoints, telemetry, and immutable run evidence.

GitHub is the durable employer-facing code and reproducibility layer derived from validated AWS milestones.

Kaggle is used only for external submission delivery/status where required.

A GitHub merge does not retrain a model, mutate AWS, or submit a candidate.

## Public/private boundary

Public GitHub excludes:

- raw competition data
- raw third-party response archives
- fitted private weights and large checkpoints
- credentials and private cloud locations
- complete private execution runners
- unreleased competition-specific active feature combinations

This boundary preserves reviewability and scientific provenance without distributing restricted data or active competitive IP.