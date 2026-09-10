# Chronological confirmation and target-statistic ablation

The initial temporal model changes representation and architecture together. Its
development score cannot identify which features helped. The fixed equal blend
also needs temporal confirmation before it can become a deployment candidate.

This study uses the three existing chronological folds entirely inside the
original 192-game training partition. Their training/evaluation sizes are
94/41, 135/28 and 163/29 games. Neither the 32-game development partition, the
previously inspected reserved set nor Kaggle outcomes select these fits.

Each fold independently fits its role-ridge baseline and chronological historical
encoder. All full-training baseline predictions, target residuals and historical
statistics in the reusable observed tensors are replaced before training.
Encodings exclude every outcome on the current training date. Evaluation uses
the frozen earlier-training table. Tensor geometry is observed-only and can be
reused without fitting it again.

Two models per fold share the exact architecture, parameter count, seed, batch
order, reflection draws, optimizer, schedule and 40-epoch final EMA selection:

| Variant | Target-derived input statistics | Counts and cold-start flags |
|---|---|---|
| Full representation | Player and role x/y motion-error means and dispersion | Retained |
| Without target statistics | The six channels are fixed to zero | Retained |

This isolates target-derived statistics from identity frequency and cold-start
information. It does not separately estimate the contribution of player versus
role statistics, and it does not ablate the role-ridge residual formulation.

The plan is written and hashed before any fit. Each arm has a separate checkpoint
containing the model, optimizer, EMA, RNG and batch cursor. Repeating a completed
arm verifies its checkpoint and keyed-error hashes without training another
epoch. All runs retain the original bounded training settings, UTC progress and
15-second heartbeats. A budget-stopped arm cannot enter the completed report.

The official coordinate RMSE is computed over every requested coordinate. Pooled
RMSE combines squared errors and row counts; it is not the average of fold RMSEs.
Paired uncertainty resamples whole games with the same 2,000 draws and seed 2026.
Every neural comparison and fixed 50/50 tree blend requires exactly the same
forecast keys as the preserved fold reference.

Before inspecting the completed fits, the decision criteria are:

- Claim a consistent benefit from target statistics only if full inputs improve
  all three fold scores and the pooled paired 95% interval excludes zero.
- Advance the full-input equal blend toward inference validation only if it
  improves every fold, reduces pooled tree RMSE by at least 1%, and its paired
  interval excludes zero. This is a research gate, not automatic deployment.
- Report failed or mixed criteria directly. No weight fitting, checkpoint
  selection, seed search or additional holdout evaluation is part of this study.

These are reused historical folds and one seed, so confidence intervals describe
the measured predictions conditional on this study. They do not establish
independent replication or a new Kaggle score. Other neural feature families and
reliability checks remain open even if these criteria pass.

```bash
uv run --locked scripts/temporal_research.py
uv run --frozen python scripts/evaluate_temporal_research.py
```

Run from a restored checkout whose manifests verify the completed data and
temporal inputs. Raw tracking and individual predictions remain private.
