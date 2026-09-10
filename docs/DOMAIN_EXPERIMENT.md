# Controlled domain-feature screening

Declared on 2026-09-10 before fitting any arm in this experiment.

The question is whether explicit football information corrects errors of the
saved chronological temporal models. This is a conditional feature screen, not
final training or proof that an architecture is optimal. No new Kaggle run,
development evaluation or reserved-holdout evaluation is part of this protocol.

Use the three existing expanding chronological folds, wholly within the original
192 training games. Load each fold's hash-verified final-EMA checkpoint and
fold-local tensors, including its earlier-date target encodings. Freeze that
network. Extract its existing decoder inputs and predictions on its own training
games and on the later evaluation games. Fitting a correction on in-sample
training residuals is intentional and identical in every arm; its possible
residual-distribution shift limits this screen's interpretation.

The four candidate families are motion-state reliability and recent dynamics;
landing-time arrival constraints and soft boundary-value paths; observed
receiver/opponent interaction histories; and field/pass geometry. Use the
organizer-provided landing point, role, and forecast horizon, with no future
player positions. Proposed motion paths are inputs, never forced outcomes.

Fit ten arms per fold: an existing-input control, four single-family additions,
all four families, and four leave-one-family-out ablations. Every arm has the
same total input width, random initialization, architecture, training order,
optimizer and twelve-epoch budget. Absent families are zeroed **after** train-only
standardization. The correction head has hidden widths 64 and 32 with SiLU,
zero-initialized two-coordinate output, and no dropout. Multiply the correction
by forecast seconds. AdamW uses learning rate 0.002, cosine decay to 0.0001,
weight decay 0.01, gradient clipping 1, seed 2026 and batches of 4096 rows.
Coordinate MSE weights all requested coordinates equally. No validation-epoch,
seed, learning-rate or ensemble-weight selection is permitted.

Screen constant and exact-duplicate candidate columns on training rows only;
retain a deterministic canonical representative. Record every rejection and
the training means/scales. Clip standardized inputs at ten standard deviations
as a predeclared numerical guard, and report evaluation clipping rates. Do not
remove or clip outcomes, drop difficult evaluation plays, or change the metric.

The predeclared joint candidate is all four families. It clears the screen only
if its pooled coordinate RMSE improves over the matched control by at least 1%,
all three fold differences favor it, and the paired whole-game bootstrap 95%
interval excludes zero. Assess each family from **both** its addition and its
leave-one-out comparison. For the eight family comparisons report Bonferroni
simultaneous intervals (99.375% individually), as well as raw effect sizes.
Do not silently select the smallest of ten scores as a validated winner.

Keep the saved tree and temporal errors as references. An equal-weight blend
with the tree is a declared secondary diagnostic. It does not select an arm or
authorize a submission. Report pooled row-weighted scores, per-fold scores,
role/horizon/cold-history slices, learning curves, and training error
concentration. Validation frames are never treated as independent bootstrap
observations. These reused development folds and a single seed cannot establish
an independent generalization result or the 0.46 target.

Persist every completed feature cache, screen, optimizer/RNG checkpoint,
prediction and source fingerprint. Resume only matching verified artifacts.
The feature completion gate remains open regardless of this screen's outcome.
