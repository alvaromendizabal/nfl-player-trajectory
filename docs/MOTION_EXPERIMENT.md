# Adaptive investigation of the motion-feature signal

Declared after the completed 30-fit domain screen, before any fit below.
The parent screen identifies motion as helpful in all three chronological folds
and in both simultaneous-interval family comparisons. This investigation asks
which parts carry that signal. It is explicitly adaptive and uses the same
development folds; it cannot supply an independent generalization estimate.

Partition the 57 motion candidates into instantaneous velocity/reliability (5),
body-facing alignment (5), smoothed observed state (30), recent maneuver dynamics
(9), and query-time physical displacement hypotheses (8). Freeze the partition
before fitting. Reuse each fold's verified feature matrix, training-only screen,
and normalization. The existing control and complete-motion fits are exact
references and must not be retrained.

Fit each of five subfamilies alone and remove each from the complete motion bank:
ten new arms per fold. Use the parent's identical 463-input, 31,842-parameter
correction network, initialization, row order, optimizer, and twelve-epoch final
checkpoint. Zero excluded columns after training-only standardization. Do not
select a validation epoch, learning rate, seed, or blend weight.

Report all ten comparisons with Bonferroni simultaneous 95% intervals from
10,000 paired whole-game bootstrap replicates. Also report every fold's point
estimate and the inherited complete-motion-versus-control comparison. A
subfamily merits an end-to-end representation experiment only when both its
addition and removal effects favor inclusion in the pooled comparison and in
all three folds; intervals distinguish strong from uncertain evidence. A
negative result is conditional on the frozen neural representation and fixed
small correction head, not proof that the football mechanism is irrelevant.

As a declared secondary diagnostic, compare the complete-motion model's fixed
50/50 tree blend with the already completed attention/tree blend. Do not choose
an ensemble weight or evaluate the original development games, reserved
holdout, or Kaggle. Checkpoint every epoch and reuse matching completed arms.
The feature-research completion gate remains open.
