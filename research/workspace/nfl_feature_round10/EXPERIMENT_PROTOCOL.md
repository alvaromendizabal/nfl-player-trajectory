# Frozen protocol · Grouped goal-frame feature ablation

## Question and source of the hypothesis

Round 9 reports nontrivial temporal differences but little prediction response through the relationship pathway, with nonzero gradients and changed weights. The proposed model makes four observed relationship groups separately available to the decoder and lets requested forecast time condition attention. Context normalization is shared by both arms. These choices are hypotheses, not findings from the audit.

The primary experiment tests only the **six additional numerical goal-frame values**. It does not separately attribute group pooling, query conditioning, normalization or model capacity. The architecture is a compact experimental implementation, not a reproduction of a winning competition model.

## Inputs and availability

Use only the existing Round 8 fold-2 tensors: 851 selected plays, 26,039 training forecast rows and 5,886 evaluation rows from 14 later games. Original chronological game separation remains in force. All labels and the tree reference are reused unchanged.

The eight Cartesian directed pair channels and 10 node channels remain. Add the six unchanged Round 9 goal channels: along/across separation, along/across relative velocity, peer-minus-self goal distance and closing speed. Fixed scales are 20 yards for distances and 10 yards/second for velocities. Near-zero goal direction (<0.1 yard) is masked. Missing pair observations are never filled across time. Self relationships are masked. Ball landing point, role and requested frame IDs are supplied task inputs, not future player targets.

Four groups use supplied side/role: same-side, opponent, targeted receiver and passer. They overlap intentionally; they do not identify defensive responsibility. Group aggregation takes the union of observed support across the observed window, with channel ages retained. It does not fabricate current player positions. Base normalization uses training rows only; evaluation arrays are not opened for feature scaling, profiling or training.

## Arms

- `cartesian`: eight actual Cartesian pair channels plus six numerical zero channels; **all goal validity masks and ages are retained**.
- `goal`: same inputs, except actual values fill the six goal-frame channels.

Both use the same 14-channel temporal encoder, group masks, query-conditioned dot-product attention, non-affine LayerNorm on each group context, player encoder, 72 base features, and decoder. Zeroing goal values isolates their numerical contribution conditional on availability; it does not estimate the contribution of their masks. Shared parameters include weights associated with those zeroed values in the control.

## Fitting and cost

Two independent models, one selected fold, no ensemble. Each jointly predicts two residual coordinates relative to the existing constant-velocity baseline. Error differences in these residual coordinates preserve the coordinate-RMSE metric. No future coordinates enter features, no clipping or removal of requested evaluation rows occurs.

Same seed 20260912, same initial weights, AdamW lr 0.001 / weight decay 0.0001, same shuffled play order, batch size 8, 24 epochs, gradient norm cap 5.0 and fixed final-step selection. Save optimizer, model, RNG, step and objective history every 25 steps and at epoch boundaries. A training batch's denominator is fixed using total training coordinates / batches, avoiding equal weighting of unequal-length plays.

Preflight 90 seconds; feature smoke 120; full extension preparation 360; runtime checks 180; profile 180; each scientific arm 360; evaluation 180; replay 180; report 60. A 16-step training-only profile on eight largest training examples must conservatively project <=300 seconds per arm before either scientific fit. That is a cost guard, not a runtime promise. It also requires finite nonzero goal-input gradients after disposable updates, without choosing a minimum gradient magnitude. No budget escalation or additional fold is automatic.

## Evaluation and decision

Wait for both complete arms. The sole primary contrast is goal minus Cartesian coordinate RMSE on all requested evaluation rows. Require >=1% relative gain and a paired whole-game 95% upper difference bound below zero (10,000 resamples, seed 20260912). Compute pooled errors from all coordinates, not the mean of play/fold RMSE.

Before considering more compute, the goal arm must also be no worse than the preserved tree on the same rows. This is an additional performance hurdle, not a causal attribution to the new features. No role- or horizon-specific model selection is permitted from diagnostic plots.

These are repeatedly inspected games, one seed and a small subset of the competition. The interval is not a correction for all preceding project decisions. A pass earns consideration for replication, not promotion, a leaderboard claim, or release. Failure stops the exact treatment unchanged.

## Recovery and publication

Old source, arrays, targets and weights stay read-only. New extension arrays have exact replay/checksum receipts. Changed source, inputs or numerical environment stop reuse. Independent model reloads must reproduce the training probe; fresh-process evaluation replay must reproduce both complete predictions and summary with zero optimizer steps. The export allowlist excludes raw data, row keys, predictions, model weights and private feature arrays.

No AWS provisioning, IAM change, S3 write, Git commit, push, merge, package download or Kaggle submission is performed by this package. A runtime cache miss is a stop. The project repository and earlier kit folders remain required.
