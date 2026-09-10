# Velocity supervision: a new matched reconstruction

Experiment ID: `velocity-isolation-20260910-v1`  
Phase: implementation and synthetic validation; **scientific fitting is disabled**.  
Source parent: `95c8f03ed0c3311834071f00add8f794792629ae`.

## Question and evidence boundary

Does velocity-only auxiliary supervision improve a shared motion representation relative to identical coordinate-only training? The historical joint velocity/acceleration study cannot be replayed because its exact implementation and weights are missing. This implementation is **new**, not a recovered version of that model. It has no NFL validation score yet.

The project research ledger records grouped temporal filters and auxiliary motion tasks in leading NFL solutions. Its prior controlled motion studies also support reliable motion representations, while the smoothed-input continuation and soft-coverage correction failed their respective gates. Those findings motivate this test; they do not prove that this encoding or auxiliary loss will help. This milestone does not reproduce the winning solution or close feature research.

Source basis: the pinned repository's `docs/DOMAIN_RESEARCH.md`, `docs/RECOVERY_STATUS.md`, `docs/MOTION_TARGET_AUDIT.md`, `src/nfl_trajectory/temporal_data.py`, `temporal_model.py`, and `motion_targets.py`. External author writeup referenced by that ledger: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution. The writeup's full body was not independently re-read for this implementation milestone. No external data are introduced.

## Information available to the model

Retain the maintained 20 observed frames, 30 dynamic channels, 20 static channels, and 12 directed player-pair channels. Keep observed-history masks, role/side embeddings, supplied landing geometry, requested horizon, and the fold-local role-ridge residual origin. Player IDs are alignment keys, not numeric predictors. The cache's historical encoders are inherited and were not reconstructed by the latest schema preflight.

`observed_batch()` strips future coordinate labels before collation. The model accepts an explicit whitelist of input tensors and rejects extra fields. `supervised_batch()` returns labels separately. The velocity head is **not an input** to coordinate prediction. Both heads share the decoder representation.

The new stem applies three per-signal grouped convolutions with dilations 1, 2, and 4 to the already observed window. Thirty signals plus presence each receive four filters (124 internal channels); a pointwise projection returns the maintained decoder's 48-channel temporal interface. Width-96 player attention and a continuous-time coordinate residual decoder are retained. The stem is not claimed identical to the missing historical stem.

## Labels, masking, and normalization

Reuse maintained consecutive-frame target construction: canonical displacement is baseline displacement plus coordinate residual truth. Velocity labels are backward finite differences over actual consecutive forecast frames at 0.1 seconds. They are interval velocities, not independently measured instantaneous velocities. Frame zero is anchored only when the last observed position is current; gaps and stale endpoints are masked, not interpolated. No future motion target is provided to the forward pass.

Fit a single x/y velocity RMS on training samples only, with the maintained minimum scale of 0.1. No validation statistic enters that scale. Acceleration is not used for supervision or normalization; the existing derivative utility may calculate it, but this model and loss do not consume it. The reported acceleration tails remain an unresolved research question; no clipping or label deletion is introduced.

The coordinate objective is coordinate SSE divided by the coordinate count. The treatment adds `0.1 * normalized_velocity_MSE`; the control uses weight zero and never reads auxiliary targets in its loss. Reported competition RMSE will be `sqrt(sum(dx^2 + dy^2)/(2N))`, not mean Euclidean distance or mean per-game RMSE.

Scientific batching must use fixed denominators derived from the fold totals and number of batches. Equal averaging of differently sized batch means would reweight short plays and is explicitly not the scientific contract. The loss API's local-batch defaults are for isolated synthetic tests.

## Matched arms and remaining experimental specification

Both arms have identical parameter counts, initialization, input representation, optimizer settings, dropout draws, training order, reflections, exposure, and final-EMA selection. Only auxiliary loss weight differs. Model width is 96, seed 2026, AdamW initial learning rate 0.001, weight decay 0.01, gradient norm cap 1, and EMA decay 0.98. Never select the best observed validation epoch.

`MatchedState` is a CPU training-step primitive, not a complete scientific trainer. Scientific epoch count, batch size, and learning-rate schedule deliberately remain unset. A later training-only throughput smoke must select one common exposure before validation access. A real-data cursor/order manifest, learning-rate scheduler, complete evaluation, and private per-arm S3 publication must be integrated and reviewed before fitting. This package cannot silently run an under-specified experiment.

Expected private input: `artifacts/temporal/research/inner_1/samples.pkl`, SHA256 `d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d`. The user's returned preflight reports 4,951 training plays / 94 games / 193,452 rows, and 2,103 validation plays / 41 games / 83,938 rows. All rows are retained, including the 368 training rows after frame 48 and maximum frame 94. The new code has been tested on synthetic long-horizon requests, not run against this private cache here.

These expanding inner folds have already been inspected in earlier research. A future positive result is internal evidence, not untouched confirmation, cross-season proof, or a Kaggle score.

## Persistence contract

The state payload includes model, EMA, optimizer, Torch dropout RNG, completed-step cursor, loss counters, seed, width, loss weight, and scale. Checkpoints use immutable content-addressed tensor blobs and an atomic pointer published last. Each blob is reread and hashed before pointer advancement. A publication interruption preserves the previous committed generation. Checkpoints reject different signatures, changed bytes, regressed cursors, and changed exact-replay runtimes. Only tensor/primitive payloads are loaded, using `weights_only=True`.

Tests copy a checkpoint to an independent local directory, delete the original, resume in a fresh process, and compare all state with uninterrupted training. That proves the synthetic **local** recovery contract. It does not prove remote durability. `remote_verified` remains false; an independent S3 upload/download verification and a real-data runtime resume test are still mandatory before scientific fitting. Old immutable generations are not automatically deleted.

## Running this milestone

On a prepared Linux environment, the locked command is:

```bash
uv run --locked scripts/motion_supervision.py --self-test
```

The command accepts self-test mode only. It does not read competition data, install packages itself, contact AWS, create jobs, push Git, publish notebooks, or submit to Kaggle. `uv` may download the declared runtime when invoked by the operator or CI. Do not install this stack in the nearly full CloudShell home directory.

The dependency declaration and lock are copied exactly from the maintained `scripts/train_temporal.py` runtime: Python 3.11 and CPU Torch 2.8.0. The local validation environment is recorded separately in the implementation receipt; it is **not** represented as a locked-runtime replay.

The runner bounds the pytest stage to 180 seconds, prints UTC heartbeats every 15 seconds, stops its test process group on timeout, and writes per-run logs/XML/JSON plus `artifacts/quality/motion_supervision.json`. The new CI step has a five-minute outer bound, inside the existing 15-minute Quality job. All existing Quality checks remain enabled.

Tests cover analytical derivatives, gaps/stale endpoints, training-only normalization, future-label isolation, player permutation, padding, reflection, frame-94 requests, parameter/init parity, active auxiliary gradients, short synthetic learning, full-width execution, checkpoint corruption, publication interruption, and exact fresh-process restoration. Synthetic optimization is an engineering test, not an NFL experiment.

## Gates and next decision

Before a scientific fit: preserve source on a reviewed branch, pass locked-runtime and repository-wide CI, integrate the real-data runner, freeze exposure using training-only throughput, and verify private remote persistence. Proposed first-fold cap: 600 seconds per arm, with epoch-level checkpoints. The runtime of real-data fitting is not yet measured; no AWS dollar estimate is asserted.

The predeclared candidate continuation gate is at least 1% lower coordinate RMSE and a paired whole-game bootstrap upper bound below zero, with both matched arms complete and independently restored. Do not add folds after incomplete fits or a failed gate. Role/horizon diagnostics describe failures; they are not permission to tune post-hoc gates. Acceleration, learned assignments, long-horizon parameterizations, and older-season alignment remain separate open research avenues.

This stage changes no canonical notebook outputs because it produces no scientific result. Historical 0.70090 Kaggle private, 0.64160270 preserved internal blend, and 0.62708199 incomplete historical evidence remain distinct and unchanged.
