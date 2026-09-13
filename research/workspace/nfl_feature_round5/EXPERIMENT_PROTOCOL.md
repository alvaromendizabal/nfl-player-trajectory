# Round 5 · Direct-state temporal replication

**Feature engineering remains open.** This is validation of an existing feature representation, not a new family, model search, or production release.

## Hypothesis and motivation

Round 4's 62 direct observed-state fields reduced first-fold fixed-tree coordinate RMSE from 0.6831604368 to 0.6676231770. The six-look adjusted upper game-bootstrap difference endpoint is only −0.0000108868, so the next high-value experiment is temporal replication rather than adding more covariates to the same repeatedly inspected 16 games.

The winning author's public solution separately represents observed dynamic tracking and static information, including roles, horizon and terminal/landmark positions. This motivates preserving information explicitly, but does not imply that our tabular representation matches the winning architecture or gains. The author also used substantial augmentation and ensembling; this kit does not assert that features alone explain the leaderboard gap.

Primary sources reviewed:
- Author's solution: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
- scikit-learn cross-validation guidance: https://scikit-learn.org/stable/modules/cross_validation.html

## Frozen representation and availability

Control: the original 72 base fields. Treatment: those exact fields plus the same 62 direct-state fields from Round 4. The definitions in `origin_features.py` and `state_features.py` are copied byte-for-byte. They cover observed player state, supplied landing-point displacement, synchronized receiver/passer context, multiwindow motion/receiver summaries, role and input-support flags. No ID is used as a numeric predictor. Observed input and query keys only enter the feature builder; no target, future position, saved error or learned player prior is a feature argument.

The old builder also returns its 26 geometry candidates for historical parity; they are never selected into the model matrix. There are **zero new feature definitions** this round. This avoids quietly changing the hypothesis while calling the study replication.

## Data and split reuse

Keep the original 1,024 selected plays and three chronological game-fold definitions. Do not choose a new sample, change labels, reread raw output CSVs, add a season, or touch the reserved outer holdout. The first-fold predictions and four original coordinate models are independently forward-replayed. They are not fitted again.

Reuse Round 4 per-play state files read-only. Only missing states for later selected plays are built. The 32-play smoke uses fold-2 training plays and independently reconstructs their observed features, including equality after each parent shard's historical float32 or float64 storage conversion. Equality must remain exact; no tolerance relaxation.

Screen only near-constant columns (range ≤1e-10) on the individual fold's training rows. Evaluation distributions, targets and errors cannot select columns. Both arms use identical training/evaluation rows and ordering. Forward chronological order, expanding training sets, unique keys, complete declared game coverage, and disjoint evaluation games are verified.

The fold definitions were previously used in the research program. No statement of independent confirmation or cross-season stability is warranted. Expanding training can contain earlier folds' evaluation games; the evaluation sets themselves must not overlap, and each model's training must precede its evaluation dates.

## Estimator and exposure

Same HistGradientBoostingRegressor as Round 4: squared-error loss, learning rate 0.06, 120 iterations, max leaves 15, minimum leaf size 30, L2=1.0, 127 bins, no early stopping, warm start, seed 20260911. No algorithm/parameter search, clipping, ensemble, role-gating, row weighting, or augmentation.

Fit control x/y and direct-state x/y on each later fold: **eight new coordinate fits maximum**. Existing first-fold models are read-only. Save verified checkpoints every 30 iterations. Checkpoint signatures bind numerical environment, settings, source, data, feature manifest, fold, arm, axis and training keep-mask. Replay cannot fit a missing or incomplete model. A fresh-process replay checks full saved predictions, keys and targets.

## Staging and futility

Preflight → 32 training-play smoke → missing-state preparation → fold 2 → fold 3 if not futile → no-fit replay → pooled review → report.

After fold 2, a relative deterioration of **5% or greater** prevents fold 3. This is a cost guard, not an acceptance criterion. Modest negative results are retained and tested on fold 3 without retuning. If stopped, report the available evidence and do not modify the cutoff.

Limits: preflight 150 seconds, smoke 180, preparation 360, each four-fit fold 240, replay 180, standalone summarization 120. Two CPU threads; 15-second heartbeats. Budgets are hard stops, not runtime promises. Reuse successful work across repeated commands with unchanged source/data/settings. A deterministic failure requires diagnosis, not unchanged retries.

## Evaluation and decision

Coordinate RMSE is sqrt(sum((x−xhat)^2+(y−yhat)^2)/(2N)). Pool by summing squared errors and rows, never averaging fold RMSE.

A paired game-block bootstrap uses 10,000 resamples, fixed seed 20260911, and quantiles .05/(2×10), 1−.05/(2×10). The nominal ten-look accounting covers the prior six declared comparison slots plus four new exploratory summaries: fold 2, fold 3, later-only pooled, all-available pooled. It is deliberately not presented as a guarantee of simultaneous coverage over the whole adaptive project. Reuse of games, model dependence across expanding folds, and feature selection from earlier results remain limitations.

Primary gate, evaluated only when both later folds finish:
1. Both later folds reduce RMSE.
2. Their pooled relative reduction is at least 1%.
3. The later-only adjusted upper paired-game difference bound is below zero.

First-fold discovery performance cannot rescue a failed later-only gate. All-fold pooling, horizon/role slices and leave-one-game-out gain ranges are descriptive. Leave-one-game-out ranges are **not confidence intervals** and cannot be used to drop games or choose a model. A pass earns review for a controlled deployment-path integration, not automatic acceptance/submission. A failure keeps research open and requires diagnosis before expanding this interface.

## Publication and privacy

This kit and its notebooks stay outside the canonical repository. Its launcher may populate a separate cached script environment through the existing verified lock, offline. It does not change `.venv`, download interpreters/packages/data, run the final fitter, commit, push, merge, launch cloud resources, or submit to Kaggle. No first-fold source/artifact migration is performed.

Report export is an explicit allowlist of aggregate JSON receipts. It omits raw tracking, per-row keys, predictions, targets, feature matrices and model blobs. Source/report hashes enable review without publishing competition data.
