# Frozen Round 3 protocol

## Objective
Try to close the gap toward the historical 0.46340 private leaderboard target by testing meaningful representations. Feature engineering remains open. This is a bounded transfer test in the same estimator *family* used by the repository's submitted tree model, not that exact trained model or a guaranteed improvement.

## Evidence-driven changes
Preserve the full arrival control after Round 2 removal tests. Do not repeat its failed turning, braking, orientation or earlier-origin mixture. Increase the original-origin sample from 256 to 1,024 plays to reduce dependence on a tiny sample, without using errors or targets to select additions. Preserve the original 256 base X/y rows; do not refit any prior ridge model. Their six control/arrival models are read and replayed to verify the baseline lineage.

## New representation
Four frozen partner slots: two opponents, one teammate, and targeted receiver. Partners are selected by most recent joint observation, then separation, then stable ID to resolve ties. Partner identities do not switch across lags. No slot is a purported coverage assignment. Five observed lags: 0, 2, 5, 10, 19 frames before the prediction origin. Eight channels: dx, dy, relative vx/vy, separation, closing speed, lateral relative speed and velocity alignment.

160 numeric channel values + 160 channel-availability indicators + 4 joint-observation ages = **324 candidate columns per view**. These reuse underlying tracking signals; they are not 324 new independent facts. Their core ideas overlap existing repository representations. The new work is the matched terminal-versus-history interface and the controlled fixed-tree test, not a claim to have invented temporal relationships.

History uses actual lagged observations. Terminal control repeats the last jointly observed pair vector at each available historical slot. A channel is valid only when it is available both at that historical slot and at the terminal joint frame; this mask is identical in both arms. Both arms get the same terminal ages. Invalid values are zero placeholders with explicit masks. Missing frames are not filled and player IDs are not predictors. No future coordinates, target encodings, outside data or fitted parent predictions enter these inputs. Earlier observed data is intentionally replaced by terminal information in the control; it is legal information available at the prediction origin, not a claim of causal online processing at every historical slot.

## Four matched arms
A: 72-column existing motion/context control.
B: A + 24 existing role-conditioned arrival columns (96 columns before screening).
C: B + 324 terminal-pair columns.
D: B + 324 pair-history columns.

All use scikit-learn HistGradientBoostingRegressor 1.8.0 with the repository's existing final-fit dependency lock. Fixed settings: squared error, learning rate .06, 120 iterations, 15 leaves, 30 minimum leaf rows, L2 1, 127 bins, seed 20260911. No hyperparameter search, ensembling or automatic early stopping. This is intentionally smaller than the 128 GiB full-bank refit; **never run scripts/fit_final.py for this milestone**. Only its dependency lock is reused. Two coordinate estimators per arm; eight estimators in the first fold.

Training-only constant-column screening. C and D share the union of training-variable columns so they use the same retained dimensions. Same labels, rows, ordering, model settings and fixed final boosting exposure in every arm. Capacity is matched by hyperparameters; added information can still change effective statistical complexity. No validation score chooses a tree checkpoint.

## Sample and leakage controls
Only games present in the verified Round 1 training-side dataset are eligible (94 games in the real contract). The former 256 plays remain. Add 768 by fixed-key hashing and round-robin allocation across games. No outcome, horizon, role or prediction error guides selection. Retain every requested player/frame in each selected play. Freeze the exact chosen play IDs and three original chronological game splits before reading new labels. All raw files are checked against the supplied SHA256 inventory. New targets use complete original-origin output rows, not augmentation. Private held-out games, other seasons and Kaggle test data are excluded.

The game splits have been repeatedly inspected. Larger samples within them are not untouched confirmation. Three predeclared contrasts: B-A, C-B, D-C. Compare coordinate RMSE sqrt(sum(dx^2+dy^2)/(2N)); residual-coordinate errors equal absolute-coordinate errors because the same reference is restored to predictions and targets.

## Stage decision
Execute **fold 1 only** now. Continue eligibility requires at least 1% improvement and a negative upper paired-game percentile bootstrap bound adjusted for six planned looks (three first-fold and three potential pooled comparisons), 10,000 resamples. This is an exploratory decision rule, not a formal adaptive-study coverage guarantee. Stop if none passes. Return the first-fold report even if one passes. No notebook automatically runs folds 2/3. Later confirmation, if separately reviewed, must use unchanged settings, all four arms, every eligible validation row, pooled evidence and improvement in at least two of three folds. The runner refuses later folds when fold 1 failed. Later pooled reporting is not implemented in this first-fold milestone; do not infer it from separate fold summaries.

Horizon/role slices are descriptive only. No dropping difficult rows, post-hoc role gating, threshold tuning or new blend weights. Negative findings stop this exact interface from scaling, not all learned temporal relationship research.

## Recovery and budgets
Existing Python 3.11 notebook environment remains unchanged. Runtime setup defaults to offline reuse of the existing uv cache. An explicit `runtime --online` action may retrieve only the versions in the existing lock into an isolated uv script cache. No new interpreter is downloaded. Exact existing lock Git-blob verification precedes use.

Command caps: preflight 120 s; runtime 180 s; key indexing 180 s; 32-play smoke 180 s; preparation 600 s; one fold 480 s; replay 180 s; report 60 s. These are hard limits, not runtime forecasts. Two CPU threads. No AWS API, new cloud instance, GPU, Git write, Kaggle download or submission. User's existing Studio instance still incurs its ordinary running charge. No dollar cost estimate is asserted.

Per-play feature files are atomic and hash verified. Model checkpoints every 30 trees have source/data/environment signatures, immutable hash-named blobs and atomic pointer receipts. Successful checkpoints survive later failures. Fresh-process numerical replay refuses to fit missing work. An interrupted model can continue the exact same data, hyperparameters and source; never alter them to resume. A completed model is not retrained. Internal model pickle files are only self-generated local artifacts; never load unknown pickles or substitute someone else's files.
