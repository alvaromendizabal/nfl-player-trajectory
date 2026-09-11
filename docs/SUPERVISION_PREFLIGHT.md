# Training-only supervision preflight

This is an engineering prerequisite to the new velocity-isolation experiment, not a new feature result or trained model. The existing notebook environment and verified inner_1 cache are reused. The feature/representation completion gate remains **open**.

## Research question

Can the coordinate-only and velocity-supervised arms use precisely the same deterministic training-play order and uniform-coordinate objective, without dropping long plays or allowing evaluation labels to influence normalization? A successful preflight establishes that contract; only a later matched fit and ablation can establish predictive value.

The rationale and domain evidence remain in DOMAIN_RESEARCH.md, MOTION_TARGET_AUDIT.md and MOTION_SUPERVISION_EXPERIMENT.md. No new external data, new academic performance claim, or assertion that prior features are exhausted is introduced here.

## Implementation

`src/nfl_trajectory/supervision_plan.py::training_plan` rejects evaluation samples before reading their labels, orders plays by stable game/play identity, and produces an epoch permutation from NumPy SeedSequence([seed, epoch]). Both arms must consume that same mapping. Seed is 2026. A batch size of 64 is a preflight candidate, not a throughput-approved scientific schedule.

The plan retains all training rows, including frames beyond 48. It computes shared-axis velocity RMS only from supported training labels. Finite differences follow the maintained real-frame and stale-endpoint contract. It does not compute validation metrics, fit a model, select feature families, or alter cached samples.

For K batches, fixed coordinate and velocity loss denominators are the respective full-training coordinate counts divided by K. This avoids weighting a short final batch or short plays more heavily merely because their local means are averaged equally. The same exposure, initialization, reflection mapping, optimizer and final-EMA policy remain mandatory across arms.

## Tests and stop conditions

Tests cover reversed input order, no mutation, duplicate plays, rejection of validation samples before label access, long forecasts, gaps, exact denominators and direct RMS parity. Private execution must additionally verify the cache SHA256, expected 4,951 training plays / 94 games / 193,452 rows, 368 training rows beyond frame 48, and an independent private S3 write/read-back receipt.

The complete real-data training/evaluation loop, training-only throughput selection of epochs and learning-rate schedule, reflection cursor, matched-arm checkpoint round-trip and final keyed predictions remain separate gates. A JSON receipt round-trip is **not** a trained-model checkpoint recovery test. No scientific fitting may be inferred from a passed preflight.

## Research remains open

After the prerequisite passes, the immediate comparison remains coordinate-only versus velocity-only auxiliary supervision. Additional folds depend on the predeclared metric/uncertainty gate. Acceleration tails, long-horizon displacement parameterization, learned defender-receiver relationships, role-conditioned arrival, and eligible historical-season alignment remain independent avenues. Do not repeat the rejected fixed soft-coverage ridge interface or call these mechanisms exhausted because one encoding underperformed.


## Executed AWS prerequisite — 11 September 2026

The actual `nfl-trajectory-dev` notebook and locked CPU research runtime were
verified. All **55 supervision/data/model/recovery tests passed** in Python
3.11.16, NumPy 2.4.6 and Torch 2.8.0+cpu.

The private training-only plan retains 4,951 plays, 94 games and 193,452 rows,
including all 368 rows beyond frame 48. Velocity support is 100%; the shared-axis
training RMS is 4.0287780777 yards/second. The candidate 64-play batch plan has
78 batches, with a fixed coordinate denominator of 4,960.3076923 per batch.
Batch size is still a candidate, not a throughput-approved scientific schedule.

Both width-96 synthetic arms were interrupted after two updates, uploaded to the
existing private S3 project bucket, downloaded to an independent directory and
resumed in a fresh process through update six. Model, EMA, optimizer, RNG, cursor
and loss counters matched the uninterrupted state exactly. The four input
checkpoint files totaled 9,185,502 bytes. This is a real S3 model-state recovery
proof, but uses synthetic data and does not establish scientific predictive value.

The proof runner is `scripts/check_supervision_recovery.py`. Run `--stage prepare`
and `--stage verify` in separate processes under the locked motion-supervision
runtime, with `--directory` inside `artifacts/quality`. Between stages, upload
the two `upload/<arm>/` generations, independently download them to
`download/<arm>/`, and record their exact bytes, SHA256, S3 key and version in
`transfer.json`. The runner validates that receipt and the downloaded generations
before continuing either model. No credentials or tensors are published in Git.

[Measured acceptance evidence](results/supervision_aws_readiness.json) records
provenance and remaining gates. **No new scientific fit, RMSE, or feature-maturity
claim** follows from these checks. The real-data training/evaluation loop and
training-only throughput selection must precede the fixed matched one-fold study.
