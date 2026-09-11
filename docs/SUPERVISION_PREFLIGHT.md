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
