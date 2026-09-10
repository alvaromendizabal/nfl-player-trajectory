# Reconstructed motion-label prerequisite

This is new source, not recovery of the missing motion-supervision implementation.
Feature engineering remains open. No trained weights or score improvement are
claimed by this audit. It prepares the next coordinate-only versus velocity-only
auxiliary-supervision experiment.

The historical joint-supervision experiment recorded a 5.18% matched improvement,
but cannot separate velocity from acceleration and lost its trained checkpoints.
The [first-place training notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)
motivated that hypothesis. Reconstructing labels correctly is a low-cost prerequisite
before rebuilding any neural model. Future coordinates may supervise training;
they must never enter observed features, normalization of inference inputs, or
the model forward path.

## Frozen audit contract

- Use only the hash-verified `inner_1` sample cache already in the independently
  downloaded private soft-coverage backup. Verify its original hash before loading.
- Recover canonical future displacement as baseline displacement plus residual
  truth. Difference at 0.1 seconds within player and play, using actual frame IDs.
- Require consecutive predecessors. Frame zero is the zero displacement only
  when the player's last observed endpoint has zero age. Mask gaps and stale
  endpoints; do not interpolate targets, truncate horizons or remove hard rows.
- Compute velocity and acceleration scale from training labels only. Use a
  shared x/y RMS for each vector family, floored at 0.1. Reject validation samples
  passed to the scale fitter. Preserve reflections and all row keys.
- Audit all 7,054 cached plays in at most 300 seconds on local CPU. Log progress
  every 1,000 plays and heartbeat every 15 seconds. Save aggregate support,
  magnitude quantiles, source/input hashes and train-fitted scales atomically.
- Fail on malformed keys, duplicate labels, inconsistent player identity or
  nonfinite labels. Expose large derivatives for inspection; do not silently clip
  them or call label availability evidence of predictive value.
- Re-run the saved audit to verify unchanged report hash and mtime. The report
  and tests are public; individual training labels stay in the existing private
  input checkpoint and are reconstructed deterministically.

## Next experiment gate

After this prerequisite and model recovery tests pass, declare a new one-fold
matched coordinate-only versus velocity-only experiment. Freeze initialization,
architecture, rows, loss scaling, optimizer, exposure, seed and selection policy
before fitting. Persist and independently read back each completed arm before
the next fit. Acceleration-only and joint arms are later attribution questions,
not automatic extra runs. No Kaggle submission or holdout reuse is authorized by
this document. No feature-completion or employer-readiness claim follows from
this preparatory audit.
