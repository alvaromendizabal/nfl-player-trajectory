# Model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026 Prediction. Inputs are observed tracking, player roles, the supplied ball landing point, and requested forecast horizon.

## Current deep ensemble

The strongest verified competition-facing system is a **seven-model equal ensemble** built from five source-faithful game-grouped base models, one independently initialized seed-1 model, and one protected context-dropout model. Its post-competition private score is **0.46547 coordinate RMSE**, improving the prior five-base private score of **0.46615** by **0.00068**. The final first-place private score is **0.46340**, leaving a gap of **0.00207**. No official competition rank is claimed.

The five source-faithful base models produce **0.468143852 pooled OOF RMSE** over 561,607 retained rows. Every OOF row is predicted only by its excluded-fold model. Seed and context variants were not promoted as standalone systems when their fixed confirmation or uncertainty gates failed; they were retained as complementary ensemble members and only credited after an exact scored deployment improved the private metric.

### Current limitations

The complete first-place ensemble diversity has not been reproduced. The current system has one primary five-fold split family plus two additional Fold-0 diversity models. Checkpoint selection and repeated research inspection limit independence. The next experiment is a complete second game-grouped five-fold split family; it is pending and has no published metric yet.

## Historical feature-engineering system

The sections below document an earlier tree-based research path retained for provenance. Its development and reserved-holdout metrics are not pooled with the newer deep-model OOF or Kaggle private scores.

