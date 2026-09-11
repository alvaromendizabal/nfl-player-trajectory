# Temporal interaction research — representation prototype

11 September 2026. **Feature research is open; predictive benefit is unmeasured.**
This milestone builds and tests observed inputs. It does not train a model,
recover missing historical weights, or claim that the 0.46 objective is reached.

## Hypothesis and primary evidence

Retaining pair relationships across the observed window may preserve relative
motion and matchup changes that terminal geometry or early pooling loses.
Song et al., *Decoding Defensive Coverage Responsibilities* (2026), sections
3.1–3.3, use temporal/agent attention and receiver–defender pairing. They also
use coverage annotations and contextual fields not established as available
for our inference. We use the representation hypothesis, not their annotations,
trained weights, classification scores, or post-throw observations.
Source: https://arxiv.org/html/2603.25901v1

Dutta, Yurko and Ventura, *Unsupervised Methods for Identifying Pass Coverage*,
motivate relational movement information. Their coverage task is not our
post-throw coordinate-RMSE evaluation. No responsibility label is fabricated
from proximity or presented as calibrated matchup probability.
Source: https://arxiv.org/abs/1906.11373

## Implemented input contract

`src/nfl_trajectory/temporal_edges.py` returns a tensor with separate source-player,
destination-player, observed-frame, and channel axes. It uses eleven channels:
relative position/velocity, separation, closing/lateral motion, velocity alignment,
bearing/separation rates, and same-side status. Channel-specific masks distinguish
unavailable measurements from real zeros. Self edges are masked. Frame gaps are
not filled or compressed; differences require adjacent joint observations.
Physical scales are constants, not estimates from evaluation data. The adapter
reads only observed history, observation masks, terminal observed anchors, and
side; it does not read labels, requested future rows, or fitted baseline outputs.

This is **not** a rerun of the failed fixed soft-affinity/ridge correction.
The current prototype does not generate learned matchup probabilities or yet
connect an edge encoder to the forecasting model. Existing experiment code,
settings, locks, checkpoints, and the three canonical historical notebooks
are unchanged. The focused notebook `03_interaction_research.ipynb` contains
executed synthetic diagnostics and an optional real training-only smoke path.
Synthetic figures must never be described as NFL private-data results.

## Current acceptance and bounded AWS action

Local tests cover known-motion units, masks, gap handling, missing-value poison,
stationary/coincident players, reflection parity, permutation equivariance,
translation, observed-time causality, finite output, bounded size, and label
exclusion. They establish software behavior, not feature-value evidence.

The next AWS action first inspects the **existing** job
`nfl-motion-scientific-20260911-055510-d265d9d`. The read-only collector downloads
at most 64 MiB, checks account/source identity and immutable hashes, retains
errors/checkpoints privately, verifies matching row keys and reported RMSE, and
never starts or stops cloud compute. Byte verification is not numerical model
replay. Its job status is a timestamped observation, not a perpetual live status.
The notebook recomputes the paired game-bootstrap interval when saved evidence
is available. Independent numerical restoration still precedes model promotion.

The feature smoke reads the already verified local cache, checks its exact hash
before unpickling, and uses the first 32 training plays in canonical game/play
order. It does not load new data from Kaggle, score validation, or fit anything.
A missing or changed cache is a stop, not a reason to redownload large archives.

## Future comparison — NOT enabled by this milestone

First close the existing coordinate/velocity result using its frozen >=1% gain
and paired-game uncertainty rule. Do not change that old experiment to include
these new features or launch a duplicate to resolve unknown status.

Then specify a separately versioned, parameter-matched temporal-edge experiment:
same edge-encoder architecture in both arms, same masks, normalization, ordering,
seed, reflections, labels, loss, and optimizer exposure. The control receives
repeated terminal pair geometry across valid observed slots; the treatment
receives the actual observed pair sequence. This isolates temporal relational
information, conditional on the same added encoder capacity. Its precise
terminal/missing-derivative policy requires tests before implementation.

Before any scientific fit, validate raw-input adapter parity, screen input support
on training only, benchmark the new path on training only, fix exposure and a
hard budget, and prove exact independent checkpoint recovery. Do not reuse the
old model's throughput estimate for this larger tensor. One first-fold comparison
precedes replication; declared acceptance should require >=1% official RMSE gain
and a negative upper paired-game 95% difference bound. Retain every requested
row and fixed final EMA. Mark reused development folds as reused. Do not tune
against the already inspected reserved holdout or claim cross-season stability.

Role-dependent arrival, long-horizon parameterization, strictly observed-origin
augmentation, and permitted historical-season alignment remain independent open
hypotheses. Acceleration remains excluded pending its tail audit. A scientifically
negative experiment is committed and retained, not silently retried or relabeled.

## Publication and reproducibility

The bundle synchronizes only by fast-forward to an explicitly accepted immutable
Git revision and does not discard tracked edits or untracked files. Each run saves
a local status report and a small return ZIP; raw errors and checkpoint bytes
remain outside Git. New notebook output is generated from existing evidence or
clearly labeled synthetic examples. Updating `main` does not itself deploy AWS.
