# Recovered velocity-isolation result

**11 September 2026 — saved scientific evidence recovered; feature research remains open.**

The existing SageMaker Processing job `nfl-motion-scientific-20260911-055510-d265d9d`
is now verified as **Completed** through the live AWS API. Its saved status and full
scientific summary were read from private S3. Do not launch a duplicate of this job.

## Reported matched result

| Arm | Coordinate RMSE (yards) |
|---|---:|
| Coordinate-only control | 0.861603856 |
| Velocity-supervised treatment | 0.720381558 |

The recorded relative reduction is **16.3906%** over the same 83,938 requested rows
from 41 chronological evaluation games. Both arms used 16 epochs and 1,248 optimizer
steps. The scientific runner recorded 614.516 seconds; the original job used one
CPU instance with a 1,800-second maximum runtime.

The reported paired-game bootstrap interval for treatment-minus-control RMSE is
**[-0.155414, -0.126956] yards** (10,000 resamples, seed 2026). This passes the frozen
one-fold screening rule: at least 1% relative reduction and an upper confidence
bound below zero. This is still one seed on a reused development fold. The interval
does not correct for earlier exploratory searches on those games.

The exact reported numbers, source revision, S3 report key and verification limits
are preserved in [the machine-readable review](results/velocity_isolation_recovered.json).

## What this does not establish

The numbers above come from the independently retrieved saved summary, not a newly
completed error-file recomputation or forward pass through restored model weights.
The original report records successful final checkpoint remote read-back. That
original job's recovery receipt is distinct from reloading those weights today.
The connector read the private error CSV schema, but its bulk retrieval task did
not yield a retrievable result. No independently recomputed row-level metric is
claimed from that attempt.

This compact reconstruction is not the historical 0.62708 three-fold blend whose
checkpoint recovery remains incomplete. It is also not a new Kaggle submission.
The maintained 0.70090 Kaggle private record, the earlier replayable 0.64160 internal
blend, and the 0.46340 historical winning target refer to different populations or
model lineages. They must not be displayed as directly matched comparisons.

No model source, scientific configuration, old notebook, checkpoint, or Kaggle
submission is changed by publishing this review. No new AWS job was launched.

## Where the error profile points

For the velocity arm, reported coordinate RMSE rises from 0.33668 in the first
forecast-second bucket to 1.12559, 2.12745 and 2.39788 in the subsequent buckets.
The final bucket has only 64 rows. Using each bucket's row count times its squared
RMSE, requests after one second represent about **24.80% of rows and 83.57% of total
squared coordinate error**. This is a descriptive derivation from the saved summary,
not a fresh model evaluation or permission to tune horizon thresholds on validation.

All requested horizons remain in the official metric. Do not select clipping,
subgroup gates or loss weights from these inspected outcomes.

## Next bounded gates

1. Verify the preserved error CSV SHA256 values, exact matching unique keys, all
   83,938 rows and all 41 games; independently recompute coordinate-weighted RMSE and
   the paired-game screening rule. Restore and forward-replay both final weights
   separately before calling the model independently reproduced.
2. Audit the training-only simultaneous-observation support and actual joint-frame
   age of the already implemented synchronized relationship representation. Include
   pairs with no joint observation, preserve gaps and exclude self-pairs. A mask
   availability audit does not establish predictive value.
3. Freeze the next learned relationship interface and its equal-capacity terminal
   joint-observation control. Complete real-data loss, throughput and checkpoint
   recovery smoke tests before a bounded chronological comparison. Geometry,
   relative-motion and role-context group ablations must be separate attributable
   tests, not an open-ended architecture or feature-count sweep.

The passed velocity screen merits a frozen replication after artifact review, not
uncontrolled retuning. The no-fit relationship support prerequisite can proceed
without bypassing that requirement. Role-conditioned arrival, long-horizon motion,
strictly observed earlier-origin augmentation and permitted historical-season
alignment remain open hypotheses. Keep the rejected exact soft-coverage and
smoothed-state treatments stopped. Acceleration supervision remains disabled
pending its training-label tail audit.

See [the existing representation contract](RELATION_HISTORY.md) and
[the feature inventory](FEATURE_STATUS.md) for the prior evidence and information
availability constraints. Older readiness paragraphs in those documents describe
an earlier inspection; this dated review supplies the recovered job status and
reported scientific result without silently changing their original evidence.
