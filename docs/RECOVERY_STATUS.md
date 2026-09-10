# Motion-study recovery status

Verified 10 September 2026. **Recovery is incomplete. The 0.46 objective is
unmet and feature research remains open.**

The latest saved experiment reports a fixed motion/tree blend of **0.62708**,
compared with the prior declared domain/tree blend of **0.64160** on the same
202,361 internal evaluation rows. This is a 2.26% reduction. The six newest
trained checkpoint sets and five source/test changes have not been recovered.
The saved score therefore cannot currently be replayed from those model weights.
Recorded Kaggle private RMSE remains **0.70090**.

## What failed

The execution environment was reset after the six models completed and while
the final source publication was incomplete. The former temporary checkout,
training files and live execution state were absent at recovery. Twenty-two
uploaded Git objects survived, but no branch or commit referenced them.
The last durable private archive predates the new six-model study.

This is a persistence and handoff failure. The local replay test had verified
reuse before the reset; it did not establish recovery from durable storage.
Checkpoint files needed to be saved and verified before publication and before
continuing to subsequent work. The underlying cause of the execution-environment
reset and the separate ChatGPT failure notices is not established by this audit.

Other avoidable overhead is recorded rather than hidden. The
[domain review](DOMAIN_RESEARCH.md) identifies 30 reproduction fits after a
typing correction, separate from 60 scientific feature comparisons. The newer
[historical quality receipt](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/8d7078f4874e8d8fd4c51581eb0d4e9baed7b125/docs/results/supervision_notebooks.json)
records a reporting-type error and accumulated synthetic-test output exhausting
a storage gate; both were corrected without retraining the six new models.
Training counters are not a credit or billing ledger, and cannot account for
the reported 700 minutes of ChatGPT usage.

## What has been recovered and verified

The surviving files are preserved in immutable commit
[`8d7078f`](https://github.com/alvaromendizabal/nfl-player-trajectory/commit/8d7078f4874e8d8fd4c51581eb0d4e9baed7b125)
on `recovery/motion-supervision-evidence`. This branch is an incomplete historical
evidence checkpoint and must not be merged as an executable model release.

- All 22 recovered files match their original Git blob identifiers.
- Sixteen available files match the historical 19-file publication seal; the
  other three source files are explicitly missing.
- All three recovered canonical notebooks match their execution-receipt hashes,
  contain the recorded executed cells, and contain no saved error outputs.
  They have not been re-executed in the incomplete recovery checkout.
- All seven pooled model RMSEs recompute from the saved fold scores and row
  counts. Row counts and fold identifiers match the earlier domain study.
  This verifies report arithmetic, not a new evaluation of individual predictions.
- The earlier private archive matches SHA-256
  `ef73a55e88678985a7c1b7bc1ed67ce98babc202f12b62e8d90da785e6582175`.
  Every one of its 412 content-addressed objects was read and verified, covering
  450 current logical files and one historical cache entry. It contains 96
  checkpoint paths: 66 scientific fits and 30 historical reproduction fits.
  None belongs to the newer motion-supervision study.
- The configured S3 bucket contains 67 snapshot manifests; the newest is dated
  10 September at 01:13:02 UTC. Exact object checks for the six new checkpoint
  hashes and three missing source hashes return `404 Not Found`.
- No NFL SageMaker training or processing job was in progress in `us-west-2`
  at the check. This is scoped to those named services, project and region.

The machine-readable [recovery receipt](results/recovery_status.json) preserves
the file inventory, hashes, missing paths, cloud checks and numerical verification.
No new training epochs or Kaggle submission were performed during this recovery.

## What the recorded experiment supports

| Comparison on the same three internal folds | Coordinate RMSE |
|---|---:|
| Matched position-only supervision | 0.69635 |
| Position plus velocity/acceleration supervision | 0.66029 |
| Fixed motion-supervision/tree blend | 0.62708 |
| Prior declared domain/tree blend | 0.64160 |

The controlled auxiliary-supervision comparison records a **5.18%** reduction,
with improvement in every fold and a saved paired 95% difference interval of
**−0.05587 to −0.01636 yards**. The current recovery cannot recompute this
bootstrap interval without the missing per-row predictions.

Both new arms use the same initialization, 281,430 parameters, observed inputs,
48 training passes and optimizer. The treatment adds motion labels only during
training. There are no new independent observed fields; the learned temporal
channels must not be counted as new raw domain measurements. Improvements over
the older attention model also combine changes in temporal stem, exposure,
batch size and learning rate. Only the treatment/control contrast isolates the
joint auxiliary-supervision package.

These are reused development folds and one seed. They do not establish a new
Kaggle result, state-of-the-art performance or exhausted feature engineering.
Read the preserved
[complete result](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/8d7078f4874e8d8fd4c51581eb0d4e9baed7b125/docs/MOTION_SUPERVISION_RESULTS.md)
with the recovery limitation above.

## Feature research priorities remain substantive

The preserved
[primary-source research extension](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/8d7078f4874e8d8fd4c51581eb0d4e9baed7b125/docs/MOTION_LEARNING_RESEARCH.md)
and [domain review](DOMAIN_RESEARCH.md) cover football motion, pass-arrival
geometry, player history, uncertain coverage assignments, route intent,
representation learning, historical seasons and their inference boundaries.
Fourteen mechanism families remain in the extension's open coverage ledger.

The most pressing measured weakness is motion beyond one second: the saved
motion model places **89.19% of squared error** there. Defensive coverage players
account for **84.57% of its squared error**. These are diagnostics of the standalone
motion model, not of the 0.62708 blend and not causal feature rankings. The next
experiments must isolate velocity versus acceleration supervision, evaluate
stable motion parameterizations, and test temporal defender/receiver relationships
and role-dependent arrival behavior. Nearest-opponent summaries already failed;
that does not settle uncertain, changing coverage responsibility.

## Required recovery sequence

1. Recover the exact missing source and six checkpoint/prediction sets if an
   additional original source is found. Their expected hashes are preserved.
   Do not relabel reconstructed source or newly trained weights as original.
2. Before another expensive fit, commit the exact runnable source and verify a
   durable checkpoint round trip from a separate local destination. Treat a
   failed save as a stopping condition before the next fit.
3. Save each completed fit and its input/source lineage before proceeding to
   the next arm. A local replay receipt alone is insufficient.
4. Preserve the same chronological rows and declared attribution comparisons.
   Record any required reconstruction as a new experiment with separate cost,
   lineage and results. Do not reopen or call the previously inspected holdout
   untouched, and do not submit to Kaggle while feature research remains open.

The earlier complete archive and public repository remain the durable starting
point. The new study's aggregate evidence is preserved; its missing executable
artifacts remain a concrete blocker to resuming that exact trained model.
