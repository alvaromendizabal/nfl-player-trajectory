# Current research status — September 2026

## Competitive state

- Strongest recorded late private submission: **0.46547 RMSE**.
- Published first-place private score: **0.46340 RMSE**.
- Private-score gap: **0.00207**.
- Strongest completed local system: **0.4631723213 OOF RMSE** over 561,607 rows from 20 fixed models across four grouped-CV split families.
- Stretch research target: **0.44 RMSE**. No local result is claimed as equivalent to a private-leaderboard score.

## Latest milestone: observed-context supervision

Earlier-origin augmentation already removes recent observed frames before forecasting. The latest study reused those real removed pre-pass trajectories as additional labels for non-requested context players while keeping unknown post-pass motion masked. The backbone, inputs, original requested-player targets, augmentation, and parameter count stayed fixed.

The matched control and treatment used the same initialization, examples, optimizer schedule, effective batch, sample order, and validation protocol. The treatment only added the fixed 0.2 context objective.

### Result

| Metric | Fold 0 | Fold 1 |
|---|---:|---:|
| Reference blend RMSE | 0.453726 | 0.475846 |
| Context blend RMSE | 0.452149 | 0.476089 |
| Gain vs reference | +0.001577 | -0.000243 |
| Matched-control incremental gain | +0.000893 | +0.000093 |
| Decision | Discovery passed | **No confirmation** |

Fold 0 cleared the locked reference gate and the matched-control gate. Fold 1 did not confirm either effect. The study therefore ended with `NO_CONFIRMATION`; no post-hoc weight search or selective fold promotion was used.

## Execution and GPU evidence

- AWS instance: `ml.g6e.2xlarge` in `us-west-2`, one NVIDIA L40S 48 GB GPU.
- Runner time: **1095.7 s**.
- New fits: **4**.
- Optimizer updates: **8,416**.
- Runtime-based compute estimate: **$0.852** at **$2.80/hour**; excludes startup, idle time outside the runner, storage, and API requests.
- Delivery tests: **220 passed**.
- Fold-0 benchmark selected microbatch 256 and six loader workers. The measured optimizer benchmark was about 5.6k samples/s and the reported end-to-end step time was about 0.086 s.

## Scientific conclusions

1. Extra real supervision can create a material first-fold improvement without adding parameters, but the effect was not stable enough to promote.
2. The matched control confirms that Fold-0 benefit was not merely additional fine-tuning; the incremental treatment gain was positive and its paired-game interval was positive on Fold 0.
3. Fold 1 failed confirmation, so the family is retired in its fixed form.
4. Multi-split diversity remains the strongest completed local result. Future work should prioritize materially new feature-configuration or heterogeneous-representation diversity rather than near-duplicate fine-tuning.
5. The existing private score remains 0.46547; no unscored local metric is used to claim a leaderboard win.

## Public/private boundary

This public snapshot intentionally excludes raw tracking data, player-level private outputs, fitted weights, private S3 paths, credentials, and large checkpoint archives. AWS remains canonical for those artifacts. GitHub contains aggregate result receipts, executed notebooks, selected figures, and documentation suitable for technical review.
