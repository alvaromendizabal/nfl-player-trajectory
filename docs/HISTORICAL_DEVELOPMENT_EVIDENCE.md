# Historical velocity development evidence

This dated record preserves the missing aggregate evidence from [PR #30](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/30), [PR #31](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/31), and [PR #32](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/32). These proposals describe the same September 11 development experiment. **Consolidation does not change the final selected model or its reported results.** The [current results](RESULTS.md) remain authoritative for project closeout.

## What was measured

The [aggregate receipt](results/velocity_development_history.json) retains all three measurement contexts on **83,938 forecast rows across 41 games per arm**. Coordinate RMSE is `sqrt(SSE / (2 × rows))`, in yards.

| Historical measurement | Coordinate RMSE | Velocity RMSE | Verification scope |
|---|---:|---:|---|
| PR #30 | 0.8616038787456273 | 0.7203815956554687 | Full saved residual CSVs independently recomputed in float64 at that time |
| PR #31 | 0.8616038787456286 | 0.7203815956554684 | Separate complete-CSV inventory |
| PR #32 | 0.861603856086731 | 0.7203815579414368 | Original saved summary; complete residual recomputation was not completed in this inventory |

The saved paired-game 95% interval for **velocity minus coordinate RMSE** is **[-0.1554135397037707, -0.1269560761641106]**, using 10,000 bootstrap resamples. The interval was reported by the historical experiment; it was not recomputed during this consolidation. The receipt preserves exact pooled SSE, horizon aggregates and the separately recorded defensive-coverage SSE fraction. Values from different verification contexts remain separate, including their last-bit differences.

The original improvement justified continued development on that population. It does not establish a new competition result, an untouched-test guarantee, or an improvement to the final model. No private predictions or model weights were replayed for this publication.

## Byte provenance and publication scope

The receipt links each original JSON path at its exact PR commit and records its SHA-256. The following mapping describes a selective consolidation, not a byte-identical copy.

| Original JSON source | Original SHA-256 |
|---|---|
| [PR #30](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/2861366277903c70c1a5c0fc7908f94f8948d989/docs/results/velocity_recovery_inventory.json) | `d35c3ec57584e8d89451e10a33ab43f28a4efb9f894bb1045b6feb016111b8dd` |
| [PR #31](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/dc193a16d599e9f42fa0bdbd1b3c6bceb56a4a20/docs/results/feature_restart.json) | `f7552901a67fc724a6390c4ee2d90b2a5e7d2be2909dc8677d4aaef9389dede2` |
| [PR #32](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/c49c2032ef6a0dc3d86d77f6e31c6c50c194216d/docs/results/velocity_isolation_recovered.json) | `5fa1a5534531daa13c3a0bf9353251b50e0ea03d5a03607b8b27f7daec84adc2` |

Published file: `docs/results/velocity_development_history.json`  
Published SHA-256: `c039a34481ec0fdb90d1ecbcb84369bc0fa55fcd5fde43c8e60967f2f45f8620`

Deployment locators, private artifact paths, training schedules, workspace inventories and obsolete next-step plans were excluded. These omissions do not alter the preserved measurements. A source hash identifies the original record; it does not make withheld residuals, checkpoints or the historical bootstrap publicly reproducible.
