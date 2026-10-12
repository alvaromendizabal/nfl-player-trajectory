# Velocity recovery review — September 11, 2026

## Decision

Do not rerun `nfl-motion-scientific-20260911-055510-d265d9d`: it completed. Read-only AWS inspection recovered the summary, both complete error CSVs and checkpoint-object metadata. No new fit or cloud compute was launched for this inventory.

Independent float64 recomputation of `sqrt(sum(dx**2 + dy**2)/(2*N))` gives **0.861603879** for coordinate-only control and **0.720381596** for velocity supervision, approximately **16.39% relative improvement**. Both files contain the same 83,938 unique forecast keys across 41 games; all error values are finite. The summary reports a paired-game 95% RMSE-difference interval of **[-0.155414, -0.126956]** and passage of the frozen continuation rule. The interval is reported, not newly bootstrapped in this inventory. The small float32/float64 differences do not change the conclusion.

Each arm completed 1,248 steps / 16 epochs. The saved scientific runtime is 614.516 seconds. Both final checkpoint objects exist at their expected sizes, but weights were not loaded for fresh inference here. Full CSV hashes were not recomputed in the restricted remote analysis runtime; the downloadable recovery lab performs SHA256 verification on restoration. See [machine-readable evidence](results/velocity_recovery_inventory.json).

**This is not a new best model.** The historical 0.62708 internal blend is not reconstructed: its original trained checkpoint sets remain missing. The project-recorded Kaggle private result is 0.70090; authenticated current submission history was not inspected. The historical winning private score is 0.46340. These results use different evaluation protocols and cannot be treated as one leaderboard.

## Where the error is

In the velocity arm, the 20,813 forecast rows after the first second account for about **24.8% of rows and 83.6% of total squared coordinate error**. This motivates investigation of remaining-time geometry, endpoint miss, pursuit/arrival demand and evolving receiver-defender relations. It does not prove these features will improve the official metric, nor establish that every long-horizon error comes from representation.

## Source, workspace and data boundaries

PR #28's observed-only synchronized relationship module is merged at `ed8ce417bb8fec29f0a3daa3d4120013d2e6bb5b`; reviewed-head Quality run 34640025249 succeeded. It is software evidence, not an ablated NFL accuracy result.

The last saved workspace verification is `d265d9decb4ca5cd829781ceb78bd043480b5c35`, September 11 at 05:55:09 UTC. At this inventory, `nfl-trajectory-dev` persisted but its JupyterLab application had status `Deleted` and remote access was disabled. No current on-disk synchronization with newer GitHub changes is claimed. This documentation publication is not an AWS deployment.

All 31 raw objects in the inspected snapshot passed S3 HEAD size checks. The snapshot contains 18 raw 2023 input weeks, example test files and evaluator support. It has no raw output-label CSV entries. This is a snapshot-scoped observation, not proof of label absence everywhere: cached research samples contain labels. Before any new supervised experiment, verify exactly which inputs, labels and checkpoint signatures it requires. The first input-only feature audit needs one early training week, not all archives.

## Highest-value next sequence

1. Complete an observed-input-only support audit for synchronized relationship histories and explicit ball-relative arrival histories. Check real frame clocks, missingness, masks, role strata and runtime. Ball landing is an organizer-supplied landmark, not every player's enforced destination.
2. Hold the architecture, training schedule, velocity-supervision setting, row keys and seed fixed. Compare temporal relationship histories against their equal-shape last-joint-observation control. Investigate arrival-demand and turn-alignment families separately before combining them. Predeclare official row-weighted RMSE and paired-game uncertainty; report all role/horizon slices without post-hoc slice selection.
3. Replicate a passing treatment on established chronological folds. Already-inspected development labels are not a fresh holdout. Preserve at least 1% relative matched RMSE improvement and a negative upper confidence bound as an advancement gate, not a record guarantee.
4. Then investigate shared earlier forecast-origin augmentation, role-conditioned trajectory parameterizations and permitted historical-season expansion in separately bounded studies. Verify external-data permission and inference availability before use.

Do not repeat the stopped linear soft-coverage probe, promote previously unsuccessful neural smoothing, clip acceleration labels silently, or revive the unrecovered historical blend by changing its name. Maintain the established feature bank and negative evidence. Feature research remains open.

Primary references: [organizer input contract](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data), [winning author's solution](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution), [final leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard), and [Song et al., coverage-responsibility modeling](https://arxiv.org/html/2603.25901v1). The coverage paper supports temporal relationship modeling as a hypothesis; its proprietary annotations are not assumed available for this competition.
