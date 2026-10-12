# Feature research restart — verified inventory

11 September 2026. Feature engineering remains open. The competitive target is strictly below the historical final private Kaggle winner, **0.46340**, on a comparable evaluation. No leaderboard improvement is claimed by this milestone.

## Repository, workspace and score provenance

The inspected main commit was `ed8ce417bb8fec29f0a3daa3d4120013d2e6bb5b`. PRs 27 and 28 are merged. Main Quality workflow **34640469714** completed successfully. PR 28's synchronized player relationships are implemented and tested, but their predictive contribution remains unmeasured.

Live AWS reads found the persistent `nfl-trajectory-dev` space in `us-west-2`, domain `d-njhxv1erusdc` / `QuickSetupDomain-20260902T115323`. Its JupyterLab app was stopped (`Deleted`), its configured instance is `ml.m5.large`, its volume is 50 GB, and remote access is disabled. The last discovered verified workspace synchronization receipt is dated `2026-09-11T05:55:09.699645+00:00` and references `d265d9decb4ca5cd829781ceb78bd043480b5c35`. The receipt does not establish that the newer GitHub changes are on the Studio disk. No fresh filesystem synchronization or complete raw-data inventory is claimed.

Keep the scores separate: the user's approximate 0.62 test report has not been resolved to a specific receipt/split; the repository's historical 0.62708 internal blend lacks all original components for replay; the repository-recorded Kaggle result is 0.70090. These are not interchangeable. The official winner is 0.46340.

## The existing motion experiment completed

Do not relaunch `nfl-motion-scientific-20260911-055510-d265d9d`. Live `DescribeProcessingJob` reports Completed. The job used one CPU `ml.m5.2xlarge` instance and an 1,800-second runtime cap. Its processing interval was 780.314 seconds; the scientific runner reported 614.516 seconds.

Both complete saved residual CSVs were read and independently checked for duplicate keys and nonfinite errors. Their key sets match exactly: **83,938 rows in 41 games**. Float64 recomputation of `sqrt(sum(dx**2 + dy**2)/(2*N))` gives:

| Arm | Coordinate RMSE |
|---|---:|
| Coordinate objective control | 0.8616038787456286 |
| Auxiliary velocity objective | 0.7203815956554684 |

The relative improvement is **16.3906%**. Both arms completed 16 epochs / 1,248 optimizer steps. The saved 10,000-resample game-bootstrap 95% interval for the RMSE difference is **[-0.155413540, -0.126956076]**. That interval was read from the scientific summary, not independently bootstrapped again in this inventory. The original >=1% gain and negative upper confidence-bound continuation gate passed.

This is an auxiliary-supervision result, not a new inference-feature ablation, not the historical 0.62708 blend, and not a new Kaggle score. It demonstrates why the remaining performance gap cannot yet be assigned entirely to features. The private artifacts exist and were read; their byte-level SHA256 read-back is an explicit workspace-restoration gate rather than a newly completed claim here.

For the improved arm, forecasts beyond one second account for **24.795% of rows but 83.573% of squared error**. Defensive-coverage players account for **82.668% of squared error**. These are useful hypothesis priorities, but the inspected validation data must not be relabeled untouched confirmation data.

## Feature priorities and controlled rounds

1. **Recover, synchronize and audit support without fitting.** Restore the verified minimal input archive and completed-study artifacts. Check account, source commit, schemas, hashes and the 4,951-play / 94-game / 193,452-row training population. Use a separate lab directory so the previous Studio checkout and notebooks remain intact. The first support audit is capped at 256 training plays, 16-play atomic chunks and 300 seconds. This cache is sufficient for that milestone, not all later raw-data work.
2. **Synchronized learned player relationships.** Compare terminal-only and full observed relationship histories with matched capacity, supervision, row coverage and training exposure. Separately investigate directed lagged velocity/acceleration/turn-rate coupling. The proposed lag family has 24 correlations and 48 explicit support columns, not 72 independent proven signals. Require real-data masks, finite losses, same keys and checkpoint replay before scientific fitting. Short-series correlations do not establish causal reaction times or true coverage assignments.
3. **Long-horizon and earlier-origin representations.** Reconstruct earlier forecast origins from legal observed training histories, with an explicit origin-offset feature and strict separation between inputs and newly supervised future frames. Verify raw input availability before implementation. Do not truncate long scored trajectories or force every player to the landing point.
4. **Role-conditioned route/arrival, motion quality and historical context.** Audit the existing 330-candidate domain bank before inventing duplicate columns. It already represents smoothed motion, arrival deficits, turning/braking, reaction-delay hypotheses, coverage summaries and field geometry. Test useful families in the appropriate representation; freeze prior tables and normalizers on the training prefix, never using same-game/date outcomes or validation updates.
5. **Confirmation before ensemble expansion.** Establish which later allowed chronological partitions have not influenced selection; label reused folds exploratory. Require family add/remove ablations, all-row coordinate RMSE, game-cluster uncertainty, and stability across chronological windows and subsequently more than one seed. A 1% relative gain with a negative upper 95% paired-game bound is a proposed continuation gate, not a record-beating guarantee.

Preserve negative controls: the previously tested fixed soft-affinity ridge treatment and smoothed-motion temporal treatment failed their predeclared gates. Do not rerun those unchanged or generalize their failure to every nonlinear use of the information. The repository's thousands of candidate columns do not establish feature maturity.

## Research basis

The winning author's account uses compact observed motion and landmark-relative inputs, player interaction modeling, auxiliary objectives and earlier-origin augmentation, with substantially more training and extensive ensembling. More columns alone are not a supported explanation of the entire gap. Coverage studies motivate changing relative-motion relationships, but retrospective coverage labels and after-throw inputs from those studies are unavailable features for this task.

- [Official final leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard)
- [Winning author's first-place solution](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
- [Dutta, Yurko and Ventura: coverage features](https://arxiv.org/html/1906.11373v3)
- [Song et al.: factorized temporal/player coverage modeling](https://arxiv.org/html/2603.25901v1)
- [Official external-data rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/rules)

## Completion boundary

This publication records verified inventory, recovered metric evidence and the next bounded research protocol. It does not modify existing training code, model checkpoints or frozen configurations, launch paid jobs, submit predictions, or claim a completed workspace deployment. Downloadable feature-lab helpers are engineering artifacts supplied separately; their real-data feature audit and predictive ablations are not yet completed or promoted into the production training path. Record the actual commit and hashes whenever those helpers are installed.

After every meaningful milestone report attempted/completed work, passes/failures, actual metrics or `not measured`, durable artifacts, GitHub status, learning, next step and its expected information value. Keep original files and completed work; stop on mismatches instead of destructive resets or unchanged retries.
