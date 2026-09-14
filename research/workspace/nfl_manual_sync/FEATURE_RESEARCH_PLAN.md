# NFL feature-research continuation — evidence before expansion

11 September 2026. **Research remains open.** This plan is separate from the manual workspace setup. It does not claim the strongest attainable representation has been found, that every candidate will help, or that the historical 0.46340 target will be beaten.

## What the prediction mechanism implies

The task forecasts player x/y after the throw, conditional on observed tracking and organizer-supplied context. The landing point and forecast horizon are legitimate supplied inputs for this competition; observed future player coordinates are not. This is not a live pre-pass system that must infer an unknown landing point. Position, heading, relative motion, role, reachable arrival geometry, and synchronized interactions are therefore meaningful hypotheses—not a license to use post-throw information.

Use the exact official coordinate metric: `sqrt(sum(dx**2 + dy**2) / (2*N))`, over every required coordinate row. Euclidean displacement RMSE without the factor 2, averages of subgroup RMSEs, and averages of fold RMSEs are not interchangeable with pooled official RMSE. Keep forecast keys and row counts identical between arms. Weight pooled results by summed coordinate squared errors and row counts.

Source: [project data contract](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/docs/DATA_CARD.md), [official scorer](https://www.kaggle.com/code/metric/nfl-2025), [recorded evaluation](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/START_HERE.md).

## Research-to-feature translation

**Temporal relationships, not just nearest neighbors.** Song et al. (2026) separate temporal and player interactions to infer defensive responsibilities. Their task and labels differ from coordinate forecasting, but motivate preserving changing pair relationships rather than terminal proximity alone. Their coverage annotations, predicted coverage inputs and later observations cannot simply be imported into this competition. Dutta, Yurko and Ventura (2019) likewise motivate relational movement features for coverage analysis; their coverage outputs are not evidence of an improvement in this project's RMSE. Treat proximity/soft affinities as representations, not ground-truth matchup assignments.

Primary sources: [Song et al., factorized attention and coverage responsibilities](https://arxiv.org/html/2603.25901v1); [Dutta et al., unsupervised pass coverage](https://arxiv.org/abs/1906.11373). Relevant primary competition leads to examine in full before reproducing details: [first-place author's write-up](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution), [first-place discussion](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814), [third-place write-up](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution). Their complete dynamically rendered bodies were not available in the final browser extraction; no undocumented implementation detail or leaderboard-equivalent replication is claimed here.

## Existing evidence must constrain the next experiment

The repository already contains thousands of feature candidates and multiple motion, context and representation studies. Quantity does not establish value. In the recorded matched continuation, smoothed-state inputs scored 0.71631 versus 0.71745 for the control and failed the declared gate. A fixed coverage treatment also failed in its prior representation. Do not rerun either unchanged simply under a new feature-family name.

The 0.62708 internal blend is an incompletely recovered historical result, not a deployable current model. The saved Kaggle submission record is 0.70090; a later owner-reported approximately 0.62 needs its own receipt. The separate coordinate/velocity-isolation job is completed, but the manual collector must first verify its immutable errors/weights and the existing notebook's paired-game inference. Reading checkpoint bytes is not a numerical forward replay.

The latest merged additions are `relation_history.py` and `temporal_edges.py`, with explicit synchronized observation masks and tests. Their prototypes are **implemented but do not yet have an attributable forecasting gain**. The next work should validate and integrate these, not create a third redundant relation tensor.

Source: [current protocol](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/docs/TEMPORAL_EDGE_PROTOCOL.md), [PR #28](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/28), [PR #29](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/29), [recorded study history](https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/START_HERE.md).

## Ordered, bounded investigation rounds

| Round | Hypothesis and representation | Leakage / attribution controls | Evidence needed before proceeding |
|---|---|---|---|
| 0 — setup and recovery | The latest code, original raw data, exact frozen cache and completed-run artifacts are accessible without reconstruction. | No fitting, no new split, no arbitrary overwrite, checksum verification before loading serialized objects. | This kit's sync/data receipts; existing-run saved-error verification; separate numerical checkpoint replay before promotion. |
| 1 — synchronized relationships | Separation change, closing/lateral motion, relative velocity, alignment, bearing change and defender/receiver motion context across time contain information terminal geometry loses. | Pre-throw jointly observed frames only; real clock gaps; no self edges, stale fill or invented assignment labels. Equal encoder capacity and optimizer exposure in history and terminal arms. | 32-play smoke first; representative training-only support/age audit next; raw-adapter parity and recovery test; one fixed chronological fold before replication. |
| 2 — arrival and role | Roles react differently to the supplied landing point; radial/tangential velocity, arrival slack, turning demand, and relative arrival ordering describe distinct movement objectives. | Use only supplied role/horizon/landing point and observed state; do not force every player to the ball; fit physical/prior scales on training only; ablate within the same model. | Role × horizon × distance support; aggregate error concentration; family add/remove ablations on identical forecast keys. Treat low-support subgroups descriptively. |
| 3 — trajectory state and turns | Causal multi-window velocity, heading change, curvature, deceleration and body/movement disagreement distinguish turning from steady pursuit. | Adjacent-frame derivatives, unit/reflection tests, explicit zero-speed masks; no interpolation through future frames. Acceleration/jerk remain gated by tail/support audits. | Prove a changed hypothesis relative to rejected smoothing; train-only outlier/sensor audit; bounded addition and omission comparisons. |
| 4 — origin and horizon representation | A better reference frame and horizon-conditioned representation may reduce accumulating long-horizon error without adding labels. | Compare absolute/local/landing-relative inputs with the same downstream capacity; preserve all requested rows. Do not add previously hidden post-throw observations under the label of augmentation. | Error/SSE accounting by horizon, train-only geometric augmentation tests, one attributable treatment at a time. Decoder/loss changes must be labeled separately from input-feature effects. |
| 5 — team structure and permitted histories | Spatial leverage, safety depth/width, receiver spacing, congestion, role-conditioned historical tendencies and opponent-relative context may add information beyond isolated motion. | Only actually available fields; historical summaries as-of earlier games, smoothed fold-locally with unseen-entity fallback. No full-season target encoding or PFF label import. | Availability audit and inference adapter tests first; support-adjusted add/remove ablations; game-blocked chronological replication. |
| 6 — external-season alignment | Additional legitimate labeled historical tracking may improve support for rare trajectories if usable under the rules. | Verify rules, access, labels, schema, sample selection and timing before use. Do not substitute Analytics data or assume January calendar dates imply a new season. | Written eligibility and mapping contract, held-out-season transfer test when independent labels actually exist, explicit budget. Not enabled by this kit. |

Rounds 2–6 are proposals to implement and test, not a claim that these families are absent in their entirety. Audit the existing feature dictionary before adding anything. Record which subfamilies already exist, which were rejected, which were never integrated, and which lack sufficiently isolated evidence.

## Round 1 experiment specification to freeze before fitting

**Question:** Does retaining the observed pair sequence help beyond the terminal-information control, given the same model capacity?

Hold player/edge encoder dimensions, losses, velocity supervision, optimizer steps, seed, augmentation, labels, normalizers, split, forecast keys and output horizon fixed. The two arms differ only in the information retained by the relationship input. Specify how terminal derivatives, joint observation time, unavailable channels and time encoding enter the control; silently turning missing derivatives into zeros would confound the comparison. Compare both with the unchanged baseline to separate adding capacity from adding temporal information.

Before one scientific fold: independent physics/unit tests; translation, reflection and permutation counterfactuals; poisoned missing-value tests; observed-time causality; label/query exclusion; raw-adapter parity; stable sample ordering; cache SHA and source signature; training-only support/age distributions; and a fresh-process checkpoint continuation test. Measure actual training-path memory and throughput—do not substitute the cheap NumPy tensor timing for end-to-end training cost.

A proposed continuation gate, to be frozen before scores are observed, is at least **1% lower pooled official coordinate RMSE** than the matched terminal control and a paired-game bootstrap upper 95% delta bound below zero. This gate is conditional evidence, not a leaderboard claim. If it fails, preserve both arms and diagnose representation or optimization before another run. A passing one-fold screen earns only later chronological folds and seed replication, not final-model replacement.

Compute allocation is not an unbounded dollar authorization. No cost estimate for a new learned encoder is asserted without a measured training smoke and the selected instance's rate. Begin with a small training-only throughput/recovery trial, then predeclare the wall-clock/spend cap for each single-fold run. Do not run the historical 128 GiB full-bank reproduction sequence on the small notebook space.

## Validation and presentation requirements

The downloaded Prediction inventory documents only labeled 2023-season data. January games retain the organizer's season designation. The project has repeatedly inspected development and already evaluated a reserved holdout; neither can be relabeled pristine. Feature screening and fitted encodings belong inside training-only chronological partitions. Any future truly untouched evaluation must have a genuine independent provenance.

For every family preserve a registry row: research source, formula/input contract, prediction-time availability, implementation/source hash, synthetic and raw parity tests, cache/input hash, fold definition, feature count, peak memory, elapsed compute, seeds, pooled SSE/N, per-game errors privately, confidence interval, addition/removal result, failure notes and keep/reject decision. Publish aggregate Plotly evidence; keep competition rows and weights private.

The distinction between representation and architecture is empirical. Prioritize features as requested, but do not assert the entire performance gap is caused by features without controlled comparisons. Ensembling comes after feature attribution and calibrated validation, not as a substitute for investigating the representation.

## Milestone report required at every stop

Record: attempted work; completed work; passes; failures; actual metric (or explicitly no new RMSE); saved artifacts; GitHub branch/commit/CI and separate AWS deployment status; what was learned; next highest-value question; and why its bounded compute is justified. Negative results are retained. Activity alone is not improvement.
