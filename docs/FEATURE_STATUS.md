# Current features, models and research decision

Verified 10 September 2026. **Feature research remains open; the 0.46 target is unmet.**

## Current score and model status

| Evidence | Coordinate RMSE | Meaning |
|---|---:|---|
| Recorded Kaggle private submission | 0.70090 | The actual submitted model's score. |
| Earlier replayable domain/tree blend | 0.64160 | Three reused chronological folds; separate from Kaggle. |
| Latest historical motion-supervision/tree blend | 0.62708 | Recorded three-fold evidence; six model sets and exact source remain missing. |
| New soft-coverage correction probe | 0.66895 | One predefined fold; not comparable to pooled three-fold scores. |

The submitted model is histogram gradient boosting of residual coordinates on
engineered features. Physical and role-conditioned ridge models provide baselines.
Temporal neural models learn observed motion and attention across players. These
are legitimate model classes, but the submitted system has not demonstrated
leaderboard-leading performance. The strongest recorded neural variant is not
currently replayable, which is a release-readiness defect.

Going from 0.70090 to 0.46 requires about **34.4% lower RMSE** on the same test
population. Internal validation cannot establish that reduction on Kaggle. We
cannot allocate the remaining leaderboard gap numerically among features, data,
training, architecture and ensemble size without matched experiments.

## Exact tabular candidate inventory

Counts below come directly from `docs/results/feature_catalog.csv`. They describe
candidate columns, not distinct raw signals or proven useful inputs. The bank has
7,999 candidates; its development metadata-free fit has 6,308 screened columns and
953 active tree inputs. The final refit has 951 active inputs. These counts refer
to different fits and stages.

| Family | Candidates | Football information represented |
|---|---:|---|
| `history_lags` | 560 | Exact recent positions, velocity, acceleration and landmark-relative motion. |
| `history_summaries` | 1,568 | Observed-window trends and summaries. |
| `availability` | 31 | Telemetry and input availability indicators. |
| `interactions` | 204 | Relative-player and mechanism interactions. |
| `context` | 12 | Additional observed contextual summaries. |
| `forecast` | 6 | Requested time and supplied horizon. |
| `forecast_interactions` | 462 | Forecast time crossed with observed state. |
| `multiscale` | 504 | Motion summaries over different observed time scales. |
| `nonlinear` | 84 | Nonlinear transforms of physically motivated inputs. |
| `role_response` | 1,848 | Role-dependent motion and forecast response. |
| `projected_geometry` | 98 | Kinematic projections relative to landmarks and players. |
| `player_history` | 10 | Smoothed player/role residual statistics from strictly earlier dates. |
| `metadata` | 108 | Player body and position information; weak in controlled tests. |
| `robust_history` | 576 | Robust dispersion, path and motion distributions. |
| `matched_history` | 684 | Synchronized opponent/receiver motion histories. |
| `peer_context` | 48 | Within-play relative player context and ranks. |
| `reachability` | 52 | Arrival feasibility and required motion. |
| `graph_pool` | 704 | Distance-weighted summaries of the surrounding player set. |
| `role_destination` | 120 | Role-conditioned destination hypotheses. |
| `route_representation` | 320 | Training-only observed-route components and prototypes. |

The neural representation separately uses 20 observed frames with 30 dynamic
channels, static state/history features and 12 pair-geometry channels. Its domain
correction bank has 330 candidates: motion state 57, arrival constraints 87,
coverage dynamics 147 and field geometry 39. Adding these counts to 7,999 would
exaggerate unique information because the banks overlap. The new soft-coverage
bank adds 11 terminal and 72 temporal affinity candidates; 76 additional columns
survive the matched training-only screen in the full probe.

## What has earned evidence, and what has not

- Expanded tabular representation improved fixed-estimator development RMSE
  from 0.80120 to 0.68805, a 14.12% controlled improvement.
- Chronological player/role histories improved attention from 0.77414 to 0.72063
  across the same three folds. Those histories use strictly earlier dates.
- Explicit motion summaries helped conditional neural corrections. Additional
  nearest-opponent coverage summaries hurt that tested interface.
- Repeating smoothed state in the old temporal encoder produced only a 0.159%
  gain and failed its declared feature gate.
- Historical auxiliary velocity/acceleration supervision improved a matched
  coordinate-only neural control from 0.69635 to 0.66029, but missing checkpoints
  prevent replay of that exact result. It cannot be treated as a released model.

## New bounded soft-coverage result

The [declared protocol](SOFT_COVERAGE_EXPERIMENT.md) tests changing affinities over
all eligible opponents, including an unmatched state. It takes inspiration from
[NFL/AWS coverage-responsibility research](https://arxiv.org/html/2603.25901v1),
while excluding its unavailable annotations and later observations. Independent
[unsupervised NFL coverage research](https://arxiv.org/html/1906.11373v3) also
motivates soft assignments from motion relationships. These papers support the
hypothesis; they do not prove its trajectory-RMSE value.

| Fixed ridge correction | RMSE | Retained columns |
|---|---:|---:|
| Existing decoder/control + motion state | 0.672064 | 182 |
| Add terminal soft affinities | 0.670920 | 192 |
| Add terminal + temporal soft affinities | 0.668952 | 258 |

Training used 193,452 rows from 94 games; evaluation used all 83,938 rows from
41 later games, exactly matching `inner_1`. Preparation and all three fits finished
in 21.8 seconds on local CPU. No AWS/GPU training job was launched.
The primary gain is **0.463%**, with paired game-bootstrap 95% RMSE-difference
interval **[-0.006314, +0.000060] yards**. It fails the predeclared >=0.5% gate
and the interval crosses zero. **Stop this linear correction interface.** Do not
spend additional folds or neural compute on this exact treatment, tune its fixed
scales against this fold, or promote it by selecting a secondary comparison.

The role diagnostic explains the mixed effect: defensive-coverage RMSE improves
from 0.72055 to 0.71349, while targeted-receiver RMSE worsens from 0.53109 to
0.54112. This is descriptive evidence of a shared-head tradeoff; it does not
authorize choosing a role-gated treatment after inspecting this fold.

[Results](results/soft_coverage.json) and [replay receipt](results/soft_coverage_recovery.json)
preserve the measured evidence. This is a single reused fold and an inexpensive
linear correction of parent in-sample training residuals. It does not exclude
end-to-end learned temporal assignment representations.

## Next high-value bounded milestone

Reconstruct a compact motion-learning model as a **new, separately versioned
experiment**, using the preserved motion-supervision protocol rather than claiming
to have recovered missing weights. First compare matched coordinate-only versus
velocity-only auxiliary supervision on one fold under a fixed small budget.
Save and download-verify each checkpoint before continuing. Acceleration-only and
joint-supervision attribution follow only if the initial model and durable replay
pass. This direction has stronger prior measured support than more heuristic
coverage columns; the missing six-model study recorded 5.18% matched improvement.

Further substantive avenues remain: learned temporal defender/receiver relations,
role-dependent arrival behavior, stable displacement/velocity parameterizations,
strictly observed earlier-origin augmentation, aligned historical NFL seasons,
and seed/fold replication. The [domain ledger](DOMAIN_RESEARCH.md) records their
availability limits. Historical-data use requires competition-rule and event
alignment verification; it is not currently an implemented or validated input.

There is no basis to label the current deliverable 9.9/10 or call the feature
space exhausted while key trained artifacts and these comparisons are missing.

## Reproduction environment

The original probe ran in Python 3.12.14 with numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, filelock 3.32.6, threadpoolctl 3.6.0.
Use the numerical source at `c3bc276ccbb4110b66ef359451a0f855d1f3a7e5`, restore the verified
private checkpoint contents, and run `python scripts/soft_coverage.py --source-commit c3bc276ccbb4110b66ef359451a0f855d1f3a7e5`.
The saved plan rejects environment, source or input drift and reuses completed
stages. The parent plan retains all three chronological folds; this milestone
restored and evaluated only the first. The private backup includes every parent
input required by this probe, so it does not depend on the old transient checkout.
