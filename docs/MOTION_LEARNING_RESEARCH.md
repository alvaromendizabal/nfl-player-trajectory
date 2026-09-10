# NFL motion representation: research extension

Research date: 10 September 2026. Target: **0.46 coordinate RMSE**. The
feature-completion gate remains **open**. Read this alongside the preserved
[domain review](DOMAIN_RESEARCH.md), which covers the original 20-family search,
the 330-candidate neural screen and the NFL/sports literature. This extension
addresses learned motion, supervision, target conditioning and historical data.

The [completed matched study](MOTION_SUPERVISION_RESULTS.md) now reports a
**5.18% pooled RMSE reduction from joint motion supervision**, with improvement
in all three folds. This supports the tested learning mechanism while leaving
component attribution and the 0.46 objective unresolved.

## What is most important for this prediction task?

The best-supported working answer is a reliable estimate of **current motion**,
conditioned on **time to the known pass arrival**, **player role**, and **the
evolving behavior of nearby players**. The order is a research priority, not a
universal importance ranking. Our strongest matched evidence supports smoothed
motion states and chronological player/role history. Fixed nearest-opponent
coverage summaries did not improve this model. That finding leaves learned,
uncertain responsibilities open: nearest does not necessarily mean responsible.

The central representation problem is to distinguish continuing a route,
slowing for the ball, turning toward it, and reacting to another player. Raw
speed, acceleration, orientation and velocity can disagree; a model should
learn when each measurement is informative. It must also distinguish known
arrival geometry from a hard requirement that every player finish at the ball.
These are proposed football mechanisms to test, not labels present in our inputs.

## Primary-source findings and transfer decisions

| Evidence inspected | Useful mechanism | Limit that changes our experiment |
|---|---|---|
| [Released first-place training code](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train) | Per-feature temporal convolutions and auxiliary velocity/acceleration targets. | Its complete recipe also changes loss, augmentation, exposure and ensemble size. We isolate the auxiliary package within a matched model and retain every requested row. |
| [Huang, Cheng and Wang, 2025](https://arxiv.org/html/2503.24272v1) | Separate motion streams, feature injection and consistency objectives support testing motion-aware latent features. | Pedestrian data, multimodal outputs and different losses; this is evidence for a hypothesis, not an NFL performance claim. Their position/velocity consistency attempt was not uniformly successful. |
| [SEPT, revised December 2023](https://arxiv.org/html/2309.15289v4) | Masked-history reconstruction and history-tail prediction are candidate pretraining tasks. | Road maps and best-of-six metrics differ from football. Its pretraining includes label-free validation/test inputs; our chronological protocol must instead restrict all pretraining to eligible training games. |
| [NFL/AWS coverage-responsibility Transformer, March 2026](https://arxiv.org/html/2603.25901v1) | Temporal/player representations predict uncertain defender assignments and receiver matchups. | Human play-level labels can reflect late-play responsibility; predicted early-play probabilities are proxies, not known facts. The paper's private multi-season training population is unavailable to us. |
| [Nguyen and Yurko, March 2026](https://arxiv.org/html/2603.17866v2) | Persistence of steps/turns, speed-dependent turning variability and player effects. | Ball-carrier rushing analysis and one-step alternatives differ from this multi-second passing forecast. Proposed persistence features still need our own test. |
| [LED, CVPR 2023](https://arxiv.org/pdf/2303.10895) | A learned trajectory distribution combines temporal motion and social attention; its initializer separates central motion, spread and correlated alternatives. | Its NFL benchmark uses 2017 tracks, 1.6 seconds of history and a 3.2-second forecast at 5 Hz, with best-of-20 displacement metrics in meters. These are different observations, targets, units and scoring rules. |

The review prioritizes organizer contracts, author code and papers. Broad search
results about generic AI, injury prediction or marketing were excluded. Newer
papers were checked through the research date, but recency alone did not decide
the implementation. Abstract-only findings were not treated as reproduced methods.

The coverage paper also trains one task with the true team-coverage category but
uses a predicted category at inference. That distinction matters for our proposed
transfer. In a two-stage feature pipeline, use predictions cross-fitted within
training games and frozen for later games, or pretrain exclusively on eligible
earlier seasons. A jointly trained auxiliary head is another distinct design.
Do not silently substitute unavailable true categories for predicted features.

The LED full paper confirms that an NFL-specific generative result can still be
an unsuitable numerical benchmark for this competition. Its best-of-many result
does not establish the accuracy of a single deployable point forecast. The
[released repository](https://github.com/MediaBrain-SJTU/LED) documents NBA data
and training commands; it does not establish that the NFL training population is
available in our workspace. A future generative experiment needs a justified
coordinate-mean output, source-verified data and the same chronological rows.
This keeps multimodal intent in the coverage ledger without treating a diffusion
model's name or an oracle metric as evidence of progress toward 0.46.

## New implementation and its causal question

The [fixed protocol](MOTION_SUPERVISION_EXPERIMENT.md) defines two arms on each
of the same three chronological folds. Both learn per-signal temporal filters
from the existing observed channels, then combine players through attention.
Both have the same position and motion heads, parameter initialization, exposure
and optimizer. The treatment additionally trains the shared features to predict
future velocity and acceleration. Position-only supervision is the matched
control. No future coordinates, derivative targets or evaluation-fitted
statistics enter inference.

The feature-wise stem and 48 training passes are shared. Compared with the old
40-pass mixed-convolution reference, both new arms also change batch size from
64 to 128 and the learning-rate range from 0.002–0.00005 to 0.001–0.00001.
An improvement over that old reference therefore cannot be attributed entirely
to auxiliary supervision. The primary feature claim is only the treatment-control
difference. This study adds **no new independent observed data fields**. Its 124
internal channels and four auxiliary outputs are learned representations and
training tasks, not 128 new domain measurements. A positive joint result would
still require separate velocity/acceleration ablations before declaring either
component established.

Target construction respects actual frame IDs and player identity. A gap creates
missing derivative supervision, not a fabricated 0.1-second interval. The first
future velocity uses the last observed coordinate only when the endpoint is
current. Reflection transforms positions and derivative targets consistently.
The tests check physical units, stale observations, missing frames, permutations,
padding, exclusion of future outcomes and exact interrupted-training recovery.

## Training-target conditioning: a newly measured gap

The [executed audit](results/motion_conditioning.json) reads only training labels;
it fits nothing and changes none of the declared arms. Across the folds,
forecasts beyond two seconds account for **3.755–3.806% of training rows** and
**48.782–52.489% of the ridge baseline's squared error**. This is baseline error,
not the trained attention model's error share.

At the most extreme ridge-displacement request, the x displacement is
48.766, 52.770 and 55.368 yards in the three training-fold baselines. The actual
x displacement is about 11.810 yards; its y displacement is about 5.140 yards.
The request belongs to the previously identified 9.4-second play. This explains
why target parameterization deserves attention, without establishing an erroneous
tracking label. The observed trajectory was not deleted or clipped.

**Inference:** a feature representation may spend disproportionate effort
cancelling extrapolation error in its baseline. A later matched comparison of
ridge residuals, displacement outputs and integrated frame increments should
hold the encoder, observation budget and training exposure fixed. It must score
the same complete set of rows, report short- and long-horizon changes, and avoid
claiming that a lower training target variance guarantees better predictions.

## Feature coverage and unresolved high-value work

These are mechanisms, not a count of independent columns. Status refers to this
repository's executed studies, not whether a paper has explored the subject.

| Mechanism | Current evidence/status | What remains necessary |
|---|---|---|
| Current and smoothed motion | Supported in matched addition/removal; weak gain in old-encoder continuation. | Learn reliability and temporal scales; distinguish telemetry error from a real maneuver. |
| Learned velocity/acceleration representation | Six matched fits complete; joint supervision improves all three folds and passes the declared gate. | Decompose velocity versus acceleration, then check seed stability. |
| Arrival geometry and feasibility | Available inputs; explicit candidates screened. | Role-dependent soft destination structure, with matched landmark removal. |
| Dynamic coverage responsibility | Terminal attention exists; added neighbor summaries hurt. | Encode temporal pair changes and assignment uncertainty; retain no-match possibilities. |
| Player/role history | Earlier-date-only target statistics improved all three original attention folds. | Separate role and player contributions, shrink sparse identities and quantify cold-player behavior. |
| Position and movement capacity | Supplied metadata was explored in the tree search; not a separate neural feature test. | Test compact position/anthropometric inputs and conditional entity priors with train-only vocabulary, shrinkage and missingness. Avoid an unsupported large metadata branch. |
| Route phase and intent | Explicit route representations already explored in the tree studies. | Observed-only learned phase/intent targets; no full-route classification at inference. |
| Long-horizon dynamics | Training audit shows concentrated baseline error. | Matched output parameterization and realistic tail behavior; no evaluation truncation. |
| Motion pretraining | Primary-source support; not established here. | Train-only history reconstruction/tail prediction with matched total exposure. |
| Spatial/temporal augmentation | Reflection implemented. | Consistent rotations/translations and earlier cutoffs; recompute every derived channel after a cutoff or mask. |
| Field and boundary context | Explicit families tested without a stable standalone conclusion. | Soft boundary/context effects, avoiding arbitrary trajectory clipping. |
| Historical-season information | Candidate datasets and fields identified below. | Verify access, eligible use, dates, event alignment and population shift before training. |
| Multimodal trajectories | Reviewed; no demonstrated official-RMSE advantage. | An explicit conditional-mean output rule and matched scoring; oracle sample selection is insufficient. |
| Pose/video, weather and text | No compatible measured benefit or current inference contract established. | Do not invent unavailable signals; any addition needs source, time-of-availability and predictive-value evidence. |

This ledger is intentionally open. Neither a large candidate count, a literature
table nor a statistically detectable 0.16% gain demonstrates an exhausted feature
search. Finishing requires resolving realistic high-value candidates and recording
why rejected alternatives are unavailable, redundant, costly relative to evidence,
or ineffective under a fair experiment.

## Auxiliary labels and historical data: concrete opportunities and boundaries

The [2026 Analytics organizer page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data)
lists `supplementary_data.csv`, including targeted-receiver route, man/zone
coverage and coverage type. Its displayed season counts include both 2023 and
2024. This offers a concrete candidate for training-only route/coverage tasks
on the existing games, before undertaking older-season transfer. Full file
transfer and row alignment have **not** been verified. The Prediction inference
contract does not supply these labels. Any use must join unique game/play keys,
restrict to each fold's eligible training games, audit missingness and annotation
timing, and keep later-season and evaluation labels outside model fitting.

Play descriptions and result fields are retrospective audit material, not
prediction inputs. Even pre-snap fields require a compatible inference source;
being observable in football does not make them available to this competition
model. A useful auxiliary task must improve trajectory RMSE under a matched
removal, not merely classify an interesting football category accurately.

The [2021 organizer data page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data)
documents 2018 passing-play tracking, with event markers and motion fields;
linemen are omitted. This is a plausible earlier-season source, but targeted
receiver identity, pass-arrival coordinates, population differences and event
alignment must be reconstructed and audited. Outcome descriptions are not model
inputs. Competition rules govern the data.

The [2025 organizer schema](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data)
documents targeted-receiver flags, route-running indicators, coverage assignments
and primary/secondary defender matchups. Nguyen and Yurko describe this release
as the first nine weeks of the 2022 season. These fields make supervised
representation pretraining more concrete than an undefined request for “more
NFL data.” Access and permitted reuse have not been established by reading the
schema; no claim of a completed download or trained transfer model is made.

Coverage annotations could supervise a training-only auxiliary task while
inference uses predicted probabilities. They must never be joined into our
evaluation inputs. Even historical training labels can be imperfect targets for
an earlier cutoff. All learned transforms and label encoders must be fitted
inside each eligible training population.

The [NFL's public GitHub sample](https://github.com/nfl-football-ops/Big-Data-Bowl)
currently retains one 2017 game and documentation. It is useful for parser and
event-alignment checks, but should not be described as access to a full historical
tracking season.

## Evaluation and interpretation

The primary metric remains unweighted coordinate RMSE in yards, pooled from
squared errors rather than averaged fold RMSEs. Report every declared arm and
all 202,361 requests. Game-level paired intervals describe uncertainty conditional
on this experiment; repeated adaptive use of the same folds remains a limitation.
An improved internal score is not a new Kaggle score or proof of state of the art.
The recorded Kaggle result remains 0.70090 until an actual later submission is
authorized, executed and verified. No submission is part of this study.
