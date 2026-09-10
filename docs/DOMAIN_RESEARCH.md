# NFL trajectory prediction: domain evidence and feature research

Research date: 10 September 2026. The target remains **0.46 coordinate RMSE**.
The feature-completion gate is **open**. This report does not authorize a new
Kaggle submission, declare an independent test result, or claim state-of-the-art
performance for this project.

## What matters most in this particular problem

The most defensible current answer is **accurate motion state, the known time and
location of the pass arrival, and how players react to one another**. Those are
mechanisms, not an instruction to maximize column count. Their representations
must earn their place in matched experiments. The new controlled screen gives
the strongest empirical support to explicit motion features. Its extra
nearest-opponent coverage summaries make this particular model worse.

The prediction is conditional on unusually informative organizer-supplied
metadata: intended receiver role, ball landing coordinates, and the number of
future frames. These narrow the space of plausible movement substantially.
Nevertheless, a receiver need not finish at the exact landing coordinate, a
defender may be playing a receiver or an area, and multiple movement patterns
can be consistent with the same observed history. A useful representation must
preserve these distinctions instead of forcing every player onto a single
ball-seeking path.

The project's existing tree search already generated 7,999 candidates across
20 families. The selected full-training metadata-free profile retained 6,308
screened candidates; the development fitted tree used 953 distinct predictors.
Those counts describe different stages. The present 330-column bank is a
separate conditional neural experiment. Some information overlaps the tree
bank, so adding the two counts would exaggerate the amount of unique signal.

The new experiments answer a narrower causal question: does giving the same
correction network an explicit family improve predictions on the same later
games, holding capacity and training fixed? They do not prove that a family is
intrinsically useless, that the original attention architecture is optimal, or
that adding these features to a fresh end-to-end model will give the same gain.

## Information budget and the official objective

For requested rows, the primary metric is

\[
\mathrm{RMSE}=\sqrt{\frac{1}{2N}\sum_{i=1}^N
[(\hat x_i-x_i)^2+(\hat y_i-y_i)^2]}.
\]

The unit is yards. This is not mean Euclidean displacement, final displacement,
per-play RMSE, a best-of-many sampled trajectory score, or RMSE averaged across
unequal folds. Pooled scores sum squared errors and coordinate counts first.
Long plays and players with more requested frames contribute more coordinates.
Supporting ADE, FDE, tail error and role/time slices explain failure modes; they
do not replace this metric.

The [organizer's Prediction data contract](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
separates pre-throw player tracking from post-throw coordinate targets. Landing
coordinates, output horizon, player role and prediction flags are supplied
inputs. The test files illustrate the interface; live evaluation uses the
prediction API. Supplementary analytics fields are not automatically inference
features. This report treats future player positions, catch outcome, actual
coverage annotations after the throw and future observations of the ball as
unavailable inputs. All observed history is aligned by actual frame number.

Under squared error, the optimal point forecast is the conditional mean of each
coordinate. A visually plausible sample can score worse than that mean. A
multimodal model therefore needs a justified point-output rule and official
RMSE evaluation; success on oracle best-of-20 ADE does not establish progress
toward 0.46.

## What the strongest directly relevant evidence says

The evidence search covered organizer documentation and released winning
methods, NFL/AWS forecasting and coverage research, American-football movement
models, recent sports trajectory representations, and transferable multi-agent
forecasting work. Primary papers, author writeups and released code take
precedence over tutorials and model-name claims. Searches included publications
after the competition, through the research date. This is a structured evidence
review, not a claim that every unpublished method or proprietary dataset has
been discovered.

### Same competition: the most useful comparison

The [first-place writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
uses a compact sequence representation: position, orientation components,
decomposed speed/direction, and relative ball/receiver geometry. It combines a
feature-wise temporal convolutional stem with player attention and joint future
trajectory output. Auxiliary motion targets and consistent spatial and temporal
augmentation support representation learning. Its large ensemble and shuffled
game-fold validation differ from this project's single-seed chronological
experiments. The useful hypothesis is efficient temporal feature learning, not
that copying an ensemble score gives this project the same result.

The [author's released training notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)
is more specific than the prose: grouped temporal convolutions, three attention
blocks, and auxiliary velocity/acceleration outputs are implemented. Thirty-five
epochs repeat the training data six times: 210 passes, versus our original 40.
Author-reported private results for feature configurations are 0.46471 for the
ten-channel configuration, 0.46511 after removing dynamic ball/receiver offsets,
and 0.46946 with extra rotated coordinates. Static landmarks remain available in
the reduced variant. These are author reports, not our reproduced results. The
code excludes unusually long and missing-passer plays and uses a 48-frame
output. A faithful project comparison must retain every declared evaluation
row and handle longer horizons; it must also fit normalization within each
training fold rather than import the notebook's precomputed statistics.

The [second-place solution](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/2nd-place-solution)
reports mostly raw inputs, a recurrent temporal encoder, spatial interaction
processing, and prediction of frame displacements followed by cumulative
summation. Its author found elaborate features and an input Transformer
unhelpful, used older-season data directly, and held out the last five weeks
for rapid iteration. EMA and reflection helped. This conflicts with treating
one encoder family as universally superior and supports comparing learned
motion representations. Its time-decayed Huber training objective differs from
official evaluation; any such experiment here must keep unweighted coordinate
RMSE as the primary outcome.

The [third-place solution](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution)
combines compact raw/kinematic/role information with player interactions and
older NFL tracking data. Auxiliary prediction tasks and an initial simpler
representation are part of its learned-feature approach. The writeup also
reports that several additional geometric features did not help. Earlier
tracking can expand the representation-learning sample, but event alignment,
player-role construction and season-specific measurement quality need auditing.
Past training labels may define a pretraining task; they cannot become future
inference inputs or include the evaluation games.

**Inference from these three independent author accounts:** compact,
well-conditioned motion inputs plus learned temporal and interaction features
deserve higher priority than another unstructured expansion of the tree bank.
The authors disagree on the best temporal encoder and on pretraining versus
direct data pooling. Those disagreements are reasons for controlled experiments,
not grounds to choose whichever description sounds newest.

### NFL-specific mechanisms and their limits

| Primary evidence | Mechanism relevant here | What the evidence does not establish |
|---|---|---|
| [NFL/AWS Defender CNN-LSTM](https://cdn.amazon.science/8f/31/de231564410aa55e346d02c34c12/prediction-of-defensive-player-trajectories-in-nfl-games-with-defender-cnn-lstm-model.pdf), 2021 | Defender motion depends on receiver and ball context; decomposed kinematics and temporal/social representations are useful hypotheses. | Some experiments receive later receiver/ball context and filter tracking anomalies. Their information budget and evaluation population differ from this competition. |
| [Explainable defense coverage classification](https://cdn.amazon.science/a0/66/1bffb8464526b0087d52aee271d9/explainable-defense-coverage-classification-in-nfl-games-using-deep-neural-networks.pdf), 2023 | Coverage is a temporal, multi-player pattern; attention can represent assignment ambiguity. | Coverage-class accuracy is not post-throw coordinate RMSE. Training annotations are not supplied by our inference interface. |
| [Dutta, Yurko and Ventura: unsupervised pass coverage](https://arxiv.org/html/1906.11373v3), 2020 version | Relative separation, direction agreement and their variation help characterize man/zone behavior. | A cluster or nearest-opponent identity is a proxy, not a verified coverage label. Full-play features must be restricted to the observed window. |
| [NFL/AWS factorized coverage-responsibility Transformer](https://arxiv.org/html/2603.25901v1), March 2026 | Factorized temporal/player attention models responsibility, switches and contextual matchups beyond a nearest-defender heuristic. | The work uses coverage annotations and contextual inputs unavailable here; later-cutoff experiments cannot be transplanted into pre-throw inference. Its task is responsibility classification. |
| [NFL Ghosts](https://arxiv.org/html/2406.17220v2), revised 2025 | Conditional defender positioning and uncertainty depend on offensive/defensive geometry. | Its catch-time conditioning and play-value evaluation are different from forecasting all future coordinates at throw time. |
| [Nguyen and Yurko: Bayesian step-and-turn models](https://arxiv.org/html/2603.17866v2), March 2026 | Step persistence, turning persistence, speed-dependent turning variability, relative neighbors and player effects motivate motion/interaction features. | The paper studies ball carriers on run plays and one-step alternatives with other players' observed positions. It is not a multi-second pass-play forecasting benchmark. |

The coverage findings justify exploring **learned, uncertain matchup
representations**, including assignment probabilities and temporal changes. They
do not justify declaring the nearest defender to be the responsible defender.
Our 147 explicit coverage candidates summarize selected neighbors' synchronized
histories. Their failure can reflect noisy assignment, redundancy or a poor
interface to the frozen network. It does not contradict the football mechanism.

### Recent sports AI and transferable forecasting methods

| Primary evidence | Useful hypothesis | Comparability/availability boundary |
|---|---|---|
| [PlayGen-MoG](https://arxiv.org/html/2604.02447v1), April 2026 | Relative attention and shared latent scenarios represent coordinated NFL movement. | Generates offensive trajectories from formation context; diversity and ADE/FDE do not directly imply better deterministic pass-play RMSE. |
| [Hidden Context in Dynamic Movement Forecasting](https://arxiv.org/html/2605.14855v1), May 2026 | Velocity, player relationships and landmarks can make a temporal representation more effective than a generic Transformer. | NBA task, sampling, units and horizon differ. Its encoder ranking is not an NFL ranking. |
| [AdaSports-Traj](https://arxiv.org/html/2509.16095v1), September 2025 | Role/domain conditioning can support representation transfer across sports. | Trajectory completion and cross-sport adaptation do not establish performance with our throw-time information restriction. |
| [TacticGen](https://arxiv.org/html/2604.18210v1), April 2026 | Context and event encoders, coordinated player attention and large-scale trajectory pretraining motivate richer learned features. | Association football, largely private data, and best-of-20 trajectory metrics. Its strongest conditional variant receives the complete future ball path. These scores cannot be called NFL RMSE. |
| [Wayformer](https://arxiv.org/abs/2207.05844), 2022 | Factorized attention offers a controlled way to combine temporal, agent and context information. | Autonomous-driving data and map semantics; no NFL performance claim follows. |
| [QCNeXt](https://arxiv.org/abs/2306.10508), 2023 | Relative, multi-agent trajectory representations motivate joint prediction and coordinate consistency. | Road-vehicle dynamics and multimodal evaluation differ; importing weights or features needs a transfer test. |

Generative diffusion, mixtures and foundation models remain research avenues.
They are lower immediate priority than motion conditioning and historical NFL
data alignment because their published objectives and available data often do
not match our objective. This is a cost/signal decision, not a claim that newer
methods cannot work.

## Feature-space coverage ledger

Every proposal below has an information-time boundary. “Open” means a plausible
avenue is not yet scientifically exhausted. “Tested” refers to the named
representation, not every possible implementation of a football mechanism.

| Feature avenue | Why it may help | Availability and leakage control | Current disposition |
|---|---|---|---|
| Supplied velocity vector | Direct movement direction and magnitude without differencing noise | Pre-throw speed and direction; preserve missingness | Motion screen supports follow-up; isolate from derived velocity |
| Position-derived velocity/acceleration | Independent motion estimate; disagreement measures reliability | Backward differences with actual time gaps only | Existing tree and attention; compare causal smoothing |
| Multi-scale smoothed state | Noise reduction without erasing route breaks | Fit within the observed play; 3/8/20-frame windows | Thirty state summaries help in both addition/removal tests, across all three folds |
| Braking, turning and path efficiency | Predict continuation versus a cut | Observed changes only; circular angles | Standalone benefit; no reliable incremental benefit within the full motion bank |
| Orientation versus movement | Backpedaling and body alignment may distinguish defender intent | Supplied orientation with explicit validity | No reliable benefit in addition/removal comparisons |
| Motion-to-baseline discrepancy | Tells the learner how physics and the fitted prior disagree at each query time | Observed state plus fold-fitted baseline | No reliable benefit in addition/removal comparisons |
| Time/fraction/remaining time | Movement depends on both elapsed time and pass arrival | Organizer-supplied horizon and requested frame | Existing representation; verify long-horizon behavior |
| Ball-relative displacement and speed | Required movement and approach direction | Supplied landing point, never observed future player location | Existing strong tree mechanism; neural interactions remain open |
| Arrival feasibility | Running speed, turning and reaction time constrain reachable positions | Observed state and supplied horizon; thresholds are hypotheses | 87 new soft-arrival terms show no reliable incremental benefit in this screen |
| Boundary-value trajectory bases | Provide nonlinear paths with arrival/turning structure | Derived from allowed state; never forced labels | Quadratic, cubic and quintic candidates tested conditionally; no clear group gain |
| Nearest-opponent/teammate state | Immediate spacing and relative velocity constrain reaction | Observed geometry, with missing-player masks | Existing features; do not equate proximity to assignment |
| Dynamic coverage summaries | Separation variation and movement coupling can reveal tracking versus zone behavior | Synchronized observed histories only | 147 candidates hurt in both addition and removal tests |
| Learned matchup probability | Uncertain assignments and handoffs may be more useful than hard neighbors | Train on lawful prior data or self-supervised relations; no unavailable PFF labels | Open, higher priority than adding more nearest-neighbor summaries |
| Relative rank, local density and space | Congestion, leverage and available lanes | Current observed players; train-only normalization if fitted | Existing tree ranks/pools; new density summaries tested; neural ablations remain open |
| Pass-axis and field-boundary geometry | Sidelines/end zones restrict feasible movement | Known field and supplied landing point | 39 new terms show no reliable incremental benefit in this screen |
| Route phase and break timing | Receiver intent changes during a route | Observed curvature/change points; no full-route label at inference | Existing rolling/path summaries; learned phase embedding remains open |
| Categorical role and side | Offensive and defensive motion has different conditional structure | Organizer fields; unseen category must remain explicit | Implemented; exact organizer role vocabulary needs a separate contract audit |
| Player/role frequency | Sample support and familiarity affect reliability | Counts strictly before the current game date | Implemented; retained in the target-statistic ablation |
| Player/role target encoding | Historic systematic motion residuals may predict future adjustment | Entire current date excluded; shrinkage; frozen validation lookup | Six earlier-date target statistics improved the matched attention comparison across three folds |
| Conditional entity priors | Motion habits may depend on horizon, role or direction | Hierarchical shrinkage fitted only on earlier dates | Open; requires adequate counts and comparison against current player/role priors |
| Anthropometrics/position | Physical capability and role constraints | Provided input metadata, missingness, train-only fitting | Existing tree metadata search; no demonstrated reason to add a large neural branch yet |
| Player embeddings | Learned identity effects may complement motion statistics | Train fold vocabulary; cold-start fallback; no future-season fitting | Open, but high cardinality and temporal drift require matched cold-player tests |
| Coaching/team organization | Coverage tendencies may persist across games | Verified as-of roster/team/coach joins and an inference contract | Deferred, not silently invented from IDs or future rosters |
| Opponent-adjusted history | Separate player behavior from opponents faced | Strictly earlier-date opponent information and regularization | Open only after reproducible team/matchup availability is established |
| External ratings/rankings | Could summarize established ability | Timestamped source published before each game | Deferred: weak direct link to short-horizon residual motion and unverified joins |
| Weather/stadium/surface | Friction or visibility could modify movement | Reproducible game-time metadata available at inference | Lower priority; conditional motion may already capture much of the effect |
| Missingness/observation age | Stale or partial tracks are less trustworthy | Actual observed frames and telemetry validity | Implemented and counterfactually tested |
| Learned temporal features | Compress useful motion patterns without thousands of aggregates | Fit separately in each training fold | Existing attention model tested; compact grouped-convolution and recurrent encoders remain open |
| Auxiliary motion learning | Forces embeddings to preserve velocity/acceleration and temporal consistency | Training targets only; no validation-target feature generation | High-priority learned-feature experiment supported by released winning methods |
| Forecast-origin augmentation | Increases examples of useful motion transitions | Earlier observed cutoffs; transform every input/target/clock consistently | Open; must not leak later inputs or change the true throw-time evaluation |
| Older NFL tracking | More route/coverage examples and season diversity | Licensed earlier-season data, audited event alignment and role derivation | High priority; not yet an executed cross-season representation study |
| Cross-sport pretraining | Generic interaction/movement structure may transfer | Legal data and completely separate NFL evaluation games | Open but lower priority than same-domain historical data |
| NLP/LLM-derived context | Timestamped scouting text might summarize stable tendencies | No play-description or future-outcome text is supplied at inference; any external text needs an as-of join | Low immediate priority; use LLMs for research assistance, not invented motion labels or unavailable play narratives |
| Multimodal intent | More than one trajectory can be plausible | Learned from training futures; probability-weighted point prediction | Open; evaluate coordinate mean, calibration and RMSE, never oracle selection |

This ledger makes the remaining work visible. It does **not** support closing
feature engineering. In particular, learned temporal motion, auxiliary tasks,
historical NFL data and probabilistic matchup features have not reached a
demonstrated point of diminishing returns.

## Executed domain screen

The [frozen experiment protocol](DOMAIN_EXPERIMENT.md) specifies three expanding
chronological folds wholly inside the original 192 training games. Their later
evaluation windows cover 98 games and 202,361 requested rows. Each fold uses its
own earlier-date encodings, fitted baseline and saved final-EMA attention
checkpoint. The original 32-game development partition, previously inspected
48-game reserved partition, and Kaggle are not used for this new screen.

Each correction head receives the frozen attention decoder's inputs and
prediction, with role indicators, as 133 control columns. It has two hidden
layers of widths 64 and 32. All ten arms have 463 inputs and 31,842 parameters;
excluded families are set to zero after training-only normalization. Training
uses identical initialization, row ordering and twelve-epoch schedules. There
is no validation-checkpoint or ensemble-weight selection. Training corrections
use the reference model's in-sample residuals, an important distribution-shift
limitation shared by every arm.

| New family | Generated | Main question |
|---|---:|---|
| Motion state and physical propagation | 57 | Does explicit reliable motion correct the learned state? |
| Arrival constraints and soft paths | 87 | Does explicit feasibility add information beyond existing landing inputs? |
| Coverage dynamics | 147 | Do observed matchup summaries add useful temporal context? |
| Field/pass geometry | 39 | Do boundaries and pass-relative coordinates improve prediction? |
| Total | 330 | Same-capacity addition and removal experiment |

Every fold retains 313 candidates and rejects 17: eleven constants and six exact
duplicates, detected on training rows only. Retention is an unsupervised screen,
not proof of predictive usefulness. The maximum evaluation clipping fraction
for any standardized column is below 0.49%. No target is clipped and no difficult
evaluation row is removed. The complete per-column reasons are in the
[feature catalog](results/domain_feature_catalog.csv).

| Model on the same evaluation rows | Fold 1 | Fold 2 | Fold 3 | Pooled RMSE |
|---|---:|---:|---:|---:|
| Existing tree | 0.698510 | 0.672092 | 0.735295 | 0.702601 |
| Existing attention | 0.698549 | 0.670412 | 0.792132 | 0.720631 |
| Matched correction control | 0.708373 | 0.696867 | 0.787261 | 0.730023 |
| Motion family alone | 0.673035 | 0.664349 | 0.686702 | 0.674773 |
| All four feature families | 0.669366 | 0.698585 | 0.733842 | 0.697658 |
| All features, equal blend with tree | 0.630429 | 0.619527 | 0.676110 | 0.641603 |

The correction control itself is worse than the frozen attention model. This
matters: feature gains against that control cannot all be described as gains
against the original deployed model. The all-feature model improves the matched
control by 4.43% pooled but worsens fold 2, so it **fails** the predeclared
all-fold consistency gate. The motion-only arm improves every fold and scores
better than both the frozen attention and tree on the pooled rows.

The full-feature/tree blend is a declared secondary diagnostic. Its 0.641603 is
better than the earlier equal attention/tree blend's 0.678661 on the same rows.
It remains far from 0.46 and does not independently validate the adaptively
researched feature bank.

| Family | Addition: RMSE change | Removal-based inclusion effect | Simultaneous-interval interpretation |
|---|---:|---:|---|
| Motion state | -0.055250 | -0.046538 | Both favor inclusion; intervals exclude zero |
| Arrival constraints | -0.001293 | -0.003688 | Both uncertain |
| Coverage dynamics | +0.029758 | +0.023038 | Both favor exclusion; intervals exclude zero |
| Field geometry | +0.006255 | +0.000952 | No reliable incremental benefit |

Negative changes favor including the family. For the eight family comparisons,
10,000 paired whole-game bootstrap replicates produce Bonferroni simultaneous
95% intervals, each 99.375% individually. They address multiplicity within this
screen, not the entire history of adaptive project decisions. Exact intervals,
slices, curves and artifact hashes are in the
[executed results](results/domain_research.json).

## Motion follow-up and long-horizon diagnosis

The [adaptive motion protocol](MOTION_EXPERIMENT.md) partitions the 57 motion
features into five disjoint mechanisms and tests each both alone and by
removal. It reuses the complete-motion and control fits rather than treating
them as new experiments. All follow-up outcomes, including weak or negative
ones, belong in [the motion results](results/motion_research.json).

The follow-up completes thirty additional fits. **Smoothed observed state is the
only subfamily whose addition and removal both favor inclusion in every fold.**
It also passes both simultaneous-interval comparisons. This narrows the useful
mechanism beyond the broad motion label.

| Motion mechanism | Addition RMSE change | Removal-based inclusion effect | Interpretation |
|---|---:|---:|---|
| Instantaneous state/reliability | -0.014968 | -0.001347 | Helpful alone; incremental value uncertain |
| Facing alignment | +0.003895 | +0.004825 | No reliable benefit |
| Smoothed observed state | -0.043773 | -0.011542 | Both effects favor inclusion across all folds |
| Maneuvers | -0.028394 | +0.006617 | Helpful alone; redundant or adverse point effect within the full bank |
| Physical propagation | +0.004064 | +0.001152 | No reliable benefit |

These ten comparisons use simultaneous 95% intervals (99.5% individually).
The complete-motion/tree equal blend scores **0.644086**, versus the previous
attention/tree blend's **0.678661**, with a paired 95% RMSE-difference interval
of **-0.042293 to -0.027366 yards**. The all-feature blend's 0.641603 is slightly
lower, but the full feature bank's standalone consistency gate still fails.
Neither blend is a new independent test result.

The publication contains 60 distinct scientific fits. A separate 30-fit
reproduction verifies a local typing correction required by the quality gate;
its numerical syntax tree, all prediction files, learning curves and feature
caches match the original execution. It is recorded as verification overhead,
not additional scientific evidence. Both completed studies also pass restart
checks that reuse all saved fits without a new training epoch.

The initial error-concentration audit uses **training rows only**. Forecasts
after two seconds constitute approximately 3.8% of those rows but 39–44% of the
frozen attention model's squared error. One training play, 2023091100/3167,
contributes about 10–19% by itself across the three fitted models. This identifies
a concrete risk: the representation and extrapolating residual parameterization
may spend excessive capacity on a small set of long-horizon trajectories.

The [reproducible training-play audit](results/motion_training_audit.json) finds
94 future frames (9.4 seconds), eight scored players and 752 requested rows for
that play. Its observed target coordinate bounds remain inside the field and
its largest consecutive-position speed is about 7.23 yards/second. Constant
velocity scores 14.71 yards RMSE and the fold ridge baseline 10.21 yards on this
play. A large residual therefore does not by itself establish corrupted
telemetry: the long horizon and extrapolation are concrete issues to address.

The correct response is to inspect those tracks, their units, events, horizon,
telemetry and baseline behavior, then test a defensible motion representation.
Removing difficult validation plays would change the problem. Training-only
robustness or anomaly handling, if justified, needs a frozen rule and an
all-row evaluation. A rule must not be chosen merely because a specific
validation label looks inconvenient.

## Target encoding: implemented safely, not universally solved

Player and role encodings summarize earlier motion residuals with shrinkage.
Training excludes **all games on the current date**, not just the current row
or player. Validation uses a frozen training lookup. Counts and cold-start flags
are distinguished from target-derived means and dispersion. No validation
outcome updates the encoder.

The preceding matched chronological experiment removed six target-derived
statistics while preserving the counts. Pooled attention RMSE changed from
0.720631 with them to 0.774142 without them; all three fold point estimates
favored inclusion. That supports their value in this architecture, despite the
full attention model still underperforming the tree pooled. It does not prove
that every possible category should be target encoded or that entity history
will survive season changes. Conditional/hierarchical extensions need their
own earlier-date, cold-start and season-drift evidence.

## Research priorities and the completion boundary

The next decisions should follow evidence rather than novelty claims:

1. Isolate the useful motion mechanisms and inspect long-horizon training
   failures. Then put the supported information into an end-to-end temporal
   representation, preserving a matched feature control.
2. Compare compact learned motion features—feature-wise temporal convolutions
   and a recurrent alternative—with auxiliary motion targets and consistent
   forecast-origin augmentation. Keep changes attributable instead of changing
   features, loss, encoder, training duration and split simultaneously.
3. Audit and align permitted older NFL tracking, then compare same-domain
   representation learning against the exact current-data control. Record
   season, event and role construction differences.
4. Test learned matchup/coverage representations with uncertainty and observed
   temporal inputs. The present negative nearest-neighbor summary result is a
   reason to improve the representation, not to erase coverage from the domain.
5. Only after these avenues show stable marginal returns should final capacity,
   ensemble and deployment optimization resume. Model selection must still
   distinguish reused development evidence from an independent assessment.

No measured result in this report establishes 0.46, exhaustive feature research,
or a finished 9.9/10 project. The concrete progress is a source-backed domain
map, executable controlled features, measured positive and negative family
effects, and a substantially improved internal blend diagnostic. The substantial
remaining gap is explicit and measurable.
