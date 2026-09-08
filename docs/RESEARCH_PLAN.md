# Feature research and completion gate

Feature engineering is an explicit research phase. Notebook 02 is not complete
because features can be computed or a model can fit. Final refitting and the
reserved holdout require an evidence-based closure decision.

## Protocol

Keep every game intact. Use 192 training games and 32 later development games;
reserve the last 48 games. Three expanding chronological inner splits use
94/41, 135/28, and 163/29 training/evaluation games. Their evaluation sizes are
83,938, 57,177, and 61,246 forecast frames. Each fold refits the physical baseline,
historical encodings, route components, selectors, normalizers, and residual fit.

Choose variants by pooled inner-fold squared coordinate error. Development
reports paired comparisons after selection. Development has been inspected
during research and is not an untouched test set. No across-season accuracy
claim is supported by this single labelled season.

## Candidate coverage

| Bank | Candidates | Families and rationale |
|---|---:|---|
| Core research | 5,387 | Motion, lags, observed summaries, landing/peer geometry, forecast interactions, multiscale motion, smooth transforms, role responses, projections, earlier-date player/role residual histories |
| Context | 1,468 | Body/position metadata, robust path distributions, matched histories, peer ranks, required arrival motion |
| Representation | 1,144 | Geometric player-set pools, role-conditioned destinations, train-only route components and prototypes |
| Total | **7,999** | **20 families; 6,875 nonconstant on development-training rows** |

All candidates have a named family, rationale, and availability field in the
catalog. Rolling statistics use observed frames only. Matched histories join
exact frame IDs; opponent/receiver anchors are chosen from the terminal observed
state. Player/role target histories use raw constant-velocity residuals from
strictly earlier dates, smoothing 20, and a frozen evaluation lookup. Historical
sample counts provide frequency information and cold-start handling. Body age
is computed at game date. Missingness masks are explicit. Peer percentiles are
within the observed play, not fitted on evaluation outcomes.

Role crosses, ratios, products, differences, nonlinear transforms, time gates,
horizon-normalized fractions, and geometric norms search useful interactions.
Route descriptors fit 16 components and eight prototypes to position-derived
pre-throw histories inside each training fold. No pretrained external embedding
or outcome-derived season aggregate is used.

## Controlled experiments

Preserve the 64-feature landing parent before adding conditional corrections.
The initial 64-column interaction challenger displaced 23 parent features,
so its poorer score was a confounded comparison. The expanded experiments
retain the parent and test both additions and strict omissions.

Use fixed ridge regularization 0.01 for linear comparisons. The nonlinear
diagnostic uses 100 iterations, depth 4, 15 leaves, learning rate 0.07,
minimum leaf 60, L2 1, and 63 bins; early stopping is disabled.
Do not tune this estimator to compensate for weak features.
Independent train/evaluation noise is a negative control, excluded from selection.

Screen all candidates using training-only residual associations and variance.
Detect repeated names and redundant columns. Wider budgets preserve the
250-feature union, interleave families from a ranked 3,584-column pool, and
remove new columns correlated at 0.9995 on 8,192 deterministic training rows.
Compare 512, 1,024, and 2,048 features at the same estimator capacity.

Strict family removals refit without replacement columns. Family permutations
move entire trajectories within role/horizon with exact forecast-frame alignment
and a fixed physical baseline. Their five-seed range is shuffle variation, not
a confidence interval or causal effect. Paired game bootstraps quantify
development-game uncertainty. Report individual selection overlap and family
fold stability; PCA names are not fixed semantic axes across folds.

## Availability and deployment representation

Test all 7,999 candidate values under removal of optional fields against the
declared dependency contracts. Features marked positional must actually remain
unchanged. An earlier zero-fill stress test exposed severe telemetry dependence;
the corrected pipeline chooses independently fitted omission profiles.

Joint linear refits separate feature value from the limitation of sequential
coefficients. Every raw development frame must reproduce the selected fit.
The organizer gateway test uses its unchanged source and unlabelled sample.
A sample pass is not an accuracy or leaderboard result.

The 236-feature linear bundle and 212-feature positional fallback are the
current research inference path. The wider tree is a diagnostic until its own
raw-feature/standalone inference has equivalent validation.

## Remaining avenues and stopping decision

| Avenue | Current decision |
|---|---|
| Wider screened representations | Test through 2,048 retained columns; review incremental fold gains before stopping |
| Robustness of the strongest wide representation | Refit without metadata/history and without optional telemetry |
| Training-only selection and redundancy | Implemented; preserve family and fold evidence |
| External team ratings, coaching, organization, strength of schedule | Deferred: no verified as-of join, availability contract, or demonstrated relation to this frame-level task |
| Player identities and historical outcomes | Earlier-date smoothed residual/count features only; no full-season target means |
| Learned route/player interaction representations | Train-only route components/prototypes and geometric pools implemented |
| Pretrained foundation models / high-capacity sequence tuning | Not a prerequisite for feature attribution; considered only after the feature gate |
| Multi-season stability | Not established; 2024 unlabelled gateway samples cannot supply it |

Close feature engineering only when the major realistic families have explicit
evidence, fixed-estimator feature gains are robust, the latest useful representation
has a validated inference path, and remaining plausible feature gains are small.
There is no claim that a finite search proves every possible feature exhausted.
Current status: **open pending width, robustness, and representation handoff review**.

After closure, freeze the feature/selection manifest, refit on authorized training
partitions, evaluate the reserved holdout once, finalize model and data cards,
and verify the final inference artifact. Additional model complexity must earn
its place under the same validation protocol.
