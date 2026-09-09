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
250-feature union, interleave families from the entire training-eligible pool, and
remove new columns correlated at 0.9995 on 8,192 deterministic training rows.
Compare budgets of 512, 1,024, 2,048, 4,096, and 8,192 features at the same
estimator capacity. The last budget exceeds the 7,999-column catalog and therefore
tests every eligible column that survives redundancy screening. Actual retained
widths are 6,386 / 6,382 / 6,382 on the inner folds and 6,385 on development-training
rows. Run the full-bank folds sequentially within
at least 128 GiB and checkpoint each completed fold. A 64 GiB attempt completed
all three inner folds but exceeded memory on the largest development fit; its
verified stage checkpoints preserve the completed work.

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

The current inference path uses the metadata-free wide tree with 6,308 screened
refit columns and 953 active exported inputs. Its positional fallback uses 5,572
screened columns and 981 active inputs. Lossless tree conversion reproduces all
eight inner/development profile predictions exactly. Raw replay covers all 67,857
development frames in each availability scenario: 0.68805 RMSE for complete,
metadata-free, and cold-history inputs; 0.69396 without optional telemetry.

## Remaining avenues and stopping decision

| Avenue | Current decision |
|---|---|
| Wider screened representations | Complete pool tested; final pooled inner improvement is 0.071%, with mixed folds |
| Robustness of the strongest wide representation | Metadata omission, positional refits, lossless conversion, and all-frame raw stress tests verified |
| Strict group removals on the selected wide profile | Complete across all three inner folds and development; no omission meets the predeclared follow-up criterion |
| Combined omission of direct histories and generic forecast crosses | Complete; 0.170% pooled inner gain with mixed folds, below the 0.5% follow-up threshold |
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
Current status: **closed after all 15 evidence criteria passed**. The source-bound
selection manifest is frozen for the final phase. Final refitting and the
48-game reserved holdout have not run.

The [wide ablation protocol](WIDE_ABLATION_PROTOCOL.md) extends strict refits to
the selected wide availability profile, covers all 20 catalog families in
12 disjoint groups, and tests whether removing a group is still a high-value
feature-engineering avenue. This adds a current-model requirement to the
original compact-union ablations.

The [combined-omission protocol](SIMPLIFICATION_PROTOCOL.md) adds one specific
follow-up after two direct groups each slightly improved all inner folds when
removed. It uses the same 0.5% pooled-gain and 1% maximum-fold-cost thresholds.
The completed study changes pooled inner RMSE from 0.7026010 to 0.7014072, a
0.170% gain. Per-fold relative RMSE changes are −0.298%, +0.324%, and −0.399%.
This mixed, small result does not meet the recorded threshold for another
representation search. Development results were excluded from that decision.

### Operational closure thresholds

Recorded before reviewing the full-pool results: require every candidate to be
screened in each training fold, complete eligible-pool coverage, and fixed-estimator
feature gains of at least 5% in each chronological inner fold. The final width
increment must improve pooled inner RMSE by less than 0.5%, with no individual
fold improving by 1% or more. These are explicit project stopping tolerances,
not statistical laws or a claim of universal feature optimality.

Metadata omission may cost at most 1% pooled inner RMSE and the independently
fitted positional fallback at most 5%. Require family ablations, trajectory
permutations, exact inference parity, all-frame availability stress tests, and
a pass through the organizer's unchanged unlabelled sample gateway. A failed
criterion keeps the gate open. Development intervals contextualize the decision;
the reserved holdout cannot be used to choose whether these criteria pass.

Raw replay also requires finite metrics and respects the same 1% metadata and
5% positional tolerances. Clearing player-history lookups receives the 1%
tolerance. That cold-history check is a development availability stress test,
not a player-disjoint refit or a cross-season evaluation.

After closure, freeze the feature/selection manifest, refit on authorized training
partitions, evaluate the reserved holdout once, finalize model and data cards,
and verify the final inference artifact. Additional model complexity must earn
its place under the same validation protocol.

The [final-phase protocol](FINAL_PROTOCOL.md) now implements input verification
and freezes both ordered refit schemas, the 224 training games, model settings,
and reporting rules. Preparation has been executed; final fitting and reserved
scoring remain subsequent milestones.
