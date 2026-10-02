# Model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026 Prediction. Inputs are observed tracking, player roles, the supplied ball landing point, and requested forecast horizon.

## Current competition-facing neural ensemble — October 2026

The strongest measured deployment remains the **20-model multisplit ensemble** spanning four grouped-CV split families.

| Evidence | Coordinate RMSE |
|---|---:|
| Multisplit local OOF, 561,607 rows | 0.4631723213 |
| Seven-model private submission | 0.46547 |
| **Multisplit-20 private submission** | **0.46487** |
| Published first-place private comparator | 0.46340 |

The latest private result remains **0.00147 RMSE** behind the published first-place comparator. These are late post-competition measurements; no official competition rank is claimed.

### Post-PR40 frontier research

The project has now completed a broad set of controlled studies beyond the earlier dual-path sequence.

| Candidate family | Employer-facing aggregate result | Decision |
|---|---|---|
| Competition-only pseudo-supervision | Completed controlled variants without confirmation-stage promotion | No promotion |
| ST-GRU / landing-node ST-GRU | Standalone performance far behind the incumbent | No promotion |
| Multisplit meta/post-processing | Best pooled OOF gain ~0.000049 RMSE | No promotion |
| Frozen-parent feature adapters | Best fixed-blend gain ~0.00017 RMSE | No promotion |
| Coverage/physics full-parent fine-tune | Nearly indistinguishable from matched control | No promotion |
| Source-native configurations | Useful point signal but below locked gates | No promotion |
| Direct interaction / role-specific heads | Did not beat live matched control | No promotion |
| Single-target / defender / route variants | Fixed-blend gains remained ~0.00012–0.00013 | No promotion |
| Five-fold incumbent TTA | Base-family gain up to 0.001604; multisplit hybrid gain only ~0.000034 | No promotion |

These results are development/OOF evidence, not private-leaderboard scores.

The main scientific conclusion is that **small fitted-parent modifications have largely plateaued, while independent configuration and split diversity remains the strongest transferred mechanism**.

### TTA interpretation

Five-fold TTA materially improved the standalone source-faithful base family, but the improvement was almost entirely redundant after the family was inserted back into multisplit-20.

That result is deliberately retained because it distinguishes a useful component-level technique from a useful final-system technique.

### GPU / runner engineering

The private AWS research workflow now uses bounded resumable runners with checkpoint recovery, structured resource/cost heartbeats, fail-closed integrity gates, workload-specific throughput benchmarks, AMP overflow handling, and device-safe EMA restoration.

The previously published inference result remains separate: shared preparation accelerated the fixed 20-model ensemble by **4.784×** with exact prediction parity on its declared benchmark sample.

### Current research direction

The next prepared mechanism is a **fresh independently trained source-faithful feature-configuration family**, not another fitted-parent adapter or fine-tune. It is **prepared but unmeasured** until AWS execution completes.

If a new configuration demonstrates full-OOF complementarity, the next step is repeated split-family scaling—the remaining direction most closely aligned with the documented large first-place ensemble.

Private competition data, fitted weights, large checkpoints, private runners, exact object locations, and unreleased feature/interaction transforms remain in AWS. GitHub contains aggregate evidence, selected protocols, public artifacts, and documentation.

---

## Earlier feature-engineered system retained for lineage

The sections below document the earlier ridge/boosting research line and sealed holdout evaluation. They remain useful evidence of feature engineering, validation, and inference engineering, but they are **not** the current strongest competition-facing model family.

## Measured research comparisons

| Representation and estimator | Development coordinate RMSE |
|---|---:|
| Original role-conditioned physical ridge | 0.9895688 |
| Original landing residual ridge, 64 features | 0.9268684 |
| Sequential core correction, 128 features | 0.900446 |
| Sequential context correction, 186 features | 0.8643361 |
| Joint linear profile without metadata, 236 features | 0.8222565 |
| Fixed shallow boosting, landing 64 | 0.8011972 |
| Same boosting settings, engineered union 250 | 0.7280160 |
| Same boosting settings, entire screened pool, 6,385 columns | 0.6869492 |
| Same boosting settings, selected metadata-free profile, 6,308 columns | **0.6880522** |
| Same boosting settings, positional fallback, 5,572 columns | 0.6939581 |

The current controlled tree comparison attributes a **14.12% RMSE reduction** to the feature
representation at fixed estimator settings. It does not attribute the difference
between role ridge and a tree entirely to feature engineering. Wider-budget
results and their training-only choice are reported in notebook 02. The compact
250-column union previously provided a 9.13% reduction under the same settings.
The selected profile's paired 95% game-bootstrap RMSE difference from the
64-column tree reference is −0.12780 to −0.09973 yards on development games.

All development comparisons use 32 games and 67,857 frames; game-level uncertainty
and paired differences accompany the full reports. Training comprises 192 games.
Three chronological inner folds select feature variants. The final model separately refits all 224 games and is evaluated on the reserved
48 games. One labelled season and repeatedly inspected development data limit
generalization claims. There is no verified leaderboard rank.

## Sealed final evaluation

The frozen final model scores **0.80466993 coordinate RMSE in yards** on 99,266
frames, 7,965 player/play trajectories, and 48 later games. Its predeclared
2,000-resample game-bootstrap 95% interval is **0.66313158–1.00112631**.
All predictions were sealed and checksum-verified in S3 before label access.
The official score was recomputed from keyed errors; a repeated evaluation
reused the same seal and preserved all output hashes and modification times.

Frame-weighted ADE is 0.59023183, trajectory-weighted ADE 0.45187051, trajectory
FDE 1.04067574, p95 displacement 2.07112117, and coordinate MAE 0.37592893 yards.
The refitted role baseline scores 1.07251911 and constant velocity 1.81762113;
the final model beats each baseline in all 48 games. These baseline comparisons
combine feature and algorithm effects. Feature-only attribution remains the
controlled development study above.

Metadata omission leaves predictions unchanged. Positional fallback scores
0.81821003; cold history scores 0.80469145. The latter is a small nonzero change
in the final refit, unlike the research fit's exact cold-history invariance.
Weekly RMSE is 0.96174637, 0.71397310, and 0.67801815. Coverage players score
0.88805355; targeted receivers score 0.54144962.

The higher holdout error must remain visible. Forecasts after three seconds
comprise 276 rows (0.28%) but 23.67% of total squared error. One play contributes
24.40%, and one game 28.03%. Sparse long horizons and concentrated game-level
error help explain the wide confidence interval. These are descriptive
post-evaluation findings, not evidence of a causal mechanism or permission to
retune on the holdout. No observations are removed from the official metric.

## Current inference artifact

Final preprocessing and both residual-tree profiles have been refitted on all
224 games. The final bundle contains the new physical baseline, chronological
histories, route encoder, and four coordinate models. Both final profiles use
951 active inputs after lossless pruning; their portable predictions match the
original fits exactly on all 463,670 training rows. The separate final exporter
passes the unchanged organizer gateway and verified checkpoint reuse.

The following development comparisons describe the earlier 192-game research
fit. Their scores and active-column counts retain their original lineage.

The research bundle selects the full-width metadata-free profile using the inner
validation protocol. Its frozen refit representation contains 6,308 screened
columns. Only 953 enter an actual tree split; the portable JSON bundle removes
unused columns and remaps split indices without altering predictions. A future
refit must start from the frozen 6,308 definitions, because a new fit can use
columns that the research trees did not use.

Missing telemetry selects an independently fitted positional profile with 5,572
screened columns and 981 active tree inputs. The bundle contains both profiles,
the physical baseline, frozen history tables, route transform, game manifests,
and source hashes. Numeric tree arrays support prediction without scikit-learn.
All eight inner/development profile conversions reproduce the fitted estimator
exactly, with maximum absolute coordinate difference zero on every evaluation row.

Complete-input raw replay reproduces 0.6880522 RMSE on all 67,857 development
frames. Missing metadata and cold player history return the same predictions;
missing telemetry selects the positional profile and scores 0.6939581. The
exporter resolves this validated tree artifact rather than the earlier linear fit.
An independent local run also passed the organizer's unchanged unlabelled gateway:
5,837 rows, 143 plays, unique requested identifiers, finite predictions, and exact
package/standalone parity at every callback.

## Robustness and limitations

Historical target-derived features exclude the complete current date and freeze
evaluation lookup tables. Route representations are fitted inside each training
fold. Body age uses game date; observed-frame histories cannot cross the throw.
Optional-field dependency tests compare actual feature values, not just feature
names. Future x/y columns in a request are ignored as inputs.

Raw inference validation covers every development frame plus missing metadata,
missing telemetry, and cold player history. The organizer sample gateway checks
interface shape, ordering, finiteness, and package/standalone parity; it has no
labels. Sample-year diversity does not establish multi-season accuracy.

Permutation importance expresses conditional model reliance, not causal football
effects. Correlated features reduce individual identifiability. Group refits remove
the named columns without replacement; derived information in other families can
remain. Removing direct history summaries, for example, does not remove every
forecast interaction derived from motion. These are conditional representation
ablations, not claims about eliminating a physical mechanism.

Metadata and route-only corrections are weak in some comparisons; failed avenues
are retained in the research record. The organizer follows request-file play order,
which does not guarantee chronological callbacks. Inference therefore uses frozen
training histories and does not learn from preceding evaluation plays. Private
competition data and fitted artifacts are not
redistributed in the public repository.

## Intended review and research limits

Use the notebooks to assess football reasoning, leakage prevention, controlled
feature gains, engineering, and reproducibility. This is a research artifact,
not a certified production or player-evaluation system. Feature research now
passes all 15 closure criteria. Complete wide-profile group refits identify no
omission meeting the predeclared follow-up threshold; the combined removal of
direct histories and forecast crosses improves pooled inner RMSE by just 0.170%,
with mixed fold results. The final width gain is 0.071%.

The feature and refit-column manifest remains frozen. Final refitting, reserved
evaluation, final inference validation, and error-concentration analysis are
complete. Independent labelled-season validation would strengthen external
validity; the present study cannot establish it or a leaderboard ranking.
