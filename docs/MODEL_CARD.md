# Research model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026
Prediction. Inputs are pre-throw tracking, player roles, the supplied ball landing
point, and requested forecast horizon. This is not a system that infers an unknown
ball landing point at release time.

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
Three chronological inner folds select feature variants. The final 48 games remain
unscored. One labelled season and repeatedly inspected development data limit
generalization claims. There is no verified leaderboard rank.

## Current inference artifact

Final preprocessing has separately been refitted on 224 games: the physical
baseline, chronological histories, and route encoder. The new components are
not yet paired with fully trained final residual trees or an export. The final
fitter passes synthetic numerical recovery and a 15-week raw feature check;
full-scale fitting remains pending. The inference artifact
described below remains the verified 192-game research fit, so its development
score is not relabelled as a final-model result.

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

## Intended review and next gate

Use the notebooks to assess football reasoning, leakage prevention, controlled
feature gains, engineering, and reproducibility. This is a research artifact,
not a certified production or player-evaluation system. Feature research now
passes all 15 closure criteria. Complete wide-profile group refits identify no
omission meeting the predeclared follow-up threshold; the combined removal of
direct histories and forecast crosses improves pooled inner RMSE by just 0.170%,
with mixed fold results. The final width gain is 0.071%.

The feature and refit-column manifest is frozen. Final refitting, one-time
reserved holdout evaluation, and validation of the final inference artifact
remain the next phase. The measured scores above are research results.
