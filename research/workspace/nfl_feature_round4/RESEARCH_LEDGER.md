# Feature research ledger · evidence before scale

## Current decision

Round 3 completed, but none of its three feature comparisons passed. Keep those negative results; do not spend additional folds on the same wide pair representation. The next package makes one narrow test of a source-verified representation issue, with four new coordinate fits. It is not another large feature search.

## Research sources and boundaries

**Competition winner's own write-up.** The author uses recent position, orientation, velocity and ball-/receiver-relative inputs, with separate static role/horizon information. The solution also uses a learned temporal/player architecture, auxiliary training objectives, augmentation and ensembling. This supports examining representation quality; it does not establish that algorithms or training are irrelevant to the remaining gap.

Source: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution

**Fourth-place author's own write-up.** The approach uses target-specific player context and temporal/inter-player processing. Our failed flattened pair features do not test its learned player-selection mechanism. Do not copy its horizon truncation into this project or remove requested rows.

Source: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814

**Official evaluation.** Use the organizer's two-coordinate RMSE and all required prediction rows. Relative gains only make sense within a matched evaluation population. No current package establishes a hidden-test score.

Source: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview

**Estimator documentation.** Histogram gradient boosting bins individual inputs and learns tree splits. Providing direct state in addition to time-dependent products can change what a finite-capacity estimator learns, even without adding raw information. The claimed benefit is our hypothesis, not a documented performance guarantee. We keep the already verified scikit-learn 1.8.0 runtime; current documentation is not an instruction to upgrade it.

Source: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

## Investigations and disposition

| Family | Evidence currently available | Disposition |
|---|---|---|
| Required arrival response | Helped the small matched ridge study; did not help the later fixed-tree first fold | Conditional evidence only; not a universally retained feature |
| Earlier prediction-origin mixture | Failed its tested small-ridge comparison | Stop that precise mixture unchanged; do not tune against the same folds |
| Turning/braking/orientation response additions | Turning and braking worsened the tested ridge score; orientation gain negligible | Stop those exact additions; cannot generalize to every temporal encoder |
| Flattened terminal and temporal pair inputs | Failed Round 3 fixed-tree comparison | No automatic fold replication |
| Unmultiplied observed state | Source audit: all 62 fields are currently time-multiplied in the diagnostic control | Implemented and tested in Round 4; private NFL benefit unmeasured |
| Compact landing-frame geometry | Radial/lateral speed, endpoint miss, closest approach and receiver-relative comparisons | Implemented and tested in Round 4; overlaps known domain concepts, not new raw signals |
| Learned synchronized player context | Earlier implementation prototypes exist; current small flattened-tree result is not a learned-attention test | Open; requires replayable temporal baseline, capacity/exposure-matched integration and bounded throughput/recovery tests |
| Continuous trajectory/velocity parameterization | Earlier supervision results exist; current full-model lineage and test comparability remain separate | Open; measure against a reproducible model, not an unavailable historical checkpoint |
| Chronologically learned player/role history | Earlier repository studies report conditional value | Do not rebuild blindly; recover exact prior lineage and ablate in the intended model using fold-safe tables |
| Additional labeled seasons/context | Availability and legal/inference-time parity require verification | No new download or label source in this package |

## What follows the next report

A passing Round 4 representation first needs appropriate chronological replication and a controlled integration into the intended replayable forecasting model. An exploratory gain in 16 reused games is not a release. If both additions fail, stop this interface unchanged; review a learned sequence/relationship integration rather than extending an arbitrary tabular column list.

No post-hoc selection of roles, difficult-game removal, horizon clipping, threshold adjustment, guessed coverage labels or repeated tuning against the inspected outer holdout. Feature engineering is not complete until the major plausible families have measured contributions, exclusions and stability evidence.

The historical target 0.46340 remains a serious research objective, not a promised outcome. These studies cannot apportion the remaining leaderboard gap solely to features, model architecture, loss, data, validation or ensemble size without further matched evidence.
