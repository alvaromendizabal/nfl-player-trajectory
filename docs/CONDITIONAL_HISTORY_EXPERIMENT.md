# Conditional historical motion: bounded feature experiment

Declared 10 September 2026 before fitting or inspecting the new validation arms.
The feature gate remains open. This is a separately versioned experiment, not a
reconstruction of the missing motion-supervision model.

## Hypothesis and source basis

Earlier-date player/role motion summaries improved the matched attention model
from 0.77414 to 0.72063 across three chronological folds. Those summaries collapse
forecast time and the player's approach to the landing point. A coverage player
moving away from the pass and a receiver arriving near its endpoint have different
reasons to deviate from current velocity. Test whether conditioning historical
motion on those observed contexts supplies complementary information.

This is an original hypothesis derived from the project's measured historical
encoding result and the mechanism audit in [DOMAIN_RESEARCH.md](DOMAIN_RESEARCH.md).
It is not claimed to be a winning feature. The
[first-place author](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
uses compact motion and landing/receiver-relative inputs, temporal convolutions,
player attention, auxiliary motion losses and augmentation. The author does not
establish the value of the target encoding proposed here.
[Scikit-learn's target-encoding documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
explains why fitting encodings on their own labels overfits. Our date-ordered
encoding is stricter than random cross-fitting for this chronological task.

## Information and leakage contract

Twenty candidate columns: five statistics for each of four groups. The groups
are role × forecast-time bin, then that pair crossed separately with approach
speed, absolute speed, and fraction of supplied forecast horizon. Fixed boundaries
are 0.5/1/1.5/2/3 seconds; radial speed -1/1 yards per second; absolute speed
2.5/5/7.5 yards per second; and horizon fraction 1/3 and 2/3.

The historical target is actual displacement minus unfitted constant-velocity
displacement, divided by forecast seconds, expressed in landing-relative radial
and tangential axes. The cached baseline cancels when reconstructing actual
displacement. Each player/play/context bin contributes one averaged observation,
so repeated frames do not masquerade as independent support. Five outputs are
the two reconstructed canonical mean components, log support, reliability, and
vector dispersion. The coarse group shrinks to zero; finer groups shrink to its
prior using 20 pseudo-observations. No player identity is an input covariate.

Training features are emitted before any outcome on the current calendar date
updates the tables. Evaluation uses the frozen end-of-training state. Evaluation
labels are not passed to the feature builder. They are used only for scoring.
Unseen contexts back off to their earlier-date role/time prior. Identifiers align
entities and dates; their numeric values are never estimator covariates. Unit
tests cover same-date/future isolation, frozen inference, frame weighting,
reflection, permutation, sparse fallback and recovery serialization.

## Fixed experiment and stopping rule

Use the existing first chronological fold: 94 training games, 41 later evaluation
games, 83,938 evaluation frames. Keep the original development, inspected reserved
holdout and Kaggle outside this study. Reuse hash-verified parent inputs.

Three comparison arms with average-loss L2 = 0.01 and two CPU threads; reuse the
verified existing control and fit only the two new history arms:

1. Existing decoder/control plus observed motion features.
2. Add the five role/time history columns.
3. Add all twenty conditional-history columns (declared primary treatment).

All arms use the same rows, residual targets, training-only constant/duplicate
screening and scaling. Corrections vanish at forecast time zero. The third versus
second arm ablates contextual conditioning. Width changes do change linear
capacity; this is a feature screen, not a fixed-parameter neural ablation.

Continue only if the primary treatment reduces official coordinate RMSE by at
least 0.5% and its paired 5,000-draw game-bootstrap 95% difference interval is below
zero. Also require the contextual treatment to outperform the coarse history arm
before retaining the finer conditioning. All other outcomes stop this probe.
Report every arm, role/time slices, training support and the exact decision.

## Bounded execution and durable recovery

Commit runnable numerical source before the fit. Preparation and each arm run separately with a five-minute hard timeout,
15-second heartbeats and progress every 500 prepared plays. Expected total
numerical work is under five minutes; stop expansion if that budget is exceeded. Atomic,
hash-bound stages preserve covariates, fitted history, each model and predictions.
Replaying completed stages must perform zero fits and leave bytes/mtimes intact.
Before starting the second new fit, archive the first completed arm and history
state, save it privately, download separately and verify all hashes. The previously
verified parent archive supplies immutable inputs; the new archive records its
key and hash instead of duplicating 211 MB. Repeat after the second arm and report. Failed durable saving stops further work.

Limits: reused single fold, adaptive research hypothesis, fixed linear correction,
and in-sample parent training residuals. A negative result does not reject history
in a differently trained neural model; a positive result requires more dates and
neural ablation. Package versions are recorded, not misrepresented as the main
Python 3.11 environment. No new cloud compute or Kaggle submission is authorized
by this experimental protocol.

## Runtime decision

The pending velocity-supervised neural comparison remains open. PyTorch is absent
from the restored workspace and its package download did not complete. This
separate CPU feature hypothesis uses the available preserved numerical runtime;
it is not a substitute result for auxiliary neural supervision.
