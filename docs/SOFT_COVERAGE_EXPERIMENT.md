# Bounded soft-coverage feature probe

Declared 10 September 2026, before examining this probe's validation outcomes.
Feature research remains open. No Kaggle submission or final refit is part of this milestone.

## Hypothesis and research basis

Defenders may react to several plausible receivers and change responsibilities.
The existing terminal-nearest-opponent summaries cannot fully express uncertainty
over the player set. [NFL/AWS coverage-responsibility research](https://arxiv.org/html/2603.25901v1)
uses temporal and player attention to model evolving assignments. Its supervised
coverage labels and some later cutoffs are unavailable here; neither its accuracy
nor its learned model can be imported into this task. This independently written
probe tests deterministic observed-only affinities as a low-cost first step.

All eligible opponents contribute using fixed squared distance (5 yards) and
relative velocity (3 yards/second) scales. A fixed unmatched alternative with cost
2 prevents every player from being forced into a matchup. These are geometric
affinities, **not calibrated coverage probabilities or man/zone labels**. Passers
are excluded. There are 11 terminal features and 72 temporal features over 3, 8,
and 20 observed frame windows: mass, entropy, target-receiver mass, relative
geometry and motion, physical-time slopes, coverage and affinity turnover.

Only observed history, supplied roles/sides and output requests are consumed.
Future outcomes and identity values never enter features. Masks apply before
arithmetic; missing frames are not treated as consecutive observations. Reflection
and player-set permutation have explicit independent contracts.

## Scope and fixed decision

Reuse chronological `inner_1`: 94 training games and 41 later evaluation games,
83,938 evaluation rows. Original development, inspected reserved holdout and Kaggle
are excluded. Existing attention residuals and 133 decoder/control columns plus
57 motion-state candidates are reused from their hash-verified cache. The parent
training residuals are in-sample; this shared probe limitation means a negative
result rejects this correction interface, not all learned matchup models.

Fit three deterministic ridge corrections: existing control+motion, add 11 static
affinities, add all 83 affinities. Each uses the same average-loss L2 penalty 0.01
and all training rows. Screening removes constant and exact duplicate columns on
training data only. Standardization also uses training only. Corrections are
multiplied by requested seconds to vanish at the origin. There is no hyperparameter,
seed, epoch or blend-weight search. This is a linear screening experiment, not
a claim of fixed neural capacity or a reproduction of the leading solution.

Primary comparison: all affinities versus control+motion. Continue to more folds
only if official coordinate RMSE falls by at least 0.5% and a paired game-bootstrap
95% interval is below zero. The static arm tests whether temporal affinities add
more than terminal geometry. Do not select the smallest observed arm and relabel
it as the declared primary treatment. Diagnostic role/time errors are descriptive.

## Execution and persistence

One local CPU probe with two threads; no GPU or AWS training resource. A 300-second
wall budget covers preparation and fitting, with progress every 250 plays and
15-second heartbeats. Verified feature preparation, models, predictions and result
are separate atomic, source/input-hash-bound stages. Replays must leave their bytes
and modification times unchanged and execute no additional fits. Save runnable
source to a remote Git branch before the fit. Save and verify the completed probe
in the existing private S3 bucket before another scientific experiment.

The local recovery runtime differs from the main project's Python 3.11 lock. The
probe records actual package versions and uses only NumPy/Pandas/SciPy and the
existing runtime utilities; it does not silently claim execution under the
original environment. CI still checks the maintained source under its pinned lock.
