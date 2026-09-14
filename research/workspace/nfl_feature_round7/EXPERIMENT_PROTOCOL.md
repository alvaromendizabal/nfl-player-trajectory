# Round 7 · Earlier-game motion-response histories

**Status: software prepared; no new private NFL score. Feature research remains open.**

## Question and rationale

Round 6's passing-line state and history did not beat the preserved control overall. Do not rerun those treatments, the earlier-origin mixture, turning/braking/orientation, or the broad direct-state addition unchanged. A richer geometric matrix has not earned retention in this diagnostic.

This round changes the **information source**: historical outcomes from strictly earlier training dates. It asks whether player/role tendencies relative to a constant-velocity trajectory are useful beyond the current observed geometry. The maintained repository already studied chronological player/role statistics in its separate temporal model. This kit is **not a novel claim about historical encodings**, not reuse of that model's score, and not a rerun of its old experiment. The current 72-column diagnostic has no fitted historical inputs. New here: phase/horizon-conditioned, trajectory-weighted, hierarchical response priors, with support-only and player-vs-role ablations.

## Feature and information-time contract

For each previously completed training trajectory, let `y` be its saved canonical x/y residual relative to the original constant-velocity reference. At requested time `t = frame_id / 10`, define the average residual rate `y / t` (yards/second). This is an **average correction rate**, not instantaneous observed velocity.

Use the supplied forecast horizon and requested frame to select a phase third and a horizon band (boundaries 1, 2, 3 seconds). These boundaries are fixed before this run. No outcome defines a bin. Within each training player/play/phase, average its residual-rate labels once; add one donor contribution to each table. Ten correlated forecast rows are not ten independent player experiences. A donor may contribute to each of the three distinct phase bins.

Backoff hierarchy: global phase → role/phase/horizon → player/role/phase/horizon. Fixed smoothing strengths are 30, 20 and 10 trajectory contributions. Empty global history has zero mean and 2 yards/second standard deviation per axis as a **fixed weak prior**, not an empirical or physical limit. Mix first and second moments; expose dispersion without claiming it is calibrated uncertainty. No target clipping or row deletion.

Queries on a training date are encoded **before any outcome from any game on that date is added**. Process all date batches in calendar order. No random folds, leave-one-row-out, or full-data target encoder is used. Evaluation gets one table frozen after the fold's training dates; there are no updates from evaluation outcomes, even between evaluation games. The feature API accepts training targets and evaluation metadata, never evaluation targets. Player IDs are lookup keys, not numeric predictors.

The same earlier date can legitimately be training data in a later expanding fold after having been evaluated in an earlier fold. It never contributes to its own fold's evaluation features. Reused game splits are not untouched confirmation.

### Candidate inventory

- Nine support fields: log prior trajectory count, cold-start indicator, and log days since last donor, for each table level.
- Six role fields: smoothed residual-rate mean x/y; residual-rate standard deviation x/y; expected residual x/y (mean rate × requested time).
- Six player fields with the same definitions, shrunk toward the corresponding role prior.

Total: **21 new fields**. These are overlapping learned summaries of existing training outcomes, not 21 new raw signals. Late history is not available for early training dates; cold-start rows remain in training. Unknown players fall back to their current role context. The encoder is built independently per fold.

## Controlled arms

| Arm | Columns before training-only constant screening | Purpose |
|---|---:|---|
| Preserved control | 72 | Reuse already fitted Round 5 control predictions/models |
| Support only | 81 | Isolate counts, availability and recency from outcome-derived statistics |
| Role response | 87 | Add six role/cohort outcome fields to identical support |
| Player response | 93 | Add six player outcome fields to identical role/support inputs |

Estimator settings, training/evaluation rows, labels, ordering and exposure remain unchanged: 120 histogram-gradient-boosting iterations; learning rate .06; 15 leaves; minimum leaf size 30; L2 1; max bins 127; no early stopping; fixed seed 20260911. No parameter search or ensemble. Controls are replayed, not refitted. Maximum twelve new coordinate estimators across two folds.

## Stages, preservation and budgets

Preflight (150 seconds) → 32-training-play leakage smoke (180) → two fold-local feature encoders (360) → fold 2 (240) → inspect → conditional fold 3 (240) → fresh-process replay (180) → report. These are hard caps, not promised runtimes. Two CPU threads and 15-second heartbeats. Fitted trees checkpoint every 30 iterations; completed fold encoders are independent checkpoints. A killed incomplete encoder fold may reconstruct that small fold; completed encoders are reused. No raw CSV, Kaggle download, external data or cloud API call is needed.

Preserve all old artifacts. Store new private encoders, matrices and models only in `nfl-feature-round7-results`. Source/data/environment drift is a stop, not an overwrite. Replay rebuilds chronological feature arrays and reloads all completed models without scientific fitting. The ZIP export explicitly excludes historical tables, per-row labels, keys, predictions, features and weights.

## Predeclared decisions

Six contrasts are shown: support/control, role/control, role/support, player/control, player/support, player/role. Three possible reporting scopes (fold 2, fold 3, pooled) give 18 planned comparisons. Paired game-bootstrap intervals use 10,000 resamples, seed 20260911, and percentile endpoints `.05/(2*18)` and `1-.05/(2*18)`. These are exploratory adjusted intervals, not exact simultaneous coverage guarantees and not correction for the many adaptive prior rounds.

A role treatment earns continuation only if it improves **both folds**, gains at least 1% in pooled official coordinate RMSE, and has an adjusted upper difference bound below zero against **both control and support only**. Player treatment must satisfy the same conditions against **control, support only, and role**. Support alone passing does not establish historical outcome value.

If both outcome-derived arms are at least 5% worse than control on fold 2, stop fold 3. Otherwise run all three arms unchanged. No role-specific post-hoc selection, bin retuning, exclusion of hard games, or altered shrinkage. Always score every saved evaluation key with `sqrt(sum(dx²+dy²)/(2N))`.

## Evidence limits and next step

Same 1,024-play subset and repeatedly inspected chronological folds; not the full 4,951 cached training plays, a neural-model comparison, a held-out season, or Kaggle's test. The prior 0.62-ish result and 0.46340 target are different populations. Features/model training/data coverage/ensembling gaps cannot be numerically allocated from these screens.

Positive result: integrate the justified historical channels into a **replayable temporal forecaster**, use matched no-statistics input controls, then validate beyond this diagnostic before submission. Negative result: stop this encoder unchanged and stop expanding this small diagnostic with further ad hoc columns; the next representation milestone is the observed temporal/player encoder and its bounded throughput/recovery test. Neither result closes feature engineering. Full observed trajectories, learned relational interactions, role-conditioned behavior, prediction-origin support and permitted historical seasons remain research avenues, with their previously tested interfaces clearly distinguished.

## Sources and boundaries

1. Maintained project protocol, `docs/TEMPORAL_RESEARCH.md` at `402843faa1722460aa84d1bbaf27c4050a7b7ff9`: earlier-date statistics and matched presence/absence testing. Source of project-specific precedent, not new Round 7 accuracy.
2. Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, arXiv:1706.09516: target leakage and ordered statistics. We apply actual calendar order, not the paper's random permutations. We do not introduce CatBoost as an estimator.
3. Organizer overview: official coordinate RMSE, 10-Hz trajectories, play-wise inference.
4. First-place write-up: compact observed time sequences and static context motivate the wider representation program. Its architecture, augmentation, losses, ensemble and score are not reproduced by this kit.

Links are in SOURCES.md. Source reports remain separate from proposed hypotheses and local synthetic tests.
