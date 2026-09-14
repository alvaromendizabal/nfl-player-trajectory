# Rounds 14 and 15 — proposed controlled representations

## Status
Prepared, statically reviewed, unexecuted. Updated manual-only rules are in
PROJECT_RULES.md. Historical feature gates remain failed/inconclusive. Immediate
scope is tests, label readiness and two 32-play smokes; no optimizer or fit runs.
Companion fit notebooks are gated by readiness.py and an absent review_release.json.

## Hypotheses and source distinctions
Round 14 asks whether projecting observed dynamics onto the player-to-landing
axis exposes slowing, accelerating, steering and changing alignment more usefully
than only field-axis dynamics. Round 15 asks whether receiver-relative derivatives
expose differences in braking/turning beyond instantaneous receiver distance and
velocity. These are our hypotheses, not proven findings of a cited competition
solution. First-place use of ball/receiver-relative features and auxiliary motion
losses motivates their context; its augmentation, capacity, training and ensemble
also differ. The Amazon/NFL paper motivates temporal/agent separation, not these
formulas or an RMSE claim. See SOURCES.md.

## Shared representation, information time and model
All arms receive the original observed 10 player channels, the already implemented
12 motion channels from Round 13, and 12 new candidate slots. Original relationship
features, role/side masks and the 72 base query features remain as in the preserved
encoder. The shared 12-channel motion base is an exploratory reference: its
adjusted feature gate did not pass. Do not call it promoted or validated.

Each round's mask arm has all 12 numerical candidate values zeroed; core supplies
six; full supplies twelve. All retain identical validity masks. Shared masks can
encode measurement support, so this isolates numerical values conditional on that
availability, not the full information content of a family. Models start from the
same seed within and across arms. Expected parameter count from static dimensions
is 20,402, not verified by execution. Runtime tests must check equal count and
initial states. No saved prior trained model is silently resumed as a new arm.

Numerical source: families.py. Arrays are [player, observed slot, channel], bounded
by 22 players and 20 slots. Delta t = 0.1 seconds. Fixed physical scales are used;
there is no fold-global learned normalization. Derivatives require consecutive
valid observations; rolling averages require complete consecutive windows. Missing
values are zero ONLY with explicit false validity. No actual future position,
post-throw direction or evaluation target enters feature construction.

## Round 14 definitions
Let g be supplied landing position minus current observed player position; e=g/||g||
when ||g||≥0.1 yards. Let cross(e,v)=e_x v_y-e_y v_x. The first six channels are
acceleration dot/cross e, jerk dot/cross e, and velocity alignment dot/cross e.
Alignment also needs nonzero speed. Extensions project trailing three- and
five-frame acceleration onto the current goal axis, and backward-difference the
observed closing/lateral velocity in its changing goal axis. The latter includes
axis rotation; it is not the same as projecting Cartesian acceleration. Core is
6 values; extensions are 6 values. The exact scales and names are in the dictionary.

## Round 15 definitions
The first six channels are receiver-minus-player acceleration x/y, backward changes
in closing and lateral relative speed, receiver-minus-player turn rate, and the
difference in speed-change rates. Extensions are complete trailing three- and
five-frame averages of relative acceleration x/y, and relative jerk x/y.
Receiver identity uses organizer role (slot 0 in this established encoding),
requires simultaneous observations, and excludes self. No claimed man-coverage
assignment is constructed. A missing/ambiguous receiver is masked/rejected by the
existing input contract rather than guessed.

## Training-data readiness before new fitting
Scan all existing weekly organizer input/output CSV pairs. Restrict outcomes to
the SAME frozen 64 training games before processing. Verify integer unique keys,
constant scored flags/horizons per entity, requested frame set 1..horizon, finite
x/y, complete target coverage, and absence of unexpected/duplicate rows. Read only
necessary columns; use chunked parsing and per-week sealed receipts. Do not repair,
clip or silently remove problematic plays. Existing output-file hashes establish
local snapshot identity, not a new remote download-provenance assertion.
The data may share CSVs with non-training games; non-training outcomes are read
in chunks but immediately discarded without scoring, storing or summarizing them.
No larger training sample is generated in this milestone.

## Scientific phase, supplied but blocked pending review
Maintain the original 699 selected training plays, 26,039 rows, 5,886 evaluation
rows from 14 games for a pure feature comparison; no sample expansion in that
comparison. After reviewing readiness, a data-scale-only experiment may instead
be prioritized. Do not change scale and features together without a separate
planned factorial design. Candidate rounds do not read each other's numerical
features, outcomes or scores; only shared readiness can gate execution.

Each study has three joint x/y models, 24 fixed epochs and an unchanged optimizer
schedule. Train-only runtime profiling precedes fitting; its first result is
immutable and a failure remains a stop. Save model, optimizer, RNG, cursor and
configuration after bounded progress. Never select epochs using evaluation scores.
All arms must finish before evaluation. Fresh-process replay refuses missing
models and must reproduce stored predictions without fitting.

## Metric, comparisons and decisions
Coordinate RMSE = sqrt(sum(dx²+dy²)/(2N)); pool SSE and counts, not fold RMSE means.
Use paired whole-game resampling. Six planned primary comparisons across both
rounds: core-mask, full-core, full-mask in each. The existing study implementation
uses 20,000 bootstrap draws and Bonferroni tail probabilities for these six.
A required contrast needs ≥1% RMSE gain and an adjusted upper difference bound<0.
Full must beat both core and mask. Candidate must be no worse than the matched tree
before proposing another fold. Within-study adjustment does not account for all
prior adaptive decisions on these repeatedly inspected games. No causal,
independent-confirmation or Kaggle-record claim follows from a pass.
Report training/evaluation and role/horizon slices without routing models based
on inspected subgroups. Confirm on a defensible fresh assessment when feasible.

## Limits and release control
Immediate limits are documented in START_HERE.md. Future stages: prepare360s,
runtime180s, profile180s, train360s/arm, evaluate180s, replay240s. Profile must project
≤300s per arm. Maximum three scientific fits per round; no automatic extra folds.
One GiB per split of loaded numerical tensors is enforced; this is not a guarantee
about total process RSS. Existing CPU space/two threads only. No packages, AWS
resources, Git operations or Kaggle submissions are executed by the assistant.
Review release must bind tests, label and both smoke receipts and explicitly
acknowledge old failed gates, reused evaluation and the data-scale decision.
Do not copy the template into an active release to bypass review.
