# Two independent, equal-depth candidate studies

Status: implemented/prepared, not executed. Three matched arms per round; twelve
numerical channels plus masks per round. Five investigation and four experiment
figure calls per round. Current fitting is held at the review checkpoint.

## Round 16: observed route shape and directional persistence
An instantaneous derivative does not directly summarize a player's net movement
through a recent curved path. The three-, five-, and ten-frame windows use only
contiguous observed positions. Positions are stored relative to a player-specific
terminal observed anchor, but within-player differences cancel that anchor.

For each window, calculate displacement parallel and lateral to CURRENT observed
velocity, and path efficiency (net displacement / total traversed length). The
six core channels use windows three and five. The extensions use window ten plus
absolute turn accumulation, signed turn accumulation, and resultant length of
unit step directions over that window. Position masks must cover the entire
window; direction-based values need nondegenerate observed velocity/steps.

The three-frame window has two frame intervals (0.2 seconds), the five-frame
window 0.4 seconds, and ten frames span 0.9 seconds at 10 Hz. No future smoothing,
interpolation, target statistics or time compression occurs.

## Round 17: opponent traffic and closest-approach geometry
For each player and observed frame, use simultaneously observed opponents only.
Core channels: counts within 3/6 yards, closest observed separation, closing speed
of nearest opponents (ties averaged), minimum constant-relative-velocity separation
over the next one second, and its corresponding minimizing time (ties averaged).

Extensions: nearest lateral relative speed, fraction of opponents closing,
projected within-1/2-yard encounter counts, opponent count in a 2-yard-wide-on-each-side
corridor from player to supplied landing point, and average closing speed for
noncoincident opponents within six yards.

The projection uses tau = min(1, max(0, -dot(delta,dv)/dot(dv,dv))). It is a
hypothetical constant-velocity calculation, not a future observed value. The time
restriction is part of the hypothesis; it is not target clipping. The corridor is
geometric context, not actual ball flight or assigned coverage. Counts describe
available observed opponents, not an asserted complete tactical lineup. No
observed opponent makes these values unavailable, not an invented empty field.

## Controlled ablations
Every arm uses original 10 temporal channels plus the same 12 R13 motion channels,
and 12 slots for this candidate family. Candidate masks are shared exactly.
- mask: all twelve numeric candidates are zero.
- core: first six numeric candidates are supplied.
- full: all twelve are supplied.

Architecture, parameter shapes, seed, initialization, order, exposure and model
settings are identical within and across these two studies. The unchanged R13
motion features are a promising experimental reference, not a promoted model.
Projection windows and thresholds are frozen before results. No feature selection
uses evaluation labels. Any training-data normalization is fitted inside train.

Six comparisons across the two studies use game-cluster bootstrap and adjusted
intervals: core-mask, full-core and full-mask in each round. Each required contrast
needs >=1% mean RMSE reduction and a negative adjusted upper difference bound.
Full must beat core and mask. A candidate must be no worse than the preserved tree
on matching forecast keys before another fold is justified. This conditional gate
is not independent confirmation; many previous experiments used these games.

The official metric is sqrt(sum(dx^2+dy^2)/(2*N)), not Euclidean-distance RMSE or
an unweighted average of group RMSEs. No forecast rows are clipped, dropped or
weighted by post-hoc difficulty. Segment plots do not authorize segment-specific
model choice.

## Data scale and information gain
The supplied evidence establishes 3,352 eligible plays and 130,495 complete labels
within the frozen 64 training games, but the model study uses only 699 plays.
A controlled data-scale comparison is a serious alternative to releasing either
new study. Do not combine newly added training data with new features and then
attribute the resulting change to features alone. First finish the already-prepared
14/15 comparisons; then choose the next controlled change based on their evidence.

## Research sources and limits
- The organizer specifies 10 Hz tracking and coordinate RMSE. The supplied landing
  point and player role are task inputs, not inferred future outcomes.
- The first-place solution uses recent motion, receiver/ball-relative positions,
  roles/horizon, interactions, augmentation, loss design and ensembling. It does
  not prove that this exact route/traffic encoding helps.
- Nguyen and Yurko study turn-angle variability in American football ball carriers
  after reception/handoff. Their task supports the relevance of directional motion,
  not a transfer claim to this pre-throw trajectory prediction benchmark.
- Song et al. model player temporal movement and inter-player relationships for
  coverage assignment. Their annotations and accuracy do not transfer to RMSE here.

Exact source links and retrieval date are in SOURCES.md. The new implementations
are hypotheses with overlap to older feature families, not novel raw information.
