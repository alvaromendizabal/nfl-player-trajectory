# Round 2 frozen experiment protocol

## Objective and attribution boundary

Research target: the user's historical benchmark of **0.46340** coordinate RMSE. This bounded experiment is a small, **training-side diagnostic**, not a claimed route to that score by itself. Its immediate question is what information explained the observed arrival gain, and whether explicit turning/speed/orientation responses add complementary signal. No new architecture, tuning, ensembling, target encoding or earlier-origin augmentation is included.

The previously saved arrival representation is a candidate for established-model integration. Before spending that compute, this round decomposes its gain and tests three specific companion families. **Do not let cheap ridge probes become an indefinite substitute for an established-model experiment.** After this report, justified representations should receive a separate capacity-/exposure-matched established-model comparison.

## Parent evidence and preservation

Exact parent report SHA256: `8953b811c7ef3a541e7098825cb0a05c1decac4fe8990d75a092f26f22e5c98f`.
Exact parent dataset manifest: `1cc416372d04f8c32b868bcf13ea549c300429ab795d11b5ab81b2079e458bd6`.
Main revision reviewed: `402843faa1722460aa84d1bbaf27c4050a7b7ff9`.

The runner requires the successful Round 1 environment, the sealed original manifest and screen summary, all checksummed parent artifact files, and exact forward replay of six preserved control/arrival fits. It uses only the **offset-zero** X, y, roles and original forecast keys. Nonzero-offset files are checksum-verified and their key-array lengths read solely to translate saved global training indices. Their features and labels are not loaded or fitted. No arbitrary pickle is loaded by this new package.

Parent raw files, prepared arrays, models, notebooks and Git checkout are never edited. New private outputs live outside Git in `nfl-feature-round2-results`. No code path installs packages, fetches Git, calls AWS, invokes Kaggle or submits predictions.

## Experiment A: arrival attribution

For each of the original three chronological folds, reuse the saved `control` and `arrival` predictions; there are **zero parent refits**. Six new arms per fold produce **18 small new fits** total:

| Arm | Columns | Comparator | Question |
|---|---:|---|---|
| only_velocity | 72 + 8 | saved control | Does role-conditioned required-velocity response explain the gain? |
| only_acceleration | 72 + 8 | saved control | Does the quadratic-in-phase arrival response explain the gain? |
| only_receiver | 72 + 8 | saved control | Does receiver-relative motion explain the gain? |
| without_velocity | 96 − 8 | saved arrival | What is lost when this subgroup is removed? |
| without_acceleration | 96 − 8 | saved arrival | What is lost when this subgroup is removed? |
| without_receiver | 96 − 8 | saved arrival | What is lost when this subgroup is removed? |

Historical names `required_velocity` and `required_acceleration` are retained for exact lineage. The actual features are **candidate displacement corrections**, not measurements of true future velocity or acceleration. With ball-relative vector b, velocity v, observation age a, requested time t and supplied horizon h, define T=h+a and tau=t+a. Their two mechanisms are `(b-v*T)*tau/T` and `(b-v*T)*(tau/T)^2`; the third is `(v_receiver-v)*tau` with synchronized support. Each is divided by 10 and gated by four task roles. Correlation among these mechanisms can make addition and removal results differ; report both rather than pretending importance is uniquely identifiable.

## Experiment B: three new movement hypotheses

Each family is added separately to the fixed **96-column arrival** representation. Three arms per fold give **9 small new fits**, not a model search or ensemble. No combined arm is selected after viewing individual results.

| Family | Columns | Mechanics |
|---|---:|---|
| Turning arc | 24 | Four roles × two windows (5/10 frames) × x/y displacement correction plus observation-support response |
| Braking/speed change | 24 | Same structure, using observed tangential acceleration; braking stops rather than reverses the candidate path |
| Orientation–velocity mismatch | 12 | Four roles × x/y speed-scaled body-orientation correction plus availability response |

Total new candidates: **60**. The largest diagnostic arm has **120 columns**, not 156, because families are not stacked. Existing motion/orientation/role features overlap these raw signals; this is a test of a different conditional response representation, not 60 independent newly discovered signals.

The turn family integrates a constant-angular-rate motion hypothesis in the observed velocity direction and subtracts the frozen constant-velocity reference. It uses a sinc-based form continuous at zero angular rate. Angular rate is the median signed angle change per 0.1 seconds over eligible adjacent observations in the fixed window. No unsigned-angle substitute is used.

The speed family estimates median adjacent changes in speed per 0.1 seconds on the observed contiguous suffix. For v0=|v| and a<0, active time is `min(tau,v0/(-a))`; otherwise it is tau. Candidate displacement is `(v/v0)*(v0*active+0.5*a*active^2)`, minus `v*tau`. This no-reversal rule applies to the feature's hypothetical path; **neither labels nor final predictions are clipped**.

The orientation family is `(speed*unit_body_orientation - velocity)*tau`, with correct coordinate transformation and explicit absence. Orientation is not ground-truth intent or a known future route. Zero speed does not define a meaningful movement heading. All new displacement values use a fixed division by 10 yards, and role-conditioned support is multiplied by elapsed time. Observation age is included.

## Leakage and numerical contract

Only original pre-throw input CSVs are read to construct new features. No raw output CSV is reread; original targets come from exact existing Round 1 arrays. The feature API accepts observed tracking and **only the four request keys**; a request table containing x/y labels is rejected. Unknown roles, impossible horizons, duplicates, nonfinite required positions, future observed rows and stale scored players >=2 seconds stop visibly.

Missing telemetry has position-derived velocity fallback on adjacent observed frames only. Derivatives never bridge a gap; rates use a contiguous suffix. Missing velocity/orientation support is not treated as measured zero. Translation, lateral reflection, opposite play direction and row ordering are tested. Identifiers join rows; they are not numeric features. No external labels, coverage annotations, outcomes or learned histories are introduced.

Feature support figures use only the first chronological fold's TRAIN games. Deterministic features may be constructed for evaluation rows, but fitted screening/standardization always uses each arm's own training rows. Forecast keys and training indices must match the saved controls exactly; no evaluation subsampling, target clipping or row filtering is allowed. The old outer validation and reserved holdout are not used.

## Fixed fitting and uncertainty

Solver: Round 1's exact fixed ridge implementation, penalty **0.01**. Mean/standard deviation and the near-constant column mask are fitted on training only. Same y, same requested rows, same training-row order, same loss. Feature width necessarily changes between arms; therefore these are conditional representation tests under a fixed regularizer, **not parameter-count-matched neural ablations**.

Official metric formula: `sqrt(sum(dx^2+dy^2)/(2*N))`. Pooled metrics sum squared errors over all disjoint evaluation blocks, not average their RMSEs. Record per-fold, role and first-second/after-first-second diagnostics; role slices are descriptive, not permission to retrospectively choose a role-gated winner.

There are **nine predeclared pooled comparisons** (six attribution plus three motion). Use **10,000 paired game-bootstrap resamples**, seed 20260913, with ordinary 95% and Bonferroni-adjusted percentile intervals (per-comparison alpha 0.05/9). These are approximate bootstrap intervals; multiplicity handling does not make reused folds independent or undo cross-round selection.

For an addition to earn further testing: >=1% pooled RMSE reduction, adjusted upper difference bound <0, and improvement in >=2/3 folds. For conditional removal evidence: adjusted lower difference bound >0 and worsening in >=2/3 folds. Removal evidence is not a standalone deployment rule. The same selected 256 plays and already examined folds are reused; a pass is an **exploratory candidate**, not independent confirmation. Multi-season stability is not established.

## Bounded execution and restartability

Run in order: parent replay preflight (120-second cap), attribution (180), 32-play new-feature smoke (180), 256-play preparation (360), motion screen (180), fresh-process replay command (180), report (60). Caps are safeguards, not runtime estimates. One CPU space, two BLAS threads; no GPU is justified for this screen.

New per-play features and each model/prediction checkpoint are atomically saved. A forced timeout terminates the process group, preserves completed checkpoints, and returns a failing code rather than silently continuing. Fifteen-second heartbeats and per-play/per-fit counters show progress. A lock prevents concurrent notebook/terminal stages. Changed source, parent data, protocol or environment stops rather than retraining old work.

The 32 smoke plays are reused in full preparation. Models save coefficients, masks, scales, evaluation keys and predictions in non-pickled NPZ files. Exact numerical read-back is required, and a separate `replay` command starts a fresh Python process with **no fitting allowed**. A narrow interruption after atomic model save but before JSON receipt is recovered by validation, not another fit.

## Research sources and what they do not establish

1. Nguyen & Yurko (2026), *Bayesian multilevel step-and-turn models for evaluating player movement in American football*: https://arxiv.org/html/2603.17866v1. Motives: step length, signed turns, directional persistence and contextual movement. The paper evaluates ball-carrier movements and one-step alternatives; it does not demonstrate this pass-flight feature bank or our RMSE. No posterior, annotations or outcomes are imported.
2. Nguyen & Yurko (2025), *A Bayesian circular mixed-effects model for explaining variability in directional movement in American football*: https://arxiv.org/abs/2507.06122. Motives: turn-angle representation and movement variability. Its evaluation task is different.
3. Song et al. (2026), *Decoding Defensive Coverage Responsibilities...*: https://arxiv.org/html/2603.25901v1. Motives: time-varying player relationships; it supports the still-open learned relational program, not a claim that our turning tests solve coverage or that its privileged coverage labels are allowed here.

The exact 60-column design and physical integration rules are explicit hypotheses designed for this experiment, not copied winning-solution claims. Research remains open across temporal relationships, arrival behavior, origins/horizons, history, team structure and deployment.
