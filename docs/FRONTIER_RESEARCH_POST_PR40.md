# Post-PR40 frontier research review

This document summarizes controlled NFL Big Data Bowl 2026 research completed after PR #40. It is an employer-facing research record, not a mirror of the private AWS workspace.

The public record intentionally includes **aggregate evidence, validation rules, decisions, and engineering lessons** while excluding raw competition data, fitted weights, private checkpoint/object locations, complete private runners, credentials, and unreleased competition-specific transforms.

## Competitive context

- strongest recorded late private submission: **0.46487 RMSE**
- published first-place private comparator: **0.46340 RMSE**
- private gap: **0.00147 RMSE**
- strongest completed local system: **0.4631723213 pooled OOF RMSE**
- OOF rows: **561,607**
- incumbent: **20 models / four split families**
- stretch research target: **0.44 RMSE**

Private leaderboard scores and local OOF/development metrics are kept separate.

## Why the post-PR40 program changed direction

Before PR #40, the project had already reproduced large parts of the leading public solution space: the first-place temporal/player interaction family, grouped folds, EMA, augmentation and split diversity; third-place-style dual-path interaction and auxiliary supervision; target-specific sparse interaction; fixed TTA; and several optimizer/representation ablations.

The remaining question was whether more sophisticated adapters, supervision changes, target specialization, or inference-time transformations could add signal beyond the existing 20-model ensemble.

The post-PR40 sequence answered that question mostly **no** for the exact tested forms.

## Completed experiment portfolio

### Competition-only pseudo-supervision

A two-stage teacher/student adaptation used only official competition plays and valid unscored players. Controlled pseudo-label variants completed but did not earn confirmation-stage promotion.

**Decision:** retire the exact pseudo-supervision formulation.

### ST-GRU and landing-node ST-GRU

A graph/temporal recurrent family and a landing-node extension were trained on Fold 0.

Their standalone errors were far behind the incumbent reference and their fixed blends were not competitive.

**Decision:** retire the exact family rather than spending more compute making a weak architecture faster.

### Zero-fit multisplit meta/post-processing

Six predeclared transformations were evaluated directly on the existing four-family/20-model OOF bank:

- frame-34 trajectory clamp
- coordinate median
- trajectory-level velocity aggregation
- velocity aggregation plus frame clamp
- game-cross-fitted horizon weighting
- horizon weighting plus frame clamp

The best result improved pooled OOF by only about **0.000049 RMSE**, failed the minimum-gain requirement, improved only two of five folds, and had uncertainty crossing zero.

**Decision:** retire this post-processing direction.

### Frozen-parent feature adapters

Project-owned intent/geometry and temporal feature families were injected through zero-initialized adapters around the strongest parent.

The best fixed-blend gains were only about **0.00011–0.00017 RMSE**.

**Decision:** the added inputs did not become sufficiently integrated while the parent remained frozen.

### Full-parent coverage / physics fine-tuning

The next study allowed the parent to adapt while keeping a live matched control.

Coverage context, physics context, and their combination produced increments over the control of only a few **1e-5 RMSE**.

**Decision:** extra fine-tuning exposure, rather than the new context, explained nearly all movement.

### Source-native configuration variants

The project then tested stronger-parent variants based on source-native context, a landing-point node, and a shorter temporal window.

Some arms showed useful point signal, but none cleared the locked absolute, uncertainty, standalone, and matched-control gates.

**Decision:** retire the exact variants.

### Direct interaction / target-specific / role-specific heads

A live-control experiment tested:

- direct learned spatial attention bias
- target-specific sparse landing context
- role-specific trajectory correction heads

After numerical-resume recovery, all three completed Fold 0 and none beat the live control sufficiently.

**Decision:** retire the exact interaction/head variants.

### Single-target and route representations

The project recreated single-target supervision more directly and added:

- uniform single-target training
- defender-biased single-target sampling
- leakage-safe observed-route representation

All three completed Fold 0. Their fixed-blend gains stayed near **0.00012–0.00013 RMSE**, below promotion.

**Decision:** retire the exact single-target branch.

### Full five-fold winner-parent TTA audit

The final inference-only study evaluated three frozen recipes across all five source-faithful base folds, then inserted each result back into the existing multisplit-20 ensemble.

#### Equal original + horizontal flip

- base family: **0.46814385 → 0.46654003**
- gain: **+0.00160382**
- pooled bootstrap: positive
- multisplit-20 hybrid gain: only **~+0.00003379**

#### Original + horizontal flip + deterministic crop

- base family: **0.46814385 → 0.46669682**
- gain: **+0.00144703**
- hybrid multisplit RMSE: **0.46311464**
- hybrid gain: **+0.00005768**
- hybrid interval crossed zero

#### Four-way symmetry

- base family: **0.47266902**
- hybrid multisplit RMSE: **0.46437391**
- both worse than their baselines

**Decision:** TTA is real family-level signal but largely redundant inside multisplit-20. Do not roll it across the complete ensemble.

## Research interpretation

The broad post-PR40 conclusion is stronger than any individual negative.

### The project is no longer missing obvious parent-level adapters

Multiple geometric, temporal, spatial, role-specific, target-specific, and supervision-level variants have now been tested under controlled gates. The exact tested forms did not create robust incremental value beyond the existing ensemble.

### The incumbent's diversity is doing important work

The TTA audit demonstrated that a transformation can improve a component family substantially while adding almost nothing after that family is embedded inside the 20-model ensemble.

### Future gains require more independent error structure

The first-place system's documented strength was not one exotic adapter; it was breadth: many independently trained feature configurations across repeated grouped-CV splits, then simple averaging.

That remains the most important competition-data-only gap.

### Validation discipline prevented false promotion

Several branches showed favorable Fold-0 point estimates. Confirmation folds, matched controls, bootstrap intervals, or full-system replacement tests prevented those one-off wins from being advertised as robust improvements.

### Engineering reliability is part of the research result

The runner framework now carries forward reusable protections for:

- committed-stage resume
- transient AMP overflow handling
- EMA device-safe restoration
- archived baseline anchors for floating-point-sensitive replay
- collision-resistant artifacts
- resource/cost heartbeats
- workload-specific performance benchmarking
- fail-closed integrity checks

## Leading-solution reproduction matrix

### First place

Substantially covered:

- compact temporal representation
- player interaction
- Gaussian trajectory / motion supervision
- grouped game folds
- EMA
- geometric augmentation
- multiple split families
- simple averaging

Strongest transferred lesson:

**split/model diversity improved the final competition-facing system.**

Largest remaining gap:

**much broader independently trained feature-configuration × repeated-CV diversity.**

### Third place

Substantially recreated/adapted:

- dual-path temporal/spatial modeling
- auxiliary trajectory objectives
- context dropout / reorder / crop
- fine-tuning workflow
- TTA

The exact dual-path model failed cross-fold promotion, while TTA improved individual-family predictions but later proved mostly redundant inside multisplit-20.

Historical-data pretraining remains outside the active data boundary.

### Fourth place

Adapted/tested:

- landing context/node
- defender emphasis
- sparse target interaction
- single-target supervision
- frame-clamp post-processing
- symmetry TTA

No exact tested branch promoted.

### Fifth place

Adapted/tested:

- delta/future-frame modeling
- robust-objective ideas
- Muon
- EMA variants
- competition-only pseudo-supervision

No exact tested branch promoted.

## Next prepared study

The next prepared AWS milestone trains **fresh full source-faithful models from scratch** rather than modifying an existing fitted parent.

Two project-owned feature configurations preserve the successful training contract but alter the learned temporal representation. The protocol screens both configurations, continues only the better one, and requires a locked midpoint/final promotion gate before confirmation or full OOF.

This study is **prepared but unmeasured**. No performance claim is made until real AWS execution completes.

If a fresh configuration proves complementary at full OOF, the next step is additional split-family scaling—the remaining direction most directly aligned with the first-place large-ensemble strategy.

## Reproducibility boundary

This repository is deliberately semi-reproducible. It enables review of the research program, validation logic, evidence hierarchy, aggregate results, and selected implementation patterns without distributing the complete private competition system.
