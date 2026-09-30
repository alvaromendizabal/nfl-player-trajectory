# Research evidence and source archive

## Evidence hierarchy

The public record separates software correctness, local grouped-game validation, promotion-gate decisions, inference/runtime evidence, and private submission score. They are never treated as interchangeable.

The current hierarchy is:

1. **Multisplit grouped-game OOF** remains the strongest completed local system: **0.4631723213 RMSE over 561,607 rows**.
2. **Late private submission evidence** remains **0.46487 RMSE**, the strongest recorded competition-facing measurement in the project.
3. **Controlled candidate studies** test new representation and architecture families against fixed references and locked gates.
4. **Runtime evidence** is evaluated separately from predictive quality.
5. **Prepared next experiments** are labeled unmeasured until AWS execution produces a valid result.

Late submissions are performance measurements, not official competition ranks.

## Preserved source, curated presentation

`workspace/` retains selected source modules, protocols, tests, feature dictionaries, and notebooks with private outputs removed. It excludes raw data, credentials, fitted weights, large checkpoints, private object locations, and unreleased competition-specific runners/transforms.

The repository is intentionally **semi-reproducible**: reviewers can inspect the research structure, aggregate evidence, validation discipline, and selected implementation patterns without receiving a drop-in copy of the private competition system.

## Scientific conclusions since PR #38

**The restored training contract enabled credible new studies.** Two subsequent candidates completed full Fold-0 decisions without relying on the earlier drifted control.

**An expanded observed-motion representation added ensemble signal but was not promoted.** Its fixed blend improved the common reference by **0.001140 RMSE**, below the locked 0.0015 threshold, and its paired-game interval crossed zero.

**A future-conditioned delta decoder improved standalone quality but contributed less ensemble gain.** It improved the fixed blend by **0.000745 RMSE**, again with an interval crossing zero.

**GPU data feeding was materially improved.** On the current single-L4 environment, the measured full training step fell from about **0.110 s** with no loader workers to **0.0486 s** with the selected two-worker plan; more workers did not improve the benchmark.

**The next branch changes interaction structure rather than appending more kinematic channels.** A target-specific sparse-interaction family is prepared but remains unmeasured.

These negative results are preserved because they prevent repeated spending and selective promotion.

## Leading-solution reproduction boundary

The project substantially covers the strongest public solution's compact temporal/player-interaction base, grouped folds, moving-average inference, motion objectives, augmentation, and multi-split diversity. Recent work also explored a distinct future-conditioned motion decoder.

Important gaps remain in broader feature-configuration diversity and more complete standalone interaction/auxiliary-objective families. Additional-data recipes remain outside the current competition-data-only boundary.

The goal is independent recreation and controlled testing, not copying trained weights or public feature files.

## What publication does not certify

Publication CI performs no new fitting, private prediction replay, submission, or cloud mutation. It validates public bytes, notebook structure, privacy boundaries, and aggregate evidence.

No local development-fold result is presented as a private score, and no prepared-but-unrun architecture is presented as measured.
