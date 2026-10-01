# Research evidence and source archive

## Evidence hierarchy

The public record separates software correctness, local grouped-game validation, promotion-gate decisions, inference/runtime evidence, and private submission score. They are never treated as interchangeable.

1. **Multisplit grouped-game OOF:** **0.4631723213 RMSE over 561,607 rows**.
2. **Late private submission evidence:** **0.46487 RMSE**.
3. **Controlled candidate studies:** fixed references and locked promotion gates.
4. **Runtime evidence:** evaluated separately from predictive quality.
5. **Prepared experiments:** labeled unmeasured until AWS execution produces a valid result.

Late submissions are performance measurements, not official competition ranks.

## Preserved source, curated presentation

The public archive retains selected source modules, protocols, tests, feature dictionaries, aggregate evidence, and notebooks with private outputs removed. It excludes raw data, credentials, fitted weights, large checkpoints, private object locations, and unreleased competition-specific runners/transforms.

The repository is intentionally **semi-reproducible**.

## Scientific conclusions since PR #39

**Target-specific sparse interaction was useful but insufficient.** It improved the fixed blend but missed the locked promotion requirement.

**Wide/shallow dual-path interaction produced the strongest new complementary signal.** Its Fold-0 blend improved the reference by **0.002154 RMSE** with a fully positive interval, but Fold 1 did not reproduce the required gain.

**Fixed TTA transferred better than augmentation fine-tuning.** The original/flip/crop inference blend improved the parent on both tested folds; the fine-tune selected epoch 0 on both folds.

**Parent-neutral spectral and specialist adapters did not add value.** Fourier, RBF, late-horizon, and defender-only residual branches all selected the unchanged parent baseline.

**Muon did not improve the dual-path architecture under a controlled optimizer-only test.**

**GPU data feeding was materially improved.** End-to-end loader benchmarking raised dual-path throughput from about **687 to 2,137 examples/s (~3.11×)**; later runs commonly reached **~70–74% mean sampled GPU utilization with 100% peaks**.

**The next prepared mechanism changes supervision rather than architecture.** A competition-data-only two-stage/all-player pseudo-supervision study is prepared but remains unmeasured.

## Leading-solution reproduction boundary

The project now substantially covers compact temporal/player interaction, grouped folds, EMA, motion objectives, augmentation, multi-split diversity, major dual-path/auxiliary ideas, target-specific interaction, fixed TTA, and a controlled Muon axis.

Important gaps remain in **broader feature-configuration diversity**, **larger split/model diversity**, and **two-stage/all-player supervision**. Historical/external-data recipes remain outside the current competition-data-only boundary.

The goal is independent recreation and controlled testing, not copying trained weights, private runners, or public feature files.

## What publication does not certify

Publication CI performs no new fitting, private prediction replay, submission, or cloud mutation. No local development-fold result is presented as a private score, and no prepared-but-unrun architecture is presented as measured.
