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

## Scientific conclusions through the post-PR40 frontier

**Split/model diversity remains the strongest transferred mechanism.** The 20-model multisplit system still owns the strongest local OOF and private submission evidence.

**A broad sequence of parent-level adaptations plateaued.** Competition-only pseudo-supervision, ST-GRU, zero-fit post-processing, frozen-parent feature adapters, full-parent context fine-tuning, source-native context variants, target/role-specific interaction, and single-target route variants all completed controlled tests without promotion.

**TTA improved individual model families but was mostly redundant inside multisplit-20.** The best five-fold winner-parent TTA recipe improved its base family by **0.001604 RMSE**, yet improved the final multisplit ensemble by only about **0.000034 RMSE**.

**Cross-fold and matched-control gates prevented false promotion.** Several discovery-stage point gains disappeared under confirmation or full-system replacement tests.

**The project now needs independently trained error diversity rather than more small parent modifications.** The next prepared program trains fresh source-faithful feature configurations and only scales a configuration after full-OOF complementarity is demonstrated.

## Leading-solution reproduction boundary

The project substantially covers compact temporal/player interaction, grouped folds, EMA, motion objectives, augmentation, multi-split diversity, dual-path/auxiliary modeling, target-specific interaction, single-target supervision, and multiple TTA forms.

The largest remaining competition-data-only gap is **breadth of independently trained feature configurations combined with repeated grouped-CV split families**.

Historical/external-data pretraining remains outside the active data boundary.

The goal is independent recreation and controlled testing, not copying trained weights, hidden predictions, private runners, or another competitor's private feature pipeline.

## What publication does not certify

Publication CI performs no new fitting, private prediction replay, submission, or cloud mutation. No local development-fold result is presented as a private score, and no prepared-but-unrun experiment is presented as measured.
