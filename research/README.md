# Research evidence and source archive

## Evidence hierarchy

The public record separates software correctness, grouped-game validation, promotion decisions, runtime evidence, and private submission evidence.

1. **Multisplit grouped-game OOF:** `0.4631723213` RMSE over `561,607` rows.
2. **Private submission evidence:** `0.46487` RMSE.
3. **Controlled candidate studies:** fixed references and locked promotion gates.
4. **External-data evidence:** provenance, coverage, point-in-time construction, and controlled model studies.
5. **Runtime evidence:** evaluated separately from predictive quality.
6. **Prepared experiments:** explicitly labeled unmeasured until AWS execution produces a valid result.

## Preserved source, curated presentation

The public archive retains selected source modules, protocols, tests, feature dictionaries, aggregate evidence, and notebooks with private outputs removed.

It excludes raw competition data, raw third-party response archives, credentials, fitted weights, large checkpoints, private object locations, complete private runners, and unreleased competitive feature transforms.

The repository is intentionally **semi-reproducible**.

## Scientific conclusions through the October external-data frontier

**Split/model diversity remains the strongest proven system-level mechanism.**

**Standalone gains often remain ensemble-correlated.** Dense temporal supervision, zero dropout, and several source-family modifications improved component quality without adding enough independent residual signal.

**Direct external historical context is useful.** NFL Next Gen Stats improved standalone fitting, while ESPN prior-game context produced stronger recent complementarity.

**Point-in-time construction is mandatory.** Historical features exclude current-week and future observations.

**Negative experiments are preserved as evidence.** Exact failed branches are retired rather than repeatedly rescued with minor parameter changes.

**The next active program is player-specific historical context.** It reuses the already acquired direct-source data and asks whether finer player/pass-target priors create more independent residual signal.

## Public-solution research boundary

Public solution writeups and repositories may be studied for transferable mechanisms, but prepared competitor data artifacts are not used as external training data.

The current external-data program pulls directly from first-party or neutral public sources and creates its own point-in-time transformations.

## Publication does not certify

Publication CI does not retrain private models, replay private predictions, mutate AWS, or submit externally.

Prepared-but-unrun experiments are never presented as measured.
