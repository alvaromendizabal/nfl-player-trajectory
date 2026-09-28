# Research evidence and source archive

## Evidence hierarchy

The public record separates five evidence states: software correctness, local grouped-game validation, promotion-gate decisions, inference-engineering parity, and private submission score. They are never treated as interchangeable.

The current hierarchy is:

1. **Multisplit grouped-game OOF** is the strongest completed local system: **0.4631723213 RMSE over 561,607 rows**.
2. **Late private submission evidence** measures the competition-facing candidate: multisplit-20 scored **0.46487**, improving the prior 0.46547 result.
3. **Inference parity evidence** validates engineering changes separately from model quality: shared preparation produced a **4.784× measured speedup** with exact prediction parity on the declared sample.
4. **Controlled negative studies** preserve failed promotion decisions rather than retrospectively tuning around them.
5. **Prepared next experiments** are explicitly labeled unmeasured until AWS execution produces a valid result.

Late submissions are performance measurements, not official competition ranks.

## Preserved source, curated presentation

`workspace/` retains selected source modules, protocol documents, feature dictionaries, tests, and notebooks with private outputs removed. It excludes raw data, private contracts, credentials, fitted weights, numerical caches, large checkpoints, and other artifacts that would make the public repository a drop-in clone of the private competition system.

The repository is intentionally **semi-reproducible**: reviewers can inspect the research structure, validation logic, aggregate evidence, and selected implementation patterns without receiving the private competition edge.

## Scientific conclusions since the previous publication

**Split diversity transferred.** The multisplit ensemble improved the recorded private score from 0.46547 to **0.46487**.

**Engineering improvements were separated from accuracy claims.** Shared input preparation reduced measured ensemble latency by **4.784×** while preserving predictions exactly on the declared 96-play / 3,723-row parity sample.

**Several feature follow-ups were not promoted.** Their control or evaluation checks did not support a reliable positive conclusion. Normalization recovery, bounded continuation, and native moving-average evaluation did not restore a credible control.

**The recovery sequence uncovered training-contract drift.** Later speed-oriented experiments differed from the successful training recipe across multiple optimization and moving-average categories. The next feature comparison therefore returns to the verified training contract before broader confirmation.

Negative findings remain part of the record because they prevent repeated spending on untrustworthy variants.

## Leading-solution reproduction boundary

The project substantially covers the source-faithful temporal/player-interaction base, grouped folds, moving-average inference, motion objectives, augmentation, and multi-split ensemble diversity.

Important gaps remain: broader feature-configuration diversity and more complete standalone reproductions of some medal-solution training recipes. Additional-data recipes remain outside the current competition-data-only boundary.

The goal is independent recreation and testing, not copying trained weights or public feature files.

## What publication does not certify

Publication CI performs no new fitting, private prediction replay, submission, or cloud mutation. It validates aggregate arithmetic, privacy boundaries, notebook structure, and registered public bytes.

The public snapshot does not expose raw competition data, fitted states, checkpoint locations, credentials, or unreleased feature transforms. It also does not claim that a local OOF value is a private leaderboard score.
