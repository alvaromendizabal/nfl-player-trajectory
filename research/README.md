# Research evidence and source archive

## Evidence hierarchy

The public notebook reads allowlisted aggregate summaries only. A completed experiment, a passed software test, a passed promotion gate, and a Kaggle score are separate evidence states. Local OOF and private leaderboard results are never treated as interchangeable.

The current hierarchy is:

1. **Five-fold excluded-game OOF** establishes the reproduced base family: 0.468143852 across 561,607 retained rows.
2. **Controlled discovery/confirmation experiments** test new capabilities such as ROT geometry, training-seed diversity, mirror averaging, and context-player dropout.
3. **Promotion gates** require predeclared point-improvement and robustness/uncertainty conditions; several useful point gains remain unpromoted.
4. **Official Kaggle runtime and private score** measure deployment candidates. The five-base candidate scored 0.46615; the equal seven-model ensemble scored 0.46547.

Late post-competition submissions are performance measurements, not official competition ranks.

## Preserved source, curated presentation

`workspace/` retains named source modules, protocol documents, feature dictionaries, tests, and notebooks with private outputs removed. It excludes data, private contracts, credentials, numerical caches, weights, log files, raw reports, and historical duplicate installers. Paths and Python source bytes are not silently refactored. This preserves traceability without advertising standalone reproducibility that the public files do not support.

The publication manifest lists every source mirror and the explicit categories withheld. A machine-readable hash map protects the published review files. Existing `src/`, `scripts/`, `tests/`, and their Quality workflow retain their checks. Only the research archive is excluded from automatic formatting; it is checked separately for syntax, structure, hashes, metric arithmetic, and privacy boundaries.

## Scientific conclusions

The main lesson from the latest research round is that **complementarity can transfer even when a component fails standalone promotion**. Seed-1 and context-dropout variants each produced positive Fold-0 blend gains but failed a stricter cross-fold or uncertainty rule. Keeping them as low-weight complementary members in a fixed seven-model ensemble improved private RMSE from 0.46615 to 0.46547.

That gain is meaningful but incomplete: the final first-place private score remains 0.46340. The next prepared experiment tests a complete alternate game-grouped CV split family, a structurally new diversity source rather than another isolated seed. It is pending execution and has no published metric yet.

Negative findings remain part of the record. ROT failed fixed confirmation; mirror averaging failed a per-fold robustness rule; seed confirmation and context dropout each had confidence intervals crossing zero. Those results prevent retrospective promotion while still informing ensemble design.

## Leading-solution reproduction boundary

The repository now contains a validated source-faithful base architecture, adapted seed diversity, adapted protected player-context dropout, and a verified heterogeneous equal-weight deployment ensemble. Alternate CV split diversity is prepared but not yet measured. Large multi-CV/feature-diverse ensembles and pretraining plus multi-auxiliary transformer supervision remain incompletely reproduced.

The goal is not to copy a public solution. Each mechanism is independently implemented, tested under this project's validation design, and either retained or rejected based on evidence.

## What publication does not certify

Publication CI performs no new model fitting, private prediction replay, submission, or cloud mutation. It validates aggregate arithmetic, privacy boundaries, notebook structure, and hash registration. The 0.46547 score is supported by the recorded private submission; the pending alternate-CV study is not presented as completed.
