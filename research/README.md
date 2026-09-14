# Research evidence and source archive

## Evidence hierarchy

The public notebook reads allowlisted aggregate summaries from the owner's current AWS filesystem during publication. Summary hashes establish which source reports were used. Arithmetic checks reconstruct coordinate RMSE when an aggregate supplies SSE and row count. A saved replay receipt is evidence reported by the earlier run; publication does not independently forward-replay private weights.

A completed experiment, a passed software test, a passed feature gate, and a Kaggle submission are separate states. A training-only smoke test does not establish feature value. All score comparisons must preserve evaluation populations.

## Preserved source, curated presentation

`workspace/` retains named source modules, protocol documents, feature dictionaries, tests, and notebooks with their private outputs removed. It excludes data, private contracts, credentials, numerical caches, weights, log files, raw reports, and historical duplicate installers. Paths and Python source bytes are not silently refactored. This preserves traceability without advertising standalone reproducibility that the public files do not support.

The publication manifest lists every source mirror and the explicit categories withheld. A machine-readable hash map protects the published review files. Existing `src/`, `scripts/`, `tests/`, and their Quality workflow retain their checks. Only the research archive is excluded from automatic formatting; it is checked separately for syntax, structure, hashes, metric arithmetic, and privacy boundaries.

## Scientific conclusions

Use each matched comparison and its uncertainty, not the smallest score across incompatible experiments. Several mean improvements have failed predeclared replication or uncertainty criteria. Historical averages, learned interaction inputs, goal-relative derivatives, and route/traffic prototypes have different evidence states. Feature research is still open.

The observed-input and training-label audits identify unused eligible plays in the same training games. A data-scale-only comparison is a distinct next question; it should not be combined with new feature definitions and optimization changes in a way that prevents attribution.

## What publication does not certify

No new model fitting, private prediction replay, submission, production deployment, or record-breaking result is implied. Publication tests do not validate every archived model implementation. GitHub CI results apply to the exact commit they tested.
