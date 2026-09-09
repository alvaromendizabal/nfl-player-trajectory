# Validation record

The final-preprocessing milestone passes **269 tests** in the locked environment,
including four new integration tests.
The tests cover motion/coordinate geometry, official-metric arithmetic, chronological
splits and histories, exact-frame joins, optional-input dependencies, source and
artifact integrity, interrupted-stage recovery, and standalone predictor parity.
Lint, format, type checks, warnings as errors, notebook execution, and isolated
export checks form the canonical quality gate.

## Evidence levels

| Evidence | What it establishes |
|---|---|
| Synthetic unit/integration checks | Mathematical and software contracts on controlled examples |
| Completed chronological feature folds | Out-of-time representation comparisons within 2023 |
| Fixed-estimator comparisons | Gains attributable to feature representation at unchanged estimator settings |
| Strict removals and trajectory permutations | Conditional family contributions, with stated limitations |
| Full raw-input development replay | The saved inference path reproduces the measured feature pipeline |
| Organizer unlabelled sample gateway | Interface, row ordering, finite output, and standalone parity |
| Reserved holdout | Not run; future final evaluation after the feature gate |

Every experiment records source/dependency/input signatures. A reusable checkpoint
needs completed status and matching output hashes. Publication verifies model and
evaluation receipts, recomputes RMSE from frame errors, checks inner-fold selection,
and refuses stale inference or report artifacts. Notebook outputs are executed
before replacing canonical files. Enabled training/export switches block automated
publication before any cell runs.

The wide search fits deterministic 512/1,024/2,048/4,096/8,192 budgets and preserves the parent
features. Missingness, variance, duplicate names, schema, finite values, and
training-only correlation screening are checked. Exact dependency tests confirm
that declared positional features remain unchanged when telemetry/metadata vanish.

Private S3 checkpoints preserve expensive work by content hash. SageMaker
processing jobs have an enforced runtime bound and no persistent endpoint.
Completed jobs stop their compute. The current run's final quality, publication,
gateway, and checkpoint receipts supply the exact completion status; a scheduled
step is never represented as passed.

No test or publication step submits to Kaggle. Owner output and quality output
directories remain separate.

## Feature-research closure and recovery

All 15 feature-gate criteria pass after the complete current-profile group refits
and combined-omission study. The frozen manifest records 6,308 refit columns and
953 active inputs for the selected research trees; it does not treat lossless
pruning of an existing fit as feature selection for a future refit.

The final AWS review successfully verified the feature gate, raw inference,
organizer gateway, and all three notebooks. Its subsequent quality command
failed because Ruff included downloaded organizer Python in a source archive
without Git metadata. Explicit exclusions now keep raw data and generated
artifacts outside lint and format discovery. Two regression cases reproduce the
archive environment, preserve vendor bytes, and still reject invalid project
source. Recovery restored 215 files with verified hashes, including 106 complete
wide-ablation and simplification experiment files totaling 677,275,790 bytes,
from snapshot `30e0ce438b9d9f57a184487a55e0de20e47e59eff82ed4945b612c37c2f367e1`.

The fresh feature-gate review passed all 15 criteria and reused its verified
numerical evidence. Local publication executed notebooks 00, 01, and 02 with
29 consecutive code-cell outputs, nine Plotly outputs, eight embedded PNG
figures, and no errors or stderr outputs. All eight figures were inspected.

The complete public quality workflow passed in an isolated source archive with
no Git metadata or private fitted artifacts: 239 tests, lint, formatting, mypy
on 47 source files, synthetic execution/recovery, three notebook executions,
and isolated export validation. Run `20260909T020202Z-a62332af` completed all
13 checks in 115.75 seconds. An earlier attempt correctly stopped at the shared
workspace's free-space check; the successful review used a directory with more
than 5 GiB available and retained the original storage requirement.

The recovery reuses completed numerical experiments. It does not refit a final
model or score the reserved holdout.

## Final input preparation

The executable [final protocol](FINAL_PROTOCOL.md) has been run against the
verified research artifacts. Run `20260909T022425Z-2588324b` completed in 43.64
seconds: 72 input files, 224 games, 463,670 forecast rows, and 38,080 trajectories
passed the review. The rows comprise 395,813 original training rows and 67,857
development rows. The metadata-free and positional refit schemas retain 6,308
and 5,572 ordered columns. The 48 reserved games remain unscored.

The protocol source signature is
`c5b431034ad8457f8e70f027a726ec1cf3b8a66ba846ae73caaafbba7e97325f`.
A repeated preparation, `20260909T022624Z-e568bbf3`, completed in 38.25 seconds,
reused the verified input-review stage, and preserved the protocol bytes and
modification time exactly. The public preparation receipt matches the local
summary byte-for-byte.

The 26 added regression cases reject changed feature schemas and lineage,
unavailable input dependencies, holdout or mislabelled rows, missing frames,
reordered caches, extra player entities, missing or duplicated games, modified
source inputs, and silent replacement of the frozen plan. A fixture with no
outcome arrays or holdout files confirms that preparation needs neither.

The full public quality workflow passed in an isolated source archive with
11.87 GiB free and no private fitted artifacts. Run
`20260909T022705Z-1628cad6` completed all 13 checks in 165.62 seconds: 265 tests,
lint and formatting, mypy on 49 source files, synthetic execution and recovery,
all three canonical notebook executions, and isolated export validation. The
tested source files were compared byte-for-byte with the working checkout.
Preparation does not constitute a final model fit or a reserved accuracy result.

## Executed final preprocessing

Run `20260909T025200Z-55fc6931` refitted the physical baseline, strictly
earlier-date historical encodings, and route encoder on the frozen 224-game
partition in 29.41 seconds. Coverage checks confirm 927,340 coordinates, 38,080
trajectories, 1,111 historical player records, two roles, 16 route components,
and eight prototypes. Its source signature is
`eb6f17e15a1d2548d5738f92932b72ebf0b11b352255991ba6053893bb5592c6`.

Run `20260909T025326Z-ee41a0ef` reused all three completed components in
3.83 seconds. Their bytes and modification times were unchanged; the summary
was byte-identical. The integration test additionally interrupts route fitting,
then verifies that baseline and history checkpoints survive and are reused.
Other checks cover deterministic route ordering, excluded games, duplicate
entities, incomplete coordinate coverage, first-date history leakage,
nonfinite history values, and the prohibition on refitting after model sealing.

The six approved final-preparation files and their manifest were uploaded to
the private project bucket and downloaded back with exact SHA-256 and size
verification: 844,786 bytes. Snapshot
`91715ecdcb0625899e91c8a80c0481b0cf7bb4d913991353b188aed129183145`
contains 1,517 entries. The newly fitted preprocessing artifacts are verified
locally; their separate cloud upload has not yet been completed.

Final residual-tree fitting, prediction sealing, and reserved evaluation remain
unfinished. These preprocessing fits do not produce a new accuracy result.

The complete public quality workflow passed all 13 checks in 146.49 seconds
(`20260909T025414Z-3fbf3a31`): 269 tests, lint and formatting, mypy on 51 source
files, synthetic recovery, all three canonical notebooks, and isolated export
validation. It ran in a source archive without private fitted artifacts, and
the tested code was compared byte-for-byte with the working checkout.
