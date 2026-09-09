# Validation record

The final release has **296 automated tests**, including final lineage,
export, immutable evaluation, error concentration, and notebook-cache contracts.
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
| Reserved holdout | Frozen final model: 0.80466993 RMSE on 48 later games, with game-cluster uncertainty |

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


## Final fitter implementation and executed checks

The final fitter now consumes the newly fitted preprocessing and both frozen
ordered schemas. It preserves independent x/y checkpoints, rejects stale
preprocessing and altered fit plans, verifies every portable training prediction,
and binds exports to both coordinate-model hashes. A regression test corrupts one
coordinate checkpoint and verifies that rebuilding it invalidates only its own
profile's export while preserving the other completed fits.

The complete quality run `20260909T032709Z-87ed12b7` passed all 13 checks in
194.79 seconds: **280 tests**, Ruff lint/format, mypy on 54 source files,
synthetic execution and recovery, all three canonical notebook executions,
and isolated exports. The source and test bytes match the reviewed checkout.
A report typing issue found by the first run was corrected before this clean run.

The separately locked sklearn 1.8.0 numerical check fitted 1,024 synthetic rows
and 20 columns, retained 16 active inputs, and achieved exactly zero portable
prediction difference. It recovered a controlled y-fit interruption while
preserving x bytes and modification time. A completed repeat needed no training
matrix. The [numerical receipt](results/final_fit_validation.json) records its
scope and implementation hashes; CI runs this check in addition to the offline
suite.

The [real-input feature receipt](results/final_feature_validation.json) verifies
one complete play per week across all 15 training weeks, totaling 738 forecast
rows. Both the 6,308-column primary schema and the 5,572-column positional subset
matched raw-input reconstruction exactly. Run `20260909T032707Z-04157bd8`
completed in 40.36 seconds without opening reserved outcomes.

Full-scale final-model training has not run. The preprocessing snapshot is now uploaded and download-verified; final raw inference,
prediction sealing, and reserved scoring remain unfinished. None of these checks
adds a new accuracy estimate or replaces the completed research predictor.


The approved preprocessing backup completed with all 10 files and its manifest
verified by download, byte count, and SHA-256 (4,774,589 bytes). The cloud runner
now supports the final fitting phase and rejects a 64 GiB worker before any AWS
access. Full-scale fit results are reported only after execution and inspection.

The cloud-runner checkpoint passed all 13 quality checks with **281 tests**,
54 typed source files, and all three canonical notebook executions. Run
`20260909T035227Z-96a13d58` completed in 183.21 seconds. Snapshot verification
also matched all 55 private inputs bound by the final protocol.

## Final fitting and inference release

SageMaker job `nfl-final-fit-20260909-040036` completed all four coordinate
fits and both full-row portable conversions in 683.698 seconds of fitting.
The worker completed after 991.309 seconds including preparation and backup.
No replacement job was launched. Both schemas were refitted on all 224 games:
6,308 and 5,572 input columns; 951 active columns in each exported model; zero
original/portable difference on each profile’s 463,670 training rows. These
training predictions establish conversion integrity, not generalization.

All 18 required model files (29,362,995 bytes) were downloaded from the completed
1,557-file snapshot and independently hash checked. The final loader also checks
source, protocol, preprocessing, coordinate and export receipts.

The local full suite passed all 13 quality checks, including 288 tests and all
three canonical notebooks, in 251.636 seconds (run `20260909T041738Z-a16bea69`).
Subsequent targeted tests verified preserved request order through outer-key
scoring joins; the final gateway integration separately verifies its named RPC
endpoint. The exact published revision also runs through GitHub CI.

New tests exercise rejected stale models, final date boundaries, standalone
parity, partial/nonfinite telemetry, target-coordinate poisoning, holdout
request construction without outcomes, missing/duplicate scoring keys, a
controlled prediction-stage interruption, immutable seals, and resuming the
same evaluation without reopening outcomes.

The actual final-model gateway passed all **5,837 rows across 143 plays**
(run `20260909T042021Z-7493b516`, 313.948 seconds). Every package and standalone
prediction matched exactly. Metadata omission, missing telemetry, cold history,
shuffled request/input rows and ignored target coordinates also passed. The
initial local integration failure registered the callback under the wrong name;
using the organizer-required `predict` endpoint resolved it. The organizer
source was not modified, and no additional AWS job was needed.

## Sealed evaluation and final publication

The expanded local quality run `20260909T044516Z-4a0dd1dc` passed all 13 checks
in 219.997 seconds: 294 tests, 60 typed files, three canonical notebook
executions, and both synthetic recovery runs. Two subsequently added numerical
diagnostic tests check exact error mass and refuse incomplete reference keys;
the final release contains 296 tests and 61 typed files. Exact-revision CI is
required before merge.

The sealed snapshot contains 1,576 entries. All 16 newly uploaded objects,
including the manifest, were downloaded back and verified (11,551,607 bytes).
S3 records the manifest at 04:49:14 UTC, before outcome access at 04:51:09 UTC.
The [backup receipt](results/final_sealed_backup.json) records its hash.

Run `20260909T045104Z-36b7569c` evaluated all 99,266 requested rows with exact
one-to-one key coverage in 17.651 seconds. The official coordinate RMSE is
0.8046699305544704. A 2,000-resample game bootstrap gives 0.66313–1.00113.
Replaying under the same seal reused the completed stage and preserved error,
summary, outcome-access and seal bytes and modification times. Published
fit, inference, seal and score reports have a separate verified manifest.

The actual final exported notebook matches all 5,837 organizer sample outputs
exactly. Its initial gateway run took 181.849 seconds and its immediate rerun
4.415 seconds using completed per-play checkpoints. After a type-only generator
correction, the generated notebook hash remained identical and both verified
reruns reused the same outputs. The public export receipt binds the current
generator, shared template, model bundle, and notebook hashes.

The final notebook explicitly distinguishes 0.68805 development RMSE from
0.80467 reserved RMSE. The error-concentration audit retains every observation
and confirms both the recomputed official score and reference key coverage.
Long-horizon support, weekly results, role differences, and concentrated error
are shown beside the headline metrics. These observations cannot trigger new
holdout-driven model selection.

Strict publication also detected six missing historical frame-error files in the
recovered local workspace. All six were restored from the existing fitted S3
snapshot and verified against their original hashes: 714,037,121 bytes. No
research experiment was refitted. Execution used a canonical source path after
relocating the workspace to storage with adequate room, preserving provenance
checks instead of changing the frozen research implementation.

Final publication run `20260909T050014Z-0d532dcd` completed in 84.092 seconds.
All 41 published file hashes matched. The three canonical notebooks contain
31 consecutively executed code cells, 11 Plotly figures and 10 embedded PNG
figures, with no error or stderr output. The two new final-result figures were
visually inspected for readable axes, legends and complete rendering. The
[publication receipt](results/final_notebooks.json) binds their exact bytes.
