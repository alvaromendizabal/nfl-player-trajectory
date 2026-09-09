# Recovery

Run the same download, audit, demo, or benchmark command after a failure. Atomic completion
records ensure incomplete work is rerun and completed, verified work is reused.
Local logs record both total elapsed time and stage/file progress.

Download resume has two levels: completed files are verified and skipped; an incomplete
file is retained with the official Kaggle client's identity marker, allowing byte-range
resume when the server supports it. Incomplete bytes are never accepted as complete.

Before ending a phase, run `nfl backup` while other pipeline writers are stopped.
The private S3 bucket has versioning and encryption enabled. Identical bytes reuse a
content-addressed object. A manifest is published only after all selected files upload.
An upload interrupted mid-file may restart that file; already completed objects are reused.
Active logs, partial downloads, credentials, environment files, and local AWS config
are not included. Do not delete a space before a backup succeeds.

To recover after losing the space, clone or extract the same code into a fresh project,
run bootstrap, then use the exact manifest identifier from the successful backup:

```bash
.venv/bin/nfl restore --bucket YOUR_PRIVATE_BUCKET --manifest snapshots/YOUR_MANIFEST_HASH.json
.venv/bin/nfl audit
.venv/bin/nfl status
```

Replace both placeholders with the recorded values. Restore checks the manifest hash
and every file hash. It refuses divergent existing files so it cannot overwrite newer
work. Restoring into a fresh directory is the safest recovery path. If the source or
dependency lock changed, audit checkpoints are invalidated and recomputed.

The benchmark checkpoints weekly feature extraction, training statistics, model fitting,
validation errors and latency. It hashes input files, the frozen split, numerical
source and the dependency lock; each completed output is checksum verified before
reuse. An interrupted or corrupted stage is recomputed. Unchanged verified stages
are retained. Reports are regenerated cheaply from these numerical artifacts.

The feature gate is closed and final input preparation is implemented. The next
fit uses the frozen shallow-tree estimator; no neural trainer is part of the
final protocol. Resume preparation with `scripts/prepare_final.py`. It rechecks
the research evidence, verifies input hashes, and reuses the completed
`final-input-review` stage. The canonical `artifacts/final/protocol.json` is
immutable on ordinary reruns. Preserve it, its source commit, and its input
review before starting the final fit. The published preparation receipt in
`docs/results/final_protocol.json` records its signature and verified counts.

The approved preparation snapshot is
`91715ecdcb0625899e91c8a80c0481b0cf7bb4d913991353b188aed129183145` in the
existing private project bucket. All six added files and its manifest were
downloaded back and verified by SHA-256 and byte count: 844,786 uploaded bytes,
1,517 manifest entries. It extends the preserved research snapshot.

Final preprocessing now checkpoints the baseline, historical encodings, and route
encoder independently. Run `scripts/refit_final.py` to reuse valid components or
finish an interrupted stage. The integration test interrupts route fitting and
verifies that earlier component bytes and modification times are preserved.
Final residual-tree fitting is implemented with separate coordinate checkpoints.
Its full-scale execution and prediction sealing remain unfinished. Portable
conversion checks every training row, and its checkpoint binds both coordinate
model hashes. A completed fit reuses its verified models without materializing
the training matrix again. Future evaluation must bind the same model and
predictions on resume. Holdout outcomes remain unscored. See
[FINAL_PROTOCOL.md](FINAL_PROTOCOL.md).

The new preprocessing fits are verified locally and their compact receipt is
versioned with the source. They have not yet been added to a verified cloud
snapshot. Preserve that checkpoint before the expensive residual-tree stage.
