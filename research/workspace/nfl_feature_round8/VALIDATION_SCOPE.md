# Round 8 — validation scope and verification boundaries

## What was actually verified here

This is a local engineering validation, not a run on private NFL data and not a
claim that a historical leaderboard result has improved.

- **48 base tests passed** (`evidence/base_tests.log`): observed-only feature
  construction, exact frame masks, physics/units, translation/reflection,
  player/query permutation, invalid inputs, original storage precision,
  source/manifest changes and private-report boundaries.
- **20 model/recovery tests passed** (`evidence/model_tests.log`): common initial
  model state, masks, input immutability, nontrivial permutation checks,
  deterministic batching, exact clean-versus-resumed updates, full optimizer
  restoration, corrupt/changed checkpoint rejection and no-fit replay.
- The compact encoder has **14,931 trainable parameters in each arm**. The
  parameter count is also verified in the user's runtime before comparison.
- A **40-play synthetic pipeline**, including 32 training plays and 8 evaluation
  plays from two synthetic evaluation games, exercised the actual data-preparation
  and model pipeline. It retained 384 training rows and 96 evaluation rows.
  The fixture writes and then reads raw CSV bytes. Parent feature shards include
  both historical float32 and float64 storage. Old tree replay alone was
  substituted with a synthetic reference because private old weights are absent.
- All 40 completed feature checkpoints were reused on repetition. The terminal
  arm was deliberately interrupted at two optimizer steps and resumed. Both
  arms completed the fixed 24-epoch fixture exposure. No sample, model setting,
  label or stopping threshold was selected using a favorable fixture score.
- A **fresh Python process** reloaded both completed models and reproduced their
  predictions exactly with **zero new optimizer steps**. SHA256 checks verified
  **178 raw, parent, feature and model/checkpoint files** unchanged across replay.
  See `evidence/integration_final.log` and `evidence/synthetic_integration.json`.
  Metrics in this receipt are clearly labeled synthetic and are not NFL scores.
- Both notebook **copies** executed in a local synthetic harness:
  notebook 18: **6 code cells / 5 Plotly outputs / 0 errors**;
  notebook 19: **8 code cells / 4 Plotly outputs / 0 errors**.
  The harness substituted local paths and a fixture launcher. Data-stage cells
  verified precomputed synthetic preparation receipts rather than performing
  private AWS preflight. Actual plotting, training-only profiling, completed
  model replay, evaluation, and report export ran. The training notebook cells
  replayed previously fitted synthetic models, not additional scientific fits.
  See `evidence/notebook_validation.json` and its log.
- All nine Plotly figure functions validated and exported to HTML. The standalone
  `ROUND7_REVIEW.html` shows only actual supplied Round 7 aggregate results.
  Canonical delivery notebooks have empty outputs and no execution counts, so
  synthetic validation outputs cannot be mistaken for new private-data evidence.

## Earlier local failures retained

The first model-test fixture accidentally assigned the targeted-receiver role to
multiple players; the input adapter correctly rejected the ambiguous anchor. The
fixture was corrected to use unique anchor roles. The first full synthetic
integration fixture had only one evaluation game; the paired-game bootstrap
correctly refused it. The fixture now contains two independent evaluation games.
The original error logs remain in `evidence/`. Neither correction relaxed a
scientific check, changed private data, launched cloud compute or tuned the model.

## Exact environment boundary

Local numerical environment:

| Component | Local validation | Required AWS scientific runtime |
|---|---|---|
| Python | 3.13.5 | 3.11 (observed base environment previously 3.11.15) |
| NumPy | 2.3.5 | 2.4.6 |
| pandas | 2.2.3 | 3.0.5 |
| PyTorch | 2.10.0+cpu | 2.8.0+cpu |
| Plotly | 6.5.2 | 7.0.0 |
| Device / CPU threads | CPU / 2 | CPU / 2 |

The **exact pinned AWS runtime and its launcher were not executed here**. The
user's `runtime` stage verifies the repository's pinned CPU lock, imports exact
versions, and runs the 20 synthetic model/recovery tests before scientific fitting.
A missing CPU package cache permits only the documented isolated, bounded
`runtime --online` setup branch; the existing `.venv` is not replaced.

Within-runtime exact recovery is tested. Cross-version, cross-platform or
CPU/GPU byte-identical results are not claimed. PyTorch explicitly distinguishes
those guarantees in its official reproducibility guidance:
https://docs.pytorch.org/docs/stable/notes/randomness.html

## Uploaded report verification

The uploaded Round 7 ZIP was read directly. Its SHA256 and a per-arm numerical
accounting receipt are in `evidence/report_accounting.json`. Both fold-summary
hashes matched the supplied receipts. Fit and replay numerical content agreed.
Fold RMSE matched the sum of slice squared errors divided by twice the row count;
pooled metrics agreed with the fold sums. Historical table support diagnostics
come from the supplied preparation receipt.

I did **not** replay the user's private Round 7 models, read its private lookup
tables, reconstruct its bootstrap from individual errors or inspect live AWS
storage. Those are report-derived observations, clearly separated from the local
software tests. GitHub main was read through the connected API; the new kit was
not committed or merged.

## What this milestone still has to establish

Real-data support, speed, within-runtime recovery, actual paired RMSE benefit and
comparison with the preserved tree remain user-run gates. One successful fold
would remain exploratory on reused games. The test does not establish a new
Kaggle result, record performance, fully exhausted features, a 9.9/10 quality
rating or model-release readiness. The historical target remains 0.46340.
