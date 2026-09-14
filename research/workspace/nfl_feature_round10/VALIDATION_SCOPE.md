# Validation scope · Round 10

## Verified in this response

- Inspected all six entries in the uploaded Round 9 report. Verified the signal-summary SHA256 against the replay receipt, audit-signature agreement, 32-play/1,188-row population agreement and 2,376-coordinate intervention counts. Did not reconstruct hidden per-play probe aggregates or execute private weights.
- Connected GitHub read returned main `402843faa1722460aa84d1bbaf27c4050a7b7ff9`. No Git write or AWS deployment was performed.
- **67 base tests passed**: inherited goal-frame geometry tests plus matched-view, mask, missingness, no-target-input, immutable file, report-exclusion and offline-lock launcher tests.
- **24 model/recovery tests passed**: same initial weights, equal arm masks, preserved Cartesian channels, input/parameter gradients, deterministic updates, variable-player padding, player/query permutation, no-peer behavior, corruption rejection, exact optimizer continuation and metric/bootstrap checks.
- Synthetic end-to-end: 40 plays (32 training / 8 evaluation), 256 training rows, 64 evaluation rows. Both models completed 96 steps; the goal arm was interrupted at step 17 and resumed for 79. A separate fresh Python process reproduced both predictions and the summary exactly, with zero new optimizer steps. A subsequent call to each complete arm also reported zero new steps. **96 synthetic parent files and 97 model/tensor files remained byte-identical**. The input contract was replaced only inside an isolated fixture kit copy, never in the delivered kit.
- Both notebook copies executed in a local synthetic harness: **12 code cells, 9 Plotly outputs, zero stored errors**. The harness replaced paths and the pinned-runtime launcher and called actual stage functions on the separately contracted fixture. Runtime setup was substituted; training cells reused completed synthetic models. This is not the user's exact AWS runtime.
- All nine Plotly outputs serialized and exported to self-contained HTML. Static PNG export was attempted but unavailable because Kaleido is not installed locally; no static screenshot or pixel inspection is claimed.

The earlier intermediate test passes (65 base / 24 model) and first synthetic integration are retained as intermediate engineering evidence. The final test totals are 67 and 24, not their sum across repeated runs. No private NFL fitting occurred.

## Local versus required runtime

Local: Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, Plotly 6.5.2.

User neural worker requires the previously verified CPU lock: Python 3.11, NumPy 2.4.6 and PyTorch 2.8.0+cpu. It runs the 24 model/recovery tests in that environment before any private scientific fitting. No cross-version byte-identity claim is made. Recovery is verified within a single runtime; a numerical-environment change invalidates the protocol.

## Not established

- New private NFL RMSE, statistical feature contribution, out-of-sample improvement, success against the historical 0.46340 target or a new Kaggle score.
- Full competition-scale inference, matched replication across additional folds or seasons, checkpoint portability across platforms or a new production-quality deployment.
- Independent validation of Round 9's hidden per-play diagnostics; only its reported aggregates and internal receipt consistency were inspected.
- Individual benefits of shared group pooling, query conditioning or context normalization. The new within-study contrast attributes only added goal-channel values conditional on that shared interface.
- AWS launch, remote package installation, user-space execution, Git commit/push/merge, or S3/Kaggle writes.

## Evidence location

`evidence/base_tests_final.log`, `evidence/model_tests_final.log`, `evidence/integration_final.log`, `evidence/synthetic_integration.json`, `evidence/notebook_validation.json`, and `evidence/report_accounting.json`. Shipped notebooks have no prefilled synthetic experiment outputs; the first figures load the user's real Round 9 aggregates.
