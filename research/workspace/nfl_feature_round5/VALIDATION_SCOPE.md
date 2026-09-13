# Validation scope · Round 5

## Completed here

- Parsed the entire uploaded Round 4 report archive and retained its original aggregate JSON bytes.
- Matched scientific fields between its summary and replay; reconstructed each RMSE from horizon and role squared-error sums. This is aggregate accounting, not an independent run of the private AWS model.
- **79 base tests passed**, including the retained raw-feature/float32/float64 parity tests and new chronology, train-only screening, game-weighted pooling, bootstrap signs/determinism, input validation, severe-loss stopping, report privacy, path safety and result-checksum tests.
- **10 model/recovery tests passed**, including interrupted warm-start continuation, fresh-process continuation, exact predictions, no-refit replay and tamper rejection.
- A synthetic 64-play integration fixture exercised the production numerical functions: 40 prior per-play state files reused, 24 missing states generated, 32 raw training-play examples independently checked, two later folds with eight new coordinate models, and fresh-process exact replay with zero new fits. Original parent data/source/model files retained identical bytes and modification times. New weight bytes and modification times were unchanged during replay.
- The synthetic fixture's discovery authorization was explicitly injected to exercise the continuation path, since that artificial data did not satisfy the real discovery gate. No synthetic score is claimed as NFL feature evidence. Production CLI has no skip-identity or fixture-authorization flag.
- Both notebook copies executed: **11 code cells, 9 Plotly outputs, zero stored errors**. Local paths and the stage launcher were replaced by the explicit synthetic harness. Numerical stages were run/reused, not fabricated as NFL results. The delivered notebooks keep the real SageMaker paths, empty new output cells and the normal offline launcher.
- All final Python sources parse with Python 3.11 syntax. The bundled numerical modules are exact copies from Round 4. Archive checksums and notebook structures were validated.

## Local environment

Python 3.13.5; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; Plotly 6.5.2; nbformat 5.10.4; nbclient 0.10.4.

The reported AWS environment is Python 3.11.15 / NumPy 2.4.6 / pandas 3.0.5 / scikit-learn 1.8.0. These are different environments. The production preflight requires the previously verified AWS versions; local success is not substituted for that check.

## Explicit limits

No private NFL arrays or model weights were available to this assistant environment. No new private NFL RMSE was computed. The exact SageMaker offline `uv` launcher was not executed locally. No GitHub commit/push/merge, AWS resource change, cloud job, data download or Kaggle submission was performed.

Interactive Plotly figures were serialized, exported to HTML and produced in executed notebook copies. Static PNG raster review was unavailable because the optional Kaleido package was not installed; no package was installed for that purpose. The deliverables require interactive Plotly, not Kaleido.

Engineering tests do not establish predictive feature value, independence of reused game folds, leaderboard comparability, or an exhausted feature space. See the experiment protocol and research ledger for the next scientific gate.

## Receipts

`evidence/base_tests.log`, `evidence/model_tests.log`, `evidence/synthetic_integration.json`, `evidence/integration.log`, `evidence/notebook_validation.json`, `evidence/notebook_validation.log`, `evidence/report_accounting.json`, and `evidence/local_validation.json` retain the successful checks and exact source hashes.
