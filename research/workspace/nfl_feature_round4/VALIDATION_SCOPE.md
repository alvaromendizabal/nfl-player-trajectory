# Validation scope · Round 4

## Completed locally

- 53 base feature/file-safety tests passed. They cover observed-state independence from query time, physical units, masks, stationary/coincident cases, reflection/translation, key/row permutations, target exclusion, float32 and float64 exact storage parity, one-ULP mismatch rejection, source/checkpoint drift and report exclusions.
- 10 model/recovery tests passed against the reused fixed-tree checkpoint implementation. These include controlled interruption/resume, exact no-fit replay, missing-checkpoint rejection and paired-comparison checks.
- A separate 64-play synthetic parent was constructed; its first fold used 40 plays. The actual new builder reused 32 smoke checkpoints, completed the remaining eight, then reused all 40 without rebuilding. The synthetic parent deliberately mixed float32 and float64 stored legacy X to exercise the previous failure boundary.
- Four new coordinate models completed. A separate process reloaded all new models and the two preserved control models and reproduced predictions exactly with zero new fits. Parent source/data/checkpoint hashes and modification times remained unchanged; new model hashes and mtimes remained unchanged during replay.
- Both notebooks executed through a local synthetic stage harness: 13 code cells, nine Plotly outputs and zero stored errors. All notebook-stage computational functions used the implemented production modules. Test copies changed paths and their subprocess invoker to the synthetic harness; they did not exercise the AWS identity check or the offline uv launcher. Explicitly labeled executed test copies are in `evidence/`.
- All delivered Python modules and notebook code cells parse with Python 3.11 syntax. Delivered canonical notebooks retain their actual AWS paths and have no synthetic outputs masquerading as NFL results.
- Actual Round 3 aggregate metrics were checked against the uploaded fit/replay reports and independently recomputed from reported squared-error totals and row counts. The prior failed smoke receipt was correctly identified as historical.

## Environment boundary

Local validation: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Plotly 6.5.2.

Target: the user's preserved Python 3.11.15 / NumPy 2.4.6 / pandas 3.0.5 environment and its locked scikit-learn 1.8.0 / Plotly 7.0.0 script dependencies. The first AWS command checks that environment and exactly replays the existing control before fitting anything new.

No new NFL accuracy, cloud execution, full-repository CI result, GitHub publication or Kaggle submission is claimed. Synthetic results test software behavior, not predictive usefulness. The original Round 3 stored replay is user-supplied evidence; its private model weights were not available in this container.

## Files retained

`evidence/base_tests.log`, `evidence/model_tests.log`, `evidence/integration.log`, `evidence/synthetic_integration.json`, `evidence/notebook_validation.json`, the two labeled executed synthetic notebook copies, `evidence/report_accounting.json`, and `evidence/local_validation.json` document these checks. `CHECKSUMS.sha256` covers delivered files other than itself.
