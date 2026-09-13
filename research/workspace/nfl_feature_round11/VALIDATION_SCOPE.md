# Validation scope

## Completed in this response

- 49 software/base tests passed. They cover exact feature shapes, units, phase wrapping, stationary cases, receiver identity and self masking, per-channel missingness, masked poison, input immutability, reflection and player permutation, past-feature independence from later observed slots, label exclusion, atomic output, checkpoint reuse, path/symlink checks, no-training/no-online CLI and aggregate-only report export.
- Eight model/metric tests passed, using the original 19,250-parameter model definition copied unchanged. The no-optimizer test inspects actual called methods rather than rejecting descriptive `optimizer_steps: 0` metadata. The initial overly broad text-based test failed on that metadata; its log is retained.
- A synthetic 48-training-play / 288-row end-to-end fixture ran all six worker stages, including a fresh-process replay. It used deterministic untrained fixture weights; zero model optimization was performed. It verified 212 parent/source/raw fixture files and 108 feature/prediction checkpoint files unchanged. Evaluation tensor files were intentionally absent. Raw coordinate columns contained nonnumeric placeholders; only identity columns were parsed.
- Both notebook copies executed in a synthetic local harness: 11 code cells, nine Plotly outputs, zero stored errors on the successful pass. Actual stages ran through the worker entrypoint with fixture paths. The first combined container command had a 20-second execution limit and terminated the second notebook's kernel; the worker integration had passed. Notebook validation was rerun separately with a sufficient container limit, without changing the scientific code or AWS stage budgets. Both logs are retained.
- Source/report hashes and Round 10 squared-error accounting were checked. The uploaded replay-summary checksum matches.

## Not verified here

No private AWS tensor or model weight was present locally. The actual cached `uv` launcher, Python 3.11.15 / NumPy 2.4.6 / PyTorch 2.8.0+cpu environment, full 18-file raw-key scan, full private training replay, and live notebook execution remain for the user's run. The local environment was Python 3.13.5 / NumPy 2.3.5 / PyTorch 2.10.0+cpu. Exact cross-version numerical equivalence is not claimed; the launcher requires the original protocol environment and training probe hashes before proceeding.

No new private NFL accuracy metric, validation outcome, leaderboard rank, GitHub commit/CI run, cloud job, or Kaggle submission was produced. The delivered notebooks have real AWS paths and cleared code outputs, not synthetic results presented as NFL execution. Historical Round 10 charts use the uploaded aggregates. The observed feature candidates have not been fitted, screened for predictive gain, ablated or replicated.

The source files `audit_io.py`, `goal_frame.py`, `grouped_features.py` and `grouped_model.py` are verbatim copies from the reviewed Round 10 package; their hash equality is checked. The frozen model module contains the prior model definitions and helper functions, but the new CLI exposes no fitting action and the audit creates no optimizer.
