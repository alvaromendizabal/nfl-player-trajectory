# Validation scope

## Completed locally

38 feature/file-handling tests and 10 tree/recovery tests passed (48 total). The model tests include fresh-process interrupted-checkpoint continuation against clean fitting, exact completed replay, source/data drift rejection, and refusal to refit missing models during replay. See evidence/base_tests.log and evidence/model_tests.log.

A synthetic 64-play integration fixture exercised preserved parent rows, key-only sample expansion, 32-play feature checkpoint reuse, eight first-fold coordinate estimators at the frozen 120-tree exposure, a separate-process zero-refit replay, report creation, and unchanged raw/model hashes. A repeated preparation reused all 64 feature shards. This used no private NFL tracking or labels.

All delivered Python files and 11 code cells in two notebooks passed Python 3.11 grammar checks; notebook structures passed nbformat validation. Nine Plotly figures serialized and exported to standalone HTML. The two Round 2 review figures use actual uploaded aggregates. Other locally tested figures use explicitly synthetic fixtures, which are not inserted into the delivered AWS notebooks.

## Not completed locally

The full two AWS notebooks and the copied-lock uv setup were not executed in this local environment. Local Python is 3.13.5, NumPy 2.3.5, pandas 2.2.3, Plotly 6.5.2 and scikit-learn 1.8.0. The user runtime remains locked to Python 3.11 and the existing repository's dependencies. The runtime stage must pass its synthetic recovery self-test in that environment before an NFL fit.

Browser screenshot review was unavailable: Playwright's default browser was absent and the installed system Chromium blocked navigation under administrator policy. HTML export and Plotly schema checks passed; no pixel-perfect browser review is claimed. No browser installation or policy bypass was attempted.

No live AWS compute, GitHub writes, Kaggle submissions or new NFL scores were produced here. Software tests do not establish feature benefit.

## Review correction

Original and replay Round 2 reports have identical numerical metrics, fold metrics, slices and decisions, but execution counters change from new fits to reused fits. They are not byte-identical. Their pooled RMSE values were independently reconstructed from their fold row counts and RMSE.
