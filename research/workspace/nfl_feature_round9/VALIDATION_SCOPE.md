# Validation scope — Round 9

## Actually executed locally

- 41 base tests passed. Covers known-unit projection, inverse projection within explicit float32 tolerance, reflection/permutation behavior, channel-specific missingness, same-side/opponent masks, exact role indexing, immutable inputs, label/query exclusion, per-slot causality, file safety, exact checkpoint reuse and aggregate-order stability.
- 14 model/signal tests passed against the exact original Round 8 Forecast source. A zero-output-head fixture produces zero sensitivity; enabling a synthetic head yields a nonzero pair-input response and nonzero input gradients. Weights stay byte-identical and no parameter gradients are accumulated. Repeat probes agree exactly; invalid masks and decoder hook cleanup are tested.
- A synthetic 32-play integration completed feature construction and both frozen-weight audits, reused all completed per-play diagnostics, and replayed in a fresh process with exact agreement. 171 synthetic parent/result files remained byte-identical. The evaluation file was intentionally absent, and target arrays were object-typed poison: successful execution demonstrates that those arrays were not unpacked by the audit.
- No model was fitted during this validation. Two untrained deterministic synthetic weight states were used; they are not NFL checkpoints or accuracy results.
- Both notebook copies executed in a local harness: 10 code cells, nine Plotly outputs, zero stored errors. The harness substituted local paths and direct worker calls, with the actual feature/audit functions. Fresh-process replay was tested separately through the integration subprocess.
- Existing Round 8 metrics were independently reconstructed from the uploaded report's horizon SSE and row totals. Completion/replay training objectives agree. The report's paired bootstrap interval was read, not independently reconstructed.

## Corrections retained in evidence

The first integration attempt caught a JSON dictionary-order issue: the first aggregation used insertion order while resumed JSON records used sorted keys. Sorting the presentation keys fixed exact aggregate replay; a regression test protects it. No feature formula, target, model, or metric was changed. The initial failure log is retained.

HTML exports and Plotly MIME outputs were generated. A separate pixel-rendering check was not completed: the default browser executable was absent, and the available managed browser blocked local file navigation. No browser controls were changed or bypassed. Pixel-level layout is therefore not independently certified.

## Environment and limits

Local: Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu. The user's completed environment is Python 3.11.15, NumPy 2.4.6, PyTorch 2.8.0+cpu. The delivered launcher uses the existing pinned CPU lock offline and requires environment equality with the completed Round 8 protocol. Cross-version numerical identity is not claimed.

The exact AWS uv launcher, user-private input arrays, saved NFL weights, and current SageMaker filesystem have not been executed or inspected in this response. Source and report checks cannot substitute for that boundary. The delivered notebooks retain the real AWS paths and have no fabricated new NFL execution outputs.

New goal-frame tensors are implemented and tested but have not been integrated into a trained model or shown to improve RMSE. Sensitivity on 32 training plays is not validation, an importance ranking, or a causal football explanation. No new private score, Kaggle submission, GitHub write, or AWS resource modification occurred here.
