# Validation scope and known limits

## Completed locally

- 49 base tests passed, covering independent geometric units, exact endpoint
  offsets, missing landmarks, degenerate axes, role ambiguity, query identity and
  bounds, missing-value support, future-row rejection, irrelevant-target-column
  exclusion, input immutability, translation/rotation/reflection behavior, source
  paths, checkpoint checksums, result immutability, aggregate export privacy, and
  paired-comparison accounting.
- 10 preserved fixed-tree/recovery tests passed against the unchanged parent
  tree implementation used by this package.
- A synthetic 64-play, full ancestor-chain fixture completed the actual new
  worker entrypoint: source verification, parent-control replay, 32-play raw
  smoke, feature preparation/reuse, both fold fits (8 new coordinate models),
  and fresh-process exact prediction replay with zero refits. Parent bytes and
  modification times were unchanged. New weight bytes and modification times
  were unchanged on replay. Both legacy float32 and float64 storage were used.
- Identity checks remained enabled in the Round 6 worker. The fixture used its
  own recorded source hashes, actual temporary Git commit, numerical environment,
  and data receipts. A predecessor Round 4 feature-gate annotation was injected
  only into synthetic fixture reports to exercise the old continuation code;
  no actual NFL result was modified and no synthetic metric is claimed as NFL
  predictive evidence.
- A separate cost-guard fixture made both first-fold comparisons worse by
  changing synthetic comparison annotations. The actual fold-3 entrypoint
  recorded futility with zero fits and no fold-3 model directory.
- Both notebook copies executed: 11 code cells, 9 Plotly outputs, zero stored
  errors on the successful attempt. They used local fixture paths and the actual
  worker entrypoint in fresh subprocesses instead of the AWS offline-uv launcher.
  The initial validation call's 20-second tool cap interrupted the second
  notebook's kernel cleanup. That traceback is retained. Re-running under a
  60-second local cap completed both notebooks. No private compute was used.
- The uploaded Round 5 metrics, fold receipt hashes, fit/replay consistency, and
  pooled/slice squared-error accounting were checked. This is verification of
  the uploaded aggregates, not an independent replay of private AWS predictions.

## Environment boundary

Local Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Plotly 6.5.2.
AWS reports Python 3.11.15 and NumPy 2.4.6 in the recorded model environment.
The production runner uses the user's existing verified tree script lock via
OFFLINE uv; that exact AWS launcher/environment was not executed here. Python
3.11 syntax was separately checked. No dependency upgrade is recommended.

## Not completed or claimed

No new feature scores on private NFL data, no execution inside the user's AWS
space, no proof of inference parity on Kaggle, no newly submitted model, no
GitHub commit/push/merge, no top-score improvement, no independent holdout, and
no across-season validation. The unexecuted delivery notebooks contain the
actual AWS paths; synthetic execution outputs are not inserted as supposed NFL
results. Local synthetic evidence is retained under evidence/ and labeled.

The code does not recover the historical approximately 0.62708 blend or deploy a
new full-data forecast model. It answers a narrower feature-representation
question against the verified manual-study controls. Feature research is open.
