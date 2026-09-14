# Validation scope and actual results

## Completed locally

- **81 base tests passed**: preserved Round 11 formulas, two new complete feature
  banks, exact core parity, finite output, real-zero vs missing support, receiver
  exclusion, ambiguous receiver rejection, contiguous differences/windows,
  reflection and permutation behavior, target/query exclusion, time causality,
  source/input mutation rejection, precision preservation and report privacy.
- **28 model/metric tests passed**: 19,826-parameter models, identical initialization
  and masks across arms, isolated numeric ablations, unchanged original pair
  values, zero initial outputs, input gradients, exact checkpoint restoration,
  clean-versus-interrupted continuation, metric factor, paired resampling and
  no-label model inputs. These runtime tests are rerun in the user's pinned CPU
  environment before experiments.
- Both **synthetic end-to-end studies** passed the real verifier, source/input
  contracts, feature preparation, profiles, all 3 new models, and evaluation.
  Each used 32 training plays / 256 rows and 8 evaluation plays / 64 rows.
  All models used the unchanged 24 epochs (96 optimizer steps in the fixture).
  Each full arm was interrupted at step 17, then resumed. A fresh subprocess
  reconstructed every feature and reproduced predictions with zero optimizer
  steps. Rerunning completed training functions produced zero new steps.
  **180 completed result files per round** and the recorded synthetic parent
  files remained byte-identical. See `evidence/synthetic_integration.json`.
- All **four notebook copies** executed: **26 code cells, 18 Plotly outputs,
  zero stored errors on their successful runs**. The delivered notebooks do not
  contain fabricated new NFL outputs; they have the real AWS paths and blank
  execution state for the new stages. Historical plots read supplied aggregates.
- Aggregate Round 11 RMSE accounting and its three replay hashes were independently
  verified from the uploaded report bytes. No private forward pass was performed
  here. The original aggregate receipts are retained under `evidence/`.

## Notebook-harness interruptions retained

The first combined notebook command exceeded its outer local tool deadline after
notebook 26 completed. The initial notebook-27 invocation also exceeded that
outer deadline; separate execution then passed using completed fixture models.
The combined 28/29 command completed notebook 28 before its outer deadline ended
notebook 29. Notebook 29 subsequently passed with the real worker entrypoint
called in the fixture kernel rather than spawning a worker process per cell.
No failed attempt is counted as a successful run; logs are retained. These were
local validation-harness interruptions, not AWS runs or scientific model failures.
The separate integration tests already establish fresh-process replay for both
rounds; an in-kernel notebook call is not substituted for that evidence.

## Environment and execution boundaries

Local environment: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, Plotly 6.5.2,
PyTorch 2.10.0+cpu, x86_64. User's previously verified environment: Python 3.11.15,
NumPy 2.4.6, PyTorch 2.8.0+cpu, two CPU threads. Cross-environment numerical parity
is not claimed. The package checks the exact parent numerical runtime and uses
the existing verified CPU script lock; no online setup is permitted.

The synthetic fixture has separate source contracts, a separate local Git
repository, synthetic data and local environment expectations. It exercises the
real authentication logic but cannot verify availability or bytes in the user's
stopped or running AWS filesystem. The notebook test copies use substituted paths
and launchers. The actual AWS offline `uv` launcher was not executed here.

## Not performed

No AWS compute/job/resource change, GitHub push/commit/merge, Kaggle submission,
private NFL training, new private metric, unused-play label verification, external
training-data acquisition, or new model promotion. Feature support on additional
channels is unmeasured until the user runs each preparation. Test success is not
evidence of predictive superiority, independent replication or a beaten record.
