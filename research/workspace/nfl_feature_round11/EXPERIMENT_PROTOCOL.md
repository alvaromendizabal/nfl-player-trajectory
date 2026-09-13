# Round 11 — training fit, coverage, and observed representation readiness

## Frozen scope

No model fitting, optimizer steps, new evaluation predictions, hyperparameter selection, additional fold, ensemble or submission. Reuse both completed Round 10 models; evaluate them only on their selected training rows. Historical evaluation metrics are read from the existing summary. Training error is in-sample and cannot establish new predictive value.

## Questions

1. Does the final model already fit its training rows well, or is training error also high? A gap is descriptive: data quantity, distribution shift, regularization and optimization remain confounded. No numerical threshold chooses a model or budget automatically.
2. What fraction of all observed plays in the exact same training games does the small diagnostic subset include? Count only game/play IDs. Do not inspect unused labels, change the split, or select new plays. Availability of output labels for unused plays is not established by this count.
3. Are direct receiver-relative histories and adjacent-frame motion dynamics well supported and numerically stable in the current input adapter?

## Feature contract — 12 channels in two families

**Receiver (6):** relative x/y, relative vx/vy, separation, closing speed. Use the single organizer-labelled targeted receiver and simultaneous valid pair measurements. Self/receiver relations are masked; no receiver means missing values, not a guessed receiver. More than one labelled receiver is an input error. The original pair measurements are reused, not re-estimated from unsynchronized positions.

**Observed dynamics (6):** velocity-difference ax/ay; acceleration parallel and normal to current velocity; signed velocity-heading turn rate; change in speed magnitude. Differences require adjacent valid observations on the existing 20-slot, 10 Hz clock. Never bridge gaps or compress time. Direction-dependent terms are masked at near-zero speed. Use fixed documented scales and preserve true signed values. No clipping, outcome-derived normalization, future player position or hidden evaluation data is used.

These concepts overlap features in the larger repository and earlier tabular experiments. Their explicit 20-frame player-centric representation is being checked, not presented as new raw information or a proven gain. The saved neural models are not modified to accept them.

## Execution order

`preflight → coverage → smoke → features → audit → replay → report`

Each stage has a hard process limit and a 15-second launcher heartbeat. All stages remain offline. Coverage parses only identity columns from the existing 18 weekly input CSVs and retains per-file receipts. Feature checkpoints persist by play. Frozen-model predictions persist by eight-play batch and are private. A repeat requires exact agreement; a replay may not create missing features or predictions. File/metadata mismatches stop before use.

The first eight training plays must reproduce both original model probe hashes. No optimizer is constructed. The precise original numerical environment is required for that forward pass. A difference is a diagnostic stop, not permission to alter tolerances.

## Decision after the report

If substantial unused training-play coverage exists, investigate a **data-scale-only** comparison with fixed features and an unchanged evaluation contract; verify labels and budget before constructing a larger training set. If training fit is poor, investigate optimization/representation capacity before inferring that additional data or features alone solve the issue. If fit is strong but evaluation remains weaker, prioritize training support, distribution differences and controlled regularization/augmentation investigations. The current evidence cannot numerically allocate the leaderboard gap.

Only after reviewing this report should one new representation be wired into a same-capacity paired experiment. Do not revive failed geometric, direct-state, historical-response, or relationship treatments unchanged. No automated continuation is enabled.

## Data and privacy

No raw output CSV is opened. New feature construction never unpacks `y`. The frozen training audit does read already-authorized cached training labels. Raw input identity columns are scanned across files, but only current training-game plays are counted. Evaluation tensor files are not opened. Private per-game coverage, predictions, candidate tensors and weights are excluded from the return ZIP. SHA256 verifies retained artifacts, while numerical replay additionally verifies the computations it actually performs.
