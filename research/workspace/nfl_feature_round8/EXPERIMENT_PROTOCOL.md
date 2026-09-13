# Round 8 — matched learned pair-history representation

## Question and scientific boundary

Can a compact learned temporal encoder use changing player relationships more
successfully than an otherwise identical encoder receiving terminal relationships?

This is a **new forecasting model and new input interface**, not recovered old
neural weights, not a rerun of Round 3's tree columns, not an ensemble, and not a
claim that the leaderboard gap has been allocated to features rather than models.
It remains feature-representation research. Both experimental arms have identical
parameter counts, initial weights, optimizer, batch ordering, loss, and exposure.
The only treatment change is the observed pair sequence supplied to the encoder.

## Information and representation

Retain the final 20 actual observed frame slots for **all players with observations
in that window**, at most 22. Require each scored player to have observed support;
do not silently drop forecast rows. Do not compress gaps. Ten common focal-player value channels:
position relative to the player's latest observation, velocity, ball displacement,
orientation, ball distance, and closing speed. Each has an explicit validity mask.
Velocities use supplied telemetry when valid and consecutive observed differences
otherwise. No derivative bridges a missing frame.

Eight directed player-pair channels: relative position and velocity, separation,
closing speed, lateral relative speed, and velocity alignment. Channel-specific
validity masks distinguish missing measurements from real zeros. All self-pairs
are excluded. Values use fixed physical scaling, not validation statistics.

The history arm receives these actual sequences. The terminal arm repeats each
channel's latest valid observation over that channel's available slots. Both arms
receive exactly the same masks, source/destination roles and side, and the age of
the latest valid channel measurement. This is an ablation at the final forecast
origin; the terminal control is not claimed to be a causal prediction at earlier
observed slots. It never accesses post-origin positions.

A shared temporal convolution encodes each player and pair; learned weighting
aggregates other players. Weights are not labelled or calibrated coverage
responsibilities. Both arms also receive the preserved 72 base fields, scaled with
means and standard deviations from this fold's training rows only. This holds
common focal motion information fixed while attributing pair history.

No Round 7 response table, failed arrival/direct-state/goal bank, raw numerical ID,
future x/y, model prediction, or evaluation outcome enters this new feature builder.
Identifiers select tensor slots and join output requests only.

## Frozen experiment

Use the original Round 3 selection and **existing fold 2 only**: 26,039 training
rows / 64 games, 5,886 evaluation rows / 14 later games. Preserve every requested
row. No new game selection and no fold 3 fit is implemented in this release.
These are reused, selected training-side games, not independent confirmation.

Two CPU neural models jointly predict the x/y residual from the existing
constant-velocity reference. Training targets retain their original keys and
bytes; casting a working training tensor to float32 does not rewrite the labels.
Scoring uses the original float64 target array and all coordinate errors.

Settings: 24 epochs; 8 plays per batch; AdamW learning rate 0.001 and decay 0.0001;
seed 20260912; gradient norm cap 5; two CPU threads; no dropout, target augmentation,
validation checkpoint selection, early stopping, or fitted ensemble. Per-batch
squared-error sums use a fixed expected batch-row denominator, rather than
averaging each play equally. Each forecast row contributes to the training sum;
longer trajectories are not reduced to the same total weight as shorter ones. The plotted
training objective changes parameters within the epoch; it is not held-out RMSE.

## Engineering gates before any scientific fit

1. Re-read the completed Round 7 receipts and the retained source/data chain.
2. Independently replay the two preserved fold-2 control models; never refit them.
3. Build 32 training plays, rechecking the raw adapter after original float32 or
   float64 storage conversion. Require exact equality, not relaxed tolerances.
4. Prepare only this fold's selected plays; save per-play checksummed tensors.
5. Verify the pinned PyTorch CPU runtime and run all supplied model/recovery tests.
6. Profile 16 disposable optimizer steps on the largest training-only batch.
   This does not retain a scientific model and does not score evaluation data.
   The conservatively projected per-arm training time must be at most 300 seconds,
   within the enforced 360-second arm cap. Otherwise stop and return the report.

## Resumption, scoring and decision

Each scientific checkpoint saves the model, optimizer, CPU RNG, completed step,
initial-weight identity, and training loss history. Save every 25 steps and each
epoch. Interrupted stages resume the next step; completed models never refit.
Both arms use identical initialization and exposure, verified before scoring.
No validation RMSE is released by the single-arm training command. Evaluate only
when both arms are complete. Then run a separate fresh-process exact prediction
replay that refuses missing or incomplete artifacts.

Primary contrast: history minus terminal coordinate RMSE. Predeclared exploratory
gate: at least 1% relative reduction and a paired-game bootstrap 95% upper delta
bound below zero (10,000 draws, seed 20260912). This is one planned contrast on
reused games, not sequential-family error control over eight rounds of decisions.

A history win establishes only conditional first-fold feature evidence. Before
allocating a later-fold run, the history model must also be no worse than the
preserved tree on the same rows. The neural/tree comparison changes the model and
shared input representation; never attribute that difference solely to features.
No comparisons with the historical 0.62 blend or Kaggle leaderboard are valid
as controlled improvements. Neither model is submitted or promoted automatically.

Failure of both neural arms to reach the tree warrants a diagnosis of this compact
forecasting interface, exposure and representation, not a claim that history can
never help and not an automatic larger run.
