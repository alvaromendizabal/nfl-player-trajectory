# End-to-end test of the observed motion-state feature signal

Declared before fitting any model in this study. This is an adaptive follow-up
to the completed domain and motion ablations, on their same three expanding
chronological folds. The feature-completion gate stays open. The original
development partition, reserved partition and Kaggle remain outside this study.

Initialize two equal-capacity models in each fold from that fold's verified
40-epoch attention EMA. Append the thirty previously defined 3/8/20-frame
smoothed motion states to each observed temporal input. Initialize the added
convolution weights to zero and copy every original weight, including the
observation-mask channel. Initial predictions must match the saved reference
within 0.00002 yards. The control fixes the added standardized channels to zero;
the feature arm exposes them. Both train every original layer.

Compute these states for every observed player, including context players.
Fit the constant/duplicate screen and normalization on training-player states
and their deterministic lateral reflections only. Reflection occurs before
standardization; clip standardized inputs to [-10, 10], retaining masks and actual
time gaps. No target, player-ID value,
later observation or evaluation statistic enters this representation.

Both arms receive twelve additional epochs, the same batch size of 64 plays,
seed 2026, coordinate MSE, reflection probability 0.5, gradient clipping 1,
weight decay 0.01 and fresh AdamW with a cosine learning rate from 0.0002 to
0.00001. Use the final EMA with decay 0.98. This is a feature comparison within
a matched continuation budget, not a comparison between fresh and pretrained
models. No epoch, seed, feature subset or blend weight is selected on evaluation.
The per-arm training-time cap is 450 seconds; exceeding it yields an incomplete
experiment, not a reduced-budget result presented as complete.

The primary contrast is smoothed-state versus matched continuation control:
at least 1% pooled official coordinate-RMSE reduction, lower RMSE in all three
folds, and an upper paired whole-game 95% interval below zero, using 10,000
bootstrap draws. Report both arms against the frozen reference and every
0.5/1/2-second horizon slice. Equal 50/50 tree blends are secondary diagnostics.
Reused-fold uncertainty does not establish independent generalization.

Keep the historical categorical slots unchanged so pretrained predictions are
preserved. Audit actual input role names separately: a vocabulary naming error
is not automatically evidence of lost predictive information or a score gain.
Checkpoint optimizer, EMA, random state and batch position; verify interruption
and completed-run reuse before relying on resumability.
