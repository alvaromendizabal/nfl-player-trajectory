# Wide representation refit protocol

Recorded before running these group refits. The existing 250-feature strict
removals establish conditional contributions in that compact representation.
They do not establish the same effects in the full screened representation.

Use the width and availability profile already selected by pooled chronological
inner-fold RMSE. Hold the physical baseline, training/evaluation games, candidate
definitions, and fixed histogram-boosting settings unchanged. Begin with all
screened columns in that profile, before lossless pruning of unused tree inputs.
Remove each predeclared group without replacement and refit both coordinate
regressors. The disjoint groups in `src/nfl_trajectory/wide_ablation.py` cover all
20 catalog families. If a group is absent under the selected availability policy,
report an unchanged structural control and explicitly record that no fit occurred.

Run all three inner folds and development. Record coordinate RMSE, ADE, FDE,
p95 displacement, feature counts, and paired game-bootstrap RMSE differences.
The same game draws pair each comparison. These are descriptive intervals with
no multiple-comparison adjustment; development cannot choose a representation.
Retain individual-family trajectory permutations alongside these broader group
refits because correlated families can substitute within and between groups.

An omission remains a high-value feature avenue if it improves pooled inner RMSE
by at least **0.5%** and costs no individual inner fold more than **1%**. Such an
outcome keeps feature engineering open for a smaller-representation follow-up.
Otherwise, the refits supply attribution and support the previously declared
width/availability stopping criteria. These are project decision tolerances,
not statistical significance thresholds or proof of a global feature optimum.

Each fit and evaluation has its own source/input/output receipt. A verified
complete fold skips materialization on resume. Checkpoint each completed fold
privately; preserve partial completed fits on failure. The 48 reserved games
remain excluded from this experiment.
