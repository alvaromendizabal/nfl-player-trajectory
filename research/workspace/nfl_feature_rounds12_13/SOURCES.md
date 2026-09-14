# Primary sources and their limits

## User-provided evidence

`nfl_feature_round11_report.zip` (hash in `input_contract.json`), inspected directly.
Original JSON receipts are retained under `evidence/round11_*`. This evidence—not
web sources—establishes the current selected sample, training metrics and audit
status. No private model re-execution was performed while preparing this package.

## Domain research checked for this package

1. First-place author, NFL Big Data Bowl 2026 Prediction solution.
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
   Describes receiver-relative frame inputs, velocity decomposition, static role
   information, displacement targets, augmentation, auxiliary losses and neural
   modeling. Motivates the receiver representation; does not prove these extensions
   work. The winner's complete solution is not reduced to feature count alone.
2. Third-place authors, competition solution.
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution
   Discusses relative geometry and temporal modeling, but also notes more geometry
   did not help and uses additional training/modeling choices. Supports testing
   rather than assuming useful information from every plausible derived feature.
3. Song et al., Decoding Defensive Coverage Responsibilities in American Football
   Using Factorized Attention Based Transformer Models (2026).
   https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
   Distinguishes temporal movement and player relations in a different coverage
   classification task. No labels, post-throw observations, scores or weights are
   imported here. Its results do not establish our trajectory RMSE.
4. Competition overview / metric.
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview
   10 Hz tracking and coordinate RMSE over the requested target rows.

All new mathematical transforms are explicit experimental hypotheses. Their
formulas are not attributed to the above authors unless stated. No external
training data or paid service is accessed by these two rounds.
