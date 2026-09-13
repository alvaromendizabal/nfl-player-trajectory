# Primary sources and evidence boundaries

## User-provided evidence

`nfl_feature_round8_report.zip`: exact source hash is stored in `input_contract.json` and `evidence/report_accounting.json`. Scores, gates, runtime state, completion and replay receipts come from this archive. Its per-game errors and final checkpoint blobs are absent; their numerical evaluation was not independently rerun during package preparation.

## Outside research (checked in this response)

1. Kaggle, **1st Place Solution**, NFL Big Data Bowl 2026 Prediction. The page's title is “1st Place Solution” despite its `public-3rd-solution` URL slug.
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
   Motivates dynamic/static representation and player-interaction research. It attributes its full solution to data augmentation, losses and architecture as well as features. It does not demonstrate the value of this package's six formulas.
2. Song et al., **Decoding defensive coverage responsibilities in American football using factorized attention based transformer models**, Amazon Science, ISACE 2026.
   https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
   Separates temporal and agent information in coverage prediction. Its labels, timing and classification scores are not our permitted trajectory-input contract or coordinate RMSE. No annotations or trained weights are imported.
3. PyTorch **2.8 autograd documentation**.
   https://docs.pytorch.org/docs/2.8/autograd.html
   Supports input-gradient inspection. Input sensitivity does not establish predictive feature value. The package's tests and read-only weight checks are additional local engineering evidence.

The goal-frame prototype is an explicit geometric construction from existing supplied inputs, motivated by the task—not a feature ranking asserted by these sources. No external data is downloaded.
