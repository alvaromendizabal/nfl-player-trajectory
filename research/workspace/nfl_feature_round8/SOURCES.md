# Primary sources checked for this milestone

## User evidence (the source of all Round 7 numbers)

`nfl_feature_round7_report.zip`, supplied in this conversation. Reviewed aggregate
receipts are bundled under `evidence/round7_*.json`. See REPORT_REVIEW.md and the
archive hash in input_contract.json. No public source supplies your private scores.

## External research (motivations, not measured gains for this code)

- Organizer: NFL Big Data Bowl 2026 Prediction overview and metric.
  https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview
- Official final leaderboard: historical winner 0.46340.
  https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard
- Winning author's 1st Place Solution: twenty pre-pass frames, dynamic/static
  information and a learned forecasting architecture. This package is not a copy
  or a reproduction of the winning system, its training or loss functions.
  https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
- Song et al., Decoding Defensive Coverage Responsibilities in American Football
  Using Factorized Attention Based Transformer Models. Separates temporal and
  agent relationships. Its annotations, coverage targets, accuracy and later
  observations are not imported into our coordinate-RMSE study.
  https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
- PyTorch reproducibility notes: exact replay is a property tested within the
  fixed environment, not a promise of equality across releases or hardware.
  https://docs.pytorch.org/docs/stable/notes/randomness.html

## Project runtime/source reads

Pinned GitHub main: 402843faa1722460aa84d1bbaf27c4050a7b7ff9.
The existing `scripts/motion_supervision.py.lock` uses torch 2.8.0 from the official
CPU wheel index; Git blob d50c46c0d6dbb2363bb2b6c2e2a9b0e35ace1140.
The data/control reader reuses `scripts/fit_final.py.lock`, Git blob
7214e46519f6e55c4e089e88add4d9240770ef3f. No original cloud runner is executed.
