# Evidence sources and boundaries

## User artifacts
- Uploaded `nfl_feature_round9_report.zip`; all entries inspected. Exact hashes and independent consistency checks appear in `evidence/report_accounting.json`.
- Earlier Round 8/9 code archives in this conversation. `goal_frame.py`, `parent_audit.py`, and the core local I/O/checkpoint routines are reused with provenance in the source. Copies do not modify the old kits.
- Connected GitHub read: main `402843faa1722460aa84d1bbaf27c4050a7b7ff9`, unchanged when inspected. AWS filesystem synchronization is not inferred from GitHub; the runner checks the local revision.

## Primary outside research (checked for this milestone)
1. Kaggle, *1st Place Solution*, NFL Big Data Bowl 2026 Prediction. https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
   The write-up uses 20 observed frames, ball/receiver-relative movement, supplied roles and inter-player modeling. It motivates representations; our compact two-arm experiment is not its model, training regime or ensemble.
2. Song et al. (2026), *Decoding defensive coverage responsibilities in American football using factorized attention based transformer models*. https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
   Separates temporal and agent dimensions and addresses receiver/defender interactions. Its coverage labels, later observations and classification accuracy do not enter our model or establish trajectory accuracy.
3. Official competition overview / metric. https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview
   The coordinate RMSE is sqrt(sum(dx²+dy²)/(2N)). All requested rows are retained in our matched evaluation. No claim of leaderboard equivalence follows from a matching formula on different data.

New grouping, forecast-conditioned attention and LayerNorm are implementation hypotheses motivated by the local sensitivity audit. They are not asserted to be proven by these external sources.
