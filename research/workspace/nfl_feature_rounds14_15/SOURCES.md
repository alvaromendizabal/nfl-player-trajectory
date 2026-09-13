# Sources and attribution

Reviewed 12 September 2026. No connected project account was accessed.

1. Official final private leaderboard:
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard
   The winner scored 0.46340; the private leaderboard uses all test data. This is
   the competitive reference, not comparable to the local 14-game screens.
2. ohkawa3, First Place Solution, 4 December 2025:
   https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
   Uses 20 pre-pass frames, ball/receiver-relative information and player
   interactions. Augmentation, losses, architecture, training and ensembling
   also differ from our diagnostic pipeline. These new formulas are not claimed
   to be individually validated winning features.
3. Song et al., Decoding defensive coverage responsibilities, Amazon Science, 2026:
   https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
   Temporal/agent separation motivates representing movement and relationships
   distinctly. Coverage annotations and classification accuracy are not
   trajectory-RMSE evidence for the present features.
4. uv official CLI documentation:
   https://docs.astral.sh/uv/reference/cli/
   Offline execution uses cached packages. Frozen script execution preserves its
   lock. A missing cache is not permission to alter versions or reinstall.

New goal-aligned and receiver-response derivatives are our explicit hypotheses,
not demonstrated findings of these sources. Existing competition data is the
only input; no additional dataset is introduced.
