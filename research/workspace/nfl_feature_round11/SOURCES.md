# Primary sources and how they are used

1. Competition overview / organizer RMSE and 10 Hz tracking: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview
2. First-place author, Feature Engineering sections: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
   The page is titled *1st Place Solution* despite its legacy URL. It describes receiver-relative dynamic inputs, static role/horizon information, auxiliary motion losses, augmentation and a much larger modeling/training program. It motivates the receiver-reference hypothesis; it does not validate this 12-channel implementation or isolate the reason for our gap.
3. Third-place authors: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution
   Used to cross-check that representation, data construction, augmentation and training all contributed; not to assert our features alone explain performance.
4. Song et al., factorized temporal/player representation for coverage: https://www.amazon.science/publications/decoding-defensive-coverage-responsibilities-in-american-football-using-factorized-attention-based-transformer-models
   Different task and additional annotations. No coverage labels, classifier scores or post-throw observations are imported.

User-provided report and source packages are the authority for project results. Web sources motivate hypotheses and are not substitutes for those measurements. No raw competition data is redistributed.
