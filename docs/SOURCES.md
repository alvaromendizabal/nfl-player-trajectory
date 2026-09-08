# Sources and research provenance

## Official task and input contract

- [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
- [Official data description](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
- [Organizer inference example](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)
- [NFL Big Data Bowl background](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)

The private competition snapshot supplies the actual organizer gateway, inference
server, protobuf/relay implementation, and unlabelled sample inputs. The validation
script records their individual SHA-256 hashes and the original snapshot digest.
It executes the unchanged API in an isolated quality directory. The source CSVs
supply the landing point, horizon, pre-throw tracking, role, and player metadata;
post-throw labels are unavailable to prediction.

Public Kaggle pages may require authentication. The downloaded source files and
their checksums are the operative interface evidence. No authenticated leaderboard
position or reproduced winning score is claimed.

## Methodology

- [Histogram boosting, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html):
  fixed iteration/leaf/bin settings and disabled random early stopping in the diagnostic.
- [Permutation importance, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/permutation_importance.html):
  reliance of a fitted estimator, sensitivity to correlated features, and repeated shuffles.
- [SageMaker Processing API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html):
  bounded resource and runtime configuration for resumable CPU experiments.

The experiment scripts pin scikit-learn 1.8.0 and their complete dependency
environment; the main project lock is unchanged. This project adapts permutation
to complete role/horizon-matched trajectories and combines it with chronological
refit ablations. Football feature hypotheses are tested here; citations are not
substitutes for local ablation evidence.

Earlier project notes linked a competition third-place writeup. Its implementation
and rank were not independently reproduced, so neither supports this project's
performance claims. External ratings/coaching data are deferred until an as-of
join, prediction-time availability, and legitimate use can be verified.
