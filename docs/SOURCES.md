# Sources and research provenance

## Official task and input contract

- [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
- [Official data description](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
- [Organizer inference example](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)
- [NFL Big Data Bowl background](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)

The private competition snapshot supplies the actual organizer gateway, inference server, protobuf/relay implementation, and unlabelled sample inputs. Validation records their hashes and executes the unchanged interface against private artifacts. Competition data and private neural champion weights are withheld. A small fitted public baseline is documented separately.

## Temporal model source

- [chack3 public first-place training reference](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)

The integrated baseline preserves the temporal/cross-player architecture, fixed normalization convention, optimizer behavior, EMA inference and grouped game folds needed for controlled comparisons. Credit for that architecture and training reference remains with its author; project-specific extensions are evaluated separately.

## Published modeling methods

- [Public first-place solution review: large feature-configuration / CV-diversity ensemble](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
- [Third-place solution writeup: dual-path modeling, auxiliary losses, cropping, fine-tuning and TTA](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution)
- [Fourth-place solution discussion: target-specific sparse pooling, landing context, single-player modeling and symmetry TTA](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814)
- [Fifth-place solution writeup: two-stage training, delta/future-frame modeling and historical-data use](https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/5th-place-solution)
- [Public NFL trajectory repository used as a non-medal feature-engineering hypothesis source](https://github.com/PatelRis/NFL-Big-Data-Bowl-2026-Player-Trajectory-Prediction)

These sources informed concrete modeling hypotheses and implementation choices. The project does not copy trained weights, hidden predictions, private datasets, or a competitor's complete feature implementation.

Each adapted technique receives its own project-owned implementation, leakage checks, validation, uncertainty analysis, and promotion/retirement decision.

## Research coverage and attribution

The research program evaluated temporal/player interactions, grouped-game folds, moving-average inference, augmentation, alternative supervision, interaction representations and ensemble diversity. An investigated technique is not automatically an accepted contribution: the dated evidence records its validation scope and promotion or retirement decision.

The completed system and its measured results are documented in [Results](RESULTS.md). This source catalog preserves the references that informed the work; it preserves attribution and the scope of each evaluated contribution.

No detailed second-place mechanism is claimed without a sufficiently detailed technical source. The project-owned additions emphasized data integration, controlled validation, efficient inference, recoverable execution and evidence publication.

## Earlier feature-engineering methodology

- [Histogram boosting, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Permutation importance, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/permutation_importance.html)
- [SageMaker Processing API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html)
- [Yuanzhiyi public NFL trajectory repository](https://github.com/YZY0108/nfl-player-trajectory-prediction)
- [Deep Sets (Zaheer et al., NeurIPS 2017)](https://arxiv.org/abs/1703.06114)
- [Trajectron++ (Salzmann et al., ECCV 2020)](https://arxiv.org/abs/2001.03093)

Earlier feature hypotheses and tree-model evidence remain preserved in RESEARCH_REVIEW.ipynb and the model card. They are not pooled numerically with the later neural-fold studies or private evaluation scores.
