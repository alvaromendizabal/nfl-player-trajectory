# Sources and research provenance

## Official task and input contract

- [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
- [Official data description](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
- [Organizer inference example](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)
- [NFL Big Data Bowl background](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)

The private competition snapshot supplies the actual organizer gateway, inference server, protobuf/relay implementation, and unlabelled sample inputs. Validation records their hashes and executes the unchanged interface against private artifacts. Competition data and fitted weights are not redistributed.

## Reproduced neural baseline

- [chack3 public first-place training reference](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)

The source-faithful reproduction preserves the temporal/cross-player architecture, fixed normalization convention, optimizer behavior, EMA inference, and grouped game folds needed for controlled comparisons. Reproducing one training component is not a claim that the complete winning ensemble has been reproduced.

## Leading-solution mechanisms used as research hypotheses

- [Public first-place solution review: large feature-configuration / CV-diversity ensemble](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
- [Third-place solution writeup: dual-path modeling, auxiliary losses, cropping, fine-tuning and TTA](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution)
- [Fourth-place solution discussion: target-specific sparse pooling, landing context, single-player modeling and symmetry TTA](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814)
- [Fifth-place solution writeup: two-stage training, delta/future-frame modeling and historical-data use](https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/5th-place-solution)
- [Public NFL trajectory repository used as a non-medal feature-engineering hypothesis source](https://github.com/PatelRis/NFL-Big-Data-Bowl-2026-Player-Trajectory-Prediction)

These sources are used to identify mechanisms to recreate independently. The project does not copy trained weights, hidden predictions, private datasets, or a competitor's complete feature implementation.

Each adapted technique receives its own project-owned implementation, leakage checks, validation, uncertainty analysis, and promotion/retirement decision.

## Current reproduction boundary

Substantially tested/covered:

- first-place temporal/player interaction base
- grouped game folds and EMA
- strong augmentation and multiple split families
- dual-path temporal/spatial interaction and auxiliary supervision
- target-specific interaction and landing context
- single-target supervision
- multiple TTA recipes
- delta/future-frame decoding ideas
- Muon and robust-objective-related axes
- competition-only pseudo-supervision adaptations

Largest remaining competition-data-only gap:

**independently trained feature-configuration breadth combined with repeated grouped-CV split diversity at substantially larger ensemble scale.**

Historical-data pretraining remains blocked by the active competition-data-only project boundary.

No detailed second-place mechanism is claimed because no sufficiently detailed trustworthy technical writeup has been established in this project.

## Earlier feature-engineering methodology

- [Histogram boosting, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Permutation importance, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/permutation_importance.html)
- [SageMaker Processing API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html)
- [Yuanzhiyi public NFL trajectory repository](https://github.com/YZY0108/nfl-player-trajectory-prediction)
- [Deep Sets (Zaheer et al., NeurIPS 2017)](https://arxiv.org/abs/1703.06114)
- [Trajectron++ (Salzmann et al., ECCV 2020)](https://arxiv.org/abs/2001.03093)

Earlier feature hypotheses and tree-model evidence remain preserved in RESEARCH_REVIEW.ipynb and the model card. They are not pooled numerically with the newer reproduced neural folds or private leaderboard scores.
