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

The source-faithful reproduction preserves the temporal/cross-player architecture, fixed normalization convention, optimizer behavior, EMA inference, and grouped game folds needed for controlled comparisons. The captured upstream source used in the private reproduction has SHA-256 `6be46a4a8beccf14a65bbe3a18fd233cd00fc7d2281b1f396c08e5ba7e6f0fc2`. Reproducing one training component is not a claim that the complete winning ensemble has been reproduced.

## Leading-solution mechanisms used as research hypotheses

- [Public solution writeup: large ensemble / CV diversity reference](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
- [Third-place solution writeup: player dropout and broader training recipe](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution)

These sources are used to identify mechanisms to reproduce independently. This repository has evaluated training-seed diversity, protected context-player dropout, geometric variants, mirror inference, and equal-weight ensembling. Alternate game-grouped CV split diversity is the next prepared test. Broader pretraining, multi-auxiliary supervision, and substantially larger feature/CV ensembles remain open gaps.

The project does not import another competitor's trained weights, private data, or hidden predictions. Each adapted technique receives its own implementation, leakage checks, validation, uncertainty analysis, and promotion decision.

## Earlier feature-engineering methodology

- [Histogram boosting, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Permutation importance, scikit-learn 1.8](https://scikit-learn.org/1.8/modules/permutation_importance.html)
- [SageMaker Processing API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html)
- [Yuanzhiyi public NFL trajectory repository](https://github.com/YZY0108/nfl-player-trajectory-prediction)
- [Deep Sets (Zaheer et al., NeurIPS 2017)](https://arxiv.org/abs/1703.06114)
- [Trajectron++ (Salzmann et al., ECCV 2020)](https://arxiv.org/abs/2001.03093)

Earlier feature hypotheses and tree-model evidence remain preserved in `RESEARCH_REVIEW.ipynb` and the model card. They are not pooled numerically with the newer reproduced neural folds or private leaderboard scores.
