# Sources checked September 6–7, 2026

- [Competition overview, evaluation, deadlines, and code requirements](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview)
- [Official data dictionary and file inventory](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
- [Official inference notebook by Sohier Dane and organizers](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)
- [NFL Big Data Bowl background and completed 2026 event](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
- [Official Kaggle CLI and authentication](https://github.com/Kaggle/kaggle-cli)
- [AWS CreateSpace API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateSpace.html)

The competition page explicitly supplies ball landing coordinates and output horizon,
uses 10 Hz tracking, and requires the evaluation API to return x/y for one play at a time.
The official training-file description names `input_2023_w[01-18].csv` and matching outputs.
The code discovers the actual inventory so it can handle additional published seasons.

The public page shows a disabled Late Submission button while signed out. This does
not establish whether the user's authenticated account can submit late. Authenticated download and all 18 weekly audits succeeded in the user's SageMaker
space. Phase 1 restores the private snapshot for development evaluation. The official
gateway and leaderboard submission remain pending.

