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

## Completed feature study and inference interface

- The owner's real-data feature run completed on 2026-09-07 at 23:48 UTC. Its
  uploaded summary matched the private S3 object SHA-256
  `d0af7f89e3b375e97e8cf5d74386a613539784d08412fb81811de3db80339371`.
- `docs/results/feature_selection.json` contains only selected-feature membership
  and standardized coefficient diagnostics derived from completed model object
  `c6ee5b69f0aa9a8acc0b12a25daf616c80784424d184663a92c4a6fdbcc9f6a0`.
  It is not a submission or an independently fitted model. `selection_study` in
  `src/nfl_trajectory/research.py` reproduces it from the local completed model.
- [Official organizer inference example](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission).
  The exporter follows this interface; package/export parity is tested separately
  from official gateway execution, which is not claimed complete.
- [Third-place competition writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution).
  Its pre-training/fine-tuning and small trusted feature set are research context,
  not a reproduced architecture, score, or guarantee for this project.
