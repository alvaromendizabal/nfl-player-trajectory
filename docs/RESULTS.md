# Results and evidence

The completed project delivered **0.46468 recorded private coordinate RMSE** in NFL Big Data Bowl 2026 Prediction. Across its recorded submissions, error decreased **33.7%**, from **0.70090 to 0.46468**. This describes successive systems, not a controlled single-component ablation.

The authoritative closeout summary is [project_closeout.json](results/project_closeout.json). Provenance distinguishes public historical evidence from aggregate facts checked against owner-retained execution records. A hash identifies a record; it does not make a private record publicly reproducible.

## Submission history

| Recorded system | Private coordinate RMSE | Submission reference |
|---|---:|---:|
| First recorded system | 0.70090 | 56129348 |
| Preserved base ensemble | 0.46615 | 56470102 |
| Expanded ensemble | 0.46547 | 56478271 |
| Previous accepted system | 0.46487 | 56623023 |
| Final accepted system | **0.46468** | **56928100** |

Earlier entries are preserved in the [public history](results/frontier_submission.json); the final result was checked against the completed scored response. These are **late-submission private evaluation results**, with no official competition placement claimed. The final improvement over the preceding accepted system is 0.00019 RMSE.

## Validation populations

Coordinate RMSE is the square root of mean squared error across both x and y coordinates, in yards. It is not mean Euclidean distance.

| Evaluation | Rows | Games | Coordinate RMSE |
|---|---:|---:|---:|
| Final recorded private submission | Hidden | Hidden | **0.46468** |
| Supported-population OOF, final policy | 561,607 | 272 | **0.4629258204** |
| Complete-population OOF, final policy | 562,936 | 272 | **0.5242026277** |

The complete audit restores **1,329 rows** outside the historically supported subset, including difficult long-horizon and missing-role cases. Its higher RMSE reflects a broader population. It must not be described as a regression against the smaller-population metric or conflated with the hidden private score.

Local OOF uses game-heldout predictions. Development results have been inspected repeatedly; they are not an untouched test guarantee. The earlier 0.4631723213 OOF result remains in dated snapshots and belongs to an earlier inference policy. Historical chronological holdouts and the synthetic demo also have distinct scopes.

## Measured inference engineering

On **96 plays / 3,723 rows**, shared preparation accelerated the fixed ensemble **4.784×**, from **0.541114 to 0.113100 seconds per play**, with **zero maximum coordinate difference** on that sample. This is a measured implementation improvement under the recorded setup, not an across-hardware production latency guarantee. The original [aggregate measurements](results/frontier_submission.json) remain public.

## Public demonstration evidence

[Route Lab](https://alvaro-nfl-route-lab.tartmacaw2.chatgpt.site) computes fixed motion forecasts over authored synthetic trajectories. Its coordinate RMSE, average displacement error and final displacement error describe the selected demonstration scope only. Changing horizon or damping changes the actual forecast; it does not tune or evaluate the historical neural ensemble. The Python report remains a separate deterministic two-reference example.

## Evidence beyond a score

- Audited **4,880,579 observed player-frames**, **562,936 forecast player-frames**, and **272 games**, with explicit prediction-time contracts.
- Built direct-source historical context with point-in-time restrictions and player/game identity reconciliation. Coverage is recorded in the [dated research snapshot](results/current_research_snapshot.json).
- Used whole-game uncertainty, matched controls and negative-result retirement to distinguish promising ideas from accepted improvements.
- Delivered recovery, artifact identities, resource checks, cost telemetry and source-to-submission lineage on AWS.

The [public demo](REPRODUCIBILITY.md) reproduces engineering and evaluation contracts on invented data. It does not reproduce private-model scores. The [model card](MODEL_CARD.md) records intended use and limits.
