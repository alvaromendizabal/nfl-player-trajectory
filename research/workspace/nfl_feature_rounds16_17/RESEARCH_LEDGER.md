# Feature and experiment ledger

| Family/workstream | Available evidence | Current action |
|---|---|---|
| R13 full observed motion | Supplied internal RMSE 0.694695, uncertainty gate failed | Promising experimental reference, not promoted |
| R14 goal-aligned derivatives | 32-play smoke passed, no supplied trained result | Resume existing three-arm study after recovery/profile |
| R15 receiver-relative dynamics | Notebook 32 records 32-play smoke passed | Resume existing three-arm study; do not tune from R14 |
| R16 route shape/persistence | Source and tests prepared; no execution | Equal-depth candidate; fitting held |
| R17 opponent approach/traffic | Source and tests prepared; no execution | Equal-depth candidate; fitting held |
| Larger same-game training sample | 2,653 extra plays and all 130,495 target rows reported valid | Explicit data-scale-only decision before more small-sample fits |
| Further game/fold confirmation | Many repeated decisions on 14 evaluation games | Needed before promotion/leaderboard claims |
| Inference integration | No new release from these kits | Must check raw inference, schemas and artifact lineage |
| Model/optimization/augmentation | Not ruled out by feature-first priority | Investigate when controlled evidence identifies a limitation |

Overlap is explicit: route summaries and encounter geometry existed in other
representations. R16/17 test their player-by-observed-frame interface conditional
on the same motion reference; they are not proof that a larger feature bank helps.
Negative, inconclusive, invalid and execution-blocked results are distinct.
No source/weights are automatically pushed, merged, uploaded or submitted.
