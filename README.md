# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with controlled sequence modeling, ensemble diversity, GPU engineering, and reproducible AWS research.**

[Latest frontier review](docs/FRONTIER_RESEARCH_POST_PR40.md) · [Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [Model card](docs/MODEL_CARD.md) · [Recent model review](research/RECENT_MODELS.ipynb) · [Run and reproduce](START_HERE.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. Competition slug: nfl-big-data-bowl-2026-prediction. AWS/SageMaker is the canonical research workspace; Kaggle is used only for the required submission surface.

## Current research snapshot

The strongest recorded late post-competition private submission remains **0.46487 coordinate RMSE**. The published first-place private comparator is **0.46340**, leaving a **0.00147 RMSE gap**. No official competition rank is claimed for late submissions.

The strongest completed local system remains the **20-model, four-split-family equal-weight ensemble** at **0.4631723213 OOF RMSE over 561,607 rows**. Local OOF and private leaderboard measurements are intentionally kept separate.

Since PR #40, the project completed a much broader controlled frontier program. The key finding is increasingly clear: **small adaptations of an already fitted parent have mostly plateaued, while independently trained model/split diversity remains the strongest proven path.**

| Research family | Aggregate evidence | Decision |
|---|---|---|
| Competition-only pseudo-supervision | Completed controlled variants; no confirmation-stage promotion | Retired exact branch |
| ST-GRU / landing-node ST-GRU | Standalone Fold-0 scores were far behind the incumbent | Retired |
| Zero-fit meta/post-processing | Best full-OOF gain was only ~0.00005 RMSE | Retired |
| Frozen-parent intent/temporal feature adapters | Best fixed-blend gain was ~0.00017 | Retired |
| Full-parent coverage/physics fine-tune | Candidate gains were nearly indistinguishable from matched control | Retired |
| Source-native configuration variants | Ball-node / source-context variants showed signal but missed promotion gates | Retired exact variants |
| Direct interaction / role-specific heads | Completed after numerical-resume recovery; no candidate beat matched control | Retired |
| Single-target / defender-focus / route representation | All three completed Fold 0 and failed locked promotion gates | Retired |
| Full five-fold winner-parent TTA audit | TTA improved the base family by up to **0.001604**, but only **~0.000034** inside multisplit-20 | Retired as ensemble upgrade |

### What transferred

- **Split/model diversity remains the strongest proven mechanism.** The 20-model multisplit ensemble still owns the strongest local OOF and private submission evidence.
- **TTA can improve a single family but is mostly redundant inside a diverse ensemble.** The full five-fold audit made that distinction explicit.
- **Cross-fold confirmation matters.** Several one-fold signals disappeared under confirmation, so the project does not promote discovery-fold wins.
- **Negative results are first-class evidence.** Retired branches are recorded so compute is not repeatedly spent on the same question.
- **Performance engineering is workload-specific.** End-to-end loader and inference benchmarks are measured separately from predictive quality.

## Current highest-value research direction

The next prepared experiment moves away from fitted-parent adapters and trains **fresh source-faithful feature configurations from scratch** under the preserved successful training contract. It is explicitly **prepared but unmeasured** until a real AWS run completes.

This is the closest remaining competition-data-only reproduction of the first-place system's documented strength: many independently trained feature configurations combined across repeated grouped-CV splits.

## Public reproducibility boundary

This repository is intentionally **semi-reproducible**. It publishes aggregate metrics, validation rules, selected protocols, decision history, public artifacts, and privacy-safe implementation patterns.

It does **not** publish raw competition data, fitted weights, private checkpoint locations, credentials, complete private experiment runners, or unreleased competition-specific transforms that provide a competitive edge.

## Evidence map

- [Post-PR40 frontier research](docs/FRONTIER_RESEARCH_POST_PR40.md)
- [Machine-readable post-PR40 snapshot](docs/results/post_pr40_frontier_research.json)
- [Current research status](docs/CURRENT_RESEARCH_STATUS.md)
- [Competition-facing score lineage](docs/results/frontier_submission.json)
- [Earlier post-PR39 review](docs/FRONTIER_RESEARCH_POST_PR39.md)
- [Recent reproduced-model review](research/RECENT_MODELS.ipynb)

## Validation and limitations

The official coordinate metric is sqrt(sum(dx^2 + dy^2) / (2N)). Pooled OOF uses row-weighted squared errors, not an unweighted average of fold RMSEs. Development-fold results, local OOF, private leaderboard scores, and late submissions are kept separate. A mechanism is not promoted from one favorable fold; confirmation and uncertainty gates are part of the research protocol.
