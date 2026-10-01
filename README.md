# NFL Big Data Bowl 2026 - Prediction

**Player-motion forecasting with controlled sequence modeling, ensemble diversity, GPU engineering, and reproducible AWS research.**

[Frontier research review](docs/FRONTIER_RESEARCH_POST_PR39.md) · [Current research status](docs/CURRENT_RESEARCH_STATUS.md) · [Model card](docs/MODEL_CARD.md) · [Recent model review](research/RECENT_MODELS.ipynb) · [Run and reproduce](START_HERE.md)

Predict selected players' future x/y locations after a pass using observed tracking, organizer-supplied landing location, player roles, and forecast horizon. Competition slug: `nfl-big-data-bowl-2026-prediction`. AWS/SageMaker is the canonical research workspace; Kaggle is used only for the required submission surface.

## Current research snapshot

The strongest recorded late post-competition private submission remains **0.46487 coordinate RMSE**. The published first-place private comparator is **0.46340**, leaving a **0.00147 RMSE gap**. No official competition rank is claimed for late submissions.

The strongest completed local system remains the **20-model, four-split-family equal-weight ensemble** at **0.4631723213 OOF RMSE over 561,607 rows**. Local OOF and private leaderboard measurements are intentionally kept separate.

Since the previous public snapshot, the project completed a broader sequence of controlled neural studies:

| Candidate family | Key evidence | Decision |
|---|---|---|
| Target-specific sparse interaction | Best Fold-0 candidate **0.456751**; fixed blend **0.452395**, +0.001330 vs reference | No promotion |
| Wide/shallow dual-path | Fold 0 fixed blend **0.451572**, +0.002154 and passed; Fold 1 blend **0.474765**, +0.001081 and failed | No promotion after confirmation |
| Dual-path + fixed TTA | Parent standalone improved to **0.462773** on Fold 0 and **0.491123** on Fold 1 | Retained inference mechanism |
| Augmentation fine-tune | Selected epoch 0 on both folds | No promotion |
| Fourier / RBF spatial adapters | Incremental gain over parent+TTA was effectively zero | No promotion |
| Late-horizon / defender residual experts | Incremental gain over parent+TTA was effectively zero | No promotion |
| Muon optimizer | Best candidate **0.471918**; no measurable incremental gain beyond parent+TTA | No promotion |

The important positive result is **complementarity**: the dual-path family produced a strong predeclared Fold-0 blend improvement even though its standalone score was weaker than the reference. The important negative result is **cross-fold instability**: the same family did not reproduce the locked gain on Fold 1, so it was not promoted.

## What changed technically

- **Interaction architecture:** recreated a wide/shallow dual-path family with individual-motion and inter-player paths plus displacement, endpoint, and dense correspondence auxiliaries.
- **Inference robustness:** a fixed original/flip/crop TTA scheme produced repeatable standalone gains and remains in the research toolkit.
- **Medal-solution ablations:** target-specific sparse pooling, spectral pair encodings, late-horizon specialists, defender specialists, augmentation fine-tuning, and Muon were tested under locked gates and retired when they failed to add robust incremental value.
- **GPU engineering:** an end-to-end loader benchmark exposed data starvation that a compute-only benchmark had missed. On the dual-path workload, measured throughput improved from roughly **687 to 2,137 examples/s (~3.11×)**, and later runs reached **~70–74% mean sampled GPU utilization with 100% peaks**.
- **Research discipline:** Fold-0 wins require Fold-1 confirmation and incremental gain beyond the strongest parent+TTA baseline before expensive full-OOF work.

The next prepared branch studies **competition-data-only two-stage / all-player pseudo-supervision**. It is explicitly unmeasured until a real AWS run completes.

## Public reproducibility boundary

This repository is intentionally **semi-reproducible**. It publishes aggregate metrics, validation rules, selected protocols, decision history, public artifacts, and privacy-safe implementation patterns.

It does **not** publish raw competition data, fitted weights, private checkpoint locations, credentials, complete private experiment runners, or unreleased competition-specific transforms that provide a competitive edge.

## Evidence map

- [Post-PR39 frontier research](docs/FRONTIER_RESEARCH_POST_PR39.md)
- [Machine-readable aggregate snapshot](docs/results/post_pr39_frontier_research.json)
- [Current research status](docs/CURRENT_RESEARCH_STATUS.md)
- [Competition-facing score lineage](docs/results/frontier_submission.json)
- [Earlier reproduced-model review](research/RECENT_MODELS.ipynb)

## Validation and limitations

The official coordinate metric is sqrt(sum(dx^2 + dy^2) / (2N)). Pooled OOF uses row-weighted squared errors, not an unweighted average of fold RMSEs. Development-fold results, private leaderboard scores, and late submissions are kept separate. A mechanism is not promoted from one favorable fold; confirmation and uncertainty gates are part of the research protocol.
