# Current research status — October 2026

## Competitive state

- Strongest recorded late private submission: **0.46487 RMSE**.
- Published first-place private comparator: **0.46340 RMSE**.
- Remaining comparable gap: **0.00147 RMSE**.
- Strongest completed local system: **0.4631723213 OOF RMSE** over 561,607 rows from a 20-model, four-split-family ensemble.
- Stretch research target: **0.44 RMSE**. No local development-fold result is presented as an equivalent private score.

## Controlled research completed since PR #39

| Family | Fold-0 evidence | Fold-1 evidence | Decision |
|---|---|---|---|
| Target-specific sparse interaction | Candidate best **0.456751**; fixed blend **0.452395**; +0.001330 vs reference | Not run | **NO_PROMOTION** |
| Wide/shallow dual-path | Candidate **0.465845**; fixed blend **0.451572**; +0.002154; interval fully positive | Candidate **0.493347**; fixed blend **0.474765**; +0.001081; interval crossed zero | **NO_PROMOTION after confirmation** |
| Dual-path + fixed TTA | Parent standalone **0.462773** | Parent standalone **0.491123** | **Retain TTA mechanism** |
| Augmentation fine-tune + TTA | Selected epoch 0; fixed blend **0.451607** | Selected epoch 0; fixed blend **0.474849** | **NO_PROMOTION** |
| Fourier / RBF adapters | Selected epoch -1; no incremental parent+TTA gain | Not run | **NO_PROMOTION** |
| Late-horizon / defender experts | Selected epoch -1; no incremental parent+TTA gain | Not run | **NO_PROMOTION** |
| Muon optimizer | Best candidate **0.471918**; blend **0.451556**; parent+TTA incremental gain effectively zero | Not run | **NO_PROMOTION** |

The dual-path Fold-0 result is the strongest new complementary signal discovered in this sequence, but the project does not promote a one-fold win. Fold 1 did not reproduce the required gain, so the exact configuration was retired.

## What transferred

### Fixed test-time augmentation

A fixed original / horizontal-flip / cropped-input inference blend improved the selected dual-path parent on both tested folds:

- Fold 0: **0.465845 → 0.462773 RMSE**
- Fold 1: **0.493347 → 0.491123 RMSE**

The augmentation fine-tune itself did not improve beyond the starting parent; both folds selected epoch 0.

### GPU and input-pipeline engineering

An end-to-end loader benchmark corrected an earlier benchmark that timed only GPU compute and excluded data-fetch latency. On the dual-path workload:

- 0 workers: about **687 examples/s**
- selected worker plan: about **2,137 examples/s**
- throughput improvement: about **3.11×**

Later bounded runs commonly reached **~70–74% mean sampled GPU utilization** with **100% peaks**. Worker choice is benchmarked per workload because the optimum varied by model and fold.

This GPU engineering is separate from the previously published **4.784× inference acceleration** for the fixed 20-model ensemble.

## Scientific interpretation

1. **Complementarity matters more than standalone RMSE.**
2. **One-fold wins are not enough.**
3. **TTA transferred better than augmentation fine-tuning.**
4. **Small parent-neutral adapters did not escape the ceiling.**
5. **Muon did not improve this architecture under a controlled optimizer-only comparison.**
6. **The remaining major public-solution gap is system-level diversity and broader supervision.**
7. **Negative experiments remain first-class evidence.**

## Current next direction

The next prepared AWS experiment is a **competition-data-only two-stage / all-player pseudo-supervision** study. The selected project-owned dual-path model acts as teacher; two predeclared student variants test plain and uncertainty-weighted pseudo-label consistency for valid unscored players. It remains **unmeasured** until a real AWS execution completes.

If pseudo-supervision does not transfer, the next major program should emphasize **first-place-style feature/configuration and split diversity**, not additional rescue variants of retired branches.

## Public/private boundary

Public GitHub contains aggregate score and latency evidence, validation logic, selected protocols, decision history, public notebooks, and privacy-safe implementation patterns.

Private AWS retains competition data, fitted states, large checkpoints, object locations, private runners, and unreleased feature/interaction transforms. Kaggle remains the organizer-required submission surface only.
