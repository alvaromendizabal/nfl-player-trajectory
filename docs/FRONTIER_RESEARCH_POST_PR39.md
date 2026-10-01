# Post-PR39 frontier research review

This document summarizes controlled NFL Big Data Bowl 2026 research completed after PR #39. It preserves experiment logic, aggregate metrics, and engineering conclusions while excluding raw competition data, fitted weights, private checkpoint locations, private runners, and unreleased transforms.

## Competitive context

- Strongest recorded late private submission: **0.46487 RMSE**
- Published first-place private comparator: **0.46340 RMSE**
- Strongest completed local system: **0.4631723213 OOF RMSE** over 561,607 rows
- Stretch research target: **0.44 RMSE**

Private leaderboard scores and local development-fold metrics are not treated as interchangeable.

## Experiment sequence

| Family | Candidate evidence | Fixed blend evidence | Outcome |
|---|---|---|---|
| Target-specific sparse interaction | Best Fold-0 candidate 0.456751 | 0.452395, +0.001330 vs reference | No promotion |
| Wide/shallow dual-path | Fold 0: 0.465845; Fold 1: 0.493347 | Fold 0: 0.451572 (+0.002154); Fold 1: 0.474765 (+0.001081) | Fold 0 passed; Fold 1 failed confirmation |
| Fixed TTA on dual-path parent | Fold 0: 0.462773; Fold 1: 0.491123 | Retained as parent baseline | Retained mechanism |
| Augmentation fine-tune + TTA | Fold 0: 0.462706; Fold 1: 0.491058 | Fold 0: 0.451607; Fold 1: 0.474849 | Selected epoch 0; no incremental value |
| Fourier spatial adapter | Parent-equivalent | 0.451556 | No promotion |
| RBF distance adapter | Parent-equivalent | 0.451556 | No promotion |
| Late-horizon residual expert | Parent-equivalent | 0.451556 | No promotion |
| Late-horizon defender expert | Parent-equivalent | 0.451556 | No promotion |
| Muon optimizer | Best candidate 0.471918 | 0.451556; effectively zero incremental gain vs parent+TTA | No promotion |

## What transferred

### Dual-path complementarity

The wide/shallow dual-path model was weaker standalone than the fixed multisplit reference, yet its errors were complementary enough to produce the strongest locked Fold-0 blend improvement in this research sequence. The required confirmation fold then failed, showing why the project does not promote one-fold discoveries.

### Fixed TTA

A fixed original / horizontal-flip / cropped-input blend improved the selected dual-path parent on both tested folds:

- Fold 0: 0.465845 → **0.462773**
- Fold 1: 0.493347 → **0.491123**

Fine-tuning with related augmentation did not add further robust value.

### GPU engineering

A compute-only loader benchmark initially overstated throughput because it started timing after batch retrieval. Replacing it with an end-to-end benchmark exposed data starvation.

- 0 workers: about **687 examples/s**
- selected worker plan: about **2,137 examples/s**
- improvement: about **3.11×**
- later bounded runs: **~70–74% mean sampled GPU utilization**, **100% peaks**

This is separate from the previously published **4.784× inference acceleration** for the fixed 20-model ensemble.

## What did not transfer

Target-specific sparse interaction, augmentation fine-tuning, Fourier/RBF adapters, late-horizon and defender-only residual specialists, and Muon all failed predeclared incremental gates. These results are retained rather than hidden; each closes a branch and reduces repeated compute.

## Leading-solution coverage and next gap

The project now covers substantial pieces of the strongest public approaches: compact temporal/player representation, grouped folds, EMA, augmentation, multi-split diversity, wide/shallow dual-path interaction with auxiliary supervision, target-specific interaction, delta/RoPE and Muon as separate axes, and fixed TTA.

The largest remaining system-level gaps are broader **feature-configuration diversity**, larger **split/model diversity**, and **two-stage/all-player supervision**.

## Next prepared study

The next prepared branch is a **competition-data-only two-stage/all-player pseudo-supervision** experiment. A selected project-owned dual-path model acts as teacher; plain and uncertainty-weighted pseudo-label consistency variants are predeclared for valid unscored players. The branch remains unmeasured until AWS execution completes.

## Reproducibility boundary

Public GitHub contains aggregate metrics, validation logic, selected protocols, decision history, and privacy-safe implementation patterns. The private AWS workspace retains competition data, fitted states, large checkpoints, complete experiment runners, object locations, and unreleased transforms.
