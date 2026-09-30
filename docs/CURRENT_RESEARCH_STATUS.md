# Current research status — September 2026

## Competitive state

- Strongest recorded late private submission: **0.46487 RMSE**.
- Published first-place private comparator: **0.46340 RMSE**.
- Remaining comparable gap: **0.00147 RMSE**.
- Strongest completed local system: **0.4631723213 OOF RMSE** over 561,607 rows from a 20-model, four-split-family ensemble.
- Stretch research target: **0.44 RMSE**. No local result is presented as an equivalent private score.

## Post-PR38 controlled neural studies

The source-recipe recovery made it possible to compare new candidates against a credible archived anchor without extending the earlier drifted control.

| Evidence | Expanded motion | Future-delta decoder |
|---|---:|---:|
| Standalone Fold-0 RMSE | 0.460633 | **0.459976** |
| Multisplit reference RMSE | 0.453726 | 0.453726 |
| Fixed 80/20 blend RMSE | **0.452585** | 0.452981 |
| Gain vs reference | **+0.001140** | +0.000745 |
| Locked minimum gain | +0.001500 | +0.001500 |
| Paired-game interval | -0.000407 to +0.002498 | -0.000682 to +0.002049 |
| Decision | **NO_PROMOTION** | **NO_PROMOTION** |

The expanded-motion candidate also improved the equivalent archived-anchor blend by **0.001059 RMSE**, while the future-delta candidate improved it by **0.000663 RMSE**. In both cases the paired-game interval crossed zero, so neither family advanced to Fold 1.

These are useful negative results: both candidates contained signal, but the evidence was not strong enough for promotion. No post-hoc fold selection, blend-weight search, or threshold relaxation followed the result.

## GPU and runtime evidence

The current AWS training environment uses a single NVIDIA L4. A complete forward/backward/optimizer benchmark on the future-delta architecture measured:

| Loader workers | Median full step |
|---:|---:|
| 0 | 0.11018 s |
| **2** | **0.04855 s** |
| 4 | 0.04862 s |
| 12 | 0.05090 s |

Two workers were selected. The result shows that the input pipeline was materially improved without changing the statistical training recipe; additional host workers did not improve the measured step.

This GPU benchmark is separate from the previously published **4.784× inference acceleration** for the fixed 20-model ensemble.

## Scientific interpretation

1. **Split diversity remains the strongest proven new ensemble axis.** It is the only recent mechanism that transferred to a stronger private score.
2. **Small representation changes can add complementarity without earning promotion.** The expanded-motion model improved the fixed blend but missed the predeclared gain and uncertainty gates.
3. **A different decoder alone was insufficient.** Future-conditioned incremental motion improved standalone Fold-0 RMSE relative to the expanded-motion candidate, but contributed less to the fixed ensemble.
4. **The next studies should change interaction structure, not merely append more kinematics.** A target-specific sparse-interaction family is prepared for AWS execution and has no published accuracy result yet.
5. **Negative experiments remain first-class evidence.** Retiring a family after a locked no-promotion decision prevents repeated compute spend and selective reporting.

## Public/private boundary

Public GitHub contains aggregate score and latency evidence, validation logic, executed aggregate notebooks, selected protocols, and high-level research decisions.

Private AWS retains competition data, fitted states, large checkpoints, object locations, private runners, and unreleased feature/interaction transforms. Kaggle remains the organizer-required submission surface only.
