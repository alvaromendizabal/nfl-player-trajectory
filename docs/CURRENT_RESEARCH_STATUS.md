# Current research status — September 2026

## Competitive state

- Strongest recorded late private submission: **0.46487 RMSE**.
- Previous recorded private best: **0.46547 RMSE**.
- Private improvement: **0.00060 RMSE**.
- Published first-place private comparator: **0.46340 RMSE**.
- Remaining comparable gap: **0.00147 RMSE**.
- Strongest completed local system: **0.4631723213 OOF RMSE** over 561,607 rows from a 20-model, four-split-family ensemble.
- Stretch research target: **0.44 RMSE**. Local OOF is not treated as an equivalent private score.

## Latest scored milestone: multisplit-20

The multi-split ensemble converted a strong local result into a stronger private submission: **0.46487 RMSE**, improving the prior seven-model private result by **0.00060**. The score is a late post-competition measurement, not an official competition rank.

The result supports one clear conclusion: **cross-validation split diversity transferred better than several later correction-style feature families**. It does not establish that every winning-solution mechanism has been exhausted.

## Inference engineering milestone

A separate AWS study optimized the fixed 20-model ensemble without changing its weights or predictions.

| Path | Median seconds / play | Numerical result |
|---|---:|---|
| Repeated per-model input preparation | 0.541114 | Reference |
| **Shared input preparation** | **0.113100** | **Bitwise identical** |

Measured speedup: **4.784×** on the declared timing sample.

Parity was checked over **96 plays and 3,723 requested rows**, with maximum coordinate difference **0.0 yards** for the promoted path. More aggressive vectorized variants were rejected when they exceeded the numerical tolerance.

## Research outcomes since the previous GitHub publication

| Milestone | Outcome | Public conclusion |
|---|---|---|
| Multisplit-20 scoring recovery | **Promoted evidence** | New private best: 0.46487 |
| Shared-preparation inference | **Promoted engineering** | 4.784× measured speedup with exact parity |
| Scratch feature-configuration study | Not promoted | Control/evaluation credibility failed |
| Normalization and checkpoint recovery | Retired | Evaluation repair did not restore a competitive control |
| Bounded control continuation | Retired | Additional training still missed the control gate |
| Native moving-average bridge | Retired | Saved averaged state remained outside the credibility gate |
| Source-recipe audit | **Next direction** | Restore the verified training contract before new feature confirmation |

These negative outcomes are retained because they prevent repeated spending on variants that did not establish a trustworthy comparison.

## Engineering lesson from the recovery sequence

The recovery work found that later high-throughput experiments had drifted from the earlier successful training contract across several categories, including optimizer semantics, statistical batch/update behavior, scheduling, precision, clipping, and moving-average lifecycle.

The public repository records that **training-contract drift existed**, but intentionally does not publish every private configuration value or unreleased feature transform. The next experiment restores the verified contract and tests a new feature configuration from that stable baseline.

## Reproducibility boundary

Public GitHub contains:

- aggregate score and latency evidence;
- validation and promotion logic;
- executed aggregate notebooks;
- selected source/protocol snapshots;
- negative-result decisions and limitations.

Private AWS retains:

- competition data;
- fitted weights and large checkpoint archives;
- private object locations and credentials;
- unreleased feature transforms and competition-specific implementation details.

Kaggle is used only for the organizer-required submission surface. Training, validation, research, and experiment tracking remain in AWS.
