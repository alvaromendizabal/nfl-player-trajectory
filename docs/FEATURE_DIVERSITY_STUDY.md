# Confirmation-stage feature-diversity study

## Executive summary

This study tests whether **full-model feature-configuration diversity** adds useful information to the accepted grouped-split ensemble without changing the core temporal architecture.

The accepted system remains the **20-model multisplit ensemble at 0.4631723213 pooled OOF RMSE across 561,607 rows and 272 games**.

A maturity-matched development-fold study produced a strong challenger result:

- incumbent RMSE: **0.4537257476**
- maturity-matched native-control blend: **0.4547125988**
- four-model feature-portfolio blend: **0.4484976512**
- improvement versus incumbent: **0.0052280964**
- improvement versus native-control blend: **0.0062149476**

The challenger is **not promoted**. It has earned separate-fold confirmation.

## Why this experiment exists

The strongest validated project result to date came from **diversity across grouped split families**.

The next major unanswered question was whether the same core architecture could gain additional ensemble value from **independently trained feature configurations**, a mechanism used heavily in strong public competition systems.

The goal was not to search arbitrary blend weights or add another shallow adapter. The study instead trained full models to a common maturity endpoint and evaluated a fixed portfolio against both:

1. the accepted ensemble; and
2. a maturity-matched native control.

## Experimental design

Public-safe design details:

- one native-control full model
- four full treatment models with complementary feature configurations
- identical core architecture class
- identical model parameter count: **5,005,452 per model**
- common 35-epoch maturity endpoint
- final **EMA** weights only
- fixed portfolio construction
- exact coordinate RMSE
- grouped whole-game uncertainty
- leave-one-game-out robustness
- role and horizon accounting

The exact active feature recipes are intentionally withheld from the public repository. They remain part of the private AWS research state.

No row-level predictions, fitted weights, checkpoints, private cloud paths, or complete private runners are published.

## Development-fold evidence

Population:

- scored rows: **109,144**
- validation games: **55**

| Comparison | RMSE |
|---|---:|
| Existing ensemble | **0.4537257476** |
| Existing ensemble + native-control blend | **0.4547125988** |
| Existing ensemble + feature-portfolio blend | **0.4484976512** |

Effect sizes:

- improvement versus incumbent: **0.0052280964 RMSE**
- improvement versus native-control blend: **0.0062149476 RMSE**

Adjusted whole-game intervals for improvement:

- versus incumbent: **[0.0002693, 0.0099451]**
- versus native-control blend: **[0.0013023, 0.0114739]**

Robustness:

- games improved: **38 / 55**
- improvement direction after single-game removal: **55 / 55 positive**
- displayed horizon bands improved: **all**
- scored player roles improved: **both**

These checks support a confirmation-stage decision, not promotion.

## A research-process correction

An earlier version of the feature-portfolio program was stopped at an epoch-9 screen.

A later audit compared that training age with a historical native learning curve and found a concrete false-negative example: the historical model was also weak at the same age, then improved materially later.

The project did not erase or reinterpret the earlier rejection.

Instead it:

1. preserved the original decision;
2. audited the maturity mismatch;
3. defined a new fixed 35-epoch maturity-matched protocol;
4. added a native control;
5. reran the comparison prospectively.

This is an important systems lesson: **early spending gates must be calibrated to the learning dynamics of the model family they govern**.

## Separate-fold confirmation

The next gate uses a different grouped validation fold:

- validation rows: **112,694**
- validation games: **54**
- fresh initialization: **required**
- discovery-fold trained tensor reuse: **forbidden**
- endpoint: **35 outer epochs**
- candidate portfolio weights: **fixed**
- grouped uncertainty: **required**

A failed confirmation cannot be rescued by pooling the discovery and confirmation folds.

If confirmation passes, the remaining original folds can be completed and evaluated through full pooled OOF before any accepted-system change.

## What this demonstrates to employers

This study is useful beyond the competition context because it shows:

- full-model ablation design rather than one-off parameter tuning
- matched controls
- grouped uncertainty and robustness analysis
- correction of an over-aggressive early-stopping rule
- preservation of negative evidence
- checkpointed long-running GPU workflows
- explicit separation between discovery, confirmation, and promotion
- a public/private reproducibility boundary that protects restricted data and active IP

## Reproducibility boundary

Public:

- aggregate metrics
- study design
- validation population
- fixed decision rules
- robustness summaries
- machine-readable public snapshot

Private:

- exact active feature recipes
- row-level predictions
- private training data
- fitted checkpoints
- complete runners
- cloud object locations

See [Current research status](CURRENT_RESEARCH_STATUS.md), [Model card](MODEL_CARD.md), and [Research system](RESEARCH_SYSTEM.md).
