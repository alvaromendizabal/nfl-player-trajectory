# October 2026 research progress

This document summarizes the controlled research completed after the previous public milestone. It intentionally records mechanisms, aggregate metrics, validation outcomes, engineering decisions, and negative results without publishing raw competition data, fitted private weights, private cloud paths, complete private runners, or active competitive feature formulas.

## Executive summary

The October program progressed through four broad phases:

1. **representation, objective, architecture, and optimization studies**
2. **direct-source external historical context**
3. **player-history, longer-history, and retrieval studies**
4. **full-OOF diversity auditing and fresh split-family expansion**

The strongest recurring lesson is that **component quality and ensemble value are different objectives**.

Several models improved standalone source-family fitting while remaining too correlated with the accepted ensemble. The clearest system-level result remains repeated grouped-split diversity.

## Current headline evidence

- strongest recorded private submission: **0.46487 RMSE**
- strongest completed local ensemble: **0.4631723213 pooled OOF RMSE**
- OOF population: **561,607 rows / 272 games**
- ensemble: **20 models / four grouped split families**
- measured shared-preparation inference speedup: **4.784×**

These evaluation settings are intentionally kept distinct.

## Controlled experiment ledger

| Family | Public-safe result | System decision |
|---|---|---|
| fresh independent source configurations | one arm survived early screening but failed the midpoint ensemble gate | Retired |
| temporal Huber / horizon-weighted objectives | no promotion under locked gates | Retired |
| dense temporal correspondence | ~0.00097 standalone RMSE improvement versus matched source checkpoint; no fixed-ensemble gain | Retired exact recipe |
| zero-dropout source family | ~0.00183 standalone improvement; ensemble uncertainty crossed zero | Retired exact recipe |
| wide/shallow / optimizer / physics transfers | did not earn promotion under matched controls | Retired exact recipes |
| direct NFL NGS priors | strong standalone historical signal; exact integration remained ensemble-correlated | Retired exact recipe |
| direct ESPN team/PBP context | positive fixed-blend signal; uncertainty still crossed zero | Retired exact recipe |
| player-specific PBP histories | repaired data path reached the model; midpoint gain insufficient and uncertain | Retired exact recipe |
| joint temporal/player history | treatments underperformed matched short-history control | Retired |
| source-anchored longer history | parent replay reproduced tightly; longer-history treatments did not beat control | Retired |
| trajectory-memory retrieval | all three retrieval memories completed; gains were small and uncertain | Retired |
| full-OOF diversity audit | robust evidence that four split families outperform two across folds/games | Validated |
| fresh split-4 source-family pilot | 28/35 epochs complete; no promotion yet | Experimental |

## Representative findings

### Dense temporal supervision

- matched source checkpoint RMSE: `0.463555`
- candidate RMSE: `0.462590`
- standalone improvement: ~`0.000966`
- fixed ensemble: no improvement

### Zero dropout

- source checkpoint RMSE: `0.463555`
- candidate RMSE: `0.461730`
- standalone improvement: ~`0.001825`
- fixed-blend gain: ~`0.000297`
- uncertainty crossed zero

### Team-level prior-game PBP

- candidate Fold-0 RMSE: `0.463441`
- source checkpoint RMSE: `0.463555`
- fixed-blend gain: ~`0.000793`
- uncertainty crossed zero

### Player-specific PBP

After repairing the player-history parser and verifying real gradients, the selected candidate reached approximately `0.465762` standalone RMSE at its locked midpoint. Its fixed blend improved the incumbent by approximately `0.000686`, but the paired-game interval crossed zero and the gain missed the predeclared threshold.

The exact configuration was retired; the direct-source historical infrastructure remains reusable.

### Source-anchored longer history

The frozen source parent was reproduced against archived predictions with an RMSE difference of approximately `3.7e-9` and maximum coordinate difference of approximately `3.8e-6` yards.

The short-history control and both longer-history treatments then produced nearly identical fixed-blend results. Neither treatment beat the matched control, so the exact direction was retired.

### Trajectory-memory retrieval

Three fixed retrieval hypotheses—observed-motion, relationship, and forecast-shape memory—were compared against an unmodified parent and a pooled residual control.

The best fixed blend improved the incumbent by only approximately `0.000235`, with uncertainty crossing zero. Residual correlation remained high. The exact retrieval recipes were retired without a neighbor-count or correction-weight rescue.

## Full-OOF diversity audit

The project then returned to the strongest validated system mechanism: repeated grouped-split diversity.

The 20-model prediction bank reproduced:

- pooled OOF RMSE: **0.4631723213**
- rows: **561,607**
- games: **272**

Comparing two versus four split families produced:

- incremental gain: **0.00217835 RMSE**
- paired-game 95% interval: **[0.000629, 0.003861]**
- original folds improved: **5/5**
- positive direction after each single-game removal: **272/272**

This audit evaluated all 15 non-empty family subsets, 75 fold/subset comparisons, residual dependence, leave-one-family-out contributions, and leave-one-game-out sensitivity.

It supported a bounded fresh-split pilot rather than an immediate large-scale model launch.

## Active fresh-split pilot

The current pilot is a newly initialized source-family model on split seed 4 / fold 0.

Public-safe status:

- completed epochs: **28/35**
- best standalone RMSE: **0.46597135**
- exact-population four-family incumbent: **0.45634058**
- fixed 80/20 blend: **0.45655750**
- current blend delta: **−0.00021691**
- remaining prespecified epochs: **7**

The earlier continuation gates passed because the model was fitting credibly. The final promotion gate has not passed, and no expansion claim is made.

## Direct-source external-data program

### NFL Next Gen Stats

Direct acquisition produced:

- 3,920 rows
- 691 historical players
- 401 competition-player matches
- 95.74% passer-prior coverage
- 85.35% targeted-receiver-prior coverage

### ESPN

Direct acquisition produced:

- 36 weekly scoreboards
- 544 game summaries
- 272/272 competition games mapped
- 100% play-team mapping
- 97.15% player-prior coverage
- 100% dual team-PBP coverage

A first game bridge used exact calendar date and missed prime-time games because UTC dates can shift relative to local game date. The corrected bridge uses season/week plus roster overlap, with date retained as a diagnostic/tie-break signal.

## Point-in-time leakage control

For a 2023 competition play in week `w`:

- earlier seasons are historical context
- 2023 observations must come from weeks `< w`
- current-week and future observations are excluded
- learned normalization/reference statistics use eligible training history

The same principle governs team, player, NGS, and PBP-derived historical features.

## Systems-engineering lessons

Avoidable execution failures are retained as engineering evidence and converted into regression tests.

Examples during the October program included:

- live external-source identity-schema drift
- UTC date semantics in a first bridge
- stale benchmark configuration after an experiment-family change
- duplicate telemetry keyword construction
- archive prediction-column mismatch
- precision mismatch between source training and source evaluation

The operator contract remains:

**one artifact → one command → one outer return bundle**

Long-running work is checkpointed and resumable; valid completed stages are not recomputed after downstream failures.

## Current research decision

The project is prioritizing **validated diversity expansion over repeated correlated residual tweaks**.

The fresh split-family pilot must finish its prespecified validation before any broader family expansion. A completed pilot that fails its fixed blend and uncertainty gates stops that exact expansion path; a qualifying pilot advances to the remaining folds of the same family before full-OOF promotion.
