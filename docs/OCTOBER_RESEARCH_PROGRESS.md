# October 2026 research progress

This document summarizes the controlled research completed after the previous public milestone. It is intentionally public-safe: it records mechanisms, aggregate metrics, validation outcomes, and engineering lessons without publishing private weights, full runners, raw competition data, private cloud paths, or active competitive feature formulas.

## Executive summary

The project moved through three distinct phases:

1. **representation and objective exploration**
2. **architecture / optimization / physics studies**
3. **direct-source external historical context**

The strongest recurring lesson is that **component quality and ensemble value are different objectives**. Several candidates improved the source-family model while remaining too correlated with the 20-model incumbent ensemble.

The current research therefore prioritizes **new information sources and residual complementarity** rather than repeated parameter tuning.

## Controlled experiment ledger

| Family | Public-safe result | System decision |
|---|---|---|
| Fresh independent source configurations | One arm survived early screening but failed the midpoint ensemble gate | Retired exact branch |
| Temporal Huber / horizon-weighted objectives | No promotion under locked gates | Retired |
| Dense temporal correspondence | ~0.00097 standalone RMSE improvement vs matched source checkpoint; no fixed-ensemble gain | Retired exact recipe |
| Zero-dropout source family | ~0.00183 standalone RMSE improvement; ensemble uncertainty crossed zero | Retired exact recipe |
| Wide/shallow source transfer | Did not outperform matched source control | Retired |
| Layer-wise optimizer transfer | Weak at early screen | Retired |
| Explicit physics integration | Survived screen; failed midpoint ensemble gate | Retired |
| Physics/set residual heads | Rejected at early screen | Retired |
| Direct NFL NGS static priors | Strong standalone gain; exact injection recipe remained ensemble-correlated | Retired exact recipe |
| Direct ESPN team/PBP context | Positive fixed-blend signal; bootstrap interval still crossed zero | Retired exact recipe |
| Player-specific prior-game PBP | Packaged and tested | Current active milestone |

## Standalone-versus-ensemble examples

### Dense temporal supervision

- source-family checkpoint RMSE: `0.463555`
- candidate RMSE: `0.462590`
- standalone improvement: ~`0.000966`
- fixed ensemble: did not improve the incumbent

### Zero dropout

- source-family checkpoint RMSE: `0.463555`
- candidate RMSE: `0.461730`
- standalone improvement: ~`0.001825`
- fixed-blend gain: ~`0.000297`
- uncertainty crossed zero

### Team-level historical PBP

- candidate Fold-0 RMSE: `0.463441`
- source checkpoint RMSE: `0.463555`
- fixed-blend gain: ~`0.000793`
- uncertainty crossed zero

The external-context result is smaller standalone than zero dropout, but more promising from an ensemble-complementarity perspective. That is why the current program continues toward finer player-level context.

## Direct-source external data program

### NFL Next Gen Stats

Direct acquisition produced:

- 3,920 rows
- 691 historical players
- 401 competition-player matches
- 95.74% passer coverage
- 85.35% targeted-receiver coverage

The live API identity schema differed from the first fixture, so the bridge was hardened around multiple identity forms and deterministic name/position fallbacks.

### ESPN

Direct acquisition produced:

- 36 weekly scoreboards
- 544 game summaries
- 272/272 competition games mapped
- 100% play-team mapping
- 97.15% player-prior coverage
- 100% dual team-PBP prior coverage

A first bridge used exact calendar date and missed prime-time games because UTC dates can shift relative to the local game date. The repaired bridge uses competition week + roster overlap, with date as a diagnostic rather than the primary key.

## Point-in-time leakage control

For a 2023 competition play in week `w`:

- 2022 historical data are eligible
- 2023 observations must come from weeks `< w`
- current-week and future observations are excluded
- scaling/reference statistics are fitted from historical training periods rather than future validation information

This rule applies to team, player, NGS, and PBP-derived priors.

## Systems engineering lessons

Recent avoidable execution failures became permanent regression tests, including a live NGS identity-schema mismatch, a stale benchmark arm name after an experiment-family change, and multi-return-file orchestration ambiguity.

The current operator contract is:

**one `.pyz` → one terminal command → one outer return ZIP**

Child scientific evidence is nested inside the outer bundle for provenance.

## Performance engineering

Recent L4 source-family training typically selected zero data-loader workers after representative end-to-end benchmarking. The project does not assume that setting universally; each materially changed workload re-benchmarks `0/4/8`.

The fixed 20-model inference stack previously achieved **4.784×** speedup through shared preparation with exact prediction parity on its benchmark.

## Current research question

The active milestone asks whether **player-specific historical tendencies** add more independent signal than team-level context.

Two controlled arms test each player's own prior-week usage/tendency context and the same player-level context plus passer/target historical context broadcast across the play.

The experiment is prepared and tested but remains unmeasured until AWS execution completes.

## Promotion rule

A candidate must survive early screening, fixed-blend comparison, paired-game bootstrap uncertainty, grouped-fold confirmation, and full OOF evaluation before system promotion.

This intentionally makes the public research record conservative.
