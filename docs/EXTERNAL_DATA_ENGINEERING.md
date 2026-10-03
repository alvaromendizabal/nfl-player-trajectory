# External data engineering

## Purpose

This project uses external historical context only when it can be acquired directly from a public first-party or neutral source, transformed reproducibly, and made point-in-time safe for the competition prediction problem.

Competitor-prepared datasets, participant mirrors, and competitor-hosted training artifacts are not part of the external-data pipeline.

## Architecture

The AWS-canonical pipeline separates:

1. **raw acquisition**
2. **immutable provenance receipt**
3. **identity resolution**
4. **point-in-time aggregation**
5. **coverage audit**
6. **model integration**
7. **promotion/retirement decision**

Successful raw acquisition is cached and reused across experiments.

## NFL Next Gen Stats

The project acquired historical passing, receiving, and rushing context directly from NFL public endpoints.

| Metric | Value |
|---|---:|
| NGS rows | 3,920 |
| Historical NGS players | 691 |
| Competition players bridged | 401 |
| Passer prior coverage | 95.74% |
| Targeted-receiver prior coverage | 85.35% |

The identity bridge accepts multiple public identity forms and uses deterministic public-safe fallbacks when direct IDs are not available.

For a 2023 play in week `w`, only NGS observations from weeks `< w` are eligible. Earlier seasons may be used as historical context.

## ESPN historical context

| Asset | Coverage |
|---|---:|
| Weekly scoreboards | 36 / 36 |
| Game summaries | 544 / 544 |
| Competition games mapped | 272 / 272 |
| Play-team mapping | 100% |
| Player-prior coverage | 97.15% |
| Dual team-PBP prior coverage | 100% |

### Game bridge

The first bridge used exact calendar date plus roster overlap. That failed for many prime-time games because ESPN timestamps are UTC and can fall on the next date.

The repaired bridge uses competition season/week, maximum roster/player overlap, and date only as a diagnostic/tiebreak signal. This produced complete competition-game mapping.

## Historical feature families

Public documentation intentionally describes feature groups rather than exact active feature formulas.

Current families include passing history, receiving history, rushing history, prior-game team tendencies, opponent historical context, play-by-play usage tendencies, player-specific historical tendencies, passer/target context, and schedule-level inference-available context.

## Point-in-time rule

For any feature associated with a competition play in week `w`:

- do not use week `w`
- do not use future weeks
- preserve the source timestamp and season/week
- fit learned normalization only on eligible training history
- audit coverage by role/source before expensive fitting

## Model integration

External features are injected conservatively.

When the source representation is widened, new external channels begin at **zero influence**, while the existing source channels retain their previous initialization. This lets training learn whether the new information is useful without perturbing the original path at initialization.

Exact active feature combinations remain private research IP.

## Reproducibility and provenance

Each acquisition milestone records source class, acquisition timestamp, raw response hash, file size, schema summary, row/player counts, mapping coverage, rejected/ambiguous records, and downstream feature-store hash.

Raw responses and private feature stores remain in AWS and are not redistributed through the public repository.

## Operational reliability

External-data pipeline failures are treated as engineering evidence, not model evidence.

Recent examples include live identity-schema drift, UTC date semantics in the first game bridge, and a stale benchmark configuration referencing an obsolete experiment arm.

Each avoidable failure became a deterministic regression test before the next cost-bearing run.

## Public/private boundary

Public: acquisition architecture, coverage metrics, point-in-time rules, aggregate model outcomes, and selected reliability patterns.

Private: raw responses, private cache locations, exact active feature formulas, private weights/checkpoints, and complete experiment runners.
