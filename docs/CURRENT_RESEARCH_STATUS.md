# Current research status — October 2026

## Competitive state

- Strongest recorded late private submission: **0.46487 RMSE**
- Published first-place private comparator: **0.46340 RMSE**
- Remaining comparable gap: **0.00147 RMSE**
- Strongest completed local system: **0.4631723213 OOF RMSE** over **561,607 rows**
- Ensemble size: **20 models across four split families**
- Stretch research target: **0.44 RMSE**

Local OOF, development-fold metrics, and private leaderboard scores are intentionally not treated as interchangeable.

## Research completed since PR #40

The post-PR40 sequence deliberately tested several missing mechanisms from strong public approaches while preserving locked promotion gates and the competition-data-only boundary.

| Family | Evidence | Decision |
|---|---|---|
| Competition-only two-stage pseudo-supervision | Controlled variants completed without confirmation-stage promotion | **NO_PROMOTION** |
| ST-GRU / landing-node ST-GRU | Fold-0 standalone RMSE roughly 0.636 / 0.605 versus ~0.454 reference | **NO_PROMOTION** |
| Zero-fit multisplit meta frontier | Best candidate improved full OOF by only ~0.000049 RMSE | **NO_PROMOTION** |
| Frozen-parent feature adapters | Intent / temporal variants produced fixed-blend gains around 0.00011–0.00017 | **NO_PROMOTION** |
| Full-parent coverage/physics fine-tune | Candidate gains over matched control were only a few 1e-5 RMSE | **NO_PROMOTION** |
| Source-native ball/context/window variants | Best Fold-0 blend gain ~0.00118 at an intermediate checkpoint, below final gate | **NO_PROMOTION** |
| Direct interaction / Entmax-ball / role heads | All three completed Fold 0; none beat matched control | **NO_PROMOTION** |
| Single-target / defender-focused / route representation | All three completed Fold 0; fixed-blend gains remained ~0.00012–0.00013 | **NO_PROMOTION** |
| Full five-fold winner-parent TTA | Best base-family gain **0.001604**; multisplit-20 hybrid gain only **~0.000034** | **NO_PROMOTION** |

## Five-fold TTA result

The TTA audit is the clearest recent example of why local family gains and final-system gains must be separated.

### Equal original + horizontal-flip recipe

- base-family RMSE: **0.46814385 → 0.46654003**
- base-family gain: **+0.00160382**
- pooled bootstrap evidence: positive
- multisplit-20 hybrid gain: only **~+0.00003379**
- hybrid uncertainty crossed zero

### Original / flip / deterministic crop recipe

- base-family RMSE: **0.46814385 → 0.46669682**
- base-family gain: **+0.00144703**
- multisplit-20 hybrid RMSE: **0.46311464**
- multisplit-20 hybrid gain: **+0.00005768**
- hybrid 95% interval crossed zero

### Four-way symmetry recipe

- base-family RMSE: **0.47266902**
- multisplit-20 hybrid RMSE: **0.46437391**
- both materially worse

Conclusion: **TTA is useful at the family level but largely redundant with the error diversity already present in multisplit-20.** No 20-model TTA rollout is justified from this evidence.

## What the project has learned

1. **Split/model diversity is the strongest transferred mechanism.**
2. **Complementarity matters more than standalone quality, but it must survive confirmation.**
3. **One-fold wins are insufficient.**
4. **Adapters and fine-tunes around the same fitted parent mostly remained in the same error basin.**
5. **Target-specific, role-specific, and single-target variants did not robustly escape that basin.**
6. **TTA can improve a component model while adding almost nothing to an already diverse ensemble.**
7. **Negative experiments reduce future search cost when they are recorded and retired.**
8. **The remaining major competition-data-only gap is independently trained feature-configuration breadth combined with repeated grouped-CV diversity.**

## Current next direction

The next prepared study trains **fresh full first-place-style models from scratch** under the verified source training contract rather than adapting an existing fitted parent.

Two project-owned feature configurations are screened in a staged protocol. The study is **prepared but unmeasured** until a real AWS execution completes.

If a fresh configuration demonstrates credible standalone quality and leakage-safe ensemble complementarity, it advances to confirmation, full OOF, and then additional split-family scaling.

Historical-data pretraining remains a high-value **blocked** hypothesis because the active project boundary is competition-data-only.

## Engineering state

Recent runners use bounded, resumable execution with:

- collision-resistant run IDs
- checkpointed epoch/stage recovery
- structured JSONL + human-readable logs
- CPU/RAM/GPU/disk telemetry
- measured cost accumulation
- workload-specific worker/batch benchmarks
- fail-closed integrity checks
- notebook/Plotly persistence gates

Important operational lessons are retained: transient AMP overflow should use GradScaler semantics; EMA checkpoint restoration must preserve destination device/dtype; floating-point kernel-path drift should not be mistaken for model drift; and previously slower/invalid CUDA-stream overlap should not be retried unchanged.

## Public/private boundary

Public GitHub contains aggregate metrics, validation logic, selected protocols, decision history, privacy-safe implementation patterns, and machine-readable research snapshots.

Private AWS retains competition data, fitted states, large checkpoints, exact object locations, complete private runners, and unreleased feature transforms. Kaggle remains the organizer-required submission surface only.
