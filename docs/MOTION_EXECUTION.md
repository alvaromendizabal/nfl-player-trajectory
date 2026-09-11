# Matched motion execution and research review

Velocity-only auxiliary supervision is a representation hypothesis, not an extra inference-time input. The matched comparison must preserve initialization, all requested coordinates, fold-local preprocessing, deterministic order/reflections, optimizer exposure and final averaged weights. Feature research remains open.

## Primary research reviewed on 11 September 2026

[Huang, Cheng and Wang](https://arxiv.org/html/2503.24272v1), sections 3.3–3.4, motivate motion representations, cross-stream injection and consistency objectives. Their direct position/velocity consistency penalty was unsatisfactory. Our supervised auxiliary head does not reproduce their multi-trajectory pedestrian model; their ADE/FDE results do not establish NFL coordinate-RMSE performance. Isolate velocity first rather than adding every plausible loss. Acceleration remains excluded pending the training-tail audit; consistency would need a separate ablation.

[Dutta, Yurko and Ventura](https://arxiv.org/html/1906.11373v3) motivate uncertain pass-coverage representations. Coverage identification is not post-throw trajectory prediction. The failed fixed soft-affinity ridge treatment rejects that interface, not all learned temporal assignments. Features must use synchronized pre-throw observations, never retrospective coverage labels or future positions.

The first- and second-place Kaggle writeup retrievals exposed titles without article bodies. No newly inspected implementation details or benchmark claims are attributed to them here. Earlier project-attributed findings remain historical in DOMAIN_RESEARCH.md.

## Tested execution contracts

TrainingBatches canonicalizes by game/play identity. Epoch order uses SeedSequence([2026, epoch]); lateral reflections use SeedSequence([2026, epoch, 1]) indexed by canonical play. A persisted optimizer cursor recovers the same order and reflection after interruption. No future label controls this mapping. Cache arrays remain unchanged and the short final batch and long forecasts remain included.

The warmup/cosine schedule is a pure function of cursor and frozen exposure. Global loss denominators come from the training-only plan. Tests cover exact model/EMA/optimizer/RNG/cursor/loss-counter equality across interrupted epoch boundaries for both arms.

The canonical --profile-training mode times first, middle, short-final and largest-row training batches at width 96 and batch size 64 for both arms. Eight diagnostic optimizer steps are engineering work, not scientific fits. It uses no validation labels or validation metric and reuses matching source/runtime-bound evidence.

## Gates still open

Choose equal exposure from measured training throughput before validation prediction. Complete and test the full train/evaluate runner and checkpoint integration; synthetic S3 recovery alone does not certify it. Retain the one-fold >=1% RMSE gain plus negative paired-game confidence-bound continuation gate.

Still investigate velocity/acceleration attribution, long-horizon parameterization, learned changing defender–receiver relations, role-conditioned arrival, route phase, observed-origin augmentation, permitted historical-season alignment and seed/fold replication. Each needs provenance, training-only screening, matched ablations and stability evidence. Feature counts alone prove none of these.
