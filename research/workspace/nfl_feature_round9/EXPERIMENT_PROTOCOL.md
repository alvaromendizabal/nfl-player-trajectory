# Round 9 protocol — diagnose before the next scientific experiment

## Fixed scope

This is a software/representation diagnostic, not an accuracy experiment. Zero model fits, zero optimizer steps, no evaluation input-array loading, no future-coordinate array unpacking, no validation scoring. Whole input-container hashes cover all bytes, including bytes of arrays that are never unpacked. The completed model checkpoint also contains optimizer state; it is deserialized through `torch.load(weights_only=True)` but no optimizer is constructed or stepped.

Choose the first 32 training plays ordered numerically by game/play ID from the sealed Round 8 manifest. Selection uses no scores or labels. This small diagnostic sample is not representative evidence for all roles, games, or seasons.

Verify the uploaded report's immutable summary/completion/runtime hashes, original Round 8 Python source set, protocol digest, dataset digest, and selected input hashes. Verify the final checkpoint pointers, steps, initial hash, sizes and blob hashes. The uploaded aggregate report does not independently contain final weight-blob hashes: blob integrity is checked against existing local checkpoint receipts. Seal their fingerprint in Round 9 preflight and require it unchanged on subsequent stages.

## Frozen-weight probes

Use both saved terminal/history models separately. No `neural_study.protocol`, `train_arm`, old evaluator, or optimizer is invoked. Use original normalization and architecture. Require the pinned CPU numerical environment and exact initial weight hash before loading final weights.

For each model and play, compare predictions with: the opposite terminal/history view; reversed valid pair values within each channel while preserving masks; zero pair values with masks retained; and zeroing each of the base, node, and context segments immediately before the decoder. Also zero each original pair channel separately.

These are stress tests. Some are physically inconsistent or off-distribution. Prediction-change RMS measures dependence, not forecast error, counterfactual football reality, or causal importance. Different branch-zeroing interventions are not comparable controlled feature-importance estimates.

Estimate local per-channel Jacobian energy with two fixed Rademacher signed-output projections (seeds 1103/1109). No labels or loss gradients are needed; model parameters have gradients disabled and input gradients do not accumulate into parameters. These estimates are noisy local sensitivity summaries, not proof of generalization benefit. Compare saved module weights with original initialization descriptively; parameter movement alone does not prove useful learning.

Save per-play aggregate checks and new feature tensors as durable checkpoints. Require exact fresh-process diagnostic replay and unchanged protected files. No automatic train/continue gate exists in this package.

## New representation prototype

Use six goal-aligned pair channels with fixed physical scales:

- Source-goal parallel/lateral peer separation (yards / 20).
- Source-goal parallel/lateral relative velocity (yards per second / 10).
- Peer-minus-source distance to the goal (yards / 20).
- Peer-minus-source goal closing speed (yards per second / 10).

The first four rotate known vectors into the direction from the source player to the organizer-supplied landing point. This is invertible on valid nondegenerate geometry, not new raw information. Role/side partitions identify same-side, opposing-side, targeted-receiver and passer peers. They are overlapping subsets of supplied roles, not inferred coverage assignments or probabilities. Passer is role index 2 in this adapter; other route runner is index 3.

A goal distance below 0.1 yards invalidates axis-dependent quantities. Zero-speed/degenerate positions and missing measurements retain explicit masks. Slots use current observations only; no forward fill or derivatives across gaps. No model consumes these new channels during Round 9. Their accuracy contribution is UNMEASURED.

## Decisions after the audit

If original terminal/history arrays do not differ on supported channels, investigate construction and information availability before fitting. If they differ but weights barely respond, investigate the relational pathway, normalization and decoder reliance before changing model capacity. If the existing models materially use histories but still underperform, stop this exact interface unchanged and propose a separate matched representation experiment.

Any learned integration of goal-frame or role-partitioned relations must first pass raw-adapter parity, training-only recovery and throughput checks. It needs a separately frozen equal-capacity control and valid chronological evaluation. No threshold chosen from this diagnostic can establish RMSE benefit. No extra epochs, altered loss, new augmentation or new validation fold is authorized by the diagnostic code.
