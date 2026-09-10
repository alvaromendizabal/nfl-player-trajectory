# Learned motion supervision: fixed experiment

Declared before the first scientific fit, 10 September 2026. Target: 0.46
coordinate RMSE. Feature-completion gate: open. No Kaggle submission.

## Question and attribution

Do velocity and acceleration training targets improve a feature-wise temporal
representation of observed NFL movement? Compare `motion_supervised` with
`coordinate`. Both have identical parameter counts, starting weights, observed
inputs, player attention, ridge residual output, optimizer, augmentation and
training exposure. Both contain the auxiliary heads and calculate their losses;
the control multiplies those terms by zero. All model parameters are trained.

This comparison isolates the *joint motion-supervision package*. It does not
separately identify velocity versus acceleration, the grouped stem versus the
old mixed stem, or an effect of increasing exposure from 40 to 48 epochs. The
old attention and tree are frozen secondary references. Their differences from
the new model must not be described as a pure feature effect.

## Representation and availability

Keep all 30 original observed tracking channels, observed masks, 20 static
features, role/side embeddings and 12 directed player-pair features. Historical
player/role encodings retain their verified earlier-date-only construction.
Three feature-wise convolutions use four learned channels per signal and
dilations 1, 2 and 4, then mix signals into 48 channels. Missing time slots are
masked after every layer. Twenty frames feed the same width-96 player attention
and continuous-time decoder as the reference. These are learned features, not
124 new independent sources of information or a new hand-built candidate count.

The coordinate output remains a correction above the fold-local ridge. Four
auxiliary outputs share its final hidden features: two velocity components and
two acceleration components. Future targets are reconstructed as saved ridge
displacement plus saved residual truth. Targets never enter the forward path.
Differences use 0.1-second consecutive frame IDs within each player. First-frame
velocity may use the observed endpoint only if its age is zero; second-frame
acceleration then uses that endpoint and the first future position. Gaps, stale
endpoints and missing supervision are masked, never filled with invented labels.
No 48-frame truncation, long-play deletion or missing-passer exclusion is used.

Fit one RMS scale per vector family on training targets alone, sharing the scale
between x and y so reflection preserves normalization. Floor each scale at 0.1.
Position loss remains coordinate MSE in yards. Add 0.1 times normalized velocity
MSE and 0.1 times normalized acceleration MSE in the treatment. Both arms use the
same row-count correction for uniform play sampling. No label clipping or
time-dependent position weights are used.

## Population, exposure and decision

Reuse the three verified expanding chronological folds within the original 192
training games: 94/41, 135/28 and 163/29 train/evaluation games. Evaluation has
202,361 requests in 98 distinct games. These folds have already been used for
research and are not independent confirmatory test data. The previous development
and reserved-game scores are not used for selection in this study.

Six fits: two arms per fold. Forty-eight passes, batch 128 plays, seed 2026,
AdamW learning rate 0.001 cosine-decaying to 0.00001, weight decay 0.01,
gradient norm cap 1, EMA 0.98, horizontal reflection probability 0.5. Use two CPU
threads and deterministic operations. Per-arm training wall-time cap: 1,200 s;
an incomplete fit cannot be reported as a completed result. Fixed final EMA;
no validation-selected checkpoint, learning-rate search or best-arm selection.
Small synthetic recovery tests are engineering checks, not scientific fits.

The sole primary contrast is motion-supervised minus coordinate-only RMSE.
The promotion gate requires improvement in all three folds, at least 1% pooled
RMSE reduction, and a negative upper bound of the paired game-bootstrap 95%
interval (10,000 resamples). The confidence interval does not correct for all
previous adaptive use of these folds. Equal 50/50 tree blends, horizon slices
and comparison with the previous attention are descriptive secondary outcomes.
Even a passed gate does not establish 0.46, independent generalization or feature
completion; successful components still require decomposition and later testing.

## Recovery and provenance

Before training, freeze hashes of this protocol, numerical sources, locked
runtime, parent models and verified sample caches. Refuse changed signatures.
Save model, EMA, optimizer, epoch/batch pointer, RNG state and curve atomically
after every epoch and on caught interruption. Completed fits must be hash-checked
and reused. Logs include UTC timestamps and heartbeats. Verify true interrupted
recovery in tests and run the completed pipeline a second time to demonstrate
zero extra epochs and unchanged fitted artifacts. Persist private checkpoints
and publish executed notebook evidence after the quality gate.

## Primary evidence and transfer limits

The [released first-place training notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)
implements feature-wise temporal convolution and auxiliary motion targets. This
experiment borrows the learning hypothesis, not its precomputed normalizers,
excluded plays, fixed-length outputs, likelihood loss or ensemble score. It is
an independent implementation and a smaller compute study, not a reproduction.

[Huang, Cheng and Wang (2025)](https://arxiv.org/html/2503.24272v1)
study separate position, velocity and acceleration streams with feature injection
and motion consistency. Their pedestrian benchmarks and multimodal losses do not
establish NFL coordinate-RMSE performance. Our single-output experiment tests the
simpler transferable auxiliary-supervision mechanism; it does not reproduce their
three-stream architecture, branch selection or self-supervision objective.
