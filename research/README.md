# Research evidence and source archive

## Evidence hierarchy

The public record separates software correctness, validation evidence, promotion decisions, runtime evidence, and private submission evidence.

1. **Multisplit grouped-game OOF:** `0.4631723213` RMSE over `561,607` scored rows.
2. **Private submission evidence:** `0.46487` RMSE.
3. **Full-OOF diversity evidence:** the accepted four-family system improved on the two-family system by `0.00217835` RMSE with a positive paired-game interval.
4. **Controlled candidate studies:** matched controls, fixed references, and locked promotion gates.
5. **External-data evidence:** provenance, coverage, point-in-time construction, and controlled model studies.
6. **Runtime evidence:** evaluated separately from predictive quality.
7. **Experimental work:** explicitly labeled unpromoted until required validation completes.

## Preserved source, curated presentation

The public archive retains selected source modules, protocols, tests, feature dictionaries, aggregate evidence, and notebooks with private outputs removed.

It excludes raw competition data, raw third-party response archives, credentials, fitted weights, large checkpoints, private object locations, complete private runners, and unreleased active competitive transforms.

The repository is intentionally **semi-reproducible**.

## Scientific conclusions through the current October milestone

**Split/model diversity remains the strongest proven system-level mechanism.**

The full-OOF audit showed the two-to-four-family improvement across all five original folds and under every single-game removal.

**Standalone gains often remain ensemble-correlated.**

Dense temporal supervision, zero dropout, external priors, player-specific history, longer-history models, and retrieval variants each produced useful information, but their exact tested recipes did not add enough robust independent signal for promotion.

**Direct external historical context is useful.**

NFL Next Gen Stats and ESPN historical game context are acquired directly from first-party/neutral public sources and transformed through point-in-time-safe pipelines.

**Point-in-time construction is mandatory.**

Historical 2023 features exclude current-week and future observations.

**Negative experiments are preserved as evidence.**

Exact failed branches are retired rather than repeatedly rescued with small parameter or blend changes.

**The current active program is confirmation of full-model feature-configuration diversity.**

A maturity-matched development-fold study improved the incumbent from **0.45372575 to 0.44849765 RMSE** under a fixed portfolio construction, with positive adjusted whole-game intervals and positive direction under every single-game removal.

The challenger is not promoted. A separate grouped fold with fresh model initialization is the next required gate, followed by full pooled-OOF evaluation if confirmation succeeds.

The exact active feature recipes remain private; the public archive records aggregate evidence, validation contracts, and the decision boundary.

## Public-solution research boundary

Public solution writeups and repositories may be studied for transferable mechanisms, but competitor-prepared training datasets and private artifacts are not used as external training inputs.

Mechanisms are reimplemented inside the project's own validation and provenance contracts.

## Aggregate notebook scope

`RESEARCH_REVIEW.ipynb` is the frozen aggregate review generated from its registered evidence set.

The later October narrative in `docs/` extends the public-safe record through the current split-diversity program. That documentation does not silently rewrite the frozen notebook's historical evidence.

## Publication does not certify

Publication CI does not:

- retrain private models
- replay private predictions
- mutate AWS
- submit externally
- establish that an experimental challenger is promoted

Prepared or running experiments are never presented as measured wins.
