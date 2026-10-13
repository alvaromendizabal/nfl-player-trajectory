# Forecast motion. Account for every trajectory.

**Alvaro Mendizabal · Temporal modeling / sports analytics / inference engineering**

A trajectory-forecasting project connecting player-motion representations, game-aware validation, complete-population evaluation, and measured inference acceleration.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| ML engineering | **Optimize inference without changing predictions.** Measure latency against a fixed ensemble and retain parity checks rather than crediting a speedup from a changed task. | [Code](../src/nfl_trajectory/tree_inference.py) · [Tests](../tests/test_tree_inference.py) · [Evidence](../docs/EMPLOYER_CASE_STUDY.md) |
| Data science | **Report the complete evaluation population.** Keep supported-population and full-population error distinct so unsupported rows do not disappear from the claim. | [Code](../src/nfl_trajectory/final_evaluation.py) · [Tests](../tests/test_final_evaluation.py) · [Evidence](../docs/REPRODUCIBILITY.md) |
| Applied AI | **Represent time before adding model complexity.** Use temporal data and motion contracts to connect historical observations with future-coordinate targets. | [Code](../src/nfl_trajectory/temporal_data.py) · [Tests](../tests/test_temporal_data.py) · [Evidence](../START_HERE.md) |
| ML engineering | **Make the demonstration independently inspectable.** Generate synthetic trajectories, reports, and manifests from a dependency-free public implementation. | [Code](../src/nfl_trajectory/portfolio_demo.py) · [Tests](../tests/test_portfolio_demo.py) · [Evidence](../scripts/validate_portfolio_release.py) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `motion-fixture` | python | [`scripts/run_portfolio_demo.py`](../scripts/run_portfolio_demo.py) |
| `release-evidence` | python | [`scripts/validate_portfolio_release.py`](../scripts/validate_portfolio_release.py) |
| `browser-motion-engine` | node | [`tools/test_public_demo_engine.mjs`](../tools/test_public_demo_engine.mjs) |
| `browser-controls` | node | [`tools/test_public_demo.mjs`](../tools/test_public_demo.mjs) |

## What is actually reproduced

The Python review executes a deterministic synthetic motion demo and its release validator. Optional browser checks test the actual route engine and controls. Recorded OOF and late-evaluation measurements remain separate from the synthetic demonstration.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint covers this curated review surface, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. HTML links follow the current default branch; the receipt is the record of the local bytes inspected. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Raw tracking data, private target rows, fitted ensembles, exact blend recipes, competition outputs, and full private serving state remain outside this review.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
