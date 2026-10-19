"""Curate the final public trajectory delivery from the reviewed source tree."""
from pathlib import Path
import ast,json,subprocess
ROOT=Path.cwd()
subprocess.run(['git','diff','--exit-code','06f9888568de2b0681690462593b85842a4bc324','HEAD','--','.',':(exclude).github/workflows',':(exclude).cleanup'],check=True)
def write(name,text):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def selected_source(name,functions,header):
 p=ROOT/name;old=p.read_text();lines=old.splitlines(keepends=True);nodes=ast.parse(old).body
 return header+'\n\n'+'\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]).rstrip() for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in functions)+'\n'
score=selected_source('src/nfl_trajectory/final_evaluation.py',{'score_predictions'},'''"""Key-aligned scoring primitives; labels enter only after predictions exist."""
from __future__ import annotations
import numpy as np
import pandas as pd
from nfl_trajectory.motion import ENTITY, KEYS, require_keys''')
metrics=selected_source('src/nfl_trajectory/benchmark.py',{'error_metrics','bootstrap_scores'},'')
fixture=selected_source('src/nfl_trajectory/report.py',{'synthetic_play'},'''"""Small authored tracking fixture for independent numerical contract tests."""
from __future__ import annotations
import pandas as pd''')
keep_source={'__init__.py','motion.py','models.py','runtime.py','data.py','portfolio_demo.py','report.py','final_evaluation.py','cli.py'}
keep_tests={'test_models.py','test_motion.py','test_data.py','test_runtime.py','test_portfolio_demo.py','test_portfolio_release.py','test_final_evaluation.py'}
keep_docs={'DATA_CARD.md','MODEL_CARD.md','EMPLOYER_CASE_STUDY.md','RESULTS.md','REPRODUCIBILITY.md','PROJECT_CLOSEOUT.md','CURRENT_RESEARCH_STATUS.md','SOURCES.md'}
keep_results={'project_closeout.json','frontier_submission.json','model.json'}
keep_scripts={'run_portfolio_demo.py','validate_portfolio_release.py','quality.py'}
tracked=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
removed=[]
for name in filter(None,tracked):
 p=Path(name);parts=p.parts
 retire=parts[0] in {'research','notebooks','kaggle','configs'}
 retire |= name.startswith('src/nfl_trajectory/') and (p.name not in keep_source or len(parts)!=3)
 retire |= parts[0]=='tests' and p.name not in keep_tests
 retire |= parts[0]=='scripts' and p.name not in keep_scripts
 retire |= name.startswith('docs/') and len(parts)==2 and p.name not in keep_docs
 retire |= name.startswith('docs/archive/')
 retire |= name.startswith('docs/results/') and p.name not in keep_results
 if retire and (ROOT/name).is_file():(ROOT/name).unlink();removed.append(name)
write('src/nfl_trajectory/final_evaluation.py',score+metrics)
write('src/nfl_trajectory/report.py',fixture)
write('src/nfl_trajectory/cli.py','''"""Public data audit and deterministic reference, never private training or submission."""
from __future__ import annotations
import argparse
from pathlib import Path
from nfl_trajectory.data import audit, download
from nfl_trajectory.portfolio_demo import run_demo
from nfl_trajectory.runtime import Run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "audit", "download"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio"))
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "src/nfl_trajectory").is_dir():
        parser.error("Run from the repository root.")
    if args.command == "demo":
        run_demo(args.output)
    else:
        with Run(root, args.command) as run:
            {"audit": audit, "download": download}[args.command](root, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')
write('scripts/quality.py','''"""Run all maintained package tests and the public evaluation, without cloud writes."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "src", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-S", "scripts/run_portfolio_demo.py", "--output", "artifacts/portfolio"],
        [sys.executable, "-S", "scripts/validate_portfolio_release.py",
         "--demo-output", "artifacts/portfolio"],
    ]
    for command in commands:
        print("CHECK", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True, timeout=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')
write('tests/test_final_evaluation.py','''"""Retained final scoring contracts, independent of retired fitting campaigns."""
import math
import numpy as np
import pandas as pd
import pytest
from nfl_trajectory.final_evaluation import bootstrap_scores, error_metrics, score_predictions
from nfl_trajectory.motion import KEYS


def fixture():
    return pd.DataFrame([[1, 1, 1, 1, 0.0, 0.0], [2, 1, 1, 1, 2.0, 3.0],
                         [2, 1, 1, 2, 3.0, 4.0]], columns=KEYS + ["x", "y"])


def test_outer_join_and_coordinate_denominator():
    truth = fixture()
    keys = truth[KEYS].iloc[::-1].reset_index(drop=True)
    values = keys.merge(truth, on=KEYS)[["x", "y"]].to_numpy() + [3, 4]
    errors = score_predictions(truth, keys, values)
    pd.testing.assert_frame_equal(errors[KEYS], keys)
    assert error_metrics(errors)["coordinate_rmse_yards"] == pytest.approx(math.sqrt(12.5))
    for bad in (truth.iloc[1:], pd.concat([truth, truth.iloc[:1]])):
        with pytest.raises(ValueError):
            score_predictions(bad, keys, values)
    with pytest.raises(ValueError):
        score_predictions(truth, keys, np.full((3, 2), np.nan))


def test_whole_game_bootstrap_pools_sse_and_preserves_replication():
    errors = fixture()[KEYS].assign(dx=[10.0, 0.0, 0.0], dy=0.0)
    scores = bootstrap_scores(errors, repeats=200, seed=7)
    np.testing.assert_array_equal(scores, bootstrap_scores(errors, repeats=200, seed=7))
    allowed = (0.0, math.sqrt(100/6), math.sqrt(100/2))
    assert all(any(math.isclose(x, y, abs_tol=1e-12) for y in allowed) for x in scores)
    with pytest.raises(ValueError):
        bootstrap_scores(errors.iloc[:0])
''')
p=ROOT/'pyproject.toml';s=p.read_text()
for dep in ['plotly==7.0.0','boto3==1.43.89','ipykernel==7.3.0','nbformat==5.11.1','matplotlib==3.10.8']:s=s.replace('  "'+dep+'",\n','')
a=s.index('[tool.ruff.lint.per-file-ignores]');b=s.index('[tool.mypy]',a);s=s[:a]+s[b:];p.write_text(s)
write('README.md','''# NFL player trajectory forecasting

**Temporal modeling · game-aware evaluation · inference engineering**

![NFL trajectory project](docs/assets/route-lab-hero.svg)

A completed trajectory-forecasting system with a **20-model neural ensemble**,
audited game-held-out evaluation, and prediction-preserving inference acceleration.
The public repository exposes the data, evaluation and recovery contracts, a small
trainable reference control, and a local-first interactive demonstration.

| Recorded result | Scope |
|---|---|
| **0.46468 private coordinate RMSE** | Final late submission 56928100; no official placement claimed |
| **0.4629258204 supported OOF** | 561,607 rows across 272 games |
| **0.5242026277 complete OOF** | All 562,936 rows, including difficult unsupported cases |
| **4.784× inference acceleration** | Fixed 20-model workload, 96 plays, 3,723 rows; exact prediction parity |

[Case study](docs/EMPLOYER_CASE_STUDY.md) · [Results](docs/RESULTS.md) ·
[Model card](docs/MODEL_CARD.md) · [Run and verify](docs/REPRODUCIBILITY.md) ·
[Route Lab](https://alvaro-nfl-route-lab.tartmacaw2.chatgpt.site)

## Inspect the work

The evaluation distinguishes missing-population error from model quality, aligns
all four forecast identifiers, and pools squared errors before taking a square root.
The native reference tests exercise fitted numerical controls, geometric invariance,
missing-role fallback, invalid input rejection, and recoverable artifact writes.
The browser and Python reports expose every synthetic trajectory and export their
inputs and results for independent recomputation.

```bash
python3 -S review/run.py --output artifacts/review --with-node
python3 -S review/run.py --output artifacts/review --with-node --check
```

Python 3.11+ and Node.js are sufficient for this public review. Open the generated
`index.html` for the implementation map and `forecast-diagnostics.html` for keyed
trajectory and whole-game comparisons. No account, GPU, cloud space or raw NFL data
is needed. The [full package checks](docs/REPRODUCIBILITY.md) additionally exercise
the fitted reference and organizer-data contracts in the committed environment.

## Repository map

| Path | What it contributes |
|---|---|
| `src/nfl_trajectory/models.py` | Small fitted motion reference and observation-only prediction |
| `src/nfl_trajectory/final_evaluation.py` | Exact keyed scoring and whole-game resampling |
| `src/nfl_trajectory/data.py` | Organizer input audit, safe download and schema checks |
| `src/nfl_trajectory/runtime.py` | Atomic artifacts, source signatures and verified stage reuse |
| `public-demo/` | Interactive observed-motion forecasts on authored examples |
| `review/` and `tools/` | Source-bound walkthrough, independent CSV audit and browser tests |
| `docs/results/` | Final-system aggregates and the explicitly separate public reference |

## Publication boundary

The public control is **not the final neural ensemble**. The final neural weights,
exact ensemble recipe, private forecast arrays and competition data are not
distributed. Synthetic reference scores do not estimate the private score. Original
measurements retain their population and late-evaluation qualifications.
[Source attribution](docs/SOURCES.md) distinguishes adapted architecture references
from this project's integration, evaluation, optimization and delivery work.
''')
write('START_HERE.md','''# Technical review

Start with the [case study](docs/EMPLOYER_CASE_STUDY.md), then
[reproduce the public review](docs/REPRODUCIBILITY.md).

The maintained code covers organizer-data auditing, observation-only reference
forecasting, fitted reference controls, keyed coordinate scoring, whole-game
resampling, and atomic recovery. The final neural ensemble is documented through
its [original aggregate evidence](docs/results/project_closeout.json); its private
weights and exact recipe are not shipped.

| Review question | Implementation and tests |
|---|---|
| Are forecasts aligned to every requested row? | `final_evaluation.py`, `test_final_evaluation.py` |
| Can labels leak into reference prediction? | `models.py`, `test_models.py` |
| Can a stale or partial stage be reused? | `runtime.py`, `test_runtime.py` |
| Are reported errors independently reproducible? | `tools/audit_public_forecasts.py`, `tools/test_forecast_audit.py` |

Run all package tests with `uv sync --frozen --group dev` followed by
`uv run --frozen python scripts/quality.py`. This does not call AWS or submit to Kaggle.
''')
p=ROOT/'docs/RESULTS.md';s=p.read_text();s=s[:s.index('## Historical development records')];s='\n'.join(line for line in s.splitlines() if 'dated research snapshot' not in line)+'\n';p.write_text(s)
p=ROOT/'docs/DATA_CARD.md';s=p.read_text();a=s.index('The complete [feature dictionary]');b=s.index('## Validation and limits',a);s=s[:a]+s[b:];a=s.index('The earlier chronological research track');b=s.index('## Access and reproducibility',a);s=s[:a]+'''Only one labeled season is established by the audited inventory. Calendar-year
diversity in an unlabelled gateway sample checks interface coverage, not accuracy
on a second season. Repeatedly inspected OOF cannot be reused as an untouched test.

'''+s[b:];s=s.replace('Selected historical source and a small fitted baseline are already public;','A small fitted numerical reference is public;');p.write_text(s)
p=ROOT/'docs/EMPLOYER_CASE_STUDY.md';s=p.read_text();s='\n'.join(x for x in s.splitlines() if not x.startswith('| Data engineering |'))+'\n';a=s.index('## Data and reliability work');b=s.index('## Make trajectory errors inspectable',a);s=s[:a]+s[b:];s=s.replace('A small fitted baseline and historical source archives remain public.','A small fitted baseline remains public as a reference control, not as the final neural model.');p.write_text(s)
p=ROOT/'docs/REPRODUCIBILITY.md';s=p.read_text();s=s.replace('Additional research self-tests and their optional dependencies\nare recorded in the existing quality workflow.','All retained native reference and scoring tests run in the quality workflow.');a=s.index('The repository also publishes research code');b=s.index('The private neural ensemble',a);s=s[:a]+'''The package retains a small fitted motion reference at `docs/results/model.json`,
its numerical training and inference implementation, organizer-data contracts,
keyed scoring and atomic stage recovery. Native tests fit reference controls on
invented data; they do not fit or restore the private ensemble.

'''+s[b:];p.write_text(s)
p=ROOT/'docs/MODEL_CARD.md';s=p.read_text().replace('Selected historical implementation and a small fitted baseline are already public.','The public implementation contains a small fitted reference, data contracts, keyed scoring and atomic recovery.').replace('Selected historical source','Selected public source');p.write_text(s)
write('docs/PROJECT_CLOSEOUT.md','''# Completed trajectory-delivery project

The [results](RESULTS.md) identify the final accepted 20-model system, its exact
late-submission scope, and separate supported/full-population OOF measurements.
The original [aggregate receipt](results/project_closeout.json) remains unchanged.

The maintained public package is deliberately focused: organizer data contracts,
a trainable motion reference, keyed evaluation, recoverable artifacts, the
interactive demo, and independent output auditing. Superseded experiment workspaces
and launchers are not part of the maintained release. Git history was not rewritten.

The public checks require no deleted AWS resources. Full private model replay would
require the original licensed data and private fitted artifacts; no public demo is
presented as a replacement for those requirements.

[Run and inspect](REPRODUCIBILITY.md) · [Model card](MODEL_CARD.md) ·
[Sources and attribution](SOURCES.md)
''')
write('docs/CURRENT_RESEARCH_STATUS.md','''# Release status

Completed research project. The final recorded private result is **0.46468 coordinate
RMSE**, late submission **56928100**. No official placement is claimed.

The [final results](RESULTS.md) and [original aggregate receipt](results/project_closeout.json)
are authoritative. Public tests validate the maintained data, reference, evaluation,
recovery and demonstration components, not the withheld neural ensemble.
No private training, automatic submission, or AWS workspace is required by the
[public review](REPRODUCIBILITY.md).
''')
p=ROOT/'docs/SOURCES.md';s=p.read_text();a=s.find('## Earlier feature-engineering methodology');s=s[:a] if a>=0 else s;s=s.replace('The integrated baseline preserves','The privately evaluated neural system integrated').replace('Each adapted technique receives its own project-owned implementation, leakage checks, validation, uncertainty analysis, and promotion/retirement decision.','Adapted techniques were evaluated with project-specific leakage checks, uncertainty analysis and selection controls. The public numerical reference is not that neural system.');p.write_text(s)
p=ROOT/'review/project.json';d=json.loads(p.read_text());d['source_files']=[x for x in d['source_files'] if (ROOT/x).is_file()];d['cards'][0].update(title='Recover stages without accepting stale artifacts.',decision='Bind reusable outputs to their input/source identity, and retain atomic file writes.',boundary='These are public artifact-integrity contracts, not a production availability claim.',implementation='src/nfl_trajectory/runtime.py',tests='tests/test_runtime.py');d['cards'][2].update(title='Keep fitted reference predictions observation-only.',decision='Check keyed forecasting, geometric invariance and global fallback with a small native fitted control.',boundary='The reference control is not the private 20-model ensemble.',implementation='src/nfl_trajectory/models.py',tests='tests/test_models.py');p.write_text(json.dumps(d,indent=2)+'\n')
print('Retired',len(removed),'files; original final-result aggregate bytes retained')
