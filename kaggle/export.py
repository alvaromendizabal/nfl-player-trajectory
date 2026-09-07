"""Export the tested reference predictor into a standalone Kaggle notebook."""

from __future__ import annotations

import argparse
import ast
import json
import pprint
import subprocess
import sys
from pathlib import Path

import nbformat

from nfl_trajectory.runtime import Run, atomic_bytes


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["constant_velocity", "role_ridge"], default="constant_velocity"
    )
    parser.add_argument("--weights", type=Path, default=root / "artifacts/benchmark/model.json")
    args = parser.parse_args()
    with Run(root, "export_kaggle") as run:
        paths = [root / "src/nfl_trajectory/motion.py"]
        if args.model == "role_ridge":
            paths.append(root / "src/nfl_trajectory/models.py")
        definitions: list[ast.stmt] = []
        for path in paths:
            parsed = ast.parse(path.read_text())
            definitions.extend(
                node
                for node in parsed.body
                if not isinstance(node, (ast.Import, ast.ImportFrom))
                and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
            )
        imports = (
            "from __future__ import annotations\nimport importlib\nimport os\n"
            "import sys\nfrom pathlib import Path\n"
        )
        if args.model == "role_ridge":
            imports += "from typing import Any\n"
        imports += "import numpy as np\nimport pandas as pd\n"
        source = imports + ast.unparse(ast.Module(body=definitions, type_ignores=[])) + "\n"
        if args.model == "role_ridge":
            from nfl_trajectory.models import BASIS

            fitted = json.loads(args.weights.read_text())
            if fitted.get("basis") != BASIS or fitted.get("format") != 1:
                raise ValueError("Use weights produced by nfl benchmark.")
            source += (
                "trajectory_predict = predict\nFITTED_MODEL = "
                + pprint.pformat(fitted, width=85, sort_dicts=True)
                + "\n"
            )
        ordered = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "I,UP",
                "--fix",
                "--stdin-filename",
                "model.py",
                "-",
            ],
            input=source,
            text=True,
            capture_output=True,
            check=True,
        )
        source = ordered.stdout
        setup = """COMPETITION_PATH = Path('/kaggle/input/nfl-big-data-bowl-2026-prediction')
if not COMPETITION_PATH.exists():
    candidates = list(Path('/kaggle/input').glob('competitions/nfl-big-data-bowl-2026-prediction'))
    if len(candidates) != 1:
        raise FileNotFoundError('Attach the NFL Big Data Bowl 2026 Prediction competition.')
    COMPETITION_PATH = candidates[0]
sys.path.insert(0, str(COMPETITION_PATH))
inference_module = importlib.import_module('kaggle_evaluation.nfl_inference_server')
"""
        interface = """def predict(test, test_input):
    # Preserve the exact incoming target row order. Only public, observed input is used.
    target = test.to_pandas() if hasattr(test, 'to_pandas') else test
    observed = test_input.to_pandas() if hasattr(test_input, 'to_pandas') else test_input
    predictions = constant_velocity(observed, target[KEYS])
    return predictions[['x', 'y']]

server = inference_module.NFLInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    server.serve()
else:
    server.run_local_gateway((str(COMPETITION_PATH),))
"""
        if args.model == "role_ridge":
            interface = interface.replace(
                "constant_velocity(observed, target[KEYS])",
                "trajectory_predict(observed, target[KEYS], 'role_ridge', FITTED_MODEL)",
            )
        description = (
            "Role-conditioned ridge model fitted on the frozen training games. "
            "The learned weights are embedded; inference requires no external model download. "
            if args.model == "role_ridge"
            else "Constant velocity reference using only observed pre-throw coordinates. "
        )
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "# NFL trajectory reference submission\n\n"
                    + description
                    + "Enable the official competition input, use CPU, and disable internet. "
                    "Run the local gateway before attempting a late submission. "
                    "No Kaggle score has been obtained.\n\n"
                    "Interface: [official organizer example]"
                    "(https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)."
                ),
                nbformat.v4.new_code_cell(source),
                nbformat.v4.new_code_cell(setup),
                nbformat.v4.new_code_cell(interface),
            ],
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
        )
        nbformat.validate(notebook)
        destination = root / "artifacts/kaggle/submission.ipynb"
        formatted = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--stdin-filename", "submission.ipynb", "-"],
            input=nbformat.writes(notebook),
            text=True,
            capture_output=True,
            check=True,
        )
        nbformat.validate(nbformat.reads(formatted.stdout, as_version=4))
        atomic_bytes(destination, formatted.stdout.encode())
        run.event(
            "notebook_exported",
            path=str(destination.relative_to(root)),
            model=args.model,
            official_gateway_status="not_run",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
