"""Export the tested reference predictor into a standalone Kaggle notebook."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

from nfl_trajectory.runtime import Run, atomic_bytes


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with Run(root, "export_kaggle") as run:
        source = (root / "src/nfl_trajectory/motion.py").read_text()
        source = source.replace(
            "import numpy as np",
            "import importlib\nimport os\nimport sys\nfrom pathlib import Path\n\n"
            "import numpy as np",
            1,
        )
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
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "# NFL trajectory reference submission\n\n"
                    "Constant velocity baseline, using only observed pre-throw coordinates. "
                    "This Phase 0 artifact demonstrates the inference contract. "
                    "It is not a trained competitive model. "
                    "Enable the official competition input, use CPU, and disable internet. "
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
            official_gateway_status="not_run",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
