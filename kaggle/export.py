"""User-invoked, verified export of a standalone predictor; never submits to Kaggle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pprint
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat
import numpy as np

from nfl_trajectory.runtime import Run, atomic_bytes, fingerprint, sha256, stage

MODELS = ("constant_velocity", "role_ridge", "motion_ridge", "landing_ridge", "interaction_ridge")
RESIDUAL_MODELS = MODELS[2:]


def load_parameters(root: Path, model: str, weights: Path,
                    feature_weights: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if model not in MODELS:
        raise ValueError("Choose a supported, explicitly named model.")
    if model == "constant_velocity":
        return None, None
    from nfl_trajectory.models import BASIS

    fitted = json.loads(weights.read_text())
    if fitted.get("basis") != BASIS or fitted.get("format") != 1:
        raise ValueError("Use baseline weights produced by nfl benchmark.")
    for parameters in [fitted["global"], *fitted["roles"].values()]:
        coefficients = np.asarray(parameters["coefficients"], dtype=float)
        if coefficients.shape != (len(BASIS),) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid baseline coefficients.")
    if model == "role_ridge":
        return fitted, None
    from nfl_trajectory.feature_experiment import numerical_sources
    from nfl_trajectory.features import feature_catalog

    bundle = json.loads(feature_weights.read_text())
    receipt = json.loads((root / ".state/features-report.json").read_text())
    summary_path = root / "artifacts/features/summary.json"
    summary = json.loads(summary_path.read_text())
    relative = feature_weights.resolve().relative_to(root.resolve()).as_posix()
    if (bundle.get("format") != 1 or bundle.get("source_sha256") != numerical_sources()
            or bundle.get("baseline_sha256") != sha256(weights)
            or not fitted.get("training_games")
            or bundle.get("training_games") != fitted["training_games"]
            or bundle.get("split_sha256") != fitted.get("split_sha256")
            or receipt.get("status") != "completed"
            or receipt.get("outputs", {}).get(relative) != sha256(feature_weights)
            or receipt.get("outputs", {}).get("artifacts/features/summary.json") != sha256(summary_path)
            or receipt.get("signature") != summary.get("numerical_signature")
            or summary.get("status") != "passed" or summary.get("split") != "validation"
            or summary.get("holdout_evaluation") != "not_run"
            or summary.get("screening_split") != "train"
            or summary.get("source_sha256") != bundle.get("source_sha256")
            or summary.get("baseline_sha256") != bundle.get("baseline_sha256")
            or summary.get("split_sha256") != bundle.get("split_sha256")):
        raise ValueError("Feature export requires completed, matching source/model/split evidence.")
    residual = bundle["models"][model]
    names = residual["features"]
    if not names or len(names) != len(set(names)) or not set(names).issubset(feature_catalog().feature):
        raise ValueError("Invalid residual feature names.")
    shapes = {"mean": (len(names),), "scale": (len(names),),
              "coefficients": (len(names), 2), "intercept": (2,)}
    for name, shape in shapes.items():
        value = np.asarray(residual[name], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid residual coefficient or scaler values.")
    if np.any(np.asarray(residual["scale"]) <= 0):
        raise ValueError("Residual scales must be positive.")
    return fitted, residual


def definitions(path: Path, selected: set[str] | None = None) -> list[ast.stmt]:
    """Embed the actual tested implementation, not a second hand-maintained predictor."""
    result = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if selected is None or name in selected:
            result.append(node)
    return result


def build_notebook(root: Path, model: str, fitted: dict[str, Any] | None,
                   residual: dict[str, Any] | None, signature: str) -> Any:
    """Pure notebook construction; export validates provenance before calling this function."""
    package = root / "src/nfl_trajectory"
    nodes = definitions(package / "motion.py")
    if model != "constant_velocity":
        nodes += definitions(package / "models.py")
    if model in RESIDUAL_MODELS:
        nodes += definitions(package / "features.py")
        nodes += definitions(package / "feature_experiment.py",
                             {"BATCH_ROWS", "target_state", "predict_residual"})
    nodes += definitions(package / "runtime.py",
                         {"Run", "stage", "sha256", "atomic_bytes", "atomic_json"})
    source = (
        "from __future__ import annotations\n"
        "import hashlib\nimport importlib\nimport json\nimport os\nimport sys\n"
        "import tempfile\nimport threading\nimport time\nimport uuid\n"
        "from collections.abc import Callable\nfrom dataclasses import dataclass\n"
        "from datetime import UTC, datetime\nfrom pathlib import Path\nfrom typing import Any\n"
        "import numpy as np\nimport pandas as pd\nfrom filelock import FileLock\n"
        + ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"
        + "EXPORT_SIGNATURE = " + repr(signature) + "\n"
        + "gateway_run: Run | None = None\n"
    )
    if model not in RESIDUAL_MODELS:
        source = source.replace("from dataclasses import dataclass\n", "")
    if model != "constant_velocity":
        source += "trajectory_predict = predict\nFITTED_MODEL = " + pprint.pformat(fitted) + "\n"
    if residual is not None:
        source += "RESIDUAL_MODEL = " + pprint.pformat(residual) + "\n"
    setup = """COMPETITION_PATH = Path('/kaggle/input/nfl-big-data-bowl-2026-prediction')
if not COMPETITION_PATH.exists():
    candidates = list(Path('/kaggle/input').glob('competitions/nfl-big-data-bowl-2026-prediction'))
    if len(candidates) != 1:
        raise FileNotFoundError('Attach the NFL Big Data Bowl 2026 Prediction competition.')
    COMPETITION_PATH = candidates[0]
sys.path.insert(0, str(COMPETITION_PATH))
inference_module = importlib.import_module('kaggle_evaluation.nfl_inference_server')
"""
    expression = ("constant_velocity(observed, target[KEYS])" if model == "constant_velocity"
                  else "trajectory_predict(observed, target[KEYS], 'role_ridge', FITTED_MODEL)")
    correction = (
        "        bank = build_player_features(observed, target[ENTITY])\n"
        "        prediction[['x', 'y']] += predict_residual(bank, target[KEYS], RESIDUAL_MODEL)\n"
        if residual is not None else ""
    )
    interface = """def predict(test, test_input):
    target = test.to_pandas() if hasattr(test, 'to_pandas') else test
    observed = test_input.to_pandas() if hasattr(test_input, 'to_pandas') else test_input
    require_keys(target)

    def compute():
        prediction = MODEL_EXPRESSION
CORRECTION        return prediction[['x', 'y']]

    if gateway_run is None:
        return compute()
    # Cache key includes model/source, environment, observed input, and exact target row order.
    digest = hashlib.sha256(EXPORT_SIGNATURE.encode())
    digest.update((np.__version__ + pd.__version__).encode())
    digest.update(json.dumps(list(observed.columns)).encode())
    digest.update(pd.util.hash_pandas_object(observed, index=False).to_numpy().tobytes())
    digest.update(target[KEYS].to_numpy(dtype=np.int64).tobytes())
    key = digest.hexdigest()
    destination = Path.cwd() / 'artifacts/inference' / (key + '.json')

    def save_prediction():
        prediction = compute()
        atomic_json(destination, prediction.to_numpy(dtype=float).tolist())

    stage(Path.cwd(), 'inference-' + key, key, [destination], save_prediction, gateway_run)
    values = np.asarray(json.loads(destination.read_text()), dtype=float)
    if values.shape != (len(target), 2) or not np.isfinite(values).all():
        raise ValueError('Cached predictions must match the requested rows and be finite.')
    return pd.DataFrame(values, columns=['x', 'y'])

with Run(Path.cwd(), 'kaggle_gateway') as gateway_run:
    server = inference_module.NFLInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        server.serve()
    else:
        server.run_local_gateway((str(COMPETITION_PATH),))
        gateway_run.event('local_gateway_completed', submission_status='not_submitted')
""".replace("MODEL_EXPRESSION", expression).replace("CORRECTION", correction)
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# NFL player trajectory · user-generated inference notebook\n\n"
                f"**Model:** `{model}`. Weights and tested numerical code are embedded. "
                "Attach the official competition input, use CPU, and disable internet. "
                "Run All invokes the organizer's local gateway; **nothing is uploaded or submitted**. "
                "Review the result and use Kaggle's own submission interface yourself. "
                "A local sample run is not a hidden-test score.\n\n"
                "Per-play predictions have content-addressed, checksum-verified checkpoints. "
                "They resume when the working directory is retained; a fresh Kaggle runtime "
                "does not automatically restore a previous session's disk. Save notebook outputs "
                "to preserve them. UTC logs include stages, total elapsed time, and heartbeats.\n\n"
                "[Official organizer interface](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)."
            ),
            nbformat.v4.new_code_cell(source),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_code_cell(interface),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "export": {"model": model, "signature": signature,
                             "official_gateway_status": "not_run", "submission_status": "not_submitted"}},
    )
    for i, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"nfl-export-{i}".encode()).hexdigest()[:8]
    nbformat.validate(notebook)
    return notebook


def export_notebook(root: Path, model: str, weights: Path | None = None,
                    feature_weights: Path | None = None, output: Path | None = None) -> Path:
    root = root.resolve()
    weights = weights or root / "artifacts/benchmark/model.json"
    feature_weights = feature_weights or root / "artifacts/features/model.json"
    destination = (output or root / "artifacts/kaggle/submission.ipynb").resolve()
    if not destination.is_relative_to(root / "artifacts") or destination.suffix != ".ipynb":
        raise ValueError("Export destination must be an .ipynb beneath artifacts/.")
    fitted, residual = load_parameters(root, model, weights, feature_weights)
    inputs = [Path(__file__).resolve()]
    if model != "constant_velocity":
        inputs.append(weights.resolve())
    if model in RESIDUAL_MODELS:
        inputs += [feature_weights.resolve(), root / ".state/features-report.json",
                   root / "artifacts/features/summary.json"]
    signature = fingerprint(root, inputs, {"model": model, "output": str(destination.relative_to(root))})
    with Run(root, "export_kaggle") as run:
        def write() -> None:
            notebook = build_notebook(root, model, fitted, residual, signature)
            ordered = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--select", "I,UP", "--fix",
                 "--stdin-filename", "submission.ipynb", "-"],
                input=nbformat.writes(notebook), text=True, capture_output=True, check=True,
            )
            formatted = subprocess.run(
                [sys.executable, "-m", "ruff", "format", "--stdin-filename", "submission.ipynb", "-"],
                input=ordered.stdout, text=True, capture_output=True, check=True,
            )
            nbformat.validate(nbformat.reads(formatted.stdout, as_version=4))
            atomic_bytes(destination, formatted.stdout.encode())

        key = hashlib.sha256(str(destination.relative_to(root)).encode()).hexdigest()[:12]
        stage(root, "export-kaggle-" + model + "-" + key, signature, [destination], write, run)
        run.event("notebook_exported", path=str(destination.relative_to(root)), model=model,
                  official_gateway_status="not_run", submission_status="not_submitted")
    return destination


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="constant_velocity")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--feature-weights", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export_notebook(root, args.model, args.weights, args.feature_weights, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
