"""Build a standalone inference notebook on explicit request; never submit to Kaggle."""

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

from nfl_trajectory.runtime import Run, atomic_bytes, sha256, stage

RESIDUAL_MODELS = ("motion_ridge", "landing_ridge", "interaction_ridge")


def definitions(path: Path, names: set[str] | None = None) -> list[ast.stmt]:
    """Reuse tested numerical definitions, not a separately maintained inference implementation."""
    result = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if names is None or getattr(node, "name", None) in names:
            result.append(node)
    return result


def residual_parameters(root: Path, model: str, baseline_path: Path) -> dict[str, Any]:
    from nfl_trajectory.features import feature_catalog
    from nfl_trajectory.research import load_evidence

    summary, _, label = load_evidence(root)
    if label != "Verified local experiment":
        raise ValueError("Use your completed local feature experiment to generate an export.")
    path = root / "artifacts/features/model.json"
    bundle = json.loads(path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if (
        bundle.get("format") != 1
        or bundle.get("baseline_sha256") != sha256(baseline_path)
        or bundle.get("split_sha256") != summary["split_sha256"]
        or bundle.get("training_games") != baseline.get("training_games")
        or bundle.get("source_sha256") != summary.get("source_sha256")
    ):
        raise ValueError("Residual model and baseline provenance disagree.")
    parameters = bundle["models"][model]
    names = parameters["features"]
    if (
        not names
        or len(names) != len(set(names))
        or not set(names).issubset(feature_catalog().feature)
    ):
        raise ValueError("Residual feature names are invalid.")
    for key, shape in (
        ("mean", (len(names),)),
        ("scale", (len(names),)),
        ("coefficients", (len(names), 2)),
        ("intercept", (2,)),
    ):
        values = np.asarray(parameters[key], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("Residual parameters must have valid shapes and finite values.")
        if key == "scale" and (values <= 0).any():
            raise ValueError("Residual scales must be positive.")
    return parameters


def model_source(root: Path, model: str, weights: Path) -> tuple[str, list[Path]]:
    """Build a numerical cell without project-package imports and list its dependencies."""
    folder = root / "src/nfl_trajectory"
    paths = [folder / "motion.py", folder / "runtime.py"]
    nodes = definitions(paths[0]) + definitions(
        paths[1], {"Run", "stage", "sha256", "atomic_bytes", "atomic_json"}
    )
    imports = (
        "from __future__ import annotations\nimport hashlib\nimport importlib\n"
        "import json\nimport os\n"
        "import sys\nimport tempfile\nimport threading\nimport time\nimport uuid\n"
        "from collections.abc import Callable\nfrom filelock import FileLock\n"
        "from datetime import UTC, datetime\nfrom pathlib import Path\nfrom typing import Any\n"
        "import numpy as np\nimport pandas as pd\n"
    )
    learned = model not in ("constant_velocity", "research")
    if learned:
        from nfl_trajectory.models import BASIS

        paths.extend([folder / "models.py", weights])
        nodes.extend(definitions(folder / "models.py"))
        fitted = json.loads(weights.read_text())
        if fitted.get("format") != 1 or fitted.get("basis") != BASIS:
            raise ValueError("Use weights produced by nfl benchmark.")
        for parameters in [fitted["global"], *fitted["roles"].values()]:
            array = np.asarray(parameters["coefficients"], dtype=float)
            if array.shape != (len(BASIS),) or not np.isfinite(array).all():
                raise ValueError("Invalid baseline coefficients.")
    if model in RESIDUAL_MODELS:
        imports += "from dataclasses import dataclass\n"
        paths.extend(
            [
                folder / "features.py",
                folder / "feature_experiment.py",
                root / "artifacts/features/model.json",
                root / "artifacts/features/summary.json",
                root / ".state/features-report.json",
                root / "artifacts/game_splits.csv",
                folder / "research.py",
                folder / "benchmark.py",
            ]
        )
        residual = residual_parameters(root, model, weights)
        nodes.extend(definitions(folder / "features.py"))
        nodes.extend(
            definitions(folder / "feature_experiment.py", {"target_state", "predict_residual"})
        )
    source = imports + ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"
    if model == "research":
        from nfl_trajectory.research_inference import INFERENCE_MODULES, research_bundle

        bundle = research_bundle(root)
        modules = {}
        for name in INFERENCE_MODULES:
            path = folder / (name + ".py")
            paths.append(path)
            modules[name] = path.read_text().replace("nfl_trajectory.", "_nfl_export.")
        paths.extend(
            root / "artifacts" / stage_name / "development" / "models.json"
            for stage_name in ("research", "context", "representation")
        )
        paths.append(root / "artifacts/representation/development/routes.json")
        source += (
            "import types\n"
            "_package = types.ModuleType('_nfl_export')\n"
            "_package.__path__ = []\n"
            "sys.modules['_nfl_export'] = _package\n"
            "_MODULE_SOURCES = " + pprint.pformat(modules, width=85) + "\n"
            "_MODULE_ORDER = " + repr(INFERENCE_MODULES) + "\n"
            "for _name in _MODULE_ORDER:\n"
            "    _source = _MODULE_SOURCES[_name]\n"
            "    _qualified = '_nfl_export.' + _name\n"
            "    _module = types.ModuleType(_qualified)\n"
            "    _module.__file__ = _name + '.py'\n"
            "    _module.__package__ = '_nfl_export'\n"
            "    sys.modules[_qualified] = _module\n"
            "    exec(compile(_source, _module.__file__, 'exec'), _module.__dict__)\n"
            "research_predict = sys.modules['_nfl_export.research_inference'].predict_research\n"
            "RESEARCH_MODEL = " + pprint.pformat(bundle, width=85) + "\n"
        )
    if learned:
        source += (
            "trajectory_predict = predict\nFITTED_MODEL = "
            + pprint.pformat(fitted, width=85)
            + "\n"
        )
    if model in RESIDUAL_MODELS:
        source += "BATCH_ROWS = 1024\nRESIDUAL_MODEL = " + pprint.pformat(residual, width=85) + "\n"
    source += "INFERENCE_SIGNATURE = " + repr(hashlib.sha256(source.encode()).hexdigest()) + "\n"
    source += "_gateway_run: Run | None = None\n"
    return source, paths


def build_notebook(root: Path, model: str, weights: Path) -> tuple[Any, list[Path]]:
    if model not in ("constant_velocity", "role_ridge", "research", *RESIDUAL_MODELS):
        raise ValueError("Unknown export model.")
    source, dependencies = model_source(root, model, weights)
    prediction = "predictions = constant_velocity(observed, target[KEYS])"
    if model != "constant_velocity":
        prediction = (
            "predictions = trajectory_predict(observed, target[KEYS], 'role_ridge', FITTED_MODEL)"
        )
    if model in RESIDUAL_MODELS:
        prediction += (
            "\n    bank = build_player_features(observed, target[ENTITY])"
            "\n    predictions[['x', 'y']] = predictions[['x', 'y']].to_numpy() + "
            "predict_residual(bank, target[KEYS], RESIDUAL_MODEL)"
        )
    if model == "research":
        prediction = "predictions = research_predict(observed, target[KEYS], RESEARCH_MODEL)"
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
    # Ignore target coordinates and retain the exact incoming target row order.
    target = test.to_pandas() if hasattr(test, 'to_pandas') else test
    observed = test_input.to_pandas() if hasattr(test_input, 'to_pandas') else test_input
    require_keys(target)

    def compute():
        PREDICTION
        return predictions[['x', 'y']]

    if _gateway_run is None:
        return compute()
    digest = hashlib.sha256(INFERENCE_SIGNATURE.encode())
    digest.update((np.__version__ + pd.__version__).encode())
    digest.update(json.dumps(list(observed.columns)).encode())
    digest.update(pd.util.hash_pandas_object(observed, index=False).to_numpy().tobytes())
    digest.update(target[KEYS].to_numpy(dtype=np.int64).tobytes())
    key = digest.hexdigest()
    destination = Path.cwd() / 'artifacts/inference' / (key + '.json')

    def save_prediction():
        result = compute()
        atomic_json(destination, result.to_numpy(dtype=float).tolist())

    stage(Path.cwd(), 'inference-' + key, key, [destination], save_prediction, _gateway_run)
    values = np.asarray(json.loads(destination.read_text()), dtype=float)
    if values.shape != (len(target), 2) or not np.isfinite(values).all():
        raise ValueError('Cached predictions must match the requested rows and be finite.')
    _gateway_run.event('prediction_batch_completed', rows=len(target))
    return pd.DataFrame(values, columns=['x', 'y'])

with Run(Path.cwd(), 'kaggle_gateway') as active_run:
    _gateway_run = active_run
    try:
        server = inference_module.NFLInferenceServer(predict)
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            server.serve()
        else:
            server.run_local_gateway((str(COMPETITION_PATH),))
    finally:
        _gateway_run = None

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and Path('submission.parquet').is_file():
    from IPython.display import FileLink, display

    display(FileLink('submission.parquet', result_html_prefix='Download local gateway output: '))
""".replace("PREDICTION", prediction.replace("\n", "\n    "))
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# NFL trajectory inference\n\n"
                f"**Model:** `{model}`. Generated by the project owner "
                "from verified local weights. "
                "The numerical code and fitted parameters are embedded; "
                "no model download is needed. "
                "Attach the official competition input, use CPU, and disable internet. "
                "Running this notebook invokes the organizer gateway; it does not submit anything. "
                "Local sample predictions are not hidden-test predictions or a leaderboard score. "
                "You control any subsequent Kaggle submission.\n\n"
                "Verified per-play predictions are reused while the working directory is retained. "
                "Restore saved outputs before expecting a fresh runtime to reuse checkpoints. "
                "UTC logs include stage and total timing plus a 15-second heartbeat.\n\n"
                "Interface: [official organizer example]"
                "(https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)."
            ),
            nbformat.v4.new_code_cell(source),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_code_cell(interface),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "nfl_export": {
                "model": model,
                "official_gateway_status": "not_run",
                "automatically_submitted": False,
            },
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"nfl-inference-{index}"
    compile(
        "\n".join(c.source for c in notebook.cells if c.cell_type == "code"), "inference.py", "exec"
    )
    nbformat.validate(notebook)
    return notebook, dependencies


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["constant_velocity", "role_ridge", "research", *RESIDUAL_MODELS],
        default="constant_velocity",
    )
    parser.add_argument("--weights", type=Path, default=root / "artifacts/benchmark/model.json")
    parser.add_argument("--output", type=Path, default=root / "artifacts/kaggle/submission.ipynb")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(root / "artifacts") or destination.suffix != ".ipynb":
        raise ValueError("Export destination must be an .ipynb inside this project's artifacts/.")
    with Run(root, "export_kaggle") as run:
        notebook, dependencies = build_notebook(root, args.model, args.weights)
        dependencies += (
            [Path(__file__).resolve(), root / "uv.lock"]
            if (root / "uv.lock").exists()
            else [Path(__file__).resolve()]
        )
        input_hashes = {str(p): sha256(p) for p in dependencies}
        signature = hashlib.sha256(
            json.dumps(
                {"inputs": input_hashes, "model": args.model, "output": str(destination)},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def write() -> None:
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
                    "submission.ipynb",
                    "-",
                ],
                input=nbformat.writes(notebook),
                text=True,
                capture_output=True,
                check=True,
            )
            formatted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--stdin-filename",
                    "submission.ipynb",
                    "-",
                ],
                input=ordered.stdout,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--stdin-filename",
                    "submission.ipynb",
                    "-",
                ],
                input=formatted.stdout,
                text=True,
                capture_output=True,
                check=True,
            )
            nbformat.validate(nbformat.reads(formatted.stdout, as_version=4))
            if any(sha256(p) != input_hashes[str(p)] for p in dependencies):
                raise ValueError("Export inputs changed; the previous artifact was preserved.")
            atomic_bytes(destination, formatted.stdout.encode())

        key = hashlib.sha256(str(destination.relative_to(root)).encode()).hexdigest()[:12]
        stage(root, f"kaggle-export-{key}", signature, [destination], write, run)
        run.event(
            "notebook_exported",
            path=str(destination.relative_to(root)),
            model=args.model,
            sha256=sha256(destination),
            official_gateway_status="not_run",
            automatically_submitted=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
