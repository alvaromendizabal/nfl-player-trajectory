"""Build a self-contained inference notebook; never upload or submit to Kaggle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pprint
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
from filelock import FileLock

from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256

MODELS = ("constant_velocity", "role_ridge", "motion_ridge", "landing_ridge", "interaction_ridge")

# Embedded in the exported notebook, not imported from this repository at inference time.
RUNTIME = '''
CACHE_ENABLED = False
CACHE_DIR = Path.cwd() / "nfl_inference_cache"
RUN_STARTED = time.monotonic()
LOG_PATH = Path.cwd() / "inference.jsonl"
LOG_LOCK = threading.Lock()


def emit(event, **fields):
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "stage": "inference",
        "total_elapsed_seconds": round(time.monotonic() - RUN_STARTED, 3),
        "stage_elapsed_seconds": round(time.monotonic() - RUN_STARTED, 3),
        **fields,
    }
    line = json.dumps(record, allow_nan=False)
    with LOG_LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\\n")
            handle.flush()
        print(line, flush=True)


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_prediction(prediction, count):
    if list(prediction.columns) != ["x", "y"] or len(prediction) != count:
        raise ValueError("Predictions must preserve every requested row and contain only x/y.")
    if not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be finite; coordinates are not silently clipped.")
    return prediction


def cached_predict(target, observed):
    target = target[KEYS].copy()
    require_keys(target)
    if not CACHE_ENABLED:
        return validate_prediction(model_prediction(target, observed), len(target))
    started = time.monotonic()
    digest = hashlib.sha256(MODEL_FINGERPRINT.encode())
    digest.update(json.dumps([np.__version__, pd.__version__]).encode())
    for frame in (target, observed):
        digest.update(json.dumps(list(zip(frame.columns, map(str, frame.dtypes), strict=True))).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    signature = digest.hexdigest()
    folder = CACHE_DIR / signature
    prediction_path, receipt_path = folder / "prediction.npy", folder / "complete.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        payload = prediction_path.read_bytes()
        valid = (
            receipt.get("signature") == signature
            and receipt.get("sha256") == hashlib.sha256(payload).hexdigest()
            and receipt.get("rows") == len(target)
        )
        if valid:
            values = np.load(io.BytesIO(payload), allow_pickle=False)
            result = validate_prediction(pd.DataFrame(values, columns=["x", "y"]), len(target))
            emit("prediction_reused", rows=len(target), stage_elapsed_seconds=round(time.monotonic() - started, 3))
            return result
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        pass  # Missing or invalid receipts cause recomputation, never silent reuse.
    result = validate_prediction(model_prediction(target, observed), len(target))
    buffer = io.BytesIO()
    np.save(buffer, result.to_numpy(dtype=float), allow_pickle=False)
    payload = buffer.getvalue()
    atomic_write(prediction_path, payload)
    receipt = {"signature": signature, "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(target)}
    atomic_write(receipt_path, (json.dumps(receipt, sort_keys=True) + "\\n").encode())
    emit("prediction_completed", rows=len(target), stage_elapsed_seconds=round(time.monotonic() - started, 3))
    return result


def validate_gateway_output(folder, competition_path):
    result = json.loads((folder / "result.json").read_text())
    if result.get("Succeeded") is not True:
        raise ValueError("The official local gateway did not report success.")
    path = folder / "submission.parquet"
    submission = pd.read_parquet(path)
    if list(submission.columns) != ["id", "x", "y"]:
        raise ValueError("The official gateway output must have id, x, y in that order.")
    targets = pd.read_csv(competition_path / "test.csv", dtype={"id": str})
    ordered = pd.concat([part for _, part in targets.groupby(["game_id", "play_id"], sort=False)])
    if submission["id"].astype(str).tolist() != ordered["id"].tolist():
        raise ValueError("Gateway IDs/order differ from the organizer's play-by-play order.")
    if submission["id"].isna().any() or submission["id"].duplicated().any():
        raise ValueError("Gateway row IDs must be complete and unique.")
    validate_prediction(submission[["x", "y"]], len(targets))
    return {"format": 1, "rows": len(submission), "model": MODEL_NAME,
            "model_fingerprint": MODEL_FINGERPRINT,
            "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "official_local_gateway": "passed", "preview_only": True,
            "leaderboard_score": None, "uploaded_to_kaggle": False}


def run_gateway(server, competition_path):
    global CACHE_ENABLED, RUN_STARTED
    RUN_STARTED = time.monotonic()
    rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    CACHE_ENABLED = not rerun
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(15):
            emit("heartbeat", status="running", stage_elapsed_seconds=round(time.monotonic() - RUN_STARTED, 3))
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    emit("inference_started", model=MODEL_NAME, competition_rerun=rerun)
    try:
        if rerun:
            server.serve()
            return
        output_root = Path.cwd()
        with tempfile.TemporaryDirectory(prefix=".gateway-", dir=output_root) as temporary:
            folder = Path(temporary)
            try:
                os.chdir(folder)
                server.run_local_gateway((str(competition_path),))
                manifest = validate_gateway_output(folder, competition_path)
            finally:
                os.chdir(output_root)
            # An unsuccessful gateway never overwrites the previous valid submission.
            atomic_write(output_root / "submission.parquet", (folder / "submission.parquet").read_bytes())
            atomic_write(output_root / "submission_manifest.json", (json.dumps(manifest, indent=2) + "\\n").encode())
        emit("SUBMISSION_VALIDATED", rows=manifest["rows"], preview_only=True,
             stage_elapsed_seconds=round(time.monotonic() - RUN_STARTED, 3))
        from IPython.display import FileLink, display
        display(FileLink("submission.parquet", result_html_prefix="Download your gateway output: "))
        display(FileLink("submission_manifest.json", result_html_prefix="Download validation manifest: "))
        print("This is preview inference, not a scored submission. You control Kaggle submission.")
    except BaseException as error:
        emit("inference_failed", error_type=type(error).__name__)
        raise
    finally:
        stop.set()
        worker.join()
'''


def validate_residual(model: dict[str, Any], baseline_path: Path) -> None:
    """Reject stale, mismatched, unknown, and non-finite residual weights before writing."""
    from nfl_trajectory.feature_experiment import numerical_sources
    from nfl_trajectory.features import feature_catalog

    baseline = json.loads(baseline_path.read_text())
    if (
        model.get("format") != 1
        or model.get("baseline_sha256") != sha256(baseline_path)
        or not baseline.get("split_sha256")
        or not baseline.get("training_games")
        or model.get("split_sha256") != baseline.get("split_sha256")
        or model.get("training_games") != baseline.get("training_games")
        or model.get("source_sha256") != numerical_sources()
    ):
        raise ValueError("Residual weights do not match the baseline, training split, or numerical code.")
    known = set(feature_catalog().feature)
    if not isinstance(model.get("models"), dict) or not model["models"]:
        raise ValueError("Residual model collection is missing.")
    for fitted in model["models"].values():
        names = fitted.get("features", [])
        if not names or len(names) != len(set(names)) or not set(names).issubset(known):
            raise ValueError("Residual feature schema is invalid.")
        for name, shape in (("mean", (len(names),)), ("scale", (len(names),)),
                            ("coefficients", (len(names), 2)), ("intercept", (2,))):
            array = np.asarray(fitted.get(name), dtype=float)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("Residual weights have an invalid shape or nonfinite value.")
            if name == "scale" and (array <= 0).any():
                raise ValueError("Residual scales must be positive.")


def build_source(root: Path, model: str, weights: Path, feature_weights: Path) -> tuple[str, dict[str, Any]]:
    from nfl_trajectory.models import BASIS

    if model not in MODELS:
        raise ValueError("Choose an explicitly supported model; no fallback is automatic.")
    paths = [root / "src/nfl_trajectory/motion.py"]
    residual = model not in ("constant_velocity", "role_ridge")
    if model != "constant_velocity":
        paths.append(root / "src/nfl_trajectory/models.py")
    if residual:
        paths.append(root / "src/nfl_trajectory/features.py")
    definitions: list[ast.stmt] = []
    for path in paths:
        definitions.extend(node for node in ast.parse(path.read_text()).body
                           if not isinstance(node, (ast.Import, ast.ImportFrom))
                           and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)))
    if residual:
        parsed = ast.parse((root / "src/nfl_trajectory/feature_experiment.py").read_text())
        definitions.extend(node for node in parsed.body
                           if isinstance(node, ast.FunctionDef) and node.name in ("target_state", "predict_residual"))
    source = (
        "from __future__ import annotations\n"
        "import hashlib\nimport importlib\nimport io\nimport json\nimport os\n"
        "import sys\nimport tempfile\nimport threading\nimport time\n"
        "from datetime import UTC, datetime\nfrom pathlib import Path\n"
        + ("from typing import Any\n" if model != "constant_velocity" else "")
        + ("from dataclasses import dataclass\n" if residual else "")
        + "import numpy as np\nimport pandas as pd\n"
        + ast.unparse(ast.Module(body=definitions, type_ignores=[])) + "\n"
    )
    evidence: dict[str, Any] = {"model": model, "sources": {p.name: sha256(p) for p in paths}}
    if model != "constant_velocity":
        fitted = json.loads(weights.read_text())
        if fitted.get("basis") != BASIS or fitted.get("format") != 1:
            raise ValueError("Use weights produced by nfl benchmark.")
        source += "trajectory_predict = predict\nFITTED_MODEL = " + pprint.pformat(fitted, width=85, sort_dicts=True) + "\n"
        evidence["baseline_sha256"] = sha256(weights)
    if residual:
        feature_model = json.loads(feature_weights.read_text())
        validate_residual(feature_model, weights)
        if model not in feature_model["models"]:
            raise ValueError("The selected residual model has not been trained.")
        source += "BATCH_ROWS = 1024\nRESIDUAL_MODEL = " + pprint.pformat(feature_model["models"][model], width=85, sort_dicts=True) + "\n"
        evidence["feature_model_sha256"] = sha256(feature_weights)
    if model == "constant_velocity":
        implementation = "return constant_velocity(observed, target[KEYS])[[\"x\", \"y\"]]"
    else:
        implementation = "result = trajectory_predict(observed, target[KEYS], 'role_ridge', FITTED_MODEL)\n"
        if residual:
            implementation += "bank = build_player_features(observed, target[ENTITY].drop_duplicates())\nresult[['x', 'y']] += predict_residual(bank, target[KEYS], RESIDUAL_MODEL)\n"
        implementation += "return result[['x', 'y']]"
    source += "\ndef model_prediction(target, observed):\n" + "\n".join("    " + line for line in implementation.splitlines()) + "\n"
    identity = hashlib.sha256((source + RUNTIME).encode()).hexdigest()
    source += f"\nMODEL_NAME = {model!r}\nMODEL_FINGERPRINT = {identity!r}\n" + RUNTIME
    ordered = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "I,UP", "--fix", "--stdin-filename", "model.py", "-"], input=source, text=True, capture_output=True, check=True)
    return ordered.stdout, evidence


def export_notebook(root: Path, model: str, weights: Path, feature_weights: Path) -> Path:
    """Generate a reproducible, checksummed notebook from saved weights; never retrain."""
    destination = root / "artifacts/kaggle/submission.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / "export.lock"), timeout=1), Run(root, "export_kaggle") as run:
        source, evidence = build_source(root, model, weights, feature_weights)
        setup = '''COMPETITION_PATH = Path(os.getenv("NFL_COMPETITION_PATH", "/kaggle/input/nfl-big-data-bowl-2026-prediction")).expanduser().resolve()
if not COMPETITION_PATH.is_dir():
    candidates = list(Path("/kaggle/input").glob("competitions/nfl-big-data-bowl-2026-prediction"))
    if len(candidates) != 1:
        raise FileNotFoundError("Attach the official competition, or set NFL_COMPETITION_PATH to its local data directory.")
    COMPETITION_PATH = candidates[0].resolve()
sys.path.insert(0, str(COMPETITION_PATH))
inference_module = importlib.import_module("kaggle_evaluation.nfl_inference_server")
'''
        interface = '''def predict(test, test_input):
    # Preserve incoming target order. Return x/y only; the official gateway adds IDs.
    target = test.to_pandas() if hasattr(test, "to_pandas") else test
    observed = test_input.to_pandas() if hasattr(test_input, "to_pandas") else test_input
    return cached_predict(target, observed)

server = inference_module.NFLInferenceServer(predict)
run_gateway(server, COMPETITION_PATH)
'''
        description = (
            f"# NFL trajectory inference\n\n**Model:** `{model}`. Generated by your notebook from saved weights; no training occurs here. "
            "The learned weights and required inference code are embedded. Attach the official competition, use CPU, and disable internet. "
            "Run all cells to generate and validate `submission.parquet`, then use the displayed download link. "
            "The preview is not a leaderboard score; you decide whether to submit the saved version to Kaggle.\n\n"
            "Completed local play predictions are reusable only with matching input/code/model/version signatures and checksums. "
            "An interrupted play is recomputed. Cache files survive only while their filesystem or a saved copy is retained; "
            "a new Kaggle session does not automatically restore them. Hidden competition reruns do not reuse preview caches.\n\n"
            "UTC events, per-play elapsed time, total time, and 15-second heartbeats are written to `inference.jsonl`. "
            "The previous valid output is preserved if gateway validation fails. No automatic upload or submission is performed.\n\n"
            "Interface: [official organizer example](https://www.kaggle.com/code/sohier/nfl-2026-demo-submission)."
        )
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(description), nbformat.v4.new_code_cell(source), nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(interface)], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        formatted = subprocess.run([sys.executable, "-m", "ruff", "format", "--stdin-filename", "submission.ipynb", "-"], input=nbformat.writes(notebook), text=True, capture_output=True, check=True).stdout
        result = nbformat.reads(formatted, as_version=4)
        nbformat.validate(result)
        compile("\n".join(c.source for c in result.cells if c.cell_type == "code"), "submission.py", "exec")
        # Compare before writing so a repeated export does not invalidate downstream receipts.
        # nbformat IDs are normalized for byte-for-byte reproducible generation.
        for i, cell in enumerate(result.cells):
            cell.id = f"nfl-inference-{i}"
        payload = nbformat.writes(result).encode()
        if not destination.exists() or destination.read_bytes() != payload:
            atomic_bytes(destination, payload)
        manifest = {"format": 1, **evidence, "notebook_sha256": sha256(destination), "official_gateway_status": "not_run", "uploaded_to_kaggle": False}
        atomic_json(destination.parent / "export_manifest.json", manifest)
        run.event("notebook_exported", path=str(destination.relative_to(root)), model=model, official_gateway_status="not_run", total_elapsed_seconds=round(time.monotonic() - run.started, 3))
    return destination


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="constant_velocity")
    parser.add_argument("--weights", type=Path, default=root / "artifacts/benchmark/model.json")
    parser.add_argument("--feature-weights", type=Path, default=root / "artifacts/features/model.json")
    args = parser.parse_args()
    export_notebook(root, args.model, args.weights, args.feature_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
