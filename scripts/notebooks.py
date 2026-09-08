"""Execute notebooks with checkpoints; publish only verified local benchmark results."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import nbformat
from filelock import FileLock
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from nfl_trajectory.research_evidence import EXTRA_REPORTS, extended_evidence
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, fingerprint, sha256, stage

REPORT_FILES = (
    "summary.json",
    "protocol.json",
    "model.json",
    "eda.json",
    "latency.json",
    "benchmark.png",
    "coefficients.png",
    "eda.png",
)
FEATURE_REPORTS = {
    "summary.json": "feature_summary.json",
    "model.json": "feature_model.json",
    "benchmark.png": "feature_benchmark.png",
}
RESEARCH_REPORTS = {
    "summary.json": "feature_research.json",
    "permutation.json": "feature_permutation.json",
    "stability.png": "feature_stability.png",
    "catalog.csv": "feature_catalog.csv",
}


def research_results(root: Path) -> dict[str, str]:
    folder = root / "artifacts/research"
    if not folder.exists():
        return {}
    from nfl_trajectory.research import load_research_report

    _, _, label = load_research_report(root)
    if label != "Verified local feature research":
        raise ValueError("Finish and report local feature research before publication.")
    return {f"research/{name}": sha256(folder / "report" / name) for name in RESEARCH_REPORTS}


def feature_results(root: Path) -> dict[str, str]:
    """Optional feature reports require completed, current-code evidence before publication."""
    folder = root / "artifacts/features"
    if not folder.exists():
        return {}
    paths = [folder / name for name in FEATURE_REPORTS]
    checkpoint_path = root / ".state/features-report.json"
    if not all(path.is_file() for path in [*paths, checkpoint_path]):
        raise ValueError("Feature experiment is incomplete; rerun nfl features before publication.")
    checkpoint = json.loads(checkpoint_path.read_text())
    summary = json.loads((folder / "summary.json").read_text())
    model = json.loads((folder / "model.json").read_text())
    from nfl_trajectory.feature_experiment import numerical_sources

    split_hash = sha256(root / "artifacts/game_splits.csv")
    if (
        checkpoint.get("status") != "completed"
        or summary.get("status") != "passed"
        or summary.get("holdout_evaluation") != "not_run"
        or summary.get("split") != "validation"
        or summary.get("screening_split") != "train"
        or checkpoint.get("signature") != summary.get("numerical_signature")
        or any(value.get("split_sha256") != split_hash for value in (model, summary))
        or any(value.get("source_sha256") != numerical_sources() for value in (model, summary))
        or any(
            value.get("baseline_sha256") != sha256(root / "artifacts/benchmark/model.json")
            for value in (model, summary)
        )
    ):
        raise ValueError("Feature results are stale or have inconsistent completion evidence.")
    for path in paths:
        if checkpoint.get("outputs", {}).get(str(path.relative_to(root))) != sha256(path):
            raise ValueError("Feature report hash does not match its completed checkpoint.")
    return {f"features/{name}": sha256(folder / name) for name in FEATURE_REPORTS}


def source_hash(notebook: Any) -> str:
    """Exclude outputs, so publishing a notebook does not invalidate its computation."""
    content = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def local_results(root: Path) -> dict[str, str]:
    """Require a completed local benchmark, not the bundled public snapshot."""
    folder = root / "artifacts/benchmark"
    paths = [folder / name for name in REPORT_FILES]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise ValueError("Run nfl benchmark before publishing: local report files are incomplete.")
    summary = json.loads((folder / "summary.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    model = json.loads((folder / "model.json").read_text())
    checkpoint_path = root / ".state/benchmark-summary.json"
    if not checkpoint_path.is_file():
        raise ValueError("A completed local benchmark checkpoint is required for publication.")
    checkpoint = json.loads(checkpoint_path.read_text())
    if (
        summary.get("status") != "passed"
        or summary.get("split") != "validation"
        or summary.get("holdout_evaluation") != "not_run"
        or protocol.get("holdout_evaluation") != "not_run"
        or checkpoint.get("status") != "completed"
        or not summary.get("numerical_signature")
        or checkpoint.get("signature") != summary["numerical_signature"]
        or checkpoint.get("outputs", {}).get("artifacts/benchmark/summary.json")
        != sha256(folder / "summary.json")
    ):
        raise ValueError("Local benchmark completion evidence is missing or inconsistent.")
    split_path = root / "artifacts/game_splits.csv"
    if not split_path.is_file():
        raise ValueError("The frozen local split manifest is required.")
    split_hash = sha256(split_path)
    if any(item.get("split_sha256") != split_hash for item in (summary, protocol, model)):
        raise ValueError("Benchmark results do not match the frozen local split manifest.")
    return (
        {name: sha256(folder / name) for name in REPORT_FILES}
        | feature_results(root)
        | research_results(root)
        | {f"extended/{k}": v for k, v in extended_evidence(root).items()}
    )


def execution_signature(root: Path, source: Path) -> str:
    local = root / "artifacts/benchmark"
    results = local if (local / "summary.json").is_file() else root / "docs/results"
    inputs = [root / "scripts/notebooks.py"]
    inputs.extend(results / name for name in REPORT_FILES if (results / name).is_file())
    feature_local = root / "artifacts/features"
    inputs.extend(
        path
        for original, published in FEATURE_REPORTS.items()
        if (
            path := (
                feature_local / original
                if (feature_local / "summary.json").is_file()
                else root / "docs/results" / published
            )
        ).is_file()
    )
    inputs.extend(
        path
        for name in ("audit_summary.json", "game_splits.csv")
        if (path := root / "artifacts" / name).is_file()
    )
    selection_path = root / "docs/results/feature_selection.json"
    if selection_path.is_file() and not (feature_local / "summary.json").is_file():
        inputs.append(selection_path)
    for name, published in RESEARCH_REPORTS.items():
        local_report = root / "artifacts/research/report" / name
        path = local_report if local_report.exists() else root / "docs/results" / published
        if path.exists():
            inputs.append(path)
    for name, original in EXTRA_REPORTS.items():
        local_report = root / "artifacts" / original
        path = local_report if local_report.exists() else root / "docs/results" / name
        if path.exists():
            inputs.append(path)
    return fingerprint(
        root,
        inputs,
        {
            "notebook": source.name,
            "source_sha256": source_hash(nbformat.read(source, as_version=4)),
        },
    )


def validate_executed(source: Path, executed: Any) -> None:
    nbformat.validate(executed)
    original = nbformat.read(source, as_version=4)
    if source_hash(original) != source_hash(executed):
        raise ValueError("Executed notebook source differs from the canonical notebook.")
    count = 0
    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        if cell.execution_count != count:
            raise ValueError("Every code cell must have a consecutive execution count.")
        for output in cell.outputs:
            if output.output_type == "error" or (
                output.output_type == "stream" and output.name == "stderr" and output.text
            ):
                raise ValueError("Notebook contains an error or stderr output.")
    if count == 0:
        raise ValueError("An executed notebook must contain at least one code cell.")


def validate_review_controls(notebook: Any) -> None:
    """Refuse armed manual controls before automatic execution can run any cell."""
    manual = {"RUN_FEATURE_EXPERIMENT", "GENERATE_EXPORT"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if not isinstance(node, ast.Assign):
                continue
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            armed = names & manual
            if armed and not (isinstance(node.value, ast.Constant) and node.value.value is False):
                raise ValueError(
                    "Turn off manual notebook controls before automatic publication: "
                    + ", ".join(sorted(armed))
                    + ". Run those cells interactively to train or create your own export."
                )


def execute(root: Path, source: Path, run: Run) -> None:
    destination = root / "artifacts/notebooks" / source.name
    signature = execution_signature(root, source)

    def action() -> None:
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        validate_review_controls(notebook)
        shell = InteractiveShell.instance()
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            started = time.monotonic()
            run.event("cell_started", notebook=source.name, cell=count)
            cell.outputs = []
            cell.execution_count = None
            with warnings.catch_warnings(), capture_output() as captured:
                warnings.simplefilter("error")
                result = shell.run_cell(cell.source, store_history=True)
            result.raise_error()
            if captured.stderr:
                raise RuntimeError(f"Notebook {source.name} cell {count} emitted stderr.")
            cell.execution_count = count
            if captured.stdout:
                cell.outputs.append(
                    nbformat.v4.new_output("stream", name="stdout", text=captured.stdout)
                )
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=output.data, metadata=output.metadata
                    )
                )
            run.event(
                "cell_completed",
                notebook=source.name,
                cell=count,
                elapsed_cell_seconds=round(time.monotonic() - started, 3),
            )
        notebook.metadata["execution"] = {
            "method": "isolated_process_ipython",
            "cells": count,
            "signature": signature,
            "source_sha256": source_hash(notebook),
        }
        validate_executed(source, notebook)
        if execution_signature(root, source) != signature:
            raise ValueError("Notebook inputs changed during execution; no output was replaced.")
        atomic_bytes(destination, nbformat.writes(notebook).encode())

    stage(root, f"notebook-{source.stem}", signature, [destination], action, run)


def publish(root: Path, sources: list[Path], expected: dict[str, str], run: Run) -> None:
    """Validate every candidate before replacing any canonical notebook or result file."""
    if not sources or local_results(root) != expected:
        raise ValueError("Local report changed during execution or no notebooks were selected.")
    payloads: dict[Path, bytes] = {}
    for source in sources:
        path = root / "artifacts/notebooks" / source.name
        notebook = nbformat.read(path, as_version=4)
        validate_executed(source, notebook)
        if notebook.metadata.get("execution", {}).get("signature") != execution_signature(
            root, source
        ):
            raise ValueError("Executed notebook is stale; rerun scripts/notebooks.py --publish.")
        checkpoint_path = root / ".state" / f"notebook-{source.stem}.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        if (
            checkpoint.get("status") != "completed"
            or checkpoint.get("signature") != notebook.metadata.execution.signature
            or checkpoint.get("outputs", {}).get(str(path.relative_to(root))) != sha256(path)
        ):
            raise ValueError("Executed notebook does not match its completed checkpoint.")
        payloads[source] = path.read_bytes()
    for name in REPORT_FILES:
        payload = (root / "artifacts/benchmark" / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[name]:
            raise ValueError("A report artifact changed during publication validation.")
        payloads[root / "docs/results" / name] = payload
    for name, published in FEATURE_REPORTS.items():
        key = f"features/{name}"
        if key not in expected:
            continue
        payload = (root / "artifacts/features" / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[key]:
            raise ValueError("A feature report changed during publication validation.")
        payloads[root / "docs/results" / published] = payload
    if "features/summary.json" in expected:
        from nfl_trajectory.research import selection_study

        study = selection_study(
            json.loads((root / "artifacts/features/model.json").read_text()),
            expected["features/summary.json"],
            expected["features/model.json"],
        )
        payloads[root / "docs/results/feature_selection.json"] = (
            json.dumps(study, indent=2, allow_nan=False) + "\n"
        ).encode()
    for name, published in RESEARCH_REPORTS.items():
        key = f"research/{name}"
        if key in expected:
            payload = (root / "artifacts/research/report" / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected[key]:
                raise ValueError("A research report changed during publication validation.")
            payloads[root / "docs/results" / published] = payload
    receipt = root / "artifacts/notebooks/publication.json"
    for published, original in EXTRA_REPORTS.items():
        key = f"extended/{published}"
        if key in expected:
            payload = (root / "artifacts" / original).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected[key]:
                raise ValueError("An extended experiment changed during publication.")
            payloads[root / "docs/results" / published] = payload
    atomic_json(receipt, {"status": "running"})
    try:
        for path, payload in payloads.items():
            if not path.exists() or path.read_bytes() != payload:
                atomic_bytes(path, payload)
        atomic_json(
            receipt,
            {
                "status": "passed",
                "files": {str(path.relative_to(root)): sha256(path) for path in payloads},
                "official_gateway_status": "not_run",
                "holdout_evaluation": "not_run",
            },
        )
    except BaseException:
        atomic_json(receipt, {"status": "failed"})
        raise
    run.event(
        "notebooks_published", notebooks=len(sources), report_files=len(payloads) - len(sources)
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", help="Execute one notebook by its basename.")
    parser.add_argument(
        "--publish", action="store_true", help="Refresh canonical files from local results."
    )
    args = parser.parse_args()
    if args.notebook is not None:
        if args.publish:
            parser.error("--publish executes and validates the complete notebook set.")
        if Path(args.notebook).name != args.notebook or not args.notebook.endswith(".ipynb"):
            raise ValueError("Select a notebook basename from notebooks/.")
        with Run(root, "notebook") as run:
            execute(root, root / "notebooks" / args.notebook, run)
        return 0
    (root / ".state").mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".state/notebooks.lock"), timeout=1), Run(root, "notebooks") as run:
        sources = sorted((root / "notebooks").glob("*.ipynb"))
        if not sources:
            raise ValueError("No canonical notebooks were found.")
        expected = local_results(root) if args.publish else {}
        for source in sources:
            run.event("notebook_started", notebook=source.name)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--notebook", source.name],
                cwd=root,
                check=True,
            )
            run.event("notebook_completed", notebook=source.name)
        if args.publish:
            publish(root, sources, expected, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
