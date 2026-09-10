"""Notebook execution and publication contracts; all small data here are synthetic."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from nfl_trajectory.runtime import Run, atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebooks_runner", ROOT / "scripts/notebooks.py")
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/notebooks.py", tmp_path / "scripts/notebooks.py")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "artifacts/benchmark").mkdir(parents=True)
    (tmp_path / "artifacts/game_splits.csv").write_text("synthetic split fixture\n")
    split_hash = sha256(tmp_path / "artifacts/game_splits.csv")
    summary = {
        "status": "passed",
        "split": "validation",
        "holdout_evaluation": "not_run",
        "numerical_signature": "synthetic-benchmark-signature",
        "split_sha256": split_hash,
    }
    atomic_json(tmp_path / "artifacts/benchmark/summary.json", summary)
    atomic_json(
        tmp_path / "artifacts/benchmark/protocol.json",
        {"split_sha256": split_hash, "holdout_evaluation": "not_run"},
    )
    atomic_json(tmp_path / "artifacts/benchmark/model.json", {"split_sha256": split_hash})
    for name in runner.REPORT_FILES:
        path = tmp_path / "artifacts/benchmark" / name
        if not path.exists():
            path.write_bytes(b"synthetic report fixture")
    atomic_json(
        tmp_path / ".state/benchmark-summary.json",
        {
            "status": "completed",
            "signature": summary["numerical_signature"],
            "outputs": {
                "artifacts/benchmark/summary.json": sha256(
                    tmp_path / "artifacts/benchmark/summary.json"
                )
            },
        },
    )
    return tmp_path


def notebook(root: Path, name: str = "01_analysis.ipynb", code: str = "print('executed')") -> Path:
    path = root / "notebooks" / name
    value = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    nbformat.write(value, path)
    return path


def execute(root: Path, source: Path) -> Path:
    with Run(root, "test_notebook") as run:
        runner.execute(root, source, run)
    return root / "artifacts/notebooks" / source.name


def test_source_hash_ignores_outputs_but_not_code() -> None:
    value = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")])
    initial = runner.source_hash(value)
    value.cells[0].execution_count = 1
    value.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="2")]
    assert runner.source_hash(value) == initial
    value.cells[0].source = "2 + 2"
    assert runner.source_hash(value) != initial


@pytest.mark.parametrize(
    "name", ["domain_research.json", "motion_research.json", "domain_manifest.json"]
)
def test_published_domain_research_invalidates_notebook_cache(project: Path, name: str) -> None:
    source = notebook(project)
    before = runner.execution_signature(project, source)
    path = project / "docs/results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1}')
    added = runner.execution_signature(project, source)
    path.write_text('{"version": 2}')
    assert added != before
    assert runner.execution_signature(project, source) != added


def test_execute_reuses_verified_output_and_recovers_corruption(project: Path) -> None:
    source = notebook(project)
    output = execute(project, source)
    payload = output.read_bytes()
    modified = output.stat().st_mtime_ns
    execute(project, source)
    assert output.stat().st_mtime_ns == modified
    output.write_text("corrupt")
    execute(project, source)
    assert output.read_bytes() == payload
    events = [
        json.loads(line)
        for path in (project / "logs").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(event["event"] == "stage_reused" for event in events)
    assert all("timestamp" in event and "elapsed_seconds" in event for event in events)


def test_changed_source_and_results_invalidate_execution(project: Path) -> None:
    source = notebook(project)
    output = execute(project, source)
    before = nbformat.read(output, as_version=4).metadata.execution.signature
    source = notebook(project, code="print('changed')")
    execute(project, source)
    after = nbformat.read(output, as_version=4).metadata.execution.signature
    assert before != after
    (project / "artifacts/benchmark/eda.json").write_text("new synthetic fixture")
    execute(project, source)
    assert nbformat.read(output, as_version=4).metadata.execution.signature != after


def test_changed_visual_helper_invalidates_notebook_evidence(project: Path) -> None:
    source = notebook(project)
    before = runner.execution_signature(project, source)
    helper = project / "src/nfl_trajectory/research_visuals.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# initial visual helper\n")
    added = runner.execution_signature(project, source)
    helper.write_text("# revised visual helper\n")
    changed = runner.execution_signature(project, source)
    assert len({before, added, changed}) == 3


@pytest.mark.parametrize(
    "relative",
    [
        "docs/results/final_evaluation.json",
        "docs/results/final_results_manifest.json",
        "src/nfl_trajectory/final_results.py",
    ],
)
def test_final_publication_invalidates_previous_notebook_checkpoint(
    project: Path, relative: str
) -> None:
    source = notebook(project)
    output = execute(project, source)
    previous = nbformat.read(output, as_version=4).metadata.execution.signature
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("initial final evidence\n")
    execute(project, source)
    first = nbformat.read(output, as_version=4).metadata.execution.signature
    path.write_text("changed final evidence\n")
    execute(project, source)
    second = nbformat.read(output, as_version=4).metadata.execution.signature
    assert len({previous, first, second}) == 3


@pytest.mark.parametrize(
    "code,error",
    [
        ("raise ValueError('deliberate test failure')", ValueError),
        ("import warnings; warnings.warn('deliberate test warning', UserWarning)", UserWarning),
        ("import sys; print('deliberate stderr', file=sys.stderr)", RuntimeError),
    ],
)
def test_failed_cell_preserves_last_good_output(
    project: Path, code: str, error: type[Exception]
) -> None:
    source = notebook(project)
    output = execute(project, source)
    before = output.read_bytes()
    notebook(project, code=code)
    with pytest.raises(error):
        execute(project, source)
    assert output.read_bytes() == before
    state = json.loads((project / ".state/notebook-01_analysis.json").read_text())
    assert state["status"] == "failed"


@pytest.mark.parametrize("change", ["missing", "checkpoint", "split", "summary", "holdout"])
def test_publication_rejects_invalid_local_evidence(project: Path, change: str) -> None:
    if change == "missing":
        (project / "artifacts/benchmark/eda.png").unlink()
    elif change == "checkpoint":
        (project / ".state/benchmark-summary.json").unlink()
    elif change == "split":
        (project / "artifacts/game_splits.csv").write_text("changed")
    elif change == "summary":
        (project / "artifacts/benchmark/summary.json").write_text('{"status":"passed"}')
    else:
        atomic_json(project / "artifacts/benchmark/protocol.json", {"holdout_evaluation": "scored"})
    with pytest.raises(ValueError):
        runner.local_results(project)


def test_publication_updates_canonical_files_and_is_repeatable(project: Path) -> None:
    source = notebook(project)
    output = execute(project, source)
    expected = runner.local_results(project)
    with Run(project, "test_publish") as run:
        runner.publish(project, [source], expected, run)
    assert source.read_bytes() == output.read_bytes()
    receipt = json.loads((project / "artifacts/notebooks/publication.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["official_gateway_status"] == "not_run"
    assert len(receipt["files"]) == len(runner.REPORT_FILES) + 1
    assert all(sha256(project / name) == digest for name, digest in receipt["files"].items())
    modified = source.stat().st_mtime_ns
    with Run(project, "test_publish") as run:
        runner.publish(project, [source], expected, run)
    assert source.stat().st_mtime_ns == modified


def test_invalid_second_notebook_does_not_replace_first(project: Path) -> None:
    first = notebook(project)
    second = notebook(project, "02_analysis.ipynb")
    execute(project, first)
    output = execute(project, second)
    original = first.read_bytes()
    value = nbformat.read(output, as_version=4)
    value.cells[0].execution_count = None
    nbformat.write(value, output)
    with Run(project, "test_publish") as run, pytest.raises(ValueError):
        runner.publish(project, [first, second], runner.local_results(project), run)
    assert first.read_bytes() == original
    assert not (project / "docs/results").exists()


def test_stale_notebook_is_not_published(project: Path) -> None:
    source = notebook(project)
    execute(project, source)
    (project / "artifacts/benchmark/eda.json").write_text("changed synthetic artifact")
    with Run(project, "test_publish") as run, pytest.raises(ValueError, match="stale"):
        runner.publish(project, [source], runner.local_results(project), run)


def test_changed_snapshot_is_not_published(project: Path) -> None:
    source = notebook(project)
    execute(project, source)
    expected = runner.local_results(project)
    (project / "artifacts/benchmark/eda.json").write_text("changed synthetic artifact")
    with Run(project, "test_publish") as run, pytest.raises(ValueError, match="changed"):
        runner.publish(project, [source], expected, run)


def test_subprocess_execution_and_publication(project: Path) -> None:
    first = notebook(project, code="isolated_value = 1; print('first')")
    second = notebook(
        project, "02_analysis.ipynb", "assert 'isolated_value' not in globals(); print('second')"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(project / "scripts/notebooks.py"), "--publish"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert "notebooks_published" in result.stdout
    assert result.stderr == ""
    for source in (first, second):
        assert nbformat.read(source, as_version=4).cells[0].execution_count == 1


def test_generated_demo_does_not_invalidate_notebook(project: Path) -> None:
    source = notebook(
        project,
        code=(
            "from pathlib import Path\n"
            f"folder = Path({str(project)!r}) / 'artifacts/demo'\n"
            "folder.mkdir(parents=True, exist_ok=True)\n"
            "(folder / 'metrics.json').write_text('{}')\n"
            "print('demo generated')"
        ),
    )
    assert execute(project, source).is_file()


def test_modified_executed_outputs_are_not_published(project: Path) -> None:
    source = notebook(project)
    output = execute(project, source)
    value = nbformat.read(output, as_version=4)
    value.cells[0].outputs[0].text = "unverified replacement"
    nbformat.write(value, output)
    with Run(project, "test_publish") as run, pytest.raises(ValueError, match="checkpoint"):
        runner.publish(project, [source], runner.local_results(project), run)
    assert nbformat.read(source, as_version=4).cells[0].execution_count is None
