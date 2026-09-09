"""Generate the final-model notebook using the shared, tested organizer interface."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path
from typing import Any

import nbformat
from export import build_notebook

from nfl_trajectory.final_inference import final_bundle, standalone_source
from nfl_trajectory.final_protocol import digest
from nfl_trajectory.runtime import Run, atomic_bytes, sha256, stage

KAGGLE_SOURCE_LIMIT = 1_000_000


def packed_assignment(name: str, value: Any) -> str:
    """Embed losslessly compressed JSON without external files or downloads."""
    # Module insertion order is its dependency order; preserve it through JSON.
    payload = json.dumps(value, separators=(",", ":")).encode()
    encoded = base64.b85encode(zlib.compress(payload, level=9)).decode("ascii")
    chunks = "\n".join("    " + repr(chunk) for chunk in textwrap.wrap(encoded, 76))
    return f"{name} = json.loads(zlib.decompress(base64.b85decode(\n{chunks}\n)))\n"


def final_notebook(root: Path) -> Any:
    """Retain the canonical gateway and recovery code; supply verified final inference."""
    notebook, _ = build_notebook(root, "constant_velocity", root / "artifacts/benchmark/model.json")
    bundle = final_bundle(root)
    source = notebook.cells[1].source.partition("INFERENCE_SIGNATURE =")[0]
    source = source.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\nimport base64\nimport types\nimport zlib\n",
        1,
    )
    for node in ast.parse(standalone_source(root, bundle)).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        name = (
            node.targets[0].id
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None
        )
        if isinstance(node, ast.Assign) and name in ("_sources", "FINAL_MODEL"):
            value = bundle if name == "FINAL_MODEL" else ast.literal_eval(node.value)
            source += packed_assignment(name, value)
        else:
            source += ast.unparse(node) + "\n"
    source += "INFERENCE_SIGNATURE = " + repr(hashlib.sha256(source.encode()).hexdigest()) + "\n"
    source += "_gateway_run: Run | None = None\n"
    notebook.cells[1].source = source
    original = "predictions = constant_velocity(observed, target[KEYS])"
    if notebook.cells[-1].source.count(original) != 1:
        raise ValueError("The shared organizer template changed; validate its final-model adapter.")
    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        original, "predictions = final_predict(observed, target[KEYS], FINAL_MODEL)"
    )
    notebook.cells[0].source = notebook.cells[0].source.replace("`constant_velocity`", "`final`")
    notebook.metadata.nfl_export.model = "final"
    compile(
        "\n".join(c.source for c in notebook.cells if c.cell_type == "code"),
        "final-notebook.py",
        "exec",
    )
    return notebook


def export_final(root: Path, output: Path, run: Run) -> None:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "artifacts") or output.suffix != ".ipynb":
        raise ValueError("Choose an .ipynb within this project's artifacts directory.")
    notebook = final_notebook(root)
    payload = nbformat.writes(notebook)
    source = digest(
        {
            "notebook": payload,
            "adapter": sha256(Path(__file__)),
            "template": sha256(root / "kaggle/export.py"),
        }
    )

    def action() -> None:
        formatted = payload
        for args in (("check", "--select", "I,UP", "--fix"), ("format",), ("check",)):
            result = subprocess.run(
                [sys.executable, "-m", "ruff", *args, "--stdin-filename", "submission.ipynb", "-"],
                input=formatted,
                text=True,
                capture_output=True,
            )
            if result.returncode:
                raise ValueError(
                    "Final notebook quality check failed: " + result.stdout + result.stderr
                )
            if args != ("check",):
                formatted = result.stdout
        nbformat.validate(nbformat.reads(formatted, as_version=4))
        encoded = formatted.encode()
        if len(encoded) >= KAGGLE_SOURCE_LIMIT:
            raise ValueError(
                f"Kaggle notebook exceeds its {KAGGLE_SOURCE_LIMIT:,}-byte source limit: "
                f"{len(encoded):,} bytes. Keep fitted parameters compressed."
            )
        atomic_bytes(output, encoded)

    stage(root, "final-notebook-export", source, [output], action, run)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "artifacts/kaggle/submission.ipynb")
    args = parser.parse_args()
    with Run(root, "final-notebook-export") as run:
        export_final(root, args.output, run)
