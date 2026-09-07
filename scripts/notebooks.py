"""Execute plain-Python notebook cells in a dedicated, socket-free process.

The quality gate invokes this script as a separate process. IPython executes every
cell in one shared notebook namespace and captures real text/HTML/SVG outputs.
This supports this project's notebooks, which do not need browser widget comms.
Interactive use remains JupyterLab with the registered project kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from nfl_trajectory.runtime import Run, atomic_bytes


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with Run(root, "notebook") as run:
        source = root / "notebooks" / "00_project_readiness.ipynb"
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        shell = InteractiveShell.instance()
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            result.raise_error()
            cell.execution_count = count
            cell.outputs = []
            for name, value in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if value:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data",
                        data=output.data,
                        metadata=output.metadata,
                    )
                )
            run.event("cell_completed", cell=count)
        notebook.metadata["execution"] = {"method": "isolated_process_ipython", "cells": count}
        nbformat.validate(notebook)
        atomic_bytes(
            root / "artifacts" / "notebooks" / source.name, nbformat.writes(notebook).encode()
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
