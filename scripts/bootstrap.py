"""Create the locked project environment without modifying the Jupyter base environment."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    log = root / "logs" / "bootstrap.jsonl"
    log.parent.mkdir(exist_ok=True)

    def event(status: str) -> None:
        line = json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": status,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        )
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

    def command(args: list[str], label: str) -> None:
        event(label + "_started")
        process = subprocess.Popen(args, cwd=root)
        while True:
            try:
                status = process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                event(label + "_heartbeat")
        if status:
            raise subprocess.CalledProcessError(status, args)
        event(label + "_completed")

    try:
        uv = shutil.which("uv")
        if uv is None:
            tool_env = root / ".venv-bootstrap"
            command([sys.executable, "-m", "venv", str(tool_env)], "bootstrap_environment")
            python = tool_env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            command([str(python), "-m", "pip", "install", "uv==0.11.33"], "install_uv")
            uv_command = [str(python), "-m", "uv"]
        else:
            uv_command = [uv]
        command([*uv_command, "sync", "--frozen", "--group", "dev"], "sync_environment")
        python = (
            root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        )
        command(
            [
                str(python),
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                "nfl-trajectory",
                "--display-name",
                "Python (NFL Trajectory)",
            ],
            "register_kernel",
        )
        command([str(python), "scripts/quality.py"], "quality_gate")
        event("bootstrap_completed")
        return 0
    except (OSError, subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        event("bootstrap_stopped_" + type(exc).__name__)
        print("Rerun this command after resolving the displayed error; existing work is retained.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
