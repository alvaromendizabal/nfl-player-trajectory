"""Create the first local commit from an explicit publication allowlist."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nfl_trajectory.runtime import Run, checked_command


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        print("This project is already a Git repository. Use its existing branches and history.")
        return 1
    with Run(root, "git_initialization") as run:
        checked_command(["git", "init", "-b", "main"], root, run)
        for key, value in [
            ("user.name", "Alvaro Mendizabal"),
            ("user.email", "108156083+alvaromendizabal@users.noreply.github.com"),
        ]:
            exists = subprocess.run(
                ["git", "config", "--get", key], cwd=root, capture_output=True, check=False
            )
            if exists.returncode:
                checked_command(["git", "config", "--local", key, value], root, run)
        checked_command(
            [
                "git",
                "add",
                ".github",
                ".gitattributes",
                ".gitignore",
                ".python-version",
                "pyproject.toml",
                "uv.lock",
                "README.md",
                "START_HERE.md",
                "LICENSE",
                "src",
                "scripts",
                "tests",
                "notebooks",
                "docs",
                "kaggle",
            ],
            root,
            run,
        )
        checked_command(["git", "diff", "--cached", "--stat"], root, run)
        checked_command(
            ["git", "commit", "-m", "feat: establish tested NFL trajectory research foundation"],
            root,
            run,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
