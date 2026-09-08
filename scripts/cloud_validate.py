"""Validate the cloud runner in its real container before research execution."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import boto3


def main() -> None:
    root = pathlib.Path("/opt/ml/processing/project")
    root.mkdir(parents=True, exist_ok=True)
    ref = os.environ["NFL_REPO_REF"]
    archive = root.parent / "source.tar.gz"
    with (
        urllib.request.urlopen(
            "https://codeload.github.com/alvaromendizabal/nfl-player-trajectory/tar.gz/" + ref,
            timeout=90,
        ) as source,
        archive.open("wb") as target,
    ):
        shutil.copyfileobj(source, target)
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            parts = pathlib.PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            if ".." in parts or not (member.isdir() or member.isfile()):
                raise ValueError("Unsafe repository archive.")
            path = root.joinpath(*parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                content = source.extractfile(member)
                if content is None:
                    raise ValueError("Archive member has no content.")
                with content, path.open("wb") as output:
                    shutil.copyfileobj(content, output)
    os.chdir(root)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(root.parent / "uv-tools"),
            "uv==0.11.33",
        ],
        check=True,
        env=env,
    )
    uv = str(root.parent / "uv-tools/bin/uv")
    subprocess.run([uv, "sync", "--frozen", "--group", "dev"], check=True, env=env)
    python = str(root / ".venv/bin/python")
    formatted = [
        "scripts/cloud_research.py", "scripts/cloud_validate.py",
        "src/nfl_trajectory/research_evidence.py",
        "scripts/feature_attribution.py", "scripts/validate_gateway.py",
        *[str(p.relative_to(root)) for p in root.glob("notebooks/*.ipynb")],
    ]
    formatted = [name for name in formatted if (root / name).is_file()]
    subprocess.run([python, "-m", "ruff", "format", *formatted], check=True, env=env)
    subprocess.run([python, "-m", "ruff", "check", "--fix", *formatted], check=True, env=env)
    if (root / "scripts/validate_gateway.py").exists():
        subprocess.run([uv, "lock", "--script", "scripts/validate_gateway.py"], check=True, env=env)
        formatted.append("scripts/validate_gateway.py.lock")
    s3 = boto3.client("s3", region_name="us-west-2")
    for name in formatted:
        path = root / name
        s3.put_object(
            Bucket=os.environ["NFL_BUCKET"],
            Key=os.environ["NFL_PREFIX"] + "/" + name,
            Body=path.read_bytes(),
            ServerSideEncryption="AES256",
        )
    for command in [
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "ruff", "format", "--check", "."],
        [python, "-m", "mypy", "src", "scripts", "kaggle"],
        [python, "-m", "pytest", "-q"],
    ]:
        subprocess.run(command, check=True, env=env)
    s3.put_object(
        Bucket=os.environ["NFL_BUCKET"],
        Key=os.environ["NFL_PREFIX"] + "/validation.json",
        Body=json.dumps(
            {
                "status": "passed",
                "source_commit": ref,
                "checks": ["lint", "format", "types", "tests"],
            }
        ).encode(),
        ServerSideEncryption="AES256",
    )
    print("CLOUD_ENVIRONMENT_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
