#!/usr/bin/env python3
"""Bounded, preservation-first NFL workspace synchronization and input recovery.

No training, cloud resource changes, GitHub writes, credential printing, or
unconditional overwrite. Python >=3.9 for Git/file commands; the existing
project Python 3.11 environment supplies boto3/Kaggle for explicit data reads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPOSITORY = "alvaromendizabal/nfl-player-trajectory"
REVIEWED_MAIN = "402843faa1722460aa84d1bbaf27c4050a7b7ff9"
ACCOUNT = "560403859723"
BUCKET = "sagemaker-nfl-trajectory-560403859723-us-west-2"
REGION = "us-west-2"
SAMPLE_SHA = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
SNAPSHOTS = [
    "snapshots/d7992e30f488b53dcd0027883dd2e5bf0404d166de3c2f86f64c31be1ac3a4a9.json",
    "snapshots/5d01a2afa894be877b6e0a086e630fae0045a0db0de15ba077a456dfb0e48166.json",
]
PROTECTED = {"data", "artifacts", "models", "checkpoints", "logs", ".state",
             ".venv", ".downloads", "aws.local.json", ".env", "kaggle.json"}
REQUIRED_INPUT = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y",
                  "player_to_predict", "num_frames_output", "ball_land_x", "ball_land_y"}
REQUIRED_OUTPUT = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing a symbolic-link receipt destination")
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Run:
    def __init__(self, home, command, seconds=600):
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.home / (stamp + "-" + command + "-" + uuid.uuid4().hex[:6])
        self.path.mkdir(mode=0o700)
        self.started = time.monotonic()
        self.seconds = seconds
        self.command = command
        self.events = self.path / "events.jsonl"
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        self.event("started", command=command, budget_seconds=seconds)

    def _heartbeat(self):
        while not self.closed.wait(15):
            self.event("heartbeat", command=self.command)

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError("Stage budget exceeded; completed files/receipts are retained")

    def event(self, event, **fields):
        record = {"utc": utc(), "event": event,
                  "elapsed_seconds": round(time.monotonic() - self.started, 3), **fields}
        text = json.dumps(record, sort_keys=True, allow_nan=False)
        print(text, flush=True)
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def finish(self, result):
        self.closed.set()
        result = {"utc": utc(), "command": self.command,
                  "elapsed_seconds": round(time.monotonic() - self.started, 3),
                  "scientific_fits": 0, "new_rmse": None,
                  "github_write_performed": False, "cloud_resource_changes": 0,
                  "feature_research": "open", "receipt_directory": str(self.path), **result}
        atomic_json(self.path / "result.json", result)
        atomic_json(self.home / ("latest_" + self.command + ".json"), result)
        self.event("finished", status=result.get("status"), receipt=str(self.path / "result.json"))
        return result


def sha256(path, run=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, non-symlink file: " + str(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if run:
                run.check()
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while hashing; close other writers: " + str(path))
    return digest.hexdigest()


def safe_destination(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe relative path")
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or any(p in {"..", "."} for p in relative.split("/")):
        raise ValueError("Unsafe relative path")
    root = Path(root).resolve()
    destination = root / Path(*part.parts)
    for current in (destination, *destination.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError("Symlinks are not valid restore destinations: " + relative)
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Path escapes project")
    return destination


def command(args, cwd, run, label, seconds=120, extra_env=None):
    """Log output privately, emit bounded heartbeats, kill the process group on timeout."""
    run.check()
    log = run.path / (label + ".log")
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "PYTHONUNBUFFERED": "1"})
    if extra_env:
        env.update(extra_env)
    remaining = max(0.01, min(seconds, run.seconds - (time.monotonic() - run.started)))
    with log.open("wb") as output:
        p = subprocess.Popen(args, cwd=str(cwd), env=env, stdout=output,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise TimeoutError(label + " stopped at its wall-clock limit; see private log")
    data = log.read_bytes()
    run.event("command_completed", label=label, exit_code=rc, log=str(log))
    if rc != 0:
        raise RuntimeError(label + " failed; see " + str(log))
    return data.decode("utf-8", errors="replace").strip()


def git(repo, run, *args, label=None, seconds=120):
    return command(["git", "-c", "core.hooksPath=/dev/null", "-c", "submodule.recurse=false",
                    "-c", "maintenance.auto=false", *args], repo, run,
                   label or ("git_" + uuid.uuid4().hex[:8]), seconds)


def ancestor(repo, older, newer):
    p = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=repo,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return p.returncode == 0


def validate_origin(origin):
    allowed = {"https://github.com/" + REPOSITORY, "https://github.com/" + REPOSITORY + ".git",
               "git@github.com:" + REPOSITORY + ".git", "ssh://git@github.com/" + REPOSITORY + ".git"}
    if origin not in allowed:
        raise ValueError("Origin is not the expected NFL repository; no sync performed")


def quality_for_sha(sha):
    url = "https://api.github.com/repos/" + REPOSITORY + "/commits/" + sha + "/check-runs?per_page=100&filter=latest"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "NFL-manual-readiness"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    rows = [r for r in payload.get("check_runs", []) if r.get("name") == "quality"
            and r.get("head_sha") == sha and r.get("app", {}).get("slug") == "github-actions"]
    if not rows:
        raise ValueError("No verified quality check for the fetched main SHA; do not bypass")
    record = max(rows, key=lambda r: r["id"])
    if (record.get("status"), record.get("conclusion")) != ("completed", "success"):
        raise ValueError("Latest quality check has not passed; do not sync or rerun training")
    return {k: record.get(k) for k in ("id", "name", "status", "conclusion", "head_sha", "html_url", "completed_at")}


def is_protected(relative):
    return PurePosixPath(relative).parts[0] in PROTECTED


def prepare_sync(repo, run, fetch=True, check_quality=quality_for_sha, minimum=REVIEWED_MAIN):
    repo = Path(repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise ValueError("Existing checkout not found at " + str(repo) + "; do not create a replacement")
    validate_origin(git(repo, run, "remote", "get-url", "origin", label="origin"))
    if git(repo, run, "status", "--porcelain=v1", "--untracked-files=no", label="tracked_status"):
        raise ValueError("Tracked edits or conflicts exist. They are preserved. Do not reset/clean/stash; review tracked_status.log")
    head = git(repo, run, "rev-parse", "HEAD", label="before_head")
    branch = git(repo, run, "symbolic-ref", "--short", "HEAD", label="before_branch")
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"):
        path = git(repo, run, "rev-parse", "--git-path", marker)
        if (repo / path).exists():
            raise ValueError("An unfinished Git operation must be resolved first")
    if fetch:
        git(repo, run, "fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main",
            label="fetch_main", seconds=120)
    target = git(repo, run, "rev-parse", "refs/remotes/origin/main", label="fetched_main")
    if not ancestor(repo, minimum, target):
        raise ValueError("Fetched main does not include the reviewed baseline; investigate, do not force")
    if not ancestor(repo, head, target):
        raise ValueError("Current branch has commits not in main. Preserved unchanged; review before switching")
    main_exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/main"],
                                 cwd=repo, timeout=30).returncode == 0
    if main_exists and not ancestor(repo, "refs/heads/main", target):
        raise ValueError("Local main diverges from GitHub main; preserved unchanged")
    changed = git(repo, run, "diff", "--name-only", head, target, label="incoming_paths").splitlines()
    if any(is_protected(p) for p in changed):
        raise ValueError("Incoming revision touches a protected private-data/artifact path; manual review required")
    # Reject any target path that could shadow an untracked/ignored file or directory.
    old = set(git(repo, run, "ls-tree", "-r", "--name-only", head).splitlines())
    new = git(repo, run, "ls-tree", "-r", "--name-only", target).splitlines()
    for relative in new:
        path = safe_destination(repo, relative)
        if relative not in old and (path.exists() or path.is_symlink()):
            raise ValueError("Incoming path collides with preserved local content: " + relative)
        for parent in path.parents:
            if parent == repo:
                break
            if parent.is_file():
                raise ValueError("Incoming directory collides with a local file")
    quality = check_quality(target)
    result = {"status": "safe_sync_plan", "repo": str(repo), "branch_before": branch,
              "head_before": head, "target_main": target, "local_main_exists": main_exists,
              "incoming_paths": changed, "quality": quality, "data_recovery_included": False}
    atomic_json(run.path / "sync_plan.json", result)
    return result


def inventory_raw(repo, run=None):
    root = Path(repo) / "data" / "raw"
    rows = []
    if root.is_symlink():
        raise ValueError("Raw directory is a symlink; review its intended target before continuing")
    if root.exists():
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in dirs:
                if (Path(parent) / name).is_symlink():
                    raise ValueError("Symlink within raw directory requires review")
            for name in sorted(files):
                path = Path(parent) / name
                rows.append({"path": path.relative_to(repo).as_posix(), "size": path.stat().st_size,
                             "sha256": sha256(path, run)})
    return sorted(rows, key=lambda r: r["path"])


def private_metadata(repo):
    """Metadata-only preservation check for large artifacts; not a model integrity proof."""
    rows = {}
    for prefix in ("artifacts", "models", "checkpoints", "logs", ".state"):
        root = Path(repo) / prefix
        if root.is_symlink():
            rows[prefix] = {"symlink": os.readlink(root)}
            continue
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                p = Path(parent) / name
                if p.is_symlink():
                    rows[p.relative_to(repo).as_posix()] = {"symlink": os.readlink(p)}
                elif p.is_file():
                    s = p.stat()
                    rows[p.relative_to(repo).as_posix()] = {"size": s.st_size, "mtime_ns": s.st_mtime_ns}
    return rows


def apply_sync(repo, run, plan):
    repo = Path(repo).resolve()
    if git(repo, run, "rev-parse", "HEAD") != plan["head_before"]:
        raise ValueError("Checkout changed after planning")
    if git(repo, run, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("Tracked changes appeared after planning; preserved")
    before = inventory_raw(repo, run)
    meta = private_metadata(repo)
    atomic_json(run.path / "raw_before.json", before)
    atomic_json(run.path / "private_metadata_before.json", meta)
    git(repo, run, "bundle", "create", str(run.path / "source_refs.bundle"), "--all",
        label="backup_git_refs", seconds=120)
    # Fast-forward the active branch first. An old feature branch remains available by name.
    git(repo, run, "merge", "--ff-only", "--no-overwrite-ignore", plan["target_main"], label="fast_forward")
    if plan["branch_before"] != "main":
        if plan["local_main_exists"]:
            git(repo, run, "checkout", "--no-overwrite-ignore", "main", label="switch_main")
            git(repo, run, "merge", "--ff-only", "--no-overwrite-ignore", plan["target_main"], label="advance_main")
        else:
            git(repo, run, "checkout", "--no-overwrite-ignore", "-b", "main", plan["target_main"], label="create_local_main")
    git(repo, run, "branch", "--set-upstream-to=origin/main", "main", label="upstream")
    after = inventory_raw(repo, run)
    if before != after or meta != private_metadata(repo):
        raise ValueError("Preservation comparison changed; stop all writers and inspect receipts")
    actual = git(repo, run, "rev-parse", "HEAD", label="verified_head")
    if actual != plan["target_main"] or git(repo, run, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("Final source equality check failed; do not run research")
    remote = git(repo, run, "ls-remote", "origin", "refs/heads/main", label="remote_readback").split()[0]
    return {"status": "source_synced_data_preserved" if remote == actual else "source_synced_remote_advanced",
            "repo": str(repo), "head": actual, "remote_main_at_readback": remote,
            "raw_files_sha256_verified_unchanged": len(before),
            "raw_bytes_verified_unchanged": sum(r["size"] for r in before),
            "private_artifact_metadata_unchanged": True,
            "all_model_bytes_hashed": False, "raw_data_complete": "not_yet_audited",
            "quality": plan["quality"], "git_reference_backup": str(run.path / "source_refs.bundle")}


def data_audit(repo, run):
    rows = inventory_raw(repo, run)
    by_name = {r["path"]: r for r in rows}
    weeks = []
    problems = []
    for week in range(1, 19):
        row = {"week": week}
        for prefix, required in (("input", REQUIRED_INPUT), ("output", REQUIRED_OUTPUT)):
            relative = f"data/raw/train/{prefix}_2023_w{week:02d}.csv"
            entry = by_name.get(relative)
            row[prefix + "_present"] = entry is not None
            row[prefix + "_bytes"] = entry["size"] if entry else 0
            if entry is None:
                problems.append("missing: " + relative)
                continue
            with (Path(repo) / relative).open(newline="", encoding="utf-8-sig") as stream:
                columns = next(csv.reader(stream), [])
                first = next(csv.reader(stream), None)
            if not required.issubset(columns) or first is None:
                problems.append("invalid/empty schema: " + relative)
        weeks.append(row)
    for name in ("test.csv", "test_input.csv"):
        if "data/raw/" + name not in by_name:
            problems.append("missing organizer sample: " + name)
    api_files = [r for r in rows if r["path"].startswith("data/raw/kaggle_evaluation/")]
    if len(api_files) < 11:
        problems.append("organizer API inventory below recorded 11 files")
    return {"status": "raw_readiness_passed" if not problems else "raw_readiness_incomplete",
            "repo": str(Path(repo)), "files": rows, "weeks": weeks, "problems": problems,
            "file_count": len(rows), "total_bytes": sum(r["size"] for r in rows),
            "organizer_api_files": len(api_files), "expected_recorded_inventory": 49,
            "verification_scope": "full local hashes; CSV headers and first-row existence; NOT full row-level/label audit",
            "historical_split_preserved": True, "labels_used_for_feature_selection": False}


def validate_manifest(payload, key):
    if not re.fullmatch(r"snapshots/[0-9a-f]{64}\.json", key):
        raise ValueError("Invalid snapshot key")
    if hashlib.sha256(payload).hexdigest() != PurePosixPath(key).stem:
        raise ValueError("Snapshot SHA256 mismatch")
    value = json.loads(payload)
    if value.get("format") != 1 or not isinstance(value.get("files"), list):
        raise ValueError("Unsupported snapshot format")
    seen = set()
    for item in value["files"]:
        path = item["path"]
        safe_destination(Path.cwd(), path)
        if path in seen or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid/duplicate snapshot entry")
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise ValueError("Invalid object length")
        seen.add(path)
    return value["files"]


def select_inputs(manifests):
    selected = {}
    for entries in manifests:
        for item in entries:
            p = item["path"]
            keep = p.startswith("data/raw/") or (p.startswith("artifacts/") and item["sha256"] == SAMPLE_SHA)
            if not keep:
                continue
            if p in selected and selected[p] != item:
                raise ValueError("Snapshot versions disagree about an input; no overwrite allowed")
            selected[p] = item
    return [selected[k] for k in sorted(selected)]


def install_verified(root, item, read_blocks, run):
    """Atomic create-only install. Completed files are resumable, differing files stop."""
    path = safe_destination(root, item["path"])
    if path.exists():
        if path.stat().st_size != item["size"] or sha256(path, run) != item["sha256"]:
            raise ValueError("Existing file differs; preserved unchanged: " + item["path"])
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nfl-input-", dir=path.parent)
    hasher, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as stream:
            for block in read_blocks():
                run.check()
                size += len(block)
                if size > item["size"]:
                    raise ValueError("Download exceeds expected object length")
                hasher.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size != item["size"] or hasher.hexdigest() != item["sha256"]:
            raise ValueError("Downloaded input checksum/length failed")
        # Hard-link is create-only and cannot replace a file created by another writer.
        os.link(temporary, path)
        return "restored"
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def aws_session():
    import boto3
    from botocore.config import Config
    session = boto3.Session(region_name=REGION)
    config = Config(connect_timeout=5, read_timeout=20, retries={"total_max_attempts": 2})
    if session.client("sts", config=config).get_caller_identity()["Account"] != ACCOUNT:
        raise ValueError("Unexpected AWS account; no project objects were read")
    return session.client("s3", config=config)


def restore_inputs(repo, run, apply=False):
    client = aws_session()
    manifests = []
    for key in SNAPSHOTS:
        run.check()
        response = client.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=ACCOUNT)
        body = response["Body"]
        try:
            payload = body.read(2 * 1024**2 + 1)
        finally:
            body.close()
        if len(payload) > 2 * 1024**2:
            raise ValueError("Snapshot exceeds metadata budget")
        manifests.append(validate_manifest(payload, key))
    items = select_inputs(manifests)
    if not items:
        raise ValueError("Known snapshots do not contain requested raw/cache inputs")
    atomic_json(run.path / "selected_inputs.json", {"snapshots": SNAPSHOTS, "files": items})
    missing = []
    for item in items:
        path = safe_destination(repo, item["path"])
        if path.exists():
            if path.stat().st_size != item["size"] or sha256(path, run) != item["sha256"]:
                raise ValueError("Existing input differs from snapshot; preserved: " + item["path"])
        else:
            missing.append(item)
    bytes_needed = sum(i["size"] for i in missing)
    if bytes_needed > 2 * 1024**3:
        raise ValueError("Restore exceeds 2 GiB stage cap; review selected_inputs.json")
    if shutil.disk_usage(repo).free < bytes_needed + 2 * 1024**3:
        raise ValueError("Need missing input bytes plus 2 GiB of free-space reserve")
    result = {"status": "input_restore_plan", "selected_files": len(items),
              "missing_files": len(missing), "missing_bytes": bytes_needed,
              "snapshots": SNAPSHOTS, "raw_output_files_in_snapshots": sum("/output_" in i["path"] for i in items),
              "sample_cache_paths": [i["path"] for i in items if i["sha256"] == SAMPLE_SHA]}
    atomic_json(run.path / "restore_plan.json", result)
    if not apply:
        return result
    for index, item in enumerate(items, 1):
        def blocks(item=item):
            r = client.get_object(Bucket=BUCKET, Key="objects/" + item["sha256"], ExpectedBucketOwner=ACCOUNT)
            body = r["Body"]
            try:
                if r["ContentLength"] != item["size"]:
                    raise ValueError("S3 object length differs from snapshot")
                while True:
                    block = body.read(4 * 1024 * 1024)
                    if not block:
                        break
                    yield block
            finally:
                body.close()
        status = install_verified(repo, item, blocks, run)
        run.event("input_verified", completed=index, total=len(items), path=item["path"], result=status)
        atomic_json(run.path / "last_completed_file.json", {"completed": index, "file": item})
    result["status"] = "snapshot_inputs_restored_or_reused"
    result["next"] = "Run data-audit; snapshots may not contain every original label file"
    return result


def kaggle_missing(repo, run):
    """Official per-file recovery; no overwrite, no new API credentials or Kaggle submission."""
    sys.path.insert(0, str(Path(repo) / "src"))
    from nfl_trajectory.data import kaggle_api, unpack_download, COMPETITION
    api = kaggle_api()
    files, token, tokens = [], None, set()
    while True:
        run.check()
        response = api.competition_list_files(COMPETITION, page_token=token, page_size=100)
        files.extend(response.files or [])
        token = response.next_page_token
        if not token:
            break
        if token in tokens:
            raise ValueError("Repeated Kaggle pagination token")
        tokens.add(token)
    if not files or len({i.name for i in files}) != len(files):
        raise ValueError("Missing or duplicate official competition inventory")
    selected = []
    for item in files:
        target = safe_destination(repo, "data/raw/" + item.name)
        size = getattr(item, "total_bytes", None)
        if not isinstance(size, int) or size <= 0 or size > 2 * 1024**3:
            raise ValueError("Official file size unavailable/unexpected; stop before transfer")
        if target.exists() and (not target.is_file() or target.stat().st_size != size):
            raise ValueError("Existing official file size differs; preserved: " + item.name)
        selected.append((item.name, size, target))
    transfer = sum(s for _, s, p in selected if not p.exists())
    if transfer > 2 * 1024**3 or shutil.disk_usage(repo).free < transfer + 2 * 1024**3:
        raise ValueError("Missing official inputs exceed transfer/storage limit")
    restored = 0
    partial_root = run.home / "kaggle_download_cache"
    partial_root.mkdir(exist_ok=True)
    for index, (name, size, target) in enumerate(selected, 1):
        run.check()
        if not target.exists():
            folder = partial_root / hashlib.sha256(name.encode()).hexdigest()
            folder.mkdir(exist_ok=True)
            api.competition_download_file(COMPETITION, name, path=str(folder), force=False, quiet=True)
            # Unpack to this run, never directly onto existing project content.
            staged = run.path / "staged" / name
            unpack_download(folder, name, staged)
            if staged.stat().st_size != size:
                raise ValueError("Official file length does not match inventory")
            item = {"path": "data/raw/" + name, "size": size, "sha256": sha256(staged, run)}
            def blocks(staged=staged):
                with staged.open("rb") as stream:
                    yield from iter(lambda: stream.read(4 * 1024**2), b"")
            install_verified(repo, item, blocks, run)
            restored += 1
            staged.unlink()
        run.event("kaggle_file_present", completed=index, total=len(selected), file=name)
    return {"status": "official_files_present", "official_file_count": len(selected), "new_files": restored,
            "existing_files_overwritten": 0,
            "verification": "official inventory sizes; local SHA256 recorded by subsequent data-audit; no publisher SHA supplied"}


def project_python(repo):
    python = Path(repo) / ".venv" / "bin" / "python"
    if not python.exists():
        raise ValueError("Existing .venv/bin/python is missing; no reinstall or alternate environment was attempted")
    return python


def doctor(repo, run):
    python = project_python(repo)
    program = ('import sys,json,importlib.metadata as m,importlib.util as u; '
               'p=["numpy","pandas","plotly","boto3","ipykernel"]; '
               'print(json.dumps({"python":sys.version,"executable":sys.executable,'
               '"version_ok":sys.version_info[:2]==(3,11),'
               '"packages":{x:(m.version(x) if u.find_spec(x) else None) for x in p}}))')
    payload = json.loads(command([str(python), "-c", program], repo, run, "environment", 30))
    if not payload["version_ok"] or any(v is None for v in payload["packages"].values()):
        raise ValueError("Existing project environment fails readiness; see environment.log; do not upgrade it blindly")
    payload.update(status="environment_ready", repo=str(repo), packages_installed=0)
    return payload


def find_cache(repo, run):
    # Known filename first; bounded fallback on pickle files in artifacts only.
    candidates = sorted((Path(repo) / "artifacts").rglob("*.pkl"))
    if len(candidates) > 300:
        raise ValueError("More than 300 candidate cache files; supply --cache explicitly")
    for p in candidates:
        run.check()
        if p.is_file() and not p.is_symlink() and p.stat().st_size < 1024**3 and sha256(p, run) == SAMPLE_SHA:
            return p
    raise ValueError("Exact frozen sample cache not found. Run restore-inputs before the feature smoke")


def run_feature_smoke(repo, run, cache=None):
    doctor(repo, run)
    cache = Path(cache).expanduser().resolve() if cache else find_cache(repo, run)
    if sha256(cache, run) != SAMPLE_SHA:
        raise ValueError("The sample cache hash differs; no pickle was loaded")
    output = run.path / "feature_smoke"
    command([str(project_python(repo)), "scripts/audit_temporal_edges.py", "--cache", str(cache),
             "--output", str(output)], repo, run, "training_only_32_play_audit", 150,
            {"PYTHONPATH": str(Path(repo) / "src"), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    result = json.loads((output / "feature_smoke.json").read_text())
    if result.get("status") != "training_only_feature_smoke_passed" or result.get("scientific_fits") != 0:
        raise ValueError("Feature smoke failed its expected no-training contract")
    result.update(repo=str(repo), output=str(output), cache=str(cache))
    return result


def inspect_result(repo, run):
    doctor(repo, run)
    output = run.path / "motion_inspection"
    command([str(project_python(repo)), "scripts/inspect_motion_run.py", "--output", str(output)],
            repo, run, "inspect_existing_motion_run", 210, {"PYTHONPATH": str(Path(repo) / "src")})
    result = json.loads((output / "inspection.json").read_text())
    if result.get("status") != "saved_errors_and_checkpoint_bytes_verified":
        raise ValueError("Existing run inspection did not pass; inspect its receipt, do not launch a replacement")
    result.update(repo=str(repo), output=str(output), new_rmse=None)
    return result


def return_report(run):
    destination = run.home / "nfl_workspace_report.zip"
    temporary = run.path / "return_report.zip"
    count = 0
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.home.glob("latest_*.json")):
            if path.name == "latest_return-report.json":
                continue
            if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
                raise ValueError("Unexpected receipt file; review before sharing")
            archive.write(path, path.name)
            count += 1
        archive.writestr("CONTENTS.txt", "Private diagnostic receipt export. Contains status, file paths, sizes and hashes.\n"
                          "Does not contain raw tracking, per-frame errors, model weights, credentials, or Git logs.\n"
                          "Review before sharing. This is not an experiment submission.\n")
    if destination.is_symlink():
        raise ValueError("Refusing symlink report destination")
    os.replace(temporary, destination)
    return {"status": "return_report_created", "path": str(destination), "receipt_count": count,
            "sha256": sha256(destination, run)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "sync", "data-audit", "restore-inputs", "kaggle-missing",
                                      "doctor", "feature-smoke", "inspect-result", "return-report"])
    p.add_argument("--repo", type=Path, default=Path.home() / "nfl-player-trajectory")
    p.add_argument("--receipts", type=Path, default=Path.home() / "nfl-workspace-evidence")
    p.add_argument("--apply", action="store_true", help="Restore missing S3 inputs (restore-inputs only)")
    p.add_argument("--cache", type=Path)
    p.add_argument("--seconds", type=int, default=600)
    args = p.parse_args()
    if not 30 <= args.seconds <= 900:
        p.error("Each stage must have a 30–900 second budget")
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        p.error("Existing project directory not found; no replacement was created")
    args.receipts = args.receipts.expanduser().resolve()
    if args.receipts.is_relative_to(repo):
        p.error("Receipts must be outside the Git checkout")
    os.umask(0o077)
    run = Run(args.receipts, args.action, args.seconds)
    # A watchdog bounds SDK/network operations as well as subprocesses. Finished files survive.
    def expired():
        try:
            run.finish({"status": "stopped_wall_clock_limit", "error": "Stage reached its hard time limit"})
        finally:
            os._exit(124)
    timer = threading.Timer(args.seconds + 5, expired)
    timer.daemon = True
    timer.start()
    try:
        if args.action in {"plan", "sync"}:
            plan = prepare_sync(repo, run)
            result = plan if args.action == "plan" else apply_sync(repo, run, plan)
        elif args.action == "data-audit":
            result = data_audit(repo, run)
        elif args.action == "restore-inputs":
            result = restore_inputs(repo, run, args.apply)
        elif args.action == "kaggle-missing":
            result = kaggle_missing(repo, run)
        elif args.action == "doctor":
            result = doctor(repo, run)
        elif args.action == "feature-smoke":
            result = run_feature_smoke(repo, run, args.cache)
        elif args.action == "inspect-result":
            result = inspect_result(repo, run)
        else:
            result = return_report(run)
        run.finish(result)
        return 2 if result.get("status") == "raw_readiness_incomplete" else 0
    except (Exception, KeyboardInterrupt) as exc:
        # Detailed SDK exceptions may contain request info; keep details in mode-700 private receipts.
        run.finish({"status": "stopped_review_required", "error_type": type(exc).__name__,
                    "error": str(exc), "repo": str(repo)})
        print("STOPPED: " + str(exc), file=sys.stderr)
        return 2
    finally:
        timer.cancel()
        run.closed.set()


if __name__ == "__main__":
    raise SystemExit(main())
