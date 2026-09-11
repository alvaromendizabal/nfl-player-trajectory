#!/bin/bash
# Pinned, fast-forward-only sync. Local files are never removed or overwritten.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'An exact reviewed Git commit is required.' >&2; exit 2
fi
timeout --signal=TERM --kill-after=5s 180s python3 - "$1" <<'PY'
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import boto3
from botocore.config import Config

root = Path('/home/sagemaker-user/nfl-player-trajectory')
target = sys.argv[1]
metadata = json.loads(Path('/opt/ml/metadata/resource-metadata.json').read_text())
if (metadata.get('DomainId'), metadata.get('SpaceName')) != ('d-njhxv1erusdc', 'nfl-trajectory-dev'):
    raise ValueError('Wrong workspace; no repository operation attempted.')
if root.is_symlink() or not (root / '.git').is_dir():
    raise ValueError('The existing regular Git checkout is required.')
s3 = boto3.client('s3', region_name='us-west-2', config=Config(
    connect_timeout=5, read_timeout=20, retries={'max_attempts': 2}))
bucket = 'sagemaker-nfl-trajectory-560403859723-us-west-2'
prefix = 'operations/workspace_sync/' + target


def git(*args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45).stdout


def save(key, payload):
    s3.put_object(Bucket=bucket, Key=key, Body=payload,
        ServerSideEncryption='AES256', ExpectedBucketOwner='560403859723')
    actual = s3.get_object(Bucket=bucket, Key=key,
        ExpectedBucketOwner='560403859723')['Body'].read()
    if actual != payload:
        raise ValueError('S3 read-back mismatch.')
    return {'key': key, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


origin = git('remote', 'get-url', 'origin').decode().strip()
if origin not in {'https://github.com/alvaromendizabal/nfl-player-trajectory',
                  'https://github.com/alvaromendizabal/nfl-player-trajectory.git'}:
    raise ValueError('Unexpected repository origin.')
if git('branch', '--show-current').decode().strip() != 'feat/velocity-supervision':
    raise ValueError('Unexpected branch; no branch switch attempted.')
if git('diff', 'HEAD', '--name-only') or git('diff', '--cached', '--name-only'):
    raise ValueError('Tracked edits are present; all files are preserved.')
previous = git('rev-parse', 'HEAD').decode().strip()
local_receipt = root / 'docs/results/supervision_execution.json'
preserved = None
if local_receipt.is_file() and not local_receipt.is_symlink():
    if local_receipt.stat().st_size > 1048576:
        raise ValueError('Unexpected receipt size.')
    preserved = save(prefix + '/preserved/supervision_execution.json', local_receipt.read_bytes())
git('fetch', '--no-tags', 'origin', target)
git('merge-base', '--is-ancestor', 'HEAD', target)
git('merge', '--ff-only', target)
if git('rev-parse', 'HEAD').decode().strip() != target or git('diff', 'HEAD', '--name-only'):
    raise ValueError('Exact source revision verification failed.')
archive = git('archive', '--format=tar.gz', target)
verified = {}
notebooks = {}
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as handle:
    for member in handle:
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError('Unexpected Git archive entry.')
        payload = handle.extractfile(member).read()
        path = root / member.name
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError('Tracked content differs: ' + member.name)
        verified[member.name] = hashlib.sha256(payload).hexdigest()
        if member.name.startswith('notebooks/') and member.name.endswith('.ipynb'):
            code = [c for c in json.loads(payload)['cells'] if c['cell_type'] == 'code']
            notebooks[member.name] = {'code_cells': len(code),
                'executed_cells': sum(c.get('execution_count') is not None for c in code),
                'error_outputs': sum(o.get('output_type') == 'error' for c in code for o in c.get('outputs', []))}
report = {'status': 'tracked_source_and_notebooks_synced', 'previous_commit': previous,
    'actual_commit': target, 'tracked_files': len(verified), 'tracked_sha256': verified,
    'notebooks': notebooks, 'preserved_receipt': preserved, 'scientific_fits': 0,
    'feature_completion_gate': 'open', 'utc': datetime.now(timezone.utc).isoformat(),
    'git_status': git('status', '--porcelain=v1').decode(),
    'source_archive': save(prefix + '/source.tar.gz', archive)}
save(prefix + '/receipt.json', (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())
print(json.dumps({k:v for k,v in report.items() if k != 'tracked_sha256'}), flush=True)
PY
