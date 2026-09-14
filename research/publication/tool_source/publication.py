"""User-operated NFL source publication. No ML fitting or private-data upload.

Commands are intentionally split at the public-write and merge boundaries.
The live workspace is the source; a linked worktree isolates publication edits.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile

REPO = 'alvaromendizabal/nfl-player-trajectory'
URL = 'https://github.com/' + REPO + '.git'
VERSION = '2026-09-12.publication.1'  # Saved-plan schema; compatible with the staged predecessor.
HOME = Path.home()
ORIGINAL = HOME / 'nfl-player-trajectory'
STATE = HOME / '.nfl-workspace' / 'publication'
SOURCE_DIRS = [f'nfl_feature_round{n}' for n in range(1, 12)] + [
    'nfl_feature_rounds12_13', 'nfl_feature_rounds14_15',
    'nfl_feature_rounds16_17', 'nfl_manual_sync',
]
PRIVATE_PARTS = {'data', 'artifacts', 'logs', 'evidence', 'reference', 'payload',
    '.state', '.venv', '.git', '__pycache__', '.ipynb_checkpoints',
    'labels_private', 'models', 'features', 'runtime', 'backups'}
DOC_NAMES = {'EXPERIMENT_PROTOCOL.md', 'FEATURE_RESEARCH_PLAN.md',
    'RESEARCH_LEDGER.md', 'SOURCES.md', 'DIAGNOSIS_AND_VALIDATION.md',
    'VALIDATION_SCOPE.md'}
SECRET_RE = re.compile(
    r'(?:(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{25,}|'
    r'github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'X-Amz-Signature=[a-fA-F0-9]{32,}|'
    r'(?:aws_secret_access_key|aws_session_token)\s*[:=]\s*[\"\x27][A-Za-z0-9/+=]{30,})'
)
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
MAX_FILE = 8 * 1024**2


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refusing symlink destination: ' + str(path))
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_rel(value):
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe relative path')
    return Path(*p.parts)


def regular(path, root=None):
    p = Path(path)
    if root is not None:
        rel = p.relative_to(root)
        q = Path(root)
        for part in rel.parts:
            q = q / part
            if q.is_symlink():
                raise ValueError('Symlink excluded: ' + str(q))
    if p.is_symlink() or not p.is_file():
        raise ValueError('Expected regular file: ' + str(p))
    if p.stat().st_size > MAX_FILE:
        raise ValueError('Source exceeds size cap: ' + str(p))
    data = p.read_bytes()
    if len(data) > MAX_FILE:
        raise ValueError('Source changed size during read')
    return data


def check_text(data, label):
    text = data.decode('utf-8')
    if '\x00' in text or SECRET_RE.search(text):
        raise ValueError('Binary or credential-like content requires review: ' + label)
    return text


def clean_notebook(data, source_name):
    obj = json.loads(data)
    if obj.get('nbformat') != 4 or not isinstance(obj.get('cells'), list):
        raise ValueError('Unsupported notebook: ' + source_name)
    cells = []
    for old in obj['cells']:
        kind = old.get('cell_type')
        if kind not in {'code', 'markdown', 'raw'}:
            raise ValueError('Unknown cell type')
        source = old.get('source', [])
        text = ''.join(source) if isinstance(source, list) else source
        check_text(text.encode(), source_name)
        # Markdown embedded binary attachments are not published.
        text = re.sub(r'!\[[^\]]*\]\((?:data:|attachment:)[^)]*\)',
                      '[Private embedded image omitted from public source copy.]', text)
        cell = {'cell_type': kind, 'id': sha((source_name+':'+str(len(cells))).encode())[:12],
                'metadata': {}, 'source': text.splitlines(True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    cells.insert(0, {'cell_type': 'markdown', 'id': sha((source_name+':banner').encode())[:12],
                     'metadata': {}, 'source': [
        '# Archived experiment notebook\n',
        '\nSource preserved for review. Private outputs, attachments, and execution counts '
        'are removed in this public copy; original AWS notebooks remain untouched. '
        'This copy is not evidence of a fresh execution. Use the public Research Review '
        'for aggregate results. Private input contracts and artifacts are required to '
        'reproduce this historical experiment.\n']})
    return packed({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'publication': {'source_name': source_name, 'original_sha256': sha(data),
                        'outputs_removed': True, 'execution_claim': 'not_executed_for_publication'}},
        'cells': cells})


def event(name, **kw):
    row = {'utc': now(), 'event': name, **kw}
    print(json.dumps(row, allow_nan=False), flush=True)


def command(args, cwd=None, limit=60, check=True, interactive=False, input_text=None):
    """Bounded subprocess; never prints credentials, URLs with tokens, or environment."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GH_PROMPT_DISABLED='1',
               PYTHONDONTWRITEBYTECODE='1', GH_PAGER='cat', GIT_PAGER='cat')
    if interactive:
        env.pop('GH_PROMPT_DISABLED', None)
        return subprocess.run(args, cwd=cwd, env=env, timeout=limit).returncode
    process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    start = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = process.communicate(input_text, timeout=15)
                break
            except subprocess.TimeoutExpired:
                input_text = None
                event('publication_heartbeat', command=Path(str(args[0])).name,
                      elapsed_seconds=round(time.monotonic()-start, 1))
                if time.monotonic()-start >= limit:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                    raise TimeoutError('Publication command budget exceeded; completed work retained')
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
        raise
    if check and process.returncode:
        msg = ANSI.sub('', stderr + '\n' + stdout)
        msg = SECRET_RE.sub('[REDACTED]', msg)
        raise RuntimeError(f'{Path(str(args[0])).name} exited {process.returncode}: {msg[-3500:]}')
    return process.returncode, stdout, stderr


def git(*args, cwd=ORIGINAL, check=True):
    return command(['git', *args], cwd=cwd, check=check)[1].strip()


def find_gh():
    p = shutil.which('gh')
    if not p:
        candidate = HOME / '.local/bin/gh'
        p = str(candidate) if candidate.is_file() else None
    if not p:
        raise RuntimeError('GitHub CLI missing. Run: python3 nfl_workspace.zip github-login')
    return p


def gh(*args, check=True, cwd=None):
    return command([find_gh(), *args], cwd=cwd, check=check)


def api(endpoint):
    return json.loads(gh('api', endpoint)[1])


def network_bytes(url, cap):
    request = urllib.request.Request(url, headers={'User-Agent': 'nfl-publication/1'})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read(cap+1)
    if len(body) > cap: raise ValueError('Download exceeds declared limit')
    return body


def github_login():
    try:
        exe = find_gh()
    except RuntimeError:
        machine = platform.machine()
        arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(machine)
        if platform.system() != 'Linux' or arch is None:
            raise RuntimeError('Automatic CLI setup supports Linux amd64/arm64 only')
        release = json.loads(network_bytes('https://api.github.com/repos/cli/cli/releases/latest', 2*1024**2))
        version = release['tag_name'].lstrip('v')
        name = f'gh_{version}_linux_{arch}.tar.gz'
        assets = {a['name']: a for a in release['assets']}
        asset = assets[name]
        checksum = assets[f'gh_{version}_checksums.txt']
        for a in (asset, checksum):
            if not a['browser_download_url'].startswith('https://github.com/cli/cli/releases/download/'):
                raise ValueError('Unexpected CLI release asset source')
        checks = network_bytes(checksum['browser_download_url'], 512*1024).decode()
        lines = [line.split()[0] for line in checks.splitlines() if line.split()[-1].lstrip('*') == name]
        if len(lines) != 1: raise ValueError('CLI checksum missing or ambiguous')
        payload = network_bytes(asset['browser_download_url'], 45*1024**2)
        if sha(payload) != lines[0]: raise ValueError('GitHub CLI download checksum mismatch')
        target = HOME / '.local/bin/gh'
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
            candidates = [m for m in tar.getmembers() if m.name.endswith('/bin/gh') and m.isfile()]
            if len(candidates) != 1: raise ValueError('Unexpected CLI archive layout')
            binary = tar.extractfile(candidates[0]).read()
        atomic(target, binary)
        target.chmod(0o755)
        exe = str(target)
        event('github_cli_installed', version=version, location=str(target), ml_environment_changed=False)
    # Browser/device-code authorization. No password or token is requested in ChatGPT.
    status = command([exe, 'auth', 'status', '--hostname', 'github.com'], check=False)[0]
    if status:
        code = command([exe, 'auth', 'login', '--hostname', 'github.com', '--git-protocol',
                        'https', '--web', '--scopes', 'workflow'], interactive=True, limit=600)
        if code: raise RuntimeError('GitHub browser authorization did not complete')
    headers=command([exe,'api','--include','user'],check=False)[1]
    scope=re.search(r'(?im)^x-oauth-scopes:\s*(.*)$',headers)
    if scope and 'workflow' not in scope.group(1).split(', '):
        code=command([exe,'auth','refresh','--hostname','github.com','--scopes','workflow'],interactive=True,limit=600)
        if code:raise RuntimeError('Workflow permission refresh did not complete')
    user = api('user')
    repo = api('repos/' + REPO)
    if not repo.get('permissions', {}).get('push'):
        raise PermissionError('Authenticated account cannot push to the requested repository')
    event('github_publication_access_ready', login=user['login'], repository=REPO,
          note='Workflow-file changes require workflow permission; no credentials are exported')


def git_network(*args, cwd=ORIGINAL):
    # Authenticate HTTPS per command; do not change global Git configuration.
    helper = '!' + shlex.quote(find_gh()) + ' auth git-credential'
    return command(['git', '-c', 'credential.helper=', '-c', 'credential.helper='+helper,
                    *args], cwd=cwd, limit=90)[1].strip()


def resources(name):
    # Works when this module is loaded directly from nfl_workspace.zip.
    data = __loader__.get_data(str(Path(__file__).parent / 'publication_payload' / name))
    return data


def package_integrity():
    manifest_path=str(Path(__file__).parent/'PACKAGE_MANIFEST.json')
    raw=__loader__.get_data(manifest_path)
    manifest=json.loads(raw)
    for name,expected in manifest['files'].items():
        safe_rel(name)
        data=__loader__.get_data(str(Path(__file__).parent/name))
        if sha(data)!=expected:
            raise ValueError('Maintenance ZIP checksum mismatch: '+name)
    return sha(raw)


def state_load():
    p = STATE / 'active.json'
    if not p.is_file(): raise FileNotFoundError('Run publication-plan first')
    d = read_json(p)
    if d['version'] != VERSION: raise ValueError('Publication package changed; retain the existing plan for review')
    return d


def save_state(d):
    atomic(STATE / 'active.json', packed(d))


def source_selection(folder):
    for root, dirs, filenames in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in PRIVATE_PARTS and not d.startswith('.'))
        for filename in sorted(filenames):
            p = Path(root) / filename
            rel = p.relative_to(folder)
            if p.is_symlink(): raise ValueError('Unexpected source symlink: '+str(p))
            include = (p.suffix in {'.py', '.ipynb'} or filename.endswith('.py.lock')
                       or filename in DOC_NAMES or filename in {'FEATURE_DICTIONARY.csv', 'pyproject.toml',
                       'uv.lock', 'requirements.txt', '.python-version'})
            if include:
                yield p, rel


def public_summary(d):
    """Strict aggregate allowlist. No recursive export of arbitrary private dictionaries."""
    out = {}
    for key in ('status','round','training_rows','evaluation_rows','evaluation_games','feature_research',
        'models_total','scientific_models_total','new_scientific_models','new_kaggle_score',
        'core_ready_for_replication','full_ready_for_replication','ready_for_later_fold',
        'conditional_feature_evidence','feature_replication_gate_passed','protocol_signature'):
        if key in d and (d[key] is None or isinstance(d[key], (str, int, float, bool))): out[key] = d[key]
    for key in ('metrics','training_metrics','pooled_rmse','pooled_metrics'):
        if key not in d or not isinstance(d[key], dict): continue
        metrics = {}
        for arm, item in d[key].items():
            if isinstance(item, (int,float)) and not isinstance(item,bool):
                if not math.isfinite(item): raise ValueError('Nonfinite aggregate')
                metrics[arm] = item
            elif isinstance(item,dict):
                m = {k:item[k] for k in ('rmse','rows','sse') if k in item}
                if not m: continue
                if all(k in m for k in ('rmse','rows','sse')):
                    if m['rows'] <= 0 or not math.isclose(m['rmse'], math.sqrt(m['sse']/(2*m['rows'])), abs_tol=1e-9):
                        raise ValueError('Reported RMSE arithmetic failed')
                metrics[arm] = m
        out[key] = metrics
    for key in ('contrast','comparison','contrasts'):
        if key not in d or not isinstance(d[key],dict): continue
        keep = ('delta_rmse','relative_gain','low95','high95','low_adjusted','high_adjusted',
                'passes','feature_gate_passed','resamples','planned_family_comparisons')
        if key in ('contrast','comparison'):
            out[key] = {k:d[key][k] for k in keep if k in d[key]}
        else:
            out[key] = {name:{k:item[k] for k in keep if k in item}
                        for name,item in d[key].items() if isinstance(item,dict)}
    if isinstance(d.get('slices'),list):
        out['slices']=[{k:item[k] for k in ('arm','slice','rows','rmse','sse') if k in item}
                       for item in d['slices'] if isinstance(item,dict)
                       and not re.search(r'(?:game|nfl_id|\d{7,})',str(item.get('slice','')),re.I)]
    return out


def collect_evidence():
    studies = []
    private_hashes = {}
    for n in range(1,18):
        root = HOME / f'nfl-feature-round{n}-results'
        if not root.is_dir(): continue
        summaries = [root/'summary.json']
        summaries += sorted(root.glob('fold_*/summary.json'))
        summaries += [root/'attribution/summary.json',root/'motion/summary.json']
        found = False
        for p in summaries:
            if not p.is_file(): continue
            raw = regular(p, root)
            d = json.loads(raw)
            record = public_summary(d)
            if not record: continue
            found = True
            record['round'] = n
            record['evidence_id'] = f'round{n}/'+p.relative_to(root).as_posix()
            record['original_summary_sha256'] = sha(raw)
            record['verification'] = 'aggregate arithmetic; saved replay is a source claim, not fresh model replay'
            private_hashes[str(p)] = sha(raw)
            studies.append(record)
        if not found:
            record = {'round':n,'evidence_id':f'round{n}/readiness','status':'no_published_accuracy_summary',
                      'new_kaggle_score':None}
            for name in ('preflight.json','smoke.json','tests_receipt.json','replay.json'):
                p=root/name
                if p.is_file():
                    raw=regular(p,root);j=json.loads(raw);private_hashes[str(p)]=sha(raw)
                    record[name.removesuffix('.json')] = {k:j[k] for k in ('status','plays','tests_run',
                        'failures','errors','skipped','new_optimizer_steps','new_model_fits') if k in j}
            studies.append(record)
    if not studies: raise ValueError('No current experiment receipts were found; refusing invented publication')
    label=HOME/'nfl-feature-round14-results/label_readiness.json'
    labels={}
    if label.is_file():
        raw=regular(label);d=json.loads(raw);private_hashes[str(label)]=sha(raw)
        labels={k:d[k] for k in ('status','training_games','selected_training_plays',
            'observed_training_plays','label_eligible_training_plays','additional_label_eligible_plays') if k in d}
    return {'schema':1,'source':'local AWS receipts collected during publication-plan',
            'metric':'sqrt(sum(dx^2+dy^2)/(2N)); yards; lower is better',
            'comparison_limit':'Reused internal game splits. No new Kaggle score or model replay by publication.',
            'studies':studies,'label_readiness':labels},private_hashes


def put(root, name, data):
    path = root / safe_rel(name)
    check_text(data, name)
    atomic(path, data)


def original_state():
    if not (ORIGINAL/'.git').exists(): raise ValueError('Expected existing NFL checkout is missing')
    origin=git('remote','get-url','origin')
    accepted={URL,'git@github.com:'+REPO+'.git','https://github.com/'+REPO}
    if origin not in accepted: raise ValueError('Original remote is not the requested NFL repository')
    if git('diff','--cached','--name-only'): raise ValueError('Original checkout has staged work; preserve it and finish its review first')
    return {'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
            'status':git('status','--porcelain=v1','--untracked-files=normal')}


# EOF formatting in immutable historical Python must not rewrite scientific bytes.
# All other diff --check errors (including conflict markers) remain blockers.
EOF_POLICY = 'byte-identical-archive-python-eof-v1'
LEGACY_PUBLICATION_SHA256 = '0c31c012d13359cd65b2e8cc3accd64cd961d5af028e947dc1c856d6d003f5d9'


# Git ignore rules remain unchanged. Only these generated public JSON files may
# be explicitly staged when ignored; a directory or private artifact is never forced.
STAGING_POLICY = 'exact-reviewed-public-files-v1'
PUBLIC_EVIDENCE_PATHS = frozenset({
    'research/evidence/studies.json',
    'research/evidence/publication_manifest.json',
})
PREVIOUS_EOF_HELPER_SHA256 = '4b6892b1bb35ed7014ed1bd5e1f7de15421b95743c84bbfe1ea91afccd186cff'
PREVIOUS_EOF_VALIDATOR_SHA256 = '526d703007963b78ccfa843cf8fc7fec573cd1ac36e303f49fe3255648bb9bb6'



# Scoped Git storage policy for immutable public source snapshots. This does not
# change the root attributes, ignore rules, the original checkout, or Git config.
STORAGE_POLICY = 'immutable-archive-bytes-v1'
ARCHIVE_ATTRIBUTES_PATH = 'research/workspace/.gitattributes'
ARCHIVE_ATTRIBUTES = (
    '# Immutable research snapshots: preserve the recorded bytes on add and checkout.\n'
    '# Only this archived-source subtree is affected; maintained code is unchanged.\n'
    '** -text -filter -ident -working-tree-encoding !eol '
    'whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n'
).encode('ascii')
PREVIOUS_STAGING_HELPER_SHA256 = 'd96bc16d7dc7c8963d7de85ed53c4e7cba8a1b4ad2f79a120dc0d6aed3df43ed'
PREVIOUS_STAGING_VALIDATOR_SHA256 = '526d703007963b78ccfa843cf8fc7fec573cd1ac36e303f49fe3255648bb9bb6'


def storage_policy_record():
    return {'policy': STORAGE_POLICY, 'attributes_file': ARCHIVE_ATTRIBUTES_PATH,
            'attributes_sha256': sha(ARCHIVE_ATTRIBUTES),
            'scope': 'research/workspace only', 'archive_bytes_rewritten': False,
            'root_attributes_changed': False, 'git_config_changed': False,
            'checkout_verification': 'public CI verifies raw hashes and indexed blob identities'}


def git_blob_id(data, algorithm):
    return hashlib.new(algorithm, b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def index_diagnostics(wt, expected):
    """Read ALL reviewed index entries; no filters are invoked and no file is changed.

    LF-normalized equivalence is a diagnosis only, never acceptance of a changed
    archive. Completion still requires exact raw bytes in disk, index and manifest.
    """
    algorithm = git('rev-parse', '--show-object-format', cwd=wt)
    if algorithm not in {'sha1', 'sha256'}:
        raise ValueError('Unsupported Git object format')
    entries = {}
    for entry in command(['git', 'ls-files', '--stage', '-z'], cwd=wt)[1].split('\0'):
        if not entry:
            continue
        header, name = entry.split('\t', 1)
        mode, oid, stage = header.split()
        if name in expected:
            entries.setdefault(name, []).append({'mode': mode, 'oid': oid, 'stage': stage})
    counts = {}; findings = []
    for name, digest in sorted(expected.items()):
        data = regular(wt / safe_rel(name), wt)
        raw_id = git_blob_id(data, algorithm)
        indexed = entries.get(name, [])
        normalized = data.replace(b'\r\n', b'\n')
        if sha(data) != digest:
            status = 'worktree_differs_from_manifest'
        elif not indexed:
            status = 'missing_from_index'
        elif len(indexed) != 1 or indexed[0]['stage'] != '0' or indexed[0]['mode'] not in {'100644','100755'}:
            status = 'conflicted_or_nonregular_index'
        elif indexed[0]['oid'] == raw_id:
            status = 'exact'
        elif normalized != data and indexed[0]['oid'] == git_blob_id(normalized, algorithm):
            status = 'index_is_crlf_to_lf_conversion'
        else:
            status = 'different_indexed_content'
        counts[status] = counts.get(status, 0) + 1
        if status != 'exact':
            findings.append({'path': name, 'status': status, 'expected_sha256': digest,
                'worktree_sha256': sha(data), 'worktree_bytes': len(data),
                'crlf_count': data.count(b'\r\n'), 'expected_raw_git_blob': raw_id,
                'lf_normalized_git_blob': git_blob_id(normalized, algorithm), 'index_entries': indexed})
    changed = set(command(['git', 'diff', '--cached', '--name-only', '-z'], cwd=wt)[1].split('\0')) - {''}
    return {'status': 'exact' if not findings and not (changed-set(expected)) else 'differences_found',
        'object_format': algorithm, 'reviewed_files': len(expected), 'counts': counts,
        'findings': findings, 'unreviewed_staged_paths': sorted(changed-set(expected)),
        'all_mismatches_reported': True, 'index_changed': False}


def archive_attribute_check(wt, expected):
    """Check the effective staged policy, including higher-precedence local overrides."""
    if ARCHIVE_ATTRIBUTES_PATH not in expected:
        return {'status': 'legacy_no_storage_policy'}
    if regular(wt / ARCHIVE_ATTRIBUTES_PATH, wt) != ARCHIVE_ATTRIBUTES:
        raise ValueError('Archive attribute file differs from the reviewed storage policy')
    paths = sorted(n for n in expected if n.startswith('research/workspace/'))
    attrs = ['text','filter','ident','working-tree-encoding','eol','whitespace']
    rc, out, err = command(['git','check-attr','--cached','-z','--stdin',*attrs], cwd=wt,
                          input_text='\0'.join(paths)+'\0', check=False)
    if rc or err.strip():
        raise ValueError('Cannot verify effective archive attributes: '+err[-1000:])
    parts = out.split('\0')
    if parts[-1] == '':
        parts.pop()
    if len(parts) % 3:
        raise ValueError('Malformed archive attribute response')
    wanted={'text':'unset','filter':'unset','ident':'unset','working-tree-encoding':'unset',
            'eol':'unspecified','whitespace':'blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol'}
    actual={}
    for name, attr, value in zip(parts[0::3],parts[1::3],parts[2::3]):
        if name not in paths or attr not in wanted or (name,attr) in actual:
            raise ValueError('Unexpected archive attribute response')
        actual[name,attr]=value
    for name in paths:
        for attr,value in wanted.items():
            if actual.get((name,attr)) != value:
                raise ValueError('Higher-precedence attribute conflicts with exact archive storage: '+name+' '+attr)
    return {'status':'archive_attributes_verified','files':len(paths), 'root_attributes_changed':False}


def archive_storage_update(run, wt):
    """One recoverable metadata update; never normalizes source or CSV content."""
    phase='archive-storage'
    transaction=run/(phase+'-transaction.json')
    if transaction.is_file():
        apply_publication_journal(run,wt,read_json(transaction),phase=phase)
        return
    index = read_json(wt/'research/publication/files.json')
    for name,digest in index.items():
        if sha(regular(wt/safe_rel(name),wt)) != digest:
            raise ValueError('Prepared publication changed before storage update: '+name)
    attr_path=wt/ARCHIVE_ATTRIBUTES_PATH
    if attr_path.exists() and regular(attr_path,wt) != ARCHIVE_ATTRIBUTES:
        raise ValueError('Preserve existing archive attribute edits; no automatic replacement')
    expected=expected_public_files(wt)
    check_public_staging_files(wt,expected)
    before=index_diagnostics(wt,expected)
    atomic(run/'index_before_storage.json',packed(before))
    if before['unreviewed_staged_paths']:
        raise ValueError('Unreviewed staged paths exist before archive storage update')
    # Prior staged metadata may be at a recorded transaction boundary. Only those
    # explicit old blob hashes are accepted as recoverable preimages.
    known={}
    for old_phase in ('eof-policy','evidence-stage'):
        tx=run/(old_phase+'-transaction.json')
        if not tx.is_file():
            continue
        journal=read_json(tx)
        for item in journal.get('items',[]):
            for field,hash_field in (('before_file','before_sha256'),('after_file','after_sha256')):
                value=item.get(field)
                if not value:
                    continue
                rel=safe_rel(value)
                if len(rel.parts)!=1:
                    raise ValueError('Unsafe recorded metadata preimage')
                data=regular(run/old_phase/rel,run/old_phase)
                if sha(data)!=item[hash_field]:
                    raise ValueError('Recorded metadata preimage hash differs')
                known.setdefault(item['path'],set()).add(git_blob_id(data,before['object_format']))
    for item in before['findings']:
        if item['status'] in {'missing_from_index','index_is_crlf_to_lf_conversion'}:
            continue
        if (item['status']=='different_indexed_content' and len(item['index_entries'])==1
                and item['index_entries'][0]['oid'] in known.get(item['path'],set())):
            continue
        raise ValueError('Unrecognized staged change; details are in index_before_storage.json: '+item['path'])
    mirror=read_json(wt/'research/evidence/publication_manifest.json')
    mirror['byte_storage_policy']=storage_policy_record()
    replacements={ARCHIVE_ATTRIBUTES_PATH: ARCHIVE_ATTRIBUTES,
        'research/publication/tool_source/publication.py':__loader__.get_data(__file__),
        'research/publication/validate.py':resources('validate.py'),
        'research/evidence/publication_manifest.json':packed(mirror)}
    index.update({n:sha(b) for n,b in replacements.items()})
    replacements['research/publication/files.json']=packed(index)
    journal=publication_update_journal(run,wt,replacements,phase=phase)
    apply_publication_journal(run,wt,journal,phase=phase)
    event('publication_storage_policy_prepared',diagnosis_counts=before['counts'],
          archive_scope='research/workspace',archive_bytes_rewritten=False)


def expected_public_files(wt):
    """Return only exact files from the prepared publication and its provenance."""
    wt = Path(wt)
    raw_index = regular(wt / 'research/publication/files.json', wt)
    expected = json.loads(raw_index)
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Prepared public-file map is empty or invalid')
    expected = dict(expected)
    expected['research/publication/files.json'] = sha(raw_index)
    mirror = read_json(wt / 'research/evidence/publication_manifest.json')
    for item in mirror['source_archives']:
        name = item['path']
        if name in expected and expected[name] != item['published_sha256']:
            raise ValueError('Public-file and source-provenance hashes disagree: ' + name)
        expected[name] = item['published_sha256']
    # Before-images of the historical front pages are separately checked against HEAD.
    for filename in ('README.md', 'START_HERE.md'):
        name = 'docs/archive/' + filename
        digest = sha(regular(wt / name, wt))
        if name in expected and expected[name] != digest:
            raise ValueError('Historical front-page hash differs: ' + name)
        expected[name] = digest
    return expected


def check_public_staging_files(wt, expected):
    """Check all files before staging; do not turn ignore rules off globally."""
    if not isinstance(expected, dict) or not expected:
        raise ValueError('An explicit nonempty file/hash allowlist is required')
    if not PUBLIC_EVIDENCE_PATHS.issubset(expected):
        raise ValueError('Both public evidence files must be in the publication allowlist')
    actual_evidence = set()
    folder = wt / 'research/evidence'
    if folder.is_symlink():
        raise ValueError('Evidence directory must not be a symlink')
    for path in folder.rglob('*'):
        if path.is_symlink():
            raise ValueError('Evidence symlink requires review')
        if path.is_file():
            actual_evidence.add(path.relative_to(wt).as_posix())
    if actual_evidence != PUBLIC_EVIDENCE_PATHS:
        raise ValueError('Unexpected or missing files in public evidence: '
                         + repr(sorted(actual_evidence ^ PUBLIC_EVIDENCE_PATHS)))
    for name, digest in expected.items():
        relative = safe_rel(name)
        if relative.as_posix() != name or any(ord(c) < 32 for c in name):
            raise ValueError('Noncanonical publication path')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid publication digest: ' + name)
        if name.startswith('research/evidence/') and name not in PUBLIC_EVIDENCE_PATHS:
            raise ValueError('Only the two approved aggregate evidence files may be published')
        data = regular(wt / relative, wt)
        if sha(data) != digest:
            raise ValueError('Prepared bytes changed before staging: ' + name)
        if name in PUBLIC_EVIDENCE_PATHS:
            value = json.loads(check_text(data, name))
            if not isinstance(value, dict):
                raise ValueError('Public evidence must be a JSON object: ' + name)
            packed(value)  # reject NaN/Infinity rather than publishing invalid JSON


def verify_staged_publication(wt, expected):
    """Exact byte checks over the entire reviewed set, never only the first file."""
    check_public_staging_files(wt, expected)
    result=index_diagnostics(wt,expected)
    if result['findings']:
        first=result['findings'][0]
        raise ValueError('Missing or changed staged publication bytes: '+first['path']+
            ' ['+first['status']+'; '+str(len(result['findings']))+
            ' mismatched files total; see index_diagnostics.json in publication-report]')
    if result['unreviewed_staged_paths']:
        raise ValueError('Unreviewed staged file: '+repr(result['unreviewed_staged_paths']))
    archive_attribute_check(wt,expected)
    return len(expected)


def stage_reviewed_publication(wt):
    """Stage exact files, including only allowlisted ignored aggregate evidence.

    No directory pathspecs, wildcard expansion, blanket force-add, or ignore/config
    changes. Completion verifies every reviewed file in the index, not just disk.
    """
    wt = Path(wt)
    expected = expected_public_files(wt)
    check_public_staging_files(wt, expected)
    changed = set(command(['git', 'diff', '--cached', '--name-only', '-z'], cwd=wt)[1].split('\0')) - {''}
    if changed - set(expected):
        raise ValueError('Unreviewed staged paths exist; no additional files staged')
    paths = sorted(expected)
    rc, stdout, stderr = command(
        ['git', 'check-ignore', '--no-index', '--stdin', '-z'], cwd=wt,
        check=False, input_text='\0'.join(paths) + '\0')
    if rc not in (0, 1) or stderr.strip():
        raise RuntimeError('Cannot inspect ignore policy; no files staged: '
                           + SECRET_RE.sub('[REDACTED]', stderr)[-1500:])
    ignored = set(stdout.split('\0')) - {''}
    if (rc == 0 and not ignored) or (rc == 1 and ignored) or ignored - set(expected):
        raise ValueError('Unexpected ignore-query response; no files staged')
    if ignored - PUBLIC_EVIDENCE_PATHS:
        raise ValueError('An ignored file outside the two public evidence files needs review: '
                         + repr(sorted(ignored - PUBLIC_EVIDENCE_PATHS)))
    # Standard input avoids argument-size limits; --literal-pathspecs prevents
    # archive filenames containing brackets/asterisks from matching other files.
    for names, forced in (([n for n in paths if n not in ignored], False),
                          (sorted(ignored), True)):
        if not names:
            continue
        args = ['git', '--literal-pathspecs', 'add']
        if forced:
            args.append('--force')
        args += ['--pathspec-from-file=-', '--pathspec-file-nul']
        command(args, cwd=wt, input_text='\0'.join(names) + '\0')
    if ARCHIVE_ATTRIBUTES_PATH in expected:
        # Reapply the new archive attributes even when Git's cached stat information
        # would otherwise let a previously normalized tracked CSV skip re-hashing.
        command(['git', '--literal-pathspecs', 'add', '--renormalize',
                 '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=wt,
                input_text='\0'.join(paths)+'\0')
    count = verify_staged_publication(wt, expected)
    result = {'policy': STAGING_POLICY, 'status': 'reviewed_files_staged',
              'reviewed_files': count, 'all_manifest_files_present_in_index': True,
              'staged_bytes_match_reviewed_files': True,
              'ignored_files_explicitly_staged': [{'path': n, 'sha256': expected[n]} for n in sorted(ignored)],
              'ignore_rules_changed': False, 'blanket_force_add': False,
              'scientific_source_changes': False}
    event('publication_files_staged', files=count, ignored_public_files=len(ignored),
          staged_bytes_verified=True, blanket_force_add=False)
    return result


def staging_metadata_update(run, wt):
    """A second before-image journal upgrades the helper, not the scientific archive."""
    phase = 'evidence-stage'
    transaction = run / (phase + '-transaction.json')
    if transaction.is_file():
        apply_publication_journal(run, wt, read_json(transaction), phase=phase)
        return
    index = read_json(wt / 'research/publication/files.json')
    for name, digest in index.items():
        if sha(regular(wt / safe_rel(name), wt)) != digest:
            raise ValueError('Publication changed before staging-policy update: ' + name)
    manifest = read_json(wt / 'research/evidence/publication_manifest.json')
    manifest['staging_policy'] = {
        'policy': STAGING_POLICY,
        'allowed_ignored_files': sorted(PUBLIC_EVIDENCE_PATHS),
        'all_files_staged_by_exact_literal_path': True,
        'all_manifest_files_verified_in_git_index': True,
        'gitignore_changed_by_correction': False,
    }
    replacements = {
        'research/publication/tool_source/publication.py': __loader__.get_data(__file__),
        'research/evidence/publication_manifest.json': packed(manifest),
    }
    index.update({name: sha(data) for name, data in replacements.items()})
    replacements['research/publication/files.json'] = packed(index)
    journal = publication_update_journal(run, wt, replacements, phase=phase)
    apply_publication_journal(run, wt, journal, phase=phase)


def require_selftest():
    path = STATE / 'selftest.json'
    if not path.is_file():
        raise ValueError('Run publication-self-test with the replacement ZIP first')
    receipt = read_json(path)
    if (not receipt.get('passed') or not receipt.get('tests_run')
            or receipt.get('failures') != 0 or receipt.get('errors') != 0
            or receipt.get('skipped', 0) != 0
            or receipt.get('source_sha256') != sha(__loader__.get_data(__file__))):
        raise ValueError('A passing publication-self-test for this exact helper is required')


def staged_whitespace_review(wt):
    """Allow only hash-verified, pre-existing EOF blanks in archived Python.

    Git still inspects every staged path for whitespace and conflict markers.
    No global/local git config, worktree source file, or archived byte is changed.
    """
    wt = Path(wt)
    rc, stdout, stderr = command([
        'git', '-c', 'color.ui=false', '-c', 'core.quotePath=false',
        '-c', 'core.whitespace=blank-at-eol,space-before-tab,blank-at-eof',
        'diff', '--cached', '--check', '--no-ext-diff',
    ], cwd=wt, check=False)
    base = {'policy': EOF_POLICY, 'conflict_markers_checked': True,
            'global_git_configuration_changed': False,
            'source_bytes_changed': False, 'exceptions': []}
    if rc == 0 and not stdout.strip() and not stderr.strip():
        return dict(base, status='strict_whitespace_check_passed')
    if rc != 2 or stderr.strip() or not stdout.strip():
        raise RuntimeError('Git diff validation failed; no formatting exception applies: '
                           + SECRET_RE.sub('[REDACTED]', stderr + stdout)[-3500:])
    manifest = read_json(wt / 'research/evidence/publication_manifest.json')
    mirrors = {}
    for item in manifest.get('source_archives', []):
        if item['path'] in mirrors:
            raise ValueError('Duplicate source provenance path')
        mirrors[item['path']] = item
    seen = set()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([^\r\n]+):([1-9][0-9]*): new blank line at EOF\.', line)
        if match is None:
            raise ValueError('A non-EOF whitespace or conflict-marker error remains: '
                             + SECRET_RE.sub('[REDACTED]', line))
        name, number = match.groups()
        relative = safe_rel(name)
        item = mirrors.get(name)
        if (not name.startswith('research/workspace/') or relative.suffix != '.py'
                or item is None or item.get('transformation') != 'byte-identical source'
                or item.get('source_sha256') != item.get('published_sha256')):
            raise ValueError('EOF exception refused outside byte-identical archived Python: ' + name)
        raw = regular(wt / relative, wt)
        if sha(raw) != item['published_sha256']:
            raise ValueError('Archive differs from its source hash: ' + name)
        # Worktree bytes alone do not establish what Git is about to commit.
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if git('rev-parse', ':' + name, cwd=wt) != blob:
            raise ValueError('Staged archive bytes differ from the reviewed file: ' + name)
        lines = raw.splitlines()
        if not lines or lines[-1].strip(b' \t\r'):
            raise ValueError('EOF diagnostic not supported by the archived file: ' + name)
        if name in seen:
            raise ValueError('Repeated unexpected EOF diagnostic: ' + name)
        seen.add(name)
        base['exceptions'].append({'path': name, 'line': int(number),
            'sha256': sha(raw), 'reason': 'Original archived Python ends with a blank line; bytes retained'})
    if not base['exceptions']:
        raise ValueError('Git failed without an eligible archive EOF diagnostic')
    return dict(base, status='passed_with_documented_archive_eof')


def plan_paths(d):
    """Constrain a saved operation to its existing linked worktree."""
    if d.get('repository') != REPO or d.get('version') != VERSION:
        raise ValueError('Existing publication belongs to a different repository or format')
    run, wt = Path(d['run_dir']), Path(d['worktree'])
    if run.parent != STATE or wt != run / 'worktree':
        raise ValueError('Unexpected publication worktree location')
    for p in (STATE, run, wt):
        if p.is_symlink() or not p.is_dir() or p.resolve() != p:
            raise ValueError('Publication worktree location is missing or indirect')
    if (git('rev-parse', '--show-toplevel', cwd=wt) != str(wt)
            or git('rev-parse', 'HEAD', cwd=wt) != d['base']
            or git('branch', '--show-current', cwd=wt) != d['branch']):
        raise ValueError('Saved publication branch moved; no implicit reset or recommit')
    if git('rev-parse', '--git-common-dir', cwd=wt) != git('rev-parse', '--git-common-dir', cwd=ORIGINAL):
        # Git can return a relative common-dir in the original checkout.
        a = (wt / git('rev-parse', '--git-common-dir', cwd=wt)).resolve()
        b = (ORIGINAL / git('rev-parse', '--git-common-dir', cwd=ORIGINAL)).resolve()
        if a != b:
            raise ValueError('Saved worktree does not belong to the NFL checkout')
    if original_state() != d['original']:
        raise ValueError('Original research checkout changed; preserve it and review before publishing')
    return run, wt


def verify_saved_public_files(wt):
    index = read_json(wt / 'research/publication/files.json')
    if not isinstance(index, dict) or not index:
        raise ValueError('Saved public-file hash map is missing or empty')
    for name, expected in index.items():
        path = wt / safe_rel(name)
        if not isinstance(expected, str) or sha(regular(path, wt)) != expected:
            raise ValueError('Prepared publication changed: ' + name)
    actual = {p.relative_to(wt).as_posix() for p in (wt / 'research').rglob('*') if p.is_file()}
    if actual - set(index) - {'research/publication/files.json'}:
        raise ValueError('Unreviewed files appeared in the prepared research publication')
    expected_staged = set(index) | {'research/publication/files.json',
        'docs/archive/README.md', 'docs/archive/START_HERE.md'}
    mirror = read_json(wt / 'research/evidence/publication_manifest.json')
    expected_staged.update(item['path'] for item in mirror['source_archives'])
    changed = set(git('diff', '--cached', '--name-only', cwd=wt).splitlines())
    if not changed or changed - expected_staged:
        raise ValueError('Unexpected staged publication paths; no commit performed')
    if git('diff', '--cached', '--diff-filter=D', '--name-only', cwd=wt):
        raise ValueError('Unexpected staged deletions; no deletion is required for this correction')
    pending = set(git('diff', '--name-only', cwd=wt).splitlines()) | set(git('ls-files', '--others', '--exclude-standard', cwd=wt).splitlines())
    if pending - expected_staged:
        raise ValueError('Publication worktree has additional edits/files; preserve them for review')
    # These two historical front pages were not in the original hash map.
    for filename in ('README.md', 'START_HERE.md'):
        path = wt / 'docs/archive' / filename
        raw = regular(path, wt)
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != git('rev-parse', 'HEAD:' + filename, cwd=wt):
            raise ValueError('Saved historical front page differs from publication base: ' + filename)
    return index, mirror


def live_fingerprints_from_publication(wt, manifest):
    """Recover old-plan fingerprints without recopying or rerendering its content."""
    result, discovered, recorded = {}, set(), set()
    folders = set()
    for item in manifest['source_archives']:
        name = item['path']
        relative = safe_rel(name)
        if name.startswith('research/workspace/'):
            if len(relative.parts) < 4:
                raise ValueError('Archive path is not a source file')
            dirname = relative.parts[2]
            if dirname != 'nfl_manual_sync' and re.fullmatch(r'nfl_feature_rounds?\d+(?:_\d+)?', dirname) is None:
                raise ValueError('Unknown archived source directory')
            source = HOME.joinpath(*relative.parts[2:])
            root = HOME / dirname
            folders.add(dirname)
            recorded.add(name)
        else:
            if not name.startswith(('src/', 'scripts/', 'tests/', 'kaggle/', 'notebooks/')):
                raise ValueError('Unexpected integrated source path')
            source, root = ORIGINAL / relative, ORIGINAL
        raw = regular(source, root)
        if sha(raw) != item['source_sha256']:
            raise ValueError('Live source changed after failed preparation: ' + str(source))
        result[str(source)] = sha(raw)
    for dirname in folders:
        for path, rel in source_selection(HOME / dirname):
            discovered.add('research/workspace/' + dirname + '/' + rel.as_posix())
    if discovered != recorded:
        raise ValueError('Source inventory changed after failed preparation; no hidden recapture')
    evidence, hashes = collect_evidence()
    if packed(evidence) != regular(wt / 'research/evidence/studies.json', wt):
        raise ValueError('Aggregate study evidence changed after failed preparation')
    result.update(hashes)
    return result


def publication_update_journal(run, wt, replacements, phase='eof-policy'):
    """Save exact before/after bytes before touching the isolated public view."""
    if phase not in {'eof-policy', 'evidence-stage', 'archive-storage'}:
        raise ValueError('Unknown publication transaction phase')
    policy = {'eof-policy':EOF_POLICY,'evidence-stage':STAGING_POLICY,'archive-storage':STORAGE_POLICY}[phase]
    root = run / phase
    root.mkdir(exist_ok=True)
    items = []
    for number, (name, content) in enumerate(sorted(replacements.items())):
        path = wt / safe_rel(name)
        before = regular(path, wt) if path.exists() else None
        old_name, new_name = f'{number}.before', f'{number}.after'
        if before is not None:
            atomic(root / old_name, before)
        atomic(root / new_name, content)
        items.append({'path': name, 'before_sha256': sha(before) if before is not None else None,
            'after_sha256': sha(content), 'before_file': old_name if before is not None else None,
            'after_file': new_name})
    journal = {'schema': 1, 'policy': policy, 'created_utc': now(),
        'helper_sha256': sha(__loader__.get_data(__file__)), 'items': items}
    atomic(run / (phase + '-transaction.json'), packed(journal))
    return journal


def apply_publication_journal(run, wt, journal, phase='eof-policy'):
    allowed = {'research/publication/tool_source/publication.py',
        'research/publication/validate.py', 'research/publication/files.json',
        'research/publication/whitespace_review.json', 'research/evidence/publication_manifest.json'}
    if phase not in {'eof-policy', 'evidence-stage', 'archive-storage'}:
        raise ValueError('Unknown publication transaction phase')
    policy = {'eof-policy':EOF_POLICY,'evidence-stage':STAGING_POLICY,'archive-storage':STORAGE_POLICY}[phase]
    if phase == 'evidence-stage':
        allowed = {'research/publication/tool_source/publication.py',
                   'research/publication/files.json', 'research/evidence/publication_manifest.json'}
    if phase == 'archive-storage':
        allowed = {'research/publication/tool_source/publication.py',
            'research/publication/validate.py', 'research/publication/files.json',
            'research/evidence/publication_manifest.json', ARCHIVE_ATTRIBUTES_PATH}
    helper = journal.get('helper_sha256')
    if journal.get('policy') != policy:
        raise ValueError('Unexpected recovery policy')
    current = sha(__loader__.get_data(__file__))
    if helper != current:
        accepted = {'eof-policy': {PREVIOUS_EOF_HELPER_SHA256, PREVIOUS_STAGING_HELPER_SHA256},
                    'evidence-stage': {PREVIOUS_STAGING_HELPER_SHA256}, 'archive-storage': set()}
        if helper not in accepted[phase]:
            raise ValueError('Unknown publication recovery source')
        expected_payloads = {'research/publication/tool_source/publication.py': helper}
        if phase == 'eof-policy':
            expected_payloads['research/publication/validate.py'] = (
                PREVIOUS_EOF_VALIDATOR_SHA256 if helper == PREVIOUS_EOF_HELPER_SHA256
                else PREVIOUS_STAGING_VALIDATOR_SHA256)
        items = {item['path']: item for item in journal['items']}
        for name, digest in expected_payloads.items():
            item = items.get(name)
            if item is None or item.get('after_sha256') != digest:
                raise ValueError('Legacy recovery payload does not match its known source')
    names = [item['path'] for item in journal['items']]
    if set(names) != allowed or len(names) != len(allowed):
        raise ValueError('Unexpected publication recovery write scope')
    pending = []
    for item in journal['items']:
        path = wt / safe_rel(item['path'])
        rel = safe_rel(item['after_file'])
        if len(rel.parts) != 1:
            raise ValueError('Unsafe recovery payload location')
        content = regular(run / phase / rel, run / phase)
        if sha(content) != item['after_sha256']:
            raise ValueError('Recovery payload hash mismatch')
        current = sha(regular(path, wt)) if path.exists() else None
        if current not in (item['before_sha256'], item['after_sha256']):
            raise ValueError('Concurrent or unknown publication edit: ' + item['path'])
        if current != item['after_sha256']:
            pending.append((path, content))
    # Validate every destination before the first write; only after-images are applied.
    for path, content in pending:
        atomic(path, content)


def resume_prepared_plan(d, fingerprints=None):
    """Resume whitespace/ignored-evidence stops in place; no new worktree/render."""
    require_selftest()
    if d.get('status') != 'preparing' or 'commit' in d:
        raise ValueError('Only an uncommitted preparing publication can use this recovery')
    run, wt = plan_paths(d)
    transaction = run / 'eof-policy-transaction.json'
    resumed = fingerprints is None
    staging_transaction = run / 'evidence-stage-transaction.json'
    storage_transaction = run / 'archive-storage-transaction.json'
    if storage_transaction.is_file() or staging_transaction.is_file() or transaction.is_file():
        fingerprints = read_json(run / 'source_hashes.json')
        for path, digest in fingerprints.items():
            if sha(regular(Path(path))) != digest:
                raise ValueError('Live source/evidence changed during recovery: ' + path)
        if storage_transaction.is_file():
            apply_publication_journal(run, wt, read_json(storage_transaction), phase='archive-storage')
        elif staging_transaction.is_file():
            apply_publication_journal(run, wt, read_json(staging_transaction), phase='evidence-stage')
        else:
            apply_publication_journal(run, wt, read_json(transaction))
    else:
        index, manifest = verify_saved_public_files(wt)
        recorded_source = regular(wt / 'research/publication/tool_source/publication.py', wt)
        if sha(recorded_source) not in (LEGACY_PUBLICATION_SHA256, PREVIOUS_EOF_HELPER_SHA256, PREVIOUS_STAGING_HELPER_SHA256, sha(__loader__.get_data(__file__))):
            raise ValueError('Unknown prepared publisher version; no implicit rewrite')
        if fingerprints is None:
            fingerprints = live_fingerprints_from_publication(wt, manifest)
        for path, digest in fingerprints.items():
            if sha(regular(Path(path))) != digest:
                raise ValueError('Live source/evidence changed before review: ' + path)
        whitespace = staged_whitespace_review(wt)
        atomic(run / 'source_hashes.json', packed(fingerprints))
        manifest['whitespace_policy'] = {
            'policy': EOF_POLICY, 'exceptions': len(whitespace['exceptions']),
            'receipt': 'research/publication/whitespace_review.json',
            'scope': 'Only original EOF blank lines in hash-verified, byte-identical archived Python',
            'all_other_whitespace_and_conflict_errors_block': True,
        }
        replacements = {
            'research/publication/tool_source/publication.py': __loader__.get_data(__file__),
            'research/publication/validate.py': resources('validate.py'),
            'research/publication/whitespace_review.json': packed(whitespace),
            'research/evidence/publication_manifest.json': packed(manifest),
        }
        index.update({name: sha(data) for name, data in replacements.items()})
        replacements['research/publication/files.json'] = packed(index)
        journal = publication_update_journal(run, wt, replacements)
        apply_publication_journal(run, wt, journal)
    if not storage_transaction.is_file():
        staging_metadata_update(run, wt)
    archive_storage_update(run, wt)
    # The scientific files and rendered notebook/figures have not been rewritten.
    staging = stage_reviewed_publication(wt)
    atomic(run / 'staging_check.json', packed(staging))
    atomic(run / 'index_after_storage.json', packed(index_diagnostics(wt, expected_public_files(wt))))
    command([str(ORIGINAL / '.venv/bin/python'), 'research/publication/validate.py'], cwd=wt, limit=90)
    verify_saved_public_files(wt)
    whitespace = staged_whitespace_review(wt)
    if packed(whitespace) != regular(wt / 'research/publication/whitespace_review.json', wt):
        raise ValueError('Whitespace review differs after restaging; no commit performed')
    for path, digest in fingerprints.items():
        if sha(regular(Path(path))) != digest:
            raise ValueError('Live source/evidence changed during finalization: ' + path)
    if original_state() != d['original']:
        raise ValueError('Original research checkout changed during publication recovery')
    if git('diff', '--name-only', cwd=wt) or git('ls-files', '--others', '--exclude-standard', cwd=wt):
        raise ValueError('Unreviewed files or edits appeared during publication recovery')
    tree = git('write-tree', cwd=wt)
    manifest = read_json(wt / 'research/evidence/publication_manifest.json')
    review_text = '# Review before GitHub publication\n\nRepository: ' + REPO + '\n\n'
    review_text += 'Publication worktree: `' + str(wt) + '`\n\nBase: `' + d['base'] + '`\n\n'
    review_text += ('Original research checkout, data, models, and scientific archive bytes are unchanged. '
                    'This step does not commit, push, or merge.\n\n')
    review_text += '## Staged changes\n\n```text\n' + git('diff', '--cached', '--stat', cwd=wt) + '\n```\n\n'
    review_text += ('## Historical source formatting\n\nGit checked every staged path. Only the '
        + str(len(whitespace['exceptions'])) + ' documented EOF blank-line findings in byte-identical '
        'archived Python are accepted. Every other whitespace/conflict-marker error still blocks. '
        'No Git configuration was changed and no archive file was reformatted.\n\n')
    for item in whitespace['exceptions']:
        review_text += '- `' + item['path'] + '`; original SHA256 `' + item['sha256'] + '`\n'
    review_text += ('\n## Exact-file staging\n\nOnly two generated aggregate JSON files may be '
        'explicitly staged when ignored. No directory is force-added, and ignore rules are unchanged. '
        'Every manifest file is present in the Git index with the reviewed bytes.\n\n')
    for item in staging['ignored_files_explicitly_staged']:
        review_text += '- `' + item['path'] + '`; SHA256 `' + item['sha256'] + '`\n'
    review_text += ('\n## Exact archive storage\n\n'
        'The archive-only .gitattributes preserves original line endings and disables content '
        'filters only under research/workspace. Original files, root attributes, and Git configuration '
        'are unchanged. All reviewed index blobs equal the expected raw bytes.\n\n')
    review_text += '\n## Exclusions\n\n' + '\n'.join('- ' + x['path'] + ': ' + x['reason'] for x in manifest['excluded'])
    review_text += ('\n\n## Verification boundary\n\nThe existing Quality workflow and public evidence '
        'validation must pass on the actual PR head. Archived experiments were not rerun; original private '
        'weights were not replayed by publication.\n')
    atomic(run / 'PUBLICATION_REVIEW.md', review_text.encode())
    atomic(HOME / 'NFL_PUBLICATION_REVIEW.md', review_text.encode())
    atomic(run / 'whitespace_check.json', packed(whitespace))
    atomic(run / 'publication_resumption.json', packed({
        'status': 'prepared_publication_reused' if resumed else 'publication_preparation_completed',
        'policy': EOF_POLICY, 'utc': now(), 'worktree': str(wt),
        'new_worktree_created_during_resume': False, 'charts_rerendered_during_resume': False,
        'archived_source_bytes_changed': False, 'scientific_fits': 0,
        'accepted_eof_findings': len(whitespace['exceptions']),
        'staging_policy': STAGING_POLICY,
        'ignored_public_files_explicitly_staged': len(staging['ignored_files_explicitly_staged']),
        'prior_eof_transaction_retained': True, 'prior_staging_transaction_retained': True,
        'storage_policy': STORAGE_POLICY, 'github_write_done': False}))
    previous_error = STATE / 'last_error.json'
    if previous_error.is_file():
        d['resolved_preparation_error_sha256'] = sha(regular(previous_error))
    d.update(status='ready_to_publish', tree=tree,
        review_file=str(HOME / 'NFL_PUBLICATION_REVIEW.md'),
        included_files=len(manifest['source_archives']),
        expected_checks=['quality', 'publication'], rendered_public_notebook=True,
        whitespace_policy=EOF_POLICY, staging_policy=STAGING_POLICY, resumed_existing_worktree=resumed,
        preparation_helper_sha256=sha(__loader__.get_data(__file__)))
    save_state(d)
    event('publication_ready_for_review', review_file=d['review_file'],
        resumed_existing_worktree=resumed, accepted_archive_eof_findings=len(whitespace['exceptions']),
        ignored_public_files_staged=len(staging['ignored_files_explicitly_staged']),
        worktree=str(wt), scientific_fits=0, github_write_done=False)
    report()



def build_plan():
    require_selftest()
    if (STATE/'active.json').exists():
        d=state_load()
        if d.get('status') in {'ready_to_publish','pushed','merged','synced'}:
            event('existing_publication_plan',status=d['status'],review_file=d.get('review_file'),pr=d.get('pr_url'))
            return
        if d.get('status') == 'preparing' and 'commit' not in d:
            # Resume the already-rendered/staged publication. Never silently create
            # another worktree to work around a validation failure.
            return resume_prepared_plan(d)
        raise ValueError('Existing publication needs review; export publication-report. No worktree replaced.')
    find_gh()
    ident=api('user');remote=api('repos/'+REPO)
    if not remote.get('permissions',{}).get('push'): raise PermissionError('No repository push access')
    before=original_state()
    if before['branch'] != 'main': raise ValueError('Original checkout is not main; preserve branch work and review it first')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    run=STATE/stamp;run.mkdir(parents=True,exist_ok=False)
    wt=run/'worktree';branch='publish/employer-facing-'+stamp.lower()
    d={'version':VERSION,'status':'preparing','created_utc':now(),'original':before,
       'run_dir':str(run),'worktree':str(wt),'branch':branch,'repository':REPO,
       'author_login':ident['login'],'author_id':ident['id'],'scientific_fits':0,
       'github_write_done':False}
    save_state(d)
    git_network('fetch','--no-tags',URL,'refs/heads/main:refs/remotes/origin/main')
    base=git('rev-parse','refs/remotes/origin/main');d['base']=base;save_state(d)
    # No force-reset, stash, original branch checkout, or source deletion.
    code=command(['git','merge-base','--is-ancestor',before['head'],base],cwd=ORIGINAL,check=False)[0]
    if code: raise ValueError('Local main has unpublished/divergent commits. Preserve and reconcile before publication.')
    if before['head'] != base and git('diff','--name-only'):
        raise ValueError('Remote advanced and local source edits exist; no automatic conflict resolution')
    git('worktree','add','-b',branch,str(wt),base)
    included=[];excluded=[];fingerprints={};total=0
    source_dirs=SOURCE_DIRS+sorted(q.name for q in HOME.iterdir()
        if q.is_dir() and re.fullmatch(r'nfl_feature_rounds?\d+(?:_\d+)?',q.name) and q.name not in SOURCE_DIRS)
    # Copy safe, explicitly named source mirrors. Never walk the home directory indiscriminately.
    for dirname in source_dirs:
        folder=HOME/dirname
        if not folder.is_dir():
            excluded.append({'path':dirname,'reason':'not installed; not silently recreated'})
            continue
        if folder.is_symlink(): raise ValueError('Symlink project folder: '+dirname)
        for path,rel in source_selection(folder):
            raw=regular(path,folder);fingerprints[str(path)]=sha(raw)
            data=clean_notebook(raw,dirname+'/'+rel.as_posix()) if path.suffix=='.ipynb' else raw
            target='research/workspace/'+dirname+'/'+rel.as_posix()
            put(wt,target,data);total+=len(data)
            included.append({'path':target,'source_sha256':sha(raw),'published_sha256':sha(data),
                'transformation':'private notebook outputs removed' if path.suffix=='.ipynb' else 'byte-identical source'})
            if total>45*1024**2 or len(included)>5000: raise ValueError('Source publication cap exceeded')
        excluded.append({'path':dirname+'/[private inputs, generated outputs, and local evidence]',
            'reason':'input_contract.json, local manifests, caches, raw data, checkpoints, outputs, and logs excluded'})
    # Known tracked source edits can be included. Unknown files are retained and listed for review.
    changed=git('diff','--name-only','HEAD').splitlines()
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    for name in changed+untracked:
        p=ORIGINAL/safe_rel(name)
        is_tracked=name in changed
        allowed=any(name.startswith(prefix) for prefix in ('src/','scripts/','tests/','kaggle/','notebooks/'))
        allowed=allowed and p.suffix in {'.py','.ipynb','.lock','.toml','.md'}
        if not allowed or not p.is_file():
            if is_tracked: raise ValueError('Tracked edit needs deliberate integration: '+name)
            excluded.append({'path':'nfl-player-trajectory/'+name,'reason':'untracked/non-source artifact retained privately'})
            continue
        raw=regular(p,ORIGINAL);fingerprints[str(p)]=sha(raw)
        data=clean_notebook(raw,name) if p.suffix=='.ipynb' else raw
        put(wt,name,data)
        included.append({'path':name,'source_sha256':sha(raw),'published_sha256':sha(data),'transformation':'local source integration'})
    evidence, receipt_hashes=collect_evidence();fingerprints.update(receipt_hashes)
    put(wt,'research/evidence/studies.json',packed(evidence))
    put(wt,'research/publication/tool_source/publication.py',__loader__.get_data(__file__))
    # Preserve historical front pages before replacing the active employer-facing entry.
    for name in ('README.md','START_HERE.md'):
        if (wt/name).is_file():
            put(wt,'docs/archive/'+name,regular(wt/name))
    for target,resource in [('README.md','README.md'),('START_HERE.md','PROJECT_START.md'),
        ('research/README.md','RESEARCH.md'),('research/publication/validate.py','validate.py'),
        ('research/publication/render.py','render.py'),('research/RESEARCH_REVIEW.ipynb','RESEARCH_REVIEW.ipynb'),
        ('.github/workflows/publication.yml','publication.yml')]:
        put(wt,target,resources(resource))
    # The old experimental bytes must not be reformatted. Original src/scripts/tests remain fully checked.
    py=wt/'pyproject.toml';text=py.read_text()
    match=re.search(r'(?m)^extend-exclude\s*=\s*\[([^\]]*)\]',text)
    if not match: raise ValueError('Cannot safely extend the existing Ruff archive exclusion')
    if '"research"' not in match.group(1):
        revised='extend-exclude = ['+match.group(1).rstrip().rstrip(',')+', "research"]'
        text=text[:match.start()]+revised+text[match.end():]
        atomic(py,text.encode())
    ignored=wt/'.gitignore';text=ignored.read_text() if ignored.exists() else ''
    text+='\n# Private reproduction state; not distributable research evidence.\nresearch/**/input_contract.json\nresearch/**/review_release.json\nresearch/**/.venv/\nresearch/**/artifacts/\nresearch/**/outputs/\n'
    atomic(ignored,text.encode())
    index=['# Experiment source archive\n','\nOriginal folder names and scientific Python bytes are preserved. '
           'These snapshots require private contracts, data, and existing AWS paths; they are not advertised '
           'as a stand-alone public training application. No source package is marked validated merely '
           'because it was archived. Notebook outputs are deliberately stripped.\n\n']
    for dirname in source_dirs:
        target=wt/'research/workspace'/dirname
        if target.is_dir():index.append(f'- [{dirname}]({dirname}/)\n')
    put(wt,'research/workspace/README.md',''.join(index).encode())
    # New runnable public notebook uses ONLY public aggregates; no project code is imported.
    pyexe=ORIGINAL/'.venv/bin/python'
    if not pyexe.is_file(): raise FileNotFoundError('Existing Python environment missing; no reinstall performed')
    command([str(pyexe),'research/publication/render.py'],cwd=wt,limit=90)
    manifest={'schema':1,'created_utc':now(),'original_base':base,'source_archives':included,
        'excluded':excluded,'private_checkpoint_replay_performed':False,'scientific_fits':0,
        'notebook_policy':'archived notebook outputs removed; Research Review rendered from public aggregates only'}
    manifest['byte_storage_policy']=storage_policy_record()
    put(wt,ARCHIVE_ATTRIBUTES_PATH,ARCHIVE_ATTRIBUTES)
    put(wt,'research/evidence/publication_manifest.json',packed(manifest))
    # Hash all publication files after rendering. Original repository files remain governed by existing CI.
    files={p.relative_to(wt).as_posix():sha(regular(p,wt)) for p in sorted((wt/'research').rglob('*')) if p.is_file()}
    files.update({n:sha(regular(wt/n,wt)) for n in ('README.md','START_HERE.md','.github/workflows/publication.yml','pyproject.toml','.gitignore')})
    put(wt,'research/publication/files.json',packed(files))
    for path,h in fingerprints.items():
        if sha(regular(Path(path))) != h:raise ValueError('Workspace source changed during publication: '+path)
    if original_state()!=before:raise ValueError('Original checkout changed during preparation; no push performed')
    # Stage only inside the isolated worktree; never git add . in the original.
    stage_reviewed_publication(wt)
    d['source_bytes'] = total
    save_state(d)
    resume_prepared_plan(d, fingerprints=fingerprints)


def verify_plan(d):
    if d.get('status') not in {'ready_to_publish','pushed','merged','synced'}:
        raise ValueError('Complete publication-plan and review it before publishing')
    wt=Path(d['worktree'])
    if not wt.is_dir():raise ValueError('Publication worktree missing')
    # No implicit recapture, rebase, force push, or publication of a changing source.
    for path,h in read_json(Path(d['run_dir'])/'source_hashes.json').items():
        if sha(regular(Path(path)))!=h:raise ValueError('Live source/evidence changed after review: '+path)
    verify_staged_publication(wt, expected_public_files(wt))
    if git('write-tree',cwd=wt)!=d['tree']:raise ValueError('Reviewed Git index changed')
    if git('diff','--name-only',cwd=wt):raise ValueError('Publication worktree has unstaged changes')
    return wt


def publish():
    d=state_load()
    if d['status'] in {'merged','synced'}:
        event('already_merged',pr=d.get('pr_url'));return
    wt=verify_plan(d)
    if 'commit' not in d:
        if git('rev-parse','HEAD',cwd=wt)!=d['base']:raise ValueError('Publication branch moved before commit')
        email=f"{d['author_id']}+{d['author_login']}@users.noreply.github.com"
        command(['git','-c','user.name='+d['author_login'],'-c','user.email='+email,
            'commit','-m','Publish reproducible trajectory research and employer-facing evidence'],cwd=wt,limit=60)
        d['commit']=git('rev-parse','HEAD',cwd=wt);save_state(d)
    if git('rev-parse','HEAD',cwd=wt)!=d['commit']:raise ValueError('Publication HEAD changed')
    # Push is fast-forward only; no --force or global credential modifications.
    try:
        git_network('push',URL,'HEAD:refs/heads/'+d['branch'],cwd=wt)
    except RuntimeError as exc:
        if 'workflow' in str(exc).lower():
            raise RuntimeError('Workflow scope required. Run gh auth refresh -h github.com -s workflow, then repeat publish.') from exc
        raise
    remote=git_network('ls-remote',URL,'refs/heads/'+d['branch'],cwd=wt).split()[0]
    if remote!=d['commit']:raise ValueError('Pushed ref does not match reviewed commit')
    prs=json.loads(gh('pr','list','--repo',REPO,'--head',d['branch'],'--state','all',
                     '--json','number,url,state,headRefOid')[1])
    if len(prs)>1:raise ValueError('Ambiguous pull request for publication branch')
    if prs:
        pr=prs[0]
        if pr['headRefOid']!=d['commit'] or pr['state']!='OPEN':raise ValueError('Existing PR is not the expected open publication')
    else:
        body=('## Publication scope\n\nEmployer-facing README, public evidence notebook, aggregate results, '
              'and sanitized source mirrors collected from the owner\'s AWS workspace.\n\n'
              'No raw data, private contracts, per-row outputs, weights, credentials, or notebook attachments '
              'are added. Original workspace files are preserved.\n\n'
              'Original Quality CI remains required. Publication CI checks source syntax, public-file hashes, '
              'privacy exclusions, and aggregate RMSE arithmetic. Archived experiments are not rerun by '
              'publication CI. No new model fit, Kaggle result, or live AWS deployment is implied.\n\n'
              'Experimental source is excluded from automatic reformatting to preserve its scientific hashes; '
              'maintained src/scripts/tests retain their original checks.\n')
        path=Path(d['run_dir'])/'PR_BODY.md';atomic(path,body.encode())
        url=gh('pr','create','--repo',REPO,'--head',d['branch'],'--base','main',
            '--title','Publish employer-facing trajectory research and evidence','--body-file',str(path),cwd=wt)[1].strip()
        pr=json.loads(gh('pr','view',url,'--repo',REPO,'--json','number,url,headRefOid')[1])
    d.update(status='pushed',github_write_done=True,pr_number=pr['number'],pr_url=pr['url'])
    save_state(d)
    event('publication_pushed_pr_created',pr=d['pr_url'],commit=d['commit'],merged=False)
    report()


def check_status():
    d=state_load()
    if not d.get('pr_number'):raise ValueError('Publish the reviewed branch first')
    pr=json.loads(gh('pr','view',str(d['pr_number']),'--repo',REPO,'--json',
        'number,url,state,headRefOid,mergeStateStatus,mergedAt,mergeCommit')[1])
    code,out,err=gh('pr','checks',str(d['pr_number']),'--repo',REPO,'--json',
                    'name,bucket,state,link',check=False)
    if code not in (0,1,8):raise RuntimeError('Could not read pull-request checks: '+err[-1500:])
    checks=json.loads(out) if out.strip().startswith('[') else []
    if pr['headRefOid']!=d['commit']:raise ValueError('PR head differs from reviewed publication commit')
    result={'pr':pr,'checks':checks,'all_expected_passed':False}
    byname={c['name']:c['bucket'] for c in checks}
    result['all_expected_passed']=all(byname.get(n)=='pass' for n in d['expected_checks']) and all(
        c['bucket'] in ('pass','skipping') for c in checks)
    atomic(Path(d['run_dir'])/'github_checks.json',packed(result))
    event('publication_checks',pr=d['pr_url'],all_expected_passed=result['all_expected_passed'],
          checks=[{'name':c['name'],'bucket':c['bucket']} for c in checks])
    return d,result


def merge():
    d,result=check_status();pr=result['pr']
    if pr['state']=='MERGED':
        merged=pr['mergeCommit']['oid']
    else:
        if not result['all_expected_passed']:
            event('merge_not_performed',reason='Checks pending, absent, or failed; no override applied')
            report();return
        if pr['mergeStateStatus']!='CLEAN':
            event('merge_not_performed',reason='GitHub merge state '+pr['mergeStateStatus']+'; no admin bypass')
            report();return
        gh('pr','merge',str(d['pr_number']),'--repo',REPO,'--squash',
           '--match-head-commit',d['commit'])
        confirmed=json.loads(gh('pr','view',str(d['pr_number']),'--repo',REPO,'--json','state,mergedAt,mergeCommit')[1])
        if confirmed['state']!='MERGED':
            event('merge_queued_or_pending',note='No completed merge claimed');report();return
        merged=confirmed['mergeCommit']['oid']
    d.update(status='merged',merge_commit=merged);save_state(d)
    event('github_merge_verified',pr=d['pr_url'],merge_commit=merged,aws_checkout_updated=False)
    report()


def sync_original():
    d=state_load()
    if d.get('status') not in ('merged','synced'):raise ValueError('Verify merge before synchronizing original checkout')
    before=original_state()
    pinned=[]
    for dirname in SOURCE_DIRS:
        p=HOME/dirname/'input_contract.json'
        if p.is_file() and not p.is_symlink():
            contract=read_json(p)
            if contract.get('repo_head')==before['head']:
                pinned.append(dirname+'/input_contract.json')
    if pinned:
        wt=Path(d['worktree'])
        if git('diff','--name-only',cwd=wt) or git('diff','--cached','--name-only',cwd=wt):
            raise ValueError('Publication view has edits; no checkout performed')
        git_network('fetch','--no-tags',URL,'refs/heads/main:refs/remotes/origin/main',cwd=wt)
        code=command(['git','merge-base','--is-ancestor',d['merge_commit'],'origin/main'],cwd=wt,check=False)[0]
        if code:raise ValueError('Verified publication merge is no longer on remote main')
        git('switch','--detach',d['merge_commit'],cwd=wt)
        d.update(status='merged',publication_view_head=d['merge_commit'],
                 original_checkout_preserved_at=before['head'],pinned_contracts=pinned)
        save_state(d)
        event('publication_verified_original_preserved',merge_commit=d['merge_commit'],
              original_head=before['head'],reason='Active experiments require the exact original Git HEAD',
              original_checkout_fast_forwarded=False)
        report();return
    if before['branch']!='main' or git('diff','--name-only'):
        raise ValueError('Original checkout has tracked edits or another branch. Nothing discarded; merge remains published.')
    git_network('fetch','--no-tags',URL,'refs/heads/main:refs/remotes/origin/main')
    head=git('rev-parse','refs/remotes/origin/main')
    code=command(['git','merge-base','--is-ancestor',d['merge_commit'],head],cwd=ORIGINAL,check=False)[0]
    if code:raise ValueError('Remote main no longer contains the verified publication merge')
    incoming=git('diff','--name-only',before['head'],head).splitlines()
    if any(p.startswith(('data/','artifacts/','logs/','.state/','.venv/')) for p in incoming):
        raise ValueError('Incoming change touches a private workspace path; refusing automatic sync')
    tracked=set(git('ls-files').splitlines())
    for name in incoming:
        if name not in tracked and (ORIGINAL/name).exists():
            raise ValueError('Untracked or ignored file would be overwritten: '+name)
    command(['git','merge','--ff-only','--no-overwrite-ignore',head],cwd=ORIGINAL,limit=60)
    if git('rev-parse','HEAD')!=head:raise ValueError('Source synchronization verification failed')
    d.update(status='synced',aws_checkout_head=head,aws_source_sync_utc=now());save_state(d)
    event('github_and_aws_source_synced',head=head,private_artifacts_preserved=True,
          note='Private artifact bytes were not rehashed; no data paths were changed')
    report()


def report():
    STATE.mkdir(parents=True,exist_ok=True)
    destination=HOME/'nfl_publication_report.zip'
    fd,temp=tempfile.mkstemp(dir=HOME,suffix='.zip');os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            d = {}
            if (STATE/'active.json').exists():
                d=state_load();z.writestr('publication.json',packed(d));run=Path(d['run_dir'])
                for n in ('github_checks.json','PUBLICATION_REVIEW.md','whitespace_check.json','staging_check.json','publication_resumption.json',
                          'index_before_storage.json','index_after_storage.json',
                          'eof-policy-transaction.json','evidence-stage-transaction.json',
                          'archive-storage-transaction.json'):
                    p=run/n
                    if p.is_file():z.writestr(n,regular(p))
                # Files, modes, hashes and newline counts only: no source bytes or credentials.
                try:
                    wt=Path(d['worktree'])
                    if wt == run/'worktree' and run.parent == STATE and not wt.is_symlink():
                        values=index_diagnostics(wt,expected_public_files(wt))
                        z.writestr('index_diagnostics.json',packed(values))
                        for name in ('research/workspace/.gitattributes', '.gitattributes'):
                            target=wt/name
                            if target.is_file() and not target.is_symlink():
                                z.writestr('attributes/'+name.replace('/','__'),check_text(regular(target,wt),name))
                except Exception as diagnostic_error:
                    z.writestr('diagnostic_capture.json',packed({'status':'incomplete',
                        'error':SECRET_RE.sub('[REDACTED]',str(diagnostic_error))[:1500]}))
            tests=STATE/'selftest.json'
            if tests.is_file():z.writestr('publication_selftest.json',regular(tests))
            error=STATE/'last_error.json'
            if error.is_file():
                raw_error=regular(error)
                label=('historical_last_error.json' if sha(raw_error)==d.get('resolved_preparation_error_sha256') else 'last_error.json')
                z.writestr(label,raw_error)
        os.replace(temp,destination)
    finally:
        Path(temp).unlink(missing_ok=True)
    event('publication_report_created',path=str(destination))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['github-login','publication-plan','publish','publication-status',
        'merge','publication-sync','publication-report','publication-self-test'])
    args=parser.parse_args()
    package_integrity()
    STATE.mkdir(parents=True,exist_ok=True)
    if args.stage=='publication-self-test':
        import publication_tests.test_publication as tests
        import unittest
        import publication_tests.test_storage as storage_tests
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
            unittest.defaultTestLoader.loadTestsFromModule(storage_tests)])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        atomic(STATE/'selftest.json',packed({'status':'publication_tests_passed' if result.wasSuccessful() else 'publication_tests_failed',
            'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':len(result.skipped),'source_sha256':sha(__loader__.get_data(__file__)),'utc':now(),
            'failure_details':[{'test':str(t),'traceback':SECRET_RE.sub('[REDACTED]',msg)[-12000:]} for t,msg in result.failures],
            'error_details':[{'test':str(t),'traceback':SECRET_RE.sub('[REDACTED]',msg)[-12000:]} for t,msg in result.errors]}))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    # Advisory publication lock is local; does not hold or operate ML worker locks.
    import fcntl
    with (STATE/'publication.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another publication operation is active; no concurrent run allowed')
        try:
            actions={'github-login':github_login,'publication-plan':build_plan,'publish':publish,
                'publication-status':check_status,'merge':merge,'publication-sync':sync_original,
                'publication-report':report}
            if args.stage != 'github-login':
                def deadline(signum, frame):
                    raise TimeoutError('Publication stage exceeded its 480-second hard cap')
                signal.signal(signal.SIGALRM, deadline)
                signal.alarm(480)
            try:
                actions[args.stage]()
            finally:
                signal.alarm(0)
        except (Exception,KeyboardInterrupt) as exc:
            message=SECRET_RE.sub('[REDACTED]',str(exc))
            atomic(STATE/'last_error.json',packed({'utc':now(),'stage':args.stage,'error_type':type(exc).__name__,
                'message':message,'sources_preserved':True,'no_force_reset':True}))
            event('publication_stopped',stage=args.stage,error=message)
            try:report()
            except Exception:pass
            raise SystemExit(2)
