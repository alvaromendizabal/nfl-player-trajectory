"""Validate public artifacts only; no imports of private experiment code or fitting."""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / 'research/publication/files.json').read_text())
    if not manifest:
        raise ValueError('Publication hash manifest is empty')
    expected = set(manifest)
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'research').rglob('*') if p.is_file()}
    extra = actual - expected - {'research/publication/files.json'}
    if extra:
        raise ValueError('Unmanifested research files: ' + repr(sorted(extra)))
    forbidden = {'.env', 'input_contract.json', 'review_release.json', 'kaggle.json'}
    blocked_ext = {'.pt', '.pkl', '.pickle', '.npz', '.npy', '.parquet', '.zip', '.gz', '.bin'}
    files = 0
    for name, digest in manifest.items():
        path = ROOT/name
        if path.is_symlink() or not path.is_file() or '..' in Path(name).parts:
            raise ValueError('Invalid file path: ' + name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Hash mismatch: ' + name)
        if path.name in forbidden or path.suffix in blocked_ext:
            raise ValueError('Private/binary artifact in publication: ' + name)
        if path.suffix == '.png':
            if not (name.startswith('research/figures/review_') or name in {'research/figures/winner_folds.png', 'research/figures/winner_confirmation.png', 'research/figures/winner_horizon.png', 'research/figures/winner_diversity.png', 'research/figures/winner_private_progress.png'}) or not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Unexpected image artifact')
            files += 1
            continue
        text = data.decode('utf-8')
        if re.search(r'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|X-Amz-Signature=[a-fA-F0-9]{32,}', text):
            raise ValueError('Credential-like content: ' + name)
        if path.suffix == '.py':
            ast.parse(text, feature_version=(3, 11))
        if path.suffix == '.ipynb':
            nb = json.loads(text)
            if nb.get('nbformat') != 4 or not nb.get('cells'):
                raise ValueError('Invalid notebook')
            for cell in nb['cells']:
                if cell.get('attachments'):
                    raise ValueError('Notebook attachment not allowed')
                if name.startswith('research/workspace/') and (cell.get('outputs') or cell.get('execution_count') is not None):
                    raise ValueError('Archived notebook still has private outputs/counts')
        files += 1
    provenance = json.loads((ROOT/'research/evidence/publication_manifest.json').read_text())
    mirrors = {}
    for item in provenance['source_archives']:
        if item['path'] in mirrors:
            raise ValueError('Duplicate archived provenance path')
        mirrors[item['path']] = item
        data = (ROOT/item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['published_sha256']:
            raise ValueError('Published source differs from provenance: ' + item['path'])
    whitespace = json.loads((ROOT/'research/publication/whitespace_review.json').read_text())
    if whitespace.get('policy') != 'byte-identical-archive-python-eof-v1':
        raise ValueError('Unexpected archive formatting policy')
    reviewed = set()
    for item in whitespace['exceptions']:
        name = item['path']
        original = mirrors.get(name)
        if (name in reviewed or not name.startswith('research/workspace/')
                or not name.endswith('.py') or original is None
                or original.get('transformation') != 'byte-identical source'
                or original['source_sha256'] != original['published_sha256']
                or original['published_sha256'] != item['sha256']):
            raise ValueError('Unjustified archive formatting exception: ' + name)
        reviewed.add(name)
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Archive bytes changed after whitespace review')
    if provenance['whitespace_policy']['exceptions'] != len(reviewed):
        raise ValueError('Archive formatting exception count differs')
    storage = provenance.get('byte_storage_policy')
    if storage is not None:
        if storage.get('policy') != 'immutable-archive-bytes-v1' or storage.get('attributes_file') != 'research/workspace/.gitattributes':
            raise ValueError('Unexpected archive byte-storage policy')
        attrs = (ROOT/storage['attributes_file']).read_bytes()
        if hashlib.sha256(attrs).hexdigest() != storage['attributes_sha256']:
            raise ValueError('Archive storage policy hash differs')
        algorithm = subprocess.run(['git','rev-parse','--show-object-format'],cwd=ROOT,
            check=True,capture_output=True,text=True,timeout=15).stdout.strip()
        if algorithm not in ('sha1','sha256'):
            raise ValueError('Unsupported Git object format')
        raw_index = subprocess.run(['git','ls-files','--stage','-z'],cwd=ROOT,
            check=True,capture_output=True,text=True,timeout=30).stdout
        entries={}
        for record in raw_index.split('\0'):
            if not record:
                continue
            header,name=record.split('\t',1)
            mode,oid,stage=header.split()
            if name in manifest or name=='research/publication/files.json':
                if name in entries or stage!='0' or mode not in ('100644','100755'):
                    raise ValueError('Conflicted or invalid indexed public artifact: '+name)
                entries[name]=oid
        for name in set(manifest) | {'research/publication/files.json'}:
            data=(ROOT/name).read_bytes()
            oid=hashlib.new(algorithm,b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if entries.get(name)!=oid:
                raise ValueError('Public checkout/index raw-byte mismatch: '+name)
        paths=sorted(n for n in manifest if n.startswith('research/workspace/'))
        names=['text','filter','ident','working-tree-encoding','eol']
        output=subprocess.run(['git','check-attr','--cached','-z','--stdin',*names],
            cwd=ROOT,check=True,capture_output=True,text=True,timeout=30,
            input='\0'.join(paths)+'\0').stdout.split('\0')
        if output[-1]=='':
            output.pop()
        if len(output)%3:
            raise ValueError('Malformed Git attribute output')
        seen={}
        for name,attr,value in zip(output[0::3],output[1::3],output[2::3]):
            if (name,attr) in seen:
                raise ValueError('Duplicate Git attribute output')
            seen[name,attr]=value
        for name in paths:
            for attr in names:
                if seen.get((name,attr)) != ('unspecified' if attr=='eol' else 'unset'):
                    raise ValueError('Effective archive attribute conflicts with byte preservation: '+name+' '+attr)
    evidence = json.loads((ROOT/'research/evidence/studies.json').read_text())
    arithmetic = 0
    for study in evidence['studies']:
        for category in ('metrics','training_metrics','pooled_metrics','pooled_rmse'):
            for record in study.get(category, {}).values():
                if isinstance(record, dict) and all(k in record for k in ('rows','sse','rmse')):
                    if record['rows'] <= 0 or not math.isclose(record['rmse'], math.sqrt(record['sse']/(2*record['rows'])), abs_tol=1e-9):
                        raise ValueError('Aggregate RMSE arithmetic failed')
                    arithmetic += 1
    print(json.dumps({'status':'publication_validation_passed','files':files,
                      'aggregate_rmse_checks':arithmetic,'scientific_fits':0,
                      'archived_experiment_tests_run':0}))


if __name__ == '__main__':
    main()
