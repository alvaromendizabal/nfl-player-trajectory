"""Strict local I/O for a read-only-parent feature-signal audit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
import numpy as np

INPUT_FIELDS = ('ids','node','node_valid','pair','pair_valid','pair_age','role','side','node_age','query','base','train','signature')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_file(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if not parts.parts or parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe relative artifact path')
    path = Path(root).joinpath(*parts.parts)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink artifacts are not accepted')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def atomic(path: Path, writer):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink output rejected')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    atomic(Path(path), lambda f: f.write(data))


def seal_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError('Sealed result changed; keep the prior evidence and stop')
    else:
        atomic_json(path, value)


def read_observed(path: Path) -> dict:
    """Unpack explicitly allowed arrays only. In particular never unpack y or keys."""
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > 64*1024**2:
            raise ValueError('Input archive exceeds 64 MiB decoded limit')
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in INPUT_FIELDS}
    if not bool(data['train']):
        raise ValueError('Evaluation play rejected by training-only audit')
    if any(not np.isfinite(v).all() for k,v in data.items() if k != 'signature'):
        raise ValueError('Nonfinite observed input')
    return data


def checkpoint_npz(path: Path, arrays: dict) -> str:
    """Exact numerical replay, with a byte receipt; never replace divergent files."""
    path = Path(path); receipt = path.with_suffix('.json')
    if path.exists() or receipt.exists():
        meta = read_json(safe_file(receipt.parent, receipt.name))
        if digest(safe_file(path.parent, path.name)) != meta['sha256']:
            raise ValueError('Feature checkpoint checksum changed')
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(arrays) or any(z[k].dtype != v.dtype or z[k].shape != v.shape or not np.array_equal(z[k],v) for k,v in arrays.items()):
                raise ValueError('Feature replay changed')
    else:
        atomic(path, lambda f: np.savez_compressed(f, **arrays))
        atomic_json(receipt, {'sha256':digest(path)})
    return digest(path)
