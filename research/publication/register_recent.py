"""Register a fixed allowlist of reviewed public evidence; preserve archive hashes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
ADDITIONS = (
    'research/evidence/model_reproduction.json',
    'research/publication/recent_models.py',
    'research/publication/checkpoint_lineage.py',
    'research/publication/register_recent.py',
    'research/RECENT_MODELS.ipynb',
    'research/figures/winner_folds.png',
    'research/figures/winner_confirmation.png',
    'research/figures/winner_horizon.png',
)

def main():
    path = ROOT / 'research/publication/files.json'
    manifest = json.loads(path.read_text())
    validator = ROOT / 'research/publication/validate.py'
    text = validator.read_text()
    old = "not name.startswith('research/figures/review_')"
    new = "not (name.startswith('research/figures/review_') or name in {'research/figures/winner_folds.png', 'research/figures/winner_confirmation.png', 'research/figures/winner_horizon.png'})"
    if old in text:
        if text.count(old) != 1:
            raise ValueError('Unexpected image allowlist source')
        validator.write_text(text.replace(old, new, 1))
    elif new not in text:
        raise ValueError('Publication validator changed; review before registration')
    for name in (*ADDITIONS, 'research/publication/validate.py'):
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing reviewed public artifact: ' + name)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Existing publication changed unexpectedly: ' + name)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'registered_additions': len(ADDITIONS), 'existing_archive_hashes_preserved': True}))

if __name__ == '__main__':
    main()
