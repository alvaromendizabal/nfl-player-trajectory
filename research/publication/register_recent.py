"""Register the reviewed frontier evidence allowlist while preserving archive hashes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
ADDITIONS = (
    'README.md',
    'docs/MODEL_CARD.md',
    'docs/SOURCES.md',
    'docs/results/frontier_submission.json',
    '.github/workflows/recent-models.yml',
    'research/README.md',
    'research/evidence/model_reproduction.json',
    'research/publication/recent_models.py',
    'research/publication/register_recent.py',
    'research/publication/validate.py',
    'research/RECENT_MODELS.ipynb',
    'research/figures/winner_folds.png',
    'research/figures/winner_confirmation.png',
    'research/figures/winner_horizon.png',
    'research/figures/winner_diversity.png',
    'research/figures/winner_private_progress.png',
)

EXPECTED_IMAGES = {
    'research/figures/winner_folds.png',
    'research/figures/winner_confirmation.png',
    'research/figures/winner_horizon.png',
    'research/figures/winner_diversity.png',
    'research/figures/winner_private_progress.png',
}

def main():
    path = ROOT / 'research/publication/files.json'
    manifest = json.loads(path.read_text())
    validator = ROOT / 'research/publication/validate.py'
    text = validator.read_text()
    missing = sorted(name for name in EXPECTED_IMAGES if name not in text)
    if missing:
        raise ValueError('Publication validator is missing reviewed frontier images: ' + repr(missing))
    for name in ADDITIONS:
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing reviewed public artifact: ' + name)
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, expected in manifest.items():
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Registered publication artifact is missing: ' + name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Existing publication changed unexpectedly: ' + name)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'registered_allowlisted_paths': len(ADDITIONS), 'existing_archive_hashes_preserved': True}))

if __name__ == '__main__':
    main()
