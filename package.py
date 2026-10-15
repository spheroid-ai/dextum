"""Package only this explicit publication tree; never traverse the parent workspace."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'site' / 'dist'


def allowed(path):
    rel = path.relative_to(ROOT)
    return (path.is_file() and not path.is_symlink()
            and not any(x in {'.git', 'qa', '__pycache__'} for x in rel.parts)
            and path.suffix != '.zip'
            and path.name not in {'.DS_Store', 'SHA256SUMS', 'PREPARATION_MANIFEST.json'})


def archive(destination, base, files):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(base).as_posix(), (2026, 10, 15, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise ValueError(f'Archive verification failed: {destination}')


# Include the figure download in the offline page, but never a ZIP inside itself.
offline = DIST / 'dextum-v1.zip'
archive(offline, DIST, [p for p in DIST.rglob('*')
                       if p.is_file() and p != offline and not p.is_symlink()])
files = [p for p in ROOT.rglob('*') if allowed(p)]
records = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
manifest = ROOT / 'PREPARATION_MANIFEST.json'
manifest.write_text(json.dumps({'status': 'PREPARATION_NOT_FULL_TRAINING_RELEASE',
                               'freeze_utc': '2026-10-19T16:00:00Z',
                               'files': records}, indent=2) + '\n')
bundle = ROOT / 'dextum-20oct-preparation.zip'
# The page's small figure download is included; the large offline ZIP is excluded.
archive(bundle, ROOT, files + [manifest, DIST / 'assets' / 'dextum-figures.zip'])
(ROOT / 'SHA256SUMS').write_text(''.join(
    hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.relative_to(ROOT).as_posix() + '\n'
    for p in [bundle, offline, manifest]))
print(json.dumps({'preparation_zip': str(bundle), 'bytes': bundle.stat().st_size,
                  'files': len(records), 'offline_zip': str(offline)}))
