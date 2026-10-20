"""Remote-only, hash-verified range transport for the unchanged PyPI lockfile."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Cloud only')
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename, canonicalize_name


def main():
    lock, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    pins = [line.strip().split('==') for line in lock.read_text().splitlines()
            if '==' in line and not line.lstrip().startswith('#')]
    started = time.monotonic()
    def metadata(pin):
        name, version = pin
        with urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=60) as response:
            data = json.load(response)
        candidates = []
        for item in data['urls']:
            if not item['filename'].endswith('.whl'): continue
            wheel_name, wheel_version, _, tags = parse_wheel_filename(item['filename'])
            if canonicalize_name(name) != wheel_name or str(wheel_version) != version: continue
            compatible = [ranks[t] for t in tags if t in ranks]
            if compatible: candidates.append((min(compatible), item))
        if not candidates: raise RuntimeError('No compatible pinned wheel: '+name)
        item = min(candidates, key=lambda x:x[0])[1]
        assert Path(item['filename']).name == item['filename'] and 0 < item['size'] < 2**31
        return dict(name=name, version=version, filename=item['filename'], url=item['url'],
                    size=item['size'], sha256=item['digests']['sha256'])
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        wheels = list(pool.map(metadata, pins))
    total = sum(w['size'] for w in wheels)
    assert total < 10*1024**3
    result = dict(state='DOWNLOADING', wheels=wheels, total_bytes=total, received_bytes=0,
                  transport='16 concurrent range requests, 4 MiB chunks; official PyPI URLs and SHA-256',
                  lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    def save():
        result['elapsed_s'] = time.monotonic()-started
        temporary = output/'result.tmp'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(output/'result.json')
        print(json.dumps({k:v for k,v in result.items() if k!='wheels'}), flush=True)
    chunk = 4*1024**2
    tasks = []
    for wheel in wheels:
        path = output/(wheel['filename']+'.part')
        with path.open('xb') as stream: stream.truncate(wheel['size'])
        tasks += [(wheel, offset, min(offset+chunk, wheel['size'])-1)
                  for offset in range(0, wheel['size'], chunk)]
    # Small chunks complete independently; installation sees only verified files.
    def fetch(task):
        wheel, start, end = task
        for attempt in range(3):
            try:
                request = Request(wheel['url'], headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
                with urlopen(request, timeout=60) as response:
                    full = start == 0 and end+1 == wheel['size']
                    assert response.status == 206 or (full and response.status == 200)
                    if response.status == 206:
                        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{wheel["size"]}'
                    data = response.read(end-start+2)
                    assert len(data) == end-start+1
                fd = os.open(output/(wheel['filename']+'.part'), os.O_WRONLY)
                try:
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd, data[written:], start+written)
                finally: os.close(fd)
                return len(data)
            except Exception:
                if attempt == 2: raise
                time.sleep(1+attempt)
    save()
    try:
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(fetch, task) for task in tasks]
            for i, future in enumerate(cf.as_completed(futures)):
                result['received_bytes'] += future.result()
                if i % 32 == 0: save()
        for wheel in wheels:
            part = output/(wheel['filename']+'.part')
            digest = hashlib.sha256()
            with part.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2), b''): digest.update(block)
            assert digest.hexdigest() == wheel['sha256'], wheel['filename']
            part.rename(output/wheel['filename'])
            wheel['verified'] = True
        result['state'] = 'VERIFIED'; save()
    except Exception as exc:
        result.update(state='ERROR', error=type(exc).__name__+': '+str(exc)); save(); raise


if __name__ == '__main__': main()
