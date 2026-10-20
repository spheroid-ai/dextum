"""Remote setup transport recovery; preserve original logs and all run settings."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Cloud only')

BASE = Path('/root/dextum')
ROOT = BASE/'dextum_cursor_agent/runs/pass0_section7_01'


def processes():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: args = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError): continue
        if b'pass0.section7_pipeline' in args: found['supervisor'] = int(proc.name)
        if args[:3] == [b'/root/.local/bin/uv', b'pip', b'install']:
            found['installer'] = int(proc.name)
    return found


def main():
    result = dict(state='WAITING_FOR_VERIFIED_WHEELS', changes='Transport only; original lock and experiment unchanged')
    def save():
        result['updated_at'] = time.time()
        (BASE/'wheel_install_result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    save()
    while True:
        manifest = json.loads((BASE/'pinned_wheels/result.json').read_text())
        if manifest['state'] == 'ERROR': raise RuntimeError('Wheel transfer failed')
        if manifest['state'] == 'VERIFIED': break
        time.sleep(3)
    assert all(w.get('verified') for w in manifest['wheels'])
    lock = BASE/'dextum_cursor_agent/vendor/aero_official/requirements-gpu.lock.txt'
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == manifest['lock_sha256']
    current = processes()
    if (ROOT/'bootstrap/preflight.json').exists():
        assert json.loads((ROOT/'bootstrap/preflight.json').read_text())['state'] == 'PASS'
        result['state'] = 'ORIGINAL_SETUP_ALREADY_PASSED'; save(); return
    assert not (ROOT/'section7_execute_01/e3/c_touch/config.json').exists()
    supervisor = current['supervisor']
    assert json.loads((ROOT/'section7_pipeline_01/result.json').read_text())['state'] == 'WAITING_FOR_PREFLIGHT'
    os.kill(supervisor, signal.SIGSTOP)
    time.sleep(.2)
    assert '\nState:\tT' in Path(f'/proc/{supervisor}/status').read_text()
    result.update(state='SWITCHING_SETUP_TRANSPORT', supervisor_pid=supervisor, prior_installer_pid=current.get('installer')); save()
    if 'installer' in current: os.kill(current['installer'], signal.SIGTERM)
    for _ in range(60):
        if (ROOT/'bootstrap/EXIT_CODE').exists(): break
        time.sleep(.5)
    assert (ROOT/'bootstrap/EXIT_CODE').exists(), 'Original installer did not finish stopping'
    prior = ROOT/'bootstrap_stream_01'
    assert not prior.exists()
    (ROOT/'bootstrap').rename(prior)
    output = ROOT/'bootstrap'; output.mkdir()
    py = BASE/'.venv/bin/python'
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='disable')
    code = 1
    try:
        with (output/'setup.log').open('xb') as log:
            command = ['/root/.local/bin/uv', 'pip', 'install', '--python', str(py), '--offline',
                       '--no-index', '--find-links', str(BASE/'pinned_wheels'), '-r', str(lock)]
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=300)
            env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in sorted((BASE/'.venv/lib/python3.11/site-packages').glob('nvidia/*/lib')))+':/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'
            check = "import importlib.metadata as m,json; from pathlib import Path; pins=[x.split('==') for x in Path("+repr(str(lock))+").read_text().splitlines() if '==' in x and not x.startswith('#')]; actual={k:m.version(k) for k,v in pins}; assert all(actual[k]==v for k,v in pins); print(json.dumps(actual))"
            versions = subprocess.check_output([str(py), '-c', check], env=env, text=True, timeout=30)
            (output/'installed_versions.json').write_text(versions)
            subprocess.run([str(py), '-u', 'cloud/remote/preflight.py', str(output/'preflight.json')],
                           cwd=BASE/'dextum_cursor_agent', env=env, stdout=log, stderr=log, check=True, timeout=180)
        code = 0
        result.update(state='PREFLIGHT_PASSED', installed_versions=len(json.loads(versions)))
    except Exception as exc:
        result.update(state='INSTALL_FAILED', error=type(exc).__name__+': '+str(exc))
    finally:
        (output/'EXIT_CODE').write_text(str(code)+'\n')
        save()
        os.kill(supervisor, signal.SIGCONT)


if __name__ == '__main__': main()
