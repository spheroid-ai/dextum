"""Stop the dedicated masking instance at a batch boundary or execution stop."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


TERMINAL = {'COMPLETE', 'LEASE_BOUNDARY', 'EXECUTION_STOP', 'NEEDS_EXECUTION_REVIEW'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    instance = os.environ['CONTAINER_ID']
    assert instance == '55271331'
    key = os.environ.get('CONTAINER_API_KEY')
    if not key:
        boot = dict(part.split(b'=', 1) for part in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in part)
        assert boot.get(b'CONTAINER_ID') == instance.encode()
        key = boot[b'CONTAINER_API_KEY'].decode()
    target = a.run / 'INSTANCE_FINISH.json'
    target.write_text(json.dumps(dict(state='ARMED', instance_id=instance, terminal_states=sorted(TERMINAL))) + '\n')
    while True:
        try:
            state = json.loads((a.run / 'BATCH_STATE.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        if state.get('state') in TERMINAL:
            break
        time.sleep(30)
    try:
        with (a.run / 'finish_report.log').open('a') as log:
            reporting = subprocess.run([sys.executable, '-m', 'pass1c.masking_results', '--run', str(a.run)],
                                       stdout=log, stderr=log, timeout=60, check=False)
        report_code = reporting.returncode
    except (OSError, subprocess.TimeoutExpired):
        report_code = -1  # A reporting error must never prevent the stop.
    record = dict(state='STOP_REQUESTING', batch_state=state['state'], time=time.time(),
                  instance_id=instance, report_exit_code=report_code)
    while True:
        target.write_text(json.dumps(record, indent=2) + '\n')
        try:
            request = Request('https://console.vast.ai/api/v0/instances/' + instance + '/', method='PUT',
                              data=json.dumps({'state': 'stopped'}).encode(),
                              headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urlopen(request, timeout=20) as response:
                result = json.load(response)
            if result.get('success') is False:
                raise RuntimeError('Stop request rejected')
            record.update(state='STOP_REQUEST_ACCEPTED', requested_at=time.time())
            target.write_text(json.dumps(record, indent=2) + '\n')
            return
        except Exception as exc:
            record.update(state='STOP_RETRY', error_type=type(exc).__name__)
            time.sleep(30)


if __name__ == '__main__':
    main()
