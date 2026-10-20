"""Bound one identified cloud compiler; resource failure is not a physics gate."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01')


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast.ai only')
    pid, deadline = int(sys.argv[1]), float(sys.argv[2])
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        (ROOT / 'RESOURCE_GUARD_NO_TARGET.json').write_text(json.dumps(dict(
            time=time.time(), pid=pid,
            state='TARGET_ALREADY_GONE', action='No process signalled'), indent=2) + '\n')
        return
    command = (proc / 'cmdline').read_bytes()
    assert b'pass1b.train' in command and b'c_touch' in command
    # Leave over 20 GiB below the measured 241.8-GiB container memory limit.
    limit_kib = 220 * 1024 * 1024
    while time.time() < deadline and proc.exists():
        try:
            if (proc / 'cmdline').read_bytes() != command:
                return  # Never act on a reused process id.
            fields = dict(line.split(':', 1) for line in
                          (proc / 'status').read_text().splitlines() if ':' in line)
        except FileNotFoundError:
            return
        rss = int(fields.get('VmRSS', '0 kB').split()[0])
        row = dict(time=time.time(), pid=pid, rss_kib=rss, limit_kib=limit_kib)
        with (ROOT / 'resource_trace.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if rss > limit_kib:
            row.update(reason='Compiler resident memory exceeded cloud resource bound',
                       classification='Execution failure; not a new D5 criterion',
                       attempt=1)
            (ROOT / 'TRAINING_RESOURCE_STOP.json').write_text(json.dumps(row, indent=2) + '\n')
            os.kill(pid, signal.SIGTERM)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
