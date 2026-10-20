"""Finish durable evaluation boundaries before advancing a resumed learner.

This module has no simulator, JAX, or training dependencies. Existing failed
checks stay stops. Incomplete records are retained before an exact replay.
"""
import json
from pathlib import Path
import shutil
import time
import zipfile


def complete_boundary_monitor(monitor, steps, params, save_policy):
    output = Path(monitor.output)
    record = output / f'{steps}.json'
    paired = output / f'{steps}.npz'
    if record.exists():
        row = json.loads(record.read_text())
        if int(row['steps']) != steps or row['state'] != 'PASS':
            raise RuntimeError('Preserved monitor stop or step mismatch: ' + str(record))
        if not paired.is_file():
            raise RuntimeError('Completed monitor is missing its paired evidence: ' + str(paired))
        with zipfile.ZipFile(paired) as archive:
            if archive.testzip() is not None or not {'internal.npy', 'faithful.npy'} <= set(archive.namelist()):
                raise RuntimeError('Invalid paired monitor evidence: ' + str(paired))
        return row
    if paired.exists():
        # A kill between the NPZ and JSON writes is not a completed check.
        partial = output / 'interrupted' / str(time.time_ns())
        partial.mkdir(parents=True)
        shutil.copy2(paired, partial / paired.name)
    save_policy(steps, params)
    result = monitor(steps, params)
    if result['state'] != 'PASS':
        raise RuntimeError('Boundary monitor did not pass')
    return result


def replay_boundary_monitor(monitor, *, updates, within_epoch, steps, budget, params, save_policy):
    if monitor is None:
        return None
    if updates == 0:
        return complete_boundary_monitor(monitor, 0, params, save_policy)
    if within_epoch == 0 or steps >= budget:
        return complete_boundary_monitor(monitor, steps, params, save_policy)
    return None
