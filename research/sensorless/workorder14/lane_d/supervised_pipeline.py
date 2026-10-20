"""Bounded D2 or D3 lane child with the existing per-host qualifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU device for host callbacks; model execution must remain on GPU.
    from pass1c.bounded import run
    from pass1c.matched_batch import archive
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--lease', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    lease = json.loads(a.lease.read_text())
    digest = hashlib.sha256(a.registration.read_bytes()).hexdigest()
    assert lease['registration_sha256'] == digest
    assert reg['authority'] in ('work_order_2026_10_14_lane_D2', 'work_order_2026_10_14_lane_D3')
    module = 'train_dagger' if reg['authority'].endswith('D2') else 'train_encoder'
    instance = os.environ['CONTAINER_ID']
    assert str(lease['instance_id']) == instance
    timer = json.loads(Path('/root/kit_b_deadline.json').read_text())
    assert timer['state'] == 'ARMED' and str(timer['instance_id']) == instance
    assert timer['deadline'] == lease['deadline']
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    assert boot[b'CONTAINER_ID'].decode() == instance
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', registration_sha256=digest, instance_id=instance,
                 completed=[], training_children=0, deadline=lease['deadline'])

    def save():
        path = a.output / 'PIPELINE_STATE.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)

    def remaining():
        return lease['deadline'] - time.time()

    try:
        for mode in ('interfaces', 'audit', 'qualify', 'resume_check', 'train'):
            if remaining() < 2100:
                state.update(state='LEASE_BOUNDARY', next_mode=mode)
                return
            if mode == 'interfaces':
                command = ['-m', 'workorder14.lane_d.test_interfaces_gpu', '--model', reg['model'],
                           '--teacher', reg['teacher'], '--output', str(a.output / 'INTERFACES.json')]
                output = a.output
            else:
                output = Path(reg[dict(audit='detector', qualify='qualification',
                    resume_check='fresh_process_resume', train='training')[mode]])
                if mode in ('audit', 'qualify'):
                    output = output.parent
                elif mode == 'resume_check':
                    output = output.parent.parent
                assert not output.exists(), 'Preserve historical run: ' + str(output)
                command = ['-m', 'workorder14.lane_d.' + module, '--registration', str(a.registration),
                           '--mode', mode, '--output', str(output)]
            if mode == 'train':
                rate = json.loads((Path(reg['qualification']) / 'throughput.json').read_text())
                if remaining() < rate['projected_rounded_budget_s'] + 1200:
                    state.update(state='LEASE_BOUNDARY', next_mode=mode)
                    return
                state['training_children'] = 1
            state['current_mode'] = mode
            save()
            result = run(a.output, mode, command, 7200 if mode == 'train' else 1800,
                         rss_gib=lease['rss_cap_gib'])
            if mode == 'train' and result['exit_code'] and result['reason'] == 'time_cap':
                checkpoint = output / 'full_state/latest.pkl'
                guards = list((output / 'monitor').glob('*.json'))
                if checkpoint.is_file() and guards and all(json.loads(p.read_text())['state'] == 'PASS'
                                                            for p in guards) and remaining() > 7500:
                    state['training_children'] = 2
                    save()
                    result = run(a.output, 'train_continuation', command + ['--resume', str(checkpoint)],
                                 7200, rss_gib=lease['rss_cap_gib'])
            if result['exit_code']:
                state.update(state='EXECUTION_STOP', failure_mode=mode, failure=result)
                return
            state['completed'].append(mode)
            save()
        training = Path(reg['training'])
        assert json.loads((training / 'report.json').read_text())['state'] == 'COMPLETE'
        state['archive'] = archive(training / 'full_state/latest.pkl')
        if module == 'train_dagger' and remaining() > 600:
            result = run(a.output, 'export', ['-m', 'workorder14.lane_d.train_dagger', '--registration',
                         str(a.registration), '--mode', 'export', '--output', reg['export']],
                         300, rss_gib=lease['rss_cap_gib'])
            if result['exit_code']:
                state.update(state='EXPORT_STOP', failure=result)
                return
            state['completed'].append('export')
        state['state'] = 'COMPLETE'
    except BaseException as exc:
        state.update(state='EXECUTION_STOP', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_at'] = time.time()
        save()
        request = Request('https://console.vast.ai/api/v0/instances/' + instance + '/', method='PUT',
            data=b'{"state":"stopped"}', headers={'Authorization': 'Bearer ' + boot[b'CONTAINER_API_KEY'].decode(),
                                                  'Content-Type': 'application/json'})
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
        assert result.get('success') is not False
        (a.output / 'INSTANCE_STOP.json').write_text(json.dumps(dict(state='STOP_REQUEST_ACCEPTED',
            instance_id=instance, time=time.time())) + '\n')


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    main()
