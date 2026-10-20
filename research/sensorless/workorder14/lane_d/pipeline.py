"""Bounded D1 GPU queue. A stopped mask does not cancel independent masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Host diagnostics require a CPU callback device.
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
    assert reg['state'] == 'REGISTERED' and reg['authority'] == 'work_order_2026_10_14_lane_D1'
    assert lease['registration_sha256'] == digest
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(lease['instance_id'])
    timer = json.loads(Path('/root/kit_b_deadline.json').read_text())
    assert timer['state'] == 'ARMED' and timer['deadline'] == lease['deadline']
    assert str(timer['instance_id']) == instance
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', lane='D1', instance_id=instance, registration_sha256=digest,
                 deadline=lease['deadline'], completed=[], stopped={}, archives=[], cases={})

    def save():
        target = a.output / 'PIPELINE_STATE.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(target)

    def time_left():
        return lease['deadline'] - time.time()

    try:
        for case in reg['case_order']:
            job = reg['cases'][case]
            state['current_case'] = case
            state['cases'][case] = dict(completed_modes=[], training_children=0)
            row = state['cases'][case]
            save()
            for mode in ('audit', 'qualify', 'resume_check', 'train'):
                key = dict(audit='detector', qualify='qualification', resume_check='fresh_process_resume',
                           train='training')[mode]
                output = Path(job[key])
                if mode in ('audit', 'qualify'):
                    output = output.parent
                elif mode == 'resume_check':
                    output = output.parent.parent
                if output.exists():
                    raise FileExistsError('Preserve existing evidence; a new run root is required: ' + str(output))
                if time_left() < 2100:
                    state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode)
                    return
                command = ['-m', 'workorder14.lane_d.train_masking', '--registration', str(a.registration),
                           '--case', case, '--mode', mode, '--output', str(output)]
                if mode == 'resume_check':
                    command += ['--resume', str(Path(job['qualification']) / 'resume_test.pkl.gz')]
                cap = 7200 if mode == 'train' else 1800
                if mode == 'train':
                    rate = json.loads((Path(job['qualification']) / 'throughput.json').read_text())
                    if time_left() < rate['projected_rounded_budget_s'] + 1200:
                        state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode)
                        return
                    row['training_children'] = 1
                state['current_mode'] = mode
                save()
                result = run(a.output, case + '_' + mode, command, cap, rss_gib=lease['rss_cap_gib'])
                if mode == 'train' and result['exit_code'] and result['reason'] == 'time_cap':
                    checkpoint = output / 'full_state/latest.pkl'
                    guards = list((output / 'monitor').glob('*.json'))
                    safe_resume = checkpoint.is_file() and guards and all(
                        json.loads(path.read_text())['state'] == 'PASS' for path in guards)
                    if safe_resume and time_left() > 7500:
                        row['training_children'] = 2
                        save()
                        result = run(a.output, case + '_continuation', command + ['--resume', str(checkpoint)],
                                     7200, rss_gib=lease['rss_cap_gib'])
                if result['exit_code']:
                    row.update(state='EXECUTION_STOP', failure_mode=mode, failure=result)
                    state['stopped'][case] = dict(mode=mode, result=result)
                    save()
                    break
                row['completed_modes'].append(mode)
                save()
            else:
                report = json.loads((Path(job['training']) / 'report.json').read_text())
                assert report['state'] == 'COMPLETE'
                final = Path(job['training']) / 'monitor' / (str(job['final_steps']) + '.json')
                assert json.loads(final.read_text())['state'] == 'PASS'
                row.update(state='COMPLETE', final=json.loads(final.read_text()))
                state['archives'].append(archive(Path(job['training']) / 'full_state/latest.pkl'))
                state['completed'].append(case)
                save()
        # Statistics are run on this GPU host as well; the executable asserts GPU.
        if time_left() > 300:
            result = run(a.output, 'report', ['-m', 'workorder14.lane_d.report_d1',
                         '--registration', str(a.registration), '--output', str(a.output / 'RESULT.json')],
                         180, rss_gib=lease['rss_cap_gib'])
            state['report_exit'] = result
        state['state'] = 'COMPLETE' if not state['stopped'] else 'COMPLETE_WITH_PARKED_ROWS'
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
            answer = json.load(response)
        assert answer.get('success') is not False
        (a.output / 'INSTANCE_STOP.json').write_text(json.dumps(dict(state='STOP_REQUEST_ACCEPTED',
            instance_id=instance, time=time.time())) + '\n')


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    main()
