"""Finish the independent normal-only arm after a reviewed proximal stop."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    from pass1c.bounded import run
    from pass1c.matched_batch import archive
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    root = a.run
    reg = json.loads((root / 'INDEPENDENT_REGISTRATION.json').read_text())
    assert reg['authority'] == 'user_park_proximal_continue_independent'
    assert reg['deadline'] == a.deadline
    prior_state = json.loads((root / 'BATCH_STATE.json').read_text())
    assert prior_state['state'] == 'EXECUTION_STOP'
    check = json.loads((root / 'cases/proximal/c_touch/monitor/20316160.json').read_text())
    assert check['stop_reasons'] == ['exact_mcnemar_p_below_0.01']
    assert check['internal_success'] == 114 and check['faithful_success'] == 134
    state = dict(state='RUNNING', completed=[], parked={'proximal': 'Registered guard stop; user keeps it parked'},
                 archives=[], deadline=a.deadline)
    def save():
        (root / 'INDEPENDENT_STATE.json').write_text(json.dumps(state, indent=2) + '\n')
    base = root.parent
    previous = base / 'pass1_matched_01'
    output = root / 'cases/normal_only/c_touch'
    evaluation = root / 'evaluations/normal_only'
    common = ['--steps', '20000000', '--arm', 'c_touch', '--sensor-level', 'Realistic',
        '--warm-start', str(previous / 'cases/c_touch_realistic/c_touch/policy_20316160.pkl'),
        '--qualification', str(root / 'qualification.json'),
        '--qualification-registration', str(base / 'pass1_audit_01/CONTROL_REGISTRATION.json'),
        '--evaluator-agreement', str(base / 'pass1_d5_powered_01/policy_pair/paired_evaluators.json'),
        '--matched-registration', str(root / 'MATCHED_REGISTRATION.json'),
        '--throughput-qualification', str(previous / 'throughput/full_state/throughput.json')]
    jobs = [('mask_normal_only', ['-m', 'pass1c.train_masking', '--output', str(output),
             '--intervention', str(root / 'interventions/normal_only.json'), *common], 7200),
            ('t1_normal_only', ['-m', 'pass1c.evaluate_masking', '--case', str(output),
             '--output', str(evaluation)], 900)]
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    key = boot[b'CONTAINER_API_KEY'].decode()
    assert instance == os.environ['CONTAINER_ID']
    try:
        assert not output.exists() and not evaluation.exists()
        for label, command, cap in jobs:
            if a.deadline - time.time() < cap + 120:
                state.update(state='LEASE_BOUNDARY', next_case=label)
                return
            state.update(current=label)
            save()
            child = run(root, label, command, cap, rss_gib=56)
            if child['exit_code']:
                state.update(state='EXECUTION_STOP', failure=child)
                return
            state['completed'].append(label)
            if label == 'mask_normal_only':
                state['archives'].append(archive(output / 'full_state/latest.pkl'))
            save()
        state.update(state='INDEPENDENT_MASK_COMPLETE',
                     masking_disposition='All masks accounted for: proximal parked by user, other rows measured')
    except BaseException as exc:
        state.update(state='EXECUTION_STOP', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_at'] = time.time()
        save()
        for _ in range(10):
            try:
                request = Request('https://console.vast.ai/api/v0/instances/' + instance + '/', method='PUT',
                    data=json.dumps({'state': 'stopped'}).encode(),
                    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
                with urlopen(request, timeout=20) as response:
                    result = json.load(response)
                assert result.get('success') is not False
                break
            except Exception:
                time.sleep(20)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
