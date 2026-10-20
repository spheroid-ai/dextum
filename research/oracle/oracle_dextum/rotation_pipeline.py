"""One bounded O1 lease, with independent arm stops and exact continuation.

The lease must be explicitly bound to this frozen registration and instance.
It neither acquires a host nor extends a deadline. A qualification or monitor
failure parks that row; a time cap alone may continue its exact saved state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--lease', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from pass1c.bounded import run
    from pass1c.matched_batch import archive
    read = lambda p: json.loads(Path(p).read_text())
    reg, lease = read(args.registration), read(args.lease)
    digest = hashlib.sha256(args.registration.read_bytes()).hexdigest()
    assert reg['authority'] == 'user_oracle_dextum_full_state_o1_rotation'
    assert lease['registration_sha256'] == digest
    assert lease['rss_cap_gib'] == 56
    assert str(lease['instance_id']) == os.environ['CONTAINER_ID']
    timer = read('/root/kit_b_deadline.json')
    assert timer['state'] == 'ARMED' and timer['deadline'] == lease['deadline']
    assert lease['cases'] and len(set(lease['cases'])) == len(lease['cases'])
    assert set(lease['cases']) <= set(reg['cases'])
    assert reg['maximum_training_children'] == 2 and reg['child_cap_s'] == 7200
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', registration_sha256=digest, instance_id=lease['instance_id'],
                 completed=[], parked=[], archives=[], training_attempts=0, started_at=time.time())

    def save():
        state['updated_at'] = time.time()
        target = args.output/'PIPELINE_STATE.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(target)

    def execute(case, mode, target, name, cap, resume=None):
        assert lease['deadline']-time.time() > cap+300, 'Lease lacks child margin'
        command = ['-m', 'oracle_dextum.train_rotation', '--registration', str(args.registration),
                   '--case', case, '--mode', mode, '--output', str(target)]
        if resume: command += ['--resume', str(resume)]
        state.update(current_case=case, current_mode=mode)
        save()
        return run(args.output, name, command, cap, rss_gib=56)

    save()
    for case in lease['cases']:
        # Reserve a complete two-child training allowance plus qualification.
        # If insufficient, retain the row for a later bounded lease.
        if lease['deadline']-time.time() < 2*7200+3*1800+300:
            state.update(state='LEASE_BOUNDARY', next_case=case); save(); return
        job = reg['cases'][case]
        try:
            for mode, target in [('audit', Path(job['detector']).parent),
                                 ('qualify', Path(job['qualification'])),
                                 ('resume_check', Path(job['resume_check']))]:
                assert not target.exists(), 'Preserve previous qualification: '+str(target)
                result = execute(case, mode, target, case+'_'+mode, 1800)
                if result['exit_code'] != 0:
                    raise RuntimeError(f'{mode} failed: {result}')
                if mode == 'qualify':
                    rate = read(target/'full_state/throughput.json')
                    if rate['projected_rounded_budget_s'] > reg['maximum_projected_budget_s']:
                        raise RuntimeError('Measured throughput does not fit the registered budget')
            target = Path(job['training'])
            assert not target.exists(), 'Preserve previous training: '+str(target)
            resume = None
            for child in range(2):
                result = execute(case, 'train', target, case+'_train_'+str(child+1), 7200, resume)
                if result['exit_code'] == 0:
                    break
                if result['reason'] != 'time_cap':
                    raise RuntimeError(f'Training execution/guard failure: {result}')
                resume = target/'full_state/latest.pkl'
                if not resume.exists(): raise RuntimeError('Capped child has no resumable state')
            else:
                raise RuntimeError('Two training children exhausted; preserve checkpoint without automatic retry')
            report = read(target/'report.json')
            assert report['state'] == 'COMPLETE'
            checks = [read(path) for path in (target/'monitor').glob('*.json')]
            assert len(checks) == 31 and all(x['state'] == 'PASS' for x in checks)
            assert max(x['steps'] for x in checks) == 108134400
            state['archives'].append(dict(case=case, **archive(target/'full_state/latest.pkl')))
            state['completed'].append(case); state['training_attempts'] += 1
        except BaseException as exc:
            state['parked'].append(dict(case=case, error=f'{type(exc).__name__}: {exc}',
                disposition='Preserved; no guard waiver, no change of representation from held-out outcomes'))
        save()
    state['state'] = 'BATCH_COMPLETE' if not state['parked'] else 'BATCH_WITH_PARKED_ROWS'
    save()


if __name__ == '__main__': main()
