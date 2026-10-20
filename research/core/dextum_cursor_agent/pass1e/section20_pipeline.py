"""Independent §20 scratch execution with three qualifications on this host.

The frozen scratch trainer is unchanged. Each run gets its own detector,
ten-update throughput check and fresh-process full-state replay. Only a
registered time-cap stop may resume the same training attempt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    from pass1c.bounded import run
    from pass1c.matched_batch import archive
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    lease = json.loads(args.lease.read_text())
    digest = hashlib.sha256(args.registration.read_bytes()).hexdigest()
    assert lease['registration_sha256'] == digest
    assert reg['state'] == 'REGISTERED'
    assert reg['authority'] == 'review_section18_from_scratch'
    case = lease['case']
    job = reg['cases'][case]
    assert reg['review_authority'] == 'review_section20'
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(lease['instance_id'])
    assert lease['success_action'] in ('stop', 'retain_until_deadline')
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', case=case, registration_sha256=digest,
                 completed=[], training_children=0, instance_id=int(instance),
                 deadline=lease['deadline'])

    def save():
        (args.output/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')

    try:
        for mode in ('audit', 'qualify', 'resume_check', 'train'):
            key = {'audit': 'detector', 'qualify': 'qualification',
                   'resume_check': 'fresh_process_resume', 'train': 'training'}[mode]
            output = Path(job[key])
            if mode in ('audit', 'qualify'):
                output = output.parent
            elif mode == 'resume_check':
                output = output.parent.parent
            assert not output.exists(), output
            seconds = 7200 if mode == 'train' else 1800
            assert lease['deadline']-time.time() > seconds+300
            if mode == 'train':
                qualification = Path(job['qualification'])
                resume = json.loads((qualification/'resume_test.json').read_text())
                process = json.loads(Path(job['fresh_process_resume']).read_text())
                rate = json.loads((qualification/'throughput.json').read_text())
                assert resume['state'] == 'PASS' and resume['optimizer_state_nonzero']
                assert process['state'] == 'PASS' and process['fresh_process']
                assert process['checkpoint'] == str(qualification/'resume_test.pkl.gz')
                assert rate['metadata'] == resume['metadata']
                assert rate['metadata']['registration_sha256'] == digest
                assert rate['metadata']['case'] == case
                assert rate['updates'] == 10 and rate['includes_checkpoints']
                assert rate['optimizer_updates_are_discarded']
                assert rate['projected_rounded_budget_s'] <= job['maximum_projected_budget_s'] == 9600
                assert lease['deadline']-time.time() >= 2*7200+300
                state['training_children'] = 1
            state['current'] = mode
            save()
            if mode == 'resume_check':
                command = ['-m', 'pass1e.scratch_resume_check', '--registration', str(args.registration),
                           '--case', case, '--output', str(output)]
            else:
                command = ['-m', 'pass1e.scratch', '--registration', str(args.registration),
                           '--case', case, '--mode', mode, '--output', str(output)]
            result = run(args.output, mode, command, seconds, rss_gib=lease['rss_cap_gib'])
            if mode == 'train' and result['exit_code'] and result['reason'] == 'time_cap':
                report = output/'report.json'
                assert not report.exists() or json.loads(report.read_text())['state'] != 'EXECUTION_FAILURE'
                checkpoint = output/'full_state/latest.pkl'
                assert checkpoint.is_file()
                checks = list((output/'monitor').glob('*.json'))
                assert checks and all(json.loads(p.read_text())['state'] == 'PASS' for p in checks)
                assert lease['deadline']-time.time() > 7200+300
                state.update(current='train_continuation', training_children=2,
                             continuation_of_same_attempt=True)
                save()
                result = run(args.output, 'train_continuation',
                             command+['--resume', str(checkpoint)], 7200, rss_gib=lease['rss_cap_gib'])
            if result['exit_code']:
                state.update(state='EXECUTION_STOP', failure=result)
                return
            state['completed'].append(mode)
            save()
        assert json.loads((Path(job['training'])/'report.json').read_text())['state'] == 'COMPLETE'
        state.update(state='SCRATCH_COMPLETE', training_attempts=1,
                     archive=archive(Path(job['training'])/'full_state/latest.pkl'))
    except BaseException as exc:
        state.update(state='EXECUTION_STOP', error=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_at'] = time.time()
        save()
        if state['state'] != 'SCRATCH_COMPLETE' or lease['success_action'] == 'stop':
            request = Request('https://console.vast.ai/api/v0/instances/'+instance+'/', method='PUT',
                data=b'{"state":"stopped"}', headers={
                    'Authorization': 'Bearer '+boot[b'CONTAINER_API_KEY'].decode(),
                    'Content-Type': 'application/json'})
            with urlopen(request, timeout=20) as response:
                result = json.load(response)
            assert result.get('success') is not False
            (args.output/'INSTANCE_STOP.json').write_text(json.dumps(dict(
                state='STOP_REQUEST_ACCEPTED', instance_id=int(instance), time=time.time()))+'\n')


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Cloud only')
    main()
