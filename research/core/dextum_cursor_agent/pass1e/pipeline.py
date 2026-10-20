"""Bounded §18 scratch run with same-attempt continuation and provider stop."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--lease', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    reg, lease = [json.loads(x.read_text()) for x in (a.registration, a.lease)]
    digest = hashlib.sha256(a.registration.read_bytes()).hexdigest()
    assert reg['authority'] == 'review_section18_from_scratch'
    assert lease['registration_sha256'] == digest
    case = lease['case']; job = reg['cases'][case]
    assert time.time() < lease['deadline']
    a.run.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', case=case, registration_sha256=digest,
                 completed=[], training_children=0, deadline=lease['deadline'])
    def save():
        (a.run/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(lease['instance_id'])
    try:
        for mode in ('audit', 'qualify', 'train'):
            out = Path(job[{'audit':'detector','qualify':'qualification','train':'training'}[mode]])
            if mode != 'train': out = out.parent
            assert not out.exists(), out
            seconds = 1800 if mode != 'train' else 7200
            if mode == 'train':
                q = Path(job['qualification'])
                rate = json.loads((q/'throughput.json').read_text())
                resume = json.loads((q/'resume_test.json').read_text())
                assert resume['state'] == 'PASS'
                assert rate['projected_rounded_budget_s'] <= 9600
                assert rate['metadata']['registration_sha256'] == digest
                # Two complete child allocations plus stop/archive margin.
                assert lease['deadline']-time.time() >= 2*7200+300
            assert lease['deadline']-time.time() > seconds+120
            state.update(current=mode); save()
            args = ['-m','pass1e.scratch','--registration',str(a.registration),
                    '--case',case,'--mode',mode,'--output',str(out)]
            if mode == 'train': state['training_children'] = 1; save()
            result = run(a.run, mode, args, seconds, rss_gib=lease['rss_cap_gib'])
            if mode == 'train' and result['exit_code'] and result['reason'] == 'time_cap':
                # A guard exception must never be converted into a cap resume.
                report = out/'report.json'
                assert not report.exists() or json.loads(report.read_text())['state'] != 'EXECUTION_FAILURE'
                checkpoint = out/'full_state/latest.pkl'
                assert checkpoint.is_file()
                assert all(json.loads(x.read_text())['state'] == 'PASS' for x in (out/'monitor').glob('*.json'))
                assert lease['deadline']-time.time() > 7200+120
                state.update(current='train_continuation',training_children=2,
                             continuation_of_same_attempt=True); save()
                result = run(a.run, 'train_continuation', args+['--resume',str(checkpoint)],
                             7200,rss_gib=lease['rss_cap_gib'])
            if result['exit_code']:
                state.update(state='EXECUTION_STOP', failure=result); return
            state['completed'].append(mode); save()
        state['archive'] = archive(Path(job['training'])/'full_state/latest.pkl')
        state['state'] = 'SCRATCH_COMPLETE'
    except BaseException as exc:
        state.update(state='EXECUTION_STOP',error=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_at'] = time.time(); save()
        for _ in range(10):
            try:
                request = Request('https://console.vast.ai/api/v0/instances/'+instance+'/',method='PUT',
                    data=json.dumps({'state':'stopped'}).encode(),headers={
                    'Authorization':'Bearer '+boot[b'CONTAINER_API_KEY'].decode(),'Content-Type':'application/json'})
                with urlopen(request, timeout=20) as response: result = json.load(response)
                assert result.get('success') is not False
                (a.run/'INSTANCE_FINISH.json').write_text(json.dumps(dict(
                    state='STOP_REQUEST_ACCEPTED',instance_id=instance,time=time.time()))+'\n')
                break
            except Exception: time.sleep(20)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'): raise RuntimeError('Vast only')
    main()
