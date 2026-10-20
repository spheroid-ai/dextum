"""Continue the two registered mechanical controls after a dense-skin OOM.

Only orchestration changes. Original registrations, source, model, training
budget, detector and monitor guard remain frozen.
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
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    assert reg['authority'] == 'review_section16_independent_hand_controls'
    base_path = Path(reg['base_registration'])
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == reg['base_sha256']
    base = json.loads(base_path.read_text())
    previous = json.loads(Path(reg['previous_pipeline']).read_text())
    assert previous['state'] == 'EXECUTION_STOP' and previous['current'] == 'dense_qualify'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == reg['orchestrator_sha256']
    deadline = reg['deadline']
    a.run.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', completed=[], archives=[], deadline=deadline,
                 dense='GPU_MEMORY_EXECUTION_FAILURE_REQUIRES_LARGER_DEVICE', attempts=0)
    def save():
        (a.run/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID']
    try:
        for case in ('stiff_10', 'thin_stiff_10'):
            job = base['cases'][case]
            for mode, seconds in [('audit', 1800), ('qualify', 1800), ('train', 7200)]:
                if mode == 'train':
                    seconds = min(seconds, max(0, int(deadline-time.time())-120))
                    q = json.loads((Path(job['qualification'])/'throughput.json').read_text())
                    if q['state'] != 'PASS' or q['projected_rounded_budget_s'] > seconds*2/3:
                        state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode)
                        return
                if deadline-time.time() < seconds+120:
                    state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode)
                    return
                label = case+'_'+mode
                out = Path(job[{'audit':'detector', 'qualify':'qualification', 'train':'training'}[mode]])
                if mode in ('audit', 'qualify'):
                    out = out.parent
                assert not out.exists(), out
                state.update(current=label); save()
                result = run(a.run, label, ['-m', 'pass1c.followup', '--registration', str(base_path),
                             '--case', case, '--mode', mode, '--output', str(out)], seconds, rss_gib=56)
                if result['exit_code']:
                    state.update(state='EXECUTION_STOP', failure=result)
                    return
                state['completed'].append(label); save()
                if mode == 'train':
                    state['archives'].append(archive(out/'full_state/latest.pkl')); save()
        state.update(state='HAND_CONTROLS_COMPLETE')
    except BaseException as exc:
        state.update(state='EXECUTION_STOP', error=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_at'] = time.time(); save()
        for _ in range(10):
            try:
                request = Request('https://console.vast.ai/api/v0/instances/'+instance+'/', method='PUT',
                    data=json.dumps({'state':'stopped'}).encode(), headers={
                    'Authorization':'Bearer '+boot[b'CONTAINER_API_KEY'].decode(), 'Content-Type':'application/json'})
                with urlopen(request, timeout=20) as response:
                    result = json.load(response)
                assert result.get('success') is not False
                (a.run/'INSTANCE_FINISH.json').write_text(json.dumps(dict(
                    state='STOP_REQUEST_ACCEPTED', instance_id=instance, time=time.time()))+'\n')
                break
            except Exception:
                time.sleep(20)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
