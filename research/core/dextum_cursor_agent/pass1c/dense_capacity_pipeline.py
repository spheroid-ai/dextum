"""Unchanged dense-skin experiment on a device with enough memory."""
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
    p.add_argument('--capacity', type=Path, required=True)
    a = p.parse_args()
    cap = json.loads(a.capacity.read_text())
    assert cap['authority'] == 'review_section16_dense_memory_replacement'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == cap['orchestrator_sha256']
    root = a.capacity.parent
    registration = root/'REGISTRATION.json'
    assert hashlib.sha256(registration.read_bytes()).hexdigest() == cap['registration_sha256']
    reg = json.loads(registration.read_text())
    assert reg['order'] == ['dense']
    job = reg['cases']['dense']
    state = dict(state='RUNNING', completed=[], archives=[], deadline=cap['deadline'])
    def save():
        (root/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(cap['instance_id'])
    try:
        state['current'] = 'boundary_tests'; save()
        os.environ['JAX_PLATFORMS'] = 'cpu'
        result = run(root, 'boundary_tests', ['-m', 'unittest', 'pass1c.test_criteria',
            'pass1c.test_ideal_errors', 'pass1c.test_masking', 'pass1c.test_dense_skin'], 600, rss_gib=112)
        os.environ.pop('JAX_PLATFORMS', None)
        if result['exit_code']:
            state.update(state='EXECUTION_STOP', failure=result); return
        state['completed'].append('boundary_tests'); save()
        for mode, seconds in [('audit',1800), ('qualify',1800), ('train',7200), ('t1',900)]:
            if mode == 'train':
                seconds = min(seconds, max(0, int(cap['deadline']-time.time())-120))
                q = json.loads((Path(job['qualification'])/'throughput.json').read_text())
                if q['state'] != 'PASS' or q['projected_rounded_budget_s'] > seconds*2/3:
                    state.update(state='LEASE_BOUNDARY', next_mode=mode); return
            if cap['deadline']-time.time() < seconds+120:
                state.update(state='LEASE_BOUNDARY', next_mode=mode); return
            out = Path(job[{'audit':'detector', 'qualify':'qualification', 'train':'training', 't1':'t1'}[mode]])
            if mode in ('audit', 'qualify'):
                out = out.parent
            assert not out.exists(), out
            state['current'] = 'dense_'+mode; save()
            result = run(root, 'dense_'+mode, ['-m', 'pass1c.followup', '--registration', str(registration),
                '--case', 'dense', '--mode', mode, '--output', str(out)], seconds, rss_gib=112)
            if result['exit_code']:
                state.update(state='EXECUTION_STOP', failure=result); return
            state['completed'].append('dense_'+mode); save()
            if mode == 'train':
                state['archives'].append(archive(out/'full_state/latest.pkl')); save()
        state['state'] = 'DENSE_COMPLETE'
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
                (root/'INSTANCE_FINISH.json').write_text(json.dumps(dict(
                    state='STOP_REQUEST_ACCEPTED', instance_id=instance, time=time.time()))+'\n'); break
            except Exception:
                time.sleep(20)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
