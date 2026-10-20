"""Finish a capped diagnostic without changing the frozen training code."""
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
    assert cap['authority'] == 'review_section16_complete_capped_qualification'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == cap['orchestrator_sha256']
    root = a.capacity.parent
    registration = root/'REGISTRATION.json'
    assert hashlib.sha256(registration.read_bytes()).hexdigest() == cap['registration_sha256']
    reg = json.loads(registration.read_text()); job = reg['cases']['dense']
    state = dict(state='RUNNING', completed=[], archives=[], deadline=cap['deadline'])
    def save():
        (root/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(cap['instance_id'])
    try:
        previous = json.loads(Path(cap['previous_state']).read_text())
        assert previous['state'] == 'EXECUTION_STOP' and previous['failure']['reason'] == 'time_cap'
        old = Path(job['qualification'])
        assert json.loads((old/'resume_test.json').read_text())['state'] == 'PASS'
        assert cap['deadline']-time.time() > 3720
        state['current'] = 'preserve_qualification'; save()
        state['archives'].append(archive(old/'latest.pkl')); save()
        # A killed atomic write is not a continuation. Retain its size in the
        # failure record; the preceding committed checkpoint is archived above.
        partial = old/'latest.tmp'
        if partial.exists():
            state['discarded_incomplete_write'] = dict(path=str(partial), bytes=partial.stat().st_size)
            save(); partial.unlink()
        output = Path(cap['qualification_output'])
        output.mkdir(parents=True, exist_ok=False)
        (output/'checkpoints.json').write_text('[]\n')
        state['current'] = 'dense_qualify_continuation'; save()
        result = run(root, 'dense_qualify_continuation', ['-m', 'pass1c.followup', '--registration', str(registration),
            '--case', 'dense', '--mode', 'qualify', '--output', str(output), '--resume', str(old/'latest.pkl.gz')],
            3600, rss_gib=112)
        if result['exit_code']:
            state.update(state='EXECUTION_STOP', failure=result); return
        rate_path = output/'full_state/throughput.json'
        rate = json.loads(rate_path.read_text())
        proof = json.loads((output/'full_state/resume_test.json').read_text())
        assert rate['updates'] == 10 and proof['state'] == 'PASS'
        if rate['state'] != 'PASS':
            state.update(state='THROUGHPUT_STOP', qualification=rate); return
        assert not (old/'throughput.json').exists()
        rate['measurement_source'] = str(rate_path)
        rate['prior_cap_stop_preserved'] = cap['previous_state']
        (old/'throughput.json').write_text(json.dumps(rate, indent=2)+'\n')
        state['completed'].append('dense_qualify_continuation')
        state['archives'].append(archive(output/'full_state/latest.pkl')); save()
        for mode, seconds in [('train',7200), ('t1',900)]:
            if mode == 'train':
                seconds = min(seconds, max(0, int(cap['deadline']-time.time())-120))
                if rate['projected_rounded_budget_s'] > seconds*2/3:
                    state.update(state='LEASE_BOUNDARY', next_mode=mode); return
            if cap['deadline']-time.time() < seconds+120:
                state.update(state='LEASE_BOUNDARY', next_mode=mode); return
            out = Path(job[{'train':'training', 't1':'t1'}[mode]])
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
