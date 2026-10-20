"""One bounded confirmation phase; teacher cap continuations keep one attempt."""
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
    reg = json.loads(a.registration.read_text())
    lease = json.loads(a.lease.read_text())
    assert reg['authority'] == 'review_section16_confirmation'
    assert lease['registration_sha256'] == hashlib.sha256(a.registration.read_bytes()).hexdigest()
    assert time.time() < lease['deadline']
    a.run.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', completed=[], archives=[], deadline=lease['deadline'])
    state.update(registration_sha256=lease['registration_sha256'], phase=lease['phase'],
                 training_children={})
    previous = None
    if lease.get('previous_run'):
        prior_path = Path(lease['previous_run'])/'PIPELINE_STATE.json'
        previous = json.loads(prior_path.read_text())
        assert previous['state'] == 'LEASE_BOUNDARY'
        assert previous['registration_sha256'] == state['registration_sha256']
        assert previous['phase'] == state['phase']
        state['training_children'] = dict(previous['training_children'])
        state['previous_state_sha256'] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    def save():
        (a.run/'PIPELINE_STATE.json').write_text(json.dumps(state, indent=2)+'\n')
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance = boot[b'CONTAINER_ID'].decode()
    assert instance == os.environ['CONTAINER_ID'] == str(lease['instance_id'])
    try:
        for case in reg['phases'][lease['phase']]:
            job = reg['cases'][case]
            for mode in ('audit', 'qualify', 'train'):
                out = Path(job[{'audit':'detector', 'qualify':'qualification', 'train':'training'}[mode]])
                if mode in ('audit', 'qualify'):
                    out = out.parent
                    if job.get('qualification_case', case) != case:
                        assert out.exists(), out
                        continue
                label = case+'_'+mode
                resume_path = None
                if out.exists():
                    # Only a recorded lease boundary permits continuation of
                    # this phase. Guard failures are never silently resumed.
                    assert previous is not None, out
                    if mode == 'audit':
                        observed = json.loads((out/'result.json').read_text())
                        assert observed['state'] == 'PASS'
                        assert observed['registration_sha256'] == state['registration_sha256']
                        state['completed'].append(label); save(); continue
                    if mode == 'qualify':
                        observed = json.loads((out/'full_state/resume_test.json').read_text())
                        rate = json.loads((out/'full_state/throughput.json').read_text())
                        assert observed['state'] == 'PASS' and rate['updates'] == 10
                        assert rate['metadata']['registration_sha256'] == state['registration_sha256']
                        assert rate['projected_rounded_budget_s'] <= job['maximum_projected_budget_s']
                        state['completed'].append(label); save(); continue
                    report_path = out/'report.json'
                    report = json.loads(report_path.read_text()) if report_path.exists() else {}
                    if report.get('state') == 'COMPLETE':
                        assert report['registration_sha256'] == state['registration_sha256']
                        assert report['job'] == job
                        state['completed'].append(label); save(); continue
                    assert job['phase'] == 'teacher'
                    assert previous['next_case'] == case and previous['next_mode'] == 'resume'
                    assert state['training_children'].get(case) == 1
                    assert not report or report['state'] != 'EXECUTION_FAILURE'
                    resume_path = Path(previous['checkpoint'])
                    assert resume_path == out/'full_state/latest.pkl' and resume_path.is_file()
                    assert all(json.loads(path.read_text())['state'] == 'PASS'
                               for path in (out/'monitor').glob('*.json'))
                if mode != 'train':
                    seconds = 1800
                else:
                    q = json.loads((Path(job['qualification'])/'throughput.json').read_text())
                    resume = json.loads((Path(job['qualification'])/'resume_test.json').read_text())
                    assert resume['state'] == 'PASS'
                    projected = q['projected_rounded_budget_s']
                    assert projected <= job['maximum_projected_budget_s']
                    children = 2 if job['phase'] == 'teacher' else 1
                    available = min(children*7200, lease['deadline']-time.time()-120)
                    if projected > available*2/3:
                        state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode); return
                    seconds = min(7200, max(0, int(lease['deadline']-time.time())-120))
                if lease['deadline']-time.time() < seconds+120:
                    state.update(state='LEASE_BOUNDARY', next_case=case, next_mode=mode); return
                state['current'] = label; save()
                args = ['-m', 'pass1c.confirmation', '--registration', str(a.registration),
                        '--case', case, '--mode', mode, '--output', str(out)]
                if resume_path is not None:
                    args += ['--resume', str(resume_path)]
                if mode == 'train':
                    state['training_children'][case] = state['training_children'].get(case, 0)+1
                    assert state['training_children'][case] <= (2 if job['phase'] == 'teacher' else 1)
                    save()
                result = run(a.run, label, args, seconds, rss_gib=lease['rss_cap_gib'])
                if (result['exit_code'] and mode == 'train' and job['phase'] == 'teacher'
                        and result['reason'] == 'time_cap' and state['training_children'][case] == 1):
                    # Full-budget teachers are registered to span at most two
                    # capped children. Reuse the complete checkpoint verbatim.
                    checkpoint = out/'full_state/latest.pkl'
                    assert checkpoint.is_file()
                    remaining = int(lease['deadline']-time.time())-120
                    if remaining < 7200:
                        state.update(state='LEASE_BOUNDARY', next_case=case, next_mode='resume',
                                     checkpoint=str(checkpoint)); return
                    state.update(current=label+'_continuation', continuation_of_same_attempt=True); save()
                    state['training_children'][case] = 2; save()
                    result = run(a.run, label+'_continuation', args+['--resume', str(checkpoint)],
                                 7200, rss_gib=lease['rss_cap_gib'])
                if result['exit_code']:
                    state.update(state='EXECUTION_STOP', failure=result); return
                state['completed'].append(label); save()
                if mode == 'train':
                    state['archives'].append(archive(out/'full_state/latest.pkl')); save()
        state['state'] = 'CONFIRMATION_PHASE_COMPLETE'
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
                    state='STOP_REQUEST_ACCEPTED', instance_id=instance, time=time.time()))+'\n'); break
            except Exception:
                time.sleep(20)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
