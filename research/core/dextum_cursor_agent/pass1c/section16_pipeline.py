"""Bounded §16 qualification, isolated errors and remaining rotation masks."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    from pass1c.bounded import run
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--masking-run', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--instance', required=True)
    a = p.parse_args()
    assert os.environ['CONTAINER_ID'] == a.instance
    boot = dict(x.split(b'=', 1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    assert boot[b'CONTAINER_ID'].decode() == a.instance
    key = boot[b'CONTAINER_API_KEY'].decode()
    a.run.mkdir(parents=True, exist_ok=True)
    state = dict(state='RUNNING', instance_id=a.instance, deadline=a.deadline, completed=[])
    target = a.run / 'PIPELINE_STATE.json'

    def save():
        target.write_text(json.dumps(state, indent=2) + '\n')

    def child(name, args, seconds=1800, cpu=False):
        if a.deadline-time.time() < seconds+120:
            raise RuntimeError('Lease boundary before '+name)
        state.update(current=name)
        save()
        if cpu:
            os.environ['JAX_PLATFORMS'] = 'cpu'
        try:
            result = run(a.run, name, args, seconds, rss_gib=56)
        finally:
            os.environ.pop('JAX_PLATFORMS', None)
        if result['exit_code']:
            raise RuntimeError(name+' failed: '+str(result))
        state['completed'].append(name)
        save()

    try:
        child('sensor_tests', ['-m', 'unittest', 'pass1c.test_ideal_errors',
                              'pass1c.test_criteria', 'pass1c.test_masking', '-v'], 300, cpu=True)
        child('replacement_detector', ['-m', 'pass1.audit_training', '--kit', '--section13',
              '--variant', 'barrier', '--probe', '--output', str(a.run/'detector')], 1800)
        detector = json.loads((a.run/'detector/result.json').read_text())
        assert detector['state'] == 'PASS' and detector['matched_step_median'] < 1e-4
        child('isolated_errors', ['-m', 'pass1c.hardware_sweep', '--registration',
              str(a.run/'ISOLATED_SWEEP_REGISTRATION.json'), '--output', str(a.run/'isolated_errors')], 3600)
        # The existing registered batch skips every completed case. Its training
        # source, model, checkpoint and statistical criteria remain unchanged.
        from pass1c import masking_batch
        masking_batch.run = lambda root, name, args, seconds: run(root, name, args, seconds, rss_gib=56)
        state.update(current='remaining_masking')
        save()
        sys.argv = ['masking_batch', '--run', str(a.masking_run), '--deadline', str(a.deadline)]
        masking_batch.main()
        masking = json.loads((a.masking_run/'BATCH_STATE.json').read_text())
        state.update(state=('ROTATION_MASKS_AND_ISOLATED_SWEEPS_COMPLETE' if masking['state']=='COMPLETE'
                            else masking['state']), masking_state=masking['state'],
                     remaining=['Dense-skin 4a', 'required-signal attribution if applicable',
                                'two hand-cost runs', 'confirmation seeds 2–3'])
        # Reporting must not interfere with the stop below.
        child('masking_report', ['-m', 'pass1c.masking_results', '--run', str(a.masking_run)], 60)
    except Exception as exc:
        state.update(state='EXECUTION_STOP', error_type=type(exc).__name__, error=str(exc))
    finally:
        state['finished_at'] = time.time()
        save()
        for _ in range(10):
            try:
                request = Request('https://console.vast.ai/api/v0/instances/'+a.instance+'/', method='PUT',
                                  data=json.dumps({'state':'stopped'}).encode(),
                                  headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
                with urlopen(request, timeout=20) as response:
                    result = json.load(response)
                assert result.get('success') is not False
                (a.run/'INSTANCE_FINISH.json').write_text(json.dumps(dict(state='STOP_REQUEST_ACCEPTED',
                      instance_id=a.instance, time=time.time(), pipeline_state=state['state']), indent=2)+'\n')
                break
            except Exception:
                time.sleep(20)


if __name__ == '__main__':
    main()
