"""Bounded remote section-7 supervisor; no simulator imports here."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from pass0.review_gate import require_review_pass, exact_mcnemar


def main():
    assert os.environ.get('CONTAINER_ID'), 'Remote only'
    out = Path(sys.argv[1]); root = out.parent
    result = dict(state='WAITING_FOR_PREFLIGHT', training_attempts_started=0)
    def save():
        result['updated_at'] = time.time()
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    save()
    try:
        while not (root/'bootstrap/EXIT_CODE').exists(): time.sleep(5)
        assert (root/'bootstrap/EXIT_CODE').read_text().strip() == '0', 'Bootstrap failed'
        assert json.loads((root/'bootstrap/preflight.json').read_text())['state'] == 'PASS'
        qual = json.loads(Path('../dextum_plan/progress/2026-10-10-section7/D5_QUALIFICATION.json').read_text())
        require_review_pass(qual)
        result.update(state='E3_THEN_EVALUATORS_THEN_DEMO', D5='PASS'); save()
        folder = root/'section7_execute_01'; folder.mkdir(exist_ok=False)
        with (folder/'run.log').open('xb') as log:
            proc = subprocess.Popen(['bash', 'pass0/section7_execute_01.sh', str(folder)], stdout=log, stderr=log)
            (folder/'pid').write_text(str(proc.pid))
            code = proc.wait()
        (folder/'EXIT_CODE').write_text(str(code)+'\n')
        result['execution_exit_code'] = code
        result['training_attempts_started'] = int((folder/'e3/c_touch/config.json').exists())
        result['stage_codes'] = {p.stem:int(p.read_text()) for p in folder.glob('*_EXIT_CODE')}
        result['state'] = 'MAIN_RESULTS_READY' if all(result['stage_codes'].get(k) == 0 for k in
            ('training_EXIT_CODE', 'evaluation_EXIT_CODE', 'demo_EXIT_CODE')) else 'EXECUTION_INCOMPLETE'
        first, second = folder/'control_default/result.json', folder/'control_no_constant_folding/result.json'
        if first.exists() and second.exists():
            a, b = json.loads(first.read_text()), json.loads(second.read_text())
            assert a['finite'] and b['finite']
            for k in ('n', 'steps', 'checkpoint_sha256', 'source_sha256', 'bank_sha256', 'sign'):
                assert a[k] == b[k], 'Control mismatch: '+k
            x, y = a['outcomes'], b['outcomes']
            only_a = sum(i and not j for i,j in zip(x,y)); only_b = sum(j and not i for i,j in zip(x,y))
            control = dict(state='MEASURED', n=200, default_success=sum(x), alternate_success=sum(y),
                agreement=sum(i == j for i,j in zip(x,y))/200, default_only=only_a, alternate_only=only_b,
                exact_mcnemar_p=exact_mcnemar(only_a, only_b), nonblocking=True,
                interpretation='One paired compiler control; no universal noise floor inferred')
            (folder/'compiler_control.json').write_text(json.dumps(control, indent=2)+'\n')
            result['compiler_control'] = control
        save()
    except Exception as exc:
        result.update(state='PIPELINE_ERROR', error=type(exc).__name__+': '+str(exc)); save()
    # Fetch grace; the independent instance deadline remains the hard fallback.
    time.sleep(180)
    env = dict(os.environ)
    for item in Path('/proc/1/environ').read_bytes().split(b'\0'):
        k, _, v = item.partition(b'=')
        if k in (b'CONTAINER_ID', b'CONTAINER_API_KEY'): env[k.decode()] = v.decode()
    request = Request('https://console.vast.ai/api/v0/instances/'+env['CONTAINER_ID']+'/', method='PUT',
        data=json.dumps({'state':'stopped'}).encode(),
        headers={'Authorization':'Bearer '+env['CONTAINER_API_KEY'], 'Content-Type':'application/json'})
    for _ in range(3):
        try:
            with urlopen(request, timeout=20) as response: answer = json.load(response)
            if answer.get('success') is not False: return
        except Exception: pass
        time.sleep(15)


if __name__ == '__main__': main()
