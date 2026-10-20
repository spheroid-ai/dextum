"""Remote bounded review supervisor; no numerical imports in this process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from pass0.review_gate import require_review_pass


def main():
    assert os.environ.get('CONTAINER_ID'),'Remote only'
    out=Path(sys.argv[1]);root=out.parent
    result=dict(state='WAITING_FOR_PREFLIGHT',training_attempts_started=0)
    def save():
        result['updated_at']=time.time()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    def launch(name):
        folder=root/name;folder.mkdir(exist_ok=False)
        log=(folder/'run.log').open('xb')
        proc=subprocess.Popen(['bash','pass0/'+name+'.sh',str(folder)],stdout=log,stderr=log)
        (folder/'pid').write_text(str(proc.pid))
        return proc
    def finish(name,proc):
        code=proc.wait();(root/name/'EXIT_CODE').write_text(str(code)+'\n');return code
    save()
    while not (root/'bootstrap/EXIT_CODE').exists():time.sleep(5)
    if (root/'bootstrap/EXIT_CODE').read_text().strip()!='0':
        result.update(state='BOOTSTRAP_FAILED');save();return
    assert json.loads((root/'bootstrap/preflight.json').read_text())['state']=='PASS'
    result.update(state='FIXED_POLICY_EVALUATION');save()
    evaluation=launch('review_eval_01')
    eval_code=finish('review_eval_01',evaluation)
    result['evaluation_exit_code']=eval_code
    # The probe is independent of acceptance. It never vetoes execution.
    diagnostic=launch('review_probe_01')
    gate=False
    if eval_code==0:
        evidence=json.loads((root/'review_eval_01/evaluation/paired_evaluators.json').read_text())
        try:require_review_pass(evidence);gate=True
        except RuntimeError:pass
        result['D5']=evidence['state']
    if gate:
        result.update(state='TRAINING_AND_DEMO');save()
        execute=launch('review_execute_01')
        result['execute_exit_code']=finish('review_execute_01',execute)
        result['training_attempts_started']=int((root/'review_execute_01/e3/c_touch/config.json').exists())
    else:
        result.update(state='EVALUATOR_AGREEMENT_FAILED' if eval_code==0 else 'EVALUATOR_ERROR',
                      downstream='Not started: fixed-policy agreement has not passed')
        save()
    result['diagnostic_exit_code']=finish('review_probe_01',diagnostic)
    if gate:result['state']='EXECUTION_FINISHED' if result['execute_exit_code']==0 else 'EXECUTION_ERROR'
    save()
    # Leave time to fetch; independent instance timer remains the fallback.
    time.sleep(120)
    env=dict(os.environ)
    for item in Path('/proc/1/environ').read_bytes().split(b'\0'):
        k,_,v=item.partition(b'=')
        if k in (b'CONTAINER_ID',b'CONTAINER_API_KEY'):env[k.decode()]=v.decode()
    request=Request('https://console.vast.ai/api/v0/instances/'+env['CONTAINER_ID']+'/',method='PUT',
        data=json.dumps({'state':'stopped'}).encode(),headers={'Authorization':'Bearer '+env['CONTAINER_API_KEY'],'Content-Type':'application/json'})
    for _ in range(3):
        try:
            with urlopen(request,timeout=20) as response:answer=json.load(response)
            if answer.get('success') is not False:return
        except Exception:pass
        time.sleep(15)


if __name__=='__main__':main()
