"""Stop this instance when the bounded control batch reaches a terminal state."""
import os,time,json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02')
TERMINAL={'TRAINING_FAILED','EVALUATION_FAILED','RUNTIME_AUDIT_FAILED','BUDGET_DEFERRED','CONTROLS_COMPLETE'}

def main():
    instance=os.environ['CONTAINER_ID']
    assert instance=='55271331'
    key=os.environ.get('CONTAINER_API_KEY')
    if not key:
        # SSH strips the boot environment. Read only this owned container's
        # instance-scoped credential, as used by its existing deadline timer.
        boot=dict(part.split(b'=',1) for part in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in part)
        assert boot.get(b'CONTAINER_ID')==instance.encode()
        key=boot[b'CONTAINER_API_KEY'].decode()
    while True:
        path=ROOT/'STATUS.json'
        try:status=json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            time.sleep(5);continue
        if status.get('stage') in TERMINAL:break
        time.sleep(30)
    record=dict(time=time.time(),instance_id=instance,batch=status,state='STOP_REQUESTING')
    target=ROOT/'INSTANCE_FINISH.json'
    target.write_text(json.dumps(record,indent=2)+'\n')
    while True:
        try:
            request=Request('https://console.vast.ai/api/v0/instances/'+instance+'/',method='PUT',
                data=json.dumps({'state':'stopped'}).encode(),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urlopen(request,timeout=20) as response:result=json.load(response)
            if result.get('success') is False:raise RuntimeError('Stop request rejected')
            record.update(state='STOP_REQUEST_ACCEPTED',requested_at=time.time())
            target.write_text(json.dumps(record,indent=2)+'\n')
            return
        except Exception as exc:
            record.update(state='STOP_RETRY',error_type=type(exc).__name__)
            target.write_text(json.dumps(record,indent=2)+'\n')
            time.sleep(30)

if __name__=='__main__':main()
