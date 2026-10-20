"""Small stdlib-only monitor/transfer tools for the retained Vast instance."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from ..pass0 import section7_cloud as previous

api=previous.api
ROOT=previous.ROOT
RECORD=ROOT/'dextum_plan/progress/2026-10-10-pass1/cloud'
OUTPUT=ROOT/'dextum_cursor_agent/runs/pass1_audit_01'
REMOTE='/root/dextum/dextum_cursor_agent/runs/pass1_audit_01'
api.RECORD=RECORD


def fetch():
    cmd,_,host,_=api.connection();OUTPUT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['rsync','-a','--partial','--timeout=45','-e',shlex.join(cmd[:-1]),
                'root@'+host+':'+REMOTE+'/',str(OUTPUT)+'/'],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    # Preserve small progress records beside the Pass 1 register as requested.
    import shutil
    for f in OUTPUT.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.jsonl','.log','.txt','.EXIT_CODE'):
            target=RECORD.parent/'results'/f.relative_to(OUTPUT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    return str(OUTPUT)


def status():
    row=api.instance()
    result={k:row.get(k) for k in ('id','label','actual_status','dph_total')}
    if row['actual_status']=='running':
        code="""from pathlib import Path
import json
p=Path(REMOTE)
results={str(f.relative_to(p)):json.loads(f.read_text()) for f in p.glob('*/result.json')}
logs={f.name:f.read_text()[-1400:] for f in p.glob('*.log')}
print(json.dumps(dict(results=results,logs=logs)))
""".replace('REMOTE',repr(REMOTE))
        result['jobs']=json.loads(api.remote('python3 -c '+shlex.quote(code)))
    api.write(RECORD/'latest_status.json',dict(time=time.time(),**result))
    return result
