"""One bounded continuation of the existing Pass 0 machine (stdlib only)."""
import json
import shlex
import time
from . import cloud_ops as c


def main():
    lease = c.api.read(c.RECORD / 'lease.json')
    record = c.RECORD / 'continuation_01.json'
    assert not record.exists(), 'Reconcile an existing continuation before proceeding'
    assert c.api.instance()['actual_status'] == 'running'
    deadline = lease['expires_at'] + 7200
    c.api.write(record, dict(state='REGISTERED', time=time.time(), old_deadline=lease['expires_at'],
        deadline=deadline, additional_compute_bound_usd=2 * lease['offer']['dph_total'],
        scope='Diagnose D5; only after PASS run the three-shape demo and one original E3 C-touch arm',
        training_attempt_limit=2, unchanged_D5_thresholds=True))
    timer = shlex.split(c.passive_deadline(deadline).split(' > /root/kit_b_deadline.log', 1)[0])[3]
    script = '''import json, os, signal, subprocess, time
from pathlib import Path
p=Path('/root/dextum/dextum_cursor_agent/runs/pass0_d5_01')
assert (p/'EXIT_CODE').exists(), 'Original job still running'
old=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
    try: parts=f.read_bytes().split(b'\\0'); cmd=b' '.join(parts).decode()
    except (OSError, UnicodeDecodeError): continue
    pid=int(f.parent.name)
    if pid==os.getpid(): continue
    if (len(parts)>2 and parts[0]==b'python3' and parts[1]==b'-c' and 'deadline = OLD_DEADLINE' in cmd) or (parts[0]==b'bash' and len(parts)>1 and parts[1]==b'dextum_cursor_agent/cloud/remote/launch.sh' and 'pass0_d5_01 pass0/d5_job.sh' in cmd):
        old.append(pid)
assert old, 'Old timers not identified'
env=dict(os.environ)
for item in Path('/proc/1/environ').read_bytes().split(b'\\0'):
    key, sep, value=item.partition(b'=')
    if key in (b'CONTAINER_ID',b'CONTAINER_API_KEY'): env[key.decode()]=value.decode()
with open('/root/dextum_continuation_deadline.log','ab',buffering=0) as log:
    proc=subprocess.Popen(['python3','-c',TIMER],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
for _ in range(20):
    guard=json.loads(Path('/root/kit_b_deadline.json').read_text())
    if guard['deadline']==NEW_DEADLINE and guard['state']=='ARMED': break
    time.sleep(.5)
else: raise RuntimeError('New timer not armed; old timers left intact')
for pid in old:
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
result=dict(state='ARMED',deadline=NEW_DEADLINE,pid=proc.pid,replaced_pids=old)
(p/'continuation_01.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''.replace('OLD_DEADLINE', str(lease['expires_at'])).replace('NEW_DEADLINE',repr(deadline)).replace('TIMER',repr(timer))
    result = json.loads(c.api.remote('python3 -c '+shlex.quote(script), timeout=40))
    lease.update(expires_at=deadline, continuation=result,
                 projected_bound_usd=lease['projected_bound_usd']+2*lease['offer']['dph_total'])
    c.api.write(c.RECORD/'lease.json',lease)
    reg=c.api.read(record); reg.update(state='ARMED',remote=result); c.api.write(record,reg)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
