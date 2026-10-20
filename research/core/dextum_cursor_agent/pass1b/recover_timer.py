"""Historical failed live-recovery method. Use cloud.install_startup_timer while stopped."""
import json,os,re,shlex,subprocess,time
from . import cloud
api=cloud.api
lease=api.read(cloud.RECORD/'lease.json');row=api.instance();assert row['actual_status']=='exited'
timer=cloud.previous.previous.old.passive_deadline(lease['expires_at'])
# Provider metadata is already updated; the retained /root/onstart.sh was not.
api.write(cloud.RECORD/'stale_timer_interruption.json',dict(time=time.time(),deadline=lease['expires_at'],old_deadline=1791675855.493404,training_attempts=0))
remote='''import os,re,signal,subprocess,json
from pathlib import Path
env=dict(os.environ)
for entry in Path('/proc/1/environ').read_bytes().split(b'\\0'):
 if b'=' in entry:
  k,v=entry.split(b'=',1)
  if k in (b'CONTAINER_ID',b'CONTAINER_API_KEY'):env[k.decode()]=v.decode()
assert env.get('CONTAINER_ID')=='55271331' and env.get('CONTAINER_API_KEY')
removed=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  raw=f.read_bytes();parts=raw.split(b'\\0');pid=int(f.parent.name)
  if pid in (os.getpid(),os.getppid()):continue
  if len(parts)>2 and parts[1]==b'-c' and b'kit_b_deadline.json' in parts[2] and re.search(rb'^deadline = [0-9]',parts[2],re.M):
   os.kill(pid,signal.SIGTERM);removed.append(pid)
 except (OSError,ProcessLookupError):pass
p=Path('/root/onstart.sh')
if p.exists() and not Path('/root/onstart_before_pass1_realistic.sh').exists():Path('/root/onstart_before_pass1_realistic.sh').write_bytes(p.read_bytes())
p.write_text('#!/bin/bash\\n'+TIMER+'\\n');p.chmod(0o700)
subprocess.run(TIMER,shell=True,env=env,check=True)
print(json.dumps(dict(state='REARMED',removed_old_guards=removed,deadline=DEADLINE,onstart_file_updated=True)))
'''.replace('TIMER',repr(timer)).replace('DEADLINE',repr(lease['expires_at']))
direct=row.get('ports',{}).get('22/tcp')
host=row['public_ipaddr'] if direct else row['ssh_host'];port=int(direct[0]['HostPort']) if direct else int(row['ssh_port'])
assert re.fullmatch(r'[A-Za-z0-9.-]+',host)
from pathlib import Path
cmd=['ssh','-F','/dev/null','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(cloud.RECORD/'known_hosts'),'-p',str(port),'root@'+host,'python3 -c '+shlex.quote(remote)]
print(api.request('PUT',f"/api/v0/instances/{row['id']}/",{'state':'running'}),flush=True)
for attempt in range(24):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 if r.returncode==0:
  result=json.loads(r.stdout);api.write(cloud.RECORD/'timer_recovery.json',result);print(result,flush=True);break
 time.sleep(5)
else:raise RuntimeError('Could not reach retained instance to repair stale onstart')
