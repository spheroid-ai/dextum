"""Bounded Vast continuation for review section 9; local stdlib only."""
from pathlib import Path
import json,shlex,subprocess,time,re,hashlib
from dextum_cursor_agent.pass1 import cloud as previous
api=previous.api
ROOT=previous.ROOT
RECORD=ROOT/'dextum_plan/progress/2026-10-10-pass1-realistic/cloud'
OUTPUT=ROOT/'dextum_cursor_agent/runs/pass1_realistic_01'
REMOTE='/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01'
RELEASE='/root/dextum/releases/pass1_realistic_01/dextum_cursor_agent'
api.RECORD=RECORD


def prepare():
    RECORD.mkdir(parents=True,exist_ok=True)
    assert not (RECORD/'lease.json').exists()
    old=api.read(previous.RECORD/'lease.json');assert old['state']=='STOPPED'
    lease=dict(instance_id=old['instance_id'],label=old['label'],state='PREPARED',hourly_usd=old['hourly_usd'],
        expires_at=time.time()+8*3600,maximum_hours=8,compute_bound_usd=8*old['hourly_usd'],
        authorization='User review section 9: Realistic Kit B priority batch, re-segmentation/z/hold, corrected seed replication subject to budget',
        original_instance_stopped=True)
    api.write(RECORD/'lease.json',lease);return lease


def resume():
    lease=api.read(RECORD/'lease.json');row=api.instance();assert row['actual_status']=='exited'
    assert row.get('dph_total',99)<=.6
    timer=previous.previous.old.passive_deadline(lease['expires_at'])
    api.write(RECORD/'resume_request.json',dict(time=time.time(),instance_id=row['id'],deadline=lease['expires_at']))
    api.write(RECORD/'onstart_response.json',api.request('PUT',f"/api/v0/instances/update_template/{row['id']}/",{'id':row['id'],'image':row['image_uuid'],'onstart':timer}))
    assert api.instance().get('onstart')==timer
    install_startup_timer(timer)
    response=api.request('PUT',f"/api/v0/instances/{row['id']}/",{'state':'running'})
    api.write(RECORD/'resume_response.json',response);lease.update(state='START_REQUESTED',started_at=time.time());api.write(RECORD/'lease.json',lease)
    return dict(instance_id=row['id'],state='START_REQUESTED',compute_bound_usd=lease['compute_bound_usd'])


def upload(source,destination):
    cmd,_,host,_=api.connection()
    result=subprocess.run(['rsync','-a','--timeout=45','-e',shlex.join(cmd[:-1]),str(source),'root@'+host+':'+destination],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])


def fetch():
    cmd,_,host,_=api.connection();OUTPUT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['rsync','-a','--partial','--timeout=45','-e',shlex.join(cmd[:-1]),'root@'+host+':'+REMOTE+'/',str(OUTPUT)+'/'],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    import shutil
    for f in OUTPUT.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.jsonl','.log','.txt','.EXIT_CODE'):
            target=RECORD.parent/'results'/f.relative_to(OUTPUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    return str(OUTPUT)


def install_startup_timer(timer):
    """Metadata updates do not replace retained /root/onstart.sh; copy that file.

    The explicit C.instance locator is required by Vast's storage rsync daemon.
    Only one file is transferred, with no directory permission changes.
    """
    row=api.instance();assert row['actual_status']=='exited'
    folder=RECORD/'timer_files';folder.mkdir(exist_ok=True)
    path=folder/('onstart_'+str(int(time.time()))+'.sh')
    path.write_text('#!/bin/bash\n'+timer+'\n');path.chmod(0o700)
    target='C.'+str(row['id'])
    response=api.request('PUT','/api/v0/commands/rsync/',dict(client_id='me',src_id='local',dst_id=target,src_path=str(path),dst_path='/root/onstart.sh'))
    assert response['success'];host=response['dst_addr'];port=int(response['dst_port'])
    assert re.fullmatch(r'[A-Za-z0-9.:-]+',host) and 1<=port<=65535
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-p',str(port),'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(RECORD/'copy_known_hosts')]
    time.sleep(10)
    result=subprocess.run(['rsync','-t','--timeout=30','-e',shlex.join(ssh),str(path),'vastai_kaalia@'+host+'::'+target+'//root/onstart.sh'],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(result.stderr[-500:])
    api.write(RECORD/'installed_timer.json',dict(time=time.time(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),instance_id=row['id']))
