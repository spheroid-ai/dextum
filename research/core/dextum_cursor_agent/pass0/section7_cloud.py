"""Bounded remote continuation for the user's Pass 0 review; local stdlib only."""
import argparse
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from . import cloud_ops as old

api = old.api
ROOT, PROJECT = old.ROOT, old.PROJECT
RECORD = ROOT/'dextum_plan/progress/2026-10-10-section7/cloud'
OUTPUT = PROJECT/'runs/pass0_section7_01'
REMOTE_OUTPUT = '/root/dextum/dextum_cursor_agent/runs/pass0_section7_01'
LABEL = 'dextum-v12-pass0_section7_01'
api.RECORD, api.LABEL = RECORD, LABEL


def prepare():
    RECORD.mkdir(parents=True, exist_ok=False)
    previous = api.read(old.RECORD/'lease.json')
    deadline = time.time()+4*3600
    api.write(RECORD/'lease.json', dict(instance_id=previous['instance_id'], label=LABEL,
        state='PREPARED', expires_at=deadline, hourly_usd=previous['offer']['dph_total'],
        projected_bound_usd=4*previous['offer']['dph_total']+1,
        authorization='User Pass 0 review: fixed-policy D5, diagnostic, then on PASS one original E3 seed and three-shape demo',
        historical_lease=str((old.RECORD/'lease.json').relative_to(ROOT))))
    return dict(state='PREPARED', max_hours=4, instance_id=previous['instance_id'])


def resume():
    lease=api.read(RECORD/'lease.json')
    assert not (RECORD/'resume_request.json').exists(), 'Reconcile previous request'
    row=api.instance()
    assert row and row['actual_status']=='exited'
    assert row.get('dph_total',99)<=.6
    timer=old.passive_deadline(lease['expires_at'])
    api.write(RECORD/'resume_request.json',dict(time=time.time(), instance_id=row['id'],
        previous_onstart=row.get('onstart'), requested_onstart=timer, deadline=lease['expires_at']))
    result=api.request('PUT',f"/api/v0/instances/{row['id']}/",{'onstart':timer})
    api.write(RECORD/'onstart_update_response.json',result)
    updated=api.instance()
    if updated.get('onstart') != timer:
        return dict(state='ONSTART_NOT_UPDATED', instance_status=updated['actual_status'])
    result=api.request('PUT',f"/api/v0/instances/{row['id']}/",{'state':'running'})
    api.write(RECORD/'start_response.json',result)
    lease.update(state='START_REQUESTED',requested_at=time.time())
    api.write(RECORD/'lease.json',lease)
    return dict(state='START_REQUESTED',instance_id=row['id'])


def rent():
    assert not (RECORD/'rental_attempt.json').exists(), 'Reconcile rental before retry'
    rows=api.inventory()
    assert not any(r.get('label')==LABEL for r in rows)
    prior=api.read(RECORD/'lease.json')
    old_row=next(r for r in rows if r['id']==prior['instance_id'])
    assert old_row['actual_status']=='exited'
    # Explicitly cancel any queued resume before renting another GPU.
    api.request('PUT',f"/api/v0/instances/{old_row['id']}/",{'state':'stopped'})
    api.write(RECORD/'existing_instance_lease.json',prior)
    query=dict(verified={'eq':True},rentable={'eq':True},rented={'eq':False},num_gpus={'eq':1},
        gpu_name={'eq':'RTX 4090'},cuda_max_good={'gte':12.8},cpu_cores_effective={'gte':16},
        cpu_ram={'gte':32000},reliability={'gte':.98},disk_space={'gte':40},inet_down={'gte':300},
        direct_port_count={'gte':1},type='on-demand',allocated_storage=40,order=[['dph_total','asc']],limit=50)
    offers=[r for r in api.request('POST','/api/v0/bundles/',query).get('offers',[])
            if r.get('dph_total',99)<=.6 and r.get('geolocation','').rsplit(',',1)[-1].strip() in api.COUNTRIES
            and all(r.get(k) is not None and r[k]<=.1 for k in ('inet_up_cost','inet_down_cost'))]
    assert offers,'No qualifying remote offer'
    offer=min(offers,key=lambda r:r['dph_total'])
    deadline=time.time()+4*3600
    body=dict(client_id='me',image='pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime',disk=40,
              label=LABEL,runtype='ssh_direct',cancel_unavail=True,env={},onstart=old.passive_deadline(deadline))
    api.write(RECORD/'rental_attempt.json',dict(time=time.time(),offer={k:offer.get(k) for k in
        ('id','machine_id','gpu_name','cpu_name','cpu_cores_effective','cpu_ram','dph_total','geolocation')},
        request=body,max_hours=4,projected_bound_usd=4*offer['dph_total']+1,
        authorization=prior['authorization']))
    response=api.request('PUT',f"/api/v0/asks/{offer['id']}/",body)
    api.write(RECORD/'rental_response.json',response)
    assert response.get('success') and response.get('new_contract')
    lease=dict(instance_id=int(response['new_contract']),label=LABEL,state='RENTED',created_at=time.time(),
        expires_at=deadline,hourly_usd=offer['dph_total'],projected_bound_usd=4*offer['dph_total']+1)
    api.write(RECORD/'lease.json',lease)
    api.request('POST',f"/api/v0/instances/{lease['instance_id']}/ssh/",
                {'ssh_key':(Path.home()/'.ssh/id_ed25519.pub').read_text().strip()})
    return lease


def bootstrap():
    record=RECORD/'bootstrap';record.mkdir(exist_ok=False)
    lease=api.read(RECORD/'lease.json')
    cmd,_,_,_=api.connection()
    for source,destination in [(RECORD/'bootstrap_public_dependencies.tar','/root/dextum/review_base.tar'),
                               (Path(__file__).with_name('review_bootstrap.sh'),'/root/dextum/review_bootstrap.sh')]:
        with source.open('rb') as stream:
            r=subprocess.run(cmd+['mkdir -p /root/dextum && cat > '+shlex.quote(destination)],stdin=stream,capture_output=True,timeout=180)
        assert r.returncode==0,r.stderr.decode()[-1000:]
        assert api.remote('sha256sum '+shlex.quote(destination)).split()[0]==api.digest(source)
    api.write(record/'registration.json',dict(time=time.time(),base_input_sha256=api.digest(RECORD/'bootstrap_public_dependencies.tar'),
        script_sha256=api.digest(Path(__file__).with_name('review_bootstrap.sh')),scope='Pinned remote environment only; no model training'))
    script="""import os,subprocess
from pathlib import Path
p=Path(OUTPUT);p.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CONTAINER_ID=INSTANCE,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='disable',XLA_PYTHON_CLIENT_PREALLOCATE='false')
log=(p/'setup.log').open('xb')
proc=subprocess.Popen(['bash','-c',COMMAND],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(p/'pid').write_text(str(proc.pid));print(proc.pid)
""".replace('OUTPUT',repr(REMOTE_OUTPUT+'/bootstrap')).replace('INSTANCE',repr(str(lease['instance_id']))).replace('COMMAND',repr(
    'timeout 2400s bash /root/dextum/review_bootstrap.sh '+shlex.quote(REMOTE_OUTPUT+'/bootstrap')+'; code=$?; printf \'%s\\n\' "$code" > '+shlex.quote(REMOTE_OUTPUT+'/bootstrap/EXIT_CODE')))
    pid=api.remote('python3 -c '+shlex.quote(script))
    api.write(record/'launch.json',dict(time=time.time(),pid=pid.strip()))
    return dict(state='BOOTSTRAP_STARTED',pid=pid.strip())


def status():
    row=api.instance()
    result={k:row.get(k) for k in ('id','label','actual_status','status_msg','dph_total')}
    if row['actual_status']=='running':
        script="""from pathlib import Path
import json
p=Path(OUTPUT)
names={'EXIT_CODE','paired_evaluators.json','results.json','report.json','result.json'}
files={str(f.relative_to(p)):f.read_text()[-4500:] for f in p.rglob('*') if f.is_file() and (f.name in names or f.suffix=='.log')}
print(json.dumps(dict(deadline=json.loads(Path('/root/kit_b_deadline.json').read_text()),files=files)))
""".replace('OUTPUT',repr(REMOTE_OUTPUT))
        result['remote']=json.loads(api.remote('python3 -c '+shlex.quote(script)))
    api.write(RECORD/'last_status.json',dict(time=time.time(),**result))
    return result


def submit(job):
    assert job == 'section7_pipeline_01'
    lease=api.read(RECORD/'lease.json')
    remaining=int(lease['expires_at']-time.time()-180)
    assert remaining>600
    guard=json.loads(api.remote('cat /root/kit_b_deadline.json'))
    assert guard['state']=='ARMED' and guard['deadline']==lease['expires_at']
    if job!='section7_pipeline_01':
        preflight=json.loads(api.remote('cat '+shlex.quote(REMOTE_OUTPUT+'/bootstrap/preflight.json')))
        assert preflight['state']=='PASS','GPU preflight must pass first'
    record=RECORD/job
    record.mkdir(exist_ok=False)
    paths=[]
    for folder in ('dextum','envs','experiments','tools','vendor','pass0'):
        paths += [p for p in (PROJECT/folder).rglob('*') if p.is_file()
                  and 'records' not in p.parts and '__pycache__' not in p.parts
                  and p.suffix in ('.py','.json','.txt','.sh','.md')]
    paths += [ROOT/'dextum_plan/REVIEW_2026-10-10.md',
              ROOT/'dextum_plan/progress/2026-10-10-section7/REGISTER.md',
              ROOT/'dextum_plan/progress/2026-10-10-section7/D5_QUALIFICATION.json',
              old.RECORD/'d5_retry_03/registration.json',
              old.OUTPUT/'d5_retry_03/d5/qualification.json']
    historical=PROJECT/'runs/e3p1-s1/c_touch'
    paths += [historical/'policy_108134400.pkl', historical/'checkpoints.json']
    assert sum(p.stat().st_size for p in paths)<32*1024*1024
    reg=dict(time=time.time(),job=job,run='pass0_section7_01', maximum_s=remaining,
        review='dextum_plan/REVIEW_2026-10-10.md',
        input_sha256={str(p.relative_to(ROOT)):api.digest(p) for p in paths},
        bank=dict(n=200,reset_seed=2026,dr_seed=7,reset_split='second'),
        D5=dict(warmup='accepted retry-03 14/16 within 5mm',max_success_gap=.05,min_mcnemar_p_exclusive=.05,max_median_error_exclusive=1e-5),
        wrapper='unchanged retry-03; shared nominal arrays and batched randomized fields',
        xla_flags='', diagnostic='nonblocking; per-DoF/contact-count distributions; ordered settings diff',
        phases=['GPU preflight','registered section7 D5 PASS','one E3 attempt','both evaluators on final E3 policy','96-trial native demo','nonblocking faithful compiler control'],
        training=dict(attempts=1,arm='c_touch',seed=1,steps=100000000,num_envs=8192,task_reward='original'),
        demo=dict(shapes=['p0_cube55','p0_rounded48','p0_cylinder38x70'],starts=16,
                  vision=['blackout_after_first_contact','on'],seed_start=2026101000,actor='historical final seed1; preexisting stand-in'),
        compiler_control=dict(checkpoint='historical final seed1',n=200,flags=['','--xla_disable_hlo_passes=constant_folding'],nonblocking=True),
        mapping_audit=dict(design='Conventional PPO and separate physical reference; no anonymous processor introduced',
                           implementation='No Metriplector implementation claim'))
    api.write(record/'registration.json',reg)
    paths.append(record/'registration.json')
    archive=record/'inputs.tar'
    with tarfile.open(archive,'w') as tar:
        for p in paths:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    cmd,_,_,_=api.connection()
    destination='/root/dextum/'+job+'.tar'
    with archive.open('rb') as stream:
        result=subprocess.run(cmd+['cat > '+shlex.quote(destination)],stdin=stream,capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr.decode()[-1500:])
    assert api.remote('sha256sum '+shlex.quote(destination)).split()[0]==api.digest(archive)
    release='/root/dextum/releases/'+job
    script="""from pathlib import Path
import tarfile,hashlib,json
p=Path(RELEASE);p.mkdir(parents=True,exist_ok=False)
with tarfile.open(ARCHIVE) as tar:
 for m in tar.getmembers():
  assert m.isfile() and (p/m.name).resolve().is_relative_to(p)
 tar.extractall(p)
(p/'upstream').symlink_to('/root/dextum/upstream',target_is_directory=True)
r=json.loads((p/REGISTRATION).read_text())
for name,h in r['input_sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
print('Input hashes verified')
""".replace('RELEASE',repr(release)).replace('ARCHIVE',repr(destination)).replace('REGISTRATION',repr(str((record/'registration.json').relative_to(ROOT))))
    api.remote('python3 -c '+shlex.quote(script))
    output=REMOTE_OUTPUT+'/'+job
    inner='timeout --signal=TERM --kill-after=30s '+str(remaining)+'s bash pass0/'+job+'.sh '+shlex.quote(output)
    inner+='; code=$?; printf \'%s\\n\' "$code" > '+shlex.quote(output+'/EXIT_CODE')
    launch="""import os,subprocess
from pathlib import Path
p=Path(OUTPUT);p.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CONTAINER_ID=INSTANCE,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='disable')
env.pop('XLA_FLAGS',None)
log=(p/'run.log').open('xb')
proc=subprocess.Popen(['bash','-c',INNER],cwd=RELEASE+'/dextum_cursor_agent',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(p/'pid').write_text(str(proc.pid));print(proc.pid)
""".replace('OUTPUT',repr(output)).replace('INSTANCE',repr(str(lease['instance_id']))).replace('INNER',repr(inner)).replace('RELEASE',repr(release))
    api.write(record/'launch_requested.json',dict(time=time.time(),output=output))
    pid=api.remote('python3 -c '+shlex.quote(launch))
    api.write(record/'launch.json',dict(time=time.time(),pid=pid.strip(),remaining_s=remaining))
    return dict(job=job,pid=pid.strip(),remaining_s=remaining)


def fetch():
    cmd,_,host,_=api.connection()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['rsync','-a','--partial','--timeout=45','-e',shlex.join(cmd[:-1]),
        'root@'+host+':'+REMOTE_OUTPUT+'/',str(OUTPUT)+'/'],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-1500:])
    return dict(state='FETCHED',output=str(OUTPUT))


def stop():
    row=api.instance()
    result=api.request('PUT',f"/api/v0/instances/{row['id']}/",{'state':'stopped'})
    lease=api.read(RECORD/'lease.json');lease.update(state='STOP_REQUESTED',stop_requested_at=time.time())
    api.write(RECORD/'lease.json',lease)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','resume','rent','bootstrap','status','submit','fetch','stop'))
    p.add_argument('--job');a=p.parse_args()
    print(json.dumps(submit(a.job) if a.action=='submit' else globals()[a.action](),indent=2))
