"""Bounded sequential Realistic controls, fixed final checkpoints and both evaluators."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01')
OLD=ROOT.parent/'pass1_audit_01'
PY='/root/dextum/.venv/bin/python'
RELEASE=Path(__file__).resolve().parents[1]

def write(name,x):(ROOT/name).write_text(json.dumps(x,indent=2)+'\n')
def run(name,args,limit):
    write('batch_status.json',dict(stage=name,time=time.time()))
    with (ROOT/(name+'.log')).open('xb') as log:
        p=subprocess.Popen([PY,'-u',*args],cwd=RELEASE,env=os.environ,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        end=time.monotonic()+limit
        code=None
        # Supervise the child from its launch. A separately installed guard can
        # arrive after the compiler has already exhausted the container.
        next_record=0.
        while code is None:
            code=p.poll()
            if code is not None:break
            try:
                status=Path(f'/proc/{p.pid}/status').read_text()
                rss=int(next((s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:')),'0'))
            except FileNotFoundError:
                continue
            now=time.monotonic()
            if now>=next_record:
                with (ROOT/(name+'_resources.jsonl')).open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),pid=p.pid,rss_kib=rss))+'\n')
                next_record=now+20
            if rss>220*1024*1024 or now>=end:
                reason='memory_limit' if rss>220*1024*1024 else 'time_limit'
                write(name+'_resource_stop.json',dict(time=time.time(),pid=p.pid,rss_kib=rss,
                      reason=reason,classification='Execution bound; not a physics criterion'))
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                code=125 if reason=='memory_limit' else 124
                break
            time.sleep(1)
    (ROOT/(name+'.EXIT_CODE')).write_text(str(code)+'\n');return code

def main():
    stop=float(sys.argv[1]);start=time.time();assert stop-start>3600
    # Record exact implementation before the first optimizer update.
    sources={str(p.relative_to(RELEASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RELEASE/'pass1b').glob('*.py')}
    write('BATCH_REGISTRATION.json',dict(time=start,arms=['c_touch','c_notouch','a_task'],seed=1,steps_requested=100000000,num_envs=8192,stop_at=stop,sources=sources,selection='final checkpoint only',attempts_per_arm=1,
      qualification='Inherited unchanged pass1/faithful.py barrier qualification; no new D5 criteria',condition='Kit B Realistic feature-proxy v0; common geometry and randomizer',eval_starts=200))
    # The long rollout smoke compile is diagnostic, not an extra physics gate.
    # The training job performs its own finite-metric/parameter checks.
    boundary=json.loads((ROOT/'runtime_checks/result.json').read_text())
    assert boundary['state']=='PASS'
    write('preflight_nonblocking.json',dict(time=time.time(),reason='Existing wrapper qualification and measured packet/runtime boundary retained; no new D5 criterion',rollout_diagnostic='preflight06'))
    for arm in ['c_touch','c_notouch','a_task']:
        remain=stop-time.time()
        if remain<3600:write('batch_status.json',dict(stage='BUDGET_DEFERRED',arm=arm,remaining_s=remain));return
        write(arm+'_attempt.json',dict(attempt=1,started_at=time.time(),steps=100000000))
        code=run(arm+'_training',['-m','pass1b.train','--arm',arm,'--output',str(ROOT/'e3'/arm),'--steps','100000000','--seed','1','--envs','8192','--qualification',str(OLD/'TRAINING_QUALIFICATION_BARRIER.json'),'--qualification-registration',str(OLD/'CONTROL_REGISTRATION.json')],min(7200,remain-900))
        if code==0:
            run(arm+'_evaluation',['-m','pass1b.compare',str(ROOT/'e3'),'--arm',arm,'--output',str(ROOT/(arm+'_evaluation')),'--warmup',str(OLD/'TRAINING_QUALIFICATION_BARRIER.json'),'--wrapper-registration',str(OLD/'CONTROL_REGISTRATION.json')],min(1800,max(1,stop-time.time()-120)))
        else:
            write('batch_status.json',dict(stage='TRAINING_FAILED',arm=arm,exit_code=code,attempts_used=1));return
    write('batch_status.json',dict(stage='COMPLETE',time=time.time()))
if __name__=='__main__':main()
