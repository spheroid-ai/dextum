"""Section 10 controls, admitted only after compile-only checks pass."""
import os,sys,time,json,hashlib
from pathlib import Path
from pass1c.bounded import run
ROOT=Path('/root/dextum/dextum_cursor_agent/runs/pass1_realistic_02')
OLD=ROOT.parent/'pass1_audit_01'
RELEASE=Path(__file__).resolve().parents[1]
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    deadline=float(sys.argv[1])
    assert not os.environ.get('XLA_FLAGS','')
    for name in ('model_verified','cell_check','compile_training2','compile_evaluator'):
        assert json.loads((ROOT/name/'result.json').read_text())['state']=='PASS',name
    assert json.loads((ROOT/'ORIGINAL_HAND_CHECK.json').read_text())['state']=='PASS'
    admitted=json.loads((ROOT/'ADMITTED_MODEL.json').read_text())
    assert hashlib.sha256((ROOT/'model/kit_b_training.xml').read_bytes()).hexdigest()==admitted['xml_sha256']
    write('CONTROL_REGISTRATION.json',dict(time=time.time(),seed=1,steps_requested=100000000,
        num_envs=8192,arms=['c_touch','c_notouch','a_task'],attempt_numbers=[2,1,1],
        original_task_reward=True,final_checkpoint_only=True,eval_starts=200,
        model=admitted,compile_checks_consume_training_attempts=False,
        sources={str(p.relative_to(RELEASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RELEASE/'pass1c').glob('*.py')},
        sensor_error_model_sha256=hashlib.sha256((RELEASE/'pass1b/sensors.py').read_bytes()).hexdigest()))
    for arm,attempt in [('c_touch',2),('c_notouch',1),('a_task',1)]:
        remain=deadline-time.time()
        if remain<3600:
            write('STATUS.json',dict(stage='BUDGET_DEFERRED',arm=arm,time=time.time()));return
        write(arm+'_attempt.json',dict(attempt=attempt,started_at=time.time(),steps_requested=100000000))
        write('STATUS.json',dict(stage='TRAINING',arm=arm,attempt=attempt,time=time.time()))
        r=run(ROOT,arm+'_training',['-m','pass1c.train','--arm',arm,'--output',str(ROOT/'e3'/arm),
            '--steps','100000000','--seed','1','--envs','8192',
            '--qualification',str(OLD/'TRAINING_QUALIFICATION_BARRIER.json'),
            '--qualification-registration',str(OLD/'CONTROL_REGISTRATION.json')],min(7200,remain-900))
        if r['exit_code']:
            write('STATUS.json',dict(stage='TRAINING_FAILED',arm=arm,attempt=attempt,exit=r));return
        write('STATUS.json',dict(stage='EVALUATING',arm=arm,time=time.time()))
        r=run(ROOT,arm+'_evaluation',['-m','pass1c.compare',str(ROOT/'e3'),'--arm',arm,
            '--output',str(ROOT/(arm+'_evaluation')),'--warmup',str(OLD/'TRAINING_QUALIFICATION_BARRIER.json'),
            '--wrapper-registration',str(OLD/'CONTROL_REGISTRATION.json')],min(1800,max(1,deadline-time.time()-60)))
        if r['exit_code']:
            write('STATUS.json',dict(stage='EVALUATION_FAILED',arm=arm,exit=r));return
        checkpoints=json.loads((ROOT/'e3'/arm/'checkpoints.json').read_text())
        final=next(c for c in reversed(checkpoints) if c.get('kept',True))
        r=run(ROOT,arm+'_export',['-m','pass1c.export_actor',str(ROOT/'e3'/arm/final['path']),
            str(ROOT/'runtime'/arm/'actor.npz'),'--arm',arm],300)
        if r['exit_code']:
            write('STATUS.json',dict(stage='RUNTIME_AUDIT_FAILED',arm=arm,exit=r));return
    write('STATUS.json',dict(stage='CONTROLS_COMPLETE',time=time.time(),next='hold/regrasp masking'))

if __name__=='__main__':main()
