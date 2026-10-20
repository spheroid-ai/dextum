"""Bounded sensor-cost sweeps followed by registered E3 masking continuations."""
import argparse,json,os,time
from pathlib import Path
from pass1c.bounded import run
from pass1c.matched_batch import archive


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();root=a.run
    reg=json.loads((root/'MATCHED_REGISTRATION.json').read_text())
    result=json.loads(Path(reg['matched_results']).read_text())
    assert result['decision']=='MASKING_AUTHORIZED_ONE_SEED_EFFECT'
    assert reg['authority']=='review_section15_masking'
    base=root.parent;prior=base/'pass1_matched_01'
    warm=prior/'cases/c_touch_realistic/c_touch/policy_20316160.pkl'
    common=['--steps','20000000','--arm','c_touch','--sensor-level','Realistic',
        '--warm-start',str(warm),'--qualification',str(root/'qualification.json'),
        '--qualification-registration',str(base/'pass1_audit_01/CONTROL_REGISTRATION.json'),
        '--evaluator-agreement',str(base/'pass1_d5_powered_01/policy_pair/paired_evaluators.json'),
        '--matched-registration',str(root/'MATCHED_REGISTRATION.json'),
        '--throughput-qualification',str(prior/'throughput/full_state/throughput.json')]
    statefile=root/'BATCH_STATE.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(state='RUNNING',completed=[],archives=[])
    def save():statefile.write_text(json.dumps(state,indent=2)+'\n')
    jobs=[]
    for sweep in ('floor','ideal_floor','delay','noise','direction','pvdf','rate','location'):
        output=root/'sweeps'/sweep
        jobs.append(('sweep_'+sweep,['-m','pass1c.sensor_sweep','--sweep',sweep,
            '--registration',str(root/'MATCHED_REGISTRATION.json'),'--output',str(output),
            '--checkpoint',reg['ideal_checkpoint'] if sweep=='ideal_floor' else str(warm)],output))
    for group in ('none','shear','location','slip','pvdf','palm','proximal','normal_only'):
        output=root/'cases'/group/'c_touch'
        jobs.append(('mask_'+group,['-m','pass1c.train_masking','--output',str(output),
            '--intervention',str(root/'interventions'/(group+'.json')),*common],output))
        evaluation=root/'evaluations'/group
        jobs.append(('t1_'+group,['-m','pass1c.evaluate_masking','--case',str(output),
            '--output',str(evaluation)],evaluation))
    for label,args,output in jobs:
        if label in state['completed']:continue
        if a.deadline-time.time()<3600:
            state.update(state='LEASE_BOUNDARY',next_case=label);save();return
        if output.exists():
            state.update(state='NEEDS_EXECUTION_REVIEW',next_case=label);save();return
        state.update(state='RUNNING',current_case=label,started_case_at=time.time());save()
        child=run(root,label,args,7200)
        if child['exit_code']:
            state.update(state='EXECUTION_STOP',failure=child);save();return
        if label.startswith('mask_'):
            monitor=json.loads((output/'monitor/20316160.json').read_text())
            assert monitor['state']=='PASS'
            state['archives'].append(archive(output/'full_state/latest.pkl'))
        state['completed'].append(label);save()
    state.update(state='COMPLETE');save()


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
