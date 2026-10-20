"""Three controls under §15; accepted C-touch remains frozen in its original run."""
import argparse,json,os,time
from pathlib import Path
from pass1c.bounded import run
from pass1c.matched_batch import archive


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();root=a.run
    registration=json.loads((root/'MATCHED_REGISTRATION.json').read_text())
    base=Path('/root/dextum/dextum_cursor_agent/runs')
    old=base/'pass1_matched_01'
    qualification=old/'throughput/full_state/throughput.json'
    rate=json.loads(qualification.read_text())
    assert rate['state']=='PASS' and json.loads(qualification.with_name('resume_test.json').read_text())['state']=='PASS'
    assert rate['metadata']['model_sha256']==registration['model_sha256']
    warm=dict(c_touch=base/'pass0_section7_01/section7_execute_01/e3/c_touch/policy_108134400.pkl',
              c_notouch=base/'pass1_audit_01/controls_01/e3/c_notouch/policy_108134400.pkl',
              a_task=base/'pass1_audit_01/controls_01/e3/a_task/policy_108134400.pkl')
    common=['--steps','20000000','--qualification',str(root/'qualification.json'),
        '--qualification-registration',str(base/'pass1_audit_01/CONTROL_REGISTRATION.json'),
        '--evaluator-agreement',str(base/'pass1_d5_powered_01/policy_pair/paired_evaluators.json'),
        '--matched-registration',str(root/'MATCHED_REGISTRATION.json'),
        '--throughput-qualification',str(qualification)]
    path=root/'BATCH_STATE.json'
    state=json.loads(path.read_text()) if path.exists() else dict(state='RUNNING',completed=[],archives=[],
        attempts_consumed=1,review_control_attempts_consumed=0,accepted_c_touch=str(old/'cases/c_touch_realistic/c_touch'))
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    for label,arm in [('c_notouch','c_notouch'),('c_touch_ideal','c_touch'),('a_task_ideal','a_task')]:
        if label in state['completed']:continue
        if a.deadline-time.time()<max(rate['projected_rounded_budget_s']+1800,3600):
            state.update(state='LEASE_BOUNDARY',next_case=label);save();return
        state.update(state='RUNNING',current_case=label,started_case_at=time.time());save()
        output=root/'cases'/label/arm
        if output.exists():
            state.update(state='NEEDS_EXECUTION_REVIEW',error='Incomplete child preserved; resume separately');save();return
        result=run(root,label+'_train',['-m','pass1c.train','--arm',arm,'--sensor-level','Ideal',
            '--output',str(output),'--warm-start',str(warm[arm]),*common],7200)
        if result['exit_code']:
            state.update(state='EXECUTION_STOP',failure=result);save();return
        monitor=json.loads((output/'monitor/20316160.json').read_text())
        assert monitor['state']=='PASS' and monitor['criterion_version']=='review_section15'
        latest=output/'full_state/latest.pkl'
        if latest.exists():state['archives'].append(archive(latest))
        state['completed'].append(label);save()
    state.update(state='COMPLETE');save()


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
