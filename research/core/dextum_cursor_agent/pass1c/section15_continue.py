"""Finish the running independent control, then resume the accepted stopped state."""
import argparse,json,os,shutil,time
from pathlib import Path
from pass1c.bounded import run
from pass1c.matched_batch import archive


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();root=a.run;base=root.parent
    statefile=root/'BATCH_STATE.json'
    def state():return json.loads(statefile.read_text())
    def save(s):statefile.write_text(json.dumps(s,indent=2)+'\n')
    stop=root/'a_task_ideal_train_exit.json'
    while not stop.exists():
        if time.time()>a.deadline-3600:raise RuntimeError('Lease margin exhausted before continuation')
        time.sleep(20)
    child=json.loads(stop.read_text())
    if child['exit_code']:
        s=state();s.update(state='EXECUTION_STOP',failure=child,current_case='a_task_ideal');save(s);return
    case=root/'cases/a_task_ideal/a_task'
    assert json.loads((case/'report.json').read_text())['state']=='COMPLETE'
    s=state();s['archives'].append(archive(case/'full_state/latest.pkl'))
    s['completed'].append('a_task_ideal');s.pop('failure',None);save(s)
    migration=json.loads((root/'GUARD_CHECKPOINT_MIGRATION.json').read_text())
    assert migration['state']=='PASS' and migration['payload_before']==migration['payload_after']
    assert a.deadline-time.time()>3600
    case=root/'cases/c_touch_ideal/c_touch';snapshot=root/'before_c_touch_ideal_resume'
    snapshot.mkdir(exist_ok=False)
    for file in case.iterdir():
        if file.is_file():shutil.copy2(file,snapshot/file.name)
    shutil.copytree(case/'monitor',snapshot/'monitor')
    args=['-m','pass1c.train','--arm','c_touch','--sensor-level','Ideal','--output',str(case),
          '--warm-start',str(base/'pass0_section7_01/section7_execute_01/e3/c_touch/policy_108134400.pkl'),
          '--resume',migration['destination'],'--steps','20000000',
          '--qualification',str(root/'qualification.json'),
          '--qualification-registration',str(base/'pass1_audit_01/CONTROL_REGISTRATION.json'),
          '--evaluator-agreement',str(base/'pass1_d5_powered_01/policy_pair/paired_evaluators.json'),
          '--matched-registration',str(root/'MATCHED_REGISTRATION.json'),
          '--throughput-qualification',str(base/'pass1_matched_01/throughput/full_state/throughput.json')]
    s.update(state='RESUMING_ACCEPTED_CONTROL',current_case='c_touch_ideal',started_case_at=time.time(),
             delegated_guard='review_section15_clustered',resumed_optimizer_steps=7208960)
    save(s)
    child=run(root,'c_touch_ideal_continue',args,7200)
    if child['exit_code']:
        s.update(state='EXECUTION_STOP',failure=child);save(s);return
    assert json.loads((case/'report.json').read_text())['state']=='COMPLETE'
    full=case/'full_state';finished=full/'completed_20316160.pkl'
    assert not finished.exists();(full/'latest.pkl').rename(finished)
    s['archives'].append(archive(finished));s['completed'].append('c_touch_ideal')
    s.update(state='COMPLETE',historical_stop_resolved=True);save(s)
    result=run(root,'matched_results',['-m','pass1c.section15_results','--run',str(root)],120)
    if result['exit_code']:
        s.update(state='RESULT_AUDIT_STOP',failure=result);save(s)

if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
