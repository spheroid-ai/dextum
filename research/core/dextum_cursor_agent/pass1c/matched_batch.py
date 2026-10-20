"""Serial, bounded execution of the five review-required matched controls."""
import argparse,gzip,hashlib,json,os,time
from pathlib import Path
from pass1c.bounded import run


def archive(source):
    target=source.with_suffix(source.suffix+'.gz')
    h=hashlib.sha256()
    with source.open('rb') as f,gzip.open(target,'wb',compresslevel=1) as out:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):
            h.update(chunk);out.write(chunk)
    restored=hashlib.sha256()
    with gzip.open(target,'rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):restored.update(chunk)
    assert restored.digest()==h.digest()
    report=dict(source=str(source),archive=str(target),original_bytes=source.stat().st_size,
                archive_bytes=target.stat().st_size,original_sha256=h.hexdigest(),decompressed_sha256_verified=True)
    source.unlink()
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();root=a.run
    registration=json.loads((root/'MATCHED_REGISTRATION.json').read_text())
    qualification=root/'throughput/full_state/throughput.json'
    rate=json.loads(qualification.read_text());resume=json.loads(qualification.with_name('resume_test.json').read_text())
    assert rate['state']=='PASS' and resume['state']=='PASS'
    assert rate['metadata']['model_sha256']==registration['model_sha256']
    base=Path('/root/dextum/dextum_cursor_agent/runs')
    warm=dict(c_touch=base/'pass0_section7_01/section7_execute_01/e3/c_touch/policy_108134400.pkl',
              c_notouch=base/'pass1_audit_01/controls_01/e3/c_notouch/policy_108134400.pkl',
              a_task=base/'pass1_audit_01/controls_01/e3/a_task/policy_108134400.pkl')
    common=['--steps','20000000','--qualification',str(root/'matched_final/result.json'),
            '--qualification-registration',str(base/'pass1_audit_01/CONTROL_REGISTRATION.json'),
            '--evaluator-agreement',str(base/'pass1_d5_powered_01/policy_pair/paired_evaluators.json'),
            '--matched-registration',str(root/'MATCHED_REGISTRATION.json'),
            '--throughput-qualification',str(qualification)]
    state_file=root/'BATCH_STATE.json'
    state=json.loads(state_file.read_text()) if state_file.exists() else dict(state='RUNNING',completed=[],archives=[],attempts_consumed=0)
    def save():state_file.write_text(json.dumps(state,indent=2)+'\n')
    qlatest=root/'throughput/full_state/latest.pkl'
    if qlatest.exists():state['archives'].append(archive(qlatest));save()
    cases=[('c_touch_realistic','c_touch','Realistic'),('c_notouch','c_notouch','Ideal'),
           ('c_touch_ideal','c_touch','Ideal'),('a_task_ideal','a_task','Ideal'),('a_task_realistic','a_task','Realistic')]
    for label,arm,level in cases:
        if label in state['completed']:continue
        if a.deadline-time.time()<max(rate['projected_rounded_budget_s']+1800,3600):
            state.update(state='LEASE_BOUNDARY',next_case=label);save();return
        state.update(state='RUNNING',current_case=label,started_case_at=time.time());save()
        parent=root/'cases'/label;output=parent/arm
        args=['-m','pass1c.train','--arm',arm,'--sensor-level',level,'--output',str(output),
              '--warm-start',str(warm[arm]),*common]
        if output.exists():
            report=output/'report.json'
            if report.exists() and json.loads(report.read_text())['state']=='COMPLETE':result=dict(exit_code=0)
            else:
                state.update(state='NEEDS_EXECUTION_REVIEW',error='Existing incomplete child preserved');save();return
        else:result=run(root,label+'_train',args,7200)
        if result['exit_code']:
            state.update(state='EXECUTION_STOP',failure=result);save();return
        evaluation=parent/'evaluation'
        if not evaluation.exists():
            result=run(root,label+'_eval',['-m','pass1c.compare',str(parent),'--arm',arm,
                '--sensor-level',level,'--output',str(evaluation),'--warmup',str(root/'matched_final/result.json'),
                '--wrapper-registration',str(base/'pass1_audit_01/CONTROL_REGISTRATION.json')],7200)
            if result['exit_code']:
                state.update(state='EVALUATION_STOP',failure=result);save();return
        latest=output/'full_state/latest.pkl'
        if latest.exists():state['archives'].append(archive(latest))
        state['completed'].append(label);save()
    state.update(state='COMPLETE');save()


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
