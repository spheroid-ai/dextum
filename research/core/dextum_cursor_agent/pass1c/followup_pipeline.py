"""Bounded §16 dense-skin and mechanical follow-ups, strictly after masks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


def main():
    from pass1c.bounded import run
    from pass1c.matched_batch import archive
    p=argparse.ArgumentParser()
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    reg=json.loads(a.registration.read_text())
    a.run.mkdir(parents=True,exist_ok=True)
    state=dict(state='RUNNING',completed=[],archives=[],deadline=a.deadline)
    def save():
        (a.run/'PIPELINE_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    boot=dict(x.split(b'=',1) for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x)
    instance=boot[b'CONTAINER_ID'].decode();key=boot[b'CONTAINER_API_KEY'].decode()
    assert instance==os.environ['CONTAINER_ID']
    try:
        masks = json.loads(Path(reg['masking_state']).read_text())
        if reg.get('parked_mask'):
            disposition = json.loads(Path(reg['parked_mask']).read_text())
            assert disposition['authority'] == 'user_park_proximal_continue_independent'
            assert masks['state'] == 'INDEPENDENT_MASK_COMPLETE'
            assert set(masks['parked']) == {'proximal'}
        else:
            assert masks['state'] == 'COMPLETE'
        assert json.loads(Path(reg['dense_tests']).read_text())['exit_code']==0
        fixture=json.loads(Path(reg['hand_fixture']).read_text())
        assert fixture['state']=='PASS'
        prepared=json.loads(Path(reg['hand_preparation']).read_text())
        assert prepared['thin_geometry_admissible']
        for case in reg['order']:
            job=reg['cases'][case]
            for mode,seconds in [('audit',1800),('qualify',1800),('train',7200)]+([('t1',900)] if case=='dense' else []):
                if mode == 'train':
                    # A child may use a shorter cap near the fixed lease end.
                    # Retain the registered one-third throughput margin at
                    # that effective cap; never extend the lease or budget.
                    seconds = min(seconds, max(0, int(a.deadline-time.time())-120))
                    q = json.loads((Path(job['qualification'])/'throughput.json').read_text())
                    if q['state'] != 'PASS' or q['projected_rounded_budget_s'] > seconds*2/3:
                        state.update(state='LEASE_BOUNDARY',next_case=case,next_mode=mode,
                                     effective_child_cap_s=seconds)
                        save();return
                if a.deadline-time.time()<seconds+120:
                    state.update(state='LEASE_BOUNDARY',next_case=case,next_mode=mode)
                    save();return
                label=case+'_'+mode
                out=Path(job[{'audit':'detector','qualify':'qualification','train':'training','t1':'t1'}[mode]])
                if mode=='audit':out=out.parent
                if mode=='qualify':out=out.parent
                state.update(current=label);save()
                result=run(a.run,label,['-m','pass1c.followup','--registration',str(a.registration),'--case',case,
                    '--mode',mode,'--output',str(out)],seconds,rss_gib=56)
                if result['exit_code']:
                    state.update(state='EXECUTION_STOP',failure=result);save();return
                state['completed'].append(label);save()
                if mode=='train':
                    state['archives'].append(archive(out/'full_state/latest.pkl'));save()
                if mode=='qualify':
                    q=json.loads((out/'full_state/throughput.json').read_text())
                    if q['state']!='PASS':
                        state.update(state='THROUGHPUT_STOP',qualification=q);save();return
        state.update(state='FOLLOWUPS_COMPLETE',remaining='Corrected-physics seed2–3 teachers/confirmation and other Pass1 tasks')
    except BaseException as exc:
        state.update(state='EXECUTION_STOP',error=f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_at']=time.time();save()
        for _ in range(10):
            try:
                request=Request('https://console.vast.ai/api/v0/instances/'+instance+'/',method='PUT',
                    data=json.dumps({'state':'stopped'}).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urlopen(request,timeout=20) as response:result=json.load(response)
                assert result.get('success') is not False
                (a.run/'INSTANCE_FINISH.json').write_text(json.dumps(dict(state='STOP_REQUEST_ACCEPTED',instance_id=instance,time=time.time()),indent=2)+'\n')
                break
            except Exception:time.sleep(20)


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    main()
