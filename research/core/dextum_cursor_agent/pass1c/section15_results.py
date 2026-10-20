"""Compare saved final dynamic faithful outcomes, without re-evaluating policies."""
import argparse,hashlib,json,os
from pathlib import Path


def main():
    import numpy as np
    from pass0.review_gate import exact_mcnemar
    from pass1c.criteria import assess_clustered_monitor
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    base=root.parent;old=base/'pass1_matched_01/cases/c_touch_realistic/c_touch'
    reg=json.loads((root/'MATCHED_REGISTRATION.json').read_text())
    accepted=json.loads((root/'ACCEPTED_C_TOUCH.json').read_text())
    paths=dict(c_touch_realistic=old,c_notouch=root/'cases/c_notouch/c_notouch',
               c_touch_ideal=root/'cases/c_touch_ideal/c_touch',a_task_ideal=root/'cases/a_task_ideal/a_task')
    outcomes={};reports={};bank=None
    for label,path in paths.items():
        raw=path/'monitor/20316160.npz';row=json.loads(raw.with_suffix('.json').read_text())
        data=np.load(raw);outcomes[label]=data['faithful']
        assert row['n']==200 and row['steps']==20316160 and row['finite']
        assert row['faithful_success']==int(outcomes[label].sum())
        checks=[json.loads(f.read_text()) for f in (path/'monitor').glob('*.json')]
        paired_checks={}
        for check in checks:
            with np.load(path/'monitor'/f"{check['steps']}.npz") as saved:
                paired_checks[check['steps']]=(saved['internal'],saved['faithful'])
        guard_checks=[dict(steps=check['steps'],**assess_clustered_monitor(check,checks,paired_checks)) for check in checks]
        assert all(check['state']=='PASS' for check in guard_checks)
        config=json.loads((path/'config.json').read_text())
        report=json.loads((path/'report.json').read_text())
        assert config['configuration']['seed']==1
        qualification=Path(config['physics_qualification']['path'])
        assert hashlib.sha256(qualification.read_bytes()).hexdigest()==config['physics_qualification']['sha256']
        assert json.loads(qualification.read_text())['model_sha256']==reg['model_sha256']
        assert config['configuration']['num_timesteps']==20000000
        assert row['reset_seed']==2026 and row['dr_seed']==7
        if label=='c_touch_realistic':
            assert int(outcomes[label].sum())==110
            assert hashlib.sha256(raw.read_bytes()).hexdigest()==accepted['outcomes_sha256']
            assert hashlib.sha256((path/'policy_20316160.pkl').read_bytes()).hexdigest()==accepted['checkpoint_sha256']
        else:
            assert report['state']=='COMPLETE'
            current={k:data[k] for k in ('reset_keys','direct_reset_keys','dr_keys','sign')}
            if bank is None:bank=current
            else:
                for key in current:assert np.array_equal(current[key],bank[key]),key
        segment_walls=[report['wall_s']]
        if label=='c_touch_ideal':
            prior=json.loads((root/'before_c_touch_ideal_resume/report.json').read_text())
            segment_walls.insert(0,prior['wall_s'])
        reports[label]=dict(n=200,success_count=row['faithful_success'],success_percent=row['faithful_success']/2,
            faithful_drops=row.get('faithful_drops'),sensor_level=config['sensor_level'],
            hand_model_sha256=reg['model_sha256'],seed=1,steps=20316160,
            outcome_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),path=str(path),wall_s=sum(segment_walls),
            training_segment_wall_s=segment_walls,
            guard_checks=guard_checks)
    def paired(left,right):
        x,y=outcomes[left],outcomes[right];b=int(np.sum(x&~y));c=int(np.sum(~x&y))
        return dict(left=left,right=right,difference_points=(int(x.sum())-int(y.sum()))/2,
                    left_only=b,right_only=c,exact_mcnemar_p=exact_mcnemar(b,c))
    touch=paired('c_touch_realistic','c_notouch');sensor=paired('c_touch_ideal','c_touch_realistic')
    established=touch['difference_points']>=10 and touch['exact_mcnemar_p']<.05
    result=dict(state='MEASURED',authority='review_section15',arms=reports,touch_effect=touch,sensor_cost=sensor,
        contact_vs_joint_ideal=paired('c_touch_ideal','a_task_ideal'),
        hand_effect_points=91.5-reports['c_touch_ideal']['success_percent'],
        hand_effect_protocol='Bare Ideal hand 108.1M training versus corrected hand 20.3M warm adaptation; one seed',
        accepted_bank_provenance='Historical MonitorGuard: PRNG2026 split200 then second reset split; DR7; identical generation for new controls. Historical npz has outcomes only, no key arrays.',
        decision='MASKING_AUTHORIZED_ONE_SEED_EFFECT' if established else 'SEEDS_2_3_REQUIRED',
        attempts_used_c_touch_realistic=1,review_control_attempts=0)
    (root/'MATCHED_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Numerical comparisons on Vast only')
    main()
