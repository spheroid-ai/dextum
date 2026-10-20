"""Cloud replay of saved actuator commands to recover missing observer fields.

No controller is executed or changed. Reconstructed physics must reproduce
the original logged yaw errors and held success before any score is accepted.
Full-pose Ideal camera is a declared interface extension; blind pose remains
unavailable. Simulator truth is used only by the evaluator/replay backend.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import mujoco
import numpy as np
from pass0.demo import scene, quat_mul, HOME, ASSETS
from pass1.goal_observer import GoalObserver, PoseObservation


def replay(asset,trial):
    model=scene(asset);data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    cube=model.body('cube').id;qstart=model.jnt_qposadr[model.joint('cube_freejoint').id]
    rng=np.random.default_rng(trial['seed'])
    limits=model.jnt_range[model.jnt_type!=mujoco.mjtJoint.mjJNT_FREE]
    data.qpos[:qstart]=np.clip(data.qpos[:qstart]+.1*rng.normal(size=qstart),limits[:,0],limits[:,1])
    data.qpos[qstart:qstart+3]=np.array([.1,0,.05])+rng.uniform(-.01,.01,3)
    quaternion=rng.normal(size=4);data.qpos[qstart+3:qstart+7]=quaternion/np.linalg.norm(quaternion)
    data.qvel[:]=0;data.ctrl[:]=HOME;data.mocap_pos[:]=-100;mujoco.mj_forward(model,data)
    observer=GoalObserver(trial['turn_deg'],blackout=trial['blackout'])
    target=None;held=best=0;max_yaw_error=0.;frames=[]
    for row in trial['steps']:
        tick=row['tick']
        if tick==trial['first_contact_tick']:
            angle=math.radians(trial['turn_deg'])/2
            target=quat_mul([math.cos(angle),0,0,math.sin(angle)],data.xquat[cube])
            observer.issue_at_contact(PoseObservation(tick,tuple(data.xquat[cube]),float(data.qpos[qstart+2]),'camera'))
        data.ctrl[:]=row['command']
        for _ in range(5):mujoco.mj_step(model,data)
        dropped=bool(data.qpos[qstart+2]<-.05)
        yaw=180.;tilt_ok=None
        if target is not None:
            error=quat_mul(target,data.xquat[cube]*np.array([1,-1,-1,-1]))
            angle=2*math.atan2(error[3],error[0])
            yaw=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle))))
            tilt_ok=bool(error[0]**2+error[3]**2>=math.cos(math.radians(30)/2)**2)
            held=held+1 if yaw<10 and tilt_ok and not dropped else 0
            best=max(best,held)
        max_yaw_error=max(max_yaw_error,abs(yaw-row['evaluation_only_yaw_error_deg']))
        observed_pose=None if trial['blackout'] else PoseObservation(tick,tuple(data.xquat[cube]),float(data.qpos[qstart+2]),'camera')
        verdict=observer.update(tick,observed_pose)
        frames.append(dict(tick=tick,evaluation_only_yaw_error_deg=yaw,evaluation_only_tilt_ok=tilt_ok,
                           evaluation_only_hold_steps=held,observer=verdict,
                           observed_pose=None if observed_pose is None else vars(observed_pose)))
    valid=max_yaw_error<1e-7 and abs(best*.05-trial['max_hold_s'])<1e-9 and (best>=40)==trial['success'] and dropped==trial['dropped']
    return dict(asset=trial['asset'],seed=trial['seed'],blackout=trial['blackout'],
                replay_valid=valid,maximum_logged_yaw_error_difference_deg=max_yaw_error,
                original_success=trial['success'],replayed_success=best>=40,
                old_ever_observed=any(s['goal_observed'] for s in trial['steps']),
                aligned_observed=observer.ever_observed,frames=frames)


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assets={v['id']:v for v in ASSETS};rows=[]
    for source in sorted(a.records.glob('p0_*.json')):
        trial=json.loads(source.read_text());row=replay(assets[trial['asset']],trial)
        row['original_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        (a.output/source.name).write_text(json.dumps(row)+'\n')
        rows.append({k:v for k,v in row.items() if k!='frames'})
        print(source.name,row['replay_valid'],row['aligned_observed'],flush=True)
    valid=len(rows)==96 and all(r['replay_valid'] for r in rows)
    result=dict(state='VERIFIED_REPLAY' if valid else 'REPLAY_MISMATCH',n=len(rows),rows=rows,
                controller_changes=0,source_commands='Original recorded actuator commands, unchanged',
                interface='Relative quaternion yaw, tilt and hold from full-pose Ideal camera; an extension of the old face-angle packet',
                blind_pose='Unavailable; no evaluation-only pose supplied to runtime observer after blackout',
                interpretation='Recognition measurement on replayed original trajectories; not a closed-loop controller improvement')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if not valid:raise SystemExit(3)


if __name__=='__main__':main()
