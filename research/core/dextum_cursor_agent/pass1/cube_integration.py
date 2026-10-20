"""Cloud cube integration ladder, fixed actor and paired 16-start conditions.

Change handoff, randomization, start bank and geometry in named adjacent rows.
The direct-MJX/native reference uses the original training geometry. The
held-out cube remains the original Pass 0 55-mm mesh. No new policy is trained.
"""
import argparse
from collections import deque
import copy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import vendor.aero_official.vision_env
from brax.io import model as policy_io
from brax.training.agents.ppo import networks
from envs.blind_rotation import BlindRotation, EPISODE
from experiments.evaluate_e3 import step_one, summarise
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass0.demo import scene, reading, camera_angle, quat_mul, trial as legacy_trial, ASSETS
from pass0.numpy_actor import Actor, ContactMotor, HOME, actor_frame
from pass1.faithful import wrap_for_training, MODEL_KEY
from pass1.qualification import require_training_pass
from pass1.runtime import Packet, TaskLoop, Edge, reorientation_dag, scripted_action
from pass1.goal_observer import PoseObservation
from pass0.train_e3 import make_network
from dextum.touch_mjx import TouchModel
from tools.physics_ref import RANDOMISED


def apply_draw(native, source_model, randomized, index):
    # The held-out mesh removes decorative object geoms. Map friction by name
    # or by body/local ordinal, never by an assumed global geom index.
    geom_map=[]
    for g in range(native.ngeom):
        name=mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_GEOM,g)
        if name:
            old=mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_GEOM,name)
        else:
            body=native.geom_bodyid[g]
            body_name=mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_BODY,body)
            old_body=mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_BODY,body_name)
            ordinal=list(np.flatnonzero(native.geom_bodyid==body)).index(g)
            old=int(np.flatnonzero(source_model.geom_bodyid==old_body)[ordinal])
        assert old>=0
        geom_map.append(old)
    for field in RANDOMISED:
        values=np.asarray(getattr(randomized,field)[index])
        if field=='geom_friction':values=values[geom_map]
        assert getattr(native,field).shape==values.shape,field
        getattr(native,field)[:]=values
    # Same convention as registered native D5: do not recompute derived model
    # constants when MJX randomization also leaves those constants untouched.


def native_trial(native, index, use_training_start, initial, blackout, actor):
    data=mujoco.MjData(native)
    mujoco.mj_resetDataKeyframe(native,data,native.key('home').id)
    cube=native.body('cube').id;qstart=native.jnt_qposadr[native.joint('cube_freejoint').id]
    if use_training_start:
        for field in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
            getattr(data,field)[:]=np.asarray(getattr(initial.data,field)[index])
    else:
        rng=np.random.default_rng(2026101000+index)
        limits=native.jnt_range[native.jnt_type!=mujoco.mjtJoint.mjJNT_FREE]
        data.qpos[:qstart]=np.clip(data.qpos[:qstart]+.1*rng.normal(size=qstart),limits[:,0],limits[:,1])
        data.qpos[qstart:qstart+3]=np.array([.1,0,.05])+rng.uniform(-.01,.01,3)
        q=rng.normal(size=4);data.qpos[qstart+3:qstart+7]=q/np.linalg.norm(q)
        data.qvel[:]=0;data.ctrl[:]=HOME;data.mocap_pos[:]=-100
    mujoco.mj_forward(native,data)
    sign=1 if index%2==0 else -1
    motor=ContactMotor(native.actuator_ctrlrange);touch=TouchModel(native)
    history=deque(maxlen=3);age=np.zeros(11,int)
    normal=shear=np.zeros(11);slip=np.zeros(11,bool)
    target=None;goal_face=0.;held=best=0;success=False;loop=None;choice={'quantum':'pivot'}
    sensor_names=('len_if','len_mf','len_rf','len_pf','len_th1','len_th2','len_th_abd')
    trace=[]
    for tick in range(EPISODE):
        if tick<20:
            command=HOME.copy()
        else:
            action=actor(np.concatenate(list(history))) if choice['quantum']=='pivot' else scripted_action('hold',tick-20,sign)
            command=motor.step(action,normal,shear,slip)
        data.ctrl[:]=command
        for _ in range(5):mujoco.mj_step(native,data)
        normal,shear,slip,force_xyz,centroid=reading(native,data,touch)
        age=np.where(normal>.3,age+1,0)
        tick1=tick+1
        if tick1==20:
            angle=math.radians(sign*30)/2
            target=quat_mul([math.cos(angle),0,0,math.sin(angle)],data.qpos[qstart+3:qstart+7])
            goal_face=camera_angle(data,cube)+sign*30
            loop=TaskLoop(reorientation_dag(f'reorient by {sign*30} degrees'),blackout=blackout,
                          family='cube',contact_seen=True,
                          active=Edge('trained_pivot','entry','N6','pivot',280),active_since=20)
            # Absolute goal is the registered task-command interface. Current
            # pose after blackout is not supplied to the runtime observer.
            loop.observer.target=tuple(float(x) for x in target)
        motors=np.array([data.sensor(n).data[0] for n in sensor_names])
        frame=actor_frame(motors,motor,normal,shear,slip,force_xyz,centroid,age,goal_face,sign,tick1-20)
        if tick1<20:frame[15:18]=0.
        if not history:history.extend([frame.copy() for _ in range(3)])
        else:history.appendleft(frame)
        dropped=bool(data.qpos[qstart+2]<-.05)
        error_deg=180.;tilt_ok=False
        if target is not None:
            error=quat_mul(target,data.qpos[qstart+3:qstart+7]*np.array([1,-1,-1,-1]))
            angle=2*math.atan2(error[3],error[0])
            error_deg=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle))))
            tilt_ok=bool(error[0]**2+error[3]**2>=math.cos(math.radians(30)/2)**2)
            within=error_deg<10 and tilt_ok and not dropped and tick1>20
            held=held+1 if within else 0;best=max(best,held);success |= held>=40
            pose=None if blackout else PoseObservation(tick1,tuple(data.qpos[qstart+3:qstart+7]),float(data.qpos[qstart+2]),'camera')
            packet=Packet(tick1,tuple(normal),tuple(shear),tuple(slip),tuple(motors),
                          camera_shape='cube' if not blackout else None,camera_valid=not blackout,pose=pose)
            choice=loop.update(packet)
        trace.append(dict(tick=tick1,quantum=choice['quantum'],evaluation_only_yaw_error_deg=error_deg,
                           evaluation_only_tilt_ok=tilt_ok,goal_observed=bool(loop and loop.goal_observed),
                           command=[float(x) for x in command]))
        if dropped:break
    return dict(index=index,blackout=blackout,success=bool(success),dropped=dropped,
                max_hold_s=best*.05,goal_observed=bool(loop and loop.goal_observed),steps=trace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True);a=p.parse_args()
    require_training_pass(json.loads(a.qualification.read_text()))
    assert json.loads(a.actor.with_suffix('.audit.json').read_text())['state']=='PASS'
    a.output.mkdir(parents=True,exist_ok=False);jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation('c_touch');n=16
    rng=jax.random.split(jax.random.PRNGKey(2026),n);dr=jax.random.split(jax.random.PRNGKey(7),n)
    randomized,axes=domain_randomize(env.mjx_model,dr)
    wrapped=wrap_for_training(env,episode_length=EPISODE,randomization_fn=functools.partial(domain_randomize,rng=dr))
    initial=jax.jit(wrapped.reset)(rng)
    signs=jnp.where(jnp.arange(n)%2==0,1.,-1.)
    raw=initial.replace(info={**{k:v for k,v in initial.info.items() if k!=MODEL_KEY},'turn_sign':signs})
    params=policy_io.load_params(a.checkpoint);factory=networks.make_inference_fn(make_network('c_touch'))
    step=jax.vmap(functools.partial(step_one,env),in_axes=(axes,0,0))

    @jax.jit
    def reference(model,state,policy_params):
        infer=factory(policy_params,deterministic=True)
        def body(s,_):
            nxt=step(model,s,infer(s.obs,jax.random.PRNGKey(0))[0])
            return nxt,dict(success=nxt.metrics['e3/success'],done=nxt.done,
                            within=nxt.metrics['e3/within'],error=nxt.metrics['e3/abs_error_deg'])
        return jax.lax.scan(body,state,None,length=EPISODE)[1]

    values=summarise(jax.tree.map(np.asarray,reference(randomized,raw,params)),np.asarray(signs))
    np.savez(a.output/'mjx_reference.npz',**values)
    rows=[dict(condition='F_mjx_training_geometry',blackout=True,n=n,success=int(values['success'].sum()),drops=int(values['dropped'].sum()))]
    actor=Actor(a.actor);heldout=ASSETS[0]
    conditions=[('A_legacy_handoff',False,False,False),('B_aligned_handoff',False,False,False),
                ('C_randomized_physics',True,False,False),('D_training_starts',True,True,False),
                ('E_training_geometry',True,True,True)]
    for condition,randomized_physics,training_starts,training_geometry in conditions:
        for blackout in (True,False):
            trials=[]
            for i in range(n):
                if condition=='A_legacy_handoff':
                    record=legacy_trial(heldout,2026101000+i,1 if i%2==0 else -1,blackout,actor)
                else:
                    native=copy.deepcopy(env.mj_model) if training_geometry else scene(heldout)
                    if randomized_physics:apply_draw(native,env.mj_model,randomized,i)
                    record=native_trial(native,i,training_starts,initial,blackout,actor)
                trials.append(record)
                (a.output/f'{condition}_{blackout}_{i:02d}.json').write_text(json.dumps(record)+'\n')
            row=dict(condition=condition,blackout=blackout,n=n,success=sum(t['success'] for t in trials),
                     drops=sum(t['dropped'] for t in trials),goal_observed=sum(t['goal_observed'] for t in trials))
            rows.append(row);print(json.dumps(row),flush=True)
            (a.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    result=dict(state='COMPLETE',rows=rows,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                n=16,reset_seed=2026,dr_seed=7,goal_signs='alternating +/-30 degrees for all conditions',
                contrasts={'A_B':'handoff package: actor history/clock, warm-up goal timing, uninterrupted pivot entry and aligned observer',
                    'B_C':'randomized dynamics fields, original demo starts unchanged',
                    'C_D':'training reset bank, randomized fields and alternating goal signs unchanged',
                    'D_E':'training geometry instead of held-out mesh; collision representation changes with geometry',
                    'E_F':'native port vs direct MJX on training geometry, same initial states/draws; includes sensor/motor numerical-port differences'},
                limits='No one-seed touch advantage or three-shape competence claim. Goal command at warm-up boundary is the original E3 interface. '
                       'Ideal full-pose camera only for vision-on observer; blind pose unavailable. '
                       'Vision-on handoff can select hold after goal recognition; primary success is already achieved then. '
                       'Adjacent diagnostic rows are not a replacement for the frozen Pass 0 demo.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
