"""Independent CPU scoring: curriculum metrics and full orientation kept distinct."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as j
import mujoco
import numpy as np
from .cpu_eval import make_model,reset,observation
from .evaluate_r0 import official_draws,set_draw
from .orientation_curriculum import OrientationCurriculum,STAGES,effective_error,WARMUP,TIMEOUT
from .orientation_policy import OrientationPolicy
from .vision_math import multiply,angle
from .goal_yaw import yaw_valid
from .vision_oracle import advance_sensor
from .policy import TRAIN_SCALE


class CpuOrientation:
    def __init__(self,stage,strict=False,history=1):
        if history not in (1,3):raise ValueError('Unsupported history length')
        self.history=history
        self.stage=STAGES[stage] if isinstance(stage,int) else stage
        if strict:self.stage=replace(self.stage,tilt_weight=1.,tolerance=np.pi/18,hold_steps=40)
        self.env=OrientationCurriculum(self.stage)
        self.capture=jax.jit(self.env.oracle.capture);self.advance=jax.jit(advance_sensor)
        self.pack=jax.jit(self.env.observations);self.full_error=jax.jit(angle)
        self.error=jax.jit(lambda q,g:angle(effective_error(q,g,self.stage.tilt_weight),j.array([1.,0.,0.,0.])))

    def reset(self,m,seed):
        self.model=m;self.data,self.rng=reset(m,seed);self.info=self.env.reset(jax.random.PRNGKey(seed)).info
        self.last=np.zeros(7,np.float32);self.data.sensordata[:]=0
        self.base,self.rng=observation(self.data,self.last,self.rng,True)
        self.age=0;self.hold=0;self.goal_time=0;self.full_hold=0
        self._history=None;self._history_age=-1
        if self.history==3:self.packet()

    def packet(self):
        current=np.asarray(self.pack({'state':j.array(self.base),'privileged_state':j.zeros(89)},self.info,j.array(self.data.qpos[-4:]))['state'])
        if self.history==1:return current
        if self._history is None:self._history=np.tile(current,3)
        elif self._history_age!=self.age:self._history=np.r_[current,self._history[:68]]
        else:self._history[:34]=current  # Repeated reads do not advance time.
        self._history_age=self.age
        return self._history.copy()

    def step(self,action):
        active=self.age>=WARMUP
        self.data.ctrl[:]=self.model.key('home').ctrl+TRAIN_SCALE*action
        self._advance_physics()
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):raise RuntimeError('Invalid reference physics')
        self.base,self.rng=observation(self.data,self.last,self.rng,True);self.last=action
        self.age+=1;self.info['age']=j.array(self.age,j.int32);q=j.array(self.data.qpos[-4:]);p=j.array(self.data.qpos[16:19])
        if self.age==WARMUP:self.info['goal_world']=multiply(self.info['goal_delta'],q)
        key,cap,delay=jax.random.split(self.info['vision_rng'],3)
        capture,_=self.capture(cap,self.info['cameras'],p,q,j.array(self.data.geom_xpos),
            j.array(self.data.geom_xmat.reshape(-1,3,3)),self.env.palm_position,self.env.palm_quaternion)
        self.info['sensor'],self.info['packet']=self.advance(self.info['sensor'],capture,jax.random.bernoulli(delay,.2).astype(j.int32))
        self.info['vision_rng']=key
        drop=bool(self.data.sensor('cube_position').data[2]<-.05)
        error=float(self.error(q,self.info['goal_world']));full=float(self.full_error(q,self.info['goal_world']))
        valid=bool(yaw_valid(q,self.info['goal_world'])) if self.stage.tilt_weight==0 else True
        self.hold=self.hold+1 if active and error<self.stage.tolerance and valid and not drop else 0
        self.full_hold=self.full_hold+1 if active and full<.2 and not drop else 0
        self.goal_time+=int(active);self.info['hold']=j.array(self.hold,j.int32);self.info['goal_time']=j.array(self.goal_time,j.int32)
        if self.history==3:self.packet()
        return dict(error=error,full_error=full,drop=drop,success=self.hold>=self.stage.hold_steps,
            active=active,hold=self.hold,full_hold=self.full_hold,visible=bool(self.info['packet'][9]>.5))

    def _advance_physics(self):
        for _ in range(5):mujoco.mj_step(self.model,self.data)


def evaluate(checkpoint,output,stage=0,trials=40,final=False,randomized=True,strict=False,
             seed_base=None,dr_seed_base=None):
    if (seed_base is None)!=(dr_seed_base is None):raise ValueError('Override both start and physics seed banks together')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);policy=OrientationPolicy(checkpoint)
    cpu=CpuOrientation(stage,strict,history=policy.history)
    start=seed_base if seed_base is not None else (71001 if final else 61001)
    keys=dr_seed_base if dr_seed_base is not None else (72001 if final else 62001)
    draws=official_draws(range(keys,keys+trials)) if randomized else None
    if draws is not None:np.savez(output/'models.npz',**draws)
    r=dict(state='RUNNING',stage=cpu.stage.__dict__,stage_index=stage,strict=strict,trials=trials,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),camera='qualified noisy tag oracle',
        randomized=randomized,history=policy.history,final=final,seed_base=start,dr_seed_base=keys,rows=[],started_at=time.time())
    path=output/'report.json'
    try:
        for i in range(trials):
            m,_=make_model()
            if draws is not None:set_draw(m,draws,i)
            cpu.reset(m,start+i);trace=[];best=np.pi;best_full=np.pi;longest=0;full_longest=0;visible=0
            for tick in range(WARMUP+TIMEOUT):
                obs=cpu.packet();action=policy(obs);step=cpu.step(action);visible+=step['visible']
                if step['active']:
                    best=min(best,step['error']);best_full=min(best_full,step['full_error'])
                    longest=max(longest,step['hold']);full_longest=max(full_longest,step['full_hold'])
                trace.append(dict(tick=tick,**step,action=action.tolist(),cube_qpos=cpu.data.qpos[16:].tolist()))
                if step['drop'] or step['success']:break
            row=dict(trial=i,seed=start+i,success=step['success'],drop=step['drop'],reached=best<cpu.stage.tolerance,
                best_error_deg=best*180/np.pi,best_full_error_deg=best_full*180/np.pi,longest_hold=longest,
                full_wuji_success=full_longest>=5,visible_fraction=visible/(tick+1),steps=tick+1)
            (output/f'trial_{i}.json').write_text(json.dumps(dict(row,goal=np.asarray(cpu.info['goal_world']).tolist(),trace=trace))+'\n')
            r['rows'].append(row);path.write_text(json.dumps(r,indent=2)+'\n')
        r.update(state='COMPLETE',successes=sum(x['success'] for x in r['rows']),drops=sum(x['drop'] for x in r['rows']),
            reaches=sum(x['reached'] for x in r['rows']),full_wuji_successes=sum(x['full_wuji_success'] for x in r['rows']),
            mean_best_error_deg=float(np.mean([x['best_error_deg'] for x in r['rows']])))
    except BaseException as exc:r.update(state='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:r['wall_s']=time.time()-r['started_at'];path.write_text(json.dumps(r,indent=2)+'\n')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',type=int,default=0);p.add_argument('--trials',type=int,default=40);p.add_argument('--final',action='store_true')
    p.add_argument('--nominal',action='store_true');p.add_argument('--strict',action='store_true')
    p.add_argument('--seed-base',type=int);p.add_argument('--dr-seed-base',type=int);a=p.parse_args()
    r=evaluate(a.checkpoint,a.output,a.stage,a.trials,a.final,not a.nominal,a.strict,a.seed_base,a.dr_seed_base)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
