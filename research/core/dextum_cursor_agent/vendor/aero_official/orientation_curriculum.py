"""Camera PPO curriculum. Only its final stages use the full SO(3) criterion."""
from dataclasses import dataclass,asdict
import jax
import jax.numpy as j
from .goal_yaw import GoalYaw,WARMUP,TIMEOUT,yaw_valid
from .vision_env import CubeRotateZAxis
from .vision_math import unit,multiply,conjugate,axis_quat,rot6,angle
from .vision_oracle import initial_sensor,advance_sensor,sample_qualified_cameras,features


@dataclass(frozen=True)
class Stage:
    name:str
    minimum_deg:float
    maximum_deg:float
    cone_deg:float=0.
    tilt_weight:float=0.
    hold_steps:int=40
    tolerance:float=float(j.pi/18)
    uniform_so3:bool=False


STAGES=(Stage('yaw_15_60',15,60),Stage('yaw_30_180',30,180),
    Stage('tilt_25',15,45,tilt_weight=.25,hold_steps=5,tolerance=.2),
    Stage('tilt_50',15,45,tilt_weight=.5,hold_steps=5,tolerance=.2),
    Stage('full_orientation_yaw',15,45,tilt_weight=1.,hold_steps=5,tolerance=.2),
    Stage('cone_30',15,45,cone_deg=30,tilt_weight=1.,hold_steps=5,tolerance=.2),
    Stage('so3_15_45',15,45,cone_deg=180,tilt_weight=1.,hold_steps=5,tolerance=.2),
    Stage('so3_30_90',30,90,cone_deg=180,tilt_weight=1.,hold_steps=5,tolerance=.2),
    Stage('uniform_so3',0,180,cone_deg=180,tilt_weight=1.,hold_steps=5,tolerance=.2,uniform_so3=True))


def goal_delta(key,stage):
    k1,k2,k3=jax.random.split(key,3)
    if stage.uniform_so3:
        # Normalized isotropic R4 Gaussian is Haar-uniform on S3, hence SO(3).
        return unit(jax.random.normal(k1,(4,)))
    z=jax.random.uniform(k1,minval=j.cos(stage.cone_deg*j.pi/180),maxval=1.)
    az=jax.random.uniform(k2,minval=0.,maxval=2*j.pi)
    axis=j.array([j.sqrt(j.maximum(0.,1-z*z))*j.cos(az),j.sqrt(j.maximum(0.,1-z*z))*j.sin(az),z])
    theta=jax.random.uniform(k3,minval=stage.minimum_deg*j.pi/180,maxval=stage.maximum_deg*j.pi/180)
    return axis_quat(axis,theta)


def effective_error(current,goal,tilt_weight):
    """Swing/twist homotopy: yaw at0, exact geodesic orientation at1."""
    error=unit(multiply(goal,conjugate(current)))
    error=j.where(error[0]<0,-error,error)
    if tilt_weight==1.:return error
    n=j.sqrt(error[0]**2+error[3]**2)
    twist=j.where(n>1e-7,j.array([error[0],0.,0.,error[3]])/j.maximum(n,1e-7),j.array([1.,0.,0.,0.]))
    if tilt_weight==0.:return twist
    swing=unit(multiply(error,conjugate(twist)))
    phi=2*j.arctan2(j.linalg.norm(swing[1:]),j.maximum(swing[0],0.))
    return unit(multiply(axis_quat(swing[1:],tilt_weight*phi),twist))


def reward(previous,error_q,omega,delta,hold,elapsed,drop,stage,valid=True):
    d=angle(error_q,j.array([1.,0.,0.,0.]));drop=j.asarray(drop,dtype=j.bool_)
    within=(d<stage.tolerance)&j.asarray(valid)&(~drop)
    hold=j.where(within,hold+1,0);success=(hold>=stage.hold_steps)&(~drop)
    axis=unit(j.where(error_q[0]<0,-error_q[1:],error_q[1:]))
    terms=dict(progress=4*(previous-d),approach=.05*j.clip(j.dot(axis,omega),-3.,3.)*(~within),
        holding=.1*within,settling=-.0025*j.sum(omega**2)*within,time_cost=-.01*(~within),
        action_cost=-.00005*j.sum(delta**2),falling=-1.*drop,bonus=5.*success)
    return dict(**terms,reward=sum(terms.values()),error=d,hold=hold,success=success,within=within,
        timeout=(elapsed+1>=TIMEOUT)&(~success))


class OrientationCurriculum(GoalYaw):
    def __init__(self,stage=0,pose_mode='camera'):
        super().__init__(pose_mode);self.stage=STAGES[stage] if isinstance(stage,int) else stage

    def observations(self,base,info,q):
        measured=multiply(self.palm_quaternion,info['packet'][3:7])
        active=info['age']>=WARMUP
        identity=j.array([1.,0.,0.,0.])
        observed=j.where(active,effective_error(measured,info['goal_world'],self.stage.tilt_weight),identity)
        true=j.where(active,effective_error(q,info['goal_world'],self.stage.tilt_weight),identity)
        clock=j.array([j.where(active,info['goal_time']/TIMEOUT,(info['age']-WARMUP)/WARMUP)])
        return dict(state=j.concatenate((base['state'][:14],features(info['packet'],info['sensor']['age']),rot6(observed),clock)),
            privileged_state=j.concatenate((base['privileged_state'][:81],rot6(true),j.array([info['hold']/self.stage.hold_steps]),clock)))

    def reset(self,key):
        state=CubeRotateZAxis.reset(self,key)
        vision,camera,goal=jax.random.split(jax.random.fold_in(key,817),3)
        q=state.data.qpos[self._cube_qids][3:7];sensor=initial_sensor();delta=goal_delta(goal,self.stage)
        state.info.update(age=j.array(0,j.int32),goal_time=j.array(0,j.int32),hold=j.array(0,j.int32),
            goal_delta=delta,goal_world=multiply(delta,q),previous_error=angle(delta,j.array([1.,0.,0.,0.])),
            vision_rng=vision,cameras=sample_qualified_cameras(camera),sensor=sensor,
            packet=self.pose_packet(state.data) if self.pose_mode=='perfect' else sensor['last'])
        if self.pose_mode=='perfect':state.info['sensor']['age']=j.array(0,j.int32)
        for name in ('error','success','within','full_orientation_error','visible','progress','approach','holding','settling','time_cost','action_cost','falling','bonus'):
            state.metrics['target_'+name]=j.zeros(())
        return state.replace(obs=self.observations(state.obs,state.info,q))

    def step(self,state,action):
        active=state.info['age']>=WARMUP;previous=state.info['previous_error'];last=state.info['last_act']
        hold=state.info['hold'];elapsed=state.info['goal_time']
        base={k:v[:14 if k=='state' else 81] for k,v in state.obs.items()}
        state=self._advance_physics(state.replace(obs=base),action);info=state.info;info['age']+=1
        q=state.data.qpos[self._cube_qids][3:7];p=state.data.qpos[self._cube_qids][:3]
        info['goal_world']=j.where(info['age']==WARMUP,multiply(info['goal_delta'],q),info['goal_world'])
        if self.pose_mode=='perfect':
            info['packet']=self.pose_packet(state.data);info['sensor']['age']=j.array(0,j.int32)
        else:
            key,cap,delay=jax.random.split(info['vision_rng'],3)
            capture,_=self.oracle.capture(cap,info['cameras'],p,q,state.data.geom_xpos,state.data.geom_xmat,self.palm_position,self.palm_quaternion)
            info['sensor'],info['packet']=advance_sensor(info['sensor'],capture,jax.random.bernoulli(delay,.2).astype(j.int32))
            info['vision_rng']=key
        error_q=effective_error(q,info['goal_world'],self.stage.tilt_weight);drop=state.done>.5
        valid=yaw_valid(q,info['goal_world']) if self.stage.tilt_weight==0 else j.array(True)
        terms=reward(previous,error_q,self.get_cube_angvel(state.data),action-last,hold,elapsed,drop,self.stage,valid)
        info.update(hold=j.where(active,terms['hold'],0),goal_time=j.where(active,elapsed+1,0),previous_error=terms['error'])
        for name in ('progress','approach','holding','settling','time_cost','action_cost','falling','bonus'):
            state.metrics['target_'+name]=j.where(active,terms[name],j.where(drop,terms[name],0.) if name=='falling' else 0.)
        state.metrics.update(target_error=terms['error'],target_success=(active&terms['success']).astype(j.float32),
            target_within=(active&terms['within']).astype(j.float32),target_full_orientation_error=angle(q,info['goal_world']),target_visible=info['packet'][9])
        return state.replace(obs=self.observations(state.obs,info,q),reward=j.where(active,terms['reward'],j.where(drop,-1.,0.)),
            done=(drop|(active&(terms['success']|terms['timeout']))).astype(j.float32))

    def _advance_physics(self,state,action):
        return CubeRotateZAxis.step(self,state,action)


class HistoryOrientation(OrientationCurriculum):
    """Three observable frames, current first; no additional privileged actor data."""
    def reset(self,key):
        state=super().reset(key)
        history=j.tile(state.obs['state'],3)
        state.info['orientation_history']=history
        return state.replace(obs={**state.obs,'state':history})

    def step(self,state,action):
        state=super().step(state,action)
        history=j.concatenate((state.obs['state'],state.info['orientation_history'][:68]))
        state.info['orientation_history']=history
        return state.replace(obs={**state.obs,'state':history})


class SequenceOrientation(HistoryOrientation):
    """Resample successful goals without resetting physics, sensors or history."""
    def reset(self,key):
        state=super().reset(key)
        state.info.update(sequence_goal_rng=jax.random.fold_in(key,841),sequence_goals=j.array(0,j.int32))
        state.metrics['sequence_goals']=j.zeros(())
        return state

    def resample_completed(self,state,success):
        info=state.info
        key,draw=jax.random.split(info['sequence_goal_rng'])
        delta=goal_delta(draw,self.stage)
        q=state.data.qpos[self._cube_qids][3:7]
        goal=multiply(delta,q)
        info['sequence_goal_rng']=j.where(success,key,info['sequence_goal_rng'])
        info['sequence_goals']+=success.astype(j.int32)
        info['goal_delta']=j.where(success,delta,info['goal_delta'])
        info['goal_world']=j.where(success,goal,info['goal_world'])
        info['hold']=j.where(success,0,info['hold'])
        info['goal_time']=j.where(success,0,info['goal_time'])
        next_error=angle(effective_error(q,goal,self.stage.tilt_weight),j.array([1.,0.,0.,0.]))
        info['previous_error']=j.where(success,next_error,info['previous_error'])
        # Goal issuance changes the current packet, never either past frame.
        obs=self.observations(state.obs,info,q)
        history=info['orientation_history'].at[:34].set(obs['state'])
        info['orientation_history']=history
        state.metrics['sequence_goals']=success.astype(j.float32)
        return state.replace(obs={**obs,'state':history},done=j.where(success,0.,state.done))

    def step(self,state,action):
        state=super().step(state,action)
        # The base predicate excludes drops. Failed goals still terminate;
        # Brax's episode wrapper also imposes the unchanged 500-step limit.
        return self.resample_completed(state,state.metrics['target_success']>.5)
