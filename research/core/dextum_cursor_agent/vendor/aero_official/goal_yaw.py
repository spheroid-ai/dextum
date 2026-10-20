"""GY1: explicitly easier yaw-and-hold curriculum, never full-SO(3) success."""
import jax
import jax.numpy as j
from .vision_env import VisualReorient
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis
from .vision_math import multiply,conjugate,axis_quat,rot6,matrix,angle
from .vision_oracle import features,initial_sensor,advance_sensor,sample_qualified_cameras

WARMUP=20
TIMEOUT=280
HOLD=40
TOLERANCE=j.pi/18


def yaw_error(current,goal):
    q=multiply(goal,conjugate(current))
    error=2*j.arctan2(q[...,3],q[...,0])
    return j.arctan2(j.sin(error),j.cos(error))


def yaw_valid(current,goal):
    q=multiply(goal,conjugate(current))
    return q[...,0]**2+q[...,3]**2>.04


def reward_terms(previous,error,omega,delta,hold,elapsed,drop,valid=True):
    drop=j.asarray(drop,dtype=j.bool_);valid=j.asarray(valid,dtype=j.bool_)
    within=(j.abs(error)<TOLERANCE)&valid&(~drop)
    hold=j.where(within,hold+1,0);success=(hold>=HOLD)&(~drop)
    progress=4*(j.abs(previous)-j.abs(error))
    # Preserve a dense reason to continue the existing gait while approaching.
    approach=.05*j.clip(j.sign(error)*omega[2],-3.,3.)*(~within)
    holding=.1*within
    settling=-.0025*j.sum(omega**2)*within
    time_cost=-.01*(~within)
    action_cost=-.00005*j.sum(delta**2)
    falling=-1.*drop
    bonus=5.*success
    return dict(reward=progress+approach+holding+settling+time_cost+action_cost+falling+bonus,
        progress=progress,approach=approach,holding=holding,settling=settling,time_cost=time_cost,
        action_cost=action_cost,falling=falling,bonus=bonus,hold=hold,success=success,timeout=(elapsed+1>=TIMEOUT)&(~success))


class GoalYaw(VisualReorient):
    def __init__(self,pose_mode='perfect'):
        super().__init__(0,camera_mode='qualified')
        if pose_mode not in ('perfect','camera'):raise ValueError(pose_mode)
        self.pose_mode=pose_mode

    def pose_packet(self,data):
        q=data.qpos[self._cube_qids][3:7];p=data.qpos[self._cube_qids][:3]
        return j.concatenate((matrix(self.palm_quaternion).T@(p-self.palm_position),
            multiply(conjugate(self.palm_quaternion),q),j.array([1e6,1e6,1.])))

    def observations(self,base,info,q):
        packet=info['packet'];measured=multiply(self.palm_quaternion,packet[3:7])
        error=yaw_error(measured,info['goal_world']);true_error=yaw_error(q,info['goal_world'])
        active=info['age']>=WARMUP
        goal_feature=rot6(axis_quat(j.array([0.,0.,1.]),j.where(active,error,0.)))
        truth_feature=rot6(axis_quat(j.array([0.,0.,1.]),j.where(active,true_error,0.)))
        clock=j.array([j.where(active,info['goal_time']/TIMEOUT,(info['age']-WARMUP)/WARMUP)])
        return dict(state=j.concatenate((base['state'][:14],features(packet,info['sensor']['age']),goal_feature,clock)),
            privileged_state=j.concatenate((base['privileged_state'][:81],truth_feature,j.array([info['hold']/HOLD]),clock)))

    def reset(self,rng):
        state=CubeRotateZAxis.reset(self,rng);key,camera_key=jax.random.split(jax.random.fold_in(rng,711))
        q=state.data.qpos[self._cube_qids][3:7];sensor=initial_sensor()
        state.info.update(age=j.array(0,j.int32),goal_time=j.array(0,j.int32),hold=j.array(0,j.int32),
            goal_world=multiply(axis_quat(j.array([0.,0.,1.]),j.pi/6),q),previous_error=j.array(j.pi/6),
            vision_rng=key,cameras=sample_qualified_cameras(camera_key),sensor=sensor,
            packet=self.pose_packet(state.data) if self.pose_mode=='perfect' else sensor['last'])
        for name in ('yaw_error','yaw_success','yaw_within','full_orientation_error','camera_visible',
                     'yaw_progress','yaw_approach','yaw_holding','yaw_settling','yaw_time_cost','yaw_action_cost','yaw_falling','yaw_bonus'):
            state.metrics[name]=j.zeros(())
        return state.replace(obs=self.observations(state.obs,state.info,q))

    def step(self,state,action):
        old_action=state.info['last_act'];active=state.info['age']>=WARMUP
        old_error=state.info['previous_error'];old_hold=state.info['hold'];old_time=state.info['goal_time']
        base={k:v[:14 if k=='state' else 81] for k,v in state.obs.items()}
        state=CubeRotateZAxis.step(self,state.replace(obs=base),action);info=state.info
        q=state.data.qpos[self._cube_qids][3:7];p=state.data.qpos[self._cube_qids][:3]
        info['age']+=1
        goal=multiply(axis_quat(j.array([0.,0.,1.]),j.pi/6),q)
        info['goal_world']=j.where(info['age']==WARMUP,goal,info['goal_world'])
        if self.pose_mode=='perfect':info['packet']=self.pose_packet(state.data)
        else:
            key,capture_key,delay_key=jax.random.split(info['vision_rng'],3)
            capture,_=self.oracle.capture(capture_key,info['cameras'],p,q,state.data.geom_xpos,state.data.geom_xmat,
                self.palm_position,self.palm_quaternion)
            info['sensor'],info['packet']=advance_sensor(info['sensor'],capture,jax.random.bernoulli(delay_key,.2).astype(j.int32))
            info['vision_rng']=key
        error=yaw_error(q,info['goal_world']);drop=state.done>.5
        terms=reward_terms(old_error,error,self.get_cube_angvel(state.data),action-old_action,old_hold,old_time,drop,yaw_valid(q,info['goal_world']))
        info['hold']=j.where(active,terms['hold'],0);info['goal_time']=j.where(active,old_time+1,0)
        info['previous_error']=error
        for name in ('progress','approach','holding','settling','time_cost','action_cost','falling','bonus'):
            state.metrics['yaw_'+name]=j.where(active,terms[name],j.where(drop,terms[name],0.) if name=='falling' else 0.)
        state.metrics.update(yaw_error=j.abs(error),yaw_success=(active&terms['success']).astype(j.float32),
            yaw_within=(active&(terms['hold']>0)).astype(j.float32),camera_visible=info['packet'][9],
            full_orientation_error=angle(q,info['goal_world']))
        reward=j.where(active,terms['reward'],j.where(drop,-1.,0.))
        done=drop|(active&(terms['success']|terms['timeout']))
        return state.replace(obs=self.observations(state.obs,info,q),reward=reward,done=done.astype(j.float32))
