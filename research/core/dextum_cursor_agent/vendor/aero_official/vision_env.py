"""Official Aero mechanics plus simulated tag observations and orientation goals."""
import sys
import jax
import jax.numpy as j
import mujoco
from etils import epath
from .policy import PLAYGROUND,MENAGERIE
sys.path.insert(0,str(PLAYGROUND))
from mujoco_playground._src import mjx_env
mjx_env.MENAGERIE_PATH=epath.Path(MENAGERIE)
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis,domain_randomize
from .vision_math import multiply,conjugate,angle,rot6,sample_goal,task_step
from .vision_oracle import TagOracle,sample_cameras,sample_qualified_cameras,initial_sensor,advance_sensor,features

ACTOR_SIZE=34
CRITIC_SIZE=89


class VisualReorient(CubeRotateZAxis):
    def __init__(self,phase=0,recipe='vg1',hold_steps=None,camera_mode='legacy'):
        super().__init__()
        if camera_mode not in ('legacy','qualified'):raise ValueError('Unknown camera layout mode')
        self.camera_mode=camera_mode
        self.phase=phase
        self.recipe=recipe
        self.hold_steps=(40 if phase<2 else 5) if hold_steps is None else hold_steps
        self.tolerance=j.pi/18 if phase<2 else .2
        self.oracle=TagOracle(self.mj_model)
        data=mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model,data,self.mj_model.key('home').id)
        mujoco.mj_forward(self.mj_model,data)
        self.palm_position=j.array(data.xpos[self.oracle.palm_id])
        self.palm_quaternion=j.array(data.xquat[self.oracle.palm_id])
        self.cube_body=self.mj_model.body('cube').id

    def add_observation(self,base_obs,info,true_quaternion):
        packet=info['packet']
        measured_error=multiply(info['goal_palm'],conjugate(packet[3:7]))
        actual_goal=multiply(self.palm_quaternion,info['goal_palm'])
        true_error=multiply(actual_goal,conjugate(true_quaternion))
        clock=j.array([j.minimum(info['goal_time'],280)/280])
        actor=j.concatenate((base_obs['state'][:14],features(packet,info['sensor']['age']),rot6(measured_error),clock))
        critic=j.concatenate((base_obs['privileged_state'][:81],rot6(true_error),j.array([info['hold']/self.hold_steps]),clock))
        return {'state':actor,'privileged_state':critic}

    def reset(self,rng):
        state=super().reset(rng)
        key,camera_key,goal_key=jax.random.split(jax.random.fold_in(rng,711),3)
        quat=state.data.qpos[self._cube_qids][3:7]
        goal=sample_goal(goal_key,quat,self.phase)
        sensor=initial_sensor()
        cameras=(sample_cameras if self.camera_mode=='legacy' else sample_qualified_cameras)(camera_key)
        state.info.update(vision_rng=key,cameras=cameras,sensor=sensor,packet=sensor['last'],
            goal_palm=multiply(conjugate(self.palm_quaternion),goal),hold=j.array(0,j.int32),goal_time=j.array(0,j.int32),
            goal_count=j.array(0,j.int32),previous_error=angle(quat,goal))
        for name in ('goal_success','goal_timeout','goal_error','camera_visible','goal_progress','goal_bonus','goal_hold','goal_settle'):
            state.metrics[name]=j.zeros(())
        if self.recipe=='vg2':
            for name in ('goal_alignment','goal_action_cost','goal_drop_cost'):
                state.metrics[name]=j.zeros(())
        return state.replace(obs=self.add_observation(state.obs,state.info,quat))

    def step(self,state,action):
        old_action=state.info['last_act']
        base_obs={k:v[:14 if k=='state' else 81] for k,v in state.obs.items()}
        advanced=super().step(state.replace(obs=base_obs),action)
        info=advanced.info
        key,capture_key,delay_key,goal_key=jax.random.split(info['vision_rng'],4)
        q=advanced.data.xquat[self.cube_body]
        capture,_=self.oracle.capture(capture_key,info['cameras'],advanced.data.xpos[self.cube_body],q,
            advanced.data.geom_xpos,advanced.data.geom_xmat,self.palm_position,self.palm_quaternion)
        sensor,packet=advance_sensor(info['sensor'],capture,jax.random.bernoulli(delay_key,.2).astype(j.int32))
        goal=multiply(self.palm_quaternion,info['goal_palm'])
        error=angle(q,goal)
        step=task_step(info['previous_error'],error,info['hold'],info['goal_time'],
                       action-old_action,self.get_cube_angvel(advanced.data),advanced.done>.5,
                       tolerance=self.tolerance,hold_steps=self.hold_steps,recipe=self.recipe)
        next_goal=sample_goal(goal_key,q,self.phase)
        success=step['success']
        info.update(vision_rng=key,sensor=sensor,packet=packet,
            goal_palm=j.where(success,multiply(conjugate(self.palm_quaternion),next_goal),info['goal_palm']),
            hold=j.where(success,0,step['hold']),goal_time=j.where(success,0,step['time']),
            goal_count=info['goal_count']+success,previous_error=j.where(success,angle(q,next_goal),error))
        advanced.metrics.update(goal_success=success.astype(j.float32),goal_timeout=step['timeout'].astype(j.float32),
            goal_error=error,camera_visible=packet[9],goal_progress=step['progress'],goal_bonus=step['bonus'],
            goal_hold=step['holding'],goal_settle=step['settling'])
        if self.recipe=='vg2':
            advanced.metrics.update(goal_alignment=step['alignment'],goal_action_cost=step['action_rate'],
                                    goal_drop_cost=step['falling'])
        return advanced.replace(obs=self.add_observation(advanced.obs,info,q),reward=step['reward'],
            done=j.maximum(advanced.done,step['timeout'].astype(j.float32)))
