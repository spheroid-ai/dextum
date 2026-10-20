"""Remote same-input stepping diagnostic; distributions, not a D5 gate."""
import functools
import inspect
import json
import os
from pathlib import Path
import sys
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only')
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import vendor.aero_official.vision_env
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from envs.blind_rotation import BlindRotation, WARMUP, EPISODE
from experiments.evaluate_e3 import step_one
from pass0.faithful import wrap_for_training, MODEL_KEY


def stats(values):
    a=np.asarray(values).reshape(-1)
    if not len(a):return dict(n=0,median=None,p95=None,maximum=None)
    return dict(n=int(len(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a)))


def options(model):
    fields=('timestep','iterations','ls_iterations','solver','cone','impratio','tolerance','ls_tolerance',
            'integrator','jacobian','disableflags','enableflags')
    return {k:np.asarray(getattr(model.opt,k)).tolist() for k in fields if hasattr(model.opt,k)}


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    env=BlindRotation('c_touch');n=16
    rng=jax.random.split(jax.random.PRNGKey(2026),n)
    dr=jax.random.split(jax.random.PRNGKey(7),n)
    model,axes=domain_randomize(env.mjx_model,dr)
    wrapped=wrap_for_training(env,episode_length=EPISODE,randomization_fn=functools.partial(domain_randomize,rng=dr))
    initial=jax.jit(wrapped.reset)(rng)
    step_v=jax.vmap(functools.partial(step_one,env),in_axes=(axes,0,0))
    action=jnp.zeros((n,env.action_size),jnp.float32)
    candidate_pairs=np.asarray(initial.data.contact.geom[0])
    unique_pairs,pair_capacities=np.unique(candidate_pairs,axis=0,return_counts=True)

    def contacts(data):
        return jnp.sum(data.contact.dist < data.contact.includemargin,axis=-1)

    @jax.jit
    def probe(state):
        def body(s,_):
            raw=step_v(s.info[MODEL_KEY],s.replace(info={k:v for k,v in s.info.items() if k!=MODEL_KEY}),action)
            nxt=wrapped.step(s,action)
            comparable=(nxt.done<.5)&(raw.done<.5)
            return nxt,dict(qpos=jnp.abs(raw.data.qpos-nxt.data.qpos),qvel=jnp.abs(raw.data.qvel-nxt.data.qvel),
                            ctrl=jnp.abs(raw.data.ctrl-nxt.data.ctrl),
                            comparable=comparable,before=contacts(s.data),direct_after=contacts(raw.data),
                            wrapped_after=contacts(nxt.data))
        return jax.lax.scan(body,state,None,length=WARMUP)[1]

    metadata=dict(state='DIAGNOSTIC_REGISTERED',engines='Both JAX/MJX; no native/MJX contrast',
        ordered_checks=[
          dict(candidate='a_call_sequence',training_step_source=inspect.getsource(type(wrapped.env.env).step),
               common_env_step=inspect.getsource(BlindRotation.step),common_substep=inspect.getsource(mjx_env.step),
               control_dt=float(env.dt),substeps=int(env.n_substeps),
               direct='step_one -> env.step',training='auto-reset -> EpisodeWrapper -> vmap env.step',
               note='Both call the same env.step and motor clipping. Source retained for forward/velocity-clamp inspection.'),
          dict(candidate='b_options',training=options(wrapped.physics.model),direct=options(model),native=options(env.mj_model)),
          dict(candidate='c_contact_bookkeeping',mj_nconmax=int(env.mj_model.nconmax),
               mj_njmax=int(env.mj_model.njmax),mj_npair=int(env.mj_model.npair),mj_nexclude=int(env.mj_model.nexclude),
               geom_contype=np.asarray(env.mj_model.geom_contype).tolist(),
               geom_conaffinity=np.asarray(env.mj_model.geom_conaffinity).tolist(),
               data_contact_capacity=int(initial.data.contact.dist.shape[-1]),
               candidate_pair_slot_counts=[dict(geom_pair=pair.tolist(),slots=int(cap)) for pair,cap in zip(unique_pairs,pair_capacities)],
               training_and_direct_use_same_model_and_input_contact_slots=True,
               model_impl=str(getattr(model,'_impl',None).__class__)),
          dict(candidate='d_dtype',training_qpos=str(initial.data.qpos.dtype),training_qvel=str(initial.data.qvel.dtype),
               model_body_mass=str(model.body_mass.dtype),jax_enable_x64=bool(jax.config.jax_enable_x64))],
        contact_count_definition='contact.dist < contact.includemargin; reported at incoming and both outgoing control-step states',
        zero_contact_caveat='Zero at endpoints does not exclude a transient contact inside a control-step substep',
        xla_flags=os.environ.get('XLA_FLAGS',''),mujoco=mujoco.__version__,jax=jax.__version__)
    # Save the collision implementation that determines pair pruning and limits.
    import mujoco.mjx._src.collision_driver as collision
    (out/'collision_driver_source.txt').write_text(inspect.getsource(collision))
    (out/'settings.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Settings recorded; compiling same-input transition probe',flush=True)
    trace=jax.tree.map(np.asarray,probe(initial))
    mask=trace['comparable']
    count=np.maximum.reduce([trace['before'],trace['direct_after'],trace['wrapped_after']])
    model_native=env.mj_model
    cube_q=np.asarray(env._cube_qids,dtype=int)
    cube_j=int(model_native.body_jntadr[env.cube_body]);v0=int(model_native.jnt_dofadr[cube_j])
    groups={'qpos':{'cube_free_joint':cube_q,'hand_joints':np.setdiff1d(np.arange(model_native.nq),cube_q)},
            'qvel':{'cube_free_joint':np.arange(v0,v0+6),'hand_joints':np.setdiff1d(np.arange(model_native.nv),np.arange(v0,v0+6))}}
    distributions={}
    for label,chosen in [('all',mask)]+[(f'active_contacts_{k}',mask&(count==k)) for k in sorted(np.unique(count[mask]))]:
        row=dict(transitions=int(chosen.sum()),quantities={})
        row['actuator_ctrl_error']=stats(trace['ctrl'][chosen])
        for quantity in ('qpos','qvel'):
            values=trace[quantity][chosen]
            row['quantities'][quantity]=dict(per_coordinate=[stats(values[:,i]) for i in range(values.shape[-1])],
                groups={name:stats(values[:,ids]) for name,ids in groups[quantity].items()},
                transition_maximum=stats(np.max(values,axis=-1)))
        distributions[label]=row
    zero=mask&(count==0)
    zero_max=float(trace['qpos'][zero].max()) if zero.any() else None
    result=dict(state='DIAGNOSTIC_ONLY',n=16,control_steps=WARMUP,
        compared_transitions=int(mask.sum()),terminal_transitions=int((~mask).sum()),
        maximum_qpos_error=float(trace['qpos'][mask].max()),distributions=distributions,
        coordinate_indices={q:{g:ids.tolist() for g,ids in gs.items()} for q,gs in groups.items()},
        zero_contact_endpoint_maximum=zero_max,zero_contact_diagnostic_tolerance=1e-6,
        zero_contact_endpoint_check=None if zero_max is None else zero_max<=1e-6,
        settings_equal=options(wrapped.physics.model)==options(model),
        conclusion='Same-input same-engine code paths; contact solve/fusion and candidate sets remain Pass 1 investigation. Not a D5 veto.',
        contact_count_definition=metadata['contact_count_definition'],zero_contact_caveat=metadata['zero_contact_caveat'])
    np.savez_compressed(out/'transitions.npz',**trace)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='distributions'}),flush=True)


if __name__=='__main__':main()
