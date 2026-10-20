"""Existing §20 calibrated D5 instrument applied to rotation masking states.

No optimizer update. This is task qualification, not an engine experiment.
The 14 October limit is max(1e-4, ten times the direct/direct compiler floor).
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast only')
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CPU backend only serves diagnostic callbacks.
import functools
import hashlib
import inspect
import json
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import pickle
import gzip
import vendor.aero_official.vision_env
from brax.training import acting, gradients
from mujoco_playground import wrapper
from experiments.evaluate_e3 import step_one
from pass0.faithful import MODEL_KEY
from pass1c import resumable_ppo
from pass1c.environment import EPISODE
from pass1.faithful import wrap_for_training
HORIZON = 40


class AuditComplete(Exception):
    pass


def audit(env, output, registration_sha256, network_factory):
    assert jax.default_backend() == 'gpu' and all(device.platform == 'gpu' for device in jax.devices())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    n=16
    rng=jax.random.split(jax.random.PRNGKey(2026),n)
    randomize=functools.partial(env.randomize,
        rng=jax.random.split(jax.random.PRNGKey(7),n))
    _,axes=randomize(env.mjx_model)
    direct=jax.vmap(functools.partial(step_one,env),in_axes=(axes,0,0))
    traces=[];calls=[];proof={};captured={}

    def pack_state(state):
        # Brax flattens every extra with reshape(-1, ...), which cannot
        # represent zero-width MJX leaves. Omit only those empty arrays;
        # their shapes/dtypes restore them losslessly after collection.
        leaves,structure=jax.tree.flatten(state)
        layout=[(x.shape,x.dtype,bool(x.size)) for x in leaves]
        if 'state_layout' in captured:
            assert captured['state_layout']==layout
            assert captured['state_structure']==structure
        else:
            captured.update(state_layout=layout,state_structure=structure)
        return tuple(x for x in leaves if x.size)

    class Taps(wrapper.Wrapper):
        def reset(self, ignored_rng):
            state=self.env.reset(rng)
            clean=state.replace(info={k:v for k,v in state.info.items() if k!=MODEL_KEY})
            tap=dict(input_state=pack_state(clean), action=j.zeros((n,env.action_size)),
                     qpos=state.data.qpos, ctrl=state.data.ctrl,
                     force=state.data.xfrc_applied, comparable=j.zeros(n,bool),
                     done=j.zeros(n), invalid=j.zeros(n))
            return state.replace(info={**state.info,'hold_audit':tap})

        def step(self,state,action):
            info={k:v for k,v in state.info.items() if k!='hold_audit'}
            state=state.replace(info=info)
            clean=state.replace(info={k:v for k,v in info.items() if k!=MODEL_KEY})
            result=self.env.step(state,action)
            tap=dict(input_state=pack_state(clean),action=action,qpos=result.data.qpos,
                     ctrl=result.data.ctrl,force=result.data.xfrc_applied,
                     comparable=result.done<.5,done=result.done,
                     invalid=j.zeros(n))
            return result.replace(info={**result.info,'hold_audit':tap})

    def wrap(environment,episode_length=EPISODE,action_repeat=1,**unused):
        return Taps(wrap_for_training(environment,episode_length,action_repeat,
                                     randomization_fn=randomize,full_reset=True))

    original_unroll=acting.generate_unroll
    original_gradient=gradients.gradient_update_fn
    original_pmap=jax.pmap

    def tap_unroll(*args,**kwargs):
        stack=[frame.function for frame in inspect.stack()]
        assert 'collect_step' in stack
        calls.append(stack)
        kwargs['extra_fields']=(*kwargs.get('extra_fields',()),'hold_audit')
        end,data=original_unroll(*args,**kwargs)
        jax.debug.callback(lambda value:traces.append(jax.tree.map(np.asarray,value)),
                           data.extras['state_extras']['hold_audit'])
        return end,data

    def identity_gradient(*args,**kwargs):
        def update(params,normalizer,data,key,optimizer_state):
            return (j.array(0.),{'no_update':j.array(0.)}),params,optimizer_state
        return update

    def tap_pmap(fn,*args,**kwargs):
        compiled=original_pmap(fn,*args,**kwargs)
        if getattr(fn,'__name__','')!='collect_step':return compiled
        def execute(carry):
            before,state,keys=carry
            if 'models' not in captured:captured['models']=jax.tree.map(lambda x:np.asarray(x[0]),state.info[MODEL_KEY])
            result=compiled(carry)
            jax.block_until_ready(result);jax.effects_barrier()
            assert len(traces)==1
            assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
                jax.tree.leaves(before.params),jax.tree.leaves(result[0][0].params)))
            changed=state.info[MODEL_KEY].tree_replace({'body_mass':
                state.info[MODEL_KEY].body_mass.at[:,:,env.cube_body].multiply(1.1)})
            forced=state.replace(info={**state.info,MODEL_KEY:changed,
                                      'steps':j.full_like(state.info['steps'],EPISODE-1)})
            reset=compiled((before,forced,keys))
            jax.block_until_ready(reset);jax.effects_barrier()
            assert len(traces)==2
            proof['full_reset_exercised']=bool(np.all(traces[1]['done'][0]>.5))
            proof['full_reset_preserves_physics']=all(np.array_equal(np.asarray(a),np.asarray(b))
                for a,b in zip(jax.tree.leaves(changed),jax.tree.leaves(reset[0][1].info[MODEL_KEY])))
            raise AuditComplete()
        return execute

    acting.generate_unroll,gradients.gradient_update_fn,jax.pmap=tap_unroll,identity_gradient,tap_pmap
    try:
        resumable_ppo.train(environment=env,network_factory=network_factory,
            wrap_env_fn=wrap,randomization_fn=env.randomize,num_envs=n,num_eval_envs=n,
            num_timesteps=n*HORIZON,episode_length=EPISODE,batch_size=n,num_minibatches=1,
            unroll_length=40,num_updates_per_batch=1,num_evals=1,num_resets_per_eval=0,
            run_evals=False,seed=1,normalize_observations=True,learning_rate=0.,
            full_state_options=dict(directory=str(output/'unused_full_state'),metadata={'hold_audit':True}))
    except AuditComplete:
        pass
    finally:
        acting.generate_unroll,gradients.gradient_update_fn,jax.pmap=original_unroll,original_gradient,original_pmap
    assert len(traces)==2 and calls
    trace=traces[0]
    packed=iter(trace['input_state'])
    leaves=[next(packed) if present else np.zeros((HORIZON,)+shape,dtype)
            for shape,dtype,present in captured['state_layout']]
    assert next(packed,None) is None
    trace['input_state']=jax.tree.unflatten(captured['state_structure'],leaves)
    # Freeze the complete transition inputs before either direct compilation.
    # This is one shared bank for collection/A/B, never separate rollouts.
    mask=np.asarray(trace['comparable'],bool)
    assert mask.shape==(HORIZON,n) and int(mask.sum())>0
    frozen=output/'transition_inputs.pkl.gz'
    with gzip.open(frozen,'wb',compresslevel=1) as f:
        pickle.dump(dict(trace=trace,models=captured['models'],axes=axes),f,protocol=5)
    input_digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    print('Frozen rotation transitions; compiling direct A (batch16) and B (batch1)',flush=True)
    models=jax.tree.map(j.asarray,captured['models'])

    def outputs(model,state,action):
        value=step_one(env,model,state,action)
        return value.data.qpos,value.data.ctrl,value.data.xfrc_applied,value.done

    direct_a=jax.jit(jax.vmap(outputs,in_axes=(axes,0,0)))
    direct_b=jax.jit(outputs)
    single_models=[jax.tree.map(lambda v,a:v if a is None else v[index],models,axes,
                               is_leaf=lambda x:x is None) for index in range(n)]
    collected=[];floors=[];controls=[];forces=[];invalid_direct=0
    for tick in range(HORIZON):
        if not mask[tick].any():continue
        state=jax.tree.map(lambda x:j.asarray(x[tick]),trace['input_state'])
        actions=j.asarray(trace['action'][tick])
        a=jax.tree.map(np.asarray,direct_a(models,state,actions))
        for index in range(n):
            if not mask[tick,index]:continue
            one=jax.tree.map(lambda x:x[index],state)
            b=jax.tree.map(np.asarray,direct_b(single_models[index],one,actions[index]))
            collected.append(np.abs(trace['qpos'][tick,index]-a[0][index]))
            floors.append(np.abs(a[0][index]-b[0]))
            controls.append([np.max(np.abs(trace['ctrl'][tick,index]-a[1][index])),
                             np.max(np.abs(a[1][index]-b[1]))])
            forces.append([np.max(np.abs(trace['force'][tick,index]-a[2][index])),
                           np.max(np.abs(a[2][index]-b[2]))])
            invalid_direct+=int(a[3][index]>.5 or b[3]>.5)
    collected,floors=np.asarray(collected),np.asarray(floors)
    cmax,fmax=collected.max(-1),floors.max(-1)
    median,floor=float(np.median(cmax)),float(np.median(fmax))
    limit=max(1e-4,10*floor)
    finite=all(np.isfinite(x).all() for x in (collected,floors,controls,forces))
    preserved=all(proof.values())
    proof.update(state='PASS' if finite and preserved and median<=limit and
                 not np.any(trace['invalid']) and not invalid_direct else 'FAIL',
        finite=bool(finite),matched_step_median=median,
        matched_step_p95=float(np.percentile(cmax,95)),matched_step_max=float(cmax.max()),
        direct_direct_median=floor,direct_direct_p95=float(np.percentile(fmax,95)),
        direct_direct_max=float(fmax.max()),calibrated_limit=limit,floor_multiplier=10,
        base_limit=1e-4,per_dof_max=collected.max(0).tolist(),
        maximum_control_error=np.max(controls,axis=0).tolist(),
        maximum_applied_force_error=np.max(forces,axis=0).tolist(),
        comparable_transitions=len(collected),invalid_entries=int(np.count_nonzero(trace['invalid'])),
        invalid_direct_terminals=invalid_direct,
        actual_ppo_training_path=True,actual_call_stacks=calls,
        criterion_version='work_order_2026_10_14_calibrated',
        instance_id=os.environ['CONTAINER_ID'],
        model_sha256=hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest(),
        registration_sha256=registration_sha256,optimizer_steps=0,training_attempts=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transition_inputs_sha256=input_digest,same_inputs_all_three_paths=True,
        collection_compilation='actual pmap collect_step with full auto-reset',
        direct_compilations=['jit vmap, batch16','independent jit, batch1'],
        no_flags_or_solver_changes=True)
    if proof['state']!='PASS':
        fields=('timestep','iterations','ls_iterations','solver','cone','impratio','tolerance')
        options={name:np.asarray(getattr(models.opt,name)).tolist() for name in fields}
        diff=dict(state='OPTIONS_DIFF_REPORTED_ONCE',collection=options,direct_a=options,direct_b=options,
                  same_carried_model=True,substeps=int(env.n_substeps),action_repeat=1,
                  model_dtype=str(models.body_mass.dtype),model_sha256=proof['model_sha256'],
                  calls=['StatefulAutoReset -> EpisodeWrapper -> StatefulRandomization -> MaskRotation.step',
                         'step_one -> MaskRotation.step (vmap16)','step_one -> MaskRotation.step (jit1)'],
                  no_fix_applied=True,next='Ruling required before any path fix')
        (output/'options_diff.json').write_text(json.dumps(diff,indent=2)+'\n')
    np.savez_compressed(output/'comparisons.npz',collection_direct=collected,direct_direct=floors,
                        controls=controls,forces=forces)
    (output/'result.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('actual_call_stacks','per_dof_max')}),flush=True)
    if proof['state']!='PASS':raise RuntimeError('Task-calibrated rotation detector failed')
    return proof



def require_pass(record):
    limit = max(1e-4, 10 * record['direct_direct_median'])
    if not (record.get('state') == 'PASS' and record.get('finite') is True
            and record.get('full_reset_exercised') is True
            and record.get('full_reset_preserves_physics') is True
            and record.get('actual_ppo_training_path') is True
            and record['matched_step_median'] <= limit
            and record['matched_step_median'] <= 1e-2):
        raise RuntimeError('GPU actual-collection qualification required')
