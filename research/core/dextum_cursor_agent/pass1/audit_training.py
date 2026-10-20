"""Instrument actual installed PPO training_epoch -> training_step -> unroll.

Cloud only. No policy optimization: gradient update is an identity, and PPO
exits after the first real training epoch plus a forced-reset replay. Batch
size is reduced to the registered 16-start diagnostic, unroll remains 40.
Physics output taps and a separate matched-transition probe are declared
instrumentation; installed PPO/acting code is captured, not rewritten.
"""
import argparse
import copy
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only; no local numerical imports')

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import vendor.aero_official.vision_env
from brax.training import acting, gradients
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from envs.blind_rotation import BlindRotation, EPISODE, WARMUP
from experiments.evaluate_e3 import step_one
from pass0.faithful import MODEL_KEY
from pass0.train_e3 import network_factory
from tools.physics_ref import RANDOMISED


class AuditComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'barrier'], required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--kit', action='store_true', help='Section 12 corrected primitive Realistic model')
    parser.add_argument('--section13', action='store_true', help='Final split collection path; only matched-step is a hard stop')
    parser.add_argument('--starts', type=int, choices=[16,200], default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    physics = importlib.import_module('pass0.faithful' if args.variant == 'baseline' else 'pass1.faithful')
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert jax.devices()[0].platform == 'gpu'
    if args.kit:
        from pass1c.environment import BlindRotation as Environment
        from pass1c.train import network_factory as factory
        from pass1c import resumable_ppo as trainer
    else:
        Environment,factory=BlindRotation,network_factory
        trainer=ppo
    if args.section13:
        from pass1c import resumable_ppo as trainer
    final_collection = args.kit or args.section13
    env = Environment('c_touch')
    n = args.starts
    rng = jax.random.split(jax.random.PRNGKey(2026), n)
    dr = jax.random.split(jax.random.PRNGKey(7), n)
    randomizer=env.randomize if args.kit else domain_randomize
    randomize = functools.partial(randomizer, rng=dr)
    model, axes = randomize(env.mjx_model)
    direct = jax.vmap(functools.partial(step_one, env), in_axes=(axes, 0, 0))
    traces, calls, initial = [], [], {}
    result = dict(state='RUNNING', variant=args.variant, probe=args.probe,
                  n=n, reset_seed=2026, dr_seed=7, training_attempts=0,
                  wrapper_sha256=hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),
                  instrumentation='16-env batch; original 40-step nested PPO unroll and pmap; identity optimizer; output taps; optional same-state direct probe',
                  source_sha256={})
    for name, obj in [('ppo', trainer), ('generate_unroll', acting.generate_unroll), ('actor_step', acting.actor_step)]:
        source = inspect.getsource(obj)
        (args.output/(name+'_source.txt')).write_text(source)
        result['source_sha256'][name] = hashlib.sha256(source.encode()).hexdigest()
    (args.output/'registration.json').write_text(json.dumps(result, indent=2)+'\n')

    class Taps(wrapper.Wrapper):
        def reset(self, ignored_rng):
            s = self.env.reset(rng)
            tap = dict(qpos=s.data.qpos, error=jnp.zeros_like(s.data.qpos),
                       contacts=jnp.zeros(n, jnp.int32), comparable=jnp.zeros(n, bool),
                       done=jnp.zeros(n), ctrl_error=jnp.zeros_like(s.data.ctrl))
            return s.replace(info={**s.info, 'audit': tap})

        def step(self, state, action):
            info = {k:v for k,v in state.info.items() if k != 'audit'}
            state = state.replace(info=info)
            nxt = self.env.step(state, action)
            error = jnp.zeros_like(nxt.data.qpos)
            ctrl_error = jnp.zeros_like(nxt.data.ctrl)
            comparable = nxt.done < .5
            if args.probe:
                raw = direct(info[MODEL_KEY], state.replace(info={k:v for k,v in info.items() if k != MODEL_KEY}), action)
                error = jnp.abs(raw.data.qpos-nxt.data.qpos)
                ctrl_error = jnp.abs(raw.data.ctrl-nxt.data.ctrl)
                comparable &= raw.done < .5
            contacts = jnp.sum(state.data.contact.dist < state.data.contact.includemargin, axis=-1)
            tap = dict(qpos=nxt.data.qpos, error=error, contacts=contacts,
                       comparable=comparable, done=nxt.done, ctrl_error=ctrl_error)
            return nxt.replace(info={**nxt.info, 'audit':tap})

    def wrap(e, episode_length=EPISODE, action_repeat=1, **kwargs):
        return Taps(physics.wrap_for_training(e, episode_length=episode_length,
                    action_repeat=action_repeat, randomization_fn=randomize, full_reset=True))

    original_unroll = acting.generate_unroll
    original_gradient = gradients.gradient_update_fn
    original_pmap = jax.pmap

    def tap_unroll(*a, **kw):
        stack = [f.function for f in inspect.stack()]
        assert ('collect_step' if final_collection else 'training_step') in stack, stack
        calls.append(stack)
        kw['extra_fields'] = (*kw.get('extra_fields', ()), 'audit')
        end, data = original_unroll(*a, **kw)
        jax.debug.callback(lambda values: traces.append(jax.tree.map(np.asarray, values)),
                           data.extras['state_extras']['audit'])
        return end, data

    def identity_gradient(*a, **kw):
        def update(params, normalizer, data, key, optimizer_state):
            return (jnp.array(0.), {'audit_no_update': jnp.array(0.)}), params, optimizer_state
        return update

    def instrument_pmap(fn, *a, **kw):
        compiled = original_pmap(fn, *a, **kw)
        if getattr(fn, '__name__', '') not in ('training_epoch','resumable_update','collect_step'):
            return compiled
        original_compiled=compiled
        if getattr(fn,'__name__','') in ('resumable_update','collect_step'):
            def compiled(ts,es,keys):
                output=original_compiled((ts,es,keys))
                carry,metrics=output[0],output[1]
                return carry[0],carry[1],metrics

        def execute(training_state, env_state, keys):
            # This is called by installed PPO's training_epoch_with_timing.
            initial['state'] = jax.tree.map(lambda x:x[0], env_state)
            output = compiled(training_state, env_state, keys)
            jax.block_until_ready(output)
            jax.effects_barrier()
            assert len(traces) == 1
            assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x,y in zip(
                jax.tree.leaves(training_state.params), jax.tree.leaves(output[0].params)))
            changed = env_state.info[MODEL_KEY].tree_replace({
                'body_mass': env_state.info[MODEL_KEY].body_mass.at[:,:,env.cube_body].multiply(1.1)})
            forced = env_state.replace(info={**env_state.info, MODEL_KEY:changed,
                        'steps':jnp.full_like(env_state.info['steps'], EPISODE-1)})
            reset_output = compiled(training_state, forced, keys)
            jax.block_until_ready(reset_output)
            jax.effects_barrier()
            assert len(traces) == 2
            result['full_reset_exercised'] = bool(np.all(traces[1]['done'][0] > .5))
            result['full_reset_preserves_physics'] = all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(
                jax.tree.leaves(changed),jax.tree.leaves(reset_output[1].info[MODEL_KEY])))
            result['policy_parameters_unchanged'] = True
            raise AuditComplete()
        return (lambda carry:execute(*carry)) if getattr(fn,'__name__','') in ('resumable_update','collect_step') else execute

    acting.generate_unroll, gradients.gradient_update_fn, jax.pmap = tap_unroll, identity_gradient, instrument_pmap
    print('Running installed PPO train/epoch/step/unroll with no policy updates', flush=True)
    try:
        options=dict(full_state_options=dict(directory=str(args.output/'initial_full_state'),metadata={'diagnostic':'actual training path, identity optimizer'})) if final_collection else {}
        trainer.train(environment=env, network_factory=factory('c_touch'),
                  wrap_env_fn=wrap, randomization_fn=randomizer,
                  num_envs=n, num_eval_envs=n, num_timesteps=n*40, episode_length=EPISODE,
                  batch_size=n, num_minibatches=1, unroll_length=40, num_updates_per_batch=1,
                  num_evals=1, num_resets_per_eval=0, run_evals=False, seed=1,
                  normalize_observations=True, deterministic_eval=True, learning_rate=0.,**options)
    except AuditComplete:
        pass
    finally:
        acting.generate_unroll, gradients.gradient_update_fn, jax.pmap = original_unroll, original_gradient, original_pmap
    assert calls and (('collect_step' in calls[0]) if final_collection else ('training_step' in calls[0] and 'training_epoch' in calls[0]))
    result['actual_ppo_training_path']=True
    result['actual_call_stacks'] = calls
    if args.kit:
        result['model_sha256']=hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest()
    s0 = initial['state']
    train = traces[0]['qpos'][:WARMUP]
    cpu = np.zeros_like(train, dtype=np.float64)
    for i in range(n):
        native = copy.deepcopy(env.mj_model)
        for field in RANDOMISED:
            getattr(native,field)[:] = np.asarray(getattr(model,field)[i])
        data = mujoco.MjData(native)
        for field in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
            getattr(data,field)[:] = np.asarray(getattr(s0.data,field)[i])
        for tick in range(WARMUP):
            data.ctrl[:] = np.asarray(env._default_ctrl) if hasattr(env,'_default_ctrl') else np.asarray(s0.data.ctrl[i])
            from dextum.contact_space import HOME
            data.ctrl[:] = np.asarray(HOME)
            for _ in range(env.n_substeps): mujoco.mj_step(native,data)
            cpu[tick,i] = data.qpos
    z = int(np.asarray(env._cube_qids)[2])
    errors = np.abs(train[-1,:,z]-cpu[-1,:,z])
    result.update(within_5mm=int(np.sum(errors<.005)), height_error_mm=(errors*1000).tolist(),
                  finite=bool(np.isfinite(train).all() and np.isfinite(cpu).all()))
    if args.probe:
        mask=traces[0]['comparable'][:WARMUP]
        error=traces[0]['error'][:WARMUP][mask]
        maximum=error.max(-1)
        result.update(matched_transitions=int(mask.sum()),matched_step_median=float(np.median(maximum)),
                      matched_step_p95=float(np.percentile(maximum,95)),matched_step_max=float(maximum.max()),
                      maximum_control_error=float(traces[0]['ctrl_error'][:WARMUP][mask].max()))
    result['warmup_pass'] = result['within_5mm']>=14 and result['finite'] and result['full_reset_exercised'] and result['full_reset_preserves_physics']
    result['state'] = ('PASS' if result['warmup_pass'] and result['matched_step_median']<1e-5 else 'FAIL') if args.probe else ('WARMUP_PASS' if result['warmup_pass'] else 'WARMUP_FAIL')
    if args.section13:
        xyz = np.asarray(env._cube_qids)[:3]
        positions_train, positions_native = train[-1][:,xyz], cpu[-1][:,xyz]
        position_error = np.linalg.norm(positions_train-positions_native,axis=-1)
        from envs.grasp_edge import DROP_Z
        terminated=(traces[0]['done'][:WARMUP]>.5).any(0)
        retained_train, retained_native = (positions_train[:,2]>=DROP_Z)&~terminated, positions_native[:,2]>=DROP_Z
        failing = (position_error>=.005) | (retained_train!=retained_native)
        result.update(criterion_version='review_section15',detector_median_limit=1e-4,
            collection_path='resumable_ppo.collect_step -> acting.generate_unroll with full auto-reset',
            within_5mm_position=int(np.sum(position_error<.005)),
            position_agreement_rate=float(np.mean(position_error<.005)),
            position_error_median_mm=float(np.median(position_error)*1000),
            position_error_p95_mm=float(np.percentile(position_error,95)*1000),
            training_terminal_starts=np.flatnonzero(terminated).tolist(),
            continuous_error_excludes_auto_reset=dict(n=int((~terminated).sum()),
                median_mm=float(np.median(position_error[~terminated])*1000),
                p95_mm=float(np.percentile(position_error[~terminated],95)*1000)),
            qualitative_disagreement_count=int(np.sum(retained_train!=retained_native)),
            qualitative_disagreement_rate=float(np.mean(retained_train!=retained_native)),
            in_palm_definition='Existing E3 retained/not-dropped boundary: cube centre z >= DROP_Z and no earlier training terminal event; xyz also reported; no geometric retuning',
            drop_z=float(DROP_Z),
            failing_starts=[dict(start=int(i),position_error_mm=float(position_error[i]*1000),
                training_xyz_m=positions_train[i].tolist(),native_xyz_m=positions_native[i].tolist(),
                training_auto_reset_before_warmup_end=bool(terminated[i]),
                training_in_palm=bool(retained_train[i]),native_in_palm=bool(retained_native[i]),
                both_in_palm=bool(retained_train[i] and retained_native[i])) for i in np.flatnonzero(failing)],
            solver=dict(iterations=int(env.mj_model.opt.iterations),ls_iterations=int(env.mj_model.opt.ls_iterations),
                        timestep=float(env.mj_model.opt.timestep),substeps=int(env.n_substeps)),
            sensor_level=os.environ.get('DEXTUM_SENSOR_LEVEL','Realistic') if args.kit else 'Ideal',
            bank_sha256={name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name,value in [('reset',rng),('dr',dr)]})
        result['state'] = ('PASS' if result['finite'] and result['full_reset_exercised'] and
            result['full_reset_preserves_physics'] and result.get('matched_step_median',float('inf'))<1e-4 else 'FAIL') if args.probe else 'MEASURED'
    np.savez_compressed(args.output/'traces.npz', native=cpu, **{'training_'+k:v for k,v in traces[0].items()},
                        **{'forced_reset_'+k:v for k,v in traces[1].items()})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='actual_call_stacks'}),flush=True)
    if result['state'] in ('FAIL','WARMUP_FAIL'):raise SystemExit(3)


if __name__ == '__main__':main()
