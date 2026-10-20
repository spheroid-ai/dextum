"""Registered §16 follow-ups using the qualified PPO and full-state loop.

No engine changes: this module selects a sensor/model intervention, verifies
its frozen inputs, and reuses the existing collection, optimizer and monitor.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network_factory(width):
    from brax.training.agents.ppo import networks
    from experiments.train_e2 import HIDDEN, _preprocess
    def factory(*args, **kwargs):
        kwargs.update(preprocess_observations_fn=functools.partial(_preprocess, actor_width=width),
            policy_hidden_layer_sizes=HIDDEN, value_hidden_layer_sizes=HIDDEN,
            policy_obs_key='state', value_obs_key='privileged_state')
        return networks.make_ppo_networks(*args, **kwargs)
    return factory


def setup(reg, case):
    assert reg['authority'] == 'user_review_section16_followups'
    assert not os.environ.get('XLA_FLAGS', '')
    root = Path(__file__).resolve().parents[1]
    for name, digest in reg['source_sha256'].items():
        assert sha(root/name) == digest, name
    for name, digest in reg['evidence_sha256'].items():
        assert sha(name) == digest, name
    job = reg['cases'][case]
    for key in ('model', 'checkpoint'):
        assert sha(job[key]) == job[key+'_sha256'], key
    os.environ.update(DEXTUM_KIT_XML=job['model'], DEXTUM_SENSOR_LEVEL=job['sensor_level'], DEXTUM_LEGACY_PREFIX='1')
    if job['kind'] == 'dense':
        assert sha(job['layout']) == job['layout_sha256']
        os.environ['DEXTUM_DENSE_LAYOUT'] = job['layout']
        from pass1c.dense_skin import BlindRotation
    else:
        from pass1c.environment import BlindRotation
    env = BlindRotation(job.get('arm', 'c_touch'))
    return job, env


def main():
    import jax
    import jax.numpy as j
    import numpy as np
    import vendor.aero_official.vision_env
    from brax.io import model
    from brax.training.agents.ppo import networks
    from pass1c.environment import CRITIC_SIZE, EPISODE
    from pass1c import resumable_ppo
    from pass1 import faithful
    from pass1.qualification import require_training_pass
    from mujoco_playground.config import manipulation_params
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--mode', choices=['audit','qualify','train','t1'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    job, env = setup(reg, a.case)
    jax.config.update('jax_default_matmul_precision', 'highest')
    assert any(d.platform=='gpu' for d in jax.devices())
    width = 3*env.frame
    factory = network_factory(width)
    net = factory(observation_size={'state':width,'privileged_state':CRITIC_SIZE}, action_size=env.action_size)
    if a.mode == 'audit':
        from pass1 import audit_training
        from pass1c import environment, train
        # The diagnostic's wrapper/unroll instrumentation is unchanged. Bind
        # the exact environment and network width in this isolated process.
        environment.BlindRotation = lambda arm: env
        train.network_factory = lambda arm: factory
        sys.argv = ['audit_training','--kit','--section13','--variant','barrier','--probe','--output',str(a.output)]
        audit_training.main()
        path = a.output/'result.json'
        result = json.loads(path.read_text())
        result.update(followup_case=a.case, followup_registration_sha256=sha(a.registration),
            followup_source_sha256=reg['source_sha256'], actor_width=width)
        path.write_text(json.dumps(result,indent=2)+'\n')
        require_training_pass(result)
        return
    if a.mode == 't1':
        evaluate(job, env, net, a.output)
        return
    detector = json.loads(Path(job['detector']).read_text())
    require_training_pass(detector)
    assert detector['model_sha256'] == job['model_sha256']
    assert detector['followup_registration_sha256'] == sha(a.registration)
    if a.mode == 'train':
        q = Path(job['qualification'])
        rate, resume = json.loads((q/'throughput.json').read_text()),json.loads((q/'resume_test.json').read_text())
        assert rate['state']=='PASS' and resume['state']=='PASS'
        assert rate['metadata']['registration_sha256'] == sha(a.registration)
        assert rate['projected_rounded_budget_s'] <= 4800
    a.output.mkdir(parents=True,exist_ok=bool(a.resume))
    config = manipulation_params.brax_ppo_config('AeroCubeRotateZAxis').to_dict()
    config.pop('network_factory')
    config.update(episode_length=EPISODE,num_envs=8192,num_timesteps=20000000,seed=job['seed'],
        log_training_metrics=True,training_metrics_steps=327680,deterministic_eval=True,num_eval_envs=128)
    per_update = config['batch_size']*config['num_minibatches']*config['unroll_length']
    config['num_evals'] = max(2,20000000//(10*per_update)+1)
    params = model.load_params(job['checkpoint'])
    layers = params[1]['params']
    first = next(k for k in layers if 'kernel' in layers[k])
    old_width = layers[first]['kernel'].shape[0]
    assert old_width%3==0 and width>=old_width
    def restore(fresh):
        from pass1c.warm_start import expand
        changed = expand(params, old_width//3, width//3, fresh) if width!=old_width else params
        # The old actor is identical on equal old channels and zero new inputs.
        old_net=network_factory(old_width)(observation_size={'state':old_width,'privileged_state':CRITIC_SIZE},action_size=env.action_size)
        old_fn=networks.make_inference_fn(old_net)(params,deterministic=True)
        new_fn=networks.make_inference_fn(net)(changed,deterministic=True)
        x=jax.random.uniform(jax.random.PRNGKey(72),(16,3,old_width//3),minval=-2,maxval=2)
        y=j.pad(x,((0,0),(0,0),(0,(width-old_width)//3)))
        def action(fn,x):
            return fn({'state':x.reshape(16,-1),'privileged_state':j.zeros((16,CRITIC_SIZE))},jax.random.PRNGKey(0))[0]
        error=float(j.max(j.abs(action(old_fn,x)-action(new_fn,y))))
        parity=dict(state='PASS' if error<1e-5 else 'FAIL',maximum_action_error=error,old_width=old_width,new_width=width)
        (a.output/'warm_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        assert parity['state']=='PASS'
        return changed
    metadata=dict(case=a.case,configuration=config,registration_sha256=sha(a.registration),
        model_sha256=job['model_sha256'],checkpoint_sha256=job['checkpoint_sha256'],sources=reg['source_sha256'])
    record=dict(state='STARTING',job=job,configuration=config,actor_inputs=width,
        registration_sha256=sha(a.registration),metadata=metadata,started_at=time.time())
    (a.output/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic()
    checkpoints=json.loads((a.output/'checkpoints.json').read_text()) if a.resume else []
    def progress(steps, metrics):
        values={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
        assert all(np.isfinite(v) for v in values.values())
        row=dict(steps=int(steps),elapsed_s=time.monotonic()-started,metrics=values)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(steps=int(steps),elapsed_s=row['elapsed_s'])),flush=True)
    def save(steps, make_policy, values):
        del make_policy
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(values))
        if any(c['steps']==int(steps) for c in checkpoints):return
        target=a.output/f'policy_{int(steps)}.pkl'
        model.save_params(target,values)
        checkpoints.append(dict(steps=int(steps),path=target.name,elapsed_s=time.monotonic()-started,kept=True))
        (a.output/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    monitor=None
    if a.mode=='train':
        from pass1c.compare import MonitorGuard
        monitor=MonitorGuard(env,networks.make_inference_fn(net),a.output/'monitor')
    try:
        resumable_ppo.train(environment=env,network_factory=factory,
            wrap_env_fn=functools.partial(faithful.wrap_for_training,full_reset=True),
            randomization_fn=env.randomize,progress_fn=progress,policy_params_fn=save,
            restore_params=restore,run_evals=False,
            full_state_options=dict(directory=str(a.output/'full_state'),metadata=metadata,
                resume=str(a.resume) if a.resume else None,qualify=a.mode=='qualify',monitor=monitor),**config)
        record.update(state='COMPLETE',training_attempts=(0 if a.mode=='qualify' or job.get('review_control') else 1))
    except BaseException as exc:
        record.update(state='EXECUTION_FAILURE',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(wall_s=time.monotonic()-started,checkpoints=checkpoints)
        (a.output/'report.json').write_text(json.dumps(record,indent=2)+'\n')


def evaluate(job,env,net,output):
    import jax
    import jax.numpy as j
    import numpy as np
    from brax.io import model
    from brax.training.agents.ppo import networks
    from experiments.evaluate_e3 import CONDITIONS,HELD_OUT,make_rollout,summarise
    case=Path(job['training'])
    assert json.loads((case/'report.json').read_text())['state']=='COMPLETE'
    checkpoint=case/'policy_20316160.pkl'
    params=model.load_params(checkpoint)
    factory=networks.make_inference_fn(net)
    models,axes=env.randomize(env.mjx_model,jax.random.split(jax.random.PRNGKey(7),200))
    keys=jax.vmap(jax.random.split)(jax.random.split(jax.random.PRNGKey(2026),200))[...,1]
    @jax.jit
    def run(parameters,m,offset):
        infer=factory(parameters,deterministic=True)
        return make_rollout(env,lambda obs:infer(obs,jax.random.PRNGKey(0))[0],axes)(m,keys,offset)
    output.mkdir(parents=True,exist_ok=False)
    rows,arrays={},{}
    for name,c in CONDITIONS.items():
        m=models;mj=env.mj_model
        if 'friction' in c:
            ids=np.flatnonzero(mj.geom_bodyid!=0)
            m=m.tree_replace({'geom_friction':m.geom_friction.at[:,ids,0].set(c['friction'])})
        if 'gain' in c:
            m=m.tree_replace({'actuator_gainprm':m.actuator_gainprm.at[:,:,0].multiply(c['gain']),
                'actuator_biasprm':m.actuator_biasprm.at[:,:,1].multiply(c['gain'])})
        if 'mass' in c:
            cube=env.cube_body;inertia=j.asarray(mj.body_inertia[cube])*c['mass']/mj.body_mass[cube]
            m=m.tree_replace({'body_mass':m.body_mass.at[:,cube].set(c['mass']),'body_inertia':m.body_inertia.at[:,cube].set(inertia)})
        trace,sign=run(params,m,j.asarray(c.get('offset',(0.,0.)),j.float32))
        trace=jax.tree.map(np.asarray,trace);assert all(np.isfinite(v).all() for v in trace.values())
        values=summarise(trace,np.asarray(sign))
        rows[name]=dict(n=200,success=int(values['success'].sum()),drops=int(values['dropped'].sum()))
        arrays.update({name+'/'+k:v for k,v in values.items()})
        np.savez_compressed(output/'per_start.npz',**arrays,reset_keys=np.asarray(keys))
        (output/'result.json').write_text(json.dumps(dict(state='RUNNING',conditions=rows),indent=2)+'\n')
        print(json.dumps(dict(condition=name,**rows[name])),flush=True)
    result=dict(state='COMPLETE',conditions=rows,held_out_macro_success=float(np.mean([rows[k]['success']/200 for k in HELD_OUT])),
        model_sha256=job['model_sha256'],checkpoint_sha256=sha(checkpoint),scope='E3 cube physical-variation subset; one seed')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Vast only')
    main()
