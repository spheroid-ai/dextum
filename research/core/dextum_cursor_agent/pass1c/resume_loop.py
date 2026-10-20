"""Engine-independent full-pytree checkpoint boundary for pinned Brax PPO.

Only our own checkpoints may be loaded (pickle is not an untrusted format).
The engine's complete environment state is opaque to this module.
"""
import hashlib
import gzip
import json
import os
import pickle
import time
from pathlib import Path
import jax
import numpy as np
from pass1c.boundary_monitor import complete_boundary_monitor, replay_boundary_monitor


def save(path, payload, metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        record=dict(version=1,metadata=metadata,payload=jax.device_get(payload))
        if path.suffix=='.gz':
            with gzip.GzipFile(fileobj=f,mode='wb',compresslevel=1) as compressed:
                pickle.dump(record,compressed,protocol=5)
        else:
            pickle.dump(record,f,protocol=5)
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def load(path, metadata):
    path=Path(path)
    with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as f:record=pickle.load(f)
    if record['version']!=1 or record['metadata']!=metadata:
        raise ValueError('Full checkpoint configuration/source/model mismatch')
    return jax.tree.map(lambda x:jax.device_put(x) if isinstance(x,np.ndarray) else x,record['payload'])


def compare(expected, actual):
    x,xt=jax.tree.flatten(jax.device_get(expected));y,yt=jax.tree.flatten(jax.device_get(actual))
    if xt!=yt:return dict(passed=False,reason='pytree mismatch')
    largest=0.;bad=0
    for a,b in zip(x,y):
        a,b=np.asarray(a),np.asarray(b)
        if a.dtype.kind in 'fc':
            delta=float(np.max(np.abs(a-b))) if a.size else 0.
            largest=max(largest,delta)
            bad+=not np.allclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=False)
        else:bad+=not np.array_equal(a,b)
    return dict(passed=bad==0,mismatched_leaves=int(bad),maximum_absolute_difference=largest,rtol=1e-5,atol=1e-6)


def canonical(payload):
    # Match Brax's _strip_weak_type before compiled epochs. Host checkpoint
    # arrays cannot encode weak_type; both uninterrupted and restored inputs
    # therefore use explicit dtypes, avoiding a different compiled signature.
    return jax.tree.map(lambda x:jax.numpy.asarray(x,dtype=x.dtype) if hasattr(x,'dtype') else x,payload)


def checkpoint_due(updates, within_epoch, steps, budget, interval, *, qualification_end=False):
    """Always preserve boundaries; optionally amortize intermediate disk writes."""
    if type(interval) is not int or interval < 1:
        raise ValueError('Checkpoint interval must be a positive integer')
    return (updates % interval == 0 or within_epoch == 0 or steps >= budget
            or qualification_end)


def run(*,update,collect,training_state,env_state,local_key,key_envs,reset_fn,devices,
        steps_per_epoch,reset_per_epoch,num_timesteps,steps_per_update,
        make_policy,progress_fn,policy_params_fn,options):
    out=Path(options['directory']);out.mkdir(parents=True,exist_ok=True)
    checkpoint_interval=options.get('checkpoint_interval_updates',1)
    checkpoint_due(0,0,0,num_timesteps,checkpoint_interval)
    metadata=options['metadata']
    metadata={**metadata,'steps_per_epoch':int(steps_per_epoch),'reset_per_epoch':bool(reset_per_epoch),'devices':devices}
    epoch_key,local_key=jax.random.split(local_key)
    payload=dict(carry=(training_state,env_state,jax.random.split(epoch_key,devices)),
                 local_key=local_key,key_envs=key_envs,within_epoch=0,updates=0)
    if options.get('resume'):
        payload=load(options['resume'],metadata)
    def prepare(p):
        p=canonical(p)
        if 'pending' not in p:
            carry,data,key_sgd=collect(p['carry'])
            jax.block_until_ready((carry,data,key_sgd))
            p={**p,'carry':carry,'pending':(data,key_sgd)}
        return canonical(p)
    payload=prepare(payload)
    latest=out/'latest.pkl'
    save(latest,payload,metadata)
    monitor=options.get('monitor')
    # A child can stop after saving the boundary but before its monitor. Finish
    # that exact policy's check before any update, including at the final budget.
    ts=payload['carry'][0]
    replay_boundary_monitor(monitor, updates=payload['updates'], within_epoch=payload['within_epoch'],
        steps=int(jax.tree.map(lambda x:x[0],ts.env_steps)), budget=num_timesteps,
        params=jax.tree.map(lambda x:x[0],(ts.normalizer_params,ts.params.policy,ts.params.value)),
        save_policy=lambda steps,params:policy_params_fn(steps,make_policy,params))
    def advance(p):
        p=prepare(p)
        data,key_sgd=p['pending']
        carry,metrics=update((p['carry'],data,key_sgd));jax.block_until_ready((carry,metrics))
        p={k:v for k,v in p.items() if k!='pending'}
        p.update(carry=carry,within_epoch=p['within_epoch']+1,updates=p['updates']+1)
        if p['within_epoch']==steps_per_epoch:
            keys=jax.vmap(lambda x,s:jax.random.split(x[0],s),in_axes=(0,None))(p['key_envs'],p['key_envs'].shape[1])
            state=reset_fn(keys) if reset_per_epoch else carry[1]
            epoch_key,next_key=jax.random.split(p['local_key'])
            p.update(carry=(carry[0],state,jax.random.split(epoch_key,devices)),local_key=next_key,key_envs=keys,within_epoch=0)
        jax.block_until_ready(p)
        return canonical(p),metrics
    start=time.monotonic();timings=[];metrics={}
    qualification=bool(options.get('qualify'))
    if options.get('expected_next'):
        expected=canonical(load(options['expected_next'],metadata))
        payload,metrics=advance(payload)
        report=dict(loss=compare(expected['loss'],metrics),full_next_state=compare(expected['payload'],payload),
                    fresh_process=True,checkpoint=str(options['resume']))
        report['state']='PASS' if report['loss']['passed'] and report['full_next_state']['passed'] else 'FAIL'
        (out/'process_resume.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PROCESS_RESUME',json.dumps(report),flush=True)
        if report['state']!='PASS':raise RuntimeError('Fresh-process continuation differs')
        ts=payload['carry'][0]
        params=jax.tree.map(lambda x:x[0],(ts.normalizer_params,ts.params.policy,ts.params.value))
        return make_policy,params,metrics
    if qualification:
        # First actual PPO update compiles and populates optimizer/sensor state.
        before=time.monotonic();payload,metrics=advance(payload)
        compile_first_s=time.monotonic()-before
        payload=prepare(payload)
        save(out/'resume_test.pkl.gz',payload,metadata)
        expected,expected_loss=advance(payload)
        save(out/'expected_next.pkl.gz',dict(payload=expected,loss=expected_loss),metadata)
        # The reference is audit data, not a second resident training run.
        # Release its device buffers before restoring the complete checkpoint.
        # Neither compiled collection nor learning is changed by this transfer.
        input_host=jax.device_get(payload)
        expected_host=jax.device_get(expected)
        expected_loss_host=jax.device_get(expected_loss)
        checkpoint_optimizer_nonzero=any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(input_host['carry'][0].optimizer_state))
        payload=expected=expected_loss=None
        import gc
        gc.collect()
        restored=canonical(load(out/'resume_test.pkl.gz',metadata))
        input_parity=compare(input_host,restored)
        actual,actual_loss=advance(restored)
        report=dict(input_state=input_parity,loss=compare(expected_loss_host,actual_loss),full_next_state=compare(expected_host,actual),metadata=metadata,
                    serialized_bytes=(out/'resume_test.pkl.gz').stat().st_size,first_update_compile_and_run_s=compile_first_s,
                    optimizer_state_nonzero=checkpoint_optimizer_nonzero,checkpoint_sha256=hashlib.sha256((out/'resume_test.pkl.gz').read_bytes()).hexdigest(),
                    qualification_archives='gzip; rolling latest.pkl remains uncompressed on persistent disk')
        report['state']='PASS' if report['input_state']['passed'] and report['loss']['passed'] and report['full_next_state']['passed'] and report['optimizer_state_nonzero'] else 'FAIL'
        (out/'resume_test.json').write_text(json.dumps(report,indent=2)+'\n')
        print('RESUME',json.dumps(report),flush=True)
        if report['state']!='PASS':raise RuntimeError('Full-state next-update resume check failed')
        payload=actual
        del restored,actual,input_host,expected_host,expected_loss_host
        gc.collect()
        if options.get('resume_only'):
            ts=payload['carry'][0]
            params=jax.tree.map(lambda x:x[0],(ts.normalizer_params,ts.params.policy,ts.params.value))
            return make_policy,params,metrics
    def count(ts):return int(jax.tree.map(lambda x:x[0],ts.env_steps))
    while count(payload['carry'][0])<num_timesteps:
        before=time.monotonic();payload,metrics=advance(payload)
        if count(payload['carry'][0])<num_timesteps:payload=prepare(payload)
        elapsed=time.monotonic()-before
        steps=count(payload['carry'][0])
        save_s=0.
        wrote_checkpoint=checkpoint_due(payload['updates'],payload['within_epoch'],steps,
            num_timesteps,checkpoint_interval,qualification_end=qualification and len(timings)==9)
        if wrote_checkpoint:
            save_before=time.monotonic();save(latest,payload,metadata);save_s=time.monotonic()-save_before
        timings.append(dict(update=payload['updates'],seconds=elapsed,checkpoint_s=save_s))
        values={f'training/{k}':float(np.mean(np.asarray(v))) for k,v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite PPO loss')
        values.update({'training/sps':steps_per_update/(elapsed+save_s),'training/checkpoint_s':save_s})
        progress_fn(steps,values)
        # Full continuation is always durable before every evaluator boundary.
        if payload['within_epoch']==0 or steps>=num_timesteps:
            ts=payload['carry'][0]
            params=jax.tree.map(lambda x:x[0],(ts.normalizer_params,ts.params.policy,ts.params.value))
            if monitor is not None:
                complete_boundary_monitor(monitor,steps,params,
                    lambda tick,weights:policy_params_fn(tick,make_policy,weights))
            else:
                policy_params_fn(steps,make_policy,params)
        if qualification and len(timings)>=10:break
    if qualification:
        total=sum(x['seconds']+x['checkpoint_s'] for x in timings)
        rate=steps_per_update*len(timings)/total
        rounded_budget=((num_timesteps+steps_per_update-1)//steps_per_update)*steps_per_update
        record=dict(state='PASS' if rounded_budget/rate<=4800 else 'TOO_SLOW',updates=len(timings),
                    env_steps_per_s=rate,projected_20M_s=20000000/rate,includes_checkpoints=True,
                    rounded_budget_steps=int(rounded_budget),projected_rounded_budget_s=rounded_budget/rate,
                    excludes_initial_compilation_and_resume_verification=True,timings=timings,
                    checkpoint_interval_updates=checkpoint_interval,
                    optimizer_updates_are_discarded=True,training_attempts=0,metadata=metadata)
        (out/'throughput.json').write_text(json.dumps(record,indent=2)+'\n')
        print('THROUGHPUT',json.dumps(record),flush=True)
    ts=payload['carry'][0]
    params=jax.tree.map(lambda x:x[0],(ts.normalizer_params,ts.params.policy,ts.params.value))
    policy_params_fn(count(ts),make_policy,params)
    return make_policy,params,metrics


def check_ideal(directory, resume_directory=None):
    """Independent continuation test on the previously qualified Ideal model.

    Two small PPO updates and a disk-restored replay; parameters discarded.
    This does not qualify corrected-model physics or full-scale throughput.
    """
    import functools
    import vendor.aero_official.vision_env
    from envs.blind_rotation import BlindRotation, EPISODE
    from pass1.train_e3 import network_factory
    from pass1.faithful import wrap_for_training
    from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
    from pass1c.resumable_ppo import train
    if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
    jax.config.update('jax_default_matmul_precision','highest')
    train(environment=BlindRotation('c_touch'),num_timesteps=1280,num_envs=16,num_eval_envs=16,
          episode_length=EPISODE,unroll_length=40,batch_size=16,num_minibatches=1,
          num_updates_per_batch=4,num_evals=2,num_resets_per_eval=1,
          normalize_observations=True,learning_rate=3e-4,discounting=.97,entropy_cost=.01,
          seed=1,network_factory=network_factory('c_touch'),run_evals=False,
          wrap_env_fn=functools.partial(wrap_for_training,full_reset=True),randomization_fn=domain_randomize,
          full_state_options=dict(directory=directory,qualify=True,resume_only=True,
              resume=str(Path(resume_directory)/('resume_test.pkl.gz' if (Path(resume_directory)/'resume_test.pkl.gz').exists() else 'resume_test.pkl')) if resume_directory else None,
              expected_next=str(Path(resume_directory)/('expected_next.pkl.gz' if (Path(resume_directory)/'expected_next.pkl.gz').exists() else 'expected_next.pkl')) if resume_directory else None,
              metadata=dict(test='Ideal C-touch: real Brax PPO next update after disk restore',
                            scope='Engine-independent continuation; corrected-model D5 still failed; no throughput qualification',envs=16,attempts=0)))


if __name__=='__main__':
    import sys
    check_ideal(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
