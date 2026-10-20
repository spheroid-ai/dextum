"""Compile actual Brax functions at full registered dimensions; never optimize."""
import os,sys,json,time,resource,functools,inspect,hashlib
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import jax
from brax.training import acting
from brax.training.agents.ppo import train as ppo
from brax.envs.wrappers import training
import vendor.aero_official.vision_env
from mujoco_playground.config import manipulation_params
from pass1 import faithful
from pass1c.environment import BlindRotation,EPISODE
from pass1c.train import network_factory

class Complete(Exception):pass

def main():
    phase=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
    assert phase in ('training','evaluator','materialized_evaluator')
    assert not os.environ.get('XLA_FLAGS','')
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.devices()[0].platform=='gpu'
    start=time.monotonic();result=dict(state='STARTING',phase=phase,training_attempts=0,optimizer_steps=0,
        envs=8192,eval_envs=128,unroll_length=40,xla_flags='',function_compiled_not_executed=True)
    (out/'registration.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,obj in [('ppo',ppo),('acting',acting)]:
        source=inspect.getsource(obj);(out/(name+'_source.txt')).write_text(source)
        result[name+'_sha256']=hashlib.sha256(source.encode()).hexdigest()
    old_pmap=jax.pmap;old_eval=acting.Evaluator.run_evaluation;old_init=acting.Evaluator.__init__
    context={}
    def finish(fn,*args):
        before=time.monotonic();print('LOWER_AND_COMPILE',phase,flush=True)
        compiled=fn.lower(*args).compile()
        memory=compiled.memory_analysis()
        result.update(state='PASS',compile_s=time.monotonic()-before,total_s=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      executable_bytes=getattr(memory,'generated_code_size_in_bytes',None))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);raise Complete()
    def pmap(fn,*args,**kwargs):
        if getattr(fn,'__name__','')=='training_epoch':
            def stop_before_optimization(*a):
                # Brax UInt64.__post_init__ cannot reconstruct JAX ArgInfo
                # during lower(). Flatten only the serialization boundary;
                # the installed epoch, optimizer and unroll remain intact.
                leaves,structure=jax.tree.flatten(a)
                def flat_epoch(*values):
                    return jax.tree.leaves(fn(*jax.tree.unflatten(structure,values)))
                result['compile_input_adapter']='Flat pytree boundary for Brax UInt64; original epoch body unchanged'
                finish(old_pmap(flat_epoch,*args,**kwargs),*leaves)
            return stop_before_optimization
        return old_pmap(fn,*args,**kwargs)
    def init(self,eval_env,eval_policy_fn,*args,**kwargs):
        context.update(env=eval_env,factory=eval_policy_fn)
        return old_init(self,eval_env,eval_policy_fn,*args,**kwargs)
    def evaluation(self,params,training_metrics,*args,**kwargs):
        if phase=='evaluator':finish(self._generate_eval_unroll,params,jax.random.PRNGKey(0))
        env=training.EvalWrapper(context['env'])
        state=jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0),128))
        jax.block_until_ready(state)
        @jax.jit
        def rollout(state,params,key):
            return acting.generate_unroll(env,state,context['factory'](params),key,EPISODE)[0]
        finish(rollout,state,params,jax.random.PRNGKey(0))
    try:
        jax.pmap=pmap;acting.Evaluator.__init__=init;acting.Evaluator.run_evaluation=evaluation
        cfg=manipulation_params.brax_ppo_config('AeroCubeRotateZAxis').to_dict();cfg.pop('network_factory')
        cfg.update(episode_length=EPISODE,num_envs=8192,num_timesteps=100000000,seed=1,
            num_eval_envs=128,num_evals=31,deterministic_eval=True,
            log_training_metrics=True,training_metrics_steps=327680,run_evals=phase!='training')
        env=BlindRotation('c_touch')
        ppo.train(environment=env,network_factory=network_factory('c_touch'),
            wrap_env_fn=functools.partial(faithful.wrap_for_training,full_reset=True),randomization_fn=env.randomize,**cfg)
        raise RuntimeError('Dry-run interception did not trigger')
    except Complete:pass
    finally:
        jax.pmap=old_pmap;acting.Evaluator.run_evaluation=old_eval;acting.Evaluator.__init__=old_init

if __name__=='__main__':main()
