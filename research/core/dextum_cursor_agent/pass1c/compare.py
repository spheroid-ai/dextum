"""Remote paired comparison of training and post-training E3 physics paths."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
if not os.environ.get("CONTAINER_ID"):
    raise RuntimeError("Evaluate on Vast.ai only")

import jax
import jax.numpy as jnp
import numpy as np
import vendor.aero_official.vision_env  # noqa: F401
from pass1c.environment import BlindRotation, EPISODE
from experiments.evaluate_e3 import make_rollout, summarise
from pass1c.evaluate import load_policy
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass1.faithful import wrap_for_training
from pass1.qualification import require_training_pass
from pass0.review_gate import exact_mcnemar
from pass1c.criteria import assess_clustered_monitor


class MonitorGapError(RuntimeError):
    """Execution failure: the two evaluation paths no longer agree."""


class MonitorGuard:
    """Actual Brax internal evaluator against the faithful direct shared bank.

    Parameters are dynamic jit inputs; one executable serves all checkpoints.
    No optimizer updates occur during this measurement.
    """
    def __init__(self, env, make_policy, output):
        from brax.training import acting
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.key=jax.random.PRNGKey(2026)
        rng=jax.random.split(self.key,200)
        dr=jax.random.split(jax.random.PRNGKey(7),200)
        self.bank=dict(reset_keys=np.asarray(rng),dr_keys=np.asarray(dr))
        self.keys=jax.vmap(jax.random.split)(rng)[...,1]
        self.bank['direct_reset_keys']=np.asarray(self.keys)
        self.randomized,axes=env.randomize(env.mjx_model,dr)
        factory=functools.partial(make_policy,deterministic=True)
        wrapped=wrap_for_training(env,episode_length=EPISODE,full_reset=True,
            randomization_fn=functools.partial(env.randomize,rng=dr))
        self.evaluator=acting.Evaluator(wrapped,factory,200,EPISODE,1,jax.random.PRNGKey(0))
        @jax.jit
        def direct(params,randomized,keys):
            infer=factory(params)
            policy=lambda obs:infer(obs,jax.random.PRNGKey(0))[0]
            return make_rollout(env,policy,axes)(randomized,keys,jnp.zeros(2,jnp.float32))
        self.direct=direct

    def __call__(self, steps, params):
        import time
        started=time.monotonic()
        joint=self.evaluator._generate_eval_unroll(params,self.key)
        metrics=jax.tree.map(np.asarray,joint.info['eval_metrics'].episode_metrics)
        trace,sign=self.direct(params,self.randomized,self.keys)
        trace=jax.tree.map(np.asarray,trace);sign=np.asarray(sign)
        values=summarise(trace,sign)
        x=metrics['e3/success']>0;y=values['success']
        finite=all(np.isfinite(v).all() for d in (metrics,trace) for v in d.values())
        b=int(np.sum(x&~y));c=int(np.sum(~x&y))
        gap=abs(int(x.sum())-int(y.sum()))/200
        row=dict(steps=int(steps),n=200,internal_success=int(x.sum()),faithful_success=int(y.sum()),
            gap=gap,internal_only=b,faithful_only=c,exact_mcnemar_p=exact_mcnemar(b,c),
            finite=bool(finite),wall_s=time.monotonic()-started,reset_seed=2026,dr_seed=7,
            faithful_drops=int(values['dropped'].sum()))
        history=[json.loads(path.read_text()) for path in self.output.glob('*.json')]
        paired={int(steps):(x,y)}
        for old in history:
            step=int(old['steps'])
            if step >= int(steps):continue
            with np.load(self.output/f'{step}.npz') as saved:
                for key,value in self.bank.items():
                    if key not in saved or not np.array_equal(saved[key],value):
                        raise ValueError('Monitor history uses a different or unrecorded bank: '+key)
                paired[step]=(saved['internal'],saved['faithful'])
        row.update(assess_clustered_monitor(row,history,paired))
        np.savez_compressed(self.output/f'{steps}.npz',internal=x,faithful=y,
            faithful_dropped=values['dropped'],sign=sign,**self.bank)
        (self.output/f'{steps}.json').write_text(json.dumps(row,indent=2)+'\n')
        print('MONITOR',json.dumps(row),flush=True)
        if row['state']!='PASS':raise MonitorGapError('Internal monitor–faithful evaluator gap: '+str(row))
        return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--starts", type=int, default=200)
    p.add_argument("--arm", choices=("c_touch", "c_notouch", "a_task"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=Path, required=True)
    p.add_argument("--wrapper-registration", type=Path, required=True)
    p.add_argument("--direct-only", action="store_true", help="Section 11 partial-checkpoint diagnostic")
    p.add_argument("--steps", type=int)
    p.add_argument("--ideal-policy", type=Path, help="Frozen Ideal C-touch checkpoint with legacy feature prefix")
    p.add_argument("--sensor-level",choices=["Ideal","Realistic"],default="Realistic")
    a = p.parse_args()
    if a.starts != 200:
        raise ValueError("The reviewed bank contains exactly 200 starts")
    warmup = json.loads(a.warmup.read_text())
    registration = json.loads(a.wrapper_registration.read_text())
    wrapper_hash = hashlib.sha256((Path(__file__).resolve().parents[1]/'pass1/faithful.py').read_bytes()).hexdigest()
    if registration['input_sha256']['dextum_cursor_agent/pass1/faithful.py'] != wrapper_hash:
        raise RuntimeError('Wrapper differs from the accepted retry-03 warm-up release')
    if os.environ.get('XLA_FLAGS', ''):
        raise RuntimeError('The reviewed retry-03 default uses no extra XLA flags')
    require_training_pass(warmup)
    if warmup['wrapper_sha256'] != wrapper_hash:
        raise RuntimeError('Qualification and evaluator wrapper differ')
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    print('Loading fixed policy and 200-start bank', flush=True)
    os.environ["DEXTUM_SENSOR_LEVEL"]=a.sensor_level
    env = BlindRotation(a.arm)
    if a.ideal_policy:
        from brax.io import model
        from brax.training.agents.ppo import networks
        from experiments.train_e3 import make_network
        from envs.blind_rotation import frame_size as ideal_frame
        assert os.environ.get('DEXTUM_LEGACY_PREFIX')=='1'
        params=model.load_params(a.ideal_policy)
        infer=networks.make_inference_fn(make_network(a.arm))(params,deterministic=True)
        def policy(obs):
            batch=obs['state'].shape[:-1]
            legacy=obs['state'].reshape((*batch,3,env.frame))[...,:ideal_frame(a.arm)].reshape((*batch,3*ideal_frame(a.arm)))
            return infer({**obs,'state':legacy},jax.random.PRNGKey(0))[0]
        steps=a.steps;checkpoint=a.ideal_policy
    else:
        policy, steps = load_policy(a.run, a.arm, a.steps)
        checkpoints = json.loads((a.run/a.arm/'checkpoints.json').read_text())
        chosen = next(c for c in checkpoints if c['steps'] == steps)
        checkpoint = a.run/a.arm/chosen['path']
    dr = jax.random.split(jax.random.PRNGKey(7), a.starts)
    rng = jax.random.split(jax.random.PRNGKey(2026), a.starts)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    randomized, axes = env.randomize(env.mjx_model, dr)
    if a.direct_only:
        trace,sign=make_rollout(env,policy,axes)(randomized,keys,jnp.zeros(2,jnp.float32))
        trace=jax.tree.map(np.asarray,trace);sign=np.asarray(sign)
        values=summarise(trace,sign)
        finite=all(np.isfinite(x).all() for x in trace.values())
        result=dict(state='MEASURED' if finite else 'NONFINITE',diagnostic_only=True,
                    arm=a.arm,steps=steps,n=a.starts,reset_seed=2026,dr_seed=7,reset_split='second',
                    engine='MuJoCo-faithful direct MJX, explicit randomized model',
                    checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    model_xml=os.environ.get('DEXTUM_KIT_XML','original section10 ten-pad XML'),
                    success_count=int(values['success'].sum()),success=float(values['success'].mean()),
                    drops=float(values['dropped'].mean()),finite=finite,training_attempts=0,optimizer_steps=0)
        np.savez_compressed(a.output/'per_start.npz',**values,reset_keys=np.asarray(keys),dr_keys=np.asarray(dr))
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return
    wrapped = wrap_for_training(env, episode_length=EPISODE,
                               randomization_fn=functools.partial(env.randomize, rng=dr))
    s0 = jax.jit(wrapped.reset)(rng)
    bank = dict(reset_keys=np.asarray(rng), direct_reset_keys=np.asarray(keys), dr_keys=np.asarray(dr))
    np.savez(a.output/'start_bank.npz', **bank)

    @jax.jit
    def run(state):
        def body(s, _):
            s = wrapped.step(s, policy(s.obs))
            return s, {"done": s.done, "success": s.metrics["e3/success"],
                       "drop": jnp.maximum(s.metrics["e3/drop"], s.metrics["e3/warmup_drop"]),
                       "within": s.metrics["e3/within"], "error": s.metrics["e3/abs_error_deg"]}
        return jax.lax.scan(body, state, None, length=EPISODE)[1]

    print('Compiling/running training-path evaluator', flush=True)
    training = jax.tree.map(np.asarray, run(s0))
    print('Compiling/running faithful direct MJX evaluator', flush=True)
    direct, sign = make_rollout(env, policy, axes)(randomized, keys, jnp.zeros(2, jnp.float32))
    direct = jax.tree.map(np.asarray, direct)
    sign = np.asarray(sign)
    if not np.array_equal(np.asarray(s0.info['turn_sign']), sign):
        raise RuntimeError('Goal signs differ across paired starts')
    finite = all(np.isfinite(x).all() for trace in (training, direct) for x in trace.values())
    r = {}
    arrays = {}
    for name, trace in (("training", training), ("post_training", direct)):
        values = summarise({k: np.asarray(v) for k, v in trace.items()}, np.asarray(sign))
        if name == 'training':
            # EpisodeWrapper sets done at the time limit. Count physical drops
            # before the first terminal event, not timeouts, in the drop column.
            done = trace['done'] > .5
            end = np.where(done.any(0), done.argmax(0), EPISODE-1)
            alive = np.arange(EPISODE)[:, None] <= end[None]
            values['dropped'] = ((trace['drop'] > .5) & alive).any(0)
        r[name] = dict(success=float(values["success"].mean()), drops=float(values["dropped"].mean()))
        arrays.update({name+"_"+k: v for k, v in values.items()})
    disagreements = int(np.sum(arrays['training_success'] != arrays['post_training_success']))
    train_count = int(arrays['training_success'].sum())
    direct_count = int(arrays['post_training_success'].sum())
    x, y = arrays['training_success'], arrays['post_training_success']
    paired = dict(both_success=int(np.sum(x & y)), training_only=int(np.sum(x & ~y)),
                  direct_only=int(np.sum(~x & y)), both_fail=int(np.sum(~x & ~y)))
    p_value = exact_mcnemar(paired['training_only'], paired['direct_only'])
    result = dict(steps=steps, n=a.starts, reset_seed=2026, dr_seed=7,
                  state='MEASURED' if finite else 'NONFINITE', criterion_version='pass1_section8', arm=a.arm,
                  warmup_within_5mm=warmup['within_5mm'], warmup_source=str(a.warmup),
                  warmup_sha256=hashlib.sha256(a.warmup.read_bytes()).hexdigest(),
                  wrapper_sha256=wrapper_hash, xla_flags=os.environ.get('XLA_FLAGS',''),
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  matmul_precision='highest', finite=bool(finite),
                  engines=dict(training='JAX/MJX through Pass 1 carried-model barrier wrapper',
                               post_training='JAX/MJX explicit randomized model direct rollout'),
                  success_counts=dict(training=train_count, post_training=direct_count),
                  outcome_agreement=(a.starts-disagreements)/a.starts,
                  paired_counts=paired, exact_mcnemar_p=p_value,
                  evaluator_agreement=bool(finite and abs(train_count-direct_count) <= 10 and p_value > .05),
                  criteria=dict(maximum_success_gap=.05, minimum_mcnemar_p_exclusive=.05,
                                outcome_agreement='diagnostic only; withdrawn by section 7'),
                  rollout_qpos_limit_is_diagnostic_only=True,
                  initial_reset_split="Shared wrapper second split for both paths", results=r,
                  success_disagreements=disagreements,
                  absolute_success_gap=abs(r["training"]["success"]-r["post_training"]["success"]))
    (a.output/"paired_evaluators.json").write_text(json.dumps(result, indent=2)+"\n")
    result['model_sha256']=hashlib.sha256(Path(os.environ['DEXTUM_KIT_XML']).read_bytes()).hexdigest()
    result.update(criterion_version='review_section13',sensor_level=a.sensor_level,
        policy_test_valid=bool(finite and min(train_count,direct_count)>=60),pretraining_blocking=False,
        policy_test_outcome=('INCONCLUSIVE_LOW_SUCCESS' if min(train_count,direct_count)<60 else
            'PASS' if result['evaluator_agreement'] else 'REPORTED_DISAGREEMENT'))
    (a.output/"paired_evaluators.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(a.output/"paired_evaluators.npz", **arrays)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
