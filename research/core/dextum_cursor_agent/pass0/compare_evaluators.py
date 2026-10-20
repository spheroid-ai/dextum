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
from envs.blind_rotation import BlindRotation, EPISODE
from experiments.evaluate_e3 import load_policy, make_rollout, summarise
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass0.faithful import wrap_for_training
from pass0.review_gate import exact_mcnemar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--starts", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=Path, required=True)
    p.add_argument("--wrapper-registration", type=Path, required=True)
    a = p.parse_args()
    if a.starts != 200:
        raise ValueError("The reviewed bank contains exactly 200 starts")
    warmup = json.loads(a.warmup.read_text())
    registration = json.loads(a.wrapper_registration.read_text())
    wrapper_hash = hashlib.sha256(Path(__file__).with_name('faithful.py').read_bytes()).hexdigest()
    if registration['input_sha256']['dextum_cursor_agent/pass0/faithful.py'] != wrapper_hash:
        raise RuntimeError('Wrapper differs from the accepted retry-03 warm-up release')
    if os.environ.get('XLA_FLAGS', ''):
        raise RuntimeError('The reviewed retry-03 default uses no extra XLA flags')
    if warmup['within_5mm'] < 14 or not warmup['finite'] or not warmup['full_reset_preserves_physics']:
        raise RuntimeError('Missing accepted warm-up evidence')
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    print('Loading fixed policy and 200-start bank', flush=True)
    env = BlindRotation("c_touch")
    policy, steps = load_policy(a.run, "c_touch")
    checkpoints = json.loads((a.run/'c_touch/checkpoints.json').read_text())
    chosen = next(c for c in checkpoints if c['steps'] == steps)
    checkpoint = a.run/'c_touch'/chosen['path']
    dr = jax.random.split(jax.random.PRNGKey(7), a.starts)
    rng = jax.random.split(jax.random.PRNGKey(2026), a.starts)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    randomized, axes = domain_randomize(env.mjx_model, dr)
    wrapped = wrap_for_training(env, episode_length=EPISODE,
                               randomization_fn=functools.partial(domain_randomize, rng=dr))
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
                  state='MEASURED' if finite else 'NONFINITE', criterion_version='post_training_section7',
                  warmup_within_5mm=warmup['within_5mm'], warmup_source=str(a.warmup),
                  warmup_sha256=hashlib.sha256(a.warmup.read_bytes()).hexdigest(),
                  wrapper_sha256=wrapper_hash, xla_flags=os.environ.get('XLA_FLAGS',''),
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  matmul_precision='highest', finite=bool(finite),
                  engines=dict(training='JAX/MJX through retry-03 auto-reset wrapper',
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
    np.savez_compressed(a.output/"paired_evaluators.npz", **arrays)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
