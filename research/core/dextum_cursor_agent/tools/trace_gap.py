"""Step-by-step comparison of Brax's evaluation path and evaluate_e3.py's rollout on one checkpoint.

Same starts and DR draws through both paths (the Brax auto-reset wrapper's key split is
replicated), recording qpos, actor inputs, actions and E3 metrics per step; reports the first
step where the paths diverge. Also: Brax's Evaluator with full_reset off vs on (128 episodes).

    python tools/trace_gap.py runs/e3p1-s1 --output runs/<job> [--arm c_touch]
"""
import argparse
import functools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers.training import EvalWrapper
from brax.io import model
from brax.training import acting
from brax.training.agents.ppo import networks

import vendor.aero_official.vision_env  # noqa: F401
from envs.blind_rotation import EPISODE, BlindRotation
from experiments.evaluate_e3 import reset_one, step_one
from experiments.train_e3 import make_network
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize

N_TRACE = 16
N_EVAL = 128


def record(s, a):
    return dict(qpos=s.data.qpos, obs=s.obs["state"], act=a, success=s.metrics["e3/success"],
                within=s.metrics["e3/within"], tick=s.info["tick"], sign=s.info["turn_sign"], done=s.done)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", default="c_touch")
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    a.output.mkdir(parents=True, exist_ok=True)
    chosen = [c for c in json.loads((a.run / a.arm / "checkpoints.json").read_text()) if c.get("kept", True)][-1]
    params = model.load_params(a.run / a.arm / chosen["path"])
    make_policy = networks.make_inference_fn(make_network(a.arm))
    infer = make_policy(params, deterministic=True)
    policy = lambda obs: infer(obs, jax.random.PRNGKey(0))[0]
    env = BlindRotation(a.arm)
    out = dict(arm=a.arm, steps=chosen["steps"])

    dr_keys = jax.random.split(jax.random.PRNGKey(7), N_TRACE)
    rng = jax.random.split(jax.random.PRNGKey(2026), N_TRACE)
    brax_env = EvalWrapper(wrapper.wrap_for_brax_training(
        env, episode_length=EPISODE, action_repeat=1, full_reset=True,
        randomization_fn=functools.partial(domain_randomize, rng=dr_keys)))

    @jax.jit
    def brax_path(rng):
        s0 = brax_env.reset(rng)

        def body(s, _):
            act = policy(s.obs)
            n = brax_env.step(s, act)
            return n, record(n, act)
        return record(s0, jnp.zeros((N_TRACE, env.action_size))), jax.lax.scan(body, s0, None, length=EPISODE)[1]

    model_v, axes = domain_randomize(env.mjx_model, dr_keys)
    reset_v = jax.vmap(functools.partial(reset_one, env), in_axes=[axes, 0, None])
    step_v = jax.vmap(functools.partial(step_one, env), in_axes=[axes, 0, 0])

    @jax.jit
    def our_path(model_v, keys):
        s0 = reset_v(model_v, keys, jnp.zeros(2, jnp.float32))

        def body(s, _):
            act = policy(s.obs)
            n = step_v(model_v, s, act)
            return n, record(n, act)
        return record(s0, jnp.zeros((N_TRACE, env.action_size))), jax.lax.scan(body, s0, None, length=EPISODE)[1]

    # The auto-reset wrapper resets with the second half of a per-env split of the given keys.
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    b0, b = jax.tree.map(np.asarray, brax_path(rng))
    m0, m = jax.tree.map(np.asarray, our_path(model_v, keys))
    out["initial_max_abs_diff"] = {k: float(np.max(np.abs(b0[k].astype(np.float64) - m0[k]))) for k in ("qpos", "obs", "tick", "sign")}
    per_step = {k: np.max(np.abs(b[k].astype(np.float64) - m[k]), axis=tuple(range(1, b[k].ndim))) for k in ("qpos", "obs", "act")}
    out["first_step_over_1e-4"] = {k: (int(np.argmax(v > 1e-4)) if (v > 1e-4).any() else None) for k, v in per_step.items()}
    out["max_abs_diff_at_steps"] = {k: {str(t): float(v[t]) for t in (0, 1, 2, 5, 10, 19, 20, 21, 30, 50, 100, 299)} for k, v in per_step.items()}
    if out["first_step_over_1e-4"]["obs"] is not None:
        t = out["first_step_over_1e-4"]["obs"]
        d = np.abs(b["obs"][t].astype(np.float64) - m["obs"][t])
        out["obs_columns_differing_first"] = sorted(set(np.flatnonzero(d.max(0) > 1e-4).tolist()))[:40]
    for name, tr in (("brax", b), ("ours", m)):
        alive = np.cumsum(tr["done"] > 0.5, axis=0) == 0
        out[f"{name}_success"] = (tr["success"] * alive).sum(0).astype(int).tolist()
        out[f"{name}_within"] = (tr["within"] * alive).sum(0).astype(int).tolist()
        out[f"{name}_tick_at"] = {str(t): tr["tick"][t].tolist()[:4] for t in (0, 19, 20, 299)}
    print(json.dumps(out, indent=1), flush=True)
    np.savez_compressed(a.output / "trace_gap.npz", **{f"brax_{k}": v for k, v in b.items()}, **{f"ours_{k}": v for k, v in m.items()})

    eval_dr = jax.random.split(jax.random.PRNGKey(7), N_EVAL)
    for full_reset in (False, True):
        ev = acting.Evaluator(wrapper.wrap_for_brax_training(
            env, episode_length=EPISODE, action_repeat=1, full_reset=full_reset,
            randomization_fn=functools.partial(domain_randomize, rng=eval_dr)),
            functools.partial(make_policy, deterministic=True), num_eval_envs=N_EVAL, episode_length=EPISODE,
            action_repeat=1, key=jax.random.PRNGKey(5))
        s = [float(np.mean(np.asarray(ev.run_evaluation(params, {}, aggregate_episodes=False)["eval/episode_e3/success"]) > 0))
             for _ in range(2)]
        out[f"brax_evaluator_full_reset_{full_reset}"] = s
        print(f"Brax Evaluator full_reset={full_reset}: success {s}", flush=True)
    (a.output / "trace_gap.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
