"""Remote D5: 16 frozen warm-up starts, stateful training vs C MuJoCo."""
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path

if not os.environ.get("CONTAINER_ID"):
    raise RuntimeError("Run physics qualification on Vast.ai, never on the laptop")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs
from envs.blind_rotation import BlindRotation, EPISODE, WARMUP
from experiments.evaluate_e3 import reset_one, step_one
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from pass0.faithful import MODEL_KEY, wrap_for_training
from tools.physics_ref import RANDOMISED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    assert all(d.platform == "gpu" for d in jax.devices()), "GPU required"
    n = 16
    env = BlindRotation("c_touch")
    dr_keys = jax.random.split(jax.random.PRNGKey(7), n)
    rng = jax.random.split(jax.random.PRNGKey(2026), n)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]
    randomize = functools.partial(domain_randomize, rng=dr_keys)
    m, axes = randomize(env.mjx_model)
    fixed = wrap_for_training(env, episode_length=EPISODE, randomization_fn=randomize)
    zeros = jnp.zeros((n, env.action_size))

    @jax.jit
    def training_path(state):
        def body(s, _):
            s = fixed.step(s, zeros)
            return s, s.data.qpos
        return jax.lax.scan(body, state, None, length=WARMUP)

    # Reset is materialized before rollout exactly as PPO does. The randomization
    # arrays enter the training jit as state arguments, not closure constants.
    s0 = jax.jit(fixed.reset)(rng)
    final, train = training_path(s0)
    reset_v = jax.vmap(functools.partial(reset_one, env), in_axes=[axes, 0, None])
    step_v = jax.vmap(functools.partial(step_one, env), in_axes=[axes, 0, 0])

    @jax.jit
    def direct(model, keys):
        state = reset_v(model, keys, jnp.zeros(2, jnp.float32))
        def body(s, _):
            s = step_v(model, s, zeros)
            return s, s.data.qpos
        return jax.lax.scan(body, state, None, length=WARMUP)[1]

    train, reference = np.asarray(train), np.asarray(direct(m, keys))
    cpu = np.zeros_like(train, dtype=np.float64)
    for i in range(n):
        model = copy.deepcopy(env.mj_model)
        for field in RANDOMISED:
            getattr(model, field)[:] = np.asarray(getattr(m, field)[i])
        data = mujoco.MjData(model)
        for field in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(data, field)[:] = np.asarray(getattr(s0.data, field)[i])
        for t in range(WARMUP):
            data.ctrl[:] = np.asarray(cs.HOME)
            for _ in range(env.n_substeps):
                mujoco.mj_step(model, data)
            cpu[t, i] = data.qpos
    z = int(np.asarray(env._cube_qids)[2])
    height_error = np.abs(train[-1, :, z] - cpu[-1, :, z])
    agreement = int((height_error < .005).sum())
    direct_error = float(np.max(np.abs(train - reference)))
    # A full auto-reset must preserve supplied dynamic physics, not restore the
    # captured nominal draws. Perturb a parameter and force every env to reset.
    prior_model = final.info[MODEL_KEY]
    changed = prior_model.tree_replace({"body_mass": prior_model.body_mass.at[:, env.cube_body].multiply(1.1)})
    reset_input = final.replace(info={**final.info, MODEL_KEY: changed,
                                     "steps": jnp.full((n,), EPISODE - 1)})
    after_reset = jax.jit(fixed.step)(reset_input, zeros)
    preserved = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(
        jax.tree.leaves(after_reset.info[MODEL_KEY]), jax.tree.leaves(changed)))
    reset_counted = bool(np.all(np.asarray(after_reset.done) > .5))
    finite = bool(np.isfinite(train).all() and np.isfinite(cpu).all())
    # User review 2026-10-10: trajectory identity is diagnostic only. This
    # component supplies warm-up evidence; fixed-policy agreement completes D5.
    passed = agreement >= 14 and preserved and reset_counted and finite
    result = dict(state="WARMUP_PASS" if passed else "WARMUP_FAIL", n=n,
                  criterion_version="review_20261010_warmup",
                  fixed_policy_agreement_required=True,
                  rollout_qpos_limit_is_diagnostic_only=True,
                  requirement="At least 14/16 end-of-warmup cube heights within 5 mm of C MuJoCo",
                  within_5mm=agreement, height_error_mm=(height_error * 1000).tolist(),
                  training_vs_direct_max_qpos_error=direct_error,
                  model_storage="Shared nominal arrays, batched randomized arrays, all dynamic",
                  randomized_fields=list(RANDOMISED),
                  model_state_bytes_total=sum(v.size * v.dtype.itemsize for v in jax.tree.leaves(changed)),
                  full_reset_preserves_physics=preserved,
                  full_reset_exercised=reset_counted, finite=finite,
                  in_palm={name: int((path[-1, :, z] < -.025).sum()) for name, path in
                           (("training", train), ("direct", reference), ("cpu", cpu))},
                  mujoco=mujoco.__version__, jax=jax.__version__,
                  xla_flags=os.environ.get('XLA_FLAGS',''),
                  wrapper_sha256=hashlib.sha256((Path(__file__).with_name('faithful.py')).read_bytes()).hexdigest(),
                  scope="Registered warm-up D5 reference, not general long-horizon equivalence")
    (a.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(a.output / "trajectories.npz", training=train, direct=reference, cpu=cpu)
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
