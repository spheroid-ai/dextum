"""Warm-up drop through four code paths, no policy (E3 ignores actions during the warm-up).

Same reset keys and DR draws as tools/trace_gap.py. Records qpos after each warm-up step through:
  brax   Brax's training wrapper stack (randomised model captured as a constant),
  arg    evaluate_e3.py's rollout (randomised model passed as a jit argument),
  const  that rollout with the randomised model captured as a constant,
  cpu    C MuJoCo, each env's randomised fields copied into an MjModel; derived constants are left
         stale, as MJX leaves them.

    python tools/physics_ref.py --output runs/<job> [--envs 16]
"""
import argparse
import copy
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
import mujoco
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs
from envs.blind_rotation import EPISODE, WARMUP, BlindRotation
from experiments.evaluate_e3 import reset_one, step_one
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize

RANDOMISED = ("geom_friction", "body_mass", "body_inertia", "body_ipos", "qpos0", "dof_frictionloss", "dof_armature",
              "dof_damping", "actuator_gainprm", "actuator_biasprm")
PALM_Z = -0.025


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--arm", default="c_touch")
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    a.output.mkdir(parents=True, exist_ok=True)
    n = a.envs
    env = BlindRotation(a.arm)
    zeros = jnp.zeros((n, env.action_size))
    dr_keys = jax.random.split(jax.random.PRNGKey(7), n)
    rng = jax.random.split(jax.random.PRNGKey(2026), n)
    keys = jax.vmap(jax.random.split)(rng)[..., 1]    # the auto-reset wrapper's split
    model_v, axes = domain_randomize(env.mjx_model, dr_keys)

    brax_env = wrapper.wrap_for_brax_training(env, episode_length=EPISODE, action_repeat=1, full_reset=True,
                                              randomization_fn=functools.partial(domain_randomize, rng=dr_keys))

    def unroll(s0, step):
        def body(s, _):
            s = step(s)
            return s, s.data.qpos
        return jax.lax.scan(body, s0, None, length=WARMUP)[1]

    reset_v = jax.vmap(functools.partial(reset_one, env), in_axes=[axes, 0, None])
    step_v = jax.vmap(functools.partial(step_one, env), in_axes=[axes, 0, 0])

    def direct(model_v, keys):
        s0 = reset_v(model_v, keys, jnp.zeros(2, jnp.float32))
        return unroll(s0, lambda s: step_v(model_v, s, zeros))

    cz = int(np.asarray(env._cube_qids)[2])
    runs = dict(brax=lambda: jax.jit(lambda r: unroll(brax_env.reset(r), lambda s: brax_env.step(s, zeros)))(rng),
                arg=lambda: jax.jit(direct)(model_v, keys),
                const=lambda: jax.jit(lambda k: direct(model_v, k))(keys))
    paths = {}
    for name, run in runs.items():
        paths[name] = np.asarray(run(), np.float64)
        print(name, "z mm at warm-up end:", np.round(paths[name][-1, :, cz] * 1000).astype(int).tolist(), flush=True)

    s0 = jax.jit(lambda m, k: reset_v(m, k, jnp.zeros(2, jnp.float32)))(model_v, keys)
    mj = env.mj_model
    home = np.asarray(cs.HOME, np.float64)
    cpu = np.zeros_like(paths["arg"])
    for i in range(n):
        m = copy.deepcopy(mj)
        for f in RANDOMISED:
            getattr(m, f)[:] = np.asarray(getattr(model_v, f)[i])
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = (np.asarray(x[i]) for x in (s0.data.qpos, s0.data.qvel, s0.data.ctrl))
        d.mocap_pos[:], d.mocap_quat[:] = np.asarray(s0.data.mocap_pos[i]), np.asarray(s0.data.mocap_quat[i])
        for t in range(WARMUP):
            d.ctrl[:] = home
            for _ in range(env.n_substeps):
                mujoco.mj_step(m, d)
            cpu[t, i] = d.qpos
    paths["cpu"] = cpu
    print("cpu z mm at warm-up end:", np.round(cpu[-1, :, cz] * 1000).astype(int).tolist(), flush=True)

    opt = mj.opt
    out = dict(arm=a.arm, envs=n, n_substeps=int(env.n_substeps), timestep=float(opt.timestep),
               solver=int(opt.solver), iterations=int(opt.iterations), ls_iterations=int(opt.ls_iterations),
               tolerance=float(opt.tolerance), cone=int(opt.cone), impratio=float(opt.impratio),
               noslip_iterations=int(opt.noslip_iterations),
               numeric={mj.numeric(i).name: np.asarray(mj.numeric(i).data).tolist() for i in range(mj.nnumeric)})
    out["z_mm_at_warmup_end"] = {k: np.round(v[-1, :, cz] * 1000).astype(int).tolist() for k, v in paths.items()}
    out["in_palm_at_warmup_end"] = {k: int((v[-1, :, cz] < PALM_Z).sum()) for k, v in paths.items()}
    out["z_mm_trajectory_env"] = {str(e): {k: np.round(v[:, e, cz] * 1000).astype(int).tolist() for k, v in paths.items()}
                                  for e in (2, 6, 8, 11) if e < n}
    names = list(paths)
    out["max_abs_qpos_diff"] = {f"{x}-{y}": {str(t): float(np.abs(paths[x][t] - paths[y][t]).max()) for t in (0, 1, 2, 5, WARMUP - 1)}
                                for i, x in enumerate(names) for y in names[i + 1:]}
    out["envs_matching_cpu_at_warmup_end_within_5mm"] = {k: int((np.abs(v[-1, :, cz] - cpu[-1, :, cz]) < 0.005).sum())
                                                       for k, v in paths.items() if k != "cpu"}
    print(json.dumps(out, indent=1), flush=True)
    (a.output / "physics_ref.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(a.output / "physics_ref.npz", **paths)


if __name__ == "__main__":
    main()
