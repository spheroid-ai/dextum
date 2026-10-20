"""E3 evaluation: the training distribution and the T1 held-out set (spec §9), paired starts.

Each condition sits on top of the training DR draws and changes one factor:
  friction 0.35 / 1.2         sliding friction of every hand geom and the cube
  gain x0.6 / x1.4            actuator gain (kp in gain and bias)
  mass 50 g / 100 g           cube mass, inertia scaled from the nominal model
  offset x/y +/-20 mm         cube start position
Every arm sees the same starts, goals and physics draws. T1 pass: a touch arm's
mean success over the held-out conditions >= its no-touch arm's + 10 points.
Paired comparisons on the training distribution use the exact McNemar test.

Matmuls run at "highest" precision, as in training: the GPU default (TF32)
changes the simulated physics.

    python experiments/evaluate_e3.py runs/e3p1-s1 --starts 200 [--output DIR]
"""
import argparse
import contextlib
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks

import vendor.aero_official.vision_env  # noqa: F401
from envs.blind_rotation import ARMS, EPISODE, WARMUP, BlindRotation
from experiments.train_e3 import make_network
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize

CONDITIONS = {"train": {}, "friction 0.35": dict(friction=0.35), "friction 1.2": dict(friction=1.2),
              "gain x0.6": dict(gain=0.6), "gain x1.4": dict(gain=1.4),
              "mass 50 g": dict(mass=0.05), "mass 100 g": dict(mass=0.10),
              "offset x+20 mm": dict(offset=(0.02, 0.0)), "offset x-20 mm": dict(offset=(-0.02, 0.0)),
              "offset y+20 mm": dict(offset=(0.0, 0.02)), "offset y-20 mm": dict(offset=(0.0, -0.02))}
HELD_OUT = [k for k in CONDITIONS if k != "train"]
PAIRS = (("a_touch", "a_notouch"), ("c_touch", "c_notouch"))
COMPARISONS = (("c_touch", "a_touch"), ("c_notouch", "a_notouch"), ("c_touch", "a_task"), ("a_notouch", "a_task"),
               ("c_touch", "c_notouch"), ("a_touch", "a_notouch"))


def mcnemar(x, y):
    """Exact two-sided McNemar test on paired successes: (only x, only y, p)."""
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def load_policy(run, arm, steps=None):
    checkpoints = [c for c in json.loads((run / arm / "checkpoints.json").read_text()) if c.get("kept", True)]
    chosen = checkpoints[-1] if steps is None else next(c for c in checkpoints if c["steps"] == steps)
    params = model.load_params(run / arm / chosen["path"])
    infer = networks.make_inference_fn(make_network(arm))(params, deterministic=True)
    return (lambda obs: infer(obs, jax.random.PRNGKey(0))[0]), chosen["steps"]


def physics(env, n, dr_seed, condition):
    m, axes = domain_randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(dr_seed), n))
    mj = env.mj_model
    if "friction" in condition:
        ids = np.flatnonzero(mj.geom_bodyid != 0)
        m = m.tree_replace({"geom_friction": m.geom_friction.at[:, ids, 0].set(condition["friction"])})
    if "gain" in condition:
        f = condition["gain"]
        m = m.tree_replace({"actuator_gainprm": m.actuator_gainprm.at[:, :, 0].multiply(f),
                            "actuator_biasprm": m.actuator_biasprm.at[:, :, 1].multiply(f)})
    if "mass" in condition:
        cube = env.cube_body
        inertia = jnp.asarray(mj.body_inertia[cube]) * condition["mass"] / mj.body_mass[cube]
        m = m.tree_replace({"body_mass": m.body_mass.at[:, cube].set(condition["mass"]),
                            "body_inertia": m.body_inertia.at[:, cube].set(inertia)})
    return m, axes


@contextlib.contextmanager
def swapped(env, mjx_model):
    old = env._mjx_model
    env._mjx_model = mjx_model
    try:
        yield env
    finally:
        env._mjx_model = old


def reset_one(env, m, key, offset):
    """Reset one episode under model m, with the cube start moved by offset (x, y) in metres."""
    qx = int(np.asarray(env._cube_qids)[0])
    with swapped(env, m):
        s = env.reset(key)
    return s.replace(data=s.data.replace(qpos=s.data.qpos.at[qx:qx + 2].add(offset)))


def step_one(env, m, s, a):
    with swapped(env, m):
        return env.step(s, a)


def make_rollout(env, policy, axes):
    reset_v = jax.vmap(functools.partial(reset_one, env), in_axes=[axes, 0, None])
    step_v = jax.vmap(functools.partial(step_one, env), in_axes=[axes, 0, 0])

    @jax.jit
    def run(model_v, keys, offset):
        state = reset_v(model_v, keys, offset)

        def body(state, _):
            nxt = step_v(model_v, state, policy(state.obs))
            m = nxt.metrics
            return nxt, dict(done=nxt.done, success=m["e3/success"], within=m["e3/within"],
                             error=m["e3/abs_error_deg"])

        _, trace = jax.lax.scan(body, state, None, length=EPISODE)
        return trace, state.info["turn_sign"]

    return run


def summarise(trace, sign):
    done = trace["done"] > 0.5
    T = done.shape[0]
    end = np.where(done.any(0), done.argmax(0), T - 1)
    alive = np.arange(T)[:, None] <= end[None]
    onset = (trace["success"] > 0) & alive
    success = onset.any(0)
    when = np.where(success, onset.argmax(0) + 1 - WARMUP, -1)
    return dict(success=success, dropped=done.any(0), success_step=when, sign=sign,
                within_steps=((trace["within"] > 0) & alive).sum(0), final_error=trace["error"][end, np.arange(len(end))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--arms", nargs="+", default=sorted(ARMS))
    p.add_argument("--starts", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dr-seed", type=int, default=7)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--precision", default="highest", help='"default" leaves the platform default (TF32 on GPU)')
    p.add_argument("--output", type=Path, default=None, help="folder for evaluation_e3.json/npz (default: the run)")
    a = p.parse_args()
    if a.precision != "default":
        jax.config.update("jax_default_matmul_precision", a.precision)
    out = a.output or a.run
    out.mkdir(parents=True, exist_ok=True)
    keys = jax.random.split(jax.random.PRNGKey(a.seed), a.starts)
    results, arrays = {"matmul_precision": jax.config.jax_default_matmul_precision or "default",
                       "devices": [str(d) for d in jax.devices()]}, {}
    for arm in a.arms:
        if not (a.run / arm / "checkpoints.json").exists():
            print(f"{arm}: no checkpoints, skipped", flush=True)
            continue
        env = BlindRotation(arm)
        policy, steps = load_policy(a.run, arm)
        run = None
        results[arm] = dict(steps=steps, conditions={})
        for name in a.conditions:
            t0 = time.time()
            condition = CONDITIONS[name]
            model_v, axes = physics(env, a.starts, a.dr_seed, condition)
            run = run or make_rollout(env, policy, axes)
            trace, sign = run(model_v, keys, jnp.asarray(condition.get("offset", (0.0, 0.0)), jnp.float32))
            r = summarise({k: np.asarray(v) for k, v in trace.items()}, np.asarray(sign))
            ok = r["success"]
            results[arm]["conditions"][name] = dict(
                n=int(len(ok)), success=float(ok.mean()), drops=float(r["dropped"].mean()),
                success_plus=float(ok[r["sign"] > 0].mean()), success_minus=float(ok[r["sign"] < 0].mean()),
                median_time_s=float(np.median(r["success_step"][ok]) * env.dt) if ok.any() else None,
                median_final_error_deg=float(np.median(r["final_error"])))
            for k, v in r.items():
                arrays[f"{arm}/{name}/{k}"] = v
            c = results[arm]["conditions"][name]
            print(f"{arm:9s} {name:15s} success {c['success']:.3f} (+30: {c['success_plus']:.2f}, -30: {c['success_minus']:.2f}) "
                  f"drops {c['drops']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        held = [results[arm]["conditions"][k]["success"] for k in HELD_OUT if k in results[arm]["conditions"]]
        results[arm]["held_out_mean"] = float(np.mean(held)) if held else None
    t1 = {}
    for touch, notouch in PAIRS:
        if touch in results and notouch in results and results[touch]["held_out_mean"] is not None:
            gap = results[touch]["held_out_mean"] - results[notouch]["held_out_mean"]
            t1[touch] = dict(versus=notouch, gap=gap, passed=bool(gap >= 0.10))
    results["t1"] = t1
    paired = {}
    for x, y in COMPARISONS:
        kx, ky = f"{x}/train/success", f"{y}/train/success"
        if kx in arrays and ky in arrays:
            only_x, only_y, p = mcnemar(arrays[kx], arrays[ky])
            paired[f"{x} vs {y}"] = dict(difference=float(arrays[kx].mean() - arrays[ky].mean()),
                                         only_first=only_x, only_second=only_y, p=p)
    results["paired_train"] = paired
    (out / "evaluation_e3.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(out / "evaluation_e3.npz", **arrays)
    for arm in a.arms:
        if arm in results:
            print(f"{arm:9s} held-out mean success {results[arm]['held_out_mean']}")
    print("T1:", json.dumps(t1))
    for k, v in paired.items():
        print(f"{k:24s} difference {v['difference']:+.3f}  discordant {v['only_first']} / {v['only_second']}  p {v['p']:.3g}")


if __name__ == "__main__":
    main()
