"""Why E2 episodes fail: final checkpoints under the training DR, one episode per start.

Every arm sees the same starts and the same physics draws, so outcomes can be
paired. Classes, in order: success; drop; no grasp (never N3); grasp without
a 5 mm lift; lifted grasp whose hold never lasted 20 steps (with what broke it).

    python tools/diagnose_e2.py runs/e2p1-s1b --arms c_touch c_notouch --n 256
"""
import argparse
import functools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs
from envs import grasp_edge as ge
from experiments.train_e2 import make_network
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

CLASSES = ("success", "drop", "no grasp", "grasp, no lift", "lift, hold < 20")


def load_policy(run, arm, steps=None):
    checkpoints = [c for c in json.loads((run / arm / "checkpoints.json").read_text()) if c.get("kept", True)]
    chosen = checkpoints[-1] if steps is None else next(c for c in checkpoints if c["steps"] == steps)
    params = model.load_params(run / arm / chosen["path"])
    infer = networks.make_inference_fn(make_network(arm))(params, deterministic=True)
    return (lambda obs: infer(obs, jax.random.PRNGKey(0))[0]), chosen["steps"]


def rollout(arm, policy, n, seed, dr_seed):
    env = ge.GraspEdge(arm)
    wrapped = BraxDomainRandomizationVmapWrapper(
        env, functools.partial(domain_randomize, rng=jax.random.split(jax.random.PRNGKey(dr_seed), n)))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    read = jax.vmap(env.touch_model.read, in_axes=(wrapped._in_axes, 0))
    linvel, angvel = jax.vmap(env.get_cube_linvel), jax.vmap(env.get_cube_angvel)
    membership = jnp.asarray(cs.MEMBERSHIP)
    qids = np.asarray(env._cube_qids)

    @jax.jit
    def run():
        state = wrapped.reset(keys)

        def body(state, _):
            action = policy(state.obs)
            nxt = wrapped.step(state, action)
            m, info = nxt.metrics, nxt.info
            r = read(wrapped._mjx_model_v, nxt.data)
            rec = dict(done=nxt.done, success=m["e2/success"], n3=m["e2/n3"], held=m["e2/held"],
                       lifted=m["e2/lifted"], lift_mm=m["e2/lift_mm"], grip_slips=m["e2/grip_slips"],
                       hold=info["hold"], speed=jnp.linalg.norm(linvel(nxt.data), axis=-1),
                       spin=jnp.linalg.norm(angvel(nxt.data), axis=-1),
                       grip_slip_any=jnp.any(info["grip_slip"], axis=-1),
                       digit_force=r.f_n @ membership.T, n_firm=jnp.sum(r.f_n >= ge.FIRM_N, axis=-1),
                       quality=m["e2/quality"], mode=info["g"].mode, f_ref=info["g"].f_ref,
                       sensed=info["sense"].force, contact=info["sense"].contact, ctrl=nxt.data.ctrl,
                       cube=nxt.data.qpos[:, qids])
            return nxt, rec

        _, trace = jax.lax.scan(body, state, None, length=ge.EPISODE)
        return trace

    trace = {k: np.asarray(v) for k, v in run().items()}       # (T, N, ...)
    mv = wrapped._mjx_model_v
    tips = [env.mj_model.geom(g).id for g in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")]
    physics = dict(tip_friction=np.asarray(mv.geom_friction)[:, tips[0], 0],
                   kp_scale=np.asarray(mv.actuator_gainprm)[:, cs.PRESS, 0].mean(1) / 10000.0)
    return trace, physics


def max_run(x):
    best = run = 0
    for v in x:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify(trace):
    T, N = trace["done"].shape
    done = trace["done"] > 0.5
    end = np.where(done.any(0), done.argmax(0), T - 1)
    alive = np.arange(T)[:, None] <= end[None]
    active = (np.arange(T) >= ge.WARMUP)[:, None] & alive      # trace row i is the step taken at tick i
    ok = lambda k: (trace[k] > 0) & active
    success = ok("success").any(0)
    dropped = done.any(0)                                       # no time limit inside this rollout: done means drop
    grasped = ok("n3").any(0)
    lifted_grasp = (ok("n3") & ok("lifted")).any(0)
    label = np.select([success, dropped, ~grasped, ~lifted_grasp], [0, 1, 2, 3], 4)
    return label, active, end


def report(arm, steps, trace, physics):
    label, active, end = classify(trace)
    N = label.size
    print(f"\n== {arm} (checkpoint {steps}), {N} episodes")
    for c, name in enumerate(CLASSES):
        print(f"  {name:18s} {np.mean(label == c):5.2f}")
    t = np.arange(trace["done"].shape[0])[:, None] - ge.WARMUP + 1
    first_n3 = np.where((trace["n3"] > 0) & active, t, 10_000).min(0)
    success_at = np.where((trace["success"] > 0) & active, t, 10_000).min(0)
    print(f"  first N3 step (median, grasped episodes): {np.median(first_n3[first_n3 < 10_000]):.0f}; "
          f"success step median {np.median(success_at[label == 0]) if (label == 0).any() else float('nan'):.0f}, "
          f"90th pct {np.percentile(success_at[label == 0], 90) if (label == 0).any() else float('nan'):.0f} of {ge.ACTIVE}")

    # What blocked the hold while the grasp was lifted (class 4 and the time before success in class 0).
    lifted = (trace["n3"] > 0) & (trace["lifted"] > 0) & active
    blocked = lifted & (trace["held"] == 0)
    moving = (trace["speed"] >= ge.STILL_SPEED) | (trace["spin"] >= ge.STILL_ANGULAR)
    for name, sel in (("lift, hold < 20", label == 4), ("success", label == 0)):
        if not sel.any():
            continue
        b = blocked[:, sel]
        total = max(b.sum(), 1)
        runs = [max_run(trace["held"][:, i] > 0) for i in np.flatnonzero(sel)]
        print(f"  {name}: lifted-grasp steps {lifted[:, sel].sum(0).mean():.0f}/ep, held {(trace['held'][:, sel] > 0).sum(0).mean():.0f}/ep, "
              f"longest hold median {np.median(runs):.0f}; blocked by motion {np.sum(b & moving[:, sel]) / total:.2f}, "
              f"by grip slip {np.sum(b & trace['grip_slip_any'][:, sel]) / total:.2f}")

    # Start and physics features by class.
    rest = trace["cube"][ge.WARMUP - 1]                           # (N, 7) at the end of the warm-up
    quat = rest[:, 3:7]
    w, x, y, z = quat.T
    up = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)   # R^T e_z
    flat = np.abs(up).max(-1)
    feats = dict(rest_x_mm=1000 * rest[:, 0], rest_y_mm=1000 * rest[:, 1], rest_z_mm=1000 * rest[:, 2], flatness=flat,
                 tip_friction=physics["tip_friction"], kp_scale=physics["kp_scale"])
    print("  start/physics means by class:  " + "  ".join(f"{k:>12s}" for k in feats))
    for c, name in enumerate(CLASSES):
        sel = label == c
        if sel.sum() >= 3:
            print(f"  {name:18s} n={sel.sum():3d}   " + "  ".join(f"{v[sel].mean():12.3f}" for v in feats.values()))

    # Controller use at the last active step: modes, force references, measured force, closure of the press tendons.
    last = np.clip(end, 0, trace["done"].shape[0] - 1)
    pick = lambda k: trace[k][last, np.arange(N)]
    mode, f_ref, true_f = pick("mode"), pick("f_ref"), pick("digit_force")
    ctrl = pick("ctrl")
    lo, hi = cs.bounds(ge.GraspEdge(arm).mj_model.actuator_ctrlrange)
    closure = (hi[cs.PRESS] - ctrl[:, cs.PRESS]) / (hi[cs.PRESS] - lo[cs.PRESS])   # 0 open .. 1 fully closed
    names = ("free", "engage", "hold", "release")
    for name, sel in (("success", label == 0), ("failures", label != 0)):
        if not sel.any():
            continue
        share = {d: "/".join(f"{np.mean(mode[sel, i] == k):.2f}" for k in range(cs.N_MODES)) for i, d in enumerate(cs.DIGITS)}
        print(f"  {name} final modes ({'/'.join(names)}): {share}")
        print(f"    force ref N {np.round(f_ref[sel].mean(0), 1)}, true digit force N {np.round(true_f[sel].mean(0), 1)}, "
              f"closure {np.round(closure[sel].mean(0), 2)}")
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--arms", nargs="+", default=["c_touch", "c_notouch"])
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dr-seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=None, help="checkpoint step (default: last kept)")
    a = p.parse_args()
    labels = {}
    for arm in a.arms:
        policy, steps = load_policy(a.run, arm, a.steps)
        trace, physics = rollout(arm, policy, a.n, a.seed, a.dr_seed)
        labels[arm] = report(arm, steps, trace, physics)
        np.savez_compressed(a.run / f"diagnosis_{arm}.npz", label=labels[arm], **physics,
                            **{k: v for k, v in trace.items() if k in ("n3", "held", "lifted", "lift_mm", "speed", "spin", "done", "success", "cube")})
    if len(labels) >= 2:
        (x, lx), (y, ly) = list(labels.items())[:2]
        sx, sy = lx == 0, ly == 0
        print(f"\npaired starts: both succeed {np.mean(sx & sy):.2f}, only {x} {np.mean(sx & ~sy):.2f}, "
              f"only {y} {np.mean(~sx & sy):.2f}, neither {np.mean(~sx & ~sy):.2f}")
        same = np.mean(lx[~sx & ~sy] == ly[~sx & ~sy]) if (~sx & ~sy).any() else float("nan")
        print(f"  among starts both fail: same failure class {same:.2f}")


if __name__ == "__main__":
    main()
