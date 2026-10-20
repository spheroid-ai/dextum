"""E3 reward check before training (MJX, nominal physics, CPU).

Programs in the E3 environment: doing nothing, the E2 grasp program (grip and
wait), random 0.5 s contact programs and random tendon programs. Reports success,
drops, and the episode return (undiscounted, and discounted with gamma 0.97 from
the first active step) by outcome, so that the reward's ranking can be checked:
a reward that pays more for gripping or waiting than for reaching the goal would
test the reward, not the action space.

    python tools/scripted_e3.py --random 256
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs
from envs.blind_rotation import ACTIVE, EPISODE, GAMMA, NODES, REWARD, WARMUP, BlindRotation
from scripted_e2 import program

SEGMENT = 10                        # steps per random program segment (0.5 s)


def random_contact(rng, n):
    out = np.zeros((n, ACTIVE, cs.ACTION_SIZE), np.float32)
    for i in range(n):
        for s in range(0, ACTIVE, SEGMENT):
            out[i, s:s + SEGMENT] = cs.command_action(rng.integers(0, cs.N_MODES, cs.N_DIGITS),
                                                      rng.uniform(0, cs.F_MAX, cs.N_DIGITS),
                                                      rng.uniform(-1, 1, cs.N_DIGITS), rng.uniform(-1, 1))
    return out


def random_tendon(rng, n):
    a = rng.uniform(-1, 1, (n, ACTIVE // SEGMENT, 7)).astype(np.float32)
    return np.repeat(a, SEGMENT, axis=1)


def run(arm, groups, seed):
    """groups: {name: actions (n, ACTIVE, size)}; every group uses the first n starts of one bank."""
    env = BlindRotation(arm)
    names = list(groups)
    actions = np.concatenate([groups[k] for k in names])
    sizes = [len(groups[k]) for k in names]
    bank = jax.random.split(jax.random.PRNGKey(seed), max(sizes))
    keys = jnp.concatenate([bank[:n] for n in sizes])
    n = len(actions)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    state = reset(keys)
    zero = jnp.zeros((n, env.action_size))
    dropped = np.zeros(n, bool)
    terms = {k: np.zeros(n) for k in REWARD}
    discounted = np.zeros(n)
    success_at = np.full(n, -1)
    nodes = {k: np.zeros(n) for k in NODES}
    error_end = np.zeros(n)
    t0 = time.time()
    for tick in range(EPISODE):
        t = tick - WARMUP
        state = step(state, zero if t < 0 else jnp.asarray(actions[:, t]))
        m = {k: np.asarray(v) for k, v in state.metrics.items()}
        alive = ~dropped
        r = np.zeros(n)
        for k in REWARD:
            v = np.where(alive, m["e3/reward_" + k], 0.0)
            terms[k] += v
            r += v
        if t >= 0:
            discounted += GAMMA ** t * r
            for k in NODES:
                nodes[k] += np.where(alive, m["e3/" + k], 0.0)
        success_at = np.where((m["e3/success"] > 0) & (success_at < 0) & alive, t + 1, success_at)
        error_end = np.where(alive, m["e3/abs_error_deg"], error_end)
        dropped |= np.asarray(state.done) > 0
        if tick == 0:
            print(f"{arm}: compiled and first step in {time.time() - t0:.0f} s", flush=True)
    print(f"{arm}: {n} episodes in {time.time() - t0:.0f} s", flush=True)
    out, start = {}, 0
    for k, size in zip(names, sizes):
        sl = slice(start, start + size)
        out[k] = dict(success=success_at[sl] >= 0, success_at=success_at[sl], dropped=dropped[sl],
                      total=sum(v[sl] for v in terms.values()), discounted=discounted[sl],
                      terms={j: v[sl] for j, v in terms.items()}, nodes={j: v[sl] for j, v in nodes.items()},
                      error_end=error_end[sl])
        start += size
    return env.dt, out


def report(name, r, dt):
    ok, drop = r["success"], r["dropped"]
    other = ~ok & ~drop
    def mean(x, mask):
        return f"{x[mask].mean():6.2f}" if mask.any() else "     -"
    print(f"{name:24s} n={len(ok):4d} success {ok.mean():.2f} drops {drop.mean():.2f}  "
          f"time to success {np.median(r['success_at'][ok]) * dt if ok.any() else float('nan'):4.1f} s  "
          f"final |error| median {np.median(r['error_end']):5.1f} deg")
    for label, x in (("return", r["total"]), ("discounted", r["discounted"])):
        print(f"    {label:10s} all {x.mean():6.2f} | success {mean(x, ok)} | no success, kept {mean(x, other)} | dropped {mean(x, drop)}")
    print("    terms (mean):", {k: round(float(v.mean()), 2) for k, v in r["terms"].items()})
    print("    node steps (mean of", ACTIVE, "):", {k: round(float(v.mean()), 1) for k, v in r["nodes"].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random", type=int, default=256)
    p.add_argument("--scripted", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-seed", type=int, default=2026)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    home_c = np.zeros((a.scripted, ACTIVE, cs.ACTION_SIZE), np.float32)
    grip = np.tile(np.stack([program(t) for t in range(ACTIVE)])[None], (a.scripted, 1, 1))
    dt, c = run("c_touch", {"C home": home_c, "C grip and wait": grip,
                            "C random programs": random_contact(rng, a.random)}, a.start_seed)
    home_a = np.zeros((a.scripted, ACTIVE, 7), np.float32)
    _, t = run("a_touch", {"A home": home_a, "A random programs": random_tendon(rng, a.random)}, a.start_seed)
    print()
    for name, r in {**c, **t}.items():
        report(name, r, dt)


if __name__ == "__main__":
    main()
