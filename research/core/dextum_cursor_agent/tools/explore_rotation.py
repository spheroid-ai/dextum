"""E3 feasibility: can contact commands (map G) turn the cube about z at all?

Random piecewise-constant programs in contact space (C) and in tendon space (A),
plus a few hand-written pushes, from the official random starts in the E2
environment (cube settled on the palm, nominal physics). Reports the yaw turned
relative to the resting yaw, and how often a program ends within 10 deg of a
+/-30 deg turn without dropping the cube.

    python tools/explore_rotation.py --programs 512
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs
from envs.grasp_edge import ACTIVE, EPISODE, WARMUP, GraspEdge

SEGMENT = 10                        # steps per program segment (0.5 s)
TURN, TOLERANCE = np.pi / 6, np.pi / 18


def twist(q_now, q_before):
    """Rotation about world z of q_now * conj(q_before), wrapped to (-pi, pi] (the task's yaw measure)."""
    w1, x1, y1, z1 = np.moveaxis(q_now, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q_before, -1, 0) * np.array([1, -1, -1, -1])[:, None]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    angle = 2 * np.arctan2(z, w)
    return np.angle(np.exp(1j * angle))


def random_contact(rng, n):
    segments = ACTIVE // SEGMENT
    out = np.zeros((n, ACTIVE, cs.ACTION_SIZE), np.float32)
    for i in range(n):
        for s in range(segments):
            mode = rng.integers(0, cs.N_MODES, cs.N_DIGITS)
            force = rng.uniform(0, cs.F_MAX, cs.N_DIGITS)
            slide = rng.uniform(-1, 1, cs.N_DIGITS)
            out[i, s * SEGMENT:(s + 1) * SEGMENT] = cs.command_action(mode, force, slide, rng.uniform(-1, 1))
    return out


def random_tendon(rng, n):
    segments = ACTIVE // SEGMENT
    a = rng.uniform(-1, 1, (n, segments, 7)).astype(np.float32)
    return np.repeat(a, SEGMENT, axis=1)


def scripted():
    """Hand-written pushes: one digit presses hard, the others stay free."""
    F, R, H, E = cs.FREE, cs.RELEASE, cs.HOLD, cs.ENGAGE
    progs = {}

    def const(mode, force, slide=(0,) * 5, thumb=0.0):
        a = cs.command_action(mode, force, slide, thumb)
        return np.tile(a, (ACTIVE, 1))

    for k, name in enumerate(cs.DIGITS[:4]):
        mode = [F] * 5
        mode[k] = H
        force = [0.0] * 5
        force[k] = 8.0
        progs[f"{name} push 8 N"] = const(mode, force)
    progs["thumb push 8 N"] = const([F, F, F, F, H], [0, 0, 0, 0, 8.0])
    for sign in (1.0, -1.0):
        progs[f"thumb 3 N, slide {sign:+.0f}"] = const([F, F, F, F, H], [0, 0, 0, 0, 3.0], thumb=sign)
        progs[f"thumb free, slide {sign:+.0f}"] = const([F] * 5, [0] * 5, thumb=sign)
    # Alternating index pushes: hold 1 s, release 0.5 s.
    cycle = []
    for t in range(ACTIVE):
        phase = t % 30
        mode = [F] * 5
        mode[0] = H if phase < 20 else R
        cycle.append(cs.command_action(mode, [8.0, 0, 0, 0, 0], (0,) * 5, 0.0))
    progs["index push/release cycle"] = np.stack(cycle)
    progs["index + middle push 8 N"] = const([H, H, F, F, F], [8.0, 8.0, 0, 0, 0])
    progs["ring + little push 8 N"] = const([F, F, H, H, F], [0, 0, 8.0, 8.0, 0])
    return progs


def simulate(env, actions, seed):
    """actions: (n, ACTIVE, size). Returns yaw turned (n, ACTIVE), dropped (n,), final speed (n,)."""
    n = actions.shape[0]
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    qids = np.asarray(env._cube_qids)
    turned = np.zeros((n, ACTIVE))
    dropped = np.zeros(n, bool)
    zero = jnp.zeros((n, env.action_size))
    previous, total = None, np.zeros(n)
    for tick in range(EPISODE):
        t = tick - WARMUP
        state = step(state, zero if t < 0 else jnp.asarray(actions[:, t]))
        q = np.asarray(state.data.qpos[:, qids[3:7]])
        dropped |= np.asarray(state.done) > 0
        if t >= 0:
            total = total + twist(q, previous)                          # unwrapped sum of per-step twists
            turned[:, t] = total
        previous = q
    speed = np.linalg.norm(np.asarray(state.data.qvel[:, np.asarray(env._cube_qids)[:3]]), axis=-1)
    return turned, dropped, speed


def summarise(name, turned, dropped):
    final = turned[:, -1]
    peak = np.abs(turned).max(1)
    reached = (peak >= TURN) & ~dropped
    ends = (np.abs(np.abs(final) - TURN) < TOLERANCE) & ~dropped
    print(f"  {name:28s} n={len(final):4d}  drops {dropped.mean():.2f}  |turn| median {np.degrees(np.median(np.abs(final))):5.1f} deg, "
          f"peak median {np.degrees(np.median(peak)):5.1f}  reach 30 deg {reached.mean():.2f}  end at 30+-10 {ends.mean():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--programs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    env_c = GraspEdge("c_touch")
    progs = scripted()
    names = list(progs)
    turned, dropped, _ = simulate(env_c, np.stack([progs[k] for k in names]), seed=11)
    print("scripted contact programs (start seed 11, one start each):")
    for i, k in enumerate(names):
        print(f"  {k:28s} turned {np.degrees(turned[i, -1]):7.1f} deg (peak {np.degrees(np.abs(turned[i]).max()):6.1f}), dropped {bool(dropped[i])}")
    # The same programs over 16 starts each, to see the spread.
    reps = 16
    stack = np.concatenate([np.repeat(progs[k][None], reps, 0) for k in names])
    turned, dropped, _ = simulate(env_c, stack, seed=12)
    print(f"scripted programs over {reps} starts each (signed final turn, deg: mean / sd; drops):")
    for i, k in enumerate(names):
        f = np.degrees(turned[i * reps:(i + 1) * reps, -1])
        print(f"  {k:28s} {f.mean():7.1f} / {f.std():5.1f}   drops {dropped[i * reps:(i + 1) * reps].sum()}")
    print(f"[{time.time() - t0:.0f} s]", flush=True)

    print(f"random programs, {a.programs} each, 0.5 s segments, same starts (seed 13):")
    turned, dropped, _ = simulate(env_c, random_contact(rng, a.programs), seed=13)
    summarise("contact space (C, touch G)", turned, dropped)
    env_a = GraspEdge("a_touch")
    turned, dropped, _ = simulate(env_a, random_tendon(rng, a.programs), seed=13)
    summarise("tendon space (A)", turned, dropped)
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
