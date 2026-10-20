"""Scripted contact programs in the E2 environment (MJX, nominal physics, CPU).

thumb + middle + ring hold 3 N; the thumb slides to abduction ~0.6 and th1 to
its flexed end (the program the numpy probe found). Reports success, time to
success, drops and lift over random official starts.
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
from dextum.touch_mjx import PAD_NAMES
from envs.grasp_edge import ACTIVE, ARMS, EPISODE, REWARD, WARMUP, GraspEdge


def program(t, force=3.0, fingers=(1, 2), abd_steps=4):
    """Contact command at active step t."""
    mode = [cs.FREE] * cs.N_DIGITS
    target = [0.0] * cs.N_DIGITS
    for k in (*fingers, 4):
        mode[k], target[k] = cs.HOLD, force
    slide = [0.0] * cs.N_DIGITS
    slide[4] = -1.0 if t < abd_steps else 0.0
    return cs.command_action(mode, target, slide, thumb_slide=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="c_touch", choices=sorted(ARMS))
    p.add_argument("--policy", default="program", choices=("program", "home"),
                   help="home: zero action (C: all digits free, no slide; A: the home tendon command)")
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--force", type=float, default=3.0)
    a = p.parse_args()
    if a.policy == "program" and ARMS[a.arm][0] != "contact":
        p.error("the contact program needs a contact-space arm")
    env = GraspEdge(a.arm)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(123), a.starts)
    t0 = time.time()
    state = reset(keys)
    success_at = np.full(a.starts, -1)
    dropped = np.zeros(a.starts, bool)
    lifts, n3 = [], []
    returns = {k: np.zeros(a.starts) for k in REWARD}
    rest = np.zeros(env.action_size, np.float32)
    onsets_by_phase = np.zeros(3)                  # before N3 was first reached, while in N3, after leaving N3
    onsets_by_pad = np.zeros(len(PAD_NAMES))
    reached = np.zeros(a.starts, bool)
    previous_n3 = np.zeros(a.starts, bool)
    for tick in range(EPISODE):
        t = tick - WARMUP
        action = program(t, a.force) if (t >= 0 and a.policy == "program") else rest
        previous_slip = np.asarray(state.info["grip_slip"])
        state = step(state, jnp.tile(jnp.asarray(action), (a.starts, 1)))
        m = {k: np.asarray(v) for k, v in state.metrics.items()}
        for k in REWARD:
            returns[k] += np.where(dropped, 0.0, m["e2/reward_" + k])
        onset = np.asarray(state.info["grip_slip"]) & ~previous_slip & ~dropped[:, None] & (t >= 0)
        phase = np.where(previous_n3, 1, np.where(reached, 2, 0))
        np.add.at(onsets_by_phase, phase, onset.sum(1))
        onsets_by_pad += onset.sum(0)
        previous_n3 = m["e2/n3"] > 0
        reached |= previous_n3
        success_at = np.where((m["e2/success"] > 0) & (success_at < 0), t + 1, success_at)
        dropped |= np.asarray(state.done) > 0
        lifts.append(m["e2/lift_mm"])
        n3.append(m["e2/n3"])
        if tick == 0:
            print(f"compiled and first step in {time.time() - t0:.0f} s", flush=True)
    lifts, n3 = np.array(lifts), np.array(n3)
    ok = success_at >= 0
    print(f"{a.arm} {a.policy}: success {ok.sum()}/{a.starts}, drops {dropped.sum()}, "
          f"median time to success {np.median(success_at[ok]) * env.dt if ok.any() else float('nan'):.2f} s; "
          f"max lift median {np.median(lifts.max(0)):.1f} mm; N3 steps median {np.median(n3.sum(0)):.0f} of {ACTIVE}")
    print("per start (success step, max lift mm):", list(zip(success_at.tolist(), np.round(lifts.max(0), 1).tolist())))
    total = sum(returns.values())
    print("episode return mean", f"{total.mean():.2f}", "| success starts", f"{total[ok].mean():.2f}" if ok.any() else "-",
          "| others", f"{total[~ok].mean():.2f}" if (~ok).any() else "-")
    print("  per term (mean):", {k: round(float(v.mean()), 2) for k, v in returns.items()})
    print("  slip onsets per episode: before N3 %.1f, in N3 %.1f, after N3 %.1f" % tuple(onsets_by_phase / a.starts))
    print("  by pad:", {n: round(float(c / a.starts), 1) for n, c in zip(PAD_NAMES, onsets_by_pad) if c})
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
