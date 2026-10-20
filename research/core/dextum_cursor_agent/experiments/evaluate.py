"""E7-lite evaluation banks, batched in MJX (§16.2 tests 1-4).

Every arm sees identical starts, goals, cameras, physics draws and perturbation
times: the schedule is generated here from a fixed seed and written into the
episode state after reset, so one compiled rollout serves every condition.
"""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np

from envs.touch_orientation import (BASE_ACTOR, BASE_CRITIC, DROP, FREEZE, HISTORY, Perturbations,
                                    TouchOrientation, actor_frame_size)
from dextum import policy_io, touch_mjx
from vendor.aero_official.goal_yaw import TIMEOUT, WARMUP
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

DT = 0.05
HORIZON = WARMUP + TIMEOUT
BLACKOUT_TICKS = (0, 10, 20, 40, 60)          # 0, 0.5, 1, 2, 3 s
BLACKOUT_WINDOW = (WARMUP + 5, WARMUP + 25)   # mid-rotation
PUSH_WINDOW = (WARMUP + 5, WARMUP + 60)
PUSH_NEWTON = 1.0
PUSH_TICKS = Perturbations().push_ticks


def conditions():
    out = [("clean", None, 0, DROP)]
    for mode, code in (("drop", DROP), ("freeze", FREEZE)):
        out += [(f"{mode}_{t * DT:.1f}s", "blackout", t, code) for t in BLACKOUT_TICKS if t]
    out.append((f"push_{PUSH_NEWTON:.0f}N", "push", 0, DROP))
    return out


def schedule(kind, ticks, mode, n, seed):
    rng = np.random.default_rng(seed)
    blackout_start = rng.integers(BLACKOUT_WINDOW[0], BLACKOUT_WINDOW[1] + 1, n)
    push_start = rng.integers(PUSH_WINDOW[0], PUSH_WINDOW[1] + 1, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    none = -np.ones(n, np.int32)
    s = dict(blackout_start=none, blackout_end=none, blackout_mode=np.full(n, mode, np.int32),
             push_start=none, push_force=np.zeros((n, 3), np.float32))
    if kind == "blackout":
        s.update(blackout_start=blackout_start.astype(np.int32),
                 blackout_end=(blackout_start + ticks).astype(np.int32))
    elif kind == "push":
        s.update(push_start=push_start.astype(np.int32),
                 push_force=(PUSH_NEWTON * np.stack((np.cos(angle), np.sin(angle), np.zeros(n)), -1)).astype(np.float32))
    return {k: jnp.asarray(v) for k, v in s.items()}


def touch_columns(touch):
    """Actor-input columns holding touch features, over all history frames."""
    frame = actor_frame_size(touch)
    if touch == "none":
        return np.zeros(0, int)
    return np.concatenate([f * frame + np.arange(BASE_ACTOR, frame) for f in range(HISTORY)])


def make_rollout(policy, touch, n, seed, dr_seed, dump=False, ablate_touch=False):
    env = TouchOrientation(touch=touch, perturb=Perturbations(blackout_prob=0.0, push_prob=0.0))
    dr_keys = jax.random.split(jax.random.PRNGKey(dr_seed), n)
    wrapped = BraxDomainRandomizationVmapWrapper(env, functools.partial(domain_randomize, rng=dr_keys))
    keys = jax.vmap(jax.random.PRNGKey)(seed + jnp.arange(n))
    engaged_idx = jnp.array([BASE_CRITIC + 7 * p + 6 for p in range(touch_mjx.N_PADS)])
    read = jax.vmap(env.touch_model.read, in_axes=(wrapped._in_axes, 0))
    ablated = jnp.asarray(touch_columns(touch)) if ablate_touch else None

    def act(obs):
        if ablated is not None:
            obs = dict(obs, state=obs["state"].at[:, ablated].set(0.0))
        return policy(obs)

    @jax.jit
    def rollout(sched):
        state = wrapped.reset(keys)
        state.info["schedule"] = sched

        def body(state, _):
            action = act(state.obs)
            nxt = wrapped.step(state, action)
            m = nxt.metrics
            rec = dict(done=nxt.done, success=m["target_success"], error=m["target_error"],
                       z=nxt.data.qpos[:, 18], speed=jnp.linalg.norm(nxt.data.qvel[:, 16:19], axis=-1),
                       blackout=m["dextum_blackout"], push=m["dextum_push"],
                       visible=m["target_visible"], quality=m["dextum_quality"], shear=m["dextum_shear"],
                       engaged=nxt.obs["privileged_state"][:, engaged_idx] > 0.5, action=action)
            if dump:
                r = read(wrapped._mjx_model_v, nxt.data)
                rec.update(pad_force=r.force, pad_fn=r.f_n, pad_ft=r.f_t, pad_slip=r.slip, pad_centroid=r.centroid,
                           pad_point=r.point_obj, pad_normal=r.normal_obj, pad_margin=r.margin,
                           cube_qpos=nxt.data.qpos[:, 16:23], cube_qvel=nxt.data.qvel[:, 16:22],
                           goal=nxt.info["goal_world"], age=nxt.info["age"])
            return nxt, rec

        _, trace = jax.lax.scan(body, state, None, length=HORIZON)
        return trace

    return rollout


def settle_time(speed, dropped, still=0.01, hold=5):
    """Seconds after the push ends until the cube stays below `still` m/s for `hold` steps."""
    if dropped:
        return math.nan
    calm = speed < still
    for k in range(len(calm) - hold + 1):
        if calm[k:k + hold].all():
            return k * DT
    return math.nan


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def summarize(trace, sched, kind):
    t = {k: np.asarray(v) for k, v in trace.items()}            # (T, N, ...)
    T, N = t["done"].shape
    done = t["done"] > 0.5
    first = np.where(done.any(0), done.argmax(0), T - 1)
    idx = np.arange(N)
    success = t["success"][first, idx] > 0.5
    drop = t["z"][first, idx] < -0.05
    timeout = ~success & ~drop
    out = dict(n=N, success=int(success.sum()), drops=int(drop.sum()), timeouts=int(timeout.sum()))
    out["success_rate"] = success.mean()
    out["success_ci95"] = wilson(int(success.sum()), N)
    out["drop_rate"] = drop.mean()
    tts = (first[success] + 1 - WARMUP) * DT
    out["time_to_success_s"] = dict(mean=float(tts.mean()) if tts.size else None,
                                    median=float(np.median(tts)) if tts.size else None)
    active = (np.arange(T)[:, None] >= WARMUP) & (np.arange(T)[:, None] <= first[None])
    a = t["action"]
    da = np.linalg.norm(np.diff(a, axis=0), axis=-1)            # (T-1, N)
    jerk = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)
    out["mean_abs_da"] = float((da * active[1:]).sum() / max(active[1:].sum(), 1))
    out["mean_jerk"] = float((jerk * active[2:]).sum() / max(active[2:].sum(), 1))
    shear = np.where(active, t["shear"], np.nan)
    out["shear_variance"] = float(np.nanmean(np.nanvar(shear, axis=0)))
    eng = t["engaged"]                                          # (T, N, P)
    alive = (np.arange(T)[:, None] <= first[None])[..., None]
    eng = eng & alive
    seen = np.cumsum(eng, axis=0) > 0
    regained = (~eng[:-1] & eng[1:]) & seen[:-1]
    out["regrasps_per_episode"] = float(regained.sum((0, 2)).mean())
    out["visible_fraction"] = float((t["visible"] * active).sum() / max(active.sum(), 1))
    if kind == "push":
        # Trace index k holds tick k + 1; the push starts at tick `start`.
        start = np.asarray(sched["push_start"]) - 1
        q = t["quality"]
        before = q[start - 1, idx]
        by_quality, settle = [], []
        for i in range(N):
            if first[i] < start[i]:
                continue
            after = np.where(q[start[i]:first[i] + 1, i] >= before[i])[0]
            by_quality.append(after[0] * DT if after.size else math.nan)
            settle.append(settle_time(t["speed"][start[i] + PUSH_TICKS:first[i] + 1, i], drop[i]))
        by_quality, settle = np.array(by_quality), np.array(settle)
        out["push_recovery_s"] = dict(median=float(np.nanmedian(by_quality)) if np.isfinite(by_quality).any() else None,
                                      recovered_fraction=float(np.isfinite(by_quality).mean()) if by_quality.size else None)
        out["push_settle_s"] = dict(median=float(np.nanmedian(settle)) if np.isfinite(settle).any() else None,
                                    mean=float(np.nanmean(settle)) if np.isfinite(settle).any() else None,
                                    settled_fraction=float(np.isfinite(settle).mean()) if settle.size else None)
        out["push_episode_drops"] = int((drop & (first >= start)).sum())
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def evaluate(checkpoint, output, n=200, seed=81001, dr_seed=82001, only=None, dump=None, ablate_touch=False):
    run_dir = Path(checkpoint).parent
    config = json.loads((run_dir / "config.json").read_text())
    touch = config["touch"]
    policy = policy_io.Policy(checkpoint)
    rollout = make_rollout(policy, touch, n, seed, dr_seed, ablate_touch=ablate_touch)
    arm = config["arm"] + ("+touch-zeroed" if ablate_touch else "")
    report = dict(checkpoint=str(checkpoint), arm=arm, touch=touch, n=n, seed=seed, dr_seed=dr_seed,
                  ablate_touch=ablate_touch, started_at=time.time(), conditions={})
    if dump:
        traced = make_rollout(policy, touch, n, seed, dr_seed, dump=True)(schedule(None, 0, DROP, n, seed))
        np.savez_compressed(dump, **{k: np.asarray(v) for k, v in traced.items()})
        del traced
    for name, kind, ticks, mode in conditions():
        if only and name not in only:
            continue
        sched = schedule(kind, ticks, mode, n, seed)
        t0 = time.time()
        trace = rollout(sched)
        jax.block_until_ready(trace)
        report["conditions"][name] = dict(summarize(trace, sched, kind), wall_s=time.time() - t0)
        print(json.dumps(dict(condition=name, **{k: report["conditions"][name][k] for k in
                                               ("success", "drops", "timeouts", "time_to_success_s")})), flush=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    report["wall_s"] = time.time() - report["started_at"]
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def checkpoint_in(run_dir, which):
    steps = sorted(int(p.stem.split("_")[1]) for p in Path(run_dir).glob("policy_*.pkl"))
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return Path(run_dir) / f"policy_{steps[-1] if which == 'final' else steps[0]}.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--run", type=Path, help="run directory; with --which picks a checkpoint")
    p.add_argument("--which", choices=("final", "initial"), default="final")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=81001)
    p.add_argument("--dr-seed", type=int, default=82001)
    p.add_argument("--only", nargs="*")
    p.add_argument("--dump", type=Path, help="write clean-condition per-pad traces (E1) to this .npz")
    p.add_argument("--ablate-touch", action="store_true", help="zero the policy's touch inputs (reliance test)")
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    checkpoint = a.checkpoint or checkpoint_in(a.run, a.which)
    evaluate(checkpoint, a.output, a.n, a.seed, a.dr_seed, a.only, a.dump, a.ablate_touch)


if __name__ == "__main__":
    main()
