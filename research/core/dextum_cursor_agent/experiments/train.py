"""PPO for one E7-lite arm, warm-started from the working vision rotator (OR1).

Both arms start from the same function: OR1's actor and critic with zero weights
on every new input (past frames, touch, perturbation state). PPO settings are
OR1's own, so the only change at step 0 is the training distribution.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import train as ppo

from envs.touch_orientation import Perturbations, TouchOrientation
from dextum import policy_io
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize
from mujoco_playground.config import manipulation_params

ARMS = {"vision": "none", "touch": "ideal"}


def perturbations(name, push_max=None):
    if name == "none":
        return Perturbations(blackout_prob=0.0, push_prob=0.0)
    p = Perturbations()
    if push_max is not None:
        p = Perturbations(push_newton=(min(p.push_newton[0], push_max), push_max))
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--evals", type=int, default=None)
    p.add_argument("--perturb", choices=("spec", "none"), default="spec")
    p.add_argument("--push-max", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="CPU-sized batches for a pipeline check")
    a = p.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    if not a.smoke and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Full training requires a GPU; use --smoke for a CPU check")
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    a.output.mkdir(parents=True)

    touch = ARMS[a.arm]
    perturb = perturbations(a.perturb, a.push_max)
    source = model.load_params(a.source)
    scaling = policy_io.frozen_scaling(source, touch)
    restored = policy_io.warm_start(source, touch)
    policy_io.save_scaling(a.output / "scaling.npz", scaling)

    config = manipulation_params.brax_ppo_config("AeroCubeRotateZAxis").to_dict()
    config.pop("network_factory")
    config.update(discounting=0.995, entropy_cost=0.01, num_envs=a.envs, num_timesteps=a.steps,
                  seed=a.seed, log_training_metrics=True, training_metrics_steps=327680,
                  deterministic_eval=True)
    if a.smoke:
        config.update(num_envs=a.envs, batch_size=a.envs // 2, num_minibatches=2, unroll_length=10,
                      num_eval_envs=4, training_metrics_steps=a.envs * 10, num_resets_per_eval=0)
    per_iteration = config["batch_size"] * config["num_minibatches"] * config["unroll_length"]
    config["num_evals"] = a.evals or max(2, a.steps // (20 * per_iteration) + 1)

    record = dict(state="STARTING", arm=a.arm, touch=touch, perturbations=perturb.__dict__,
                  source=dict(path=str(a.source), sha256=sha256(a.source)), configuration=config,
                  actor_inputs=int(scaling["actor_mean"].shape[0]), critic_inputs=int(scaling["critic_mean"].shape[0]),
                  device=str(jax.devices()), smoke=a.smoke, started_at=time.time(),
                  warm_start="OR1 actor/critic, zero weights on new inputs, OR1 exploration, fresh Adam",
                  scaling="frozen OR1 statistics for motor/privileged; fixed elsewhere")
    (a.output / "config.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    started = time.monotonic()
    checkpoints = []

    def progress(steps, metrics):
        values = {str(k): float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).size == 1}
        bad = [k for k, v in values.items() if not np.isfinite(v)]
        if bad:
            raise RuntimeError(f"Nonfinite PPO metrics: {bad[:5]}")
        row = dict(steps=int(steps), elapsed_s=time.monotonic() - started, metrics=values)
        with (a.output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        keys = ("eval/episode_target_success", "episode/target_success", "eval/episode_target_falling",
                "eval/episode_reward")
        print(json.dumps(dict(steps=int(steps), t=round(row["elapsed_s"]),
                              **{k.split("/")[-1] + ("_eval" if k.startswith("eval") else ""): round(values[k], 4)
                                 for k in keys if k in values})), flush=True)

    def save(steps, make_policy, params):
        del make_policy
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
            raise RuntimeError("Nonfinite parameters")
        if not checkpoints:
            for x, y in zip(jax.tree.leaves(params[1:]), jax.tree.leaves(restored[1:])):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            record["initial_checkpoint_exact"] = True
        path = a.output / f"policy_{int(steps)}.pkl"
        model.save_params(path, params)
        checkpoints.append(dict(steps=int(steps), elapsed_s=time.monotonic() - started, path=path.name))
        (a.output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")

    env = TouchOrientation(touch=touch, perturb=perturb)
    try:
        ppo.train(environment=env, network_factory=policy_io.network_factory(scaling),
                  wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
                  randomization_fn=domain_randomize, progress_fn=progress, policy_params_fn=save,
                  restore_params=restored, **config)
        record.update(state="COMPLETE")
        if len(checkpoints) >= 3:
            (s0, t0), (s1, t1) = [(c["steps"], c["elapsed_s"]) for c in checkpoints[-2:]]
            record["steady_decisions_per_s"] = (s1 - s0) / max(t1 - t0, 1e-9)
    except BaseException as exc:
        record.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(wall_s=time.monotonic() - started, checkpoints=checkpoints)
        (a.output / "report.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
