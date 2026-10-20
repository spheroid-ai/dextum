"""Locate the gap between the training-time evaluator and evaluate_e3.py on E3 checkpoints.

Final checkpoint per arm, 128 deterministic episodes per variant, "highest" precision:
  brax / eval draws   Brax's Evaluator with the DR draws ppo.train used for evaluation
  brax / our draws    the same Evaluator with evaluate_e3.py's DR draws (DR seed 7)
  ours / eval draws   evaluate_e3.py's rollout with ppo.train's evaluation DR draws
  ours / our draws    evaluate_e3.py's rollout with its own draws and starts
Both draw sets are summarised, and per-episode success is split by fingertip friction.

    python tools/eval_gap.py runs/e3p1-s1 --output runs/<job> [--arms c_touch a_task]
"""
import argparse
import functools
import json
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
from brax.training import acting
from brax.training.agents.ppo import networks

import vendor.aero_official.vision_env  # noqa: F401
from envs.blind_rotation import EPISODE, BlindRotation
from experiments.evaluate_e3 import make_rollout, summarise
from experiments.train_e3 import make_network
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand.rotate_z import domain_randomize

N = 128
TIPS = ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip")


def training_eval_key(seed):
    """The evaluation key of brax ppo.train for this seed (single process)."""
    _, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, 0)
    return jax.random.split(local_key, 3)[2]


def draw_summary(env, dr_keys):
    m, _ = domain_randomize(env.mjx_model, dr_keys)
    tip = np.asarray(m.geom_friction[:, env.mj_model.geom(TIPS[0]).id, 0])
    nominal_kp = np.asarray(env.mj_model.actuator_gainprm[:, 0])
    kp = np.asarray(m.actuator_gainprm[:, :, 0]) / np.where(nominal_kp == 0, 1, nominal_kp)
    return tip, dict(tip_friction=[float(tip.mean()), float(tip.min()), float(tip.max())], kp_scale_mean=float(kp.mean()))


def by_friction(success, tip):
    cuts = np.quantile(tip, [1 / 3, 2 / 3])
    bins = np.digitize(tip, cuts)
    return [float(success[bins == b].mean()) for b in range(3)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["c_touch", "a_task"])
    p.add_argument("--seed", type=int, default=1, help="training seed of the run")
    p.add_argument("--repeats", type=int, default=4, help="Brax evaluations per variant (new starts each)")
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    a.output.mkdir(parents=True, exist_ok=True)
    eval_key = training_eval_key(a.seed)
    draws = {"eval draws": jax.random.split(eval_key, N),
             "our draws": jax.random.split(jax.random.PRNGKey(7), 200)[:N]}
    starts = jax.random.split(jax.random.PRNGKey(2026), 200)[:N]
    results = {}
    for arm in a.arms:
        chosen = [c for c in json.loads((a.run / arm / "checkpoints.json").read_text()) if c.get("kept", True)][-1]
        params = model.load_params(a.run / arm / chosen["path"])
        make_policy = networks.make_inference_fn(make_network(arm))
        env = BlindRotation(arm)
        results[arm] = dict(steps=chosen["steps"], variants={}, draws={})
        run = None
        for draw_name, dr_keys in draws.items():
            tip, results[arm]["draws"][draw_name] = draw_summary(env, dr_keys)

            t0 = time.time()
            eval_env = wrapper.wrap_for_brax_training(
                env, episode_length=EPISODE, action_repeat=1, full_reset=True,
                randomization_fn=functools.partial(domain_randomize, rng=dr_keys))
            evaluator = acting.Evaluator(eval_env, functools.partial(make_policy, deterministic=True),
                                         num_eval_envs=N, episode_length=EPISODE, action_repeat=1, key=eval_key)
            success = []
            for _ in range(a.repeats):
                m = evaluator.run_evaluation(params, training_metrics={}, aggregate_episodes=False)
                success.append(np.asarray(m["eval/episode_e3/success"]) > 0)
            success = np.stack(success)
            v = dict(success=float(success.mean()), per_repeat=[float(s.mean()) for s in success],
                     by_tip_friction_tercile=by_friction(success.mean(0), tip),
                     within=float(np.mean(m["eval/episode_e3/within"])), drop=float(np.mean(m["eval/episode_e3/drop"])))
            results[arm]["variants"][f"brax / {draw_name}"] = v
            print(f"{arm:9s} brax / {draw_name:10s} success {v['success']:.3f} {v['per_repeat']} "
                  f"by tip friction {v['by_tip_friction_tercile']} [{time.time() - t0:.0f} s]", flush=True)

            t0 = time.time()
            model_v, axes = domain_randomize(env.mjx_model, dr_keys)
            infer = make_policy(params, deterministic=True)
            run = run or make_rollout(env, lambda obs: infer(obs, jax.random.PRNGKey(0))[0], axes)
            trace, sign = run(model_v, starts, jnp.zeros(2, jnp.float32))
            r = summarise({k: np.asarray(x) for k, x in trace.items()}, np.asarray(sign))
            v = dict(success=float(r["success"].mean()), by_tip_friction_tercile=by_friction(r["success"], tip),
                     within=float(r["within_steps"].mean()), drop=float(r["dropped"].mean()))
            results[arm]["variants"][f"ours / {draw_name}"] = v
            print(f"{arm:9s} ours / {draw_name:10s} success {v['success']:.3f} "
                  f"by tip friction {v['by_tip_friction_tercile']} [{time.time() - t0:.0f} s]", flush=True)
        print(f"{arm:9s} draws {json.dumps(results[arm]['draws'])}", flush=True)
    (a.output / "eval_gap.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
