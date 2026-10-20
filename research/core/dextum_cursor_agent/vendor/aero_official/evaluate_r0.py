"""Registered R0 CPU evaluation: continuity, deployment, DR and robustness.

Separate from training. The randomized bank calls the pinned randomizer rather
than reimplementing its distributions. CPU dynamics are still a reference
evaluation, not an assertion of CPU/MJX trajectory identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import mujoco
import numpy as np

from .cpu_eval import make_model, run_trial, manifest, sha
from .policy import ROOT, PLAYGROUND, MENAGERIE, ReleasedPolicy

DR_FIELDS = ("geom_friction", "body_mass", "body_inertia", "body_ipos", "qpos0",
    "dof_frictionloss", "dof_armature", "dof_damping", "actuator_gainprm", "actuator_biasprm")
SHIFTS = ("nominal", "tip_friction_0.35", "tip_friction_1.2", "dry_friction_3",
    "gain_0.6", "gain_1.4", "cube_mass_0.05", "cube_mass_0.1",
    "cube_size_0.045", "cube_size_0.055", "sensor_noise_2", "latency_1")


def official_draws(seeds):
    from etils import epath
    sys.path.insert(0, str(PLAYGROUND))
    from mujoco_playground._src import mjx_env
    mjx_env.MENAGERIE_PATH = epath.Path(MENAGERIE)
    from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis, domain_randomize
    env = CubeRotateZAxis()
    keys = jax.numpy.stack([jax.random.PRNGKey(seed) for seed in seeds])
    randomized, _ = domain_randomize(env.mjx_model, keys)
    return {name: np.asarray(getattr(randomized, name)) for name in DR_FIELDS}


def set_draw(m, draws, index):
    for name in DR_FIELDS:
        getattr(m, name)[:] = draws[name][index]
    mujoco.mj_setConst(m, mujoco.MjData(m))


def apply_shift(m, shift):
    if shift not in SHIFTS:
        raise ValueError(shift)
    options = dict(sensor_noise_multiplier=1., extra_latency_steps=0)
    cube = m.body("cube").id
    if shift.startswith("tip_friction_"):
        value = float(shift.removeprefix("tip_friction_"))
        for name in ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip"):
            m.geom_friction[m.geom(name).id, 0] = value
    elif shift == "dry_friction_3":
        m.dof_frictionloss[:16] *= 3
    elif shift.startswith("gain_"):
        scale = float(shift.removeprefix("gain_"))
        m.actuator_gainprm[:, 0] *= scale
        m.actuator_biasprm[:, 1] *= scale
    elif shift.startswith("cube_mass_"):
        mass = float(shift.removeprefix("cube_mass_"))
        m.body_inertia[cube] *= mass / m.body_mass[cube]
        m.body_mass[cube] = mass
    elif shift.startswith("cube_size_"):
        size = float(shift.removeprefix("cube_size_"))
        old_size = 2 * m.geom_size[m.geom("cube").id, 0]
        m.geom_size[m.geom("cube").id] = size / 2
        m.body_inertia[cube] *= (size / old_size)**2
    elif shift == "sensor_noise_2":
        options["sensor_noise_multiplier"] = 2.
    elif shift == "latency_1":
        options["extra_latency_steps"] = 1
    if shift != "nominal":
        mujoco.mj_setConst(m, mujoco.MjData(m))
    return options


def jobs(suite):
    if suite == "fixed100":
        return [("nominal", s, 15., "training_interface", False) for s in range(2001, 2101)]
    if suite == "c1":
        return [("nominal", s, 25., "deployment_interface", True) for s in range(1, 51)]
    if suite == "dr":
        return [("official_dr", s, 25., "training_interface", False) for s in range(4001, 4051)]
    if suite == "c2":
        return [(shift, s, 25., "training_interface", False) for shift in SHIFTS for s in range(1001, 1011)]
    raise ValueError(suite)


def evaluate(checkpoint, suite, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    digest = sha(checkpoint)
    report_path = output / "report.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old["checkpoint_sha256"] != digest or old["suite"] != suite:
            raise ValueError("Refusing to mix evaluations")
        if old["state"] == "COMPLETE":
            return old
    policy = ReleasedPolicy(checkpoint)
    policy(np.zeros(14, np.float32))
    _, files = make_model()
    report = dict(state="RUNNING", suite=suite, checkpoint_sha256=digest,
        manifest=manifest(files, checkpoint), rows=[],
        unique_starts=1 if suite=="c1" else (100 if suite=="fixed100" else 50 if suite=="dr" else 10),
        primary_spin="signed world-Z integral / requested episode duration; zero motion credited after a drop",
        deviation="C1 repeats one exact nominal start; other suites use declared common randomized starts")
    report["manifest"]["deviations"] = [
        "CPU MuJoCo reference, not exact MJX trajectory parity",
        "Simulation sensor/command interface; no hardware map, motor dynamics or hardware result"]
    draws = official_draws(range(9001, 9051)) if suite=="dr" else None
    if draws is not None:
        draw_file = output / "official_dr_draws.npz"
        if not draw_file.exists():
            np.savez(draw_file, **draws)
        report["draws_sha256"] = sha(draw_file)
    try:
        for index, (shift, seed, duration, mode, nominal) in enumerate(jobs(suite)):
            trial_path = output / f"{shift}_{seed}.json"
            if trial_path.exists():
                result = json.loads(trial_path.read_text())
                if result["checkpoint_sha256"] != digest:
                    raise ValueError("Trial checkpoint changed")
            else:
                m, _ = make_model()
                if draws is not None:
                    set_draw(m, draws, index)
                    options = {}
                else:
                    options = apply_shift(m, shift)
                result = run_trial(m, policy, seed, mode, duration, nominal, **options)
                result.update(checkpoint_sha256=digest, shift=shift)
                trial_path.with_suffix(".tmp").write_text(json.dumps(result)+"\n")
                trial_path.with_suffix(".tmp").replace(trial_path)
            actions = np.asarray([r["action"] for r in result["trajectory"]])
            row = {k: result[k] for k in ("seed", "shift", "retained", "elapsed_sim_s", "signed_world_z_rotation_rad", "reward")}
            row.update(spin_rad_s=result["signed_world_z_rotation_rad"]/duration,
                mean_command_change=float(np.linalg.norm(np.diff(np.vstack((np.zeros((1,7)), actions)), axis=0), axis=1).mean()))
            report["rows"].append(row)
            report_path.with_suffix(".tmp").write_text(json.dumps(report, indent=2)+"\n")
            report_path.with_suffix(".tmp").replace(report_path)
            print(json.dumps(dict(suite=suite, **row)), flush=True)
        report["summary"] = {}
        for shift in sorted({r["shift"] for r in report["rows"]}):
            rows = [r for r in report["rows"] if r["shift"]==shift]
            report["summary"][shift] = dict(trials=len(rows), retained=sum(r["retained"] for r in rows),
                mean_rotation_rad=float(np.mean([r["signed_world_z_rotation_rad"] for r in rows])),
                mean_spin_rad_s=float(np.mean([r["spin_rad_s"] for r in rows])),
                mean_command_change=float(np.mean([r["mean_command_change"] for r in rows])))
        if suite=="c2":
            base = report["summary"]["nominal"]["mean_spin_rad_s"]
            report["robustness_retention"] = float(np.mean([report["summary"][s]["mean_spin_rad_s"]/base
                for s in SHIFTS if s!="nominal"])) if base>0 else None
        report["state"] = "COMPLETE"
    except BaseException as exc:
        report.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_s"] = time.monotonic()-started
        report_path.write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--suite", choices=("fixed100", "c1", "dr", "c2"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evaluate(args.checkpoint, args.suite, args.output)
