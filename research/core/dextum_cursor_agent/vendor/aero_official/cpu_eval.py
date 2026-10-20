"""CPU qualification, not a fresh training reproduction or hardware result.

Scene/assets and policy come from the pinned official sources. Training-interface
mode preserves the upstream observation's action-history lag. Deployment-interface
mode preserves the ROS node's publish-then-compute order (one command tick delay).
Neither imports a historical Aero scene or adds its command limits/smoothing.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
import mujoco

from .policy import (ROOT, OFFICIAL, PLAYGROUND, MENAGERIE, CHECKPOINT, TASK,
                     ReleasedPolicy, TRAIN_SCALE, DEPLOY_SCALE, DEPLOY_HOME, SENSORS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_model():
    assets = {}
    files = []
    for folder, pattern in [(MENAGERIE / "tetheria_aero_hand_open/assets", "*"),
                            (TASK / "xmls", "*.xml"),
                            (TASK / "xmls/reorientation_cube_textures", "*"),
                            (TASK / "xmls/assets", "*")]:
        for p in sorted(folder.glob(pattern)):
            if p.is_file():
                assets[p.name] = p.read_bytes()
                files.append(p)
    scene = TASK / "xmls/scene_mjx_cube.xml"
    m = mujoco.MjModel.from_xml_string(scene.read_text(), assets)
    m.opt.timestep = .01
    return m, files


def sensor(data, name):
    return np.array(data.sensor(name).data, copy=True)


def reset(m, seed, nominal_keyframe=False):
    """Same reset equations and PRNG draw order as the pinned rotate_z.reset."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    if nominal_keyframe:
        d.mocap_pos[:]=-100
        mujoco.mj_forward(m,d)
        return d,jax.random.PRNGKey(seed)
    rng, pos_rng, vel_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
    del vel_rng
    d.qpos[:16] = np.clip(m.key("home").qpos[:16] + .1 * np.asarray(
        jax.random.normal(pos_rng, (16,))), m.jnt_range[:16, 0], m.jnt_range[:16, 1])
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    d.qpos[16:19] = np.array([.1, 0, .05]) + np.asarray(jax.random.uniform(
        p_rng, (3,), minval=-.01, maxval=.01))
    u, v, w = np.asarray(jax.random.uniform(quat_rng, (3,)))
    d.qpos[19:23] = [np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v),
                      np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)]
    d.qvel[:] = 0
    d.mocap_pos[:] = -100
    mujoco.mj_forward(m, d)
    return d, rng


def observation(d, previous_action, rng, noise, noise_multiplier=1.):
    # Preserve all four upstream random splits, including two unused keys.
    rng, _ = jax.random.split(rng)
    rng, noise_rng = jax.random.split(rng)
    values = np.array([sensor(d, n).item() for n in SENSORS], dtype=np.float32)
    if noise:
        values[:6] += np.asarray(2*jax.random.uniform(noise_rng, (6,))-1)*(.005*noise_multiplier)
    rng, noise_rng = jax.random.split(rng)
    if noise:
        values[6:] += np.asarray(2*jax.random.uniform(noise_rng, (1,))-1)*(.05*noise_multiplier)
    rng, _ = jax.random.split(rng)
    return np.concatenate([values, previous_action]).astype(np.float32), rng


def run_trial(m, policy, seed, mode, duration, nominal_keyframe=False,
              sensor_noise_multiplier=1., extra_latency_steps=0):
    if sensor_noise_multiplier<0 or extra_latency_steps<0:
        raise ValueError("Noise scale and latency must be nonnegative")
    d, rng = reset(m, seed, nominal_keyframe)
    deploy = mode == "deployment_interface"
    nominal = DEPLOY_HOME if deploy else np.array(m.key("home").ctrl)
    scale = DEPLOY_SCALE if deploy else TRAIN_SCALE
    last = np.zeros(7, np.float32)
    queued = nominal.copy()
    if not deploy:
        # Upstream make_data does not call forward at reset, so its first sensor
        # packet is zero (plus configured noise). Preserve that startup behavior.
        d.sensordata[:] = 0
    obs, rng = observation(d, last, rng, noise=not deploy, noise_multiplier=sensor_noise_multiplier)
    delayed_observations = [obs.copy() for _ in range(extra_latency_steps)]
    rotations = 0.
    reward = 0.
    inference_ms = []
    trajectory = []
    start = time.perf_counter()
    dropped = False
    for tick in range(round(duration/.05)):
        t = time.perf_counter()
        delayed_observations.append(obs)
        action = policy(delayed_observations.pop(0))
        inference_ms.append(1000*(time.perf_counter()-t))
        target = nominal + action * scale
        d.ctrl[:] = queued if deploy else target
        queued = target
        for _ in range(5):
            mujoco.mj_step(m, d)
            rotations += float(sensor(d, "cube_angvel")[2])*.01
        dropped = float(sensor(d, "cube_position")[2]) < -.05
        reward += .05*(float(sensor(d, "cube_angvel")[2]) -
                       np.square(action-last).sum() - 100*int(dropped))
        trajectory.append(dict(t=float(d.time), cube_position=sensor(d,"cube_position").tolist(),
                               cube_quaternion=sensor(d,"cube_orientation").tolist(),
                               action=action.tolist(), command=d.ctrl.tolist()))
        # Training step builds obs BEFORE last_act = action. Hardware computes
        # obs with the preceding inference's action, without this extra lag.
        obs, rng = observation(d, action if deploy else last, rng, noise=not deploy,
                               noise_multiplier=sensor_noise_multiplier)
        last = action
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):
            raise RuntimeError(f"Invalid physics: seed {seed}, tick {tick}, warnings {d.warning.number}")
        if dropped:
            break
    return dict(seed=seed, mode=mode, nominal_keyframe=nominal_keyframe, duration_requested_s=duration,
                sensor_noise_multiplier=sensor_noise_multiplier, extra_observation_latency_steps=extra_latency_steps,
                elapsed_sim_s=float(d.time), retained=not dropped,
                signed_world_z_rotation_rad=rotations, signed_revolutions=rotations/(2*np.pi),
                reward=float(reward), wall_s=time.perf_counter()-start,
                inference_median_ms=float(np.median(inference_ms)),
                inference_p95_ms=float(np.percentile(inference_ms,95)),
                trajectory=trajectory)


def manifest(files, checkpoint=CHECKPOINT):
    checkpoint=Path(checkpoint).resolve()
    return dict(repositories={str(p.relative_to(ROOT)): subprocess.check_output(
        ["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
        for p in (OFFICIAL, PLAYGROUND, MENAGERIE)},
        checkpoint=dict(path=str(checkpoint.relative_to(ROOT)),sha256=sha(checkpoint)),
        checkpoint_origin="released" if checkpoint==CHECKPOINT.resolve() else "local_training",
        assets={str(p.relative_to(ROOT)):sha(p) for p in files},
        packages={p:importlib.metadata.version(p) for p in
                  ("mujoco","mujoco-mjx","brax","jax","jaxlib","flax","numpy")},
        actor_inputs=list(SENSORS)+[f"previous_action_{i}" for i in range(7)],
        cpu_reference=True, hardware=False, fresh_training=False,
        deviations=["CPU MuJoCo rather than MJX: qualification, not matched-engine reproduction",
                    "Nominal physics only; domain randomization is not evaluated here",
                    "Deployment interface uses simulator transmission readings directly; no hardware map or servo dynamics",
                    "Deployment interface starts from the task reset distribution, without manual placement/startup wait"],
        metrics="Signed integral of world-frame omega_z at each physics step; not endpoint quaternion angle")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials",type=int,default=10)
    p.add_argument("--seed",type=int,default=1001)
    p.add_argument("--mode",choices=("training_interface","deployment_interface"),default="training_interface")
    p.add_argument("--seconds",type=float,default=15)
    p.add_argument("--nominal-keyframe",action="store_true",
                   help="Use the exact home keyframe, no randomized start. A deterministic diagnostic, not independent trials.")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,default=CHECKPOINT,
                   help="Trusted released or freshly trained checkpoint; never load an untrusted pickle")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous reports; choose a new output path")
    m, files = make_model()
    policy = ReleasedPolicy(args.checkpoint)
    policy(np.zeros(14,np.float32))  # Compile before measuring.
    report = dict(manifest=manifest(files,args.checkpoint), settings=vars(args)|{"output":str(args.output),"checkpoint":str(args.checkpoint)},
                  scene=dict(cube_size_m=(2*m.geom("cube").size).tolist(),
                             cube_mass_kg=float(m.body_mass[m.body("cube").id]),
                             nominal_controls=m.key("home").ctrl.tolist()), trials=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for seed in range(args.seed,args.seed+args.trials):
        row = run_trial(m,policy,seed,args.mode,args.seconds,args.nominal_keyframe)
        report["trials"].append(row)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k!="trajectory"}),flush=True)
    report["summary"] = dict(retained=sum(r["retained"] for r in report["trials"]),
        trials=args.trials, mean_rotation_rad=float(np.mean([r["signed_world_z_rotation_rad"] for r in report["trials"]])))
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
