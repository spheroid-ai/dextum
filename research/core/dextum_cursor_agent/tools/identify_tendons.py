"""§5.4 identification in simulation for the no-touch estimator.

Per pressing tendon (index, middle, ring, little, th2), free motion (cube away):
  baseline  quasi-static closing tension vs measured tendon length (16 knots
            over the shared range), nominal physics
  damping   extra tension per m/s of closing speed: engage-rate closing curve
            minus the static one, median over the interior knots
  level     RMS deviation of the free-motion model T(l) + b * closing speed from
            closing sweeps under the training randomisation extremes (joint
            friction x0.5/x2, servo gain x0.8/x1.2, joint damping x0.8/x1.2) and,
            for the thumb, thumb postures; excludes the last 1 mm before the
            extension limit, where commands beyond it make the servo push
  gain      least-squares slope (through zero) of true digit force on positive
            excess tension over the loaded steps of engage/hold programs on the
            resting cube
Writes dextum/identification_v1.json and prints the estimator's detection
quality on those programs. CPU, about a minute.
"""
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from itertools import product
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import mujoco
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import contact_space as cs, touch_mjx
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis

m = CubeRotateZAxis().mj_model
LO, HI = cs.bounds(m.actuator_ctrlrange)
KP = float(m.actuator_gainprm[cs.PRESS[0], 0])
TOUCH = touch_mjx.TouchModel(m)
SUBSTEPS = 5
DT = SUBSTEPS * m.opt.timestep
KNOTS = 16
LENGTH_NOISE = 2e-5      # m, servo-side length measurement (encoder-class resolution)
VARIANTS = [dict()] + [{k: v} for k, v in product(("friction", "gain", "damping"), (0.5, 2.0))]
VARIANTS = [v if "friction" in v else {k: (0.8 if x < 1 else 1.2) for k, x in v.items()} for v in VARIANTS]


@contextmanager
def physics(friction=1.0, gain=1.0, damping=1.0):
    saved = m.dof_frictionloss.copy(), m.actuator_gainprm.copy(), m.actuator_biasprm.copy(), m.dof_damping.copy()
    m.dof_frictionloss[:16] *= friction
    m.actuator_gainprm[:, 0] *= gain
    m.actuator_biasprm[:, 1] *= gain
    m.dof_damping[:16] *= damping
    try:
        yield
    finally:
        m.dof_frictionloss[:], m.actuator_gainprm[:], m.actuator_biasprm[:], m.dof_damping[:] = saved


def fresh(cube_away=True):
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    d.qpos[16:19] = (0.6, 0.6, 0.05) if cube_away else (0.1, 0.0, 0.05)
    d.qpos[19:23] = (1, 0, 0, 0)
    d.ctrl[:] = cs.HOME
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    return d


def control(d, cmd):
    d.ctrl[:] = cmd
    for _ in range(SUBSTEPS):
        mujoco.mj_step(m, d)


def sweep(a, rate_units, posture=None, dwell=0):
    """Close actuator `a` from its open end at `rate_units` normalised units per step.
    Rows: measured length, tension (from the nominal gain, as the estimator sees it), closing speed."""
    d = fresh()
    cmd = cs.HOME.copy()
    if posture is not None:
        cmd[cs.ABD], cmd[cs.TH1] = posture
    cmd[a] = HI[a]
    for _ in range(40):
        control(d, cmd)
    rows = []
    previous = float(d.actuator_length[a])
    while cmd[a] > LO[a]:
        cmd[a] = max(cmd[a] - rate_units * cs.SCALE[a], LO[a])
        control(d, cmd)
        for _ in range(dwell):
            control(d, cmd)
        length = float(d.actuator_length[a])
        rows.append((length, KP * (length - cmd[a]), max(previous - length, 0.0) / (DT * (1 + dwell))))
        previous = length
    return np.array(rows)


def interior(rows):
    return rows[rows[:, 0] < rows[:, 0].max() - 1e-3]


def digit_forces(d):
    f = np.zeros(touch_mjx.N_PADS)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        p1, p2 = TOUCH.pad_of_geom[c.geom1], TOUCH.pad_of_geom[c.geom2]
        pad = p1 if (p1 >= 0 and TOUCH.object_geom[c.geom2]) else (p2 if (p2 >= 0 and TOUCH.object_geom[c.geom1]) else -1)
        if pad >= 0:
            mujoco.mj_contactForce(m, d, i, f6)
            f[pad] += max(f6[0], 0.0)
    return cs.MEMBERSHIP @ f


def free_model(t, length, speed):
    return np.interp(length, t["knots"], t["baseline_n"]) + t["damping_ns_per_m"] * speed


def loaded_samples(out):
    """(digit, excess tension, true force) over engage/hold programs on the resting cube."""
    rows = []
    for variant, abd, force, fingers in product(VARIANTS[:3], (0.6, 0.9, 1.2), (1.0, 3.0, 5.0),
                                                ((0, 1), (1, 2), (2, 3), (0, 1, 2, 3))):
        with physics(**variant):
            d = fresh(cube_away=False)
            cmd = cs.HOME.copy()
            cmd[cs.ABD], cmd[cs.TH1] = abd, LO[cs.TH1]
            active = (4, *fingers)
            previous = d.actuator_length.copy()
            for _ in range(100):
                f = digit_forces(d)
                for k in active:
                    a = cs.PRESS[k]
                    if f[k] > 0.3:
                        cmd[a] -= np.clip(cs.HOLD_GAIN * (force - f[k]), -cs.HOLD_MAX, cs.HOLD_MAX)
                    else:
                        cmd[a] -= cs.ENGAGE_RATE * cs.SCALE[a]
                cmd = np.clip(cmd, LO, HI)
                control(d, cmd)
                f = digit_forces(d)
                for k in range(cs.N_DIGITS):
                    a = int(cs.PRESS[k])
                    length = float(d.actuator_length[a])
                    speed = max(previous[a] - length, 0.0) / DT
                    excess = KP * (length - cmd[a]) - free_model(out["tendons"][str(a)], length, speed)
                    rows.append((k, excess, f[k]))
                previous = d.actuator_length.copy()
    return np.array(rows)


def main():
    out = dict(source=dict(script=str(Path(__file__).relative_to(HERE)),
                           sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
               kp_nominal=KP, length_noise_m=LENGTH_NOISE, control_dt=DT, variants=VARIANTS, tendons={})
    for a in cs.PRESS:
        a = int(a)
        knots = np.linspace(LO[a], HI[a], KNOTS)
        static = interior(sweep(a, 0.005, dwell=3))
        order = np.argsort(static[:, 0])
        baseline = np.interp(knots, static[order, 0], static[order, 1])
        engage = interior(sweep(a, cs.ENGAGE_RATE))
        moving = engage[:, 2] > 0.5 * cs.ENGAGE_RATE * cs.SCALE[a] / DT
        extra = engage[moving, 1] - np.interp(engage[moving, 0], knots, baseline)
        damping = float(np.median(extra / engage[moving, 2])) if moving.any() else 0.0
        t = dict(knots=knots.tolist(), baseline_n=baseline.tolist(), damping_ns_per_m=max(damping, 0.0))
        deviations = []
        postures = [None]
        if a == cs.PRESS[4]:
            postures += list(product((LO[cs.ABD], cs.HOME[cs.ABD], HI[cs.ABD]), (LO[cs.TH1], HI[cs.TH1])))
        for variant, posture in product(VARIANTS, postures):
            with physics(**variant):
                for rate, dwell in ((0.005, 3), (cs.ENGAGE_RATE, 0)):
                    rows = interior(sweep(a, rate, posture=posture, dwell=dwell))
                    deviations.append(rows[:, 1] - free_model(t, rows[:, 0], rows[:, 2]))
        dev = np.concatenate(deviations)
        t.update(level_n=float(np.sqrt(np.mean(dev ** 2))), p99_deviation_n=float(np.percentile(np.abs(dev), 99)))
        out["tendons"][str(a)] = t
    samples = loaded_samples(out)
    for k in range(cs.N_DIGITS):
        a = int(cs.PRESS[k])
        t = out["tendons"][str(a)]
        s = samples[samples[:, 0] == k]
        threshold = 3.0 * t["level_n"]
        use = (s[:, 2] > 0.3) & (s[:, 1] > 0.0)
        e, f = s[use, 1], s[use, 2]
        gain = float(np.sum(e * f) / max(np.sum(e * e), 1e-9)) if use.sum() else 0.0
        touching = s[:, 2] > 0.3
        detected = s[:, 1] > threshold
        estimate = np.where(detected, gain * np.maximum(s[:, 1], 0.0), 0.0)
        t.update(gain=gain, contact_steps=int(touching.sum()), free_steps=int((~touching).sum()),
                 detected_when_touching=float(detected[touching].mean()) if touching.any() else 0.0,
                 detected_when_free=float(detected[~touching].mean()) if (~touching).any() else 0.0,
                 detected_above_1n=float(detected[s[:, 2] > 1.0].mean()) if (s[:, 2] > 1.0).any() else 0.0,
                 rms_error_n_touching=float(np.sqrt(np.mean((estimate - s[:, 2])[touching] ** 2))) if touching.any() else 0.0,
                 correlation=float(np.corrcoef(s[touching, 1], s[touching, 2])[0, 1]) if touching.sum() > 2 else 0.0)
    path = HERE / "dextum/identification_v1.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for k, name in enumerate(cs.DIGITS):
        t = out["tendons"][str(int(cs.PRESS[k]))]
        print(f"{name:6s}: free {min(t['baseline_n']):5.1f}..{max(t['baseline_n']):5.1f} N, damping {t['damping_ns_per_m']:6.0f} N s/m, "
              f"level {t['level_n']:.2f} N (p99 {t['p99_deviation_n']:.1f}), gain {t['gain']:.3f}; "
              f"detects {t['detected_when_touching']:.2f} of contacts ({t['detected_above_1n']:.2f} above 1 N), "
              f"{t['detected_when_free']:.2f} false; force rms {t['rms_error_n_touching']:.2f} N; corr {t['correlation']:.2f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
