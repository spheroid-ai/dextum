"""Can contact-space commands express a lifted thumb + two-finger grasp?

Plain-MuJoCo version of map G at the 20 Hz control rate with ideal pad forces:
engage closes a digit at a fixed rate until its pads report contact, hold
regulates the digit's summed normal force, the thumb's posture (abduction, th1)
is a direct position command. CPU, about a minute.
"""
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

import vendor.aero_official.vision_env  # noqa: F401
from dextum import touch_mjx
from mujoco_playground._src.manipulation.aero_hand.rotate_z import CubeRotateZAxis

m = CubeRotateZAxis().mj_model
touch = touch_mjx.TouchModel(m)
HOME = m.keyframe("home")
LO, HI = m.actuator_ctrlrange.T
if "--official-range" in sys.argv:
    # The official action space: home +/- action_scale, within the actuator range.
    SCALE = np.array([0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012])
    LO, HI = np.maximum(LO, HOME.ctrl - SCALE), np.minimum(HI, HOME.ctrl + SCALE)
NAMES = touch_mjx.PAD_NAMES
DIGIT_PADS = {0: ("index_tip", "index_pad"), 1: ("middle_tip", "middle_pad"), 2: ("ring_tip", "ring_pad"),
              3: ("little_tip", "little_pad"), 6: ("thumb_tip", "thumb_base")}
SUBSTEPS = 5
ENGAGE_RATE = 0.0004      # m per control step (0.02 normalised action units)
HOLD_GAIN = 0.0002        # m per N of force error per control step
MAX_STEP = 0.001


def fresh():
    d = mujoco.MjData(m)
    d.qpos[:] = HOME.qpos
    d.qpos[16:19] = (0.1, 0.0, 0.05)
    d.qpos[19:23] = (1, 0, 0, 0)
    d.ctrl[:] = HOME.ctrl
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    return d


def pad_forces(d, with_points=False):
    out = np.zeros(len(NAMES))
    moment = np.zeros((len(NAMES), 3))
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        p1, p2 = touch.pad_of_geom[c.geom1], touch.pad_of_geom[c.geom2]
        pad = p1 if (p1 >= 0 and touch.object_geom[c.geom2]) else (p2 if (p2 >= 0 and touch.object_geom[c.geom1]) else -1)
        if pad >= 0:
            mujoco.mj_contactForce(m, d, i, f6)
            fn = max(f6[0], 0.0)
            out[pad] += fn
            moment[pad] += fn * c.pos
    if not with_points:
        return out
    cube = m.body("cube").id
    centroid = moment / np.maximum(out, 1e-9)[:, None]
    point_obj = np.where((out > 0)[:, None], (centroid - d.xpos[cube]) @ d.xmat[cube].reshape(3, 3), 0.0)
    return out, point_obj


def grasped(d):
    """Spec N3: >= 3 firm pads (>= 1 N) including the thumb, grasp quality >= 0.3."""
    f, point_obj = pad_forces(d, with_points=True)
    engaged = f > touch_mjx.ENGAGE_N if hasattr(touch_mjx, "ENGAGE_N") else f > 0.3
    reading = touch_mjx.Reading(force=np.zeros((len(NAMES), 3)), f_n=f, f_t=np.zeros(len(NAMES)),
                                centroid=np.zeros((len(NAMES), 3)), point_obj=point_obj,
                                normal_obj=np.zeros((len(NAMES), 3)), slip=np.zeros(len(NAMES), bool),
                                margin=np.zeros(len(NAMES)), engaged=engaged)
    quality = float(touch_mjx.grasp_quality(reading))
    firm = f >= 1.0
    thumb = firm[NAMES.index("thumb_tip")] or firm[NAMES.index("thumb_base")]
    return bool(firm.sum() >= 3 and thumb and quality >= 0.3), quality


def digit_force(f, digit):
    return sum(f[NAMES.index(p)] for p in DIGIT_PADS[digit])


def program(abd, th1, f_thumb, f_finger, fingers=(0, 1), seconds=6.0, thumb_tendon=6, posture=None, trace=False):
    """Thumb (`thumb_tendon`) and `fingers` engage then hold; the other thumb
    tendon and abduction hold `posture`; other fingers stay at home."""
    d = fresh()
    rest_z = d.qpos[18]
    cmd = HOME.ctrl.copy()
    cmd[4] = abd
    other = 5 if thumb_tendon == 6 else 6
    cmd[other] = th1 if posture is None else posture
    cmd = np.clip(cmd, LO, HI)
    DIGIT_PADS[thumb_tendon] = ("thumb_tip", "thumb_base")
    active = {thumb_tendon: f_thumb, **{k: f_finger for k in fingers}}
    mode = {k: "engage" for k in active}
    lifted_for = 0
    for step in range(int(seconds / 0.05)):
        if trace and step % 10 == 0:
            f = pad_forces(d)
            print(f"   t={step*0.05:4.1f} z+{(d.qpos[18]-rest_z)*1000:5.1f}mm palm {f[NAMES.index('palm')]:.2f} " +
                  " ".join(f"{k}:{mode[k][0]}{digit_force(f, k):.1f}/{cmd[k]:.4f}" for k in active))
        f = pad_forces(d)
        for k, target in active.items():
            force = digit_force(f, k)
            if mode[k] == "engage":
                if force > 0.3:
                    mode[k] = "hold"
                else:
                    cmd[k] -= ENGAGE_RATE
            if mode[k] == "hold":
                cmd[k] -= np.clip(HOLD_GAIN * (target - force), -MAX_STEP, MAX_STEP)
            cmd[k] = np.clip(cmd[k], LO[k], HI[k])
        d.ctrl[:] = cmd
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        lift = (d.qpos[18] - rest_z) * 1000
        n3, quality = grasped(d)
        still = np.linalg.norm(d.qvel[16:19]) < 0.01
        lifted_for = lifted_for + 1 if (n3 and lift >= 5 and still) else 0
        if lifted_for >= 20:
            return dict(ok=True, t=round((step + 1) * 0.05, 2), lift=lift, quality=quality)
        if d.qpos[18] < -0.05:
            return dict(ok=False, dropped=True)
    f = pad_forces(d)
    return dict(ok=False, lift=round((d.qpos[18] - rest_z) * 1000, 1), palm=round(f[NAMES.index("palm")], 2),
                thumb=round(digit_force(f, thumb_tendon), 2), fingers=[round(digit_force(f, k), 2) for k in fingers],
                quality=round(grasped(d)[1], 3))


def random_closure(rng):
    """Tendon-space baseline: a slow ramp to a random target, then hold; same success test."""
    d = fresh()
    rest_z = d.qpos[18]
    start, target = d.ctrl.copy(), rng.uniform(LO, HI)
    held = 0
    for step in range(int(6.0 / 0.05)):
        d.ctrl[:] = start + (target - start) * min(1.0, (step + 1) / 20)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        n3, _ = grasped(d)
        lift = (d.qpos[18] - rest_z) * 1000
        held = held + 1 if (n3 and lift >= 5 and np.linalg.norm(d.qvel[16:19]) < 0.01) else 0
        if held >= 20:
            return True
        if d.qpos[18] < -0.05:
            return False
    return False


if "--trace" in sys.argv:
    for args in ((1.2, 0.030, 3.0, 3.0, (1, 2)), (1.2, 0.030, 5.0, 5.0, (0, 1))):
        print("program th2-press", args)
        print("  ->", program(*args, trace=True))
    print("program th1-press, th2 extended (the random-search lift)")
    print("  ->", program(1.23, None, 5.0, 5.0, (1, 2), thumb_tendon=5, posture=0.1108, trace=True))
    sys.exit()

if "--random" in sys.argv:
    rng = np.random.default_rng(1)
    hits = sum(random_closure(rng) for _ in range(1000))
    print(f"tendon-space random closures meeting the E2 test: {hits} / 1000")
    sys.exit()

ok, rows = [], 0
for thumb_tendon, posture_grid in ((6, np.linspace(LO[5], HI[5], 4)), (5, np.linspace(LO[6], HI[6], 4))):
    grid = list(product(np.linspace(max(0.6, LO[4]), min(1.7, HI[4]), 6), posture_grid, (1.0, 3.0, 5.0), (1.0, 3.0, 5.0),
                        ((0, 1), (1, 2), (0, 1, 2), (2, 3))))
    for abd, post, ft, ff, fingers in grid:
        rows += 1
        r = program(abd, None, ft, ff, fingers, thumb_tendon=thumb_tendon, posture=post)
        if r["ok"]:
            ok.append((thumb_tendon, abd, post, ft, ff, fingers, r))
print(f"lifted grasps held 1 s: {len(ok)} / {rows} contact programs")
for tendon in (6, 5):
    times = [r["t"] for t, *_, r in ok if t == tendon]
    print(f"  thumb pressing with tendon {tendon}: {len(times)} / {rows // 2}"
          + (f", median {np.median(times):.2f} s" if times else ""))
for thumb_tendon, abd, post, ft, ff, fingers, r in ok[:20]:
    print(f"  thumb tendon {thumb_tendon} abd {abd:.2f} posture {post:.4f} F_thumb {ft} F_finger {ff} fingers {fingers}: "
          f"success at {r['t']} s, lift {r['lift']:.1f} mm")
