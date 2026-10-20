"""Map G for the Aero (spec §5.2) with the reflex (§5.5) and the no-touch estimator (§5.3).

Contact-space action, 31 values in [-1, 1]. For each digit (index, middle, ring,
little, thumb): 4 mode scores (free, engage, hold, release; the largest wins),
a force target (0-10 N) and a slide; then a second thumb slide.

A digit presses by shortening one tendon (its finger tendon; th2 for the thumb).
Slides move a pad along the only arcs the Aero has: the flexion arc of a finger
(free mode only, since a pressing finger has no spare degree of freedom) and,
for the thumb, abduction (slide 0) and th1 (slide 1) in every mode. G runs once
per 20 Hz control step, the hardware loop rate.
"""
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import touch_mjx
from .contact import ENGAGE_N, MU_INIT, SAFETY_FACTOR

DIGITS = ("index", "middle", "ring", "little", "thumb")
N_DIGITS = len(DIGITS)
PRESS = np.array([0, 1, 2, 3, 6])        # actuator that presses each digit
ABD, TH1 = 4, 5
DIGIT_PADS = (("index_tip", "index_pad"), ("middle_tip", "middle_pad"), ("ring_tip", "ring_pad"),
              ("little_tip", "little_pad"), ("thumb_tip", "thumb_base"))
MEMBERSHIP = np.zeros((N_DIGITS, touch_mjx.N_PADS), np.float32)
for _d, _pads in enumerate(DIGIT_PADS):
    for _p in _pads:
        MEMBERSHIP[_d, touch_mjx.PAD_NAMES.index(_p)] = 1.0

FREE, ENGAGE, HOLD, RELEASE = range(4)
N_MODES = 4
PER_DIGIT = N_MODES + 2
ACTION_SIZE = N_DIGITS * PER_DIGIT + 1
STATE_FEATURES = N_DIGITS * (N_MODES + 3)

# Official action space: command = HOME + SCALE * a (metres on cables, radians on abduction).
HOME = np.array([0.09, 0.09, 0.09, 0.09, 0.75, 0.035, 0.1], np.float32)
SCALE = np.array([0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012], np.float32)

F_MAX = 10.0              # N, force target at a = +1
F_ENGAGE = 0.5            # N held by an engaged digit once in contact
ENGAGE_RATE = 0.02        # normalised units per step (spec §5.2)
RELEASE_RATE = 0.02
SLIDE_RATE = 0.05         # normalised units per step at |slide| = 1
HOLD_GAIN = 2e-4          # m of tendon per N of force error per step
HOLD_MAX = 1e-3           # m per step
BOOST_MIN, BOOST_FRAC = 0.5, 0.3
BOOST_HOLD, BOOST_RELAX = 6, 20     # 0.3 s and 1 s at 20 Hz
NOT_SLIPPED = 10_000

IDENTIFICATION = Path(__file__).with_name("identification_v1.json")


def bounds(ctrl_range):
    """Shared command bounds: the official action range within the actuator range."""
    lo, hi = np.asarray(ctrl_range, np.float32).T
    return np.maximum(lo, HOME - SCALE), np.minimum(hi, HOME + SCALE)


class GState(NamedTuple):
    cmd: jnp.ndarray          # (7,) actuator command
    mode: jnp.ndarray         # (5,) int32
    f_ref: jnp.ndarray        # (5,) force reference of the last step (N)
    boost0: jnp.ndarray       # (5,) slip boost at the last slip onset (N)
    since_slip: jnp.ndarray   # (5,) int32 steps since that onset
    mu: jnp.ndarray           # (5,) friction estimate
    slip: jnp.ndarray         # (5,) bool, slip flags of the last step


class Sense(NamedTuple):
    """What G measures per digit: pad readings with touch, tendon estimates without."""
    force: jnp.ndarray        # (5,) N
    shear: jnp.ndarray        # (5,) N, zero without touch
    slip: jnp.ndarray         # (5,) bool, False without touch
    contact: jnp.ndarray      # (5,) bool


def initial_state():
    z = jnp.zeros(N_DIGITS)
    return GState(cmd=jnp.asarray(HOME), mode=jnp.zeros(N_DIGITS, jnp.int32), f_ref=z, boost0=z,
                  since_slip=jnp.full(N_DIGITS, NOT_SLIPPED, jnp.int32), mu=jnp.full(N_DIGITS, MU_INIT),
                  slip=jnp.zeros(N_DIGITS, bool))


def no_sense():
    z = jnp.zeros(N_DIGITS)
    return Sense(force=z, shear=z, slip=jnp.zeros(N_DIGITS, bool), contact=jnp.zeros(N_DIGITS, bool))


def sense_touch(reading):
    m = jnp.asarray(MEMBERSHIP)
    force = m @ reading.f_n
    return Sense(force=force, shear=m @ reading.f_t, slip=(m @ reading.slip.astype(jnp.float32)) > 0,
                 contact=force > ENGAGE_N)


class TendonEstimator:
    """§5.3 from the §5.4 identification: excess tension over the free-motion model
    T_free(l) + b * closing speed; contact above 3x the free-motion level."""

    def __init__(self, path=IDENTIFICATION):
        ident = json.loads(Path(path).read_text())
        self.kp = float(ident["kp_nominal"])
        self.noise = float(ident["length_noise_m"])
        self.dt = float(ident["control_dt"])
        tendons = [ident["tendons"][str(a)] for a in PRESS]
        self.knots = jnp.asarray([t["knots"] for t in tendons], jnp.float32)
        self.baseline = jnp.asarray([t["baseline_n"] for t in tendons], jnp.float32)
        self.damping = jnp.asarray([t["damping_ns_per_m"] for t in tendons], jnp.float32)
        self.threshold = jnp.asarray([3.0 * t["level_n"] for t in tendons], jnp.float32)
        self.gain = jnp.asarray([t["gain"] for t in tendons], jnp.float32)

    def free_tension(self, length, speed):
        static = jnp.stack([jnp.interp(length[d], self.knots[d], self.baseline[d]) for d in range(N_DIGITS)])
        return static + self.damping * speed

    def __call__(self, actuator_length, previous_length, cmd, key):
        """Returns (sense, measured pressing-tendon lengths for the next call)."""
        length = actuator_length[PRESS] + self.noise * jax.random.normal(key, (N_DIGITS,))
        speed = jnp.maximum(previous_length - length, 0.0) / self.dt
        excess = self.kp * (length - cmd[PRESS]) - self.free_tension(length, speed)
        contact = excess > self.threshold
        force = jnp.where(contact, self.gain * jnp.maximum(excess, 0.0), 0.0)
        z = jnp.zeros(N_DIGITS)
        return Sense(force=force, shear=z, slip=jnp.zeros(N_DIGITS, bool), contact=contact), length


def decode(action):
    per = action[:N_DIGITS * PER_DIGIT].reshape(N_DIGITS, PER_DIGIT)
    return dict(mode=jnp.argmax(per[:, :N_MODES], axis=-1).astype(jnp.int32),
                f_target=F_MAX * (jnp.clip(per[:, N_MODES], -1.0, 1.0) + 1.0) / 2.0,
                slide=jnp.clip(per[:, N_MODES + 1], -1.0, 1.0),
                thumb_slide=jnp.clip(action[-1], -1.0, 1.0))


def step(state, action, sense, lo, hi, reflex=True):
    """One control step: (state, contact command, measurements) -> (state, actuator command)."""
    a = decode(action)
    mode = a["mode"]
    pressing = (mode == ENGAGE) | (mode == HOLD)

    # Reflex (§5.5): friction-cone floor and slip boost on engaged digits; suspended when not pressing.
    onset = sense.slip & ~state.slip & sense.contact & pressing & reflex
    ratio = sense.shear / jnp.maximum(sense.force, 1e-6)
    mu = jnp.where(onset, jnp.minimum(state.mu, 0.7 * state.mu + 0.3 * ratio), state.mu)
    mu = jnp.maximum(mu, 0.05)
    since = jnp.where(onset, 0, jnp.minimum(state.since_slip + 1, NOT_SLIPPED))
    boost0 = jnp.where(onset, jnp.maximum(BOOST_MIN, BOOST_FRAC * state.f_ref), state.boost0)
    boost = boost0 * jnp.clip(1.0 - (since - BOOST_HOLD) / BOOST_RELAX, 0.0, 1.0)
    cone = sense.shear / (mu * (1.0 - SAFETY_FACTOR))
    base = jnp.where(mode == HOLD, a["f_target"], F_ENGAGE)
    guarded = pressing & sense.contact & reflex
    f_ref = jnp.where(guarded, jnp.maximum(base, cone) + boost, base)
    boost0 = jnp.where(pressing, boost0, 0.0)

    s = jnp.asarray(SCALE)[PRESS]
    regulate = -jnp.clip(HOLD_GAIN * (f_ref - sense.force), -HOLD_MAX, HOLD_MAX)
    closing = -ENGAGE_RATE * s
    d_press = jnp.select(
        [mode == FREE, pressing, mode == RELEASE],
        [-SLIDE_RATE * a["slide"] * s, jnp.where(sense.contact, regulate, closing), RELEASE_RATE * s])

    cmd = state.cmd.at[PRESS].add(d_press)
    cmd = cmd.at[ABD].add(SLIDE_RATE * a["slide"][4] * SCALE[ABD])
    cmd = cmd.at[TH1].add(-SLIDE_RATE * a["thumb_slide"] * SCALE[TH1])
    cmd = jnp.clip(cmd, lo, hi)
    new = GState(cmd=cmd, mode=mode, f_ref=jnp.where(pressing, f_ref, 0.0), boost0=boost0,
                 since_slip=since, mu=mu, slip=sense.slip)
    return new, cmd


def features(state, sense):
    """(35,) actor inputs: mode one-hot, force reference / 10 N, measured force / 5 N, contact flag."""
    onehot = (state.mode[:, None] == jnp.arange(N_MODES)[None]).astype(jnp.float32)
    x = jnp.concatenate((onehot, (state.f_ref / F_MAX)[:, None], jnp.clip(sense.force / 5.0, 0.0, 5.0)[:, None],
                         sense.contact[:, None].astype(jnp.float32)), axis=-1)
    return x.reshape(-1)


def command_action(mode, f_target=None, slide=None, thumb_slide=0.0):
    """Inverse of `decode` for scripted programs and tests (numpy)."""
    mode = np.asarray(mode)
    f_target = np.zeros(N_DIGITS) if f_target is None else np.asarray(f_target, float)
    slide = np.zeros(N_DIGITS) if slide is None else np.asarray(slide, float)
    per = np.full((N_DIGITS, PER_DIGIT), -1.0, np.float32)
    per[np.arange(N_DIGITS), mode] = 1.0
    per[:, N_MODES] = 2.0 * f_target / F_MAX - 1.0
    per[:, N_MODES + 1] = slide
    return np.concatenate((per.reshape(-1), [thumb_slide])).astype(np.float32)
