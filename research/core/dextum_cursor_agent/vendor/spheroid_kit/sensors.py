"""Simulated six-pad tactile kit, independently of task and controller.

Only the 60-value vector returned by ``native``/``mjx`` and subsequently
``advance_degradation`` is an actor input. Contact helpers and degradation state
are simulator internals. Force is force exerted by the pad on its environment:
the pad's outward normal therefore has positive compression at its front face.
Gyro and accelerometer are MuJoCo site sensors, with accelerometer specific force
``a - gravity`` (a stationary horizontal pad reads +9.81 along world-up).
"""
from dataclasses import asdict, dataclass
import math

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src import support
import numpy as np


@dataclass(frozen=True)
class Pad:
    name: str
    body: str
    site: str
    geoms: tuple[str, ...]
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    fingertip: bool = True


_PAD_QUAT = (math.sqrt(.5), math.sqrt(.5), 0., 0.)
AERO_PADS = tuple(
    Pad(name, f"right_{name}_distal_link", f"kit_{name}", (geom,),
        (0., -.008, .015 if name == "thumb" else .014), _PAD_QUAT)
    for name, geom in zip(("index", "middle", "ring", "pinky", "thumb"),
                          ("if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip"))
) + (Pad("palm", "palm", "kit_palm",
         tuple(f"palm_collision_{i}" for i in range(1, 12)),
         # Outward (+local-x) surface center of palm_collision_7. Its
         # orientation is a 15-degree slope relative to the palm body.
         (.002, -.0208568189, .0763422906),
         (.79356082, .60849094, 0., 0.), False),)

KIT_SIZE = 60
FORCE_INDICES = np.array([[i * 11 + k for k in range(3)] for i in range(5)] + [[55, 56, 57]])
LOCATION_INDICES = np.array([[i * 11 + k for k in (3, 4)] for i in range(5)] + [[58, 59]])
GYRO_INDICES = np.array([[i * 11 + k for k in (5, 6, 7)] for i in range(5)])
ACCEL_INDICES = np.array([[i * 11 + k for k in (8, 9, 10)] for i in range(5)])


def add_pads_to_spec(spec, pads):
    """Append observation sites/sensors; adds no collision or inertial elements."""
    for pad in pads:
        if spec.site(pad.site) is not None:
            raise ValueError(f"Pad site already exists: {pad.site}")
        spec.body(pad.body).add_site(name=pad.site, pos=pad.position,
                                   quat=pad.quaternion, size=[.001, .001, .001], group=4)
        if pad.fingertip:
            for suffix, sensor_type in (("gyro", mujoco.mjtSensor.mjSENS_GYRO),
                                       ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
                spec.add_sensor(name=f"{pad.site}_{suffix}", type=sensor_type,
                                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=pad.site)
    return spec


def add_aero_kit_to_spec(spec):
    return add_pads_to_spec(spec, AERO_PADS)


def _descendant_bodies(model, root):
    answer = np.zeros(model.nbody, dtype=bool)
    for body in range(model.nbody):
        current = body
        while current != 0 and current != root:
            current = int(model.body_parentid[current])
        answer[body] = current == root
    return answer


def contact_forces_mjx(model, data):
    """Contact-frame wrench for each static MJX contact slot, inactive => zero.

    MuJoCo 3.3.6 stores dimension/constraint addresses as static NumPy arrays.
    The upstream grouped decoder handles both pyramidal and elliptic cones.
    """
    contacts = data._impl.contact
    forces = jnp.zeros((contacts.pos.shape[0], 6), dtype=data.qpos.dtype)
    for dim in np.unique(contacts.dim):
        wrench, ids = support.contact_force_dim(model, data, int(dim))
        forces = forces.at[ids].set(wrench)
    return forces


class KitSensors:
    """Static geometry/site binding. Construct once, then use under jit/vmap."""

    def __init__(self, model, pads=AERO_PADS, hand_root="tetheria_mount"):
        self.model = model
        self.pads = tuple(pads)
        root = model.body(hand_root).id if isinstance(hand_root, str) else hand_root
        self.hand_bodies = _descendant_bodies(model, int(root))
        self.hand_geoms = self.hand_bodies[model.geom_bodyid]
        self.site_ids = tuple(model.site(pad.site).id for pad in self.pads)
        self.pad_geoms = tuple(np.array([model.geom(name).id for name in pad.geoms], dtype=int)
                               for pad in self.pads)
        self.pad_masks = tuple(np.isin(np.arange(model.ngeom), geoms) for geoms in self.pad_geoms)
        self.motion_indices = []
        for pad in self.pads:
            if pad.fingertip:
                indices = []
                for suffix in ("gyro", "accel"):
                    sensor = model.sensor(f"{pad.site}_{suffix}")
                    address = int(sensor.adr[0])
                    if int(sensor.dim[0]) != 3:
                        raise ValueError("Expected a three-axis site sensor")
                    indices.extend(range(address, address + 3))
                self.motion_indices.append(np.array(indices))
            else:
                self.motion_indices.append(np.array([], dtype=int))
        all_geoms = np.concatenate(self.pad_geoms)
        if np.unique(all_geoms).size != all_geoms.size:
            raise ValueError("Collision geometry assigned to more than one pad")
        if not np.all(self.hand_geoms[all_geoms]):
            raise ValueError("Pad collision geometry is not part of the hand")
        self.size = sum(11 if pad.fingertip else 5 for pad in self.pads)

    def manifest(self):
        offset = 0
        pads = []
        for pad, geoms, site_id in zip(self.pads, self.pad_geoms, self.site_ids):
            size = 11 if pad.fingertip else 5
            pads.append(dict(**asdict(pad), site_id=int(site_id), geom_ids=geoms.tolist(),
                             offset=offset, size=size))
            offset += size
        return dict(pads=pads, size=self.size, force_units="N", location_units="m",
                    gyro_units="rad/s", accelerometer_units="m/s^2",
                    force_convention="pad-on-environment, outward normal compression positive",
                    centroid_weight="contact normal force magnitude",
                    accelerometer_convention="specific force a-gravity, stationary=-R.T@gravity",
                    sensor_bundle="30 force/contact-location channels plus30 fingertip IMU channels",
                    physical_imu_status="unconfirmed; ordered magnetic force sensors do not establish IMU availability",
                    physical_contact_location_status="unconfirmed; contact centroid readout needs hardware calibration/support",
                    force_only_claim="requires a separate force/location-only ablation",
                    coverage="whole listed rigid collision geometries; no unmodeled soft skin",
                    palm_shortcut="one lumped measurement over eleven palm collision geometries",
                    self_contacts_excluded=True)

    def _native_contacts(self, data):
        count = data.ncon
        force = np.zeros((count, 6))
        for i in range(count):
            mujoco.mj_contactForce(self.model, data, i, force[i])
        return (np.asarray(data.contact.geom[:count]), np.asarray(data.contact.pos[:count]),
                np.asarray(data.contact.frame[:count]).reshape(-1, 3, 3), force)

    def native(self, data):
        geom, points, frame, forces = self._native_contacts(data)
        output = []
        for site_id, mask, motion in zip(self.site_ids, self.pad_masks, self.motion_indices):
            rot = data.site_xmat[site_id].reshape(3, 3)
            first = mask[geom[:, 0]] & ~self.hand_geoms[geom[:, 1]]
            second = mask[geom[:, 1]] & ~self.hand_geoms[geom[:, 0]]
            sign = first.astype(float) - second.astype(float)
            world = np.einsum("ci,cij->cj", forces[:, :3], frame) * sign[:, None]
            force = world.sum(axis=0) @ rot
            weights = forces[:, 0] * (first | second)
            point = (points - data.site_xpos[site_id]) @ rot
            centroid = (point[:, :2] * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
            output.append(np.concatenate((force, centroid, data.sensordata[motion])))
        return np.concatenate(output).astype(np.float32)

    def mjx(self, model, data):
        contact = data._impl.contact
        forces = contact_forces_mjx(model, data)
        geom = contact.geom
        valid = jnp.all(geom >= 0, axis=-1)
        geom = jnp.maximum(geom, 0)
        hand = jnp.asarray(self.hand_geoms)
        output = []
        for site_id, mask, motion in zip(self.site_ids, self.pad_masks, self.motion_indices):
            rot = data.site_xmat[site_id]
            mask = jnp.asarray(mask)
            first = mask[geom[:, 0]] & ~hand[geom[:, 1]] & valid
            second = mask[geom[:, 1]] & ~hand[geom[:, 0]] & valid
            sign = first.astype(forces.dtype) - second.astype(forces.dtype)
            world = jnp.einsum("ci,cij->cj", forces[:, :3], contact.frame) * sign[:, None]
            force = jnp.sum(world, axis=0) @ rot
            weights = forces[:, 0] * (first | second)
            point = (contact.pos - data.site_xpos[site_id]) @ rot
            centroid = jnp.sum(point[:, :2] * weights[:, None], axis=0) / jnp.maximum(jnp.sum(weights), 1e-12)
            output.append(jnp.concatenate((force, centroid, data.sensordata[motion])))
        return jnp.concatenate(output)

    def _object_mask(self, object_body_id):
        return (_descendant_bodies(self.model, int(object_body_id))[self.model.geom_bodyid]
                & ~self.hand_geoms)

    def object_normal_force_native(self, data, object_body_id):
        """Privileged reward/evaluation only, never an observation block."""
        geom, _, _, forces = self._native_contacts(data)
        obj = self._object_mask(object_body_id)
        selected = ((self.hand_geoms[geom[:, 0]] & obj[geom[:, 1]]) |
                    (self.hand_geoms[geom[:, 1]] & obj[geom[:, 0]]))
        return float(np.sum(forces[:, 0] * selected))

    def object_normal_force_mjx(self, model, data, object_body_id):
        """Privileged reward/evaluation only, never an observation block."""
        geom = data._impl.contact.geom
        valid = jnp.all(geom >= 0, axis=-1)
        geom = jnp.maximum(geom, 0)
        hand, obj = jnp.asarray(self.hand_geoms), jnp.asarray(self._object_mask(object_body_id))
        selected = ((hand[geom[:, 0]] & obj[geom[:, 1]]) |
                    (hand[geom[:, 1]] & obj[geom[:, 0]])) & valid
        return jnp.sum(contact_forces_mjx(model, data)[:, 0] * selected)


@dataclass(frozen=True)
class Quality:
    force_noise: float
    gain_error: float
    bias_bound: float
    cross_talk: float
    saturation: float
    detection_floor: float
    delay_seconds: float
    requested_hz: float
    location_noise: float
    gyro_noise: float
    accel_noise: float


QUALITIES = {
    "ideal": Quality(0., 0., 0., 0., math.inf, 0., 0., math.inf, 0., 0., 0.),
    "realistic": Quality(.05, .05, .1, .03, 20., .3, .01, 200., .001, .02, .1),
    "poor": Quality(.2, .15, .5, .10, 10., .5, .03, 50., .003, .1, .5),
}


def sensor_timing(quality="realistic", physics_dt=.01):
    q = QUALITIES[quality]
    if physics_dt <= 0:
        raise ValueError("physics_dt must be positive")
    interval = max(1, int(math.ceil(1. / (q.requested_hz * physics_dt))))
    delay = int(math.ceil(q.delay_seconds / physics_dt - 1e-9))
    return dict(physics_dt=physics_dt, physics_hz=1 / physics_dt,
                requested_hz=None if math.isinf(q.requested_hz) else q.requested_hz,
                achieved_hz=1 / (interval * physics_dt), update_interval_steps=interval,
                delay_steps=delay, achieved_delay_seconds=delay * physics_dt,
                sample_hold="no interpolation; fresh values require a physics tick")


def initial_degradation(key, quality="realistic", physics_dt=.01):
    """Private sensor memory; no hidden object property appears in this API."""
    q = QUALITIES[quality]
    timing = sensor_timing(quality, physics_dt)
    key, gain_key, cross_key = jax.random.split(key, 3)
    gain = 1. + jax.random.uniform(gain_key, (6, 3), minval=-q.gain_error, maxval=q.gain_error)
    cross = jax.random.uniform(cross_key, (6, 3, 3), minval=-q.cross_talk, maxval=q.cross_talk)
    cross = cross * (1. - jnp.eye(3)[None]) + jnp.eye(3)[None]
    return dict(key=key, gain=gain, cross=cross, bias=jnp.zeros((6, 3)),
                held=jnp.zeros(KIT_SIZE), queue=jnp.zeros((timing["delay_steps"] + 1, KIT_SIZE)),
                tick=jnp.array(0, jnp.int32))


def advance_degradation(state, raw, quality="realistic", physics_dt=.01):
    """Call exactly once after each physics step, not once per policy decision.

    Detection uses the *measured* force norm, never exact contact existence.
    Bias is a bounded random walk (per-second sigma is 0.2 times its bound).
    Delay applies to all channels. Poor quality samples at 50 Hz and holds its
    measurement between samples; realistic can achieve only 100 Hz here.
    """
    if raw.shape != (KIT_SIZE,):
        raise ValueError(f"Expected exactly {KIT_SIZE} kit channels")
    q = QUALITIES[quality]
    timing = sensor_timing(quality, physics_dt)
    key, drift_key, force_key, location_key, gyro_key, accel_key = jax.random.split(state["key"], 6)
    tick = state["tick"] + 1
    update = tick % timing["update_interval_steps"] == 0
    sample_dt = physics_dt * timing["update_interval_steps"]
    bias = jnp.clip(state["bias"] + .2 * q.bias_bound * math.sqrt(sample_dt) *
                    jax.random.normal(drift_key, (6, 3)), -q.bias_bound, q.bias_bound)
    bias = jnp.where(update, bias, state["bias"])
    force = jnp.einsum("pij,pj->pi", state["cross"], raw[FORCE_INDICES] * state["gain"])
    force = force + bias + q.force_noise * jax.random.normal(force_key, (6, 3))
    force = jnp.clip(force, -q.saturation, q.saturation)
    engaged = jnp.linalg.norm(force, axis=-1) >= q.detection_floor
    # At ideal quality, an exactly unloaded channel and its centroid stay zero.
    engaged = engaged & (jnp.linalg.norm(force, axis=-1) > 0.)
    force = jnp.where(engaged[:, None], force, 0.)
    location = raw[LOCATION_INDICES] + q.location_noise * jax.random.normal(location_key, (6, 2))
    location = jnp.where(engaged[:, None], location, 0.)
    sample = raw.at[FORCE_INDICES].set(force).at[LOCATION_INDICES].set(location)
    sample = sample.at[GYRO_INDICES].add(q.gyro_noise * jax.random.normal(gyro_key, (5, 3)))
    sample = sample.at[ACCEL_INDICES].add(q.accel_noise * jax.random.normal(accel_key, (5, 3)))
    held = jnp.where(update, sample, state["held"])
    queue = jnp.concatenate((state["queue"][1:], held[None]), axis=0)
    return dict(state, key=key, bias=bias, held=held, queue=queue, tick=tick), queue[0]
