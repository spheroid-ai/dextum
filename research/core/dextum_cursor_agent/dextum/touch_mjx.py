"""Ideal per-pad touch readings for the Aero hand, computed inside MJX.

Every hand collision geom belongs to exactly one lumped pad, so no hand-object
force is invisible. Directional quantities are in the pad's primary body frame
(what a body-mounted sensor measures). Forces are exerted by the pad on the
object. All arrays are static-shape, so the model works under jit/vmap.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from mujoco.mjx._src import support

from .contact import CUBE_HALF_EDGE_M, ENGAGE_N, cube_reference_sigma

# Primary (frame) body first.
PAD_BODIES = (
    ("thumb_tip", ("right_thumb_distal_link",)),
    ("index_tip", ("right_index_distal_link",)),
    ("middle_tip", ("right_middle_distal_link",)),
    ("ring_tip", ("right_ring_distal_link",)),
    ("little_tip", ("right_pinky_distal_link",)),
    ("index_pad", ("right_index_middle_link", "right_index_proximal_link")),
    ("middle_pad", ("right_middle_middle_link", "right_middle_proximal_link")),
    ("ring_pad", ("right_ring_middle_link", "right_ring_proximal_link")),
    ("little_pad", ("right_pinky_middle_link", "right_pinky_proximal_link")),
    ("thumb_base", ("right_thumb_proximal_link", "right_thumb_mcp_link", "right_t_link")),
    ("palm", ("palm", "tetheria_mount")),
)
PAD_NAMES = tuple(name for name, _ in PAD_BODIES)
N_PADS = len(PAD_NAMES)

SLIP_SPEED = 0.01      # m/s tangential relative speed at a loaded contact
SLIP_MIN_FORCE = 0.05  # N; below this a contact cannot meaningfully slip
FORCE_SCALE = 1.0      # N per unit input
LOCATION_SCALE = 0.02  # m per unit input
AGE_STEPS = 20         # policy steps for a saturated contact age

ACTOR_FIELDS = ("fx", "fy", "fz", "f_n", "cx", "cy", "cz", "slip", "engaged", "age")
CRITIC_FIELDS = ("fx", "fy", "fz", "f_n", "slip", "margin", "engaged")


def _descendants(model, root):
    out = np.zeros(model.nbody, dtype=bool)
    for body in range(model.nbody):
        current = body
        while current not in (0, root):
            current = int(model.body_parentid[current])
        out[body] = current == root
    return out


class Reading(NamedTuple):
    """Per-pad readings, each of leading shape (N_PADS,); a JAX pytree."""
    force: jnp.ndarray       # (P, 3) pad frame, pad-on-object
    f_n: jnp.ndarray         # (P,) summed normal force
    f_t: jnp.ndarray         # (P,) summed tangential force magnitude
    centroid: jnp.ndarray    # (P, 3) pad frame, force-weighted, zero if unloaded
    point_obj: jnp.ndarray   # (P, 3) centroid in object frame
    normal_obj: jnp.ndarray  # (P, 3) mean pad-to-object normal in object frame
    slip: jnp.ndarray        # (P,) bool
    margin: jnp.ndarray      # (P,) min friction-cone margin of loaded contacts (N)
    engaged: jnp.ndarray     # (P,) bool, f_n > ENGAGE_N


class TouchModel:
    def __init__(self, mj_model, object_body="cube", hand_root="tetheria_mount"):
        m = mj_model
        self.object_body = m.body(object_body).id
        collide = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        hand = _descendants(m, m.body(hand_root).id)
        pad_of_geom = -np.ones(m.ngeom, dtype=int)
        self.frame_bodies = []
        for p, (_, bodies) in enumerate(PAD_BODIES):
            ids = [m.body(b).id for b in bodies]
            self.frame_bodies.append(ids[0])
            for g in np.where(collide & np.isin(m.geom_bodyid, ids))[0]:
                pad_of_geom[g] = p
        uncovered = np.where(collide & hand[m.geom_bodyid] & (pad_of_geom < 0))[0]
        if uncovered.size:
            raise ValueError(f"Hand collision geoms without a pad: {uncovered.tolist()}")
        self.pad_of_geom = pad_of_geom
        self.object_geom = collide & (m.geom_bodyid == self.object_body)
        self.geom_body = np.asarray(m.geom_bodyid)
        self.body_root = np.asarray(m.body_rootid)
        self.frame_bodies = np.array(self.frame_bodies)
        self.pad_geom_counts = np.bincount(pad_of_geom[pad_of_geom >= 0], minlength=N_PADS)

    def read(self, model, data):
        contact = data._impl.contact
        geom = jnp.maximum(contact.geom, 0)
        active = jnp.all(contact.geom >= 0, axis=-1)
        force = jnp.zeros((contact.pos.shape[0], 6), dtype=data.qpos.dtype)
        for dim in np.unique(contact.dim):
            wrench, ids = support.contact_force_dim(model, data, int(dim))
            force = force.at[ids].set(wrench)
        f = force[:, :3]

        pad_of_geom = jnp.asarray(self.pad_of_geom)
        obj = jnp.asarray(self.object_geom)
        pad1, pad2 = pad_of_geom[geom[:, 0]], pad_of_geom[geom[:, 1]]
        first = (pad1 >= 0) & obj[geom[:, 1]] & active      # pad is geom1
        second = (pad2 >= 0) & obj[geom[:, 0]] & active     # pad is geom2
        pad = jnp.where(first, pad1, pad2)
        selected = first | second
        onehot = (pad[:, None] == jnp.arange(N_PADS)[None]) & selected[:, None]   # (C, P)
        weight = onehot.astype(f.dtype)

        frame = contact.frame                                  # rows: normal, t1, t2
        sign = jnp.where(first, 1.0, -1.0)
        world = jnp.einsum("ci,cij->cj", f, frame) * sign[:, None]   # pad on object
        normal_pad_to_obj = frame[:, 0] * sign[:, None]
        f_n_c = jnp.maximum(f[:, 0], 0.0)
        f_t_c = jnp.linalg.norm(f[:, 1:3], axis=-1)
        margin_c = contact.friction[:, 0] * f_n_c - f_t_c

        # Relative tangential velocity of object versus pad body at the contact.
        pad_geom = jnp.where(first, geom[:, 0], geom[:, 1])
        pad_body = jnp.asarray(self.geom_body)[pad_geom]
        roots = jnp.asarray(self.body_root)

        def point_velocity(body, point):
            cvel = data.cvel[body]
            offset = point - data.subtree_com[roots[body]]
            return cvel[:, 3:] + jnp.cross(cvel[:, :3], offset)

        obj_body = jnp.full(pad_body.shape, self.object_body)
        rel = point_velocity(obj_body, contact.pos) - point_velocity(pad_body, contact.pos)
        rel_t = rel - jnp.sum(rel * frame[:, 0], axis=-1, keepdims=True) * frame[:, 0]
        slipping_c = (f_n_c > SLIP_MIN_FORCE) & (jnp.linalg.norm(rel_t, axis=-1) > SLIP_SPEED) & selected

        f_world = weight.T @ world                             # (P, 3)
        f_n = weight.T @ f_n_c
        f_t = weight.T @ f_t_c
        safe = jnp.maximum(f_n, 1e-9)[:, None]
        centroid_world = (weight * f_n_c[:, None]).T @ contact.pos / safe
        normal_world = (weight * f_n_c[:, None]).T @ normal_pad_to_obj
        normal_world = normal_world / jnp.maximum(jnp.linalg.norm(normal_world, axis=-1, keepdims=True), 1e-9)
        loaded = onehot & (f_n_c > SLIP_MIN_FORCE)[:, None]
        margin = jnp.min(jnp.where(loaded, margin_c[:, None], jnp.inf), axis=0)
        margin = jnp.where(jnp.isfinite(margin), margin, 0.0)
        slip = jnp.any(onehot & slipping_c[:, None], axis=0)

        rot = data.xmat[self.frame_bodies]                     # (P, 3, 3) body-to-world
        origin = data.xpos[self.frame_bodies]
        has = (f_n > 0)[:, None]
        force_pad = jnp.einsum("pji,pj->pi", rot, f_world)
        centroid_pad = jnp.where(has, jnp.einsum("pji,pj->pi", rot, centroid_world - origin), 0.0)
        obj_rot = data.xmat[self.object_body]
        obj_pos = data.xpos[self.object_body]
        point_obj = jnp.where(has, (centroid_world - obj_pos) @ obj_rot, 0.0)
        normal_obj = jnp.where(has, normal_world @ obj_rot, 0.0)
        return Reading(force=force_pad, f_n=f_n, f_t=f_t, centroid=centroid_pad, point_obj=point_obj,
                       normal_obj=normal_obj, slip=slip, margin=margin, engaged=f_n > ENGAGE_N)


def actor_features(reading, age):
    """(P * 10,) fixed-scale features; age counts consecutive engaged steps."""
    x = jnp.concatenate((
        reading.force / FORCE_SCALE,
        reading.f_n[:, None] / FORCE_SCALE,
        reading.centroid / LOCATION_SCALE,
        reading.slip[:, None].astype(jnp.float32),
        reading.engaged[:, None].astype(jnp.float32),
        (jnp.minimum(age, AGE_STEPS) / AGE_STEPS)[:, None],
    ), axis=-1)
    return jnp.clip(x, -5.0, 5.0).reshape(-1)


def critic_features(reading):
    x = jnp.concatenate((
        reading.force / FORCE_SCALE,
        reading.f_n[:, None] / FORCE_SCALE,
        reading.slip[:, None].astype(jnp.float32),
        jnp.clip(reading.margin, -1.0, 1.0)[:, None],
        reading.engaged[:, None].astype(jnp.float32),
    ), axis=-1)
    return jnp.clip(x, -5.0, 5.0).reshape(-1)


ACTOR_SIZE = N_PADS * len(ACTOR_FIELDS)
CRITIC_SIZE = N_PADS * len(CRITIC_FIELDS)


def _skew(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack((jnp.stack((zero, -z, y), -1), jnp.stack((z, zero, -x), -1),
                      jnp.stack((-y, x, zero), -1)), -2)


def grasp_quality(reading, half_edge=CUBE_HALF_EDGE_M):
    """§3.5 index from engaged pads as frictional point contacts at their centroids.

    For full 3-D point contacts G_i G_i^T does not depend on the contact frame, so
    sigma_min(G)^2 is the smallest eigenvalue of sum_i [[I, S_i^T], [S_i, S_i S_i^T]].
    """
    w = reading.engaged.astype(jnp.float32)
    s = _skew(reading.point_obj / half_edge)
    eye = jnp.eye(3)
    top = jnp.sum(w) * eye
    right = jnp.einsum("p,pji->ij", w, s)
    bottom = jnp.einsum("p,pij,pkj->ik", w, s, s)
    gram = jnp.block([[top, right], [right.T, bottom]])
    sigma = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram)[0], 0.0))
    return jnp.clip(sigma / cube_reference_sigma(half_edge), 0.0, 1.0)
