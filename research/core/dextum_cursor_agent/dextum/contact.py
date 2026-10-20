"""Contact-layer physics and geometry (spec §3.2, §3.5).

All forces are in newtons, lengths in metres. Contact points and normals for the
grasp matrix are expressed in the object frame (origin at the object centre).
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

ENGAGE_N = 0.3
LIGHT_MAX_N = 1.0
FIRM_MAX_N = 4.0
SAFETY_FACTOR = 0.3
MU_INIT = 0.4
CUBE_HALF_EDGE_M = 0.025

NONE, LIGHT, FIRM, HARD = 0, 1, 2, 3
FORCE_CLASS_NAMES = ("none", "light", "firm", "hard")


def force_class(f_n) -> np.ndarray:
    """Discretise pressing force: 0 none (<0.3 N), 1 light, 2 firm (1-4 N), 3 hard (>4 N)."""
    f = np.asarray(f_n, dtype=float)
    return np.select([f < ENGAGE_N, f < LIGHT_MAX_N, f < FIRM_MAX_N], [NONE, LIGHT, FIRM], HARD)


def cone_margin(f_n, f_t, mu) -> np.ndarray:
    """Friction-cone margin m = mu * F_n - |F_t|; positive while the contact holds."""
    return mu * np.asarray(f_n, dtype=float) - np.linalg.norm(np.asarray(f_t, dtype=float), axis=-1)


def required_normal_force(f_t, mu, safety=SAFETY_FACTOR) -> np.ndarray:
    """Pressing force that keeps |F_t| inside the cone with the safety factor."""
    return (1.0 + safety) * np.linalg.norm(np.asarray(f_t, dtype=float), axis=-1) / mu


def contact_frame(normal) -> np.ndarray:
    """Orthonormal frame with the normal as first column, tangents as the other two."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([n, t1, t2], axis=1)


def _skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def grasp_matrix(points, normals, length_scale=CUBE_HALF_EDGE_M) -> np.ndarray:
    """6 x 3k grasp matrix of frictional point contacts.

    Torque rows are divided by ``length_scale`` so force and torque rows are commensurate.
    """
    blocks = []
    for p, n in zip(np.asarray(points, float).reshape(-1, 3), np.asarray(normals, float).reshape(-1, 3)):
        frame = contact_frame(n)
        blocks.append(np.vstack([frame, _skew(p / length_scale) @ frame]))
    if not blocks:
        return np.zeros((6, 0))
    return np.hstack(blocks)


def grasp_sigma_min(points, normals, length_scale=CUBE_HALF_EDGE_M) -> float:
    g = grasp_matrix(points, normals, length_scale)
    if g.shape[1] < 6:
        return 0.0
    return float(np.linalg.svd(g, compute_uv=False)[5])


@lru_cache(maxsize=None)
def cube_reference_sigma(half_edge=CUBE_HALF_EDGE_M) -> float:
    """Best smallest singular value over 3-contact grasps at cube face centres.

    This is the normaliser of the grasp-quality index: 1.0 means as good as the best
    three-face grasp of the cube.
    """
    faces = [np.eye(3)[i] * s for i in range(3) for s in (1.0, -1.0)]
    best = 0.0
    for combo in itertools.combinations(faces, 3):
        pts = [half_edge * f for f in combo]
        nrm = [-f for f in combo]
        best = max(best, grasp_sigma_min(pts, nrm, half_edge))
    return best


def grasp_quality(points, normals, half_edge=CUBE_HALF_EDGE_M) -> float:
    """Grasp-quality index in [0, 1] (spec §3.5)."""
    return float(np.clip(grasp_sigma_min(points, normals, half_edge) / cube_reference_sigma(half_edge), 0.0, 1.0))


def cube_sdf(point, half_edge=CUBE_HALF_EDGE_M) -> float:
    """Analytic signed distance of a point (object frame) to an axis-aligned cube."""
    q = np.abs(np.asarray(point, float)) - half_edge
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(np.max(q), 0.0))


def cube_surface_normal(point, half_edge=CUBE_HALF_EDGE_M) -> np.ndarray:
    """Outward normal of the cube face nearest to ``point`` (object frame)."""
    p = np.asarray(point, float)
    axis = int(np.argmax(np.abs(p) - half_edge))
    n = np.zeros(3)
    n[axis] = np.sign(p[axis]) or 1.0
    return n


def cube_target_points(half_edge=CUBE_HALF_EDGE_M, grid_m=0.010):
    """Candidate contact points: face centres plus a grid on each face (spec §3.5)."""
    ticks = np.arange(-half_edge + grid_m / 2, half_edge, grid_m)
    points, normals = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            others = [a for a in range(3) if a != axis]
            centre = np.zeros(3)
            centre[axis] = sign * half_edge
            normal = np.zeros(3)
            normal[axis] = sign
            points.append(centre.copy())
            normals.append(normal.copy())
            for u, v in itertools.product(ticks, ticks):
                p = centre.copy()
                p[others[0]], p[others[1]] = u, v
                points.append(p)
                normals.append(normal.copy())
    return np.array(points), np.array(normals)


def update_friction_estimate(mu_hat: float, slip_onset_ratio: float | None, held_ratio: float | None) -> float:
    """Friction estimate rule of spec §3.5.

    At slip onset the estimate moves toward the observed |F_t|/F_n ratio. It is only raised
    when a slip-free hold was observed at a higher ratio, so a slip can only lower it.
    """
    if slip_onset_ratio is not None:
        return min(mu_hat, 0.7 * mu_hat + 0.3 * float(slip_onset_ratio))
    if held_ratio is not None and held_ratio > mu_hat:
        return float(held_ratio)
    return mu_hat
