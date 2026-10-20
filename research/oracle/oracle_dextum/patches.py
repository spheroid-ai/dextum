"""Patch assignment with explicit boundary ambiguity and distance reporting."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Assignment:
    patch_id: int
    distance_m: float
    boundary_ambiguous: bool


class CubePatches:
    """Six analytic faces, -x,+x,-y,+y,-z,+z. Cube-local coordinates."""
    def __init__(self, half_edge_m=.025, boundary_tolerance_m=1e-6):
        if half_edge_m <= 0 or boundary_tolerance_m < 0:
            raise ValueError('Invalid cube dimensions')
        self.half = half_edge_m
        self.tolerance = boundary_tolerance_m

    def assign(self, point, outward_normal=None):
        import numpy as np
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Expected finite object-local point')
        # Nearest point on each bounded face, including penetrating contacts.
        distances = []
        for patch in range(6):
            axis, side = divmod(patch, 2)
            projected = np.clip(point, -self.half, self.half)
            projected[axis] = self.half*(2*side-1)
            distances.append(float(np.linalg.norm(point-projected)))
        order = np.argsort(distances, kind='stable')
        first, second = map(int, order[:2])
        ambiguous = distances[second]-distances[first] <= self.tolerance
        if outward_normal is not None:
            normal = np.asarray(outward_normal, dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
                raise ValueError('Invalid contact surface normal')
            # A solver point can lie inside the surface; the oriented contact
            # normal disambiguates a box face at its edges/corners.
            axis = int(np.argmax(np.abs(normal)))
            first = 2*axis+int(normal[axis] >= 0)
            rank = np.sort(np.abs(normal)/np.linalg.norm(normal))
            ambiguous |= rank[-1]-rank[-2] <= 1e-6
        return Assignment(first, distances[first], bool(ambiguous))


class MeshPatches:
    """Nearest triangle in a frozen exported mesh, not an exact smooth surface.

    Every face must have exactly one patch label. Model geometry/mesh scale must
    be validated by the caller; a graph for a visual mesh is not automatically
    an exact collision-hull partition.
    """
    def __init__(self, mesh_path, graph_path, mesh_sha256, *, boundary_tolerance_m=1e-6):
        import numpy as np
        import trimesh
        mesh_path = Path(mesh_path)
        if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != mesh_sha256:
            raise ValueError('Patch graph mesh provenance mismatch')
        self.mesh = trimesh.load(mesh_path, force='mesh', process=False)
        graph = json.loads(Path(graph_path).read_text())
        self.labels = np.full(len(self.mesh.faces), -1, dtype=int)
        for patch in graph['patches']:
            indices = np.asarray(patch['surface_face_ids'], dtype=int)
            if (indices < 0).any() or (indices >= len(self.labels)).any():
                raise ValueError('Invalid face label')
            if (self.labels[indices] >= 0).any() or len(np.unique(indices)) != len(indices):
                raise ValueError('Overlapping patch face labels')
            self.labels[indices] = patch['id']
        if (self.labels < 0).any():
            raise ValueError('Incomplete patch partition')
        self.tolerance = boundary_tolerance_m

    def assign(self, point, outward_normal=None):
        import numpy as np
        import trimesh
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Expected finite object-local point')
        closest = trimesh.triangles.closest_point(self.mesh.triangles,
                                                  np.broadcast_to(point, (len(self.labels), 3)))
        distance = np.linalg.norm(closest-point, axis=1)
        i = int(np.argmin(distance))
        near = distance <= distance[i]+self.tolerance
        if outward_normal is not None:
            normal = np.asarray(outward_normal, dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
                raise ValueError('Invalid contact surface normal')
            candidates = np.flatnonzero(near)
            i = int(candidates[np.argmax(self.mesh.face_normals[candidates]@normal)])
        return Assignment(int(self.labels[i]), float(distance[i]),
                          len(np.unique(self.labels[near])) > 1)
