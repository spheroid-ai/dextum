"""Frozen 3mm virtual sensing grid on the existing CAD skin, no colliders added."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def build(cad, pitch=3.):
    import numpy as np
    cells = []
    for part in cad['parts']:
        name = part.get('module', '')
        if part.get('layer') != 'skin' or not (name == 'palm' or name.endswith('_pad')):
            continue
        vertices = np.asarray(part['vertices'])
        count = len(vertices)//2
        faces = np.asarray(part['faces'])
        faces = faces[(faces >= count).all(1)]-count
        lower, upper = vertices[:count], vertices[count:]
        normals = (upper-lower)/(part['band_mm'][1]-part['band_mm'][0])
        tri = upper[faces]
        low, high = tri.min((0, 1)), tri.max((0, 1))
        # Pitch is in the body's x/z projection; curved-surface distances and
        # coverage are reported. Mesh holes remain holes, never convex-filled.
        xs = np.arange(np.ceil(low[0]/pitch)*pitch, high[0]+1e-9, pitch)
        zs = np.arange(np.ceil(low[2]/pitch)*pitch, high[2]+1e-9, pitch)
        for z in zs:
            for x in xs:
                a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
                dx, dz = x-tri[:, 0, 0], z-tri[:, 0, 2]
                det = a[:, 0]*b[:, 2]-a[:, 2]*b[:, 0]
                den = np.where(abs(det)>1e-12, det, 1)
                u = (dx*b[:, 2]-dz*b[:, 0])/den
                v = (a[:, 0]*dz-a[:, 2]*dx)/den
                valid = (abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
                if not valid.any():
                    continue
                k = int(np.flatnonzero(valid)[0])
                weights = np.array([1-u[k]-v[k], u[k], v[k]])
                pos = weights@tri[k]
                normal = weights@normals[faces[k]]
                normal /= np.linalg.norm(normal)
                tx = np.array([1., 0, 0])-normal[0]*normal
                tx /= np.linalg.norm(tx)
                ty = np.cross(normal, tx)
                old = [c for c in cad['cells'] if c['module'] == name]
                nearest = min(old, key=lambda c: np.linalg.norm(pos-np.asarray(c['position_mm'])))
                cells.append(dict(module=name, body=part['body'], position_mm=pos.tolist(),
                    rotation=np.stack((tx, ty, normal), -1).tolist(),
                    old_cell_id=nearest['id'], grid_x_mm=float(x), grid_z_mm=float(z)))
    assert {c['module'] for c in cells} == {'palm','index_pad','middle_pad','ring_pad','pinky_pad'}
    assert all(c['old_cell_id']>=20 for c in cells)
    return cells


def main():
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    cad = json.loads(a.geometry.read_text())
    cells = build(cad)
    groups = {}
    for name in sorted({c['module'] for c in cells}):
        positions = np.array([c['position_mm'] for c in cells if c['module']==name])
        dist = np.linalg.norm(positions[:, None]-positions[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        nearest = dist.min(1)
        groups[name] = dict(cells=len(positions), nearest_spacing_mm=dict(
            minimum=float(nearest.min()), median=float(np.median(nearest)), maximum=float(nearest.max())))
    result = dict(state='LAYOUT_PREPARED', pitch_projected_mm=3., cells=cells, groups=groups,
        geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),
        physics='Virtual cells only; unchanged accepted XML, colliders, mass and solver',
        coverage='Existing CAD skin triangles, clipped at edges and holes; nearest-cell assignment within each pad')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cells=len(cells), groups=groups)), flush=True)


if __name__ == '__main__':
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('Numerical layout preparation runs on Vast only')
    main()
