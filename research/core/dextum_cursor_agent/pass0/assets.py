"""Offline three-shape stand-ins. Mesh generation runs on Vast only."""
import hashlib
import json
import math
import os
from pathlib import Path

ASSETS = (
    dict(id="p0_cube55", family="cube", dimensions_mm=[55, 55, 55], mass_g=92, fillet_mm=0),
    dict(id="p0_rounded48", family="rounded_cube", dimensions_mm=[48, 48, 48], mass_g=100, fillet_mm=7),
    dict(id="p0_cylinder38x70", family="cylinder", dimensions_mm=[38, 38, 70], mass_g=80, fillet_mm=0),
)


def mesh(asset):
    if not os.environ.get("CONTAINER_ID"):
        raise RuntimeError("Generate meshes on Vast.ai only")
    vertices, faces = [], []
    if asset["family"] == "cylinder":
        radius, half = asset["dimensions_mm"][0]/2000, asset["dimensions_mm"][2]/2000
        segments = 48
        for z in (-half, half):
            vertices.extend((radius*math.cos(2*math.pi*i/segments), radius*math.sin(2*math.pi*i/segments), z)
                            for i in range(segments))
        vertices += [(0, 0, -half), (0, 0, half)]
        for i in range(segments):
            j = (i+1) % segments
            faces += [(i, j, j+segments), (i, j+segments, i+segments),
                      (2*segments, j, i), (2*segments+1, i+segments, j+segments)]
    else:
        half = asset["dimensions_mm"][0]/2000
        radius = asset["fillet_mm"]/1000
        inner = half-radius
        steps = 12 if radius else 1
        lookup = {}
        for axis in range(3):
            u, v = (axis+1) % 3, (axis+2) % 3
            for sign in (-1, 1):
                grid = {}
                for i in range(steps+1):
                    for j in range(steps+1):
                        point = [0.0]*3
                        point[axis], point[u], point[v] = sign*half, half*(2*i/steps-1), half*(2*j/steps-1)
                        if radius:
                            core = [max(-inner, min(inner, x)) for x in point]
                            delta = [x-c for x, c in zip(point, core)]
                            norm = math.sqrt(sum(x*x for x in delta))
                            point = [c+radius*d/norm for c, d in zip(core, delta)]
                        key = tuple(round(x, 12) for x in point)
                        if key not in lookup:
                            lookup[key] = len(vertices)
                            vertices.append(tuple(point))
                        grid[i, j] = lookup[key]
                for i in range(steps):
                    for j in range(steps):
                        a, b, c, d = (grid[i, j], grid[i+1, j], grid[i+1, j+1], grid[i, j+1])
                        triangles = [(a, b, c), (a, c, d)]
                        faces += triangles if sign > 0 else [tuple(reversed(f)) for f in triangles]
    return vertices, faces


def patches(asset):
    if asset["family"] == "cylinder":
        return dict(source="handwritten_pass0", patches=[dict(id="side", curvature="curved"),
                    dict(id="cap_neg", curvature="flat"), dict(id="cap_pos", curvature="flat")],
                    adjacency=[["side", "cap_neg"], ["side", "cap_pos"]])
    flat = [f"face_{axis}_{sign}" for axis in "xyz" for sign in ("neg", "pos")]
    adjacency = [[a, b] for i, a in enumerate(flat) for b in flat[i+1:] if a.split('_')[1] != b.split('_')[1]]
    graph = dict(source="handwritten_pass0", patches=[dict(id=n, curvature="flat") for n in flat], adjacency=adjacency)
    if asset["family"] == "rounded_cube":
        graph["patches"] += [dict(id=f"fillet_{i}", curvature="curved") for i in range(12)]
        graph["adjacency"] = [[face, f"fillet_{i}"] for i, pair in enumerate(adjacency) for face in pair]
        graph["omission"] = "Corner blends are boundaries in the hand-written stand-in, not separately segmented patches"
    return graph


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for source in ASSETS:
        vertices, faces = mesh(source)
        obj = ''.join('v %.12g %.12g %.12g\n' % p for p in vertices)
        obj += ''.join('f %d %d %d\n' % tuple(i+1 for i in f) for f in faces)
        path = output/(source["id"]+".obj")
        path.write_text(obj)
        # Each undirected edge must have two opposite orientations.
        edges = {}
        for face in faces:
            for a, b in zip(face, (*face[1:], face[0])):
                key = tuple(sorted((a, b)))
                edges.setdefault(key, []).append((a, b))
        watertight = all(len(e)==2 and e[0]==tuple(reversed(e[1])) for e in edges.values())
        assert watertight, source["id"]
        manifest = dict(**source, mesh=path.name, mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        split="held_out_instance", training_exposure="none; checkpoint used 50 mm cube only",
                        family_held_out=source["family"]!="cube", friction=.3,
                        patch_graph=patches(source), watertight=watertight,
                        status="Pass 0 hand-written mesh; not the Pass 1 Blender exporter",
                        provenance="Generated after historical E3 checkpoint freeze; no tuning on demo outcomes")
        (output/(source["id"]+".json")).write_text(json.dumps(manifest, indent=2)+"\n")
        manifests.append(manifest)
    return manifests
