"""Print the cube/hand facts the touch environment depends on."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from vendor.aero_official.orientation_curriculum import OrientationCurriculum, STAGES
from vendor.spheroid_kit.sensors import AERO_PADS


def main():
    env = OrientationCurriculum(STAGES[0])
    m = env.mj_model
    cube = m.body("cube").id
    out = dict(
        nq=m.nq, nv=m.nv, nu=m.nu, ngeom=m.ngeom, nbody=m.nbody,
        cone="elliptic" if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal",
        timestep=m.opt.timestep, n_substeps=env.n_substeps,
        cube_mass=float(m.body_mass[cube]), cube_inertia=m.body_inertia[cube].tolist(),
        cube_size=m.geom_size[m.geom("cube").id].tolist(),
        cube_friction=m.geom_friction[m.geom("cube").id].tolist(),
        gravity=m.opt.gravity.tolist(),
    )
    pad_geoms = {g for p in AERO_PADS for g in p.geoms}
    rows = []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        body = m.body(m.geom_bodyid[g]).name
        rows.append(dict(geom=m.geom(g).name or f"#{g}", body=body,
                         type=int(m.geom_type[g]), condim=int(m.geom_condim[g]),
                         friction=m.geom_friction[g].round(3).tolist(),
                         size=m.geom_size[g].round(4).tolist(),
                         on_pad=(m.geom(g).name in pad_geoms)))
    out["collision_geoms"] = rows
    out["home_cube_qpos"] = m.keyframe("home").qpos[-7:].tolist()
    path = Path(__file__).resolve().parents[1] / "results/model_facts.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "collision_geoms"}))
    for r in rows:
        print(r["geom"], r["body"], r["type"], r["size"], r["friction"][0], "PAD" if r["on_pad"] else "")


if __name__ == "__main__":
    main()
