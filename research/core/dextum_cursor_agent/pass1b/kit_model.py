"""Physical Kit B model adapter; sensor and privileged truth bindings are separate."""
import json, os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'): raise RuntimeError('Vast.ai only')
import numpy as np
import mujoco
from mujoco import mjx
from dextum import touch_mjx

KIT=Path(os.environ.get('DEXTUM_KIT_ROOT','/root/dextum/kit_b'))

class TruthTouch(touch_mjx.TouchModel):
    """Reward-only original 11-pad interface including fixed Kit B descendants."""
    def __init__(self,m):
        self.object_body=m.body('cube').id
        collide=(m.geom_contype!=0)|(m.geom_conaffinity!=0)
        direct={m.body(b).id:p for p,(_,names) in enumerate(touch_mjx.PAD_BODIES) for b in names}
        self.frame_bodies=np.array([m.body(names[0]).id for _,names in touch_mjx.PAD_BODIES])
        self.pad_of_geom=np.full(m.ngeom,-1,int)
        for g in np.flatnonzero(collide):
            body=int(m.geom_bodyid[g])
            while body and body not in direct: body=int(m.body_parentid[body])
            if body in direct: self.pad_of_geom[g]=direct[body]
        self.object_geom=collide&(m.geom_bodyid==self.object_body)
        self.geom_body=np.asarray(m.geom_bodyid);self.body_root=np.asarray(m.body_rootid)
        self.pad_geom_counts=np.bincount(self.pad_of_geom[self.pad_of_geom>=0],minlength=11)


def install(env):
    # Same bodies/inertia; measured adaptive Kit B collision approximation. The timestep
    # is set to the registered 200-Hz contact/sensor integration cadence.
    model_path=Path(os.environ.get('DEXTUM_KIT_XML','/root/dextum/dextum_cursor_agent/runs/pass1_realistic_01/collision_shared/kit_b_training.xml'))
    m=mujoco.MjModel.from_xml_path(str(model_path))
    m.opt.timestep=.005
    env._config.sim_dt=.005;env._sim_dt=.005
    env._mj_model=m;env._mjx_model=mjx.put_model(m)
    env._post_init()
    env.touch_model=TruthTouch(m)
    cells=json.loads((KIT/'geometry.json').read_text())['cells']
    return cells
