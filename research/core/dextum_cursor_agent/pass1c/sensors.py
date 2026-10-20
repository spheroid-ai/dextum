"""Same sensor error model; contacts assigned to nearest site on the same pad."""
import os
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import numpy as np
import jax.numpy as j
from mujoco.mjx._src import support
from pass1b.sensors import Sensors as PreviousSensors, FEATURES, DT

PAD_NAMES=('thumb_tip','index_tip','middle_tip','ring_tip','pinky_tip',
           'palm','index_pad','middle_pad','ring_pad','pinky_pad')
CELL_GROUP=np.repeat(np.arange(10),[4,4,4,4,4,12,2,2,2,2])

def assignments(positions,geom_side,site_positions,lookup):
    """One Voronoi footprint within the contacted pad; no object-id filter."""
    pad=j.asarray(lookup)[j.maximum(geom_side,0)]
    distance=j.sum((positions[:,None,:]-site_positions[None,:,:])**2,-1)
    distance=j.where(pad[:,None]==j.asarray(CELL_GROUP)[None,:],distance,j.inf)
    return j.argmin(distance,axis=1),pad>=0

class Sensors(PreviousSensors):
    def __init__(self,m,cells,truth):
        self.level=os.environ.get('DEXTUM_SENSOR_LEVEL','Realistic')
        if self.level not in ('Ideal','Realistic'):raise ValueError(self.level)
        self.sites=np.array([m.site('kit_site_'+c['name']).id for c in cells])
        self.pad_geoms=np.array([m.geom('kit_pad_contact_'+n).id for n in PAD_NAMES])
        self.geoms=self.pad_geoms[CELL_GROUP]
        self.lookup=np.full(m.ngeom,-1,int);self.lookup[self.pad_geoms]=np.arange(10)
        # A conformal palm may have contiguous convex pieces. They all feed
        # the same palm sites; pieces are geometry, never sensing cells.
        for gid in range(m.ngeom):
            if m.geom(gid).name.startswith('kit_pad_contact_palm_part_'):
                self.lookup[gid]=PAD_NAMES.index('palm')
        self.body=np.asarray(m.geom_bodyid);self.roots=np.asarray(m.body_rootid)
        self.pad=truth.pad_of_geom[self.geoms];self.frames=truth.frame_bodies
        self.cell_pos=np.asarray(m.site_pos[self.sites])
        self.membership=(self.pad[:,None]==np.arange(11)).astype(np.float32)
        self.coupling=np.eye(20,dtype=np.float32)
        for group in [list(range(12))]+[list(range(i,i+2)) for i in range(12,20,2)]:
            for a in group:
                for b in group:
                    if a!=b and np.linalg.norm(self.cell_pos[a+20]-self.cell_pos[b+20])<=.0181:self.coupling[a,b]=.05

    def raw(self,model,data):
        c=data._impl.contact;geom=j.maximum(c.geom,0)
        wrench=j.zeros((c.pos.shape[0],6),data.qpos.dtype)
        for dim in np.unique(c.dim):
            force,ids=support.contact_force_dim(model,data,int(dim));wrench=wrench.at[ids].set(force)
        world=j.einsum('ci,cij->cj',wrench[:,:3],c.frame)
        forces=j.zeros((40,3));speed=j.zeros(40)
        active=j.all(c.geom>=0,-1)&(c.efc_address>=0)
        for side,sign in ((0,1.),(1,-1.)):
            cell,eligible=assignments(c.pos,c.geom[:,side],data.site_xpos[self.sites],self.lookup)
            mask=(cell[:,None]==j.arange(40))&active[:,None]&eligible[:,None]
            body=j.asarray(self.body)[geom[:,side]];other=j.asarray(self.body)[geom[:,1-side]]
            def vel(b):return data.cvel[b,3:]+j.cross(data.cvel[b,:3],c.pos-data.subtree_com[j.asarray(self.roots)[b]])
            rel=vel(other)-vel(body)
            tang=rel-j.sum(rel*c.frame[:,0],-1,keepdims=True)*c.frame[:,0]
            speed+=mask.T@(j.linalg.norm(tang,axis=-1)*j.maximum(wrench[:,0],0))
            local=j.einsum('nji,nj->ni',data.site_xmat[self.sites],mask.astype(world.dtype).T@(world*sign))
            forces+=local[:,j.array([2,0,1])]
        return forces,speed
