"""Candidate optimization; same pad/site assignment and inherited sensor model.

Not the registered training implementation until benchmark/parity review.
"""
from pass1c.sensors import Sensors as Reference, CELL_GROUP
import numpy as np
import jax.numpy as j
from mujoco.mjx._src import support

_CANDIDATES=np.zeros((10,12),dtype=np.int32)
_VALID=np.zeros((10,12),dtype=bool)
for _pad in range(10):
    _ids=np.flatnonzero(CELL_GROUP==_pad)
    _CANDIDATES[_pad,:len(_ids)]=_ids
    _VALID[_pad,:len(_ids)]=True

def assignments(positions,geom_side,site_positions,lookup):
    pad=j.asarray(lookup)[j.maximum(geom_side,0)]
    group=j.maximum(pad,0)
    candidates=j.asarray(_CANDIDATES)[group]
    distance=j.sum((positions[:,None,:]-site_positions[candidates])**2,-1)
    distance=j.where(j.asarray(_VALID)[group],distance,j.inf)
    selected=j.argmin(distance,axis=1)
    return j.take_along_axis(candidates,selected[:,None],axis=1)[:,0],pad>=0

class Sensors(Reference):
    def raw(self,model,data):
        c=data._impl.contact;geom=j.maximum(c.geom,0)
        wrench=j.zeros((c.pos.shape[0],6),data.qpos.dtype)
        for dim in np.unique(c.dim):
            force,ids=support.contact_force_dim(model,data,int(dim))
            wrench=wrench.at[ids].set(force)
        world=j.einsum('ci,cij->cj',wrench[:,:3],c.frame)
        forces=j.zeros((40,3));speed=j.zeros(40)
        active=j.all(c.geom>=0,-1)&(c.efc_address>=0)
        for side,sign in ((0,1.),(1,-1.)):
            cell,eligible=assignments(c.pos,c.geom[:,side],data.site_xpos[self.sites],self.lookup)
            valid=active&eligible
            body=j.asarray(self.body)[geom[:,side]];other=j.asarray(self.body)[geom[:,1-side]]
            def vel(b):
                return data.cvel[b,3:]+j.cross(data.cvel[b,:3],c.pos-data.subtree_com[j.asarray(self.roots)[b]])
            rel=vel(other)-vel(body)
            tang=rel-j.sum(rel*c.frame[:,0],-1,keepdims=True)*c.frame[:,0]
            speed+=j.zeros(40).at[cell].add(j.where(valid,j.linalg.norm(tang,axis=-1)*j.maximum(wrench[:,0],0),0))
            summed=j.zeros((40,3)).at[cell].add(j.where(valid[:,None],world*sign,0))
            local=j.einsum('nji,nj->ni',data.site_xmat[self.sites],summed)
            forces+=local[:,j.array([2,0,1])]
        return forces,speed
