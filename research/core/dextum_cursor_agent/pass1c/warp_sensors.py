"""Warp contact arrays to the existing Kit B sensor transducer.

UNQUALIFIED: review section 13 item-1 execution failed the Warp wrench-oracle
check. The elliptic branch below incorrectly infers consecutive addresses for
non-six-column tables. Preserved for diagnosis; not admitted for training.
See runs/pass1_d5_powered_01/warp_sensor_item1/result.json. No rerun authorized.

Physical interface only. This adapter does not change sensor noise, reward,
goal recognition, geometry or solver settings. Qualification is separate.
All contact assignment and force decoding below operate on JAX arrays.
"""
import os
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Vast.ai only')
import jax.numpy as j
from pass1c.sensors import Sensors, CELL_GROUP


def contact_wrench(cone, fields):
    """MuJoCo contact-frame wrench, including variable contact dimensions.

    Warp stores constraint addresses per contact/dimension and constraint
    forces per world. Pyramidal tangential components are opposite-edge
    differences times friction; the normal force is the edge-force sum.
    """
    world=j.clip(fields['worldid'],0,fields['efc_force'].shape[0]-1)
    address=fields['efc_address']
    if address.ndim==1:address=address[:,None]
    dim=fields['dim']
    valid=(j.arange(dim.shape[0])<fields['ncon'].reshape(-1)[0])&(address[:,0]>=0)
    if int(cone)==0:
        rows=j.where(dim==1,1,2*(dim-1))
        offsets=j.arange(10)
        idx=address[:,:1]+offsets
        force=fields['efc_force'][world[:,None],j.clip(idx,0,fields['efc_force'].shape[1]-1)]
        force=j.where(valid[:,None]&(offsets<rows[:,None])&(idx<fields['efc_force'].shape[1]),force,0)
        normal=force.sum(-1,keepdims=True)
        tangent=(force[:,::2]-force[:,1::2])*fields['friction']
        tangent=j.where(j.arange(5)[None,:]<(dim-1)[:,None],tangent,0)
        return j.concatenate((normal,tangent),axis=-1)
    if int(cone)==1:
        # Elliptic addresses are supplied for each contact dimension.
        idx=address if address.shape[1]==6 else address[:,:1]+j.arange(6)
        force=fields['efc_force'][world[:,None],j.clip(idx,0,fields['efc_force'].shape[1]-1)]
        return j.where(valid[:,None]&(j.arange(6)<dim[:,None])&(idx>=0)&(idx<fields['efc_force'].shape[1]),force,0)
    raise ValueError('Unsupported MuJoCo cone')


class WarpSensors(Sensors):
    def raw_batched(self, cone, fields):
        """Return the same (world,40,3) cell forces and weighted slip speeds."""
        batch=fields['site_xpos'].shape[0]
        count=fields['geom'].shape[0]
        geom=j.maximum(fields['geom'],0)
        wid=j.clip(fields['worldid'],0,batch-1)
        address=fields['efc_address']
        address=address[:,0] if address.ndim==2 else address
        active=(j.arange(count)<fields['ncon'].reshape(-1)[0])&(address>=0)&j.all(fields['geom']>=0,-1)
        wrench=contact_wrench(cone,fields)
        world_force=j.einsum('ci,cij->cj',wrench[:,:3],fields['frame'])
        forces=j.zeros((batch*40,3),world_force.dtype)
        speed=j.zeros(batch*40,world_force.dtype)
        cell_positions=fields['site_xpos'][:,self.sites]
        cell_frames=fields['site_xmat'][:,self.sites]
        for side,sign in ((0,1.),(1,-1.)):
            pad=j.asarray(self.lookup)[geom[:,side]]
            distance=j.sum((fields['pos'][:,None,:]-cell_positions[wid])**2,-1)
            distance=j.where(pad[:,None]==j.asarray(CELL_GROUP)[None,:],distance,j.inf)
            cell=j.argmin(distance,axis=-1)
            enabled=active&(pad>=0)
            target=wid*40+cell
            body=j.asarray(self.body)[geom[:,side]]
            other=j.asarray(self.body)[geom[:,1-side]]
            def velocity(b):
                cvel=fields['cvel'][wid,b]
                root=j.asarray(self.roots)[b]
                return cvel[:,3:]+j.cross(cvel[:,:3],fields['pos']-fields['subtree_com'][wid,root])
            rel=velocity(other)-velocity(body)
            normal=fields['frame'][:,0]
            tang=rel-j.sum(rel*normal,-1,keepdims=True)*normal
            speed=speed.at[target].add(j.where(enabled,j.linalg.norm(tang,axis=-1)*j.maximum(wrench[:,0],0),0))
            local=j.einsum('cji,cj->ci',cell_frames[wid,cell],world_force*sign)
            local=local[:,j.array([2,0,1])]
            forces=forces.at[target].add(j.where(enabled[:,None],local,0))
        return forces.reshape(batch,40,3),speed.reshape(batch,40)


def warp_fields(data, wp):
    """Zero-copy device arrays from native Warp; synchronize producer first."""
    result={name:wp.to_jax(getattr(data.contact,name)) for name in
            ('geom','pos','frame','dim','friction','efc_address','worldid')}
    result.update(ncon=wp.to_jax(data.ncon),efc_force=wp.to_jax(data.efc.force))
    result.update({name:wp.to_jax(getattr(data,name)) for name in
                   ('site_xpos','site_xmat','cvel','subtree_com')})
    return result
