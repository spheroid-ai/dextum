"""JAX tag-pose observation oracle; collision-box occlusion, no rendered images.

This is a declared simulated detector, not a trained RGB model. Shared reference
error is added after per-camera cube-tag fusion so it is not counted six times.
"""
import jax
import jax.numpy as j
import numpy as np
from .vision_math import unit,multiply,conjugate,axis_quat,matrix,angle,rot6

IDENTITY=j.array([1.,0.,0.,0.])
REFERENCE_TAG_POSITION=j.array([.04,.14,.01])


def ray_visible(camera,targets,positions,rotations,sizes):
    """Finite segment/OBB intersection; tag endpoint surface is not an occluder."""
    origins=j.einsum('gji,gj->gi',rotations,camera[None,:]-positions)
    delta=targets-camera
    vectors=j.einsum('gji,rj->rgi',rotations,delta)
    safe=j.where(j.abs(vectors)<1e-9,1.,vectors)
    near=(-sizes[None,:,:]-origins[None,:,:])/safe
    far=(sizes[None,:,:]-origins[None,:,:])/safe
    parallel=j.abs(vectors)<1e-9
    inside=j.abs(origins[None,:,:])<=sizes[None,:,:]
    lo=j.where(parallel,j.where(inside,-j.inf,j.inf),j.minimum(near,far))
    hi=j.where(parallel,j.where(inside,j.inf,-j.inf),j.maximum(near,far))
    enter=j.max(lo,axis=-1);leave=j.min(hi,axis=-1)
    # Discard intersections behind camera or within 0.2 mm of tag surface.
    limit=1.-.0002/j.maximum(j.linalg.norm(delta,axis=-1),1e-6)
    blocked=(leave>=j.maximum(enter,0.))&(leave>0.)&(enter<limit[:,None])
    return ~j.any(blocked,axis=-1)


def geometry(cube_position,cube_quaternion,reference,cube_half_size=.025):
    normals=j.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])
    u=j.array([[0.,1,0],[0.,1,0],[0.,0,1],[0.,0,1],[1.,0,0],[1.,0,0]])
    v=j.cross(normals,u)
    signs=j.array([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]])
    # Defaults reproduce the historical 50 mm cube and 36 mm tag exactly.
    tag_half_size=.018*(cube_half_size/.025)
    local=normals[:,None,:]*cube_half_size+tag_half_size*(signs[None,:,0,None]*u[:,None,:]+signs[None,:,1,None]*v[:,None,:])
    rotation=matrix(cube_quaternion)
    corners=local@rotation.T+cube_position
    world_normals=normals@rotation.T
    ref_corners=reference+j.concatenate((signs*.02,j.zeros((4,1))),axis=-1)
    return j.concatenate((corners,ref_corners[None,:,:]),axis=0),j.concatenate((world_normals,j.array([[0.,0.,1.]])))


def sample_cameras(key):
    keys=jax.random.split(key,6)
    az=jax.random.uniform(keys[0],(2,),minval=0,maxval=2*j.pi)
    z=jax.random.uniform(keys[1],(2,),minval=.5,maxval=1.)
    radial=j.sqrt(1-z*z)
    center=j.array([.1,0.,.05])
    direction=j.stack((radial*j.cos(az),radial*j.sin(az),z),axis=-1)
    distance=jax.random.uniform(keys[2],(2,1),minval=.30,maxval=.60)
    pos=center+direction*distance
    aim=center+jax.random.uniform(keys[3],(2,3),minval=-.05,maxval=.05)
    focal=800*jax.random.uniform(keys[4],(2,),minval=.9,maxval=1.1)
    enabled=j.array([True,jax.random.bernoulli(keys[5],.3)])
    return dict(position=pos,aim=aim,focal=focal,enabled=enabled)


def camera_layout_usable(cameras):
    """Static setup qualification, separate from dynamic finger occlusion.

    The reference tag must be detectable, and the initial cube region must fit
    in frame. A 55 mm half-extent covers a randomly oriented 50 mm cube plus
    its +/-10 mm initial position offsets. Historical sampling is unchanged.
    """
    signs=j.array([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]])
    reference=REFERENCE_TAG_POSITION+j.concatenate((signs*.02,j.zeros((4,1))),axis=-1)
    corners=j.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    region=j.array([.1,0.,.05])+corners*.055
    points=j.concatenate((reference,region))
    def usable(position,aim,focal):
        forward=unit(aim-position);right=unit(j.cross(forward,j.array([0.,1.,0.])));up=j.cross(right,forward)
        vectors=points-position;depth=vectors@forward
        px=focal*(vectors@right)/j.maximum(depth,1e-6)
        py=focal*(vectors@up)/j.maximum(depth,1e-6)
        framed=j.all((depth>0)&(j.abs(px)<640)&(j.abs(py)<360))
        image=j.stack((px[:4],py[:4]),axis=-1)
        side=j.min(j.linalg.norm(image-j.roll(image,1,axis=0),axis=-1))
        cosine=unit(position-REFERENCE_TAG_POSITION)[2]
        return framed&(side>=20)&(cosine>=j.cos(70*j.pi/180))
    return jax.vmap(usable)(cameras['position'],cameras['aim'],cameras['focal'])


@jax.jit
def sample_qualified_cameras(key):
    """Opt-in repair: condition original camera draws on a usable layout.

    Fixed-size rejection sampling is JAX-batchable. The fallback is a declared
    overhead setup; it does not inject a pose or bypass real-time occlusion.
    """
    candidates=jax.vmap(sample_cameras)(jax.random.split(key,32))
    valid=jax.vmap(camera_layout_usable)(candidates)  # [candidate, camera]
    indices=j.argmax(valid,axis=0);available=j.any(valid,axis=0)
    slots=j.arange(2)
    selected={name:value[indices,slots] for name,value in candidates.items() if name!='enabled'}
    fallback=dict(position=j.array([[.10,.07,.50],[.10,.07,.50]]),
                  aim=j.array([[.10,.07,.05],[.10,.07,.05]]),focal=j.array([800.,800.]))
    for name in selected:
        mask=available if selected[name].ndim==1 else available[:,None]
        selected[name]=j.where(mask,selected[name],fallback[name])
    # Keep the original 30% probability for a second enabled camera.
    selected['enabled']=sample_cameras(key)['enabled']
    return selected


def fuse_quaternions(quats,weights):
    # Weighted geodesic medoid for robust rejection; absent entries excluded.
    distances=angle(quats[:,None,:],quats[None,:,:])
    scores=j.sum(distances*weights[None,:],axis=-1)/j.maximum(j.sum(weights),1e-8)
    index=j.argmin(j.where(weights>0,scores,j.inf))
    reference=quats[index]
    kept=weights*(angle(quats,reference)<=15*j.pi/180)
    aligned=quats*j.where(j.sum(quats*reference,axis=-1,keepdims=True)<0,-1.,1.)
    q=unit(j.sum(kept[:,None]*aligned,axis=0))
    return j.where(j.sum(kept)>0,q,IDENTITY),kept


class TagOracle:
    def __init__(self,model):
        self.ids=np.flatnonzero((model.geom_group==3)&(model.geom_type==6))
        self.sizes=j.array(model.geom_size[self.ids])
        self.palm_id=model.body('palm').id
        # A declared fixture tag beside the hand, lying in the horizontal plane.
        self.reference=REFERENCE_TAG_POSITION

    def capture(self,key,cameras,cube_position,cube_quaternion,geom_position,geom_rotation,palm_position,palm_quaternion,*,cube_half_size=.025,geom_sizes=None):
        corners,normals=geometry(cube_position,cube_quaternion,self.reference,cube_half_size)
        sizes=self.sizes if geom_sizes is None else geom_sizes
        keys=jax.random.split(key,2)
        def camera_sample(key,position,aim,focal,enabled):
            center=j.mean(corners,axis=1)
            towards=position-center
            distances=j.linalg.norm(towards,axis=-1)
            cosine=j.sum(normals*unit(towards),axis=-1)
            forward=unit(aim-position)
            right=unit(j.cross(forward,j.array([0.,1.,0.])))
            up=j.cross(right,forward)
            vectors=corners-position
            depth=j.einsum('tki,i->tk',vectors,forward)
            px=focal*j.einsum('tki,i->tk',vectors,right)/j.maximum(depth,1e-6)
            py=focal*j.einsum('tki,i->tk',vectors,up)/j.maximum(depth,1e-6)
            fov=j.all((depth>0)&(j.abs(px)<640)&(j.abs(py)<360),axis=-1)
            image=j.stack((px,py),axis=-1)
            side=j.min(j.linalg.norm(image-j.roll(image,1,axis=1),axis=-1),axis=-1)
            clear=ray_visible(position,corners.reshape((-1,3)),geom_position[self.ids],geom_rotation[self.ids],sizes).reshape((7,4)).all(-1)
            visible=clear&fov&(side>=20)&(cosine>=j.cos(70*j.pi/180))&enabled
            g=j.clip((distances/.4)/j.maximum(cosine,.01),0,4.)
            rot_std=1.5*j.pi/180*g;pos_std=.0015*g
            k1,k2,k3,k4,k5=jax.random.split(key,5)
            # Random-axis noise matches the stated detector oracle (not per-axis Euler noise).
            noise=axis_quat(jax.random.normal(k1,(7,3)),jax.random.normal(k2,(7,))*rot_std)
            flip=jax.random.bernoulli(k3,.01,(6,))
            # Valid large-angle outlier, explicitly not an exact planar PnP solution.
            extra=axis_quat(j.tile(right,(6,1)),j.pi*j.ones(6))
            q=multiply(noise[:6],j.tile(cube_quaternion,(6,1)))
            q=j.where(flip[:,None],multiply(extra,q),q)
            weights=visible[:6]/j.maximum(rot_std[:6]**2,1e-10)
            fused,kept=fuse_quaternions(q,weights)
            accepted=kept>0
            pnoise=jax.random.normal(k4,(7,3))*pos_std[:,None]
            # Double camera-depth standard deviation without changing lateral noise.
            pnoise=pnoise+j.sum(pnoise*forward,axis=-1,keepdims=True)*forward
            pweights=accepted/j.maximum(pos_std[:6]**2,1e-10)
            point=cube_position+j.sum(pnoise[:6]*pweights[:,None],axis=0)/j.maximum(j.sum(pweights),1e-8)
            # Reference error is shared by all cube tags of this camera.
            ref_rotation=matrix(conjugate(noise[6]))
            point=ref_rotation@(point-self.reference-pnoise[6])+self.reference
            fused=multiply(conjugate(noise[6]),fused)
            ok=(j.sum(kept)>0)&visible[6]&(~jax.random.bernoulli(k5,.02))
            # Conservative scalar position precision includes the depth variance.
            rp=1/(1/j.maximum(j.sum(kept),1e-8)+rot_std[6]**2)
            pp=1/(4/j.maximum(j.sum(pweights),1e-8)+4*pos_std[6]**2)
            return fused,point,j.where(ok,rp,0.),j.where(ok,pp,0.),visible
        qs,ps,rp,pp,visibility=jax.vmap(camera_sample)(keys,cameras['position'],cameras['aim'],cameras['focal'],cameras['enabled'])
        q,kept=fuse_quaternions(qs,rp)
        pp=pp*(kept>0)
        pos=j.sum(ps*pp[:,None],axis=0)/j.maximum(j.sum(pp),1e-8)
        valid=j.sum(kept)>0
        q=multiply(conjugate(palm_quaternion),q)
        pos=matrix(palm_quaternion).T@(pos-palm_position)
        return j.concatenate((pos,q,j.array([j.sum(kept),j.sum(pp),valid.astype(j.float32)]))),visibility


def initial_sensor():
    packet=j.concatenate((j.zeros(3),IDENTITY,j.zeros(3)))
    return dict(queue=j.tile(packet,(3,1)),last=packet,age=j.array(20,dtype=j.int32))


def advance_sensor(state,capture,extra_delay):
    queue=j.concatenate((capture[None,:],state['queue'][:2]),axis=0)
    incoming=queue[1+extra_delay]
    valid=incoming[9]>.5
    last=j.where(valid,incoming,state['last'])
    age=j.where(valid,1+extra_delay,state['age']+1)
    # Stale pose is explicitly accompanied by zero precision and increasing age.
    packet=last.at[7:9].set(j.where(valid,incoming[7:9],0.)).at[9].set(valid)
    return dict(queue=queue,last=last,age=age),packet


def features(packet,age):
    return j.concatenate((rot6(packet[3:7]),packet[:3],packet[9:10],j.array([j.minimum(age,20)/20]),
        j.log1p(packet[7:9])))
