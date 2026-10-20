"""Shared camera/goal rotations. Quaternions are scalar-first; no cube symmetry."""
import jax
import jax.numpy as j


def unit(x):
    return x/j.maximum(j.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def conjugate(q):
    return q*j.array([1.,-1.,-1.,-1.])


def multiply(a,b):
    return j.concatenate((a[..., :1]*b[..., :1]-j.sum(a[...,1:]*b[...,1:],axis=-1,keepdims=True),
        a[..., :1]*b[...,1:]+b[..., :1]*a[...,1:]+j.cross(a[...,1:],b[...,1:])),axis=-1)


def axis_quat(axis,angle):
    return j.concatenate((j.cos(angle/2)[...,None],unit(axis)*j.sin(angle/2)[...,None]),axis=-1)


def angle(a,b):
    q=multiply(a,conjugate(b))
    return 2*j.arctan2(j.linalg.norm(q[...,1:],axis=-1),j.abs(q[...,0]))


def matrix(q):
    w,x,y,z=j.moveaxis(unit(q),-1,0)
    return j.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                    2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                    2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))


def rot6(q):
    return matrix(q)[...,:,:2].reshape(q.shape[:-1]+(6,))


def sample_goal(key,current,phase):
    k1,k2,k3=jax.random.split(key,3)
    if phase==0:
        axis=j.array([0.,0.,1.]);theta=j.array(j.pi/6)
    elif phase==1:
        axis=j.array([0.,0.,1.]);theta=jax.random.choice(k1,j.array([30.,90.,180.]))*j.pi/180
        theta*=j.where(jax.random.bernoulli(k2),1.,-1.)
    elif phase==2:
        axis=unit(jax.random.normal(k1,(3,)));theta=jax.random.uniform(k2,minval=j.pi/6,maxval=j.pi/4)
    else:
        # Uniform SO(3), conditioned on geodesic displacement >=90 degrees.
        axis=unit(jax.random.normal(k1,(3,)))
        low=j.pi/2;high=j.pi
        v=jax.random.uniform(k2,minval=low-j.sin(low),maxval=high-j.sin(high))
        def bisect(_,bounds):
            a,b=bounds;mid=(a+b)/2
            return j.where(mid-j.sin(mid)<v,mid,a),j.where(mid-j.sin(mid)<v,b,mid)
        low,high=jax.lax.fori_loop(0,24,bisect,(low,high));theta=(low+high)/2
    return unit(multiply(axis_quat(axis,theta),current))


def task_step(previous_error,error,hold,elapsed,action_delta,omega,drop,*,tolerance=.2,hold_steps=5,recipe='vg1'):
    """Truth only for reward/termination. A success cannot coincide with a drop."""
    within=(error<tolerance)&(~drop)
    new_hold=j.where(within,hold+1,0)
    success=(new_hold==hold_steps)&(~drop)
    time=elapsed+1
    timeout=(time>=280)&(~success)
    progress=2*(previous_error-error)
    holding=.05*.5*within
    settling=-.05*.05*j.sum(omega**2)*within
    if recipe not in ('vg1','vg2'):
        raise ValueError('Unregistered visual reward recipe')
    # VG2 borrows the continuous orientation tolerance and action-rate scale
    # from the pinned Playground LEAP reorientation task. VG1 stays replayable.
    alignment=.25*j.clip(1-j.maximum(error-tolerance,0.)/j.pi,0.,1.) if recipe=='vg2' else j.zeros(())
    rate=-(.00005 if recipe=='vg2' else .05)*j.sum(action_delta**2)
    falling=-5.*drop
    bonus=5.*success
    return dict(hold=new_hold,time=time,success=success,timeout=timeout,
        reward=alignment+progress+holding+settling+rate+falling+bonus,
        alignment=alignment,progress=progress,holding=holding,settling=settling,action_rate=rate,falling=falling,bonus=bonus)
