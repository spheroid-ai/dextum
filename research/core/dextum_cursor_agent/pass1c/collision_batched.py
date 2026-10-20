# Copyright 2023 DeepMind Technologies Limited (upstream routines).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Candidate batching adapter for the pinned MJX collision implementation.

Runs identical convex functions on equal-shaped mesh data together. Each
original group's broad-phase cap, contact parameters and output order survive.
No hull simplification, solver change, new contact pruning or XLA flags.
Adopted for the JAX fallback by review section 11; historical releases preserved.
"""
import jax
import jax.numpy as j
from mujoco.mjx._src import collision_driver as driver,mesh
from mujoco.mjx._src.collision_types import GeomInfo,ConvexInfo
from mujoco.mjx._src.types import GeomType

REFERENCE=driver.collision

def install():
    """Install once before tracing; only the pinned, measured implementation."""
    import mujoco
    if mujoco.__version__ != '3.3.6':
        raise RuntimeError('Collision adapter is qualified only for MuJoCo 3.3.6')
    driver.collision = collision

def _payload(m,d,key,geom):
    infos=[]
    for side in (0,1):
        ids=geom[:,side]
        info=GeomInfo(d.geom_xpos[ids],d.geom_xmat[ids],m.geom_size[ids])
        batched={'pos','mat','size'}
        if key.types[side]==GeomType.MESH:
            info=ConvexInfo(**vars(info),**vars(m._impl.mesh_convex[key.data_ids[side]]))
        elif key.types[side]==GeomType.BOX:
            info=mesh.box(info);batched.update(('face','vert'))
        info=type(info)(**{name:value if name in batched else j.broadcast_to(value,(ids.shape[0],)+value.shape)
                          for name,value in vars(info).items()})
        infos.append(info)
    return infos

def collision(m,d):
    if d._impl.ncon==0:return d
    max_pairs=driver._numeric(m,'max_geom_pairs')
    max_points=driver._numeric(m,'max_contact_points')
    groups=driver._contact_groups(m,d)
    buckets={}
    for key,contact in groups.items():
        if max_pairs>-1 and contact.geom.shape[0]>max_pairs and not set(key.types)&driver._GEOM_NO_BROADPHASE:
            p1,p2=d.geom_xpos[contact.geom.T]
            s1,s2=m.geom_rbound[contact.geom.T]
            dist=jax.vmap(j.linalg.norm)(p2-p1)-(s1+s2)
            _,idx=jax.lax.top_k(-dist,k=max_pairs)
            contact=jax.tree.map(lambda value:value[idx],contact)
            groups[key]=contact
        fn=driver._COLLISION_FUNC[key.types]
        if fn.__module__.endswith('collision_convex') and GeomType.HFIELD not in key.types:
            closure=dict(zip(fn.__code__.co_freevars,(c.cell_contents for c in fn.__closure__)))
            raw=closure['collision_fn']
            infos=_payload(m,d,key,contact.geom)
            signature=tuple(tuple((name,value.shape[1:]) for name,value in vars(info).items()) for info in infos)
            buckets.setdefault((raw,signature,fn.ncon),[]).append((key,infos))
        else:
            dist,pos,frame=fn(m,d,key,contact.geom)
            if fn.ncon>1:contact=jax.tree.map(lambda value:j.repeat(value,fn.ncon,axis=0),contact)
            groups[key]=contact.replace(dist=dist,pos=pos,frame=frame)
    for (fn,_,ncon),items in buckets.items():
        infos=[jax.tree.map(lambda *values:j.concatenate(values),*(entry[1][side] for entry in items)) for side in (0,1)]
        values=jax.vmap(fn)(*infos)
        if ncon>1:values=jax.tree.map(j.concatenate,values)
        offset=0
        for key,_ in items:
            contact=groups[key];count=contact.geom.shape[0]*ncon
            dist,pos,frame=jax.tree.map(lambda value:value[offset:offset+count],values)
            if ncon>1:contact=jax.tree.map(lambda value:j.repeat(value,ncon,axis=0),contact)
            groups[key]=contact.replace(dist=dist,pos=pos,frame=frame)
            offset+=count
    # Original insertion order is restored before contact selection, including
    # top_k ties; batching never changes a group's candidate allocation.
    by_dim={}
    for key,contact in groups.items():by_dim.setdefault(key.condim,[]).append(contact)
    contacts=[]
    for dim in sorted(by_dim):
        contact=jax.tree.map(lambda *values:j.concatenate(values),*by_dim[dim])
        if max_points>-1 and contact.geom.shape[0]>max_points:
            _,idx=jax.lax.top_k(-contact.dist,k=max_points)
            contact=jax.tree.map(lambda value:value[idx],contact)
        contacts.append(contact)
    return d.tree_replace({'_impl.contact':jax.tree.map(lambda *values:j.concatenate(values),*contacts)})
