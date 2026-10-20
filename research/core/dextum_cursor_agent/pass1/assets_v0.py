"""Cloud-only F1 batch specification and physical export from Blender meshes.

Uniform materials and zero COM offset are the registered Pass 1 v0 subset.
Patch segmentation follows the numerical rule, including its 15-degree normal
threshold; the spec's one-patch sphere example conflicts with that rule.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Asset generation and numerical validation run on Vast.ai only')
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import beta, gamma
import trimesh

MATERIALS = (
    dict(name='synthetic_wood', density_kg_m3=650., color=[.42,.19,.065,1.], roughness=.65),
    dict(name='synthetic_polymer', density_kg_m3=1050., color=[.045,.26,.52,1.], roughness=.3),
    dict(name='synthetic_rubber', density_kg_m3=1100., color=[.09,.12,.12,1.], roughness=.85),
)


def truth(a,b,c,e1,e2,density):
    area=4*a*b*gamma(1+e2/2)**2/gamma(1+e2)
    volume=area*c*e1*beta(e1/2,e1+1)
    xy_factor=4/3*gamma(1+1.5*e2)*gamma(1+.5*e2)/gamma(1+2*e2)
    x2=a**3*b*xy_factor*c*e1*beta(e1/2,2*e1+1)
    y2=b**3*a*xy_factor*c*e1*beta(e1/2,2*e1+1)
    z2=area*c**3*e1*beta(1.5*e1,e1+1)
    return dict(volume_m3=float(volume),mass_kg=float(volume*density),
                inertia_kg_m2=[float(density*(y2+z2)),float(density*(x2+z2)),float(density*(x2+y2))])


def generate_batch(output,seed=20261011):
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed);assets=[]
    while len(assets)<50:
        i=len(assets);axes=rng.uniform(.015,.045,3);e1,e2=rng.uniform(.2,1.4,2)
        if i==0:axes=np.array([.025]*3);e1=e2=1.
        if i==1:axes=np.array([.025]*3);e1=e2=.2
        material=MATERIALS[i%len(MATERIALS)]
        physical=truth(*axes,e1,e2,material['density_kg_m3'])
        if not .02<=physical['mass_kg']<=.3:continue
        assets.append(dict(id=f'f1_{i:03d}',family='F1',semi_axes_m=axes.tolist(),epsilon=[float(e1),float(e2)],
            material=material,friction=float(rng.uniform(.3,1.2)),split='train' if i<40 else 'held_out',
            seed=seed,stage_first_used=0,generator_truth=physical,com_offset_m=[0.,0.,0.],compliance='rigid'))
    (output/'batch.json').write_text(json.dumps(dict(seed=seed,assets=assets),indent=2)+'\n')
    return assets


def patch_graph(mesh,friction):
    """3-mm normal-frame quadratic fits, adjacency region growth and area merge."""
    xyz=np.asarray(mesh.vertices)*1000.;normal=np.asarray(mesh.vertex_normals)
    tree=cKDTree(xyz);curv=np.zeros((len(xyz),2));fit_counts=[]
    for i,(p,n) in enumerate(zip(xyz,normal)):
        ids=tree.query_ball_point(p,3.);fit_counts.append(len(ids))
        if len(ids)<6:raise ValueError('Mesh too sparse for registered 3-mm curvature fit')
        d=xyz[ids]-p
        axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
        x,y,z=d@u,d@v,d@n
        A=np.column_stack((x*x,x*y,y*y,x,y,np.ones(len(x))))
        q=np.linalg.lstsq(A,z,rcond=None)[0]
        # Generalized shape operator accounts for residual tangent-plane slope.
        I=np.array([[1+q[3]**2,q[3]*q[4]],[q[3]*q[4],1+q[4]**2]])
        II=np.array([[2*q[0],q[1]],[q[1],2*q[2]]])/math.sqrt(1+q[3]**2+q[4]**2)
        curv[i]=np.sort(np.abs(np.linalg.eigvals(np.linalg.solve(I,II)).real))[::-1]
    kind=np.where(curv[:,0]<1/30,0,np.where(curv[:,0]<1/6,1,2))
    area=np.zeros(len(xyz));np.add.at(area,mesh.faces.ravel(),np.repeat(mesh.area_faces*1e6/3,3))
    neighbours=[[] for _ in xyz]
    for a,b in mesh.edges_unique:neighbours[a].append(b);neighbours[b].append(a)
    labels=np.full(len(xyz),-1,int);groups=[];threshold=math.cos(math.radians(15))
    for seed in range(len(xyz)):
        if labels[seed]>=0 or kind[seed]==2:continue
        label=len(groups);labels[seed]=label;members=[seed];total=normal[seed]*area[seed];queue=[seed]
        while queue:
            current=queue.pop()
            for nxt in neighbours[current]:
                if labels[nxt]>=0 or kind[nxt]!=kind[seed]:continue
                if normal[nxt]@(total/np.linalg.norm(total))<threshold:continue
                labels[nxt]=label;members.append(nxt);total+=normal[nxt]*area[nxt];queue.append(nxt)
        groups.append(members)
    if not groups:raise ValueError('No non-sharp contact patches')
    # Sharp samples remain marked in the graph. Nearest non-sharp labels are
    # used only to form a partition for surface area/adjacency bookkeeping.
    support=labels>=0
    _,near=cKDTree(xyz[support]).query(xyz[~support]);labels[~support]=labels[support][near]
    edges=np.asarray(mesh.edges_unique)
    # Merge sub-pad patches into an adjacent patch with closest mean normal.
    while True:
        unique=np.unique(labels);areas={int(k):float(area[labels==k].sum()) for k in unique}
        tiny=[k for k in unique if areas[int(k)]<48.]
        if not tiny or len(unique)==1:break
        k=min(tiny,key=lambda k:areas[int(k)]);incident=edges[np.any(labels[edges]==k,axis=1)]
        adjacent=set(labels[incident].ravel())-{k}
        if not adjacent:raise ValueError('Disconnected patch cannot be merged')
        means={j:np.average(normal[labels==j],axis=0,weights=area[labels==j]) for j in adjacent|{k}}
        dest=max(adjacent,key=lambda j:np.dot(means[k],means[j])/(np.linalg.norm(means[k])*np.linalg.norm(means[j])))
        labels[labels==k]=dest
    remap={k:i for i,k in enumerate(np.unique(labels))};labels=np.array([remap[k] for k in labels])
    patches=[]
    for k in range(len(remap)):
        mask=labels==k;w=area[mask];mean=np.average(normal[mask],axis=0,weights=w);mean/=np.linalg.norm(mean)
        principal=np.average(curv[mask],axis=0,weights=w);dominant=int(np.argmax(np.abs(mean)))
        patches.append(dict(id=k,name=('positive_' if mean[dominant]>=0 else 'negative_')+'xyz'[dominant]+f'_{k}',
            mean_normal=mean.tolist(),centroid_mm=np.average(xyz[mask],axis=0,weights=w).tolist(),
            area_pad_units=float(w.sum()/192),curvature_class='flat' if np.average(kind[mask]==0,weights=w)>.5 else 'gently_curved',
            principal_radii_mm=[float(1/x) if x>1e-9 else None for x in principal],friction_class='uniform_synthetic',
            friction=friction,surface_vertex_ids=np.flatnonzero(mask).tolist()))
    adjacency={}
    for a,b in edges:
        x,y=int(labels[a]),int(labels[b])
        if x==y:continue
        key=tuple(sorted((x,y)));row=adjacency.setdefault(key,dict(patches=list(key),shared_boundary_length_mm=0.,sharp=False))
        row['sharp'] |= bool(kind[a]==2 or kind[b]==2)
    # Piecewise-linear segmentation boundary within each triangle.
    for face in mesh.faces:
        cuts={}
        for a,b in zip(face,np.roll(face,-1)):
            x,y=int(labels[a]),int(labels[b])
            if x!=y:cuts.setdefault(tuple(sorted((x,y))),[]).append((xyz[a]+xyz[b])/2)
        center=xyz[face].mean(0)
        for key,points in cuts.items():
            length=np.linalg.norm(points[0]-points[1]) if len(points)==2 else np.linalg.norm(points[0]-center)
            adjacency[key]['shared_boundary_length_mm']+=float(length)
    # Exact nearest-mesh distance in a 1-mm band around surface support samples.
    samples=[];sample_patch=[]
    for k in range(len(patches)):
        ids=np.flatnonzero(labels==k);ids=ids[np.linspace(0,len(ids)-1,min(32,len(ids))).astype(int)]
        for offset in (-.001,0.,.001):
            samples.extend(mesh.vertices[ids]+offset*normal[ids]);sample_patch.extend([k]*len(ids))
    samples=np.asarray(samples);closest,distance,face_id=trimesh.proximity.closest_point(mesh,samples)
    side=np.sign(np.sum((samples-closest)*mesh.face_normals[face_id],axis=1))
    sdf=dict(points_m=samples.tolist(),distance_m=(distance*side).tolist(),patch_id=sample_patch,
             convention='positive outside; exact nearest triangle distance to exported mesh; 1-mm local band')
    for (x,y),row in adjacency.items():
        row['dihedral_deg']=float(np.degrees(np.arccos(np.clip(np.dot(patches[x]['mean_normal'],patches[y]['mean_normal']),-1,1))))
    return dict(source='computed_mesh_3mm_quadratic_v0',patches=patches,edges=list(adjacency.values()),
        restricted_sdf=sdf,sharp_sample_ids=np.flatnonzero(kind==2).tolist(),fit_neighbour_count_min=min(fit_counts),
        sharp_fraction=float(np.mean(kind==2)),limits=['15-degree mean-normal rule subdivides spheres/cylinders; conflicts with canonical one-patch examples',
        'shared boundaries use piecewise-linear triangle partitions; convergence not yet measured',
        'uniform friction only; heterogeneous patch materials are not implemented'])


def export_one(folder,spec):
    mesh=trimesh.load(folder/'visual.obj',force='mesh',process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    mesh.density=spec['material']['density_kg_m3'];hull=mesh.convex_hull
    hull.export(folder/'collision.obj')
    printable=mesh.copy();printable.apply_scale(1000);printable.export(folder/'print_mm.stl')
    expected=spec['generator_truth'];mass=float(mesh.mass);inertia=np.diag(mesh.moment_inertia)
    mass_error=abs(mass/expected['mass_kg']-1);inertia_error=float(np.max(np.abs(inertia/expected['inertia_kg_m2']-1)))
    # F1 exponents <=1.4 imply convex superellipsoids. Every visual vertex must
    # lie inside the collision hull to floating-point tolerance.
    rng=np.random.default_rng(20261012);direction=rng.normal(size=(4096,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    e1,e2=spec['epsilon'];axes=np.asarray(spec['semi_axes_m'])
    radial=((np.abs(direction[:,0])**(2/e2)+np.abs(direction[:,1])**(2/e2))**(e2/e1)+np.abs(direction[:,2])**(2/e1))**(-e1/2)
    independent_surface=direction*radial[:,None]*axes
    _,distance,_=trimesh.proximity.closest_point(hull,independent_surface)
    collision_max_mm=float(np.max(distance)*1000)
    tensor_error=float(np.linalg.norm(mesh.moment_inertia-np.diag(expected['inertia_kg_m2']))/np.linalg.norm(expected['inertia_kg_m2']))

    graph=patch_graph(mesh,spec['friction']);(folder/'patches.json').write_text(json.dumps(graph)+'\n')
    validation=dict(watertight=True,mass_relative_error=mass_error,inertia_relative_error=inertia_error,
        mass_within_1pct=mass_error<=.01,inertia_within_2pct=inertia_error<=.02 and tensor_error<=.02,
        inertia_tensor_relative_error=tensor_error,com_offset_from_analytic_mm=float(np.linalg.norm(mesh.center_mass)*1000),
        volume_m3=float(mesh.volume),convex_hulls=1,
        collision_sample_max_mm=collision_max_mm,collision_sample_within_1mm=collision_max_mm<=1.,collision_samples=4096)
    manifest=dict(**{k:v for k,v in spec.items() if k!='friction'},visual=dict(mesh='visual.glb',source_mesh='visual.obj',views='views/*.png',expected_views=24),
        collision=dict(hulls=['collision.obj'],method='convex F1 single hull'),
        print_mesh=dict(file='print_mm.stl',units='millimetres',scale_from_simulation=1000,physical_weight='unmeasured; weigh the printed object'),
        inertial=dict(mass_kg=mass,inertia_kg_m2=mesh.moment_inertia.tolist(),com_m=mesh.center_mass.tolist()),
        patches='patches.json',friction=dict(uniform=spec['friction']),articulation=[],validation=validation,
        status='v0 subset: rigid uniform materials, zero COM offset; patch-boundary convergence outstanding')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import mujoco
    # Export the full tensor through its explicit principal frame. MuJoCo's
    # automatic fullinertia diagonalizer introduces ~2.5e-7 relative error on
    # f1_002; this preserves the registered 1e-8 round-trip tolerance.
    I=mesh.moment_inertia;principal,basis=np.linalg.eigh(I)
    if np.linalg.det(basis)<0:basis[:,0]*=-1
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,basis.ravel())
    diagonal=' '.join(f'{x:.17g}' for x in principal)
    orientation=' '.join(f'{x:.17g}' for x in quat)
    com=' '.join(f'{x:.17g}' for x in mesh.center_mass)
    xml=f'<mujoco><compiler meshdir="."/><asset><mesh name="object" file="collision.obj"/></asset><worldbody><body name="object" pos="0 0 .1"><freejoint/><inertial pos="{com}" mass="{mass:.17g}" diaginertia="{diagonal}" quat="{orientation}"/><geom type="mesh" mesh="object" friction="{spec["friction"]:.12g} .005 .0001"/></body></worldbody></mujoco>'
    (folder/'model.xml').write_text(xml+'\n')
    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
    validation['mujoco_load']=True;validation['mujoco_mass_error']=abs(float(model.body_mass[1])/mass-1)
    rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[1]);rotation=rotation.reshape(3,3)
    loaded=rotation@np.diag(model.body_inertia[1])@rotation.T
    validation['mujoco_tensor_relative_error']=float(np.linalg.norm(loaded-I)/np.linalg.norm(I))
    assert validation['mujoco_mass_error']<1e-8 and validation['mujoco_tensor_relative_error']<1e-8
    manifest['validation']=validation;(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(id=spec['id'],split=spec['split'],patch_count=len(graph['patches']),**validation)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['spec','export']);p.add_argument('output',type=Path);a=p.parse_args()
    if a.mode=='spec':generate_batch(a.output);return
    specs=json.loads((a.output/'batch.json').read_text())['assets'];rows=[]
    for spec in specs:
        rows.append(export_one(a.output/spec['id'],spec));print(json.dumps(rows[-1]),flush=True)
        (a.output/'export_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(state='EXPORTED_V0_SUBSET',n=len(rows),rows=rows,training_attempts=0)
    (a.output/'export_result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
