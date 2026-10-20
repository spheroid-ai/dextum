"""Cloud re-segmentation; meshes/views and historical patch graphs are immutable."""
import argparse,hashlib,json,math,os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast.ai only')
import numpy as np
from scipy.spatial import cKDTree
import trimesh

def patch_graph(mesh,friction):
    """Review section 9: flat normals; curved radius continuity, sharp boundaries."""
    xyz=np.asarray(mesh.triangles_center)*1000.;normal=np.asarray(mesh.face_normals)
    tree=cKDTree(xyz);curv=np.zeros((len(xyz),2));fit_counts=[]
    for i,(p,n) in enumerate(zip(xyz,normal)):
        ids=np.asarray(tree.query_ball_point(p,3.),dtype=int)
        ids=ids[normal[ids]@n>math.cos(math.radians(45))];fit_counts.append(len(ids))
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
    area=np.asarray(mesh.area_faces)*1e6
    neighbours=[[] for _ in xyz]
    for a,b in mesh.face_adjacency:neighbours[a].append(b);neighbours[b].append(a)
    labels=np.full(len(xyz),-1,int);groups=[];threshold=math.cos(math.radians(15))
    for seed in range(len(xyz)):
        if labels[seed]>=0 or kind[seed]==2:continue
        label=len(groups);labels[seed]=label;members=[seed];total=normal[seed]*area[seed];queue=[seed]
        while queue:
            current=queue.pop()
            for nxt in neighbours[current]:
                if labels[nxt]>=0 or kind[nxt]!=kind[seed]:continue
                if kind[seed]==0:
                    if normal[nxt]@(total/np.linalg.norm(total))<threshold:continue
                else:
                    # Compare principal radii across the local shared edge.
                    # Treat numerically planar secondary curvature as infinite.
                    ka,kb=curv[current],curv[nxt]
                    finite_a,finite_b=ka>=1e-3,kb>=1e-3
                    if np.any(finite_a!=finite_b):continue
                    ratio=np.where(finite_a,ka/np.maximum(kb,1e-12),1.)
                    if np.any((ratio<1/1.3)|(ratio>1.3)):continue
                labels[nxt]=label;members.append(nxt);total+=normal[nxt]*area[nxt];queue.append(nxt)
        groups.append(members)
    if not groups:raise ValueError('No non-sharp contact patches')
    # Sharp samples remain marked in the graph. Nearest non-sharp labels are
    # used only to form a partition for surface area/adjacency bookkeeping.
    support=labels>=0
    _,near=cKDTree(xyz[support]).query(xyz[~support]);labels[~support]=labels[support][near]
    edges=np.asarray(mesh.face_adjacency)
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
        mask=labels==k;w=area[mask];mean=np.average(normal[mask],axis=0,weights=w);mean=mean/np.linalg.norm(mean) if np.linalg.norm(mean)>1e-6 else np.zeros(3)
        principal=np.average(curv[mask],axis=0,weights=w);dominant=int(np.argmax(np.abs(mean)))
        patches.append(dict(id=k,name=('positive_' if mean[dominant]>=0 else 'negative_')+'xyz'[dominant]+f'_{k}',
            mean_normal=mean.tolist(),mean_normal_valid=bool(np.linalg.norm(mean)>.5),centroid_mm=np.average(xyz[mask],axis=0,weights=w).tolist(),
            area_pad_units=float(w.sum()/192),curvature_class='flat' if np.average(kind[mask]==0,weights=w)>.5 else 'gently_curved',
            principal_radii_mm=[float(1/x) if x>1e-9 else None for x in principal],friction_class='uniform_synthetic',
            friction=friction,surface_face_ids=np.flatnonzero(mask).tolist()))
    adjacency={}
    for a,b in edges:
        x,y=int(labels[a]),int(labels[b])
        if x==y:continue
        key=tuple(sorted((x,y)));row=adjacency.setdefault(key,dict(patches=list(key),shared_boundary_length_mm=0.,sharp=False))
        row['sharp'] |= bool(kind[a]==2 or kind[b]==2)
    for pair, edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        x,y=map(int,labels[pair])
        if x!=y:
            adjacency[tuple(sorted((x,y)))]['shared_boundary_length_mm']+=float(np.linalg.norm(mesh.vertices[edge[0]]-mesh.vertices[edge[1]])*1000)
    # Exact nearest-mesh distance in a 1-mm band around surface support samples.
    samples=[];sample_patch=[]
    for k in range(len(patches)):
        ids=np.flatnonzero(labels==k);ids=ids[np.linspace(0,len(ids)-1,min(32,len(ids))).astype(int)]
        for offset in (-.001,0.,.001):
            samples.extend(xyz[ids]/1000+offset*normal[ids]);sample_patch.extend([k]*len(ids))
    samples=np.asarray(samples);closest,distance,face_id=trimesh.proximity.closest_point(mesh,samples)
    side=np.sign(np.sum((samples-closest)*mesh.face_normals[face_id],axis=1))
    sdf=dict(points_m=samples.tolist(),distance_m=(distance*side).tolist(),patch_id=sample_patch,
             convention='positive outside; exact nearest triangle distance to exported mesh; 1-mm local band')
    for (x,y),row in adjacency.items():
        row['dihedral_deg']=float(np.degrees(np.arccos(np.clip(np.dot(patches[x]['mean_normal'],patches[y]['mean_normal']),-1,1))))
    return dict(source='computed_faces_3mm_quadratic_section9_v1',patches=patches,edges=list(adjacency.values()),
        restricted_sdf=sdf,sharp_sample_ids=np.flatnonzero(kind==2).tolist(),fit_neighbour_count_min=min(fit_counts),
        sharp_fraction=float(np.mean(kind==2)),limits=['15-degree mean-normal rule applies only to flat samples; curved joins use local principal-radius ratio <=1.3',
        'face-centered fits exclude neighbours beyond 45 degrees to avoid fitting across sharp edges; convergence not measured',
        'uniform friction only; heterogeneous patch materials are not implemented'])


def dense_box():
    mesh=trimesh.creation.box(extents=[.05,.05,.05])
    v,f=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.0016,max_iter=8)
    return trimesh.Trimesh(v,f,process=True)


def dense_cylinder():
    mesh=trimesh.creation.cylinder(radius=.020,height=.080,sections=128)
    v,f=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.0018,max_iter=8)
    return trimesh.Trimesh(v,f,process=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for spec in json.loads((a.assets/'batch.json').read_text())['assets']:
        source=a.assets/spec['id'];mesh=trimesh.load(source/'visual.obj',force='mesh',process=False)
        graph=patch_graph(mesh,spec['friction']);folder=a.output/spec['id'];folder.mkdir()
        (folder/'patches.json').write_text(json.dumps(graph)+'\n')
        row=dict(id=spec['id'],split=spec['split'],patches=len(graph['patches']),mesh_sha256=hashlib.sha256((source/'visual.obj').read_bytes()).hexdigest(),
            original_patch_sha256=hashlib.sha256((source/'patches.json').read_bytes()).hexdigest(),new_patch_sha256=hashlib.sha256((folder/'patches.json').read_bytes()).hexdigest())
        rows.append(row);print(json.dumps(row),flush=True)
        (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    sanity=[]
    for name,mesh,target in [('sphere',trimesh.creation.icosphere(subdivisions=5,radius=.025),1),('cylinder',dense_cylinder(),3),('cube',dense_box(),6)]:
        graph=patch_graph(mesh,.5);n=len(graph['patches']);sanity.append(dict(name=name,patches=n,target=target,pass_target=n==target))
        (a.output/(name+'_sanity.json')).write_text(json.dumps(graph)+'\n')
    sweep=[]
    directions=np.asarray(trimesh.creation.icosphere(subdivisions=5).vertices);faces=trimesh.creation.icosphere(subdivisions=5).faces
    for exponent in np.linspace(.2,1.,9):
        radial=(np.abs(directions)**(2/exponent)).sum(1)**(-exponent/2)
        mesh=trimesh.Trimesh(directions*radial[:,None]*.025,faces,process=False)
        graph=patch_graph(mesh,.5);n=len(graph['patches']);sweep.append(dict(exponent=float(exponent),patches=n,within_target=1<=n<=26))
    result=dict(state='RESEGMENTED',rows=rows,sanity=sanity,sweep=sweep,
        sanity_pass=all(x['pass_target'] for x in sanity) and all(x['within_target'] for x in sweep),
        rule='Flat <=15deg; gently curved local principal-radius max/min <=1.3; no curved normal limit; planar second curvature <.001/mm treated infinite',
        limits=['Sanity counts are scorecard observations, not new hard stops','No mesh, visual, old graph or old result changed','Physical interface geometry; no Metriplector latent labels'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
