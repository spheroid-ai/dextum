"""Cloud CPU z v0: invariant point encoder, set-matched patch graph decoder."""
import os,sys,json,time,hashlib
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import jax
import jax.numpy as j
from flax import linen as nn,serialization
import optax
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import trimesh
SLOTS=96
ATTR=5


def inputs(points,normals):
    # Centered coordinates, normals and neighbour displacements are equivariant.
    # Their norms/dot products are scalar invariant features; distances retain mm.
    p=points-points.mean(0);radius=np.linalg.norm(p,axis=1)
    dist,idx=cKDTree(p).query(p,k=9);idx=idx[:,1:];dist=dist[:,1:]
    ndot=np.sum(normals[:,None,:]*normals[idx],-1)
    radial=np.sum(p*normals,axis=-1)/np.maximum(radius,1e-8)
    return np.column_stack((radius/30,radial,dist/5,ndot)).astype(np.float32)

class Encoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        h=nn.silu(nn.Dense(64)(x));h=nn.silu(nn.Dense(64)(h))
        h=j.concatenate((h.mean(-2),h.max(-2)),-1)
        z=nn.Dense(64,name='z')(nn.silu(nn.Dense(128)(h)))
        h=nn.silu(nn.Dense(128)(z))
        nodes=nn.Dense(SLOTS*(ATTR+1))(h).reshape((-1,SLOTS,ATTR+1))
        edges=nn.Dense(SLOTS*16)(h).reshape((-1,SLOTS,16))
        adj=j.einsum('bik,bjk->bij',edges,edges)/4+nn.Dense(1)(h)[:,None,:]
        return z,nodes,adj


def graph_targets(graph):
    p=graph['patches'];a=[]
    for r in p:
        radius=[min(100,float(x)) if x is not None else 100 for x in r['principal_radii_mm']]
        a.append([r['area_pad_units']/10,float(r['curvature_class']=='gently_curved'),radius[0]/100,radius[1]/100,np.linalg.norm(r['centroid_mm'])/40])
    attrs=np.array(a,np.float32);adj=np.zeros((len(p),len(p)),np.float32)
    for e in graph['edges']:a,b=e['patches'];adj[a,b]=adj[b,a]=1
    return attrs,adj


def match(pred,targets):
    attrs=np.zeros((len(pred),SLOTS,ATTR),np.float32);occ=np.zeros((len(pred),SLOTS),np.float32);adj=np.zeros((len(pred),SLOTS,SLOTS),np.float32)
    for b,(x,(a,e)) in enumerate(zip(pred,targets)):
        cost=((x[:,:ATTR,None]-a.T[None])**2).mean(1)-.1*x[:,ATTR,None]
        rows,cols=linear_sum_assignment(cost);attrs[b,rows]=a[cols];occ[b,rows]=1;adj[b][np.ix_(rows,rows)]=e[np.ix_(cols,cols)]
    return attrs,occ,adj


def main():
    source,graphs,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False);start=time.monotonic()
    specs=json.loads((source/'batch.json').read_text())['assets'];rng=np.random.default_rng(20261011)
    X=[];targets=[];provenance=[]
    for spec in specs:
        mesh=trimesh.load(source/spec['id']/'visual.obj',process=False,force='mesh')
        p,face=trimesh.sample.sample_surface(mesh,2048,seed=20261011)
        n=mesh.face_normals[face];X.append(inputs(p*1000,n))
        path=graphs/spec['id']/'patches.json';targets.append(graph_targets(json.loads(path.read_text())))
        provenance.append(dict(id=spec['id'],split=spec['split'],patch_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(X)==1:
            rot=Rotation.from_rotvec([.8,-.5,.2]).as_matrix();inv_error=float(np.max(np.abs(X[-1]-inputs(p*1000@rot+np.array([7,8,9]),n@rot))))
    assert max(len(t[0]) for t in targets)<=SLOTS
    X=np.asarray(X);train=np.array([i for i,r in enumerate(specs) if r['split']=='train']);test=np.array([i for i,r in enumerate(specs) if r['split']=='held_out'])
    model=Encoder();params=model.init(jax.random.PRNGKey(1),j.asarray(X[:1]));opt=optax.adam(1e-3);state=opt.init(params)
    @jax.jit
    def predict(p,x):return model.apply(p,x)
    @jax.jit
    def update(p,s,x,a,o,e):
        def loss(p):
            z,n,adj=model.apply(p,x)
            attr=j.sum((n[:,:,:ATTR]-a)**2*o[:,:,None])/j.maximum(o.sum()*ATTR,1)
            presence=optax.sigmoid_binary_cross_entropy(n[:,:,ATTR],o).mean()
            pair=o[:,:,None]*o[:,None,:]*(1-j.eye(SLOTS)[None])
            edge=j.sum(optax.sigmoid_binary_cross_entropy(adj,e)*pair)/j.maximum(pair.sum(),1)
            return attr+presence+.25*edge
        loss,g=jax.value_and_grad(loss)(p);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,loss
    for step in range(500):
        idx=rng.choice(train,8,replace=False);_,pred,_=predict(params,X[idx]);a,o,e=match(np.asarray(pred),[targets[i] for i in idx]);params,state,loss=update(params,state,X[idx],a,o,e)
        if step%50==0:
            row=dict(step=step,loss=float(loss),seconds=time.monotonic()-start);print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    z,n,adj=map(np.asarray,predict(params,X));reports={}
    for label,ids in [('train',train),('held_out',test)]:
        a,o,e=match(n[ids],[targets[i] for i in ids]);present=n[ids,:,ATTR]>0;count_error=np.abs(present.sum(1)-o.sum(1))
        pair=o[:,:,None]*o[:,None,:]*(1-np.eye(SLOTS)[None]);edge_pred=adj[ids]>0
        tp=float((edge_pred*e*pair).sum());fp=float((edge_pred*(1-e)*pair).sum());fn=float(((1-edge_pred)*e*pair).sum())
        reports[label]=dict(patch_count_mae=float(count_error.mean()),attribute_mse=float((((n[ids,:,:ATTR]-a)**2)*o[:,:,None]).sum()/max(o.sum()*ATTR,1)),edge_f1=2*tp/max(2*tp+fp+fn,1))
    nearest=[]
    for i in test:
        k=train[np.argmin(np.linalg.norm(z[train]-z[i],axis=1))];nearest.append(dict(query=specs[i]['id'],train_neighbor=specs[k]['id'],distance=float(np.linalg.norm(z[k]-z[i]))))
    np.savez(out/'embeddings.npz',z=z,ids=np.array([r['id'] for r in specs]),train_indices=train,heldout_indices=test)
    (out/'encoder.msgpack').write_bytes(serialization.to_bytes(params))
    result=dict(state='COMPLETE',attempt=1,seed=1,steps=500,train_objects=len(train),held_out_objects=len(test),input_points=2048,z_width=64,slots=SLOTS,invariance_max_error=inv_error,metrics=reports,nearest_neighbors=nearest,provenance=provenance,seconds=time.monotonic()-start,
      T6='NOT_MEASURED: held-out feasibility tables do not yet exist; patch reconstruction is not a T6 substitute',limits=['40 training objects only; no hyperparameter or checkpoint selection on held-out data','Scalar invariant point MLP from equivariant geometry features; no pose, mesh or simulator required by exported actor','No per-quantum policy is yet conditioned on this encoder'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('provenance','nearest_neighbors')}),flush=True)
if __name__=='__main__':main()
