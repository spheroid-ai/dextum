import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pass1c.conformal_palm import hull_exit
root=Path('/root/dextum/dextum_cursor_agent/runs/pass1_matched_01');cad=json.loads(Path('/root/dextum/kit_b/geometry.json').read_text());skin=next(p for p in cad['parts'] if p.get('module')=='palm' and p.get('layer')=='skin');v=np.array(skin['vertices']);tri=v[np.array(skin['faces'])];a=np.load(root/'geometry/measurements.npz');base=a['palm_base_mm'];normal=a['palm_normal'];cells=[c for c in cad['cells'] if c['module']=='palm'];cn=np.array([c['normal'] for c in cells]);cb=np.array([c['position_mm'] for c in cells])-5.25*cn
parts=[]; rows=[]
regions=[(v[:,2].min()-1,53.5,-100,100),(53.5,68.5,-100,100)]+[(68.5,v[:,2].max()+1,l,h) for l,h in [(-18,0),(0,18),(18,100)]]+[(l,h,-100,-18) for l,h in [(68.5,77),(77,82),(82,v[:,2].max()+1)]]
# Three strips follow the three existing palm rows; no per-cell collider.
while regions:
 lo,hi,xlo,xhi=regions.pop(0)
 pts=[]
 for t in tri:
  poly=list(t)
  for axis,cut,sign in [(2,lo,1),(2,hi,-1),(0,xlo,1),(0,xhi,-1)]:
   new=[]
   for p,q in zip(poly,poly[1:]+poly[:1]):
    pp=(p[axis]-cut)*sign>=0;qq=(q[axis]-cut)*sign>=0
    if pp:new.append(p)
    if pp!=qq:new.append(p+(q-p)*(cut-p[axis])/(q[axis]-p[axis]))
   poly=new
  pts.extend(poly)
 pts=np.unique(np.round(pts,8),axis=0);pts=pts[ConvexHull(pts).vertices];sel=sorted(set(np.r_[np.argmin(pts,axis=0),np.argmax(pts,axis=0)].tolist()))
 while len(sel)<32:
  h=ConvexHull(pts[sel]);dist=(pts@h.equations[:,:3].T+h.equations[:,3]).max(1)
  if dist.max()<.1:break
  sel.append(int(dist.argmax()))
 chosen=pts[sel]
 vals=hull_exit(base,normal,chosen)
 peak=float(np.nanmax(vals))
 if peak>6.25:
  assert len(parts)+len(regions)<32, 'Conformal geometry subdivision exceeds bounded construction'
  ll=max(xlo,float(v[:,0].min()));hh=min(xhi,float(v[:,0].max()))
  if hh-ll>hi-lo:
   mid=(ll+hh)/2;regions.extend([(lo,hi,xlo,mid),(lo,hi,mid,xhi)])
  else:
   mid=(lo+hi)/2;regions.extend([(lo,mid,xlo,xhi),(mid,hi,xlo,xhi)])
  continue
 parts.append(chosen);rows.append(dict(z_bounds=[lo,hi],x_bounds=[xlo,xhi],vertices=len(chosen),max_mm=peak))
 print('accepted piece',len(parts),rows[-1],flush=True)
allv=np.array([hull_exit(base,normal,p) for p in parts]);vals=np.max(np.where(np.isfinite(allv),allv,-np.inf),axis=0);cv=np.array([hull_exit(cb,cn,p) for p in parts]);cs=np.max(np.where(np.isfinite(cv),cv,-np.inf),axis=0)
r=dict(regions=rows,parts=[p.tolist() for p in parts],max_mm=float(vals.max()),coverage=int(np.isfinite(vals).sum()),total=len(vals),cells_mm=cs.tolist());(root/'palm_accepted_geometry.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='parts'},flush=True)
