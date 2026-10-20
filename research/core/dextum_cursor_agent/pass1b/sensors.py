"""200-Hz Kit B packets. Physics is transduced here; actor sees only noisy packets.
PVDF bands are declared envelope proxies, not resolved high-frequency vibration.
"""
import os
if not os.environ.get('CONTAINER_ID'): raise RuntimeError('Vast.ai only')
import jax
import jax.numpy as j
import numpy as np
from mujoco.mjx._src import support
from dextum import touch_mjx

FEATURES=200
DT=.005

class Sensors:
    def __init__(self,m,cells,truth):
        self.sites=np.array([m.site('kit_site_'+c['name']).id for c in cells])
        self.geoms=np.array([m.geom('kit_contact_'+c['name']).id for c in cells])
        self.body=np.asarray(m.geom_bodyid);self.roots=np.asarray(m.body_rootid)
        self.lookup=np.full(m.ngeom,-1,int);self.lookup[self.geoms]=np.arange(40)
        self.pad=truth.pad_of_geom[self.geoms];self.frames=truth.frame_bodies
        self.cell_pos=np.asarray(m.site_pos[self.sites])
        self.coupling=np.eye(20,dtype=np.float32)
        for group in [list(range(12))]+[list(range(i,i+2)) for i in range(12,20,2)]:
            for a in group:
                for b in group:
                    if a!=b and np.linalg.norm(self.cell_pos[a+20]-self.cell_pos[b+20])<=.0181:self.coupling[a,b]=.05
        self.membership=(self.pad[:,None]==np.arange(11)).astype(np.float32)

    def initial(self,key):
        key,g=jax.random.split(key)
        return dict(key=key,gain=jax.random.uniform(g,(40,3),minval=.95,maxval=1.05),bias=j.zeros((40,3)),zero=j.zeros((40,3)),
            queue=j.zeros((2,FEATURES)),packet=j.zeros(FEATURES),previous=j.zeros(5),hp=j.zeros(5),
            age=j.zeros(40),open_ticks=j.array(0,j.int32),rezero=j.array(0,j.int32))

    def raw(self,model,data):
        c=data._impl.contact;geom=j.maximum(c.geom,0)
        wrench=j.zeros((c.pos.shape[0],6),data.qpos.dtype)
        for dim in np.unique(c.dim):
            f,idx=support.contact_force_dim(model,data,int(dim));wrench=wrench.at[idx].set(f)
        world=j.einsum('ci,cij->cj',wrench[:,:3],c.frame)
        forces=j.zeros((40,3));speed=j.zeros(40)
        active=j.all(c.geom>=0,-1)&(c.efc_address>=0)
        for side,sign in ((0,1.),(1,-1.)):
            cell=j.asarray(self.lookup)[geom[:,side]]
            mask=(cell[:,None]==j.arange(40))&active[:,None]
            body=j.asarray(self.body)[geom[:,side]];other=j.asarray(self.body)[geom[:,1-side]]
            def vel(b):
                return data.cvel[b,3:]+j.cross(data.cvel[b,:3],c.pos-data.subtree_com[j.asarray(self.roots)[b]])
            rel=vel(other)-vel(body)
            tang=rel-j.sum(rel*c.frame[:,0],-1,keepdims=True)*c.frame[:,0]
            speed+=mask.T@(j.linalg.norm(tang,axis=-1)*j.maximum(wrench[:,0],0))
            fworld=mask.astype(world.dtype).T@(world*sign)
            local=j.einsum('nji,nj->ni',data.site_xmat[self.sites],fworld)
            forces+=local[:,j.array([2,0,1])]
        return forces,speed

    def advance(self,state,raw,speed,hand_open):
        ideal=getattr(self,'level','Realistic')=='Ideal'
        s=dict(state);keys=jax.random.split(s['key'],8);s['key']=keys[0]
        s['bias']=s['bias']+jax.random.normal(keys[1],(40,3))*(.1*j.sqrt(DT/600))
        signal=raw*s['gain']+s['bias']
        noise=jax.random.normal(keys[2],(40,3))*j.concatenate((j.full((20,1),.05),j.full((20,1),.3)))
        measured=signal-s['zero']+noise
        if ideal:
            measured=raw
            s['bias']=j.zeros_like(s['bias']);s['zero']=j.zeros_like(s['zero'])
        a=jax.random.uniform(keys[3],(20,),minval=-j.pi/18,maxval=j.pi/18)
        x,y=measured[:20,1],measured[:20,2]
        optical=j.stack((j.clip(measured[:20,0],0,10),j.clip(x*j.cos(a)-y*j.sin(a),-4,4),j.clip(x*j.sin(a)+y*j.cos(a),-4,4)),-1)
        baro=j.clip(j.asarray(self.coupling)@measured[20:,0],0,20)
        normal=j.concatenate((optical[:,0],baro));engaged=normal>.3
        if ideal:
            optical=j.concatenate((j.maximum(raw[:20,:1],0),raw[:20,1:]),axis=1)
            baro=j.maximum(raw[20:,0],0)
            normal=j.concatenate((optical[:,0],baro));engaged=normal>0
        optical=j.where(engaged[:20,None],optical,0);baro=j.where(engaged[20:],baro,0)
        s['age']=j.where(engaged,s['age']+DT,0)
        # Host-declared open state uses tendon measurements; unload check uses packets.
        unloaded=j.max(j.abs(measured[:,0]))<.9
        s['open_ticks']=j.where(hand_open&unloaded,s['open_ticks']+1,0)
        rezero=s['open_ticks']==100
        s['zero']=j.where(rezero,signal,s['zero']);s['rezero']+=rezero.astype(j.int32)
        total=raw[:20,0].reshape(5,4).sum(1)
        hp=(s['hp']+total-s['previous'])*j.exp(-2*j.pi*1.6*DT)
        s['hp']=hp;s['previous']=total
        slide=speed[:20].reshape(5,4).sum(1)
        # Envelope proxy of strain high-pass + asperity source. No raw speed or
        # simulator slip label is included in the packet or contact reflex.
        jitter=jax.random.normal(keys[4],(5,3))
        if ideal:jitter=j.zeros_like(jitter)
        low=j.maximum(0,.02*hp**2+.02*slide+jitter[:,0]*.001)
        high=j.maximum(0,.05*slide+jitter[:,1]*.001)
        peak=j.maximum(0,j.abs(hp)*.1+jitter[:,2]*.01)
        detect=jax.random.uniform(keys[5],(5,))
        event=(j.abs(total-state['previous'])>.3)|(slide>.01)
        onset=j.where(event,detect<.90,detect<(.05*DT/.05)).astype(j.float32)
        if ideal:onset=event.astype(j.float32)
        pvdf=j.stack((low,high,peak,onset),-1)
        slip=((high>.003)&(normal[:20].reshape(5,4).sum(1)>.3)).astype(j.float32)
        weights=optical[:,0].reshape(5,4)
        pos=j.asarray(self.cell_pos[:20]).reshape(5,4,3)
        location=(weights[:,:,None]*pos).sum(1)/j.maximum(weights.sum(1)[:,None],1e-6)
        perturb=jax.random.normal(keys[6],(5,3))*.001
        perturb*=j.minimum(1,.002/j.maximum(j.linalg.norm(perturb,axis=-1,keepdims=True),1e-9))
        location=j.where(weights.sum(1)[:,None]>.3,location+perturb,0)
        if ideal:
            location=(weights[:,:,None]*pos).sum(1)/j.maximum(weights.sum(1)[:,None],1e-6)
        ratios=optical[:,1:]/j.maximum(optical[:,:1],.3)
        packet=j.concatenate((optical.ravel(),ratios.ravel(),baro,location.ravel()/.02,pvdf.ravel(),slip,j.minimum(s['age'],1)))
        s['packet']=state['queue'][0];s['queue']=j.concatenate((state['queue'][1:],packet[None]))
        if ideal:s['packet']=packet
        return s

    def reading(self,packet,data):
        optical=packet[:60].reshape(20,3);baro=packet[100:120]
        normal=j.concatenate((optical[:,0],baro));fn=j.asarray(self.membership).T@normal
        tangent=j.concatenate((j.linalg.norm(optical[:,1:],axis=-1),j.zeros(20)))
        ft=j.asarray(self.membership).T@tangent
        # G consumes only measured scalar load, shear and slip. Unused vector
        # fields stay zero instead of consulting simulator hand kinematics.
        force=j.zeros((11,3));centroid=j.zeros((11,3))
        slip=j.concatenate((packet[155:160]>.5,j.zeros(6,bool)))
        return touch_mjx.Reading(force,fn,ft,centroid,j.zeros((11,3)),j.zeros((11,3)),slip,j.zeros(11),fn>.3)
