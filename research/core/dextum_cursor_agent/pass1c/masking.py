"""Isolated §15 sensor interventions; frozen physical environment is unchanged.

Masks cover the 200-value packet, derived legacy centroid, and G's measured
inputs. Privileged reward/critic contact truth is unchanged. Cell force
patterns remain observable in the explicit-location mask.
"""
import os
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
import jax
import jax.numpy as j
import numpy as np
import vendor.aero_official.vision_env  # registered MuJoCo Playground path
from pass1b.sensors import DT,FEATURES
from pass1c.sensors import Sensors as BaseSensors
from pass1c.environment import BlindRotation as BaseEnvironment,PROPRIO,GOAL_INPUTS
from dextum import contact_space as cs,touch_mjx

GROUPS=('none','shear','location','slip','pvdf','palm','proximal','normal_only')

def mask_packet(packet,group):
    if group not in GROUPS:raise ValueError(group)
    if group in ('shear','normal_only'):
        optical=packet[:60].reshape(20,3).at[:,1:].set(0)
        packet=packet.at[:60].set(optical.ravel()).at[60:100].set(0)
    if group in ('location','normal_only'):packet=packet.at[120:135].set(0)
    if group in ('slip','normal_only'):packet=packet.at[155:160].set(0)
    if group in ('pvdf','normal_only'):packet=packet.at[135:155].set(0)
    if group=='palm':packet=packet.at[100:112].set(0).at[180:192].set(0)
    if group=='proximal':packet=packet.at[112:120].set(0).at[192:200].set(0)
    return packet

class Sensors(BaseSensors):
    def configure(self,config):
        self.group=config.get('mask','none');assert self.group in GROUPS
        self.floor=float(config.get('detection_floor_n',.3))
        self.ideal_floor=float(config.get('ideal_detection_floor_n',0))
        self.floors=np.array([float(config.get('optical_detection_floor_n',self.floor))]*20+
            [float(config.get('barometric_detection_floor_n',self.floor))]*20,dtype=np.float32)
        self.ideal_floors=np.array([float(config.get('ideal_optical_detection_floor_n',self.ideal_floor))]*20+
            [float(config.get('ideal_barometric_detection_floor_n',self.ideal_floor))]*20,dtype=np.float32)
        self.force_noise=float(config.get('optical_noise_n',.05))
        self.direction_rad=float(config.get('direction_degrees',10))*np.pi/180
        self.false_alarm=float(config.get('pvdf_false_alarm',.05))
        self.location_noise_m=float(config.get('location_noise_mm',1))/1000
        rate=float(config.get('rate_hz',200));self.sample_period=round(200/rate)
        assert rate>0 and self.sample_period>=1 and abs(rate*self.sample_period-200)<1e-9
        delay_ms=float(config.get('delay_ms',10))
        self.delay_ticks=round(delay_ms/5)
        assert 0<=self.delay_ticks<=10 and abs(self.delay_ticks*5-delay_ms)<1e-9
        self.configured=any(k!='mask' for k in config)
    def initial(self,key):
        state=super().initial(key)
        if self.configured:state['queue']=j.zeros((max(1,self.delay_ticks),FEATURES))
        if self.sample_period>1:state['sample_tick']=j.zeros((),j.int32)
        return state
    def advance(self,state,raw,speed,hand_open):
        result=(self.configured_advance(state,raw,speed,hand_open) if self.configured
                else super().advance(state,raw,speed,hand_open))
        if self.sample_period>1:
            result['packet']=j.where(state['sample_tick']%self.sample_period==0,
                                     result['packet'],state['packet'])
            result['sample_tick']=state['sample_tick']+1
        return {**result,'packet':mask_packet(result['packet'],self.group)}

    def configured_advance(self,state,raw,speed,hand_open):
        ideal=getattr(self,'level','Realistic')=='Ideal'
        s=dict(state);keys=jax.random.split(s['key'],8);s['key']=keys[0]
        s['bias']=s['bias']+jax.random.normal(keys[1],(40,3))*(.1*j.sqrt(DT/600))
        signal=raw*s['gain']+s['bias']
        noise=jax.random.normal(keys[2],(40,3))*j.concatenate((j.full((20,1),self.force_noise),j.full((20,1),.3)))
        measured=signal-s['zero']+noise
        if ideal:
            measured=raw
            s['bias']=j.zeros_like(s['bias']);s['zero']=j.zeros_like(s['zero'])
        a=jax.random.uniform(keys[3],(20,),minval=-self.direction_rad,maxval=self.direction_rad)
        x,y=measured[:20,1],measured[:20,2]
        optical=j.stack((j.clip(measured[:20,0],0,10),j.clip(x*j.cos(a)-y*j.sin(a),-4,4),j.clip(x*j.sin(a)+y*j.cos(a),-4,4)),-1)
        baro=j.clip(j.asarray(self.coupling)@measured[20:,0],0,20)
        normal=j.concatenate((optical[:,0],baro));engaged=normal>j.asarray(self.floors)
        if ideal:
            optical=j.concatenate((j.maximum(raw[:20,:1],0),raw[:20,1:]),axis=1)
            baro=j.maximum(raw[20:,0],0)
            normal=j.concatenate((optical[:,0],baro));engaged=normal>j.asarray(self.ideal_floors)
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
        onset=j.where(event,detect<.90,detect<(self.false_alarm*DT/.05)).astype(j.float32)
        if ideal:onset=event.astype(j.float32)
        pvdf=j.stack((low,high,peak,onset),-1)
        slip=((high>.003)&(normal[:20].reshape(5,4).sum(1)>.3)).astype(j.float32)
        weights=optical[:,0].reshape(5,4)
        pos=j.asarray(self.cell_pos[:20]).reshape(5,4,3)
        location=(weights[:,:,None]*pos).sum(1)/j.maximum(weights.sum(1)[:,None],1e-6)
        perturb=jax.random.normal(keys[6],(5,3))*self.location_noise_m
        perturb*=j.minimum(1,2*self.location_noise_m/j.maximum(j.linalg.norm(perturb,axis=-1,keepdims=True),1e-9))
        location=j.where(weights.sum(1)[:,None]>.3,location+perturb,0)
        if ideal:
            location=(weights[:,:,None]*pos).sum(1)/j.maximum(weights.sum(1)[:,None],1e-6)
        ratios=optical[:,1:]/j.maximum(optical[:,:1],.3)
        packet=j.concatenate((optical.ravel(),ratios.ravel(),baro,location.ravel()/.02,pvdf.ravel(),slip,j.minimum(s['age'],1)))
        s['packet']=state['queue'][0] if self.delay_ticks else packet
        if self.delay_ticks:s['queue']=j.concatenate((state['queue'][1:],packet[None]))
        if ideal:s['packet']=packet
        return s


class BlindRotation(BaseEnvironment):
    def __init__(self,arm='c_touch',intervention=None):
        super().__init__(arm)
        # The physical builder and all base step/reset code remain frozen.
        self.sensors.__class__=Sensors
        self.sensors.configure(intervention or {})
    def _frame(self,official_state,info,reading):
        frame=super()._frame(official_state,info,reading)
        if self.touch and self.sensors.group in ('location','normal_only'):
            start=PROPRIO+GOAL_INPUTS+cs.STATE_FEATURES
            prefix=frame[start:start+touch_mjx.ACTOR_SIZE].reshape(11,10)
            prefix=prefix.at[:,4:7].set(0)
            frame=frame.at[start:start+touch_mjx.ACTOR_SIZE].set(prefix.ravel())
        return frame
