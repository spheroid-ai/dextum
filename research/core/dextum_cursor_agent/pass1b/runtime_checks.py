"""Pure packet/frame/motor parity on cloud CPU, without constructing a simulator."""
import os,sys,json
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vendor.aero_official.vision_env
import numpy as np
import jax
import jax.numpy as j
from dextum import contact_space as cs
from pass1b.environment import BlindRotation
from pass1b.numpy_actor import ContactMotor,pad_signals,actor_frame,CELL_PAD,HOME,SCALE,SENSOR_ORDER
from pass1b.sensors import Sensors
from pass1b.masking import mask_packet
p=Path(sys.argv[1]);p.mkdir(exist_ok=False)
rng=np.random.default_rng(11);sensors=Sensors.__new__(Sensors);sensors.membership=(CELL_PAD[:,None]==np.arange(11)).astype(np.float32)
packet=rng.uniform(0,2,200).astype(np.float32);pn,pt,ps=pad_signals(packet);reading=sensors.reading(j.asarray(packet),None)
packet_error=max(float(np.max(np.abs(pn-np.asarray(reading.f_n)))),float(np.max(np.abs(pt-np.asarray(reading.f_t)))))
assert np.array_equal(ps,np.asarray(reading.slip));frame_error=0.;motor_error=0.
for arm in ('c_touch','c_notouch','a_task'):
 env=BlindRotation.__new__(BlindRotation);env.space='tendon' if arm=='a_task' else 'contact';env.touch=arm=='c_touch'
 motor=ContactMotor(np.stack((HOME-SCALE,HOME+SCALE),1),reflex=env.touch);g=cs.initial_state();lo,hi=map(j.asarray,cs.bounds(np.stack((HOME-SCALE,HOME+SCALE),1)))
 advance=jax.jit(lambda state,act,sense:cs.step(state,act,sense,lo,hi,reflex=env.touch))
 for _ in range(64):
  normal=rng.uniform(0,3,11);shear=rng.uniform(0,2,11);slip=rng.uniform(size=11)<.15
  force,tangent,slips,contact=motor.sense(normal,shear,slip);contact=rng.uniform(size=5)<.7
  sense=cs.Sense(*map(j.asarray,(force,tangent,slips,contact)))
  action=rng.uniform(-1,1,31);g,expected=advance(g,j.asarray(action),sense)
  actual=motor.step(action,normal,shear,slip,contact_override=contact);motor_error=max(motor_error,float(np.max(np.abs(np.asarray(expected)-actual))))
  values=HOME[SENSOR_ORDER]+rng.normal(0,.005,7);tick=int(rng.integers(0,300));angle=float(rng.uniform(-np.pi,np.pi));sign=1.
  window=rng.normal(size=200).astype(np.float32)
  info=dict(tick=j.array(tick),goal_face=j.array(angle),turn_sign=j.array(sign),u=j.asarray((actual-HOME)/SCALE),g=g,sense=sense,sensor_window=j.asarray(window))
  observed=np.asarray(env._frame(j.asarray(np.r_[values,np.zeros(7)]),info,object()))
  deployed=actor_frame(values,actual,motor,normal,shear,slip,angle,sign,tick,arm,window,contact_override=contact)
  frame_error=max(frame_error,float(np.max(np.abs(observed-deployed))))
for group in ('no_shear','no_pvdf_slip','no_barometric'):
 masked=np.asarray(mask_packet(j.asarray(packet),group));n,t,s=pad_signals(masked)
 if group=='no_shear':assert not t.any() and not masked[60:100].any()
 if group=='no_pvdf_slip':assert not s.any() and not masked[135:155].any()
 if group=='no_barometric':assert not n[5:].any() and not masked[180:200].any()
result=dict(state='PASS' if max(packet_error,frame_error,motor_error)<1e-5 else 'FAIL',packet_max_error=packet_error,frame_max_error=frame_error,motor_max_error=motor_error,mask_groups_checked=3,simulator_constructed=False,actor_checkpoint='Not yet available; weight parity is a separate queued export check')
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert result['state']=='PASS'
