"""Deployment-side deterministic E3 actor and G, with no simulator imports."""
import numpy as np

HOME = np.array([.09, .09, .09, .09, .75, .035, .1])
SCALE = np.array([.02, .02, .02, .02, .7, .003, .012])
PRESS = np.array([0, 1, 2, 3, 6])
SENSOR_ORDER = np.array([0, 1, 2, 3, 5, 6, 4])
PAD_GROUPS = ((1, 5), (2, 6), (3, 7), (4, 8), (0, 9))


class Actor:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.layers=[(data[f'w{i}'],data[f'b{i}']) for i in range(int(data['layers']))]
        self.width=int(self.layers[0][0].shape[0]);self.actions=int(self.layers[-1][1].shape[0])
        if (self.width,self.actions) not in ((759,31),(159,31),(54,7)):
            raise ValueError('Unknown registered Kit B actor interface')

    def __call__(self,observation):
        x=np.clip(np.asarray(observation,np.float32),-5,5)
        if x.shape[-1]!=self.width or not np.isfinite(x).all():raise ValueError('Invalid observable actor vector')
        for i,(w,b) in enumerate(self.layers):
            x=x@w+b
            if i+1<len(self.layers):x=x/(1+np.exp(-np.clip(x,-80,80)))
        return np.tanh(x)


class ContactMotor:
    """NumPy port of the registered E2/E3 map G; parity checked remotely."""
    def __init__(self, ctrl_range, reflex=True):
        self.reflex = bool(reflex)
        self.lo = np.maximum(np.asarray(ctrl_range)[:, 0], HOME-SCALE)
        self.hi = np.minimum(np.asarray(ctrl_range)[:, 1], HOME+SCALE)
        self.cmd = HOME.copy()
        self.mode = np.zeros(5, int)
        self.f_ref = np.zeros(5)
        self.boost0 = np.zeros(5)
        self.since_slip = np.full(5, 10000)
        self.mu = np.full(5, .4)
        self.slip = np.zeros(5, bool)

    @staticmethod
    def sense(normal, shear, slip):
        force = np.array([sum(normal[i] for i in group) for group in PAD_GROUPS])
        tangent = np.array([sum(shear[i] for i in group) for group in PAD_GROUPS])
        slips = np.array([any(slip[i] for i in group) for group in PAD_GROUPS])
        return force, tangent, slips, force > .3

    def features(self, normal, shear, slip, contact_override=None):
        force, _, _, contact = self.sense(normal, shear, slip)
        if contact_override is not None: contact=np.asarray(contact_override,bool)
        return np.concatenate((np.eye(4)[self.mode], (self.f_ref/10)[:, None],
                               np.clip(force/5, 0, 5)[:, None], contact[:, None]), axis=1).reshape(-1)

    def step(self, action, normal, shear, slips, contact_override=None):
        action = np.asarray(action)
        per = action[:30].reshape(5, 6)
        mode = per[:, :4].argmax(1)
        target = 5*(np.clip(per[:, 4], -1, 1)+1)
        slide = np.clip(per[:, 5], -1, 1)
        force, shear, slip, contact = self.sense(normal, shear, slips)
        if contact_override is not None: contact=np.asarray(contact_override,bool)
        pressing = (mode == 1) | (mode == 2)
        onset = slip & ~self.slip & contact & pressing & self.reflex
        ratio = shear/np.maximum(force, 1e-6)
        self.mu = np.maximum(np.where(onset, np.minimum(self.mu, .7*self.mu+.3*ratio), self.mu), .05)
        self.since_slip = np.where(onset, 0, np.minimum(self.since_slip+1, 10000))
        self.boost0 = np.where(onset, np.maximum(.5, .3*self.f_ref), self.boost0)
        boost = self.boost0*np.clip(1-(self.since_slip-6)/20, 0, 1)
        base = np.where(mode == 2, target, .5)
        f_ref = np.where(pressing & contact & self.reflex, np.maximum(base, shear/(self.mu*.7))+boost, base)
        self.boost0 = np.where(pressing, self.boost0, 0)
        regulate = -np.clip(2e-4*(f_ref-force), -.001, .001)
        delta = np.select([mode == 0, pressing, mode == 3],
                          [-.05*slide*SCALE[PRESS], np.where(contact, regulate, -.02*SCALE[PRESS]), .02*SCALE[PRESS]])
        self.cmd[PRESS] += delta
        self.cmd[4] += .05*slide[4]*SCALE[4]
        self.cmd[5] -= .05*np.clip(action[-1], -1, 1)*SCALE[5]
        self.cmd = np.clip(self.cmd, self.lo, self.hi)
        self.mode, self.f_ref, self.slip = mode, np.where(pressing, f_ref, 0), slip
        return self.cmd.copy()


CELL_PAD=np.array([0]*4+[1]*4+[2]*4+[3]*4+[4]*4+[10]*12+[5]*2+[6]*2+[7]*2+[8]*2)

def pad_signals(packet):
    packet=np.asarray(packet);optical=packet[:60].reshape(20,3)
    normal=np.bincount(CELL_PAD,weights=np.r_[optical[:,0],packet[100:120]],minlength=11)
    shear=np.bincount(CELL_PAD,weights=np.r_[np.linalg.norm(optical[:,1:],axis=-1),np.zeros(20)],minlength=11)
    slip=np.r_[packet[155:160]>.5,np.zeros(6,bool)]
    return normal,shear,slip


def actor_frame(motor_values,command,motor,normal,shear,slip,goal_face_rad,sign,tick,arm,touch_window=None,contact_override=None):
    sensors=(np.asarray(motor_values)-HOME[SENSOR_ORDER])/SCALE[SENSOR_ORDER]
    u=(np.asarray(command)-HOME)/SCALE
    goal=np.array([np.cos(4*goal_face_rad),np.sin(4*goal_face_rad),sign]) if tick>=20 else np.zeros(3)
    parts=[sensors,u,[(tick-20)/280],goal]
    if arm in ('c_touch','c_notouch'):parts.append(motor.features(normal,shear,slip,contact_override))
    if arm=='c_touch':
        if np.shape(touch_window)!=(200,):raise ValueError('Expected measured 200-field window')
        parts.append(touch_window)
    return np.clip(np.concatenate(parts),-5,5)
