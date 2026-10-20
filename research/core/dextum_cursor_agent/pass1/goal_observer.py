"""Relative yaw/tilt/hold on the physical observation interface only.

No simulator, mesh, policy, latent interpretation or evaluation labels. The
caller supplies measured poses; blackout rejects camera updates explicitly.
"""
from dataclasses import dataclass
import math


def multiply(a,b):
    w,x,y,z=a;v,i,j,k=b
    return (w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v)


@dataclass(frozen=True)
class PoseObservation:
    tick: int
    quaternion_wxyz: tuple[float,float,float,float]
    object_z_m: float
    source: str
    valid: bool = True

    def __post_init__(self):
        if self.source not in ('camera','pose_estimator'):
            raise ValueError('Only declared observable pose sources are accepted')
        if len(self.quaternion_wxyz)!=4 or not all(math.isfinite(x) for x in (*self.quaternion_wxyz,self.object_z_m)):
            raise ValueError('Finite orientation and position required')
        norm=math.sqrt(sum(x*x for x in self.quaternion_wxyz))
        if norm<1e-8:raise ValueError('Invalid quaternion')
        object.__setattr__(self,'quaternion_wxyz',tuple(float(x/norm) for x in self.quaternion_wxyz))
        object.__setattr__(self,'object_z_m',float(self.object_z_m))


@dataclass
class GoalObserver:
    delta_yaw_deg: float
    blackout: bool = True
    hold_steps: int = 40
    tolerance_deg: float = 10.0
    tilt_deg: float = 30.0
    target: tuple | None = None
    last_tick: int | None = None
    hold: int = 0
    ever_observed: bool = False

    def issue_at_contact(self, pose: PoseObservation | None):
        if self.target is not None:raise ValueError('Goal boundary already issued')
        if pose is None or not pose.valid:
            return False
        angle=math.radians(self.delta_yaw_deg)/2
        self.target=multiply((math.cos(angle),0.,0.,math.sin(angle)),pose.quaternion_wxyz)
        self.last_tick=pose.tick-1
        return True

    def update(self, tick: int, pose: PoseObservation | None):
        if self.last_tick is not None and tick<=self.last_tick:
            raise ValueError('Strictly increasing control ticks required')
        if self.last_tick is not None and tick!=self.last_tick+1:self.hold=0
        self.last_tick=tick
        allowed=(self.target is not None and pose is not None and pose.valid and
                 pose.tick==tick and not (self.blackout and pose.source=='camera'))
        if not allowed:
            self.hold=0
            return dict(known=False,within=False,hold_steps=0,goal_now=False,
                        ever_observed=self.ever_observed,yaw_error_deg=None,tilt_ok=None)
        w,x,y,z=pose.quaternion_wxyz
        error=multiply(self.target,(w,-x,-y,-z))
        angle=2*math.atan2(error[3],error[0])
        yaw=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle))))
        tilt_ok=error[0]**2+error[3]**2>=math.cos(math.radians(self.tilt_deg)/2)**2
        within=yaw<self.tolerance_deg and tilt_ok and pose.object_z_m>=-.05
        self.hold=self.hold+1 if within else 0
        reached=self.hold>=self.hold_steps
        self.ever_observed |= reached
        return dict(known=True,within=within,hold_steps=self.hold,goal_now=reached,
                    ever_observed=self.ever_observed,yaw_error_deg=yaw,tilt_ok=tilt_ok)
