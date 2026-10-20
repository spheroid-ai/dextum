"""Pass 1 task loop with aligned physical-interface goal recognition.

Planner, force recognizer and scripts retain Pass 0 behavior. The historical
Pass 0 module stays byte-identical. Pose must be a declared sensor estimate.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from pass0.runtime import Packet as PreviousPacket
from pass0.runtime import Goal, Recognizer, Edge, FAMILIES, shortest_path, stand_in_lattice
from pass0.runtime import reorientation_dag, scripted_action
from .goal_observer import GoalObserver, PoseObservation


@dataclass(frozen=True)
class Packet(PreviousPacket):
    pose: PoseObservation | None = None


@dataclass
class TaskLoop:
    goal: Goal
    blackout: bool = True
    family: str | None = None
    boundary_yaw: float | None = None
    contact_seen: bool = False
    recognizer: Recognizer = field(default_factory=Recognizer)
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    trace: deque = field(default_factory=lambda: deque(maxlen=32))
    failures: dict = field(default_factory=dict)
    active: Edge | None = None
    active_since: int = 0
    replans: int = 0
    goal_hold: int = 0
    last_tick: int = -1
    goal_observed: bool = False
    progress: float = 0.0
    initial_distance: float | None = None

    observer: GoalObserver = field(init=False)

    def __post_init__(self):
        self.observer = GoalObserver(self.goal.delta_yaw_deg, blackout=self.blackout,
            hold_steps=self.goal.hold_steps, tolerance_deg=self.goal.tolerance_deg)

    def update(self, packet):
        if packet.tick <= self.last_tick:
            raise ValueError("Packets must have strictly increasing timestamps")
        self.last_tick = packet.tick
        first_contact = any(n >= .3 for n in packet.normal_n) and not self.contact_seen
        # Last camera sample at acquisition is allowed. All later ones are ignored
        # after blackout, even if an upstream source mistakenly supplies them.
        camera_allowed = packet.camera_valid and (not self.blackout or not self.contact_seen)
        if camera_allowed:
            if packet.camera_shape in FAMILIES:
                self.family = packet.camera_shape
            if not self.contact_seen and packet.camera_yaw_deg is not None:
                self.boundary_yaw = packet.camera_yaw_deg
        self.contact_seen |= first_contact
        if first_contact:
            boundary = packet.pose
            if boundary is not None and boundary.source == 'camera' and not camera_allowed:
                boundary = None
            self.observer.issue_at_contact(boundary)
        observed = self.observer.update(packet.tick, packet.pose)
        error = observed['yaw_error_deg']
        self.goal_hold = observed['hold_steps']
        self.goal_observed = observed['ever_observed']
        node = self.recognizer.update(packet, self.goal_observed)
        self.history.append((node, packet.normal_n, packet.slip))
        if self.active and (node == self.active.target or packet.tick - self.active_since >= self.active.duration):
            ok = node == self.active.target
            self.trace.append(dict(edge=self.active.id, quantum=self.active.quantum, source=self.active.source,
                                   observed=node, outcome="observed" if ok else "timeout",
                                   duration=packet.tick - self.active_since))
            if not ok:
                self.failures[self.active.id] = self.failures.get(self.active.id, 0) + 1
            self.active = None
        if self.family is None:
            return {"quantum": "wait", "reason": "no initial shape observation", "node": node}
        path = shortest_path(stand_in_lattice(self.family), node, "N6", self.failures)
        distance = float(len(path)) if path else (0.0 if node == "N6" else math.inf)
        if self.initial_distance is None and math.isfinite(distance) and distance > 0:
            self.initial_distance = distance
        self.progress = (max(0.0, 1 - distance / self.initial_distance)
                         if self.initial_distance and math.isfinite(distance) else 0.0)
        if node == "N6":
            self.progress = 1.0
            self.active = None
        elif self.active is None and path:
            self.active, self.active_since = path[0], packet.tick
            self.replans += 1
        return dict(node=node, quantum=self.active.quantum if self.active else "hold",
                    edge=self.active.id if self.active else None,
                    edge_tick=packet.tick-self.active_since, progress=self.progress,
                    attempts=self.failures.get(self.active.id, 0) if self.active else 0,
                    replans=self.replans, yaw_error_deg=error,
                    pose_observable=error is not None, goal_observed=self.goal_observed,
                    no_route=not path and node != "N6")
