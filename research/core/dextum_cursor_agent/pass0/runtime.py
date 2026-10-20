"""Observable-only Pass 0 task loop; no mesh, simulator or numeric dependency.

Contact states are external DEXTUM vocabulary, never Metriplector coordinates.
Missing pose/patch estimates stay unknown. Timing out is an outcome, not success.
"""
from collections import deque
from dataclasses import dataclass, field
import heapq
import math
import re

FAMILIES = ("cube", "rounded_cube", "cylinder")


@dataclass(frozen=True)
class Packet:
    """Only fields available at the declared Ideal sensor/camera interface."""
    tick: int
    normal_n: tuple[float, ...]
    shear_n: tuple[float, ...]
    slip: tuple[bool, ...]
    motor: tuple[float, ...]
    camera_yaw_deg: float | None = None
    camera_shape: str | None = None
    camera_valid: bool = False
    yaw_estimate_deg: float | None = None
    yaw_uncertainty_deg: float = 180.0

    def __post_init__(self):
        if any(len(v) != 11 for v in (self.normal_n, self.shear_n, self.slip)) or len(self.motor) != 7:
            raise ValueError("Pass 0 packet needs 11 lumped pads and 7 actuator sensors")
        values = (*self.normal_n, *self.shear_n, *self.motor, self.yaw_uncertainty_deg)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite sensor packet")
        if any(v < 0 for v in (*self.normal_n, *self.shear_n)):
            raise ValueError("Force magnitudes must be nonnegative")
        for angle in (self.camera_yaw_deg, self.yaw_estimate_deg):
            if angle is not None and not math.isfinite(angle):
                raise ValueError("Nonfinite angle")


@dataclass(frozen=True)
class Goal:
    delta_yaw_deg: float
    tolerance_deg: float = 10.0
    hold_steps: int = 40


def reorientation_dag(prompt):
    """One deliberately narrow template. Unsupported prompts fail explicitly."""
    match = re.fullmatch(r"(?:reorient|rotate|turn)(?: the object)?(?: by)?\s+([+-]?\d+(?:\.\d+)?)\s*(?:degrees|deg|°)",
                         prompt.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError("Pass 0 supports 'reorient by <signed angle> degrees'")
    angle = float(match[1])
    if not 0 < abs(angle) <= 180:
        raise ValueError("Requested yaw must be nonzero and at most 180 degrees")
    return Goal(angle)


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    quantum: str
    duration: int
    success_prior: float = .5


def stand_in_lattice(family):
    if family not in FAMILIES:
        raise ValueError("Unknown family; no implicit cube fallback")
    turn = "roll" if family == "cylinder" else "pivot"
    edges = [Edge("approach", "N0", "G", "engage", 40),
             Edge("load", "G", "P", "engage", 60),
             Edge("press_to_pinch", "R", "P", "engage", 60),
             Edge("squeeze", "P", "H", "hold", 20),
             Edge("relax", "H", "P", "regrasp", 20),
             Edge("recover", "fingers", "P", "regrasp", 40)]
    edges += [Edge(f"{node}_turn", node, "N6", turn, 120) for node in ("G", "R", "P", "H")]
    return tuple(edges)


def shortest_path(edges, source, target, failures=None):
    """Dijkstra with recorded failure costs; three failures exclude an edge."""
    failures = failures or {}
    queue = [(0.0, source, ())]
    best = {source: 0.0}
    while queue:
        cost, node, ids = heapq.heappop(queue)
        if cost > best[node]:
            continue
        if node == target:
            lookup = {e.id: e for e in edges}
            return tuple(lookup[i] for i in ids)
        for e in sorted(edges, key=lambda e: e.id):
            if e.source != node or failures.get(e.id, 0) >= 3:
                continue
            candidate = cost + e.duration * (1 + 10 * (1 - e.success_prior)) * (1 + failures.get(e.id, 0))
            if candidate < best.get(e.target, math.inf):
                best[e.target] = candidate
                heapq.heappush(queue, (candidate, e.target, (*ids, e.id)))
    return ()


@dataclass
class Recognizer:
    """Force-regime stand-in for v1.2 rules; no unobserved speed/quality tests."""
    node: str = "N0"
    candidate: str = "N0"
    consecutive: int = 0

    def update(self, packet, goal_known=False):
        n = packet.normal_n
        thumb = n[0] + n[9]
        other = any(n[i] >= .3 for i in range(1, 9))
        total = sum(n)
        if goal_known:
            raw = "N6"
        elif thumb >= 4 and total >= 8 and not any(packet.slip):
            raw = "H"
        elif thumb >= 1 and other:
            raw = "P"
        elif n[10] >= 1.3:
            raw = "R"
        elif n[10] >= .3:
            raw = "G"
        elif any(x >= .3 for x in n):
            raw = "fingers"
        else:
            raw = "N0"
        self.consecutive = self.consecutive + 1 if raw == self.candidate else 1
        self.candidate = raw
        if self.consecutive >= 2:
            self.node = raw
        return self.node


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
        yaw = packet.camera_yaw_deg if camera_allowed else packet.yaw_estimate_deg
        confidence = 1.0 if camera_allowed else packet.yaw_uncertainty_deg
        error = None
        if self.boundary_yaw is not None and yaw is not None and confidence <= 10:
            error = (self.boundary_yaw + self.goal.delta_yaw_deg - yaw + 180) % 360 - 180
        within = error is not None and abs(error) <= self.goal.tolerance_deg and any(n >= .3 for n in packet.normal_n)
        self.goal_hold = self.goal_hold + 1 if within else 0
        self.goal_observed = self.goal_hold >= self.goal.hold_steps
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


def scripted_action(quantum, tick, turn_sign=1.0, attempts=0):
    """Registered E2 grasp stand-in plus explicit lift/reland schedules.

    G and its reflex remain the executor. Mode order: free, engage, hold, release.
    No command is derived from simulator pose, contact geometry or hidden mass.
    """
    modes, forces, slides = [0]*5, [0.0]*5, [0.0]*5
    for k in (1, 2, 4):
        modes[k], forces[k] = 2, min(4.0, 3.0 + .5*attempts)
    thumb_slide = 1.0
    if quantum == "wait":
        return [0.0]*31
    if quantum == "engage":
        slides[4] = -1.0 if tick < 4 else 0.0
    elif quantum in ("pivot", "regrasp", "roll"):
        phase = tick % 20
        moving = 1 if turn_sign > 0 else 2
        if phase < 4:
            modes[moving] = 3
        elif phase < 10:
            modes[moving], slides[moving] = 0, -turn_sign
        else:
            modes[moving], forces[moving] = 2, 1.5
        slides[4] = .3*turn_sign if quantum != "regrasp" else 0.0
    action = []
    for mode, force, slide in zip(modes, forces, slides):
        action.extend([1.0 if k == mode else -1.0 for k in range(4)])
        action.extend([2*force/10-1, slide])
    return action + [thumb_slide]
