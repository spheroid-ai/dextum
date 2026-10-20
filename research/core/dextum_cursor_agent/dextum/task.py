"""Task layer: DEXTUM graph, planner, task state and the four-layer DEXTUM state (spec §2.1, §4, §6.3)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from . import contact
from .dvector import DLayout
from .nodes import NODE_NAMES


@dataclass
class Edge:
    src: str
    dst: str
    name: str
    program: str
    default_duration_s: float = 1.0
    attempts: int = 0
    successes: int = 0
    drops: int = 0
    breaks: int = 0
    total_success_duration_s: float = 0.0

    @property
    def failure_rate(self) -> float:
        return (self.attempts - self.successes) / self.attempts if self.attempts else 0.0

    @property
    def mean_duration_s(self) -> float:
        return self.total_success_duration_s / self.successes if self.successes else self.default_duration_s

    @property
    def cost(self) -> float:
        """Duration x (1 + 10 x failure rate) (spec §4.2)."""
        return self.mean_duration_s * (1.0 + 10.0 * self.failure_rate)

    def record(self, success: bool, duration_s: float, dropped: bool = False, broke: bool = False) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
            self.total_success_duration_s += duration_s
        self.drops += int(dropped)
        self.breaks += int(broke)


@dataclass
class DextumGraph:
    nodes: dict[str, str]
    edges: list[Edge]

    def out_edges(self, node: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node]

    def plan(self, start: str, goal: str) -> list[Edge]:
        """Cheapest path by Dijkstra; empty if already at the goal or unreachable."""
        best = {start: 0.0}
        back: dict[str, Edge] = {}
        frontier = [(0.0, start)]
        while frontier:
            cost, node = heapq.heappop(frontier)
            if node == goal:
                break
            if cost > best.get(node, np.inf):
                continue
            for e in self.out_edges(node):
                c = cost + e.cost
                if c < best.get(e.dst, np.inf):
                    best[e.dst], back[e.dst] = c, e
                    heapq.heappush(frontier, (c, e.dst))
        if goal not in back:
            return []
        path, node = [], goal
        while node != start:
            path.append(back[node])
            node = back[node].src
        return path[::-1]

    def distance(self, start: str, goal: str) -> int:
        """Number of edges on the planned path (the shaping potential uses minus this)."""
        if start == goal:
            return 0
        path = self.plan(start, goal)
        return len(path) if path else len(self.nodes)


def cube_rotation_graph() -> DextumGraph:
    """Version-1 graph for cube rotation on the Aero (spec §6.3)."""
    nodes = {k: v for k, v in NODE_NAMES.items() if k != "N?"}
    edges = [
        Edge("N0", "N1", "place", "external placement of the cube on the palm", 2.0),
        Edge("N1", "N2", "cage", "engage finger pads toward the cube faces at light force; keep palm force", 0.5),
        Edge("N2", "N3", "grasp", "load thumb and two fingers to firm force on opposite faces; verify stillness", 0.5),
        Edge("N3", "N4", "advance", "hold with three pads; release one finger; rotate; re-place it one step ahead", 0.5),
        Edge("N4", "N5", "advance", "repeat the advance with the next finger", 0.5),
        Edge("N5", "N4", "advance", "repeat the advance with the next finger", 0.5),
        Edge("N4", "N6", "brake_hold", "zero the angular velocity; hold all engaged pads firm; confirm yaw", 0.5),
        Edge("N5", "N6", "brake_hold", "zero the angular velocity; hold all engaged pads firm; confirm yaw", 0.5),
        Edge("N7", "N1", "recover", "detect the drop; open; wait for re-placement; re-plan", 3.0),
    ]
    for src in ("N1", "N2", "N3", "N4", "N5", "N6"):
        edges.append(Edge(src, "N7", "drop", "unintended loss of the cube", np.inf))
    return DextumGraph(nodes, edges)


@dataclass
class ObjectTrack:
    """An object in the task table, with persistence of unseen objects (spec §4.1)."""

    id: str
    label: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    pos_sigma_m: float = 0.0
    rot_sigma_deg: float = 0.0
    location: str = "unknown"
    last_seen_s: float = 0.0
    events: list = field(default_factory=list)

    def observe(self, pos, quat, t_s: float, sigma_m: float = 0.001, sigma_deg: float = 1.0) -> None:
        self.pos, self.quat = np.asarray(pos, float), np.asarray(quat, float)
        self.pos_sigma_m, self.rot_sigma_deg, self.last_seen_s = sigma_m, sigma_deg, t_s

    def predict(self, dt_s: float) -> None:
        in_hand = self.location == "in_hand"
        self.pos_sigma_m += (0.002 if in_hand else 0.010) * dt_s
        self.rot_sigma_deg += (3.0 if in_hand else 15.0) * dt_s


@dataclass
class TaskState:
    goal: dict
    graph: DextumGraph
    node: str = "N0"
    node_prob: float = 1.0
    completed: list = field(default_factory=list)
    active_edge: Edge | None = None
    attempts: int = 0
    failures: int = 0
    objects: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def set_node(self, observed: str, prob: float, goal_node: str = "N6") -> Edge | None:
        """Accept the recognised node and re-plan when it differs from the expected one."""
        expected = self.active_edge.dst if self.active_edge is not None else self.node
        if observed != self.node:
            self.log.append(("node", self.node, observed))
        self.node, self.node_prob = observed, prob
        if self.active_edge is None or observed != expected:
            path = self.graph.plan(observed, goal_node)
            self.active_edge = path[0] if path else None
        return self.active_edge


@dataclass
class MotorLayer:
    q: np.ndarray
    qdot: np.ndarray
    u_prev: np.ndarray
    command: dict = field(default_factory=dict)


@dataclass
class BeliefLayer:
    mu_hat: float = contact.MU_INIT
    m_hat_kg: float = 0.1
    slip_prob: dict = field(default_factory=dict)
    sources: tuple = ("vision", "proprioception")


@dataclass
class DextumState:
    """The one structure holding the four layers (spec §2.1)."""

    task: TaskState
    contact: np.ndarray
    motor: MotorLayer
    belief: BeliefLayer
    layout: DLayout = DLayout()
