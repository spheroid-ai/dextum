"""Measured lattice records and fail-closed Dijkstra planning.

Sixteen draws estimate feasibility; they never establish an exact transition.
The builder supplies measurements through a connector, not invented edge labels.
"""
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    quantum: str
    trials: int
    successes: int
    mean_duration_s: float
    controller_sha256: str
    entry_bank_sha256: str
    parameter_id: str

    def __post_init__(self):
        if not 0 <= self.successes <= self.trials:
            raise ValueError('Invalid measured counts')
        if not math.isfinite(self.mean_duration_s) or self.mean_duration_s <= 0:
            raise ValueError('Duration must be finite and positive')
        if not self.controller_sha256 or not self.entry_bank_sha256:
            raise ValueError('Controller and entry provenance required')

    @property
    def cost(self):
        if not self.trials:
            return math.inf
        return self.mean_duration_s*(1+10*(1-self.successes/self.trials))


@dataclass(frozen=True)
class Plan:
    status: str
    edges: tuple[Edge, ...]
    cost: float | None


def plan(nodes, edges, start, goal, *, minimum_trials=16, minimum_success_rate=.7,
         allow_marginal=True):
    if start not in nodes or goal not in nodes:
        return Plan('UNKNOWN_NODE', (), None)
    if start == goal:
        return Plan('AT_GOAL', (), 0.)
    outgoing = {n: [] for n in nodes}
    for edge in edges:
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError('Edge references an unknown node')
        if edge.trials >= minimum_trials and edge.successes/edge.trials >= minimum_success_rate:
            outgoing[edge.source].append(edge)
    best, back, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        if node == goal:
            result = []
            while node != start:
                edge = back[node]; result.append(edge); node = edge.source
            return Plan('PATH', tuple(reversed(result)), cost)
        for edge in sorted(outgoing[node], key=lambda e: (e.target, e.quantum, e.parameter_id)):
            candidate = cost+edge.cost
            if candidate < best.get(edge.target, math.inf):
                best[edge.target], back[edge.target] = candidate, edge
                heapq.heappush(queue, (candidate, edge.target))
    if allow_marginal and minimum_success_rate > .3:
        fallback = plan(nodes, edges, start, goal, minimum_trials=minimum_trials,
                        minimum_success_rate=.3, allow_marginal=False)
        if fallback.status == 'PATH':
            return Plan('MARGINAL_PATH', fallback.edges, fallback.cost)
    return Plan('UNREACHABLE', (), None)
