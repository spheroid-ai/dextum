"""Bounded measured lattice builder. The connector owns actual simulation.

No connector succeeds by declaring its requested target node. It must return
the independently observed endpoint and physical success criterion.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from .lattice import Edge


@dataclass(frozen=True)
class Trial:
    success: bool
    observed_target: str
    duration_s: float
    dropped: bool
    entry_valid: bool
    entry_id: str


def wilson(successes, trials):
    if trials <= 0: raise ValueError('No measurement')
    z = 1.959963984540054
    p = successes/trials; scale = 1+z*z/trials
    centre = (p+z*z/(2*trials))/scale
    half = z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/scale
    return [centre-half, centre+half]


def build_cell(*, connector, object_id, hand_id, source, target, quantum,
               parameters, parameter_id, seeds, controller_sha256, entry_bank_sha256):
    """All draws recorded, including invalid entries; no rejection resampling."""
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise ValueError('Exactly sixteen independent registered draws required')
    rows = []
    for seed in seeds:
        result = connector(object_id=object_id, hand_id=hand_id, source=source,
                           quantum=quantum, parameters=parameters, seed=seed)
        if not isinstance(result, Trial): raise TypeError('Connector must return measured Trial')
        if not math.isfinite(result.duration_s) or result.duration_s <= 0:
            raise ValueError('Invalid measured duration')
        accepted = result.entry_valid and result.success and not result.dropped and \
            result.observed_target == target
        rows.append(dict(seed=seed, accepted=accepted, **asdict(result)))
    successes = sum(r['accepted'] for r in rows)
    durations = [r['duration_s'] for r in rows if r['accepted']]
    duration = sum(durations)/len(durations) if durations else sum(r['duration_s'] for r in rows)/16
    edge = Edge(source, target, quantum, 16, successes, duration,
                controller_sha256, entry_bank_sha256, parameter_id)
    return dict(edge=asdict(edge), object_id=object_id, hand_id=hand_id,
        parameters=parameters, trials=rows, success_interval_95=wilson(successes, 16),
        feasibility='feasible' if successes/16 >= .7 else 'marginal' if successes/16 >= .3 else 'infeasible',
        limits=['Empirical 16-draw estimate, not an exact lattice.',
                'Parameter-grid selection needs fresh held-out draws for O3 feasibility accuracy.',
                'Endpoint success does not establish composition without entry-distribution checks.'])
