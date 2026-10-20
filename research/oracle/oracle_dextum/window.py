"""Causal per-quantum summaries for section9.8c's typed token vocabulary.

Node signatures are explicit interface labels. They are not DEXONs or learned
latent labels. Metastability is measured by the lattice builder, never inferred
from a signature's name.
"""
from bisect import bisect_right
from dataclasses import dataclass
import math
from .schema import OracleFrame, QUANTA
from .tokenizer import ENGAGE_N, FORCE_EDGES_N, SLIP_MIN_FORCE_N, SLIP_SPEED_M_S


PARAMETERS = {
    'angle_rad': (-math.pi, math.pi, 128),
    'distance_m': (-.1, .1, 128),
    'force_n': (0., 20., 128),
    'duration_s': (0., 20., 128),
}
ENERGY_EDGES_J = (1e-5, 1e-4, 1e-3, .01, .1, 1., 10.)
COUNT_EDGES = (0, 1, 2, 4, 8, 16, 32)
MODES = ('free', 'sticking', 'sliding', 'mixed')


def node_signature(frame):
    """Signature retains patch/element identities and coarse load classes."""
    contacts = tuple((c.key, bisect_right(FORCE_EDGES_N, c.normal_n))
                     for c in frame.canonical().contacts if c.normal_n > ENGAGE_N)
    slipping = [c for c in frame.contacts if c.normal_n > ENGAGE_N and
                c.slip_speed_m_s > SLIP_SPEED_M_S]
    mode = 'free' if not contacts else 'sticking' if not slipping else \
        'sliding' if len(slipping) == len(contacts) else 'mixed'
    return dict(contacts=contacts, mode=mode, pose_class='unknown')


def quantize_parameter(name, value):
    low, high, bins = PARAMETERS[name]
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Parameter {name} outside its registered range')
    return min(bins-1, int((value-low)/(high-low)*bins))


@dataclass(frozen=True)
class Summary:
    quantum: str
    start_signature: dict
    final_signature: dict
    duration_s: float
    mean_total_normal_n: float
    contact_onsets: int
    slip_onsets: int
    parameters: tuple[tuple[str, float], ...]
    energy_j: float | None

    def atoms(self):
        # Typed atoms, rather than a Cartesian-product vocabulary. Identities
        # and signatures remain payload fields, not hashes masquerading as IDs.
        result = [('quantum', QUANTA.index(self.quantum)),
                  ('mode', MODES.index(self.final_signature['mode'])),
                  ('mean_normal', bisect_right(FORCE_EDGES_N, self.mean_total_normal_n)),
                  ('contact_onsets', bisect_right(COUNT_EDGES, self.contact_onsets)),
                  ('slip_onsets', bisect_right(COUNT_EDGES, self.slip_onsets)),
                  ('energy', 0 if self.energy_j is None else 1+bisect_right(ENERGY_EDGES_J, self.energy_j))]
        result += [('parameter/'+name, quantize_parameter(name, value)) for name, value in self.parameters]
        return tuple(result)


class QuantumWindow:
    def __init__(self, initial: OracleFrame, parameters=()):
        self.initial = self.previous = initial.canonical()
        self.parameters = tuple(sorted(parameters))
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError('Duplicate quantum parameter')
        for name, value in self.parameters: quantize_parameter(name, value)
        self.integral_normal_n_s = 0.
        self.contact_onsets = self.slip_onsets = 0

    @staticmethod
    def sets(frame):
        engaged = {c.key for c in frame.contacts if c.normal_n > ENGAGE_N}
        slipping = {c.key for c in frame.contacts if c.normal_n > SLIP_MIN_FORCE_N and
                    c.slip_speed_m_s > SLIP_SPEED_M_S}
        return engaged, slipping

    def advance(self, frame: OracleFrame):
        frame = frame.canonical()
        if frame.task_id != self.initial.task_id or frame.quantum != self.initial.quantum:
            raise ValueError('Close the window at a task or quantum boundary')
        dt = frame.time_s-self.previous.time_s
        if dt <= 0: raise ValueError('Non-increasing window time')
        old_e, old_s = self.sets(self.previous); new_e, new_s = self.sets(frame)
        self.contact_onsets += len(new_e-old_e); self.slip_onsets += len(new_s-old_s)
        self.integral_normal_n_s += sum(c.normal_n for c in self.previous.contacts)*dt
        self.previous = frame

    def finish(self, *, energy_j=None):
        duration = self.previous.time_s-self.initial.time_s
        if duration <= 0: raise ValueError('A quantum window needs measured duration')
        if energy_j is not None and (not math.isfinite(energy_j) or energy_j < 0):
            raise ValueError('Energy class requires a declared nonnegative work measure')
        return Summary(self.initial.quantum, node_signature(self.initial),
            node_signature(self.previous), duration, self.integral_normal_n_s/duration,
            self.contact_onsets, self.slip_onsets, self.parameters, energy_j)
