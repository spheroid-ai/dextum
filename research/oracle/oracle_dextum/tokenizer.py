"""Causal typed tokens plus continuous oracle records, not a learned encoder.

Typed atoms keep the vocabulary bounded without hashing distinct contact states
onto the same token. Record positions/identities remain side fields. A macro
summary is lossy by design; O1 must name which representation it tests.
"""
from dataclasses import dataclass
from bisect import bisect_right
from .schema import ContactKey, OracleFrame, ELEMENT_CLASSES, QUANTA


FORCE_EDGES_N = (.02, .05, .1, .3, .5, 1., 2., 5., 10.)
SLIP_MIN_FORCE_N = .05
SLIP_SPEED_M_S = .01
ENGAGE_N = .3


@dataclass(frozen=True)
class Token:
    kind: str
    value: int
    contact: ContactKey | None = None


class Tokenizer:
    """Reset on every episode. No hand names, joint ids or future labels."""
    vocabulary = tuple(
        [('quantum', i) for i in range(len(QUANTA))]
        + [('element', i) for i in range(len(ELEMENT_CLASSES))]
        + [('force', i) for i in range(len(FORCE_EDGES_N)+1)]
        + [('slip', i) for i in range(2)]
        + [('event', i) for i in range(4)]  # continuing, onset, release, unloaded
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = frozenset()
        self.last_time = None
        self.task_id = None

    def encode(self, frame: OracleFrame):
        frame = frame.canonical()
        if self.last_time is not None and frame.time_s <= self.last_time:
            raise ValueError('Non-increasing time; reset the tokenizer at episode entry')
        if self.task_id is not None and frame.task_id != self.task_id:
            raise ValueError('Task changed without an episode reset')
        keys = frozenset(c.key for c in frame.contacts if c.normal_n > ENGAGE_N)
        tokens = [Token('quantum', QUANTA.index(frame.quantum))]
        for c in frame.contacts:
            tokens.extend((Token('element', ELEMENT_CLASSES.index(c.key.element_class), c.key),
                           Token('force', bisect_right(FORCE_EDGES_N, c.normal_n), c.key),
                           Token('slip', int(c.normal_n > SLIP_MIN_FORCE_N and
                                             c.slip_speed_m_s > SLIP_SPEED_M_S), c.key),
                           Token('event', 3 if c.key not in keys else
                                 int(c.key not in self.previous), c.key)))
        for key in sorted(self.previous-keys):
            tokens.append(Token('event', 2, key))
        self.previous, self.last_time, self.task_id = keys, frame.time_s, frame.task_id
        return tuple(tokens)


def flatten_losslessly(frame: OracleFrame):
    """Same information for the flat control, before its declared encoder.

    Explicit keys and all fields survive. Removing IDs is a separate information
    ablation; a fixed permutation alone is not a proof of structural advantage.
    """
    return frame.canonical().to_dict()
