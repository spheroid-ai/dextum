"""Serializable oracle records. Units are SI; identities belong to interfaces."""
from dataclasses import asdict, dataclass
import math


ELEMENT_CLASSES = ('tip', 'shaft', 'palm', 'jaw', 'environment')
QUANTA = ('unknown', 'cage', 'engage', 'hold', 'roll', 'slide', 'pivot',
          'regrasp', 'release', 'place', 'push', 'insert', 'squeeze',
          'micro_lift', 'micro_slide')


@dataclass(frozen=True, order=True)
class ContactKey:
    object_id: str
    patch_id: int
    element_class: str
    element_id: int

    def __post_init__(self):
        if not self.object_id or self.patch_id < 0 or self.element_id < 0:
            raise ValueError('Object, patch and within-class element identity required')
        if self.element_class not in ELEMENT_CLASSES:
            raise ValueError('Unknown canonical element class')


@dataclass(frozen=True)
class ContactRecord:
    key: ContactKey
    normal_n: float
    # Force and point in the object frame, not hand-specific joint coordinates.
    force_object_n: tuple[float, float, float]
    point_object_m: tuple[float, float, float]
    slip_speed_m_s: float
    friction: float
    contact_count: int
    patch_assignment_distance_m: float = 0.
    patch_boundary_ambiguous: bool = False

    def __post_init__(self):
        values = (self.normal_n, *self.force_object_n, *self.point_object_m,
                  self.slip_speed_m_s, self.friction,
                  self.patch_assignment_distance_m)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite oracle contact')
        if any(x < 0 for x in (self.normal_n, self.slip_speed_m_s,
                               self.friction, self.patch_assignment_distance_m)):
            raise ValueError('Negative magnitude in oracle contact')
        if self.contact_count < 1:
            raise ValueError('A contact record must contain a contact')


@dataclass(frozen=True)
class OracleFrame:
    time_s: float
    contacts: tuple[ContactRecord, ...]
    # Only explicitly named issued goals/history belong here. No success labels.
    task_id: str
    quantum: str = 'unknown'

    def __post_init__(self):
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('Invalid timestamp')
        if self.quantum not in QUANTA:
            raise ValueError('Unknown quantum; never infer one from future success')
        keys = [c.key for c in self.contacts]
        if len(set(keys)) != len(keys):
            raise ValueError('Aggregate duplicate element/patch contacts first')

    def canonical(self):
        return OracleFrame(self.time_s, tuple(sorted(self.contacts, key=lambda c: c.key)),
                           self.task_id, self.quantum)

    def to_dict(self):
        return asdict(self.canonical())
