"""Pad vocabulary and the D-vector (spec §1, §3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from . import contact

DIGITS = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class Pad:
    name: str
    digit: str | None
    kind: str
    sensors: int


S1_PADS = (
    Pad("thumb_tip", "thumb", "tip", 4),
    Pad("index_tip", "index", "tip", 4),
    Pad("middle_tip", "middle", "tip", 4),
    Pad("ring_tip", "ring", "tip", 4),
    Pad("little_tip", "little", "tip", 4),
    Pad("index_pad", "index", "finger", 2),
    Pad("middle_pad", "middle", "finger", 2),
    Pad("ring_pad", "ring", "finger", 2),
    Pad("little_pad", "little", "finger", 2),
    Pad("palm", None, "palm", 12),
)

PAD_FIELDS = ("flag", "f_n", "f_t1", "f_t2", "loc_u", "loc_v", "slip")
GRIP_FIELDS = ("total_f_n", "n_engaged", "min_margin", "quality")
VIS_FIELDS = ("visible", "age_s", "confidence")


@dataclass
class PadReading:
    f_n: float = 0.0
    f_t: tuple[float, float] = (0.0, 0.0)
    loc: tuple[float, float] = (0.0, 0.0)
    slip: bool = False
    point_obj: np.ndarray | None = None
    normal_obj: np.ndarray | None = None

    @property
    def engaged(self) -> bool:
        return self.f_n >= contact.ENGAGE_N


@dataclass(frozen=True)
class DLayout:
    pads: tuple[Pad, ...] = S1_PADS
    pad_index: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "pad_index", {p.name: i for i, p in enumerate(self.pads)})

    @property
    def n_pad(self) -> int:
        return len(self.pads)

    pose = slice(0, 7)
    motion = slice(7, 13)

    def pad(self, name_or_index) -> slice:
        i = self.pad_index[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
        start = 13 + 7 * i
        return slice(start, start + 7)

    @property
    def grip(self) -> slice:
        start = 13 + 7 * self.n_pad
        return slice(start, start + 4)

    @property
    def vis(self) -> slice:
        start = 17 + 7 * self.n_pad
        return slice(start, start + 3)

    @property
    def size(self) -> int:
        return 20 + 7 * self.n_pad

    def pad_column(self, field_name: str) -> np.ndarray:
        """Indices of one per-pad field across all pads."""
        k = PAD_FIELDS.index(field_name)
        return np.array([13 + 7 * i + k for i in range(self.n_pad)])


def grip_summary(readings: Mapping[str, PadReading], mu_hat: float) -> np.ndarray:
    engaged = [r for r in readings.values() if r.engaged]
    if not engaged:
        return np.zeros(4)
    f_n = np.array([r.f_n for r in engaged])
    f_t = np.array([r.f_t for r in engaged])
    margin = float(np.min(contact.cone_margin(f_n, f_t, mu_hat)))
    located = [r for r in engaged if r.point_obj is not None and r.normal_obj is not None]
    quality = contact.grasp_quality([r.point_obj for r in located], [r.normal_obj for r in located]) if located else 0.0
    return np.array([f_n.sum(), float(len(engaged)), margin, quality])


def assemble(
    layout: DLayout,
    obj_pos,
    obj_quat,
    obj_linvel,
    obj_angvel,
    readings: Mapping[str, PadReading],
    mu_hat: float = contact.MU_INIT,
    visibility=(1.0, 0.0, 1.0),
) -> np.ndarray:
    """Build the D-vector; poses and velocities are in the palm frame, quaternion w-first."""
    d = np.zeros(layout.size)
    d[layout.pose] = np.concatenate([np.asarray(obj_pos, float), np.asarray(obj_quat, float)])
    d[layout.motion] = np.concatenate([np.asarray(obj_linvel, float), np.asarray(obj_angvel, float)])
    for pad in layout.pads:
        r = readings.get(pad.name, PadReading())
        d[layout.pad(pad.name)] = [float(r.engaged), r.f_n, r.f_t[0], r.f_t[1], r.loc[0], r.loc[1], float(r.slip)]
    d[layout.grip] = grip_summary(readings, mu_hat)
    d[layout.vis] = np.asarray(visibility, float)
    return d


def unpack(layout: DLayout, d: np.ndarray) -> dict:
    d = np.asarray(d)
    out = {
        "pos": d[0:3],
        "quat": d[3:7],
        "linvel": d[7:10],
        "angvel": d[10:13],
        "grip": dict(zip(GRIP_FIELDS, d[layout.grip])),
        "vis": dict(zip(VIS_FIELDS, d[layout.vis])),
        "pads": {},
    }
    for pad in layout.pads:
        out["pads"][pad.name] = dict(zip(PAD_FIELDS, d[layout.pad(pad.name)]))
    return out
