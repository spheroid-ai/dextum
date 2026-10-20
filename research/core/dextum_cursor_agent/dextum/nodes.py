"""Signatures and rule-based node recognition, version 1 (spec §3.2, §4.4, §6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contact
from .dvector import DIGITS, DLayout

NODE_NAMES = {
    "N0": "empty",
    "N1": "resting",
    "N2": "caged",
    "N3": "grasped",
    "N4": "gait phase A",
    "N5": "gait phase B",
    "N6": "at goal",
    "N7": "dropped",
    "N?": "unclassified",
}


@dataclass(frozen=True)
class NodeRules:
    """Thresholds of the version-1 recogniser; every value is recorded in the deviation register."""

    still_linvel_mps: float = 0.01
    still_angvel_rps: float = 0.25
    rotating_angvel_rps: float = 0.3
    lifted_m: float = 0.003
    rest_height_m: float = contact.CUBE_HALF_EDGE_M
    palm_region_radius_m: float = 0.07
    below_palm_m: float = -0.01
    goal_tolerance_deg: float = 10.0
    grasp_quality_min: float = 0.3
    persist_steps: int = 2


def yaw_from_quat(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def pad_classes(layout: DLayout, d: np.ndarray) -> dict[str, int]:
    f_n = d[layout.pad_column("f_n")]
    return {p.name: int(c) for p, c in zip(layout.pads, contact.force_class(f_n))}


def digit_classes(layout: DLayout, classes: dict[str, int]) -> dict[str, int]:
    """Strongest force class over the pads of each digit."""
    out = {dg: contact.NONE for dg in DIGITS}
    for p in layout.pads:
        if p.digit is not None:
            out[p.digit] = max(out[p.digit], classes[p.name])
    return out


def pose_class(layout: DLayout, d: np.ndarray, rules: NodeRules) -> str:
    pos, angvel, linvel = d[0:3], d[10:13], d[7:10]
    if not in_palm_region(pos, rules):
        return "lost"
    if abs(angvel[2]) >= rules.rotating_angvel_rps:
        return "rotating"
    still = np.linalg.norm(linvel) < rules.still_linvel_mps and np.linalg.norm(angvel) < rules.still_angvel_rps
    if pos[2] - rules.rest_height_m > rules.lifted_m:
        return "held" if still else "lifted"
    return "resting" if still else "moving"


def in_palm_region(pos, rules: NodeRules) -> bool:
    return bool(np.linalg.norm(pos[:2]) <= rules.palm_region_radius_m and pos[2] >= rules.below_palm_m)


def signature(layout: DLayout, d: np.ndarray, rules: NodeRules = NodeRules()) -> tuple:
    """Engaged pads with their force class, plus the pose class: the discretised contact state."""
    classes = pad_classes(layout, d)
    engaged = tuple(sorted((name, c) for name, c in classes.items() if c > contact.NONE))
    return engaged, pose_class(layout, d, rules)


def raw_node(layout: DLayout, d: np.ndarray, rules: NodeRules = NodeRules(), goal_yaw: float | None = None) -> str:
    classes = pad_classes(layout, d)
    digits = digit_classes(layout, classes)
    pose = pose_class(layout, d, rules)
    quality = float(d[layout.grip][3])
    palm = classes.get("palm", contact.NONE) > contact.NONE
    engaged_fingers = [dg for dg, c in digits.items() if c > contact.NONE]
    firm_digits = [dg for dg, c in digits.items() if c >= contact.FIRM]
    firm_pads = sum(c >= contact.FIRM for c in classes.values())
    any_engaged = any(c > contact.NONE for c in classes.values())

    if not any_engaged:
        # Empty and dropped look alike in one step; the recogniser separates them by history.
        return "N0/7" if pose == "lost" else "N?"
    still = pose in ("resting", "held")
    if goal_yaw is not None and still and firm_pads >= 3:
        if abs(np.degrees(wrap_angle(yaw_from_quat(d[3:7]) - goal_yaw))) <= rules.goal_tolerance_deg:
            return "N6"
    if pose == "rotating" and "thumb" in firm_digits:
        fingers_firm = [dg for dg in firm_digits if dg != "thumb"]
        free = [dg for dg in DIGITS[1:] if digits[dg] == contact.NONE]
        if len(fingers_firm) >= 2 and free:
            return "N4/5"
    if firm_pads >= 3 and "thumb" in firm_digits and quality >= rules.grasp_quality_min:
        return "N3"
    if palm and len(engaged_fingers) >= 2 and still:
        return "N2"
    if palm and not engaged_fingers:
        return "N1"
    return "N?"


V11_NAMES = {
    "N8": "palm-borne, moving",
    "N9": "airborne",
    "N10": "thumb-finger pinch",
    "N11": "palm sweep",
    "N12": "finger push",
}
NODE_NAMES_V11 = {**NODE_NAMES, "N1": "resting on palm", "N2": "palm and fingers, still", **V11_NAMES}
MODES = {"N0": "empty", "N7": "dropped", "N9": "free", "N6": "goal",
         "N1": "palm", "N2": "palm", "N8": "palm", "N11": "palm",
         "N3": "opposed", "N4": "opposed", "N5": "opposed", "N10": "opposed", "N12": "finger", "N?": "N?"}


def raw_node_v11(layout: DLayout, d: np.ndarray, rules: NodeRules = NodeRules(), goal_yaw: float | None = None) -> str:
    """Version 1.1: the spec nodes plus the contact modes the working rotator visits (E1).

    Precedence: empty/dropped, airborne, at goal (palm support counts as holding), the spec's
    grasp and gait nodes, then the support pattern: thumb opposing a finger, palm only,
    palm with fingers, fingers only.
    """
    classes = pad_classes(layout, d)
    digits = digit_classes(layout, classes)
    pose = pose_class(layout, d, rules)
    if not any(c > contact.NONE for c in classes.values()):
        return "N0/7" if pose == "lost" else "N9"
    if pose == "lost":
        return "N?"
    palm = classes.get("palm", contact.NONE) > contact.NONE
    thumb = digits["thumb"] > contact.NONE
    fingers = [dg for dg in DIGITS[1:] if digits[dg] > contact.NONE]
    still = pose in ("resting", "held")
    firm_pads = sum(c >= contact.FIRM for c in classes.values())
    if goal_yaw is not None and still and (palm or firm_pads >= 3):
        if abs(np.degrees(wrap_angle(yaw_from_quat(d[3:7]) - goal_yaw))) <= rules.goal_tolerance_deg:
            return "N6"
    spec = raw_node(layout, d, rules, None)
    if spec in ("N3", "N4/5"):
        return spec
    if thumb and fingers:
        return "N10"
    if palm and not fingers and not thumb:
        return "N1" if still else "N8"
    if palm and fingers:
        return "N2" if still else "N11"
    if fingers and not palm:
        return "N12"
    return "N?"


V12_NAMES = {"G": "glide (palm-borne, light)", "R": "press (palm-loaded)", "P": "pinch (thumb opposed)",
             "H": "squeeze-hold", "S": "strike"}
MODES.update({"G": "palm", "R": "palm", "P": "opposed", "H": "opposed", "S": "strike"})


@dataclass(frozen=True)
class RegimeRules:
    """Version 1.2 force-regime thresholds, read off the design-set learned clusters (E1)."""

    strike_total_n: float = 10.0
    strike_speed_mps: float = 0.06
    strike_yaw_rps: float = 0.6
    hold_thumb_n: float = 4.0
    hold_total_n: float = 8.0
    hold_yaw_rps: float = 0.15
    hold_speed_mps: float = 0.01
    pinch_thumb_n: float = contact.LIGHT_MAX_N
    press_palm_n: float = 1.3


def raw_node_v12(layout: DLayout, d: np.ndarray, rules: NodeRules = NodeRules(), goal_yaw: float | None = None,
                 regime: RegimeRules = RegimeRules()) -> str:
    """Version 1.2: force regime and support instead of finger identity (E1 learned structure)."""
    classes = pad_classes(layout, d)
    pose = pose_class(layout, d, rules)
    f_n = d[layout.pad_column("f_n")]
    if not any(c > contact.NONE for c in classes.values()):
        return "N0/7" if pose == "lost" else "N9"
    if pose == "lost":
        return "N?"
    names = [p.name for p in layout.pads]
    palm = float(f_n[names.index("palm")]) if "palm" in names else 0.0
    thumb = float(sum(f_n[i] for i, p in enumerate(layout.pads) if p.digit == "thumb"))
    total = float(f_n.sum())
    others = sum(c > contact.NONE for n, c in classes.items() if n != "palm" and not n.startswith("thumb"))
    yaw_rate, speed = abs(float(d[12])), float(np.linalg.norm(d[7:10]))
    still = pose in ("resting", "held")
    firm_pads = sum(c >= contact.FIRM for c in classes.values())
    if goal_yaw is not None and still and (palm >= contact.ENGAGE_N or firm_pads >= 3):
        if abs(np.degrees(wrap_angle(yaw_from_quat(d[3:7]) - goal_yaw))) <= rules.goal_tolerance_deg:
            return "N6"
    if total >= regime.strike_total_n and (speed >= regime.strike_speed_mps or yaw_rate >= regime.strike_yaw_rps):
        return "S"
    if thumb >= regime.hold_thumb_n and total >= regime.hold_total_n and yaw_rate < regime.hold_yaw_rps \
            and speed < regime.hold_speed_mps:
        return "H"
    if thumb >= regime.pinch_thumb_n and others >= 1:
        return "P"
    if palm >= contact.ENGAGE_N:
        return "R" if palm >= regime.press_palm_n else "G"
    return "N12"


@dataclass
class Recognizer:
    """Version-1 recogniser with hysteresis; splits the gait into phases A and B.

    A change of node is accepted after ``persist_steps`` consecutive raw labels, which is
    the metastability requirement of a DEXTUM node. The gait phase flips each time the set
    of free fingers changes while rotating.
    """

    layout: DLayout
    rules: NodeRules = NodeRules()
    version: str = "1"
    node: str = "N0"
    _candidate: str | None = None
    _count: int = 0
    _free: tuple = ()
    _phase: str = "N4"
    _was_in_hand: bool = False
    history: list = field(default_factory=list)

    def update(self, d: np.ndarray, goal_yaw: float | None = None) -> str:
        raw = {"1.1": raw_node_v11, "1.2": raw_node_v12}.get(self.version, raw_node)
        label = raw(self.layout, d, self.rules, goal_yaw)
        if label == "N0/7":
            label = "N7" if self._was_in_hand else "N0"
        elif label != "N?":
            self._was_in_hand = True
        if label == "N4/5":
            digits = digit_classes(self.layout, pad_classes(self.layout, d))
            free = tuple(dg for dg in DIGITS[1:] if digits[dg] == contact.NONE)
            if self._free and free != self._free:
                self._phase = "N5" if self._phase == "N4" else "N4"
            self._free = free
            label = self._phase
        if label == self.node:
            self._candidate, self._count = None, 0
        elif label == self._candidate:
            self._count += 1
            if self._count >= self.rules.persist_steps:
                self.node, self._candidate, self._count = label, None, 0
        else:
            self._candidate, self._count = label, 1
            if self.rules.persist_steps <= 1:
                self.node, self._candidate, self._count = label, None, 0
        self.history.append(self.node)
        return self.node
