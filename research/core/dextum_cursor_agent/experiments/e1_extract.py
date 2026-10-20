"""E1: DEXTUM extraction from a working rotator's traces (spec §4.4, §6.2, §10).

Input: the clean-condition per-pad trace written by `evaluate.py --dump`.
Version 1 labels every step with the rule recogniser. Version 2 clusters
metastable regions of D-vector histories (k-means micro-states, lag-5
transition matrix, spectral clustering, node count at the largest spectral gap
in 6..40) and is scored against the rule labels on rule-labelled steps.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np

from dextum import contact
from dextum.dvector import DLayout, Pad, PadReading, assemble
from dextum.nodes import MODES, NodeRules, Recognizer, signature
from dextum.touch_mjx import PAD_NAMES
from vendor.aero_official.goal_yaw import WARMUP

DIGIT = {"thumb_tip": "thumb", "thumb_base": "thumb", "palm": None}
PADS = tuple(Pad(n, DIGIT.get(n, n.split("_")[0]), "palm" if n == "palm" else
                 ("tip" if n.endswith("tip") else "finger"), 1) for n in PAD_NAMES)
LAYOUT = DLayout(pads=PADS)


def quat_rotate(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r @ v


def episode_ends(t):
    done = t["done"] > 0.5
    T = done.shape[0]
    return np.where(done.any(0), done.argmax(0), T - 1)


def calibrate_rest(t, ends):
    """Median cube position over still, palm-supported steps: the D-vector origin."""
    palm = PAD_NAMES.index("palm")
    pts = []
    for i, end in enumerate(ends):
        v = t["cube_qvel"][:end + 1, i]
        still = (np.linalg.norm(v[:, :3], axis=-1) < 0.01) & (np.linalg.norm(v[:, 3:], axis=-1) < 0.25)
        on_palm = t["pad_fn"][:end + 1, i, palm] > contact.ENGAGE_N
        pts.append(t["cube_qpos"][:end + 1, i, :3][still & on_palm])
    pts = np.concatenate(pts)
    return np.median(pts, axis=0), len(pts)


def dvectors(t, i, end, origin):
    out = []
    for k in range(end + 1):
        q = t["cube_qpos"][k, i, 3:7]
        lin = t["cube_qvel"][k, i, :3]
        ang = quat_rotate(q, t["cube_qvel"][k, i, 3:6])        # free-joint angular velocity is local
        readings = {}
        for p, name in enumerate(PAD_NAMES):
            readings[name] = PadReading(
                f_n=float(t["pad_fn"][k, i, p]), f_t=(float(t["pad_ft"][k, i, p]), 0.0),
                loc=tuple(float(x) for x in t["pad_centroid"][k, i, p, :2]),
                slip=bool(t["pad_slip"][k, i, p]),
                point_obj=t["pad_point"][k, i, p], normal_obj=t["pad_normal"][k, i, p])
        out.append(assemble(LAYOUT, t["cube_qpos"][k, i, :3] - origin, q, lin, ang, readings))
    return np.array(out)


def goal_yaw(goal_quat):
    w, x, y, z = goal_quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def version1(t, ends, origin, rules, episodes, rule_version="1"):
    labels, sigs, dv, episode = [], [], [], []
    for i in episodes:
        end = ends[i]
        d = dvectors(t, i, end, origin)
        rec = Recognizer(LAYOUT, rules, version=rule_version)
        for k in range(end + 1):
            gy = goal_yaw(t["goal"][k, i]) if t["age"][k, i] >= WARMUP else None
            labels.append(rec.update(d[k], gy))
            sigs.append(signature(LAYOUT, d[k], rules))
        dv.append(d)
        episode += [i] * (end + 1)
    return np.array(labels), sigs, np.concatenate(dv), np.array(episode)


def features(dv, episode, history=3):
    """Contact part of the D-vector (flags, forces, slip, grip) with a short history."""
    cols = np.concatenate([LAYOUT.pad_column(f) for f in ("flag", "f_n", "f_t1", "slip")])
    grip = np.arange(LAYOUT.grip.start, LAYOUT.grip.stop)
    x = np.concatenate((dv[:, cols], dv[:, grip], dv[:, 10:13]), axis=1)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-2)
    frames = [x]
    for lag in range(1, history):
        prev = np.roll(x, lag, axis=0)
        same = np.roll(episode, lag) == episode
        prev[~same] = x[~same]
        frames.append(prev)
    return np.concatenate(frames, axis=1)


def version2(x, episode, centres=1000, lag=5, k_range=(6, 40), seed=0):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    centres = min(centres, len(x) // 20)
    micro = MiniBatchKMeans(n_clusters=centres, random_state=seed, n_init=3, batch_size=4096).fit_predict(x)
    counts = np.zeros((centres, centres))
    src = np.arange(len(x) - lag)
    ok = episode[src] == episode[src + lag]
    np.add.at(counts, (micro[src[ok]], micro[src[ok] + lag]), 1.0)
    counts = counts + counts.T + 1e-6
    deg = counts.sum(1)
    sym = counts / np.sqrt(np.outer(deg, deg))
    vals, vecs = np.linalg.eigh(sym)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    lo, hi = k_range[0], min(k_range[1], centres - 2)
    gaps = vals[lo - 1:hi] - vals[lo:hi + 1]
    k = int(lo + np.argmax(gaps))
    emb = vecs[:, :k] / np.maximum(np.linalg.norm(vecs[:, :k], axis=1, keepdims=True), 1e-12)
    macro_of_micro = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)
    return macro_of_micro[micro], k, vals[:hi + 2]


def agreement(learned, rule):
    keep = rule != "N?"
    mapping = {}
    for c in np.unique(learned[keep]):
        mapping[c] = Counter(rule[keep][learned[keep] == c]).most_common(1)[0][0]
    predicted = np.array([mapping.get(c, "N?") for c in learned])
    return float((predicted[keep] == rule[keep]).mean()), {int(c): v for c, v in mapping.items()}


def describe_clusters(learned, labels, modes, dv, episode):
    """Physical summary of each learned node: what the dynamics' metastable regions are."""
    f_n = dv[:, LAYOUT.pad_column("f_n")]
    slip = dv[:, LAYOUT.pad_column("slip")]
    palm = f_n[:, PAD_NAMES.index("palm")]
    thumb = f_n[:, [PAD_NAMES.index("thumb_tip"), PAD_NAMES.index("thumb_base")]].sum(1)
    run_len = dwell(learned, episode)
    out = []
    for c in np.unique(learned):
        m = learned == c
        top = Counter(labels[m]).most_common(2)
        out.append(dict(
            cluster=int(c), fraction=round(float(m.mean()), 4), dwell_steps=run_len[int(c)],
            top_nodes=[(k, round(v / m.sum(), 3)) for k, v in top],
            top_mode=Counter(modes[m]).most_common(1)[0][0],
            yaw_rate=round(float(np.abs(dv[m, 12]).mean()), 3),
            speed=round(float(np.linalg.norm(dv[m, 7:10], axis=1).mean()), 4),
            total_fn=round(float(f_n[m].sum(1).mean()), 3), palm_fn=round(float(palm[m].mean()), 3),
            thumb_fn=round(float(thumb[m].mean()), 3),
            engaged=round(float(dv[m][:, LAYOUT.pad_column("flag")].sum(1).mean()), 2),
            slipping=round(float(slip[m].sum(1).mean()), 2)))
    return sorted(out, key=lambda r: -r["fraction"])


def dwell(labels, episode):
    """Mean run length in steps of each node, within episodes."""
    runs = {}
    start = 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start] or episode[k] != episode[start]:
            runs.setdefault(labels[start].item() if hasattr(labels[start], "item") else labels[start], []).append(k - start)
            start = k
    return {node: round(float(np.mean(r)), 2) for node, r in sorted(runs.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--centres", type=int, default=1000)
    p.add_argument("--rules", choices=("1", "1.1", "1.2"), default="1")
    p.add_argument("--episodes", default=":", help="python slice of episodes, e.g. 64: for held-out")
    a = p.parse_args()
    with np.load(a.trace) as f:
        t = {k: f[k] for k in f.files}
    ends = episode_ends(t)
    chosen = np.arange(len(ends))[slice(*[int(x) if x else None for x in a.episodes.split(":")])]
    origin, n_rest = calibrate_rest(t, ends)
    rules = NodeRules(rest_height_m=0.0)
    labels, sigs, dv, episode = version1(t, ends, origin, rules, chosen, a.rules)
    modes = np.array([MODES.get(x, "N?") for x in labels])
    occupancy = Counter(labels)
    transitions = Counter((a_, b_) for a_, b_, e1, e2 in zip(labels[:-1], labels[1:], episode[:-1], episode[1:])
                          if e1 == e2 and a_ != b_)
    sig_counts = Counter((tuple(n for n, _ in s[0]), s[1]) for s in sigs)
    learned, k, spectrum = version2(features(dv, episode), episode, a.centres)
    score, mapping = agreement(learned, labels)
    mode_score, mode_mapping = agreement(learned, modes)
    n = len(labels)
    report = dict(
        trace=str(a.trace), rules_version=a.rules, episodes=a.episodes, n_episodes=int(len(chosen)), steps=int(n),
        rest_origin=origin.tolist(), rest_samples=int(n_rest),
        rules={k_: v for k_, v in rules.__dict__.items()},
        occupancy={k_: round(v / n, 4) for k_, v in occupancy.most_common()},
        mode_occupancy={k_: round(v / n, 4) for k_, v in Counter(modes).most_common()},
        dwell_steps=dwell(labels, episode), mode_dwell_steps=dwell(modes, episode),
        transitions=[dict(src=s, dst=d, count=c) for (s, d), c in transitions.most_common(25)],
        signatures=[dict(pads=list(s), pose=pose, fraction=round(c / n, 4)) for (s, pose), c in sig_counts.most_common(25)],
        distinct_signatures=len(sig_counts),
        version2=dict(nodes=k, agreement_on_rule_labelled=round(score, 4), passes_90=score >= 0.9,
                      mode_agreement=round(mode_score, 4), mode_passes_90=mode_score >= 0.9,
                      mapping=mapping, mode_mapping=mode_mapping, spectrum=[round(float(v), 4) for v in spectrum],
                      clusters=describe_clusters(learned, labels, modes, dv, episode)),
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k_: report[k_] for k_ in ("n_episodes", "steps", "occupancy", "mode_occupancy",
                                                 "dwell_steps", "mode_dwell_steps", "distinct_signatures")}, indent=1))
    print(json.dumps({k_: v for k_, v in report["version2"].items() if k_ not in ("mapping", "mode_mapping", "spectrum")}))


if __name__ == "__main__":
    main()
