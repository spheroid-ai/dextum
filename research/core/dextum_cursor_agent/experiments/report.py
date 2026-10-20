"""Aggregate E7-lite evaluations: blackout-sweep chart, table and §16 pass rules.

    python experiments/report.py --runs runs/JOB1 runs/JOB2 ... --output results/NAME
Each run directory holds <arm>/eval_final.json (and optionally vision/eval_initial.json).
Seeds are pooled per arm as mean and 95% interval over seeds (Wilson if one seed).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

LENGTHS = (0.0, 0.5, 1.0, 2.0, 3.0)


def load(runs):
    arms = {}
    for run in runs:
        run = Path(run)
        for path in sorted(run.glob("*/eval_*.json")):
            r = json.loads(path.read_text())
            name = "start (OR1)" if path.stem == "eval_initial" else r["arm"]
            arms.setdefault(name, []).append(r)
    return arms


def condition(r, mode, length):
    return r["conditions"].get("clean" if length == 0 else f"{mode}_{length:.1f}s")


def pooled(reports, key, mode, length):
    rows = [condition(r, mode, length) for r in reports]
    rows = [x for x in rows if x is not None]
    if not rows:
        return None
    values = np.array([x[key] for x in rows], float)
    if len(rows) == 1 and key in ("success_rate", "drop_rate"):
        lo, hi = rows[0]["success_ci95"] if key == "success_rate" else (math.nan, math.nan)
        return float(values[0]), lo, hi
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else math.nan
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def table(arms):
    lines = []
    for mode in ("drop", "freeze"):
        lines.append(f"\n{mode} blackouts: success % (drops %)   [n starts x seeds]")
        lines.append("arm".ljust(14) + "".join(f"{l:>14.1f}s" for l in LENGTHS))
        for arm, reports in arms.items():
            cells = []
            for l in LENGTHS:
                s = pooled(reports, "success_rate", mode, l)
                d = pooled(reports, "drop_rate", mode, l)
                cells.append("—".rjust(15) if s is None else f"{100 * s[0]:7.1f} ({100 * d[0]:4.1f})".rjust(15))
            lines.append(f"{arm} x{len(reports)}".ljust(14) + "".join(cells))
    lines.append("\npush 1 N: success %, drops %, mean settle s; clean: jerk, |da|, regrasps/episode, shear var")
    for arm, reports in arms.items():
        p = [r["conditions"].get("push_1N") for r in reports]
        p = [x for x in p if x]
        c = [r["conditions"]["clean"] for r in reports if "clean" in r["conditions"]]
        if not p or not c:
            continue
        rec = [x["push_settle_s"]["mean"] for x in p if x.get("push_settle_s", {}).get("mean") is not None]
        lines.append(f"{arm}".ljust(14) + f"{100 * np.mean([x['success_rate'] for x in p]):7.1f} "
                     f"{100 * np.mean([x['drop_rate'] for x in p]):6.1f} "
                     f"{np.mean(rec) if rec else float('nan'):6.2f} "
                     f"{np.mean([x['mean_jerk'] for x in c]):7.4f} {np.mean([x['mean_abs_da'] for x in c]):7.4f} "
                     f"{np.mean([x['regrasps_per_episode'] for x in c]):6.2f} {np.mean([x['shear_variance'] for x in c]):8.4f}")
    return "\n".join(lines)


def pass_rules(arms, touch="touch", vision="vision", mode="drop"):
    if touch not in arms or vision not in arms:
        return None
    t, v = arms[touch], arms[vision]
    st, sv = pooled(t, "success_rate", mode, 2.0), pooled(v, "success_rate", mode, 2.0)
    dt, dv = pooled(t, "drop_rate", mode, 2.0), pooled(v, "drop_rate", mode, 2.0)
    jt = np.mean([r["conditions"]["clean"]["mean_jerk"] for r in t])
    jv = np.mean([r["conditions"]["clean"]["mean_jerk"] for r in v])

    def recovery(reports):
        vals = [r["conditions"]["push_1N"]["push_settle_s"]["mean"] for r in reports
                if r["conditions"].get("push_1N", {}).get("push_settle_s", {}).get("mean") is not None]
        return float(np.mean(vals)) if vals else math.nan

    rt, rv = recovery(t), recovery(v)
    return {
        "success_2s_gain_points": round(100 * (st[0] - sv[0]), 1), "success_rule_25": bool(st[0] - sv[0] >= 0.25),
        "drop_ratio_2s": round(dt[0] / dv[0], 3) if dv[0] > 0 else None,
        "drop_rule_0.5": bool(dt[0] <= 0.5 * dv[0]) if dv[0] > 0 else bool(dt[0] == 0),
        "push_recovery_ratio": round(rt / rv, 3) if rv and np.isfinite(rv) and rv > 0 else None,
        "jerk_ratio": round(float(jt / jv), 3), "jerk_rule_0.8": bool(jt <= 0.8 * jv), "mode": mode}


def chart(arms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, ("drop", "freeze")):
        for arm, reports in arms.items():
            pts = [pooled(reports, "success_rate", mode, l) for l in LENGTHS]
            xs = [l for l, p in zip(LENGTHS, pts) if p]
            ys = np.array([p for p in pts if p])
            ax.plot(xs, 100 * ys[:, 0], marker="o", label=f"{arm} (x{len(reports)})")
            if np.isfinite(ys[:, 1]).all():
                ax.fill_between(xs, 100 * ys[:, 1], 100 * ys[:, 2], alpha=0.15)
        ax.set_title(f"camera {mode} blackout")
        ax.set_xlabel("blackout length (s), mid-rotation")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("goal-rotation success (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    arms = load(a.runs)
    a.output.mkdir(parents=True, exist_ok=True)
    text = table(arms)
    rules = {m: pass_rules(arms, mode=m) for m in ("drop", "freeze")}
    (a.output / "summary.txt").write_text(text + "\n\npass rules: " + json.dumps(rules, indent=1) + "\n")
    chart(arms, a.output / "blackout_sweep.png")
    print(text)
    print("\npass rules:", json.dumps(rules, indent=1))


if __name__ == "__main__":
    main()
