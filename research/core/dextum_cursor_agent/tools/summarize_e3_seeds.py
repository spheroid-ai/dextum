"""E3 across seeds: training-time evaluator readings and the registered decision rules.

Reads runs/<job>/<arm>/metrics.jsonl (one JSON row per logged step; evaluations carry
eval/episode_e3/success). Readings per arm and seed: at 100M = mean of the evaluations with
75M <= steps <= 110M; at 200M = mean of those with 180M <= steps <= 220M (REGISTER, E3 seeds 2 and 3;
the last evaluation of a 100M run is at 108.1M, of a 200M run at 216.3M).

    python tools/summarize_e3_seeds.py --runs runs/e3p1-s1 runs/e3p1-s2c runs/e3p1-s3c
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEY = "eval/episode_e3/success"
WINDOWS = {"100M": (75e6, 110e6), "200M": (180e6, 220e6)}
ARMS = ("c_touch", "c_notouch", "a_task", "a_touch", "a_notouch")


def curve(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [(r["steps"], r["metrics"][KEY]) for r in rows if KEY in r.get("metrics", {})]


def reading(points, lo, hi):
    vals = [v for s, v in points if lo <= s <= hi]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    a = p.parse_args()

    table = {}  # (arm, window) -> {run: reading}
    for run in a.runs:
        for arm in ARMS:
            f = run / arm / "metrics.jsonl"
            if not f.exists():
                continue
            pts = curve(f)
            for w, (lo, hi) in WINDOWS.items():
                val, n = reading(pts, lo, hi)
                if n:
                    table.setdefault((arm, w), {})[run.name] = (val, n)

    print(f"{'arm':<10} {'window':<6} " + " ".join(f"{r.name:>12}" for r in a.runs) + f" {'mean':>8}")
    for (arm, w), per_run in sorted(table.items()):
        cells = [per_run.get(r.name) for r in a.runs]
        vals = [c[0] for c in cells if c]
        print(f"{arm:<10} {w:<6} " + " ".join(f"{c[0]:>9.3f}({c[1]:>2})" if c else f"{'-':>12}" for c in cells)
              + f" {np.mean(vals):>8.3f}")

    def get(arm, w, run):
        return table.get((arm, w), {}).get(run, (None, 0))[0]

    runs = [r.name for r in a.runs]
    print("\nDecision rules (REGISTER, E3 seeds 2 and 3):")
    # (1) ordering at 100M in every seed, three-seed mean differences >= 0.10
    for other in ("c_notouch", "a_task"):
        diffs = [(get("c_touch", "100M", r), get(other, "100M", r)) for r in runs]
        diffs = [(c, o) for c, o in diffs if c is not None and o is not None]
        holds = all(c > o for c, o in diffs)
        mean_diff = float(np.mean([c - o for c, o in diffs])) if diffs else float("nan")
        print(f"  (1) 100M: C-touch > {other} in every seed: {holds} ({len(diffs)} seeds); "
              f"mean difference {mean_diff:+.3f} (rule: >= 0.10) -> {'confirmed' if holds and mean_diff >= 0.10 else 'not confirmed'}")
    # (2) longer budget
    gains = [(get("c_touch", "200M", r), get("c_touch", "100M", r)) for r in runs]
    gains = [(b, a_) for b, a_ in gains if b is not None and a_ is not None]
    if gains:
        g = float(np.mean([b - a_ for b, a_ in gains]))
        print(f"  (2) C-touch 200M - 100M, mean over {len(gains)} seeds: {g:+.3f} (rule: >= 0.05) -> "
              f"{'longer training helps' if g >= 0.05 else 'no clear gain'}")
    # (3) C-touch vs a_task at 200M in both seeds
    pairs = [(get("c_touch", "200M", r), get("a_task", "200M", r)) for r in runs]
    pairs = [(c, o) for c, o in pairs if c is not None and o is not None]
    if pairs:
        ok = all(c >= o + 0.10 for c, o in pairs)
        print(f"  (3) 200M: C-touch >= a_task + 0.10 in every seed: {ok} "
              f"({', '.join(f'{c - o:+.3f}' for c, o in pairs)})")


if __name__ == "__main__":
    main()
