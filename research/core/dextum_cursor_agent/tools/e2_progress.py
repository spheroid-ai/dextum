"""Evaluation curves of an E2 run folder: python tools/e2_progress.py runs/e2p1-s1b"""
import json
import sys
from pathlib import Path

ARMS = ("c_touch", "c_notouch", "a_touch", "a_notouch")
KEYS = (("success", "success"), ("n3", "N3 steps"), ("held", "held"), ("drop", "drop"),
        ("grip_slips", "grip slips"), ("slip_onsets", "all slips"), ("lift_mm", "lift mm·steps"))


def rows(path):
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        m = r["metrics"]
        if "eval/episode_reward" in m:
            out.append((r["steps"], r["elapsed_s"], m))
    return out


def main(run):
    run = Path(run)
    for arm in ARMS:
        path = run / arm / "metrics.jsonl"
        if not path.exists():
            continue
        evals = rows(path)
        print(f"== {arm}: {len(evals)} evaluations")
        print(f"{'steps':>10} {'min':>5} {'return':>7} " + " ".join(f"{label:>13}" for _, label in KEYS))
        for steps, elapsed, m in evals:
            values = " ".join(f"{m.get('eval/episode_e2/' + k, float('nan')):13.2f}" for k, _ in KEYS)
            print(f"{steps:>10} {elapsed / 60:5.1f} {m['eval/episode_reward']:7.2f} {values}")
        reached = [s for s, _, m in evals if m.get("eval/episode_e2/success", 0) >= 0.8]
        print(f"   first evaluation with success >= 0.8: {reached[0] if reached else 'not reached'}")


if __name__ == "__main__":
    main(sys.argv[1])
