"""Read-only, stdlib re-scoring of historical demo records; no runtime input."""
import argparse
import hashlib
import json
from pathlib import Path


def confusion(trials, prediction):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    for t in trials:
        key = ('t' if prediction(t) == t['success'] else 'f') + ('p' if prediction(t) else 'n')
        counts[key] += 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    trials, hashes = [], {}
    for f in sorted(a.records.glob('p0_*.json')):
        trials.append(json.loads(f.read_text()))
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    assert len(trials) == len({(t['asset'],t['seed'],t['blackout']) for t in trials}) == 96
    terminal = lambda t:bool(t['goal_observed'])
    ever = lambda t:any(s['goal_observed'] for s in t['steps'])
    # This is evaluator-side oracle information, never passed to runtime.
    oracle = lambda t:t['max_hold_s'] >= 2.0
    rows = []
    for asset, blackout in sorted({(t['asset'],t['blackout']) for t in trials}):
        group = [t for t in trials if (t['asset'],t['blackout']) == (asset,blackout)]
        rows.append(dict(asset=asset, blackout=blackout, n=len(group),
                         true_success=sum(t['success'] for t in group),
                         terminal_observed=sum(terminal(t) for t in group),
                         ever_observed=sum(ever(t) for t in group),
                         terminal_confusion=confusion(group,terminal),
                         ever_confusion=confusion(group,ever),
                         oracle_hold_confusion=confusion(group,oracle)))
    result = dict(state='PARTIAL_REPLAY', n=96, controller_changes=0,
                  original_records_unchanged=True, rows=rows,
                  terminal_confusion=confusion(trials,terminal),
                  ever_confusion=confusion(trials,ever),
                  oracle_hold_confusion=confusion(trials,oracle),
                  missing_for_exact_observable_replay=['per-step tilt or full orientation',
                      'timestamped observed camera/estimated pose with uncertainty',
                      'observable retained/drop measurement'],
                  conclusion='Terminal flags undercount recognition events. Ever-event aggregation corrects reporting only. '
                      'max_hold_s reproduces evaluator labels by construction; it is an oracle bound, not independent recognition. '
                      'Exact yaw/tilt/hold observer replay cannot be recovered from these records alone.',
                  source_sha256=hashes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','rows')},indent=2))


if __name__ == '__main__':main()
