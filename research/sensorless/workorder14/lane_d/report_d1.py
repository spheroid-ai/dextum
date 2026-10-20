"""Final-checkpoint paired D1 report; run on the qualified GPU host."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('CONTAINER_ID'):
        raise RuntimeError('GPU cloud only')
    os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    import numpy as np
    from pass0.review_gate import exact_mcnemar
    assert jax.default_backend() == 'gpu' and all(d.platform == 'gpu' for d in jax.devices())
    p = argparse.ArgumentParser()
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    rows, arrays = {}, {}
    for name, job in reg['cases'].items():
        root = Path(job['training'])
        path = root / 'monitor' / f"{job['final_steps']}.json"
        report = root / 'report.json'
        if not path.exists() or not report.exists():
            rows[name] = dict(state='UNMEASURED', model_version='v2')
            continue
        row = json.loads(path.read_text())
        if row['state'] != 'PASS' or json.loads(report.read_text())['state'] != 'COMPLETE':
            rows[name] = dict(state='PARKED_GUARD_OR_EXECUTION_STOP', raw_final=row)
            continue
        with np.load(path.with_suffix('.npz')) as saved:
            arrays[name] = {key: saved[key].copy() for key in saved.files}
        rows[name] = dict(state='ACCEPTED', model_version='v2', run_id=name, n=row['n'],
            success=row['faithful_success'], drops=row['faithful_drops'], steps=job['final_steps'],
            monitor=row, checkpoint_sha256=hashlib.sha256(
                (root / f"policy_{job['final_steps']}.pkl").read_bytes()).hexdigest())
    baseline = reg['baseline']
    for name in arrays:
        if baseline not in arrays or name == baseline:
            continue
        for key in ('reset_keys', 'dr_keys', 'direct_reset_keys'):
            assert np.array_equal(arrays[baseline][key], arrays[name][key]), key
        x, y = arrays[baseline]['faithful'], arrays[name]['faithful']
        only_full, only_mask = int(np.sum(x & ~y)), int(np.sum(~x & y))
        drop = (int(x.sum()) - int(y.sum())) / 2
        rows[name]['comparison'] = dict(baseline=baseline, success_drop_points=drop,
            unmasked_only=only_full, mask_only=only_mask, exact_mcnemar_p=exact_mcnemar(only_full, only_mask),
            classification='required' if drop > 10 else 'helpful' if drop >= 3 else 'unneeded',
            scope='E3 rotation, v2, one seed, matched 20M continuations; no per-quantum hardware inference')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(row['state'] == 'ACCEPTED' for row in rows.values())
    status = 'MEASURED' if accepted == len(rows) else 'PARTIALLY_MEASURED' if accepted else 'UNMEASURED'
    a.output.write_text(json.dumps(dict(state=status, lane='D1', accepted_rows=accepted, rows=rows,
        registration_sha256=hashlib.sha256(a.registration.read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
