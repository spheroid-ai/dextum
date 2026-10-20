"""Freeze Lane D1 inputs and source; stdlib-only manifest creation, no run."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .source_closure import local_import_closure

MODEL_SHA = 'a2e0c670506cb69267bac11e80fa24c467a782cca3bedb901290adc64444c924'
TEACHER_SHA = '1c1dba7e9c222d5c9279517989f2323b2f383e1448df8d7f1e657a15a440c6a7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--historical-registration', type=Path, required=True)
    p.add_argument('--local-model', type=Path, required=True)
    p.add_argument('--local-teacher', type=Path, required=True)
    p.add_argument('--remote-model', required=True)
    p.add_argument('--remote-teacher', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.local_model) == MODEL_SHA
    assert sha(a.local_teacher) == TEACHER_SHA
    previous = json.loads(a.historical_registration.read_text())
    configuration = dict(previous['cases']['c_touch_realistic_v2']['configuration'])
    configuration.update(num_timesteps=20000000, num_evals=7)
    source_names = []
    for directory in ('dextum', 'envs', 'experiments', 'pass0', 'pass1', 'pass1b', 'pass1c', 'tools', 'vendor'):
        source_names.extend(path for path in (a.source / directory).rglob('*')
                            if path.is_file() and path.suffix in ('.py', '.json')
                            and not {'__pycache__', 'records', 'runs'} & set(path.parts))
    source_names.append(a.source / 'workorder14/__init__.py')
    source_names += [a.source / 'workorder14/lane_d' / name for name in
                     ('__init__.py', 'train_masking.py', 'calibration.py', 'pipeline.py', 'report_d1.py',
                      'register_d1.py', 'source_closure.py')]
    closure = local_import_closure(a.source, ['workorder14/lane_d/' + name for name in
        ('train_masking.py', 'calibration.py', 'pipeline.py', 'report_d1.py')])
    source_names.extend(closure)
    cases = {}
    # Equal-budget unmasked arm is first so both masks have a current baseline.
    for mask in ('none', 'normal_only', 'palm'):
        name = 'd1_' + mask + '_seed1'
        folder = Path(a.run_root) / name
        cases[name] = dict(arm='c_touch', seed=1, model_version='v2', sensor_level='Realistic',
            intervention={'mask': mask}, model=a.remote_model, model_sha256=MODEL_SHA,
            checkpoint=a.remote_teacher, checkpoint_sha256=TEACHER_SHA,
            teacher_steps=108134400, teacher_faithful_successes=176, teacher_bank_size=200,
            initialization='teacher actor critic normalizer; fresh optimizer; independent RNG seed1',
            configuration=configuration, maximum_projected_budget_s=9600,
            detector=str(folder / 'detector/result.json'),
            qualification=str(folder / 'qualification/full_state'),
            fresh_process_resume=str(folder / 'process_resume/full_state/process_resume.json'),
            training=str(folder / 'training'), final_steps=20316160)
    record = dict(authority='work_order_2026_10_14_lane_D1', state='REGISTERED',
        registered_at=time.time(), model_version='v2', accepted_model_sha256=MODEL_SHA,
        packaging_revision='r04_gpu_calibrated_detector_no_native_cpu_warmup',
        local_import_closure=sorted(str(path.relative_to(a.source)) for path in closure),
        teacher_sha256=TEACHER_SHA, source_sha256={str(path.relative_to(a.source)): sha(path)
                                                 for path in sorted(set(source_names))},
        inputs_sha256={a.remote_model: MODEL_SHA, a.remote_teacher: TEACHER_SHA},
        package_versions=previous['package_versions'], cases=cases,
        case_order=list(cases), baseline='d1_none_seed1', final_checkpoint_only=True,
        banks=dict(n=200, reset_seed=2026, randomization_seed=7, direct_reset_split=1),
        detector='median <= max(1e-4, 10*direct_direct_compiler_floor); diagnostics p95/max',
        audit_implementation='Existing workorder14.lane_d.calibration.audit: actual GPU collection path plus independent GPU direct-A/direct-B compilations',
        native_cpu_warmup='Omitted: non-gating legacy diagnostic under user no-CPU-run instruction',
        guard='Existing pass1c.compare.MonitorGuard and exact clustered criteria, unchanged',
        host_qualification=['actual collection detector', 'ten updates including checkpoints',
                            'fresh-process exact loss and full-state resume'],
        child_cap_s=7200, maximum_children=2, numerical_execution='GPU only',
        jax_platforms='cuda,cpu', host_callbacks='CPU backend initialized for existing host diagnostic callbacks only; default backend and simulation/learning device assertions require GPU',
        mapping_audit=dict(design='Conventional PPO, explicit physical sensor/action interfaces.',
                           implementation='No Metriplector component or compliance claim.'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(path=str(a.output), sha256=sha(a.output), cases=list(cases))))


if __name__ == '__main__':
    main()
