"""Freeze D3 pure-teacher data budget and train/validation split before learning."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--d2-registration', type=Path, required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d2 = json.loads(a.d2_registration.read_text())
    sources = {name: sha(a.source / name) for name in d2['source_sha256']}
    for name in ('encoder_core.py', 'train_encoder.py'):
        relative = 'workorder14/lane_d/' + name
        sources[relative] = sha(a.source / relative)
    run = Path(a.run_root)
    record = dict(state='REGISTERED', authority='work_order_2026_10_14_lane_D3',
        registered_at=time.time(), numerical_execution='GPU only',
        source_sha256=sources, package_versions=d2['package_versions'],
        inputs_sha256=d2['inputs_sha256'], model=d2['model'], teacher=d2['teacher'],
        model_version='v2', teacher_success=176, n=200,
        configuration=dict(seed=1103, worlds=8192, unroll=40, environment_steps=20000000,
            minibatch=2048, gradient_updates_per_collection=64,
            aggregate_retention='every_tenth_control_step_all_retained_no_eviction',
            retain_every_control_steps=10, hidden=[512, 256, 128],
            learning_rate=.0003, maximum_gradient_norm=1, actor_input=93, actor_output=200,
            initialization='fresh MLP',
            loss='per-channel scaled MSE against Realistic teacher sensor window',
            scales='optical60:5N; ratios40:10; barometric20:20N; location/PVDF/slip/age:1'),
        budget_interpretation='Bounded20M pure-teacher collected transitions, rounded20,316,160; retain every tenth control frame; no mixed DAgger states',
        validation_split='Entire environment/physics ids index%10==9 held out from all fitting; evaluation seed2026/dr7 absent from collection',
        evaluation_updates=[11, 22, 33, 44, 55, 62], initial_monitor=True,
        bank=d2['bank'], guard=d2['guard'], detector_criterion=d2['detector_criterion'],
        maximum_projected_budget_s=9600, maximum_children=2, child_cap_s=7200,
        detector=str(run / 'detector/result.json'), qualification=str(run / 'qualification/full_state'),
        fresh_process_resume=str(run / 'process_resume/full_state/process_resume.json'),
        training=str(run / 'training'),
        task='Frozen v2 C-touch teacher with all actor/G sensing replaced by sensorless estimate at evaluation',
        inference='Estimate replaces 200 packet values, derived legacy prefix, and G force/shear/slip/contact sensing',
        mapping_audit=d2['mapping_audit'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(path=str(a.output), sha256=sha(a.output))))


if __name__ == '__main__':
    main()
