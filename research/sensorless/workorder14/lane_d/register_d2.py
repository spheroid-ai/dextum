"""Freeze D2 DAgger design and all source before numerical qualification."""
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
    p.add_argument('--d1-registration', type=Path, required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    d1 = json.loads(a.d1_registration.read_text())
    job = d1['cases'][d1['baseline']]
    names = ['sensorless.py', 'learner.py', 'dagger_core.py', 'dagger_detector.py',
             'train_dagger.py', 'student_runtime.py', 'export_student.py',
             'test_interfaces_gpu.py', 'inferred_touch.py', 'supervised_pipeline.py', 'report_sensorless.py']
    sources = {name: sha(a.source / name) for name in d1['source_sha256']}
    for name in names:
        relative = 'workorder14/lane_d/' + name
        sources[relative] = sha(a.source / relative)
    run = Path(a.run_root)
    record = dict(state='REGISTERED', authority='work_order_2026_10_14_lane_D2',
        registered_at=time.time(), numerical_execution='GPU only',
        source_sha256=sources, package_versions=d1['package_versions'],
        inputs_sha256=d1['inputs_sha256'], model=job['model'], teacher=job['checkpoint'],
        model_version='v2', teacher_success=176, n=200,
        configuration=dict(seed=1, worlds=8192, unroll=40, environment_steps=20000000,
            minibatch=2048, gradient_updates_per_collection=640,
            aggregate_retention='all_collected_examples_no_eviction',
            teacher_mix_decay_steps=4000000,
            hidden=[512, 256, 128], learning_rate=.0003, maximum_gradient_norm=1,
            actor_input=93, actor_output=7, initialization='fresh zero output motor prior; no critic',
            loss='mean squared error of normalized teacher executed motor targets',
            replay_sampling='uniform over all examples collected so far'),
        budget_interpretation='20M simulator transitions visited by DAgger mixture; rounded to 20,316,160; qualification updates discarded',
        mixture='beta=max(0,1-collected_steps/4M), Bernoulli per world/control step; evaluation always pure student',
        aggregate_capacity=20316160, aggregate_bytes=8126464000,
        evaluation_updates=[11, 22, 33, 44, 55, 62], initial_monitor=True,
        bank=d1['banks'], guard=d1['guard'], detector_criterion=d1['detector'],
        maximum_projected_budget_s=9600, maximum_children=2, child_cap_s=7200,
        detector=str(run / 'detector/result.json'), qualification=str(run / 'qualification/full_state'),
        fresh_process_resume=str(run / 'process_resume/full_state/process_resume.json'),
        training=str(run / 'training'), export=str(run / 'export'),
        task='Unchanged E3 rotation on accepted v2 sensorised hand',
        vision='Existing qualified noisy tag-camera simulator; 13 measured features with validity/age',
        inference='No touch, G, simulator or teacher on the exported seven-target motor path',
        comparison='Teacher88%; each v2 no-touch seed and seed mean, with protocol/input labels',
        mapping_audit=d1['mapping_audit'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(path=str(a.output), sha256=sha(a.output), aggregate_bytes=record['aggregate_bytes'])))


if __name__ == '__main__':
    main()
