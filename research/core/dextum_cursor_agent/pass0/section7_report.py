"""Summarize fetched Pass 0 records using only the standard library locally."""
import hashlib
import json
from pathlib import Path
import shutil
from .review_gate import require_review_pass

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'dextum_cursor_agent/runs/pass0_section7_01'
DOC = ROOT/'dextum_plan/progress/2026-10-10-section7'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    qualification = read(DOC/'D5_QUALIFICATION.json')
    require_review_pass(qualification)
    execution = RUN/'section7_execute_01'
    training = read(execution/'e3/c_touch/report.json')
    evaluation = read(execution/'e3_evaluation/paired_evaluators.json')
    demo = read(execution/'demo/results.json')
    control = read(execution/'compiler_control.json')
    pipeline = read(RUN/'section7_pipeline_01/result.json')
    validation = read(DOC/'ARTIFACT_VALIDATION.json')
    shutdown = read(DOC/'cloud/shutdown_confirmation.json')
    rows = demo.get('rows', []) if demo else []
    complete = bool(training and training['state'] == 'COMPLETE' and evaluation and
                    evaluation['finite'] and evaluation['n'] == 200 and demo and
                    demo['state'] == 'COMPLETE' and len(rows) == 6 and
                    len({(r['asset'], r['blackout']) for r in rows}) == 6 and
                    all(r['n'] == 16 for r in rows) and pipeline and
                    pipeline.get('training_attempts_started') == 1 and
                    pipeline.get('execution_exit_code') == 0 and validation and
                    validation.get('state') == 'VERIFIED')
    record = dict(D5='PASS', pass0_closeout='MEASUREMENTS_COMPLETE' if complete else 'OPEN',
        training=training, evaluation=evaluation, demo=demo, compiler_control=control,
        pipeline=pipeline, artifact_validation=validation, shutdown=shutdown,
        mapping_confirmation='Pending Peter and Dan before Pass 2')
    (DOC/'scorecard.json').write_text(json.dumps(record, indent=2)+'\n')
    lines = ['# Pass 0 scorecard — review §7', '',
        '**D5 PASS. '+('Pass 0 close-out measurements are complete.' if complete else 'Pass 0 remains open.')+'**', '',
        'The cube E3 rerun succeeds on 90.5% of the faithful shared-bank evaluation. '
        'The separate three-shape task-loop demo remains unsuccessful on two shapes; '
        'measurement close-out does not establish reliable three-shape control.', '',
        'The revised criteria were adopted by the user after the historical results. Earlier failed gates remain unchanged.', '',
        '| D5 criterion | Measurement |', '|---|---|',
        '| Aggregate success gap ≤0.05 | Historical checkpoint: 175/200 on both paths; gap 0 |',
        '| Exact McNemar p >0.05 | 13 training-only, 13 direct-only; p=1 |',
        '| Native warm-up ≥14/16 | 14/16 within 5 mm |',
        '| Matched-transition median max qpos <1e-5 | 3.43e-7 over 320 transitions |', '',
        'Per-start agreement is 87%, diagnostic only. The matched-step p95 0.00327 and maximum 0.02892 remain nonblocking Pass 1 investigations; substep contact sets are unmeasured.', '']
    if training:
        checkpoints = training.get('checkpoints', [])
        lines += [f"E3 seed 1: **{training['state']}**, original task/reward, 8,192 environments, 100M requested steps. "
                  f"Last saved checkpoint: {checkpoints[-1]['steps'] if checkpoints else 'none'}. "
                  f"Recorded duration: {training.get('wall_s', 0)/60:.1f} minutes.", '']
        if training.get('error'): lines += ['Recorded training error: `'+training['error']+'`.', '']
    else:
        lines += ['E3 seed 1: final training report is not yet available.', '']
    if evaluation:
        counts = evaluation['success_counts']; result = evaluation['results']
        lines += ['| Corrected E3 evaluator | Success | Physical drops |', '|---|---|---|']
        for name, label in [('training','Training wrapper'), ('post_training','Faithful direct MJX')]:
            lines += [f"| {label} | {counts[name]}/200 ({counts[name]/2:.1f}%) | {100*result[name]['drops']:.1f}% |"]
        lines += ['', f"Per-start agreement {100*evaluation['outcome_agreement']:.1f}%; exact McNemar p={evaluation['exact_mcnemar_p']:.6g}. "
                  'The aggregate gap is 0.5 percentage points. This is one C-touch seed, '
                  'with no corrected no-touch control. The faithful score is 3 percentage points '
                  'above the historical 87.5%; a reliable improvement has not been established.', '']
    if validation:
        lines += [f"Internal Brax monitoring rose from {100*validation['first_internal_monitor_success']:.1f}% "
                  f"to {100*validation['final_internal_monitor_success']:.1f}%. "
                  'Its final 128-environment score differs substantially from the registered 200-start scores. '
                  'Different start/DR draws, batch sizes and reset/rollout compilation boundaries remain confounded; '
                  'a matched-bank monitor audit is registered for Pass 1 in [follow-up findings](FOLLOW_UP.md).', '']
    if rows:
        lines += ['| Held-out instance | Vision | Success | Drops |', '|---|---|---|---|']
        for row in rows:
            lines += [f"| {row['asset']} | {'Off after first contact' if row['blackout'] else 'On'} | "
                      f"{round(row['success']*row['n'])}/{row['n']} | {round(row['drops']*row['n'])}/{row['n']} |"]
        lines += ['', 'Demo uses native C MuJoCo, Ideal camera packets and 11 lumped Ideal pads. '
                  'It retains the registered historical cube actor and scripted family stand-ins. '
                  'These are not multi-shape scores for the newly trained checkpoint. '
                  'It does not establish Kit B sensor realism, learned perception or tactile odometry.', '']
    else:
        lines += ['Three-shape demo: no measured result yet.', '']
    if control:
        lines += [f"Nonblocking compiler control: default {control['default_success']}/200, alternate {control['alternate_success']}/200; "
                  f"agreement {100*control['agreement']:.1f}%; exact McNemar p={control['exact_mcnemar_p']:.6g}. "
                  'The alternate disables constant folding. This pair shows compiler sensitivity but does not '
                  'reproduce 87% agreement or fail the withdrawn 90% rule. It does not establish a universal noise floor.', '']
    else:
        lines += ['Nonblocking compiler control: no paired result available.', '']
    lines += ['Validation: 12 cloud gate/runtime checks passed; actor/motor export parity passed '
              '(3.84e-6 / 2.05e-7). All 154 cloud output files were downloaded and SHA-256 verified. '
              'The current and earlier DEXTUM instances are confirmed stopped; retained cloud disks remain allocated. '
              'No simulation, training or rendering ran locally.', '',
              'Mapping audit: conventional PPO and separately named simulator mechanics; no Metriplector processor is introduced. '
              'Design resolution at spec §9 remains pending Peter and Dan before Pass 2; no new implementation compliance is claimed.', '',
              'Sources: [registration](REGISTER.md), [D5 qualification](D5_QUALIFICATION.json), '
              '[machine-readable scorecard](scorecard.json), [copied evidence manifest](results_manifest.json), '
              '[artifact validation](ARTIFACT_VALIDATION.json), [shutdown confirmation](cloud/shutdown_confirmation.json), '
              '[Pass 1 follow-up](FOLLOW_UP.md).', '']
    (DOC/'SCORECARD.md').write_text('\n'.join(lines))
    manifest = []
    # Keep progress evidence in dextum_plan; binary models and arrays stay in RUN.
    for source in RUN.rglob('*'):
        if not source.is_file() or source.suffix not in ('.json', '.jsonl', '.log', '.csv'):
            continue
        target = DOC/'results'/source.relative_to(RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(dict(path=str(target.relative_to(DOC)), bytes=target.stat().st_size,
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (DOC/'results_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(pass0=record['pass0_closeout'], copied_evidence_files=len(manifest))))


if __name__ == '__main__': main()
