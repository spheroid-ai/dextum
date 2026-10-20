"""Generate the current reviewed scorecard using saved outputs; stdlib only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'dextum_cursor_agent/runs/pass0_review_01'
DOC=ROOT/'dextum_plan/progress/2026-10-10-review'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    fixed=read(RUN/'review_eval_01/evaluation/paired_evaluators.json')
    probe=read(RUN/'review_probe_01/diagnostic/result.json')
    execute=RUN/'review_execute_01'
    training=read(execute/'e3/c_touch/report.json') or read(execute/'e3/c_touch/config.json')
    trained=read(execute/'e3_evaluation/paired_evaluators.json')
    demo=read(execute/'demo/results.json')
    rows=[['Native warm-up','ACCEPTED','14/16 within 5 mm; retry-03 historical evidence'],
          ['Fixed-policy evaluator',fixed['state'] if fixed else 'PENDING',
           (f"Training {fixed['results']['training']['success']:.1%}; faithful direct {fixed['results']['post_training']['success']:.1%}; gap {fixed['absolute_success_gap']:.1%}; paired agreement {fixed['outcome_agreement']:.1%}; n=200") if fixed else 'Registered historical C-touch seed-1 final policy; 200 shared starts'],
          ['Rollout qpos discrepancy','DIAGNOSTIC ONLY','Former 0.001 limit removed as a hard stop by user review'],
          ['Matched-step diagnostic','LOGGED' if probe else 'PENDING',
           (f"Max {probe['maximum_qpos_error']:.7g}; {probe['compared_transitions']} transitions; zero-contact endpoint max {probe['zero_contact_endpoint_maximum']}; see per-coordinate/contact distributions") if probe else 'Nonblocking call-sequence/options/contact/dtype investigation'],
          ['Task-loop / actor / G / assets','KEEP','8 prior tests; actor/G parity; 3 watertight mesh stand-ins'],
          ['Planner / recognition / touch / estimators','KEEP STAND-INS','Dijkstra, force regimes, 11 Ideal pads, declared dummy estimators'],
          ['E3 original C-touch seed 1',training['state'] if training else 'NOT STARTED',
           'One original 100M-step attempt; 8192 environments; unchanged task and reward'],
          ['Rerun paired evaluation','MEASURED' if trained else 'PENDING',
           (f"Training {trained['results']['training']['success']:.1%}; faithful direct {trained['results']['post_training']['success']:.1%}; paired agreement {trained['outcome_agreement']:.1%}") if trained else 'Both evaluators required'],
          ['External datasets','OWNER TRACK','Sam; not Pass 0 or Pass 1 dependencies'],
          ['Hardware','OWNER TRACK','Build step 1 belongs to Pass 1'],
          ['Metriplector decision','PENDING CONFIRMATION','Separate physical reference + anonymous learned content; Peter and Dan before Pass 2']]
    if demo:
        for row in demo['rows']:
            rows.append([row['asset']+(' / blackout' if row['blackout'] else ' / vision on'),demo['state'],
                         f"Success {row['success']:.1%}; drops {row['drops']:.1%}; n={row['n']}"])
    else:rows.append(['Three-shape demo','NOT STARTED','96 trials after D5 PASS; no rates measured'])
    complete=bool(fixed and fixed['state']=='PASS' and training and training['state']=='COMPLETE'
                  and trained and demo and demo['state']=='COMPLETE')
    state='CLOSE_OUT_RESULTS_READY' if complete else ('EVALUATOR_AGREEMENT_FAILED' if fixed and fixed['state']!='PASS' else 'IN_PROGRESS')
    result=dict(state=state,review='User review 2026-10-10',rows=rows,training_attempts_used=int(training is not None),
        close_out_evidence_complete=complete,fixed_policy=fixed,rerun_evaluators=trained,demo=demo,
        mapping_audit=dict(design='Conventional Pass 0; separate physical-reference resolution pending Peter/Dan',
                           implementation='No Metriplector component or compliance claim'),
        limits='No real-hand, realistic Kit B sensor, tactile-benefit, or Metriplector competence claim')
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'scorecard.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# DEXTUM v1.2 — reviewed Pass 0 scorecard\n\nStatus: **'+state+'**.\n\n'
    text+='[Governing review](../../REVIEW_2026-10-10.md) · [Progress](../../PROGRESS.md)\n\n'
    text+='| Component | Status | Measured result / decision |\n|---|---|---|\n'
    text+=''.join('| '+' | '.join(r)+' |\n' for r in rows)
    text+='\nTraining attempts used: '+str(result['training_attempts_used'])+'. Close-out evidence complete: '+str(complete)+'.\n\n'
    text+='Mapping audit: conventional control/simulator interfaces only; proposed model split pending confirmation; no compliant Metriplector implementation claimed.\n\n'
    text+=result['limits']+'.\n'
    (DOC/'SCORECARD.md').write_text(text)
    print(json.dumps(dict(state=state,training_attempts_used=result['training_attempts_used'],scorecard=str(DOC/'SCORECARD.md'))))


if __name__=='__main__':main()
