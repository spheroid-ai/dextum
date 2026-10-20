"""Read-only result interpretation plus a small Pass 0 scorecard (stdlib only)."""
import argparse
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def build(run):
    physics_paths = [run/"d5/qualification.json", *sorted(run.glob("d5_retry_*/d5/qualification.json")),
                     *sorted(run.glob('d5_nofold_*/d5/qualification.json'))]
    physics_results = [(p, read(p)) for p in physics_paths if p.exists()]
    physics_path, physics = physics_results[-1] if physics_results else (None, None)
    demo_paths = sorted(run.glob("demo_*/demo/results.json"))
    demo = read(demo_paths[-1]) if demo_paths else None
    actor = read(run/"actor_01/actor.audit.json")
    assets = read(run/"assets_01/qualification.json")
    diagnostic = read(run/"d5_diagnostic_01/diagnostic/result.json")
    kernel = read(run/'d5_kernel_01/diagnostic/result.json')
    e3 = read(run/"e3_01/c_touch/report.json") or read(run/"e3_01/c_touch/config.json")
    runtime_path = run/"runtime_01/tests.log"
    runtime = runtime_path.exists() and runtime_path.read_text().rstrip().endswith("OK")
    rows = [
        ["D5 physics", physics["state"] if physics else "PENDING",
         (f"{physics['within_5mm']}/16 within 5 mm; training/direct max qpos error "
          f"{physics['training_vs_direct_max_qpos_error']:.6g} (limit 0.001)") if physics else "16-start cloud qualification", "hard stop"],
        ["Task-loop boundary", "PASS" if runtime else "PENDING", "8 remote checks" if runtime else "Not measured", "keep"],
        ["Actor + motor export", actor["state"] if actor else "PENDING",
         f"Actor error {actor['actor_max_error']:.3g}; G error {actor['motor_max_error']:.3g}" if actor else "Parity not measured", "keep if parity passes"],
        ["Shape assets", assets['state'] if assets else "PENDING", "3 watertight hand-written meshes and patch graphs; held-out instances" if assets else "Not measured", "stand-ins; deepen in Pass 1"],
        ["Planner / DAG", "stand-in", "Dijkstra, one yaw template, retry costs and bounded trace", "keep"],
        ["Recognition", "stand-in", "Observable force regimes; unobserved pose/patch remains unknown", "deepen"],
        ["Touch", "stand-in", "11 lumped Ideal pads; Kit B Realistic level not yet integrated", "Pass 1"],
        ["Pose and phi estimators", "stand-in", "No blind odometry; no learned physical-parameter estimator", "deepen"],
        ["E3 corrected-physics rerun", e3['state'] if e3 else "BLOCKED BY D5", "One C-touch seed, original task/reward; zero attempts started" if not e3 else "Registered single arm", "after qualification"],
        ["External touch data", "PENDING", "Release/license discovery recorded; gated TouchScale access and VTDexManip download unresolved", "complete access and cloud download"],
        ["Hardware", "PENDING", "Human-owned track; no measured hardware results", "owner review"],
        ["Metriplector", "not in Pass 0", "Physical-state design conflict recorded in Mapping audit", "separate reference or compliant redesign"],
    ]
    if demo:
        for d in demo["rows"]:
            rows.append([d["asset"]+(' / blackout' if d["blackout"] else ' / vision'),
                         "measured", f"success {d['success']:.1%}; drops {d['drops']:.1%}; n={d['n']}", "review"])
    else:
        rows.append(['Three-shape physical demo','BLOCKED BY D5','Guard stopped execution; no success rate measured','after qualification'])
    if diagnostic:
        rows.append(["D5 identical-start diagnostic", "diagnostic only",
                     f"Max qpos error {diagnostic['max_qpos_error']:.6g}; does not override qualification", "investigate"])
    if kernel:
        rows.append(['D5 matched-step diagnostic','diagnostic only',
                     f"Max qpos error {kernel['max_comparable_qpos_error']:.6g} across {kernel['compared_transitions']} comparable transitions",'investigate'])
    state = "HARD_STOP" if physics and physics["state"] != "PASS" else "IN_PROGRESS"
    result = dict(state=state, rows=rows, training_attempts_used=int(e3 is not None),
                  physics_result=str(physics_path.relative_to(run)) if physics_path else None,
                  physics_history=[dict(path=str(p.relative_to(run)), state=v['state'], within_5mm=v['within_5mm'],
                                        direct_error=v['training_vs_direct_max_qpos_error']) for p,v in physics_results],
                  pass_closed=False, review="Pending the named owners; never auto-fabricated",
                  limits="No Kit B, real-hand, tactile-benefit, learned-perception or Metriplector competence claim",
                  next_action="Resolve D5 stepping discrepancy, then complete Pass 0 E3 rerun and three-shape demo",
                  next_pass="Existing dextum_plan Pass 1, adjusted after this scorecard review")
    (run/"scorecard.json").write_text(json.dumps(result, indent=2)+"\n")
    md = "# DEXTUM v1.2 — Pass 0 scorecard\n\nStatus: **"+state+"**. Pass review remains open.\n\n"
    md += "| Component | Status | Observation | Decision |\n|---|---|---|---|\n"
    md += ''.join('| '+' | '.join(r)+' |\n' for r in rows)
    md += "\n"+result["limits"]+".\n\nMapping audit: `../../pass0/REGISTER.md`. Design and implementation compliance are recorded separately.\n"
    (run/"SCORECARD.md").write_text(md)
    table = '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in ('Component','Status','Observation','Decision'))+'</tr></thead><tbody>'
    table += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
    document = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DEXTUM Pass 0</title>'
    document += '<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#1a2433}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.8rem;border-bottom:1px solid #dce2e8}th{background:#f0f4f8}h1{font-size:2rem}p{line-height:1.6}</style>'
    document += '<h1>DEXTUM v1.2 · Pass 0</h1><p>'+html.escape(state)+' · review open</p>'+table+'<p>'+html.escape(result["limits"])+'.</p>'
    (run/"index.html").write_text(document)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(a.run), indent=2))
