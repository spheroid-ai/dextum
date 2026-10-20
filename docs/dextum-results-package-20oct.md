# DEXTUM results package — 20 October 2026

**Publication update — 20 October 2026 (v0.1.1).** The [public project page](https://spheroid-ai.github.io/dextum/) and [research repository](https://github.com/spheroid-ai/dextum) are live; the paper PDF remains pending. Use “project page (paper to follow)” in the application. The original schedule below is historical. The [publication audit](https://github.com/spheroid-ai/dextum/blob/main/docs/PUBLICATION_CORRECTIONS_20261020.md) confirms the bare-hand labels: C-touch **91.5%**, C-no-touch **74.5%**, plain PPO **80.5%**; held-out physics **77.85%, 61.15%, 64.40%** in that same order. C2 in this execution edition was already correct; the 18 October prose draft reversed the controls. D2 **70.5% / 2.5% drops** and D3 **41.5% / 10% drops** are accepted negatives at their tested budgets; sensorless deployment remains a design goal. The v2 touch and channel results now appear directly in the page, with no tab required.

Execution edition, 15 October 2026. Authority: the user's revised package and
[review §25 ruling](progress/2026-10-15-section25/USER_RULING.md). This edition
supersedes older schedules; historical registrations and results remain intact.

**Purpose.** A project page in UVTA's research-page form, a paper, and a
reproducible release. The page presents the programme and its measured subset;
the paper contains measured claims only. The current site is an owner-private
preview, with public launch targeted for 20 October.

**Scope rule.** A paper claim requires a faithful-evaluator number, a registered
protocol, and either three seeds or a paired test on a shared bank with its
p-value. Statistical significance and the registered effect-size threshold are
reported separately. A non-significant difference is not equivalence.

## 1. Claims and evidence

| ID | Claim and current evidence | Before the freeze | Figure / table |
|---|---|---|---|
| C1 | **v2 reorientation:** touch gains +9.5, +0.5, +8.5 points; p=.0043, 1, .0186. Mean +6.2, 95% interval −6.1 to +18.4. Touch 83–83.5%; no-touch 74–83%. A modest, seed-dependent benefit; necessity is not established. Historical v1 +11 remains labelled. | Closed; no more reorientation seeds. | Paired bars by seed; model versions separate. |
| C2 | Bare-hand contact-touch / joint-PPO / contact-no-touch: 91.5 / 80.5 / 74.5, one seed. Sensorised v1 comparisons remain historical. **Matched v2 plain-PPO comparison missing.** | Only if a host frees: plain PPO fresh 20.3M × 3 and fresh C-no-touch 20.3M, against v2 scratch 20.3M checkpoints. Otherwise retain labelled historical evidence. | Table by hand, version and protocol. |
| C3 | **v2 channels complete:** unmasked 87.5%, normal-only 85.0% (p=.424), palm removed 87.5% (p=1). No significant differences. Individual shear, location, slip and PVDF removals are **v1 only**. No “palm map helpful” claim on v2. | Closed. | Signal table with model labels. |
| C4 | v2 Ideal 90.5% → Realistic 88.0%, full-budget scratch, one seed; 2.5-point difference, p=.383. No demonstrated sensing penalty at this budget. | Closed. | Sensing contrast. |
| C5 | Friction-only correction: v1 73.0% → v2 90.5%, +17.5 points, p=3.3119×10⁻⁶; drops 13 → 3. Bare-hand Ideal 91.5%; matched-friction rigid diagnostic 87.5%, p=.29 versus v2 Ideal. No simulation basis for a geometry change. | Resolve every exact run ID and hash at freeze, including the rigid diagnostic. | Bare → v1 → v2 → rigid attribution. |
| C6 | Dense 3 mm skin: 62% versus 62%, **v1 only**, one seed. No v2/hold recheck. | Recheck cut from this package. | Historical signal-table row. |
| C7 | **No accepted touch-necessary result.** Sort's mechanics accepted at 4.42% under §24/§25. Runtime integration and training follow. Switch gets one settling re-gate. Hold bank v0.2 stopped at 40/256 training entries; held-out collection did not start. | Sort's three core arms → admission → C-arm seed 2 and normal-only if admitted; switch if qualified; hold only if eligible bank and free capacity. | Task success bars and per-task signal rows only when accepted. |
| C8 | E21 future-touch auxiliary head: registered, not run. | Cut. | Roadmap only. |
| C9 | Historical bare-hand held-out physics: 77.85 / 64.40 / 61.15%, one seed. **Six-arm v2 evaluation missing.** | Host 3: evaluation only on registered held-out banks. | v2 T1 table if accepted. |
| C10 | Physics-faithfulness protocol: captured-model bug, actual-collection detector, faithful evaluator, compiler floor, matched-bank guard. Threshold revisions and limitations are part of the method. | Write and bind calibration records. | Methods figure and calibration table. |
| C11 | **Preliminary Oracle O1:** structured 89%, 90% (seeds 1–2); ceiling 84%, flat 83%, contact-only 85%, patch-identity removed 87.5% (seed 1). Structured–flat +6, p=.0227, below the registered +10 bar. Structured–ceiling p=.0755: observed superiority is **not established**. O1 uses its own 2,048-world protocol. | Structured seed 3, fresh flat seed 2, flat seed 3. Three-seed means decide sufficiency and structure rules. | O1 arms/seeds, drops; D only if E22 accepted. |
| C12 | **Accepted negative D2:** sensorless DAgger student 70.5%, 2.5% drops at 20.316M, below the no-touch reference range 74–83%; no touch advantage recovered at this budget. D3 inferred touch is a separate run, now training after qualification. | Finish D3; report its own channel errors and success. | Teacher / student / inferred-touch comparison, protocols stated. |

**Cut line — §25.** Freeze at **19 October 18:00 CEST (16:00 UTC)**. Every
quoted number must have a run ID, model version and hash. Registered work not
accepted by then is labelled **in progress**, with its registration and reason.
The defensible minimum is C10, C5, C1/C3/C4, preliminary C11 and negative C12.
Touch necessity, held-out transfer, the physical architecture and cross-embodiment
transfer are pending until their own scorecards qualify them. The v1 +11 figure
is never a headline. The baseline page remains useful if no further result lands.

## 2. Three hosts, one queue each

All budgets, attempts, child caps and existing lease deadlines remain binding.
No fourth concurrent host; no engine, solver or guard development. A stopped
lane reports while independent authorized work continues.

1. **Host 1 — O1:** structured seed 3 → flat seed 2 **from scratch** → flat seed 3.
   The guard-stopped flat seed 2 is a learning stop under §25, not a result.
   Preserve its checkpoint and history; do not resume it for the comparison.
2. **Host 2 — sort:** three core arms (C-touch, C-no-touch with tendon estimator,
   plain PPO), Realistic, 20.3M warm starts for C arms, fresh plain PPO →
   admission criterion → seed 2 of C arms and normal-only if admitted → switch
   if its re-gate passes → Ideal hold v0.1, 50M and ≥60%, only if free and eligible.
   Admission remains no-touch ≤30%, touch ≥60%, gap ≥25 points, p<.01, sign and
   significance reproduced on seed 2. The accepted sort is not blocked by an
   additional mechanics criterion.
3. **Host 3:** D3 → C9 six-arm held-out evaluation → E24 human/Aero tokens → E22
   existing v2 reorientation rollouts → E23 physical model versus equal-size MLP.
   Hold is not an E22 dependency. Analyses retain reproducibility and boundary
   gates; D3 retains all learning guards. Stops are preserved, never waived by
   this schedule. Current E22 repeatability stop, missing validated E24 corpora
   and missing admitted E23 replay data must be resolved or labelled pending.
4. **Conditional capacity:** C2 only as described in the claims table.

**Single press diagnosis.** User clarified full-scale rate as the registered
**0.10 m/s press rate**. Read force after joint speed is below **0.0001 m/s** for
50 consecutive 200 Hz substeps, never before 0.25s. Re-gate the switch once;
remeasure sort for the record. No change to its accepted admission.

**Not run for this package:** E21, dense-skin recheck, E25, E12-on-hold,
embodiment-anchor rehearsal, z v0 and the three-shape transfer demo.

**GPU-only amendment.** No CPU experiments or rented CPU host. Scientific
analyses use the third GPU host; source edits, archive handling, writing and
plotting already reported values are ordinary build work. No silent CPU fallback.

## 3. Release contents and provenance

- Sensorised Aero model: accepted XML, pad colliders, forty cell sites; contact
  placeholders explicitly marked; third-party geometry provenance retained.
- Ideal / Realistic / Degraded sensor simulation, pooling and PVDF proxy.
- Faithful evaluator, collection detector, monitor guard and paired statistics.
- Registered rotation and held-out banks; hold bank only if an accepted bank
  exists, never an incomplete bank advertised as the task benchmark.
- Training configs, contact-space action, warm-start protocol and checkpoints
  behind every paper number, with model/run IDs and archive SHA-256 hashes.
- Fifty superquadric assets and exporter v0.

Planned terms: Apache-2.0 for original code/model; CC-BY-4.0 for original banks
and assets. Preserve third-party licences and verify redistribution before
including dependencies. These proposed terms do not relicense upstream meshes.
The preparation ZIP is a page/docs/evidence-index snapshot, **not** the complete
training release or a final paper. Publication checklist tracks missing items.

## 4. Page in UVTA's research-page form

1. Title, tagline, kit render, real Aero photo and labelled simulation media.
2. Three headline measurements: **89–90% structured O1 versus 84% ceiling**
   (two structured seeds; superiority not established); **90.5 → 88.0%**
   Ideal → Realistic (p=.383); **+17.5 points** from correcting pad friction.
3. Method diagram: specify kit → simulate → train → ablate → inform bench work.
4. C11, C1, C3/C4/C5 and C12 figures; C9/C7/D3 only as accepted. v1 rows labelled.
5. Faithfulness method and revisions, scope limits and reproducibility links.
6. Three programme blocks: representation, separate physical reference,
   touch kit/data engine. Pending physics and cross-hand claims stay pending.
7. Release materials, provenance, citation once the paper/release exists.

The two supplied concept PNGs remain labelled concept renders. The saved
simulation video and PNG carry their historical run contexts and are not
presented as video evidence of the accepted v2 success rates.

## 5. Paper v0 structure

Working title: *Which Touch Does a Hand Need? Testing a tactile kit and a
contact representation in simulation.* Review owns the prose; agent owns
source bindings, figures, build and release audit.

1. Questions and measured contributions: tactile sensing, modelling fidelity,
   representation sufficiency. Necessity is a test, not an assumed conclusion.
2. Kit/model, sensor assumptions and contact limitations.
3. Controllers, scratch/warm protocols, banks, paired statistics and guards.
4. Accepted v2 reorientation, friction correction, channels, preliminary O1 and
   D2 negative. Add other sections only on accepted scorecards before freeze.
5. Related work, with verified primary citations before publication.
6. Limitations: simulation, one hand/task family, finite seeds, placeholders,
   scope of every comparison, absence of validated hardware or transfer claims.
7. Programme outlook. Appendix: run IDs, hashes, guard revisions and registers.

No E21, hold, physical-model, human-transfer or held-out result section is
presented as complete merely because it appeared in an older outline.

## 6a. UVTA comparison and the three DEXTUM blocks

UVTA is the layout/reference comparison for human tactile data and real-robot
learning. Its public quantitative claims require primary-source verification
in the paper; DEXTUM does not claim to reproduce its real-robot results.

| Block | Accepted now | This queue, or pending | Outside this package |
|---|---|---|---|
| Representation / learning | Preliminary O1; v2 modest touch effect, channel table and sensor contrast; D2 negative | Sort, C9, E24 human-to-Aero likelihood; C2 if capacity | E21, z v0, E25, three-shape demo |
| Physical architecture | No accepted architecture result. C10 qualifies simulator paths, not a learned physical model. | E22 energy and E23 six gates; both require accepted inputs and checks | MPPI, energy-to-go; Metriplector estimator unless already runnable without queue cost |
| Kit / data engine | Specified simulated kit, v2 sensing results, friction attribution; dense skin v1 only | UVTA pipeline inside E24; shared episode schema | Noise refit from human data; retargeting and embodiment-anchor rehearsal |

**GPU amendment:** E24/E22/E23 run on Host 3 under their registered analysis
rules, not a CPU host. DEXTUM's current headline is a physically specified kit
tested on faithful simulation, with transparent model correction, controlled
representation comparisons and reproducibility materials. Energy, physical
prediction and human-token transfer widen that statement only when measured.

## 6. Owners

- Agent: bounded queues; release packaging; page build and figure ingestion.
- Review: paper/page text, interpretation, final audit against run IDs.
- Peter, Dan, Sam: review and objections as change requests; no implicit wait.

## 7. Five-day plan and freeze

| Date | Compute | Build/text |
|---|---|---|
| 15 October | Preserve stops; O1 structured 3; sort core arms; D3 then C9; one switch diagnosis | Repository, release inventory, page and scorecard-backed figures; text v0 |
| 16 October | O1 fresh flat 2; sort; E24 if inputs available; record hold/switch verdicts | C10 methods and C5/C11 figures |
| 17 October | O1 flat 3; sort admission/conditional arms; E22 if gates pass | Add only accepted task/cross-hand sections |
| 18 October | Hold only if eligible/free; E23 if inputs admitted; conditional C2 | Figure audit and explicit pending list |
| **19 October, 18:00 CEST** | **Freeze**; stop hosts at lane ends | Every number bound to run, model and hash; close text |
| 20 October | No added result required | Number audit; public page, paper PDF and tagged release |
| 21 October | — | Update application fields from published results |
| 22 October, before 18:00 CEST | — | Submit application |

These are queue targets, not promises that stopped or input-blocked work finishes.

## Mapping audit

Design: physical state, energy and contact laws belong to the separately named
contact-mode physical reference and input/output interface. No mechanical
identity is assigned to Metriplector latent coordinates. Implementation: no
Metriplector implementation compliance or result is established by PPO,
simulator qualification or this publication plan. Historical evidence is retained.
