# Project page text — 20 October 2026 (v0, 18 October)

*Copy supplied by the reviewer in the 18 October conversation. Preserved as received; the build's evidence corrections are recorded separately in `dextum_release_20oct/docs/PAGE_COPY_AUDIT_20261018.json`. Every number is an accepted result with model version and seed count; anything marked PENDING is replaced by the freeze scorecard (19 October, 18:00 CEST) or stays as "in progress". Figures are named by the scorecard they read.*

## Title

**Which Touch Does a Hand Need?**
*Deciding a tactile kit in simulation before building it — and testing a contact-state representation with an oracle*

Spheroid · Peter Toth, Dan Oprisa, Sam Padilla

[Paper PDF] · [Code] · [Model and banks] · [Scorecards]

## Tagline

We put a physically specified tactile kit on a commercial robot hand in simulation, trained with and without touch on faithful contact physics, and asked three questions before any hardware was built: what the hand costs, what realistic sensing costs, and which channels matter. Then we asked a fourth with an oracle: is a description of manipulation written at the contact — which hand elements touch which object patches, how hard, slipping or not — enough to control from?

## Hero figure

*Figure 1 — the sensorised Aero Hand Open in MuJoCo with the kit's ten pads and forty cells, and the touch map during a reorientation; right, the four questions and the answers.* (Render from the v2 model; the touch map from a Realistic-level rollout of the 88.0% policy.)

## Three numbers

| **89.2%** | **−2.5 points** | **17.5 points** |
|---|---|---|
| success from the contact state alone (three seeds: 89.0, 90.0, 88.5) against 84% from the full simulator state (one seed) — the contact state is a sufficient statistic for this task | what realistic sensing costs against ideal sensing (90.5 → 88.0, p = 0.38, not significant) | what one contact parameter moved the result by (73.0 → 90.5, p = 3 × 10⁻⁶), caught by a pre-registered check before a hardware decision was taken |

## The method in one figure

*Figure 2 — simulate the kit → train with touch → ablate the signals → decide the hardware → build.* The kit is specified physically (cell layout, noise, gain, drift, delay, cross-talk, onset statistics) and simulated at three fidelity levels; policies are trained on faithful physics with a pre-registered protocol; channels are masked; the signal requirement table decides what the first hardware carries. The loop runs before the hardware exists.

## Results

### 1. One contact parameter moved the result by 17.5 points

*Figure 3 — bare hand → sensorised v1 → sensorised v2 → rigid pads, from scratch, Ideal sensing, one seed, 200 shared starts.*

Bare Aero 91.5%. Sensorised hand, model v1, 73.0%. The same hand with the pads' friction matched to the hand's own geoms (model v2, nothing else changed) 90.5% — +17.5 points, p = 3 × 10⁻⁶. Rigid pad contacts on v2: 87.5% (−3, p = 0.29, not significant). On v1, changing pad stiffness and thickness recovered nothing (58 and 60 against 64): the cause was friction, not compliance. The 27.5-point "hand cost" we first measured was a parameter error, found by a pre-registered pre-check before the kit's geometry was changed. **The sensorised hand costs nothing against the bare hand.**

### 2. Realistic sensing costs 2.5 points, not significant

*In Figure 3.* From scratch at full budget on v2: Ideal sensing 90.5%, Realistic sensing (0.05 N noise, ±5% gain, bias drift, 2 mm localisation, 10 ms delay, barometric cross-talk, PVDF onset statistics) 88.0%; paired p = 0.38, one seed.

### 3. Touch under realistic sensing: a reliability gain, seed-dependent

*Figure 4 — paired bars per seed, v2; the v1 result beside it under its own label.*

Three seeds, 20.3 M warm starts from each arm's own teacher, Realistic level, v2: touch 83.5 / 83.5 / 83.0; no touch 74.0 / 83.0 / 74.5; gains +9.5 (p = 0.004), +0.5 (p = 1.0), +8.5 (p = 0.019); mean +6.2, 95% interval −6.1 to +18.4. The touch arm barely varies across seeds; the no-touch arm does. On this task touch is a modest, seed-dependent benefit once the model is right — not demonstrated necessity. The earlier +11.0 (three seeds, 95% interval +6.5 to +15.5) was measured on the mis-parameterised v1 model and is reported under that label as the finding that led to the correction.

### 4. Which channels matter: none individually

*Table 1 — the signal table, reorientation row; v2 rows and v1 rows labelled.*

v2, 20.3 M continuations from the Realistic teacher: unmasked 87.5%, normal force only 85.0% (p = 0.42), palm map removed 87.5% (p = 1). No single channel is required for reorientation; normal force with contact detection carries the benefit. v1 (labelled): shear, slip, PVDF and contact location individually unneeded; a dense 3 mm skin on the proximal pads and palm 62 against 62 — no gain on this task.

### 5. Contact-space actions and joint-space PPO

*Table 2 — one table, both hands, model versions labelled.*

Bare hand, one seed: contact-space actions with touch 91.5%, contact-space without touch 80.5%, joint-space PPO (the hand maker's baseline) 74.5%. Sensorised v1 (labelled): 55–60 / 44.5 / 42.5–51. v2 matched set: PENDING (runs only if a host frees before the freeze; otherwise this row stays v1).

### 6. Held-out physical variations

Bare hand, one seed: 77.85 / 64.40 / 61.15 on the held-out variation banks. v2 three-seed arms: PENDING C9 (evaluation only).

### 7. Oracle O1: the contact state is a sufficient statistic — on one task

*Figure 5 — ceiling, structured (three seeds), flat (two), contact-only, patch identities removed; the task stated in the caption.*

**The task, exactly:** a cube resting in the sensorised Aero's palm; rotate ±30° yaw; then hold yaw error ≤ 10° and tilt ≤ 30° for two seconds. One object, one hand, one goal pattern. The oracle supplies the exact contact state (which hand elements touch which object patches, forces, slip, events) and the DEXTUM hypothesis fields; PPO learns the controller; the structured encoder groups contacts by element and patch; the information-equivalent flat control receives the same numbers without the grouping.

| Input | Success |
|---|---|
| Privileged ceiling — the full simulator state | 84.0% (one seed) |
| Structured DEXTUM — oracle contact state and hypothesis fields only | **89.0, 90.0, 88.5** — mean 89.2 (three seeds) |
| Information-equivalent flat control | 83.0 (seed 1); 85.5 (seed 2, pending the recovery of one monitor check) |
| Contact-only ablation | 85.0 |
| Patch identities removed | 87.5 |

A policy given only the contact state and the hypothesis fields does at least as well as one given the full simulator state (the registered sufficiency criterion, met on all three seeds); superiority over the ceiling is not statistically established against one ceiling seed (p = 0.076) and is not claimed. The grouping structure adds about five points over the flat control (+6, p = 0.023; +4.5), below the 10-point bar we registered for "the structure, not the information, carries it". The decomposition: contact-only ≈ ceiling — the contact state alone carries what the full state carries; the hypothesis fields add four to five points; patch identities two; structure five over flat. This task gives structure little to do: there is nothing to compose, no new geometry, no second hand. The registered next oracle tests put work on each axis — sequences (O1b), geometry (O3′), a second hand (O5′) — and are listed below as pending.

### 8. Touch without touch: not at this budget

Distilling the Realistic touch teacher (88.0%) into a student that sees vision features, proprioception history and the task only — no touch — on the sensorised hand at 20.3 M steps: 70.5% success, 2.5% drops, inside or below the no-touch arms' range (74–83). No touch advantage is recovered by distillation at this budget. Inferred touch (an encoder from proprioception and vision to the touch features): PENDING D3.

### 9. Touch-necessary tasks

PENDING — stiffness sort (five identical-looking boxes, stiffness class drawn per episode, crush limit; admission criterion no-touch ≤ 30%, touch ≥ 60%, gap ≥ 25 at p < 0.01, reproduced on a second seed). Hold under perturbation: the first two implementations were task-definition failures (entry states not statically stable; the actor initialised from the rotation policy) and are reported as lessons, not results; hold v0.1 with hold-still initialisation and a scripted-engage bank is registered.

## The faithfulness protocol

*Figure 6 — the methods figure; Table 3 — the calibration table.*

GPU-batched training compiled a model that differed from the one the native evaluator ran: the training-time success level sat 0.2–0.3 above the faithful one for every arm, and the state residual carried the bug's signature (4 × 10⁻² against a compiler floor near 10⁻⁵). The protocol that now guards every run: a matched-step detector on the collection path (median residual ≤ max(10⁻⁴, 10× the compiler floor on the task's states), hard ceiling 10⁻²) as the hard stop; a native MuJoCo evaluator for every reported number on a shared 200-start bank with held-out variation banks; a monitor guard that compares the internal and faithful evaluators at 31 checks per run and stops on exact McNemar p < 0.01 at a check, a gap above 15 points, or a running mean above 3 with clustered p < 0.01; exact fresh-process resume; attempts counted only on reaching budget or a learning stop. Two guards we first set — a 14-of-16 warm-up criterion and a 5-point absolute monitor — fired on noise and were replaced by the calibrated rules above; the history is in the calibration table. After the fix the two evaluators agree exactly on the same checkpoint (87.5% on both paths). Every number on this page is a faithful-evaluator number.

## Releases

- The sensorised Aero model: MuJoCo XML with the ten pad colliders and forty cell sites, model v1 and v2 labelled, placeholders marked.
- The sensor simulation: Ideal / Realistic / Degraded, the noise model, the PVDF proxy, the pooling of contacts to cells.
- The faithful evaluator, the detector, the monitor guard with its calibration, the matched-start statistics.
- The 200-start banks (reorientation; hold v0.2) and the held-out variation banks.
- Training configs, the contact-space action, the warm-start protocol; the checkpoints behind every number, with hashes.
- The oracle O1 inputs: the oracle contact state, the hypothesis fields, the structured and flat encoders.
- The 50 superquadric assets and the exporter v0.
- Licence: Apache-2.0 for code and model; CC-BY-4.0 for banks and assets.

## The DEXTUM programme (roadmap, labelled)

Manipulation as a sequence of contact states; the physics of contact the same for every hand; a representation of skill at the contact — learned from humans, grounded by simulation, structured by physics — that transfers across objects, hands, sensors and tasks and lets a robot verify what it did. Measured so far: the oracle sufficiency result above, on one task. Registered next, with their decision rules and not yet run: **O1b sequences** (goal-conditioned reorientation trained on single goals, tested on held-out goals and chains never trained), **O3′ geometry** (randomised primitive families, held-out dimensions, a held-out family, meshes), **O5′ second hand** (the LEAP Hand in the oracle: frozen core plus a new connector against from scratch at a tenth of the budget); **E24** (UVTA's human contact tokens predicting the Aero's held-out sequences); **E22** energy accounting and the dexterity score; **E23** the contact-mode physical model against an equal-size network on six gates. Each has a ceiling, a structured arm and an information-equivalent control; the ceiling sets the difficulty into a 40–70% window; the claim is the interaction between structure and the held-out axis.

## Citation

```
@misc{spheroid2026whichtouch,
  title  = {Which Touch Does a Hand Need? Deciding a tactile kit in simulation before building it},
  author = {Toth, Peter and Oprisa, Dan and Padilla, Sam},
  year   = {2026},
  note   = {Project page: [URL]}
}
```
