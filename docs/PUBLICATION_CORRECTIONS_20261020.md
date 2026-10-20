# Publication audit — 20 October 2026

Page/release revision **v0.1.1**. This is a presentation and attribution update;
no experiment, accepted measurement, model version or registered rule changes.
The original 18 October reviewer copy remains preserved in
[dextum-page-20oct-text.md](dextum-page-20oct-text.md).

## Bare-hand labels: the page was correct

One seed, Ideal sensing, 108,134,400 steps, 200 shared starts, faithful direct
MJX evaluator:

| Arm | Success | Held-out physics macro success |
|---|---:|---:|
| C-touch, contact-space | 91.5% | 77.85% |
| C-no-touch, contact-space | **74.5%** | **61.15%** |
| Plain PPO, joint-space (`a_task`) | **80.5%** | **64.40%** |

The original 18 October prose draft's section 5 reversed the two control
labels. The execution edition of the results package already has the correct
C2 ordering. Use the labelled table above in the paper; never copy an
unlabelled three-number sequence.

- **C-no-touch:** run
  `pass1_audit_01/controls_01/e3/c_notouch`, checkpoint
  `policy_108134400.pkl`; faithful `post_training` count **149/200**.
  SHA-256:
  `63c858add49c11fff4c422b56c6ea219453e8b3ec846f0ef3e55ddcf8de2da86`.
  [Original evaluator report](../site/data/evidence/bare_notouch.json).
- **Plain PPO:** run
  `pass1_audit_01/controls_01/e3/a_task`, checkpoint
  `policy_108134400.pkl`; faithful `post_training` count **161/200**.
  SHA-256:
  `6120ce66cd2c10de10ae4734ec65156049b66418bcd2773f2005ec8545496902`.
  [Original evaluator report](../site/data/evidence/bare_ppo.json).
- [Held-out report](../site/data/evidence/bare_heldout.json) names the same
  arms and checkpoint hashes. Its `held_out_macro_success` fields are
  `c_touch=0.7785`, `c_notouch=0.6115`, `a_task=0.644`. These average ten
  physical conditions; they are not extra training seeds.

## V2 results made visible

Both comparisons were present in plots behind tabs. They now appear as
permanently visible sections, with HTML tables generated from the same
hash-verified records as the figures.

| Seed | C-touch | C-no-touch | Gain | Paired p |
|---|---:|---:|---:|---:|
| 1 | 83.5% | 74.0% | +9.5 points | 0.0043 |
| 2 | 83.5% | 83.0% | +0.5 points | 1.0 |
| 3 | 83.0% | 74.5% | +8.5 points | 0.0186 |

Mean **+6.2 points**, 95% seed interval **−6.1 to +18.4**.
Model v2, Realistic sensing, matched 20.316M warm starts.
Runs: `pass1_v2_touch_recheck_01/seed{1,2,3}_{c_touch,c_notouch}`.
[Three-seed report](../site/data/evidence/recheck.json).

| Input | Success | Paired p vs unmasked |
|---|---:|---:|
| Unmasked | 87.5% | Reference |
| Normal-only | 85.0% | 0.424 |
| Palm map removed | 87.5% | 1.0 |

Model v2, one seed, equal-budget 20.316M continuations. Runs:
`workorder14_lane_d1_r04/d1_{none,normal_only,palm}_seed1`.
[Channel report](../site/data/evidence/channels.json). Neither contrast is
significant; no equivalence or general claim about other tasks follows.

## Title, citation, paper and sensorless status

- The project-page citation is **DEXTUM: Manipulation, Described Through
  Contact**, version **0.1.1**, matching the README and `CITATION.cff`.
  The prospective paper title is not used to cite an unpublished paper.
- **No paper PDF is published.** Application wording is “project page
  (paper to follow)”. [Ready-to-use application text](APPLICATION_ARTIFACTS.md).
- D2 distillation: **70.5% success, 2.5% drops**. D3 inferred touch:
  **41.5% success, 10.0% drops**. Neither recovers the touch advantage
  at the tested budget. Sensorless deployment is a design goal, not a
  demonstrated capability. These are simulation results, one seed each.
- Dense skin is stated explicitly as **62.0% against 62.0% unmasked,
  model v1**.
- The 4 October film remains labelled historical, including its 12/40
  bank result. No accepted v2 video or complete saved v2 motion trajectory
  suitable for rendering was located. Evaluation outcome arrays and
  qualification traces are not substituted for a final-policy rollout.

The original source reports, their hashes and all historical registrations
remain unchanged.
