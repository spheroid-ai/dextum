# O1b registration card — supplied 18 October 2026

Build transcription of the user's `dextum-oracle-tests.md`, §6.1a. Earlier O1
registrations and results remain unchanged. Execution starts after the package
freeze, 19 October at 16:00 UTC. This card is not a runtime qualification.

Everything not listed is inherited from O1: model v2, sensorised Aero, 2,048
worlds, 108,134,400 steps per arm, PPO configuration, encoders, oracle inputs,
task-calibrated detector, fresh-process resume, checkpoint-inclusive throughput,
31-check clustered monitor guard and attempt counting. The executable registry
is the source of inherited settings, not a reconstructed approximation.

## Frames and goals

A goal is a rotation about a palm-frame axis, applied to object orientation at
issue time: `q_target = R_goal ⊗ q_issue`. Later goals are issued when the preceding
goal is confirmed held. Palm z is the normal, x the finger-to-wrist axis, y the
thumb-to-little axis. The goal input is the first two columns of its rotation
matrix (six values), constant until the next goal and identical across arms.

| Set | Goals |
|---|---|
| G_train | z +30°, z −30°, z +60°, z −60°, x +90°, x −90° |
| G_test | z +90°, z −90°, z 180°, y +90°, y −90° |

User clarification, 18 October: inherit O1 inputs exactly and add only the
six-value goal. The full structured and flat arms retain the pose and velocity
fields already in O1; contact-only retains its registered information ablation.
The original no-pose wording is superseded by this explicit decision.

## Success, timing and resets

Geodesic error ≤15° continuously for 1 second (20 steps at 20 Hz). A drop fails
the episode. Every goal must succeed in order. E0 is not comparable with O1's
yaw/tilt/two-second criterion.

User clarification, 18 October: each goal gets O1's registered horizon plus
exactly one second. O1 has 20 warm-up and 280 active steps (300 total); O1b
therefore allocates 320 control steps per goal, and n × 320 per chain. The
one-second success hold is inside this budget; it is not added a second time.
The existing physical warm-up occurs once at entry, not between chain goals.

Train on single goals only, uniform over G_train. User clarification,
18 October: retain O1's randomized palm-start resets and add success-guided
sampling (SGS) as a training-only reset distribution, starting with no outcomes.
New table/caged/held banks are not prerequisites. The held-out 200-start banks
remain fixed, outcome-independent, and never update SGS. Goal selection stays
uniform independently of reset sampling. Reward is O1's with geodesic error
replacing yaw/tilt terms. Monitor evaluation uses E0.

## Fixed evaluation sets

Each set has 200 episodes and published chain lists, with registered physical
draws. No evaluation outcome updates training sampling.

| Set | Definition |
|---|---|
| E0 | Single G_train goals |
| E1 | Single G_test goals |
| E2a | Two G_train goals; consecutive goals differ |
| E2b | Three G_train goals; at least one x goal; consecutive goals differ |
| E3 | Two goals, exactly one G_test goal in either position |

## Ceiling calibration and order

Ceiling seed 1 trains first. Primary set: first of E2a, E2b, E3 scoring 40–70%,
inclusive. If all exceed 70%, select E3 and add the registered four-goal E2c
set over G_train. If all are below 20%, the card refers to the floor rule. The
floor rule itself requires all arms' three-seed means, so calibration alone
does not authorize the chain-trained variant. Mixed outcomes with no set in
the window are reported; no unregistered primary-set choice is invented.

Order: ceiling s1; structured s1; flat s1; structured s2; flat s2; structured
s3; flat s3; ceiling s2; ceiling s3; contact-only s1. Thus ceiling, structured
and flat each have three seeds; contact-only has one.

One extension: if ceiling E0 monitor success rises by more than 2 points over
the last fifth of the budget, extend every arm by 50% once, equally.

## Decision rules

- Composition: mean structured-minus-flat ≥10 points on the primary set;
  exact paired McNemar p<0.01 on at least two seeds; mean interaction
  `(primary gap − E0 gap)` ≥5 points.
- Sufficiency: mean `(ceiling − structured)` ≤5 points on the primary set.
- Goal generalisation: E1 mean gap ≥10 points and p<0.01 on at least two seeds.
- Floor: if every arm's three-seed mean is below 20% on E2a, E2b and E3, report
  failure of single-goal training to reach chains, then run the registered
  variant trained on a fixed published list of 12 two-goal G_train orders.
  Other two-goal orders and E2b are held out; same arms, seeds and rules.

Report success, drops and first failed goal by arm/seed/set, the calibration
value and selected primary set, run ID, model version and hashes. The optional
target-contact-state goal arm remains exploratory and is not scheduled here.

## Mapping audit

Design: this is a conventional PPO/oracle-interface experiment. Physical goals,
contact records and action coordinates are interfaces; no Metriplector latent
coordinates receive mechanical meanings. The separate physical reference is
not a Metriplector. Implementation: no O1b runtime or transfer claim is accepted
until its build and registered GPU qualifications pass.
