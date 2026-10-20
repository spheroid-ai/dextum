# Application wording — 20 October 2026

The paper PDF remains unpublished. These corrections do not imply that an
application form has been edited or submitted.

## Links to relevant artifacts

The paragraph below is under the 1,000-character limit:

Project page (paper to follow): https://spheroid-ai.github.io/dextum/ — DEXTUM's accepted simulation results, labelled by model version, protocol and seed count: oracle contact-state representation, realistic sensing, channel masking, three-seed touch comparison, friction correction and physics-faithfulness checks. Code, v2 hand model, evaluation reports, figures, provenance and related-work/data links: https://github.com/spheroid-ai/dextum . Research snapshot: https://github.com/spheroid-ai/dextum/releases/tag/v0.1.1 . Company: https://spheroid.ai . Simulation video: https://spheroid-ai.github.io/dextum/#simulation (historical development replay, explicitly labelled). Paper, policy-checkpoint distribution and complete benchmark banks remain pending. No real-robot result is claimed.

## Architecture wording for the sensorless component

Sensorless deployment is a design goal. In the current simulation experiments,
distillation reached 70.5% success and inferred touch 41.5%, against the tactile
teacher's 88.0%; neither recovered the touch advantage at the tested budget.
These are measured negative results, not a demonstrated replacement for the
tactile skin.

The deployment candidate therefore remains experimental. The separately named
contact-mode physical model and the Metriplector retain their existing scope;
these conventional policy-transfer results establish no Metriplector result.
