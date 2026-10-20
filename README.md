# DEXTUM

### Manipulation, described through contact

**Spheroid · Peter Toth, Dan Oprisa, Sam Padilla**

[Project page](https://spheroid-ai.github.io/dextum/) · [Research code](research/README.md) · [Model and sensors](research/models/v2/README.md) · [Scorecards](site/data/evidence/) · [Simulation video](https://spheroid-ai.github.io/dextum/#simulation) · [Release download](https://github.com/spheroid-ai/dextum/releases/tag/v0.1.1) · [Company](https://spheroid.ai)

*Design render; hardware validation is pending. All reported performance below is from simulation.*

DEXTUM studies manipulation as contact states between hand elements and object patches, transitions between those states, and flows through feasible transitions. The programme combines a contact representation, a physical reference for contact dynamics, and a tactile kit with a common data format.

The first study, **“Which Touch Does a Hand Need?”**, puts a physically specified kit on the TetherIA Aero Hand Open in MuJoCo/MJX. Matched experiments measure sensor cost, channel contributions and the effect of correcting the contact model. Oracle O1 supplies exact simulator-derived inputs to test the representation separately from learning perception.

**Public research snapshot v0.1.1 · 20 October 2026.** Accepted evidence through 18 October; scientific source, the v2 model/assets, registered configurations, evidence-bound figures and the page. The paper PDF, policy checkpoints and complete portable training release are **not yet published**. No real-robot success rate is claimed.

## Results

| Experiment | Accepted result | Scope |
|---|---|---|
| Oracle O1 structured | 89.0 / 90.0 / 88.5%; mean **89.2%** | Three seeds; one cube, one hand, ±30° yaw and hold |
| O1 privileged reference / flat | Reference 84.0%; flat 83.0 / 85.5% | One reference seed; two accepted flat seeds |
| O1 structure effect | +6 points at seed 1, paired p = .0227 | Below the registered +10-point bar; three-seed verdict incomplete |
| Pad-friction correction | 73.0 → **90.5%**, +17.5 points, p ≈ 3.3 × 10⁻⁶ | v1 → v2, Ideal sensing, from scratch, one seed |
| Ideal → Realistic sensing | 90.5 → **88.0%**, p ≈ .383 | v2, full budget, one seed; not significant |
| Touch versus no touch | Mean +6.17 points; 95% seed interval −6.09 to +18.42 | v2, three matched warm-start seeds; interval spans zero |
| Channel ablations | Unmasked 87.5%; normal-only 85.0%; palm map removed 87.5% | v2, one seed; neither contrast significant |
| Sensorless student, D2 | 70.5% success, 2.5% drops | Negative at 20.3M collected transitions |
| Inferred touch, D3 | 41.5% success, 10% drops | Negative at its registered budget |

O1 uses **2,048 worlds** and its own 108.1M-step protocol; its absolute levels are not compared with the 8,192-world tactile experiments. Full O1 inputs include pose and velocity; contact-only is a separate ablation. The registered proximity criterion is met, but superiority over the single ceiling seed is not established (seed-1 paired p = .0755). Composition, object transfer, hand transfer and touch necessity remain unestablished.

The historical **+11-point touch result belongs to model v1**, before the friction correction; it is not the current-model headline.

Inspect the [CSV](site/dist/data/results.csv), [structured results](site/dist/data/results.json), [number-to-report index](site/dist/data/freeze-index.json), [report copies](site/data/evidence/) and [hash/field bindings](site/data/scorecard_bindings.json).

## Code and artifacts

| Artifact | Location |
|---|---|
| Page, concept renders, bench photo, historical simulation media | [site/dist](site/dist/) |
| Figure builder and report reader | [build_assets.py](site/build_assets.py), [read_scorecards.py](site/read_scorecards.py) |
| Oracle inputs, structured/flat encoders, guarded training interfaces | [research/oracle/oracle_dextum](research/oracle/oracle_dextum/) |
| Contact-space action and state | [research/core/dextum_cursor_agent/dextum](research/core/dextum_cursor_agent/dextum/) |
| Sensors, paired evaluation, resumable PPO, monitor guard | [research/core/dextum_cursor_agent/pass1c](research/core/dextum_cursor_agent/pass1c/) |
| Training-path faithfulness | [research/core/dextum_cursor_agent/pass1](research/core/dextum_cursor_agent/pass1/) |
| DAgger and inferred-touch source | [research/sensorless/workorder14/lane_d](research/sensorless/workorder14/lane_d/) |
| Accepted v2 XML, path-only portable copy, assets and sensor geometry | [research/models/v2](research/models/v2/) |
| Historical O1 registrations | [research/configs](research/configs/) |
| Source hashes and provenance | [SOURCE_MANIFEST.json](research/SOURCE_MANIFEST.json) |
| Released and outstanding artifacts | [RELEASE_INVENTORY.json](RELEASE_INVENTORY.json) |

The frozen training source retains its original GPU checks, path bindings and historical orchestration. It is provided for inspection and controlled reproduction; it is not a newly validated one-command training package. Read the [research guide](research/README.md) before executing it. Publication did not start any simulation or training.

## Inspect and rebuild the figures

```bash
git clone https://github.com/spheroid-ai/dextum.git
cd dextum

# Standard-library checks: hashes, model assets, report bindings and links.
python3 tools/verify_release.py

# Redraw from admitted reports; no simulation or statistical fitting.
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-figures.txt
python3 site/build_assets.py

# Open http://localhost:8000
python3 -m http.server 8000 --directory site/dist
```

GitHub Pages serves the same `site/dist` tree from the `gh-pages` branch.
Publishing the page does not run a learner. After committing page changes to
`main`, maintainers update the static branch with:

```bash
git subtree split --prefix site/dist -b pages-update
git push origin pages-update:gh-pages
git branch -D pages-update
```

## Relevant links

### DEXTUM materials and demos

- [Project page](https://spheroid-ai.github.io/dextum/) · [Repository](https://github.com/spheroid-ai/dextum).
- [Oracle O1 evidence](https://spheroid-ai.github.io/dextum/#oracle) · [Tactile results](https://spheroid-ai.github.io/dextum/#results).
- [Simulation video and stills](https://spheroid-ai.github.io/dextum/#simulation), with captions and provenance.
- [Historical Aero MP4](https://spheroid-ai.github.io/dextum/assets/aero-simulation-20261004.mp4): predates model v2; not footage of the 88% policy.
- [Figures: SVG and PDF](https://spheroid-ai.github.io/dextum/assets/dextum-figures.zip) · [Offline HTML](https://spheroid-ai.github.io/dextum/dextum-v1.zip).
- [Code guide](research/README.md) · [Model](research/models/v2/README.md) · [Scorecards](site/data/evidence/).
- **DEXTUM paper PDF:** pending; no paper URL or DOI is claimed.

### Spheroid and prior research

- [Spheroid](https://spheroid.ai).
- [Metriplector: From Field Theory to Neural Architecture](https://arxiv.org/abs/2603.29496), Dan Oprisa and Peter Toth. Related architecture research, separate from the explicit contact-mode physical reference. These PPO experiments are not a Metriplector result.
- [Hamiltonian Generative Networks](https://arxiv.org/abs/1909.13789), Peter Toth et al., ICLR 2020.

### Eidon hardware and data

External Eidon artifacts, linked as prior hardware/data work. They retain their own licences and are not DEXTUM experimental results or claimed DEXTUM training inputs.

- [Eidon glove](https://github.com/Eidon-AI/eidon-glove).
- [Eidon wearable tracker](https://github.com/Eidon-AI/eidon-tracker).
- [Eidon simulation tools](https://github.com/Eidon-AI/eidon-sim).
- [Tracker + egocentric video](https://huggingface.co/datasets/eidon-ai/tracker-pov).
- [Tracker + video + IMU](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu).
- [Egocentric video collection](https://huggingface.co/buckets/eidon-ai/egocentric-pov).
- [Eidon project archive](https://www.eidon.ai).

### Platform and related tactile work

- [TetherIA Aero Open simulation](https://github.com/TetherIA/aero-open-sim).
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) · [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) · [Brax](https://github.com/google/brax).
- [UVTA](https://uni-vta.github.io/): related human-to-robot tactile research and the presentation reference for this page.

## Boundaries and roadmap

One hand and one successful task family; simulation only. Pad contact, sensing and sim-to-real performance still need bench validation. No accepted task-success result for sort, switch or hold is claimed. The v2 held-out-physics evaluation stopped at its repeatability gate. O1b sequences, O3′ geometry, O5′ hand transfer, energy accounting, the physical-model comparison and human-to-Aero token transfer remain unmeasured here. Source availability is not a claim of a validated cross-embodiment tokeniser or glove recorder.

## Licences and attribution

Original software and model modifications: [Apache-2.0](LICENSE). Original released numerical results, figures and kit mesh assets: [CC BY 4.0](LICENSE-DATA.md). Third-party code and Aero geometry retain their own notices. Concept renders and photographs are excluded from the blanket software/data licences. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Mapping audit

Design: named physical state and joules belong to the explicit contact-mode physical reference and interfaces. Metriplector processor coordinates have no assigned anatomical or mechanical identity. Implementation: this snapshot contains conventional PPO, simulation and representation experiments; it establishes no Metriplector implementation result or compliance claim.

See the [20 October publication audit](docs/PUBLICATION_CORRECTIONS_20261020.md) for the verified bare-hand arm labels and [application wording](docs/APPLICATION_ARTIFACTS.md) for the current paper and sensorless-transfer status.

## Citation

```bibtex
@misc{spheroid2026dextum,
  title = {DEXTUM: Manipulation, Described Through Contact},
  author = {Toth, Peter and Oprisa, Dan and Padilla, Sam},
  year = {2026},
  howpublished = {\url{https://spheroid-ai.github.io/dextum/}},
  note = {Research snapshot v0.1.1; simulation results; paper pending}
}
```
