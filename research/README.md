# Research source snapshot

This is a source and model release, not a claim that a new portable GPU
installation has reproduced the results.

## Source groups

- `core/dextum_cursor_agent/`: frozen common source from the 18 October
  boundary-monitor release. Historical modules remain because registered
  source hashes and imports depend on them. They are not alternate accepted
  models; use the registered v2 configuration.
- `oracle/oracle_dextum/`: frozen O1 adapter, contact mapping, input contract,
  structured/flat networks, qualification and learner.
- `sensorless/workorder14/lane_d/`: separately labelled 20 October workspace
  snapshot for D2/D3. It is not asserted to be the exact launch payload of
  every historical student run.
- `models/v2/`: accepted XML, path-only portable copy, referenced assets and
  original sensor geometry.
- `configs/`: unmodified historical registrations. Absolute paths, source
  hashes, case identities and qualifications are evidence, not defaults to
  silently replace.
- `SOURCE_MANIFEST.json`: per-file SHA-256 and source group.

## Code map

| Component | Entry |
|---|---|
| Oracle state | [schema.py](oracle/oracle_dextum/schema.py) |
| Contact observations | [observations_mjx.py](oracle/oracle_dextum/observations_mjx.py) |
| Structured / flat encoders | [networks.py](oracle/oracle_dextum/networks.py) |
| O1 environment | [environment.py](oracle/oracle_dextum/environment.py) |
| Guarded trainer | [train_rotation.py](oracle/oracle_dextum/train_rotation.py) |
| Contact-space action | [contact_space.py](core/dextum_cursor_agent/dextum/contact_space.py) |
| Sensor model | [sensors.py](core/dextum_cursor_agent/pass1c/sensors.py) |
| Direct evaluation | [evaluate.py](core/dextum_cursor_agent/pass1c/evaluate.py) |
| Monitor guard | [compare.py](core/dextum_cursor_agent/pass1c/compare.py) |
| Full-state PPO resume | [resumable_ppo.py](core/dextum_cursor_agent/pass1c/resumable_ppo.py) |
| Faithful collection | [faithful.py](core/dextum_cursor_agent/pass1/faithful.py) |

## Runtime contract

Recorded O1 versions include JAX/JAXlib 0.6.2, MuJoCo/MJX 3.3.6, Brax 0.12.5,
Flax 0.10.6 and Optax 0.2.8. See the historical
[GPU package list](core/dextum_cursor_agent/vendor/aero_official/requirements-gpu.lock.txt)
and registration for the complete scientific configuration.

| Upstream dependency | Recorded revision |
|---|---|
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | `1bd244fa9c51fa7074c7a449934f556d898f7a15` |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | `1b86ece576591213e2b666ebf59508454200ca97` |
| [Aero Open simulation](https://github.com/TetherIA/aero-open-sim) | `dfeff43507fe82f515fed81592070e39a9bb92f6` |

Historical code expects a qualified GPU container, specific import paths,
model/geometry locations and registration-bound outputs. Original orchestration
modules are retained for source identity; some manage cloud hosts and are not
generic setup scripts.

A new training deployment must bind its paths and source hashes, then pass the
collection-path detector, fresh-process resume, checkpoint-inclusive throughput
and paired monitor checks. No guard is waived by publication.

The original XML's 0.25 ms file timestep is replaced with the registered 5 ms
integration step by `pass1c/kit_model.py`. Loading the XML alone does not
reproduce the training environment.

## Verification

From the repository root:

```bash
python3 tools/verify_release.py
```

This checks hashes, Python syntax without imports, asset references, path-only
XML equivalence, admitted report bindings and page links. It does not compile
physics, import a trainer, run evaluation or reproduce learning.

Policy checkpoints, complete original evaluation banks and an end-to-end
portable training harness remain pending. Fixed-start construction and
evaluation code are available; they do not constitute the complete bank release.
