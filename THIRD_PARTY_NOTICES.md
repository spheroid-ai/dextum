# Attribution and source boundaries

- **TetherIA Aero Hand Open:** base mechanics and Aero meshes from
  [aero-open-sim](https://github.com/TetherIA/aero-open-sim) and MuJoCo Menagerie's
  Aero entry. Apache-2.0; the model directory includes its licence. DEXTUM adds
  kit geometry and modifies contact/sensor parameters.
- **Brax:** `resumable_ppo.py` derives from Apache-2.0 Brax PPO. Its copyright
  notice is retained. DEXTUM adds full-state checkpoints/resume, boundary-monitor
  handling and instrumentation.
- **MuJoCo Playground:** the training interfaces use the recorded upstream
  dependency. This release does not relicense unrelated Menagerie models or
  Playground assets.
- **`vendor/aero_official`:** frozen workspace Aero control interfaces, with
  import adaptations and hashes recorded in `vendor/MANIFEST.json`. The name
  does not imply TetherIA endorsement.
- **Project media:** Aero bench photography, kit render and the two supplied
  concept illustrations are attributed to DEXTUM/Spheroid. They are excluded
  from the blanket code/data licence and do not show a manufactured tactile kit.
- **Simulation media:** the October 4 video and Kit B frame are historical
  development artifacts. Run context and hashes are in
  `site/dist/data/media-provenance.json` and `simulation-media.json`; they are
  not recordings of the accepted v2 policy.
- **UVTA:** presentation reference; no UVTA images, code or data are bundled.
- **Eidon and research links:** external references only, with their respective
  licences. Linked datasets are not bundled or claimed as DEXTUM inputs.

Apache-2.0 text is in `LICENSE`; original data/figure terms are in
`LICENSE-DATA.md`. Existing upstream source notices remain intact.
