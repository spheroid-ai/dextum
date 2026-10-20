# DEXTUM research page — v1

Buildless HTML research page, updated 20 October 2026 from accepted records through 18 October 2026.
`dist/index.html` works offline with its adjacent files or on any static host.
The public GitHub Pages edition is https://spheroid-ai.github.io/dextum/.
This research snapshot is not a completed paper or turnkey training release.

## Contents

- Actual Aero bench photograph, Kit B assembly render and the two supplied Spheroid concept renders.
- Historical 63-second Aero simulation MP4 and saved Kit B PNG, with explicit run context and media hashes.
- Five downloadable result figures (SVG/PDF), made from reported point estimates.
- Model-v2 rotation, channel ablations, O1, D2 and D3 results; v1 is explicitly historical.
- Protocol, sensor assumptions, limitations, programme status and evidence index.
- Reported values and provenance in `dist/data/`.

The numerical “no CPU run” amendment remains intact: no simulator, learner,
evaluation or statistical reanalysis ran for this page. Matplotlib draws already
reported values solely for presentation. No generated imagery is evidence.

Run `python3 build_assets.py` to redraw figures from the explicitly admitted frozen scorecards in `data/`; their hashes and field pointers are validated before drawing.
Run `python3 build_media.py` to copy the original media without altering it.
Serve `dist` using a local static server, or open `dist/index.html` directly.
`qa/` holds local-only source audit and visual QA, excluded from Site source.

Mapping audit: the page separates the contact-mode physical reference and
physical interfaces from Metriplector latent coordinates. It reports no measured
Metriplector implementation result and makes no new latent-mechanics design.

Visual structure is inspired by https://uni-vta.github.io/; all page code is new.
Project media are reused without image alteration and carry source hashes.

The research page leads with DEXTUM’s contact-state, quantum and task-flow
concepts, followed by the measured tactile study. Internal review drafts and
work orders are preserved in `reference-docs/`, outside the served `dist/`.
The downloadable offline edition includes only served site files.
The page does not claim the planned freeze or public research release is complete.
