# DEXTUM research page — v1

Buildless HTML research page, based on accepted records as of 18 October 2026.
`dist/index.html` works offline with its adjacent files or on any static host.
The Sites deployment is owner-private. This preview is not the promised final
public research release or a published paper.

## Contents

- Actual Aero bench photograph, Kit B assembly render and the two supplied Spheroid concept renders.
- Historical 63-second Aero simulation MP4 and saved Kit B PNG, with explicit run context and media hashes.
- Five downloadable result figures (SVG/PDF), made from reported point estimates.
- Model-v2 rotation, channel ablations, O1 and D2 results; v1 is explicitly historical.
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

The section25 package and GPU-only work order are in `dist/docs/`. Freeze is19October18:00CEST. Full publication preparation is in the sibling `dextum_release_20oct/` repository. Public launch remains20October; this Site stays owner-private.
