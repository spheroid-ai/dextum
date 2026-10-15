# DEXTUM — 20 October publication preparation

Private preparation repository: https://github.com/spheroid-ai/dextum (15 October 2026). Public release target: 20 October.
Results freeze: **19 October, 18:00 CEST / 16:00 UTC**.

Open [the HTML page](site/dist/index.html). The supplied concept renders, real
Aero photograph and historical simulation video/PNG are included with labels.
This package contains the page, five figures, reported data, frozen source
reports, the revised package/work order and the release inventory. It is **not**
the complete training release, a finished paper or a real-robot result.

## Build

`python3 site/build_assets.py` verifies the hashes in
`site/data/scorecard_bindings.json`, reads explicitly admitted report fields and
redraws the figures. It performs no simulation, evaluation, statistical fitting
or significance test. Requires Python 3 and Matplotlib.

`python3 package.py` builds the offline HTML ZIP, preparation ZIP and file-hash
manifest. `python3 -m http.server --directory site/dist` previews the page.

To add an accepted scorecard, copy its immutable report into `site/data/evidence`,
add its exact hash and field pointers to the bindings, and update the admitted
row/claim metadata. Do not point at a live “latest” checkpoint. Rebuild and audit
the accompanying prose before accepting the figure. Missing exact bindings stay
listed in the inventory and cannot silently pass the final freeze.

## Status and next publication steps

The accepted minimum is v2 reorientation, friction attribution, sensing/channel
contrasts, preliminary O1 and the negative D2 result. The full three-seed O1
verdict, C7, C9, E22–E24 and hold are pending. See
[the package](docs/dextum-results-package-20oct.md) and
[release inventory](RELEASE_INVENTORY.json).

The original hosted preview remains owner-private. This isolated publication
edition can be prepared even when its source hosting service is unavailable.
Synchronize it into the existing Site through the Sites workflow after service
recovery; never create a replacement public Site to bypass a source failure.

Original code/model terms are planned as Apache-2.0 and original banks/assets as
CC-BY-4.0. Third-party geometry and media retain their own rights. The final
licence/redistribution inventory is pending; these files do not relicense them.

## Mapping audit

Design: physical state and joules belong to the separately named contact-mode
physical reference and interfaces. Anonymous Metriplector processor coordinates
have no physical identities. Implementation: no Metriplector result or compliance
is inferred from these PPO and simulator records.
