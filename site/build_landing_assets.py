"""Small landing-page charts from admitted values; no new measurement or fit."""
from html import escape
from pathlib import Path

from read_scorecards import load_results

ROOT = Path(__file__).parent


def build():
    rows = load_results()["results"]

    def value(group, arm, seed=1, model=None):
        return next(r["success_percent"] for r in rows
                    if r["group"] == group and r["arm"] == arm and r["seed"] == seed
                    and (model is None or r["model"] == model))

    figures = {
        "landing-hand": ("Hand comparison · Ideal sensing · one seed", [
            ("Bare Aero", value("scratch_rotation", "Bare Aero Ideal"), "#8fa4a6"),
            ("Sensorised v2", value("scratch_rotation", "Sensorised Ideal", model="v2"), "#286f70"),
            ("Rigid pads · v2", value("scratch_rotation", "Sensorised rigid Ideal"), "#8fa4a6"),
        ]),
        "landing-sensing": ("Sensorised v2 · from scratch · one seed", [
            ("Ideal sensing", value("scratch_rotation", "Sensorised Ideal", model="v2"), "#8fa4a6"),
            ("Realistic sensing", value("scratch_rotation", "Sensorised Realistic"), "#286f70"),
        ]),
        "landing-channels": ("Sensorised v2 · matched continuations · one seed", [
            ("Unmasked", value("channel_ablation", "none"), "#286f70"),
            ("Normal-only", value("channel_ablation", "normal_only"), "#72a3a1"),
            ("Palm map removed", value("channel_ablation", "palm"), "#8fa4a6"),
        ]),
        "landing-oracle": ("Oracle O1 · its own 2,048-world protocol", [
            ("Full state · s1", value("oracle_o1", "privileged_ceiling"), "#8fa4a6"),
            *[(f"DEXTUM · s{i}", value("oracle_o1", "oracle_structured_full", i), "#286f70") for i in (1, 2, 3)],
        ]),
    }
    for name, (title, entries) in figures.items():
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 270" role="img">',
               f'<title>{escape(title)}</title>',
               '<rect width="600" height="270" rx="12" fill="#f5f8f7"/>',
               '<g font-family="Arial,sans-serif" fill="#26444a">',
               f'<text x="24" y="29" font-size="13">{escape(title)}</text>']
        for tick in (0, 25, 50, 75, 100):
            x = 175 + tick * 3.5
            svg.append(f'<path d="M{x} 49V222" stroke="#e2eae7"/>')
            svg.append(f'<text x="{x}" y="243" font-size="12" fill="#667c7d" text-anchor="middle">{tick}</text>')
        step = 170 / len(entries)
        for i, (label, success, color) in enumerate(entries):
            y = 56 + i * step + (step - 27) / 2
            svg.extend([
                f'<text x="160" y="{y+18:g}" font-size="14" text-anchor="end">{escape(label)}</text>',
                f'<rect x="175" y="{y:g}" width="{success*3.5:g}" height="27" rx="3" fill="{color}"/>',
                f'<text x="{185+success*3.5:g}" y="{y+19:g}" font-size="15" font-weight="700">{success:.1f}</text>',
            ])
        svg.extend(['<text x="350" y="263" font-size="11" text-anchor="middle" fill="#667c7d">Success on 200 shared starts (%)</text>', '</g></svg>'])
        (ROOT / "dist/assets" / (name + ".svg")).write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    build()
