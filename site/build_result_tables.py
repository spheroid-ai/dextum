"""Render visible result tables from the same hash-bound reports as the figures."""
from html import escape
from pathlib import Path

from read_scorecards import load_results


def render_tables():
    rows = load_results()["results"]

    def result(group, arm, seed=1):
        return next(r for r in rows if (r["group"], r["arm"], r["seed"]) == (group, arm, seed))

    def table(caption, headings, body):
        head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headings)
        content = []
        for run_id, cells in body:
            values = "".join(f"<td>{escape(v)}</td>" for v in cells)
            content.append(f'<tr data-run-id="{escape(run_id, quote=True)}">{values}</tr>')
        return (
            '<div class="table-wrap result-table"><table>'
            f"<caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead>"
            f'<tbody>{"".join(content)}</tbody></table></div>'
        )

    seeds = []
    for seed in (1, 2, 3):
        touch = result("touch_recheck", "c_touch", seed)
        control = result("touch_recheck", "c_notouch", seed)
        gain = touch["success_percent"] - control["success_percent"]
        seeds.append((touch["run_id"] + "; " + control["run_id"], [
            str(seed), f'{touch["success_percent"]:.1f}%', f'{control["success_percent"]:.1f}%',
            f"+{gain:.1f}", f'{touch["paired_p"]:.3f}',
        ]))
    channels = []
    for arm, label in (("none", "Unmasked"), ("normal_only", "Normal-only"), ("palm", "Palm map removed")):
        r = result("channel_ablation", arm)
        channels.append((r["run_id"], [
            label, f'{r["success_percent"]:.1f}%',
            "Reference" if r["paired_p"] is None else f'{r["paired_p"]:.3f}',
        ]))
    return {
        "seeds": table(
            "Model v2 · Realistic sensing · three seeds · matched 20.3M warm starts · 200 shared starts",
            ["Seed", "C-touch", "C-no-touch", "Gain (points)", "Paired p"], seeds),
        "channels": table(
            "Model v2 · Realistic sensing · one seed · equal-budget 20.3M continuations · 200 shared starts",
            ["Input", "Success", "Paired p vs unmasked"], channels),
    }


def update_page():
    for filename in ("index.html", "full-record.html"):
        path = Path(__file__).parent / "dist" / filename
        page = path.read_text()
        for name, content in render_tables().items():
            start, end = f"<!-- {name}-table:start -->", f"<!-- {name}-table:end -->"
            if start not in page:
                continue
            before, rest = page.split(start, 1)
            _, after = rest.split(end, 1)
            page = before + start + "\n" + content + "\n" + end + after
        path.write_text(page)


if __name__ == "__main__":
    update_page()
