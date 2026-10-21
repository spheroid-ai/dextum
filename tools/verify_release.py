"""Verify the published snapshot without importing scientific runtimes."""
from pathlib import Path
from html.parser import HTMLParser
import ast
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT / "research/SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        assert path.is_relative_to(ROOT), item["path"]
        assert path.is_file(), item["path"]
        assert digest(path) == item["sha256"], item["path"]
    sources = list((ROOT / "research").rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))

    model_dir = ROOT / "research/models/v2"
    original = model_dir / "kit_b_friction_matched.original.xml"
    assert digest(original) == "a2e0c670506cb69267bac11e80fa24c467a782cca3bedb901290adc64444c924"
    first = ET.parse(original).getroot()
    second = ET.parse(model_dir / "kit_b_friction_matched.xml").getroot()
    assets = 0
    for before, after in zip(first.iter(), second.iter(), strict=True):
        if "file" in before.attrib:
            assert (model_dir / after.attrib["file"]).is_file(), after.attrib["file"]
            before.attrib["file"] = "assets/" + Path(before.attrib["file"]).name
            assets += 1
        assert before.tag == after.tag and before.attrib == after.attrib
        assert (before.text or "").strip() == (after.text or "").strip()

    sys.path.insert(0, str(ROOT / "site"))
    from read_scorecards import load_results
    admitted = load_results()
    displayed = json.loads((ROOT / "site/dist/data/results.json").read_text())
    assert admitted["results"] == displayed["results"], "Displayed results differ from admitted reports"

    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths, self.anchors, self.ids = [], [], set()

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if "id" in values:
                assert values["id"] not in self.ids, "Duplicate page ID: " + values["id"]
                self.ids.add(values["id"])
            for field in ("href", "src", "poster"):
                value = values.get(field, "")
                if value.startswith("#"):
                    self.anchors.append(value[1:])
                elif value and not value.startswith(("http:", "https:", "data:", "mailto:")):
                    self.paths.append(value)

    pages = {}
    dist = ROOT / "site/dist"
    for filename in ("index.html", "full-record.html"):
        page = Page()
        page.feed((dist / filename).read_text())
        pages[filename] = page
    for filename, page in pages.items():
        for target in page.paths:
            url = urlsplit(target)
            path = unquote(url.path)
            assert (dist / path).is_file(), (filename, target)
            if url.fragment and path in pages:
                assert url.fragment in pages[path].ids, (filename, target)
        assert not set(page.anchors) - page.ids, (filename, "Unresolved page anchor")
    print(json.dumps({
        "state": "PASS",
        "hash_bound_files": len(manifest["files"]),
        "python_sources_parsed": len(sources),
        "model_asset_references": assets,
        "admitted_result_rows": len(admitted["results"]),
        "html_pages_checked": len(pages),
        "local_page_links": sum(len(p.paths) for p in pages.values()),
        "scientific_runtime_executed": False,
    }, indent=2))


if __name__ == "__main__":
    verify()
