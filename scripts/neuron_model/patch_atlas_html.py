"""Re-render an ALREADY BUILT atlas (spine_atlas.html) with the current page texts and scripts, without rebuilding it
(a build re-encodes ~2000 spines, mostly NAS reads: about an hour).

Everything the page shows lives in the payload embedded in the file, so the page is simply generated again from it. This
brings an old atlas up to date: the fixed 3-D viewers (the mesh used to vanish on the second selection), the spine gallery,
the current explanation texts and the page title.

    python scripts/neuron_model/patch_atlas_html.py runs/analysis/neuron-model/atlas/<run>/spine_atlas.html [--title "..."]

The original is kept next to the result as ``spine_atlas.html.bak`` (unless ``--out`` is given).
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.reports.atlas_page import DEFAULT_TITLE, render_page  # noqa: E402

PAYLOAD_RE = re.compile(r'<script type="application/json" id="payload">(.*?)</script>', re.S)
# footer sentences written by older builds
FOOTER_FIXES = [("нейронов; для оценки", "нейронов. Для оценки"), ("треугольников; все числа", "треугольников, а все числа")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=Path)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--out", type=Path, help="default: overwrite (a .bak copy is kept next to it)")
    args = parser.parse_args()
    text = args.html.read_text(encoding="utf-8")
    found = PAYLOAD_RE.search(text)
    if not found:
        raise SystemExit("no embedded payload found (not an atlas built by build_spine_atlas.py?)")
    payload = json.loads(found.group(1))
    meta = payload["meta"]
    meta["title"] = args.title
    footer = meta["footer"]
    for old, new in FOOTER_FIXES:
        footer = footer.replace(old, new)
    # the list of generated sets was joined with semicolons: "A — 200 образцов (...); B — ..."
    meta["footer"] = re.sub(r"\)\; (?=\S+ — \d+ образцов)", "), ", footer)
    html = render_page(args.title, meta, payload["sources"], payload["summary"], payload)
    out = args.out or args.html
    if out == args.html:
        args.html.with_suffix(".html.bak").write_text(text, encoding="utf-8")
    out.write_text(html, encoding="utf-8")
    print(f"re-rendered -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
