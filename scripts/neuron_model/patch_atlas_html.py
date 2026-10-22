"""Fix the 3-D viewers of an ALREADY BUILT atlas (spine_atlas.html) without rebuilding it (the build re-encodes spines from the NAS, ~1 h).

The page cleared the viewer div (``innerHTML = ''``) before every ``Plotly.react``; Plotly then kept its old state, believed the
plot was still in the DOM and drew nothing, so the mesh vanished on the second selection / the second latent step. The source
(``reports/assets/spine_atlas.js``) is fixed; this script applies the same replacements to a finished file:

    python scripts/neuron_model/patch_atlas_html.py runs/analysis/neuron-model/atlas/<run>/spine_atlas.html

``--gallery`` also adds the "Галерея шипиков" block (rows: real / each model, valid meshes far apart on the latent map) to an
atlas built before the block existed: it only needs data every atlas contains (embedded meshes, map, validity flags).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPLACEMENTS = [
    ("  $('viewer').innerHTML = '';\n  const meshes",
     "  if (!$('viewer')._fullData) $('viewer').innerHTML = '';\n  const meshes"),
    ("""  if (f.mesh < 0) { $('travView').innerHTML = '<div class="empty">Для этого шага поверхность не извлечена</div>'; }
  else {
    $('travView').innerHTML = '';""",
     """  if (f.mesh < 0) { Plotly.purge('travView'); $('travView').innerHTML = '<div class="empty">Для этого шага поверхность не извлечена</div>'; }
  else {
    if (!$('travView')._fullData) $('travView').innerHTML = '';"""),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=Path)
    parser.add_argument("--gallery", action="store_true", help="add the spine gallery block (HTML section + script)")
    parser.add_argument("--out", type=Path, help="default: overwrite (a .bak copy is kept next to it)")
    args = parser.parse_args()
    text = args.html.read_text(encoding="utf-8")
    patched = text
    for old, new in REPLACEMENTS:
        if new in patched:
            print("already patched:", old.strip().splitlines()[0][:60])
        elif old in patched:
            patched = patched.replace(old, new)
            print("patched:", old.strip().splitlines()[0][:60])
        else:
            raise SystemExit(f"snippet not found (a different atlas version?): {old.strip().splitlines()[0][:80]}")
    if args.gallery:
        if 'id="gallery"' in patched:
            print("already has the gallery")
        else:
            from src.neuron_model.reports.atlas_gallery import gallery_js, gallery_section

            anchor = '<section class="block" id="sec-quality"'
            tail = "</script></body></html>"
            if anchor not in patched or tail not in patched:
                raise SystemExit("cannot place the gallery (a different atlas version?)")
            patched = patched.replace(anchor, gallery_section() + anchor, 1)
            cut = patched.rindex(tail)
            patched = patched[:cut] + "\n" + gallery_js() + "\n" + patched[cut:]
            print("added: gallery")
    out = args.out or args.html
    if out == args.html:
        args.html.with_suffix(".html.bak").write_text(text, encoding="utf-8")
    out.write_text(patched, encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
