"""Gallery block of the spine atlas: the HTML section and the page script (``assets/spine_gallery.js``).

Used through ``atlas_page`` by ``scripts/neuron_model/build_spine_atlas.py`` (new atlases) and ``scripts/neuron_model/patch_atlas_html.py``
(re-renders an atlas that is already built: the gallery needs only data that every atlas contains).
"""

from pathlib import Path

from . import html_kit as hk

GALLERY_JS_PATH = Path(__file__).resolve().parent / "assets" / "spine_gallery.js"
GALLERY_CSS = (
    "<style>.gal-label{display:flex;gap:10px;align-items:center;margin:16px 0 6px}"
    ".gal-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}"
    ".gal-fig{margin:0;border:1px solid var(--border,#e6e6ea);border-radius:10px;overflow:hidden;background:#fff;text-align:center}"
    ".gal-cell{width:100%;height:auto;display:block;cursor:pointer}.gal-fig figcaption{padding:2px 0 6px}</style>"
)


def gallery_js() -> str:
    return GALLERY_JS_PATH.read_text(encoding="utf-8")


def gallery_section() -> str:
    controls = (
        '<div class="row" style="margin-bottom:6px;gap:18px;align-items:center">'
        '<label>Поворот вокруг оси шипика <input type="range" id="galAngle" min="0" max="345" step="15" value="30" style="max-width:220px"> <span id="galAngleText" class="mono"></span></label>'
        '<button id="galNew">Другая выборка</button></div>'
    )
    note = hk.note([
        "В каждой строке шипики одного источника. Сверху реальные, ниже сгенерированные каждой моделью. В одном столбце стоят "
        "близкие по латентному коду шипики. Сначала выбираются пять реальных шипиков из разных областей карты, а потом для каждого "
        "из них в остальных строках берётся ближайший валидный сгенерированный шипик. Так можно сразу сравнить, как выглядят "
        "похожие реальные и сгенерированные формы. Кнопка «Другая выборка» берёт новые реальные шипики.",
        "Масштаб одинаков во всех ячейках, а основания выровнены по одной линии, поэтому размеры можно сравнивать на глаз. "
        "Шипики нарисованы упрощённо, без текстуры, и их можно поворачивать вокруг оси. Нажатие на ячейку открывает шипик в блоке "
        "«Выбранный шипик» с вращением мышью и таблицей признаков.",
        "Близость определяется по двумерной карте латентного пространства, а она показывает соседство, а не величину различий. "
        "Рядом с реальным шипиком не всегда есть по-настоящему похожий сгенерированный. Тогда в столбце окажется самый близкий из "
        "имеющихся.",
    ], title="Как читать галерею", collapsible=True)
    return GALLERY_CSS + hk.section(
        "Галерея шипиков", "Реальные и сгенерированные шипики рядом: быстрое визуальное сравнение форм.",
        controls + '<div id="gallery"></div>' + note, anchor="sec-gallery",
    )
