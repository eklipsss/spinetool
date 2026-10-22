"""Gallery block of the spine atlas: the HTML section and the page script (``assets/spine_gallery.js``).

Used by ``scripts/neuron_model/build_spine_atlas.py`` (new atlases) and ``scripts/neuron_model/patch_atlas_html.py --gallery``
(adds the block to an atlas that is already built: the gallery needs only data that every atlas contains).
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
        '<label>Шипиков в строке <select id="galCount" aria-label="Число шипиков"><option>5</option><option>6</option><option>7</option></select></label>'
        '<label>Поворот вокруг оси шипика <input type="range" id="galAngle" min="0" max="345" step="15" value="30" style="max-width:220px"> <span id="galAngleText" class="mono"></span></label>'
        '<button id="galNew">Другая выборка</button></div>'
    )
    note = hk.note([
        "В каждой строке — шипики одного источника: реальные, затем сгенерированные каждой моделью. Показаны только валидные сетки; "
        "из случайного пула отбираются те, что лежат дальше всего друг от друга на карте латентного пространства, чтобы строка "
        "показывала разные формы, а не пять похожих. Кнопка «Другая выборка» берёт новый случайный пул.",
        "Масштаб одинаков во всех ячейках, а основания шипиков выровнены по одной линии: размеры и длины можно сравнивать "
        "между строками на глаз. Шипики нарисованы упрощённо (плоская закраска, без текстуры), с поворотом вокруг оси шипика; "
        "нажатие на ячейку открывает этот шипик в блоке «Выбранный шипик» с вращением мышью и таблицей признаков.",
        "Расстояния для отбора берутся на двумерной карте латентного пространства (UMAP), которая передаёт соседство, а не величину различий, "
        "поэтому «максимально далёкие» означает «из разных областей карты».",
    ], title="Как читать галерею", collapsible=True)
    return GALLERY_CSS + hk.section(
        "Галерея шипиков", "Реальные и сгенерированные шипики рядом: быстрое визуальное сравнение форм.",
        controls + '<div id="gallery"></div>' + note, anchor="sec-gallery",
    )
