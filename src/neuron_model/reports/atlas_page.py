"""Page of the spine atlas: the HTML body (all prose), the summary-table script and the full-page render.

``render_page`` needs only the payload, so a finished atlas can be re-rendered with the current texts and scripts without
re-encoding the spines (``scripts/neuron_model/patch_atlas_html.py``).
"""

from pathlib import Path
from typing import Any, Mapping

from . import html_kit as hk
from .atlas_gallery import gallery_js, gallery_section

ASSET = Path(__file__).resolve().parent / "assets" / "spine_atlas.js"
DEFAULT_TITLE = "Сравнение моделей генерации дендритных шипиков"


def body_html(meta: dict, sources: list, summary_rows: list) -> str:
    gen = [s for s in sources if s["kind"] == "generated"]
    real = next(s for s in sources if s["kind"] == "real")
    tiles = [{"label": "Реальных шипиков в эталоне", "value": f"{real['n']:,}".replace(",", " "), "note": f"сплит {meta['split']}, {meta['n_neurons']} нейронов"}]
    for s in gen:
        r = next((x for x in summary_rows if x["id"] == s["id"]), {})
        v = r.get("valid")
        tiles.append({"label": f"{s['name']}: валидных сеток", "value": "—" if v is None else f"{100 * v:.0f} %",
                      "note": f"среднее W₁/σ по признакам {r['w1']:.2f}" if r.get("w1") is not None else "", "accent": "pink" if s is gen[0] else "green"})
    head = hk.header(meta["title"], meta["subtitle"], meta["meta_lines"])
    summary = '<div class="card" style="margin-top:14px"><div class="card-body"><table class="metrics" id="summaryTable"></table></div></div>'
    intro = hk.note([
        "В атласе вместе показаны реальные шипики и шипики, которые создали модели. Их можно сравнить по положению в общем "
        "пространстве форм, по распределениям признаков и по качеству поверхности. Все модели сравниваются с одними и теми же "
        "реальными шипиками, которых не было в обучении.",
        "Общее пространство форм задаёт энкодер обученного VAE. Любой шипик, реальный или сгенерированный, превращается в облако "
        "точек на поверхности и проходит через этот энкодер. У MoGen своего энкодера нет, так что другого общего пространства для "
        "всех моделей не существует. Двумерная карта строится только по реальным шипикам, а сгенерированные просто ложатся на неё. "
        "Поэтому карта реальных данных не меняется, когда добавляется новая модель.",
        "**Как читать сводку.** W₁/σ показывает, насколько распределение признака у модели отличается от реального, в долях "
        "разброса реальных значений. Меньше 0,2 значит, что распределения близки, больше 0,5 значит, что расхождение заметное. "
        "Покрытие — это доля реальных форм, у которых нашёлся хотя бы один похожий сгенерированный шипик. 1-NNA около 0,5 значит, "
        "что по ближайшему соседу сгенерированные и реальные шипики не отличить, а около 1 значит, что они легко разделяются. "
        "Если близость к обучающим шипикам заметно меньше единицы, модель, скорее всего, копирует обучающую выборку.",
    ], title="О чём этот атлас")
    workspace = f"""
<div class="workspace">
 <aside class="sidebar">
  <div class="panel"><h3>Пространство</h3><select id="mapSelect" aria-label="Пространство карты"></select>
   <label class="field" for="colorSelect">Цвет точек</label><select id="colorSelect"></select></div>
  <div class="panel"><h3>Источники</h3><div id="sources"></div>
   <label class="check"><input type="checkbox" id="onlyValid"> только валидные сгенерированные сетки</label></div>
  <div class="panel"><h3>Реальные шипики</h3>
   <label class="field" for="cellTypeSelect">Тип клетки</label><select id="cellTypeSelect"></select>
   <label class="field" for="compartmentSelect">Компартмент</label><select id="compartmentSelect"></select></div>
 </aside>
 <div>
  <div class="grid-2">
   {hk.card("Карта форм", '<div id="map" class="plot l"></div><div id="mapLegend" class="legend"></div><div id="mapCounts" class="small muted"></div>', hint="колесо — масштаб, двойной щелчок — сброс")}
   {hk.card("Выбранный шипик", '<div class="row" style="margin-bottom:6px"><button id="pin">Закрепить для сравнения</button><button id="unpin">Открепить</button></div><div id="viewer" class="plot m"></div><div id="viewerNote" class="small muted"></div><div id="keyLine" class="mono muted" style="margin:6px 0"></div><table class="metrics" id="metricTable"></table>', hint="вращение мышью")}
  </div>
 </div>
</div>"""
    map_note = hk.note([
        "Каждая точка — один шипик. Рядом лежат шипики, которые энкодер считает похожими. Расстояния на проекции UMAP показывают "
        "только соседство, а не величину различий, поэтому сравнивать длины отрезков на карте нельзя. Вторая карта построена по "
        "стандартизованным морфометрическим признакам методом главных компонент. На ней оси и расстояния имеют смысл, но различия "
        "формы, которые признаки не отражают, не видны.",
        "**Хорошая модель** занимает те же области карты, что и реальные шипики, и с такой же плотностью. Сгенерированные точки вне "
        "облака реальных означают формы, которых в данных нет. Пустые места внутри облака реальных точек означают формы, которые "
        "модель не воспроизводит. Если раскрасить карту по признаку, и граница цвета у реальных и сгенерированных точек идёт "
        "одинаково, то модель связывает форму и размер так же, как в данных.",
        "Таблица под сеткой показывает значение каждого признака и его перцентиль среди реальных шипиков. Перцентиль ниже 5 или "
        "выше 95 выделен красным, потому что у реальных шипиков такие значения встречаются редко. Трёхмерные сетки встроены не для "
        "всех точек. Для точки без сетки показывается ближайший на карте шипик того же источника, и это отмечено звёздочкой.",
    ], title="Как читать карту", collapsible=True)
    morph = f"""
<div class="grid-2">
 {hk.card("Распределение признака", '<select id="metricSelect" aria-label="Признак" style="max-width:340px"></select><div id="dist" class="plot m"></div><div id="distAbout" class="small muted"></div>')}
 {hk.card("Корреляции признаков", '<select id="corrModel" aria-label="Модель" style="max-width:340px"></select><div id="corr" class="plot m"></div>')}
</div>
{hk.card("Сравнение распределений по всем признакам", '<div style="overflow:auto"><table class="metrics" id="compareTable"></table></div>')}"""
    morph_note = hk.note([
        "Признаки сгенерированных шипиков считаются тем же кодом, что и у реальных на этапе предобработки. Основание "
        "сгенерированного шипика неизвестно, поэтому оно ищется по геометрии, как и при оценке моделей. Признаки, которые зависят "
        "от основания (длина, площадь основания, угол раскрытия и другие), из-за этого содержат небольшую систематическую ошибку. "
        "Её не нужно приписывать генератору.",
        "Справа показаны связи между признаками. Слева корреляции у реальных шипиков, справа их разность у выбранной модели. "
        "Модель может хорошо воспроизводить каждый признак по отдельности и всё же нарушать связи между ними. Например, она может "
        "давать длинные шипики с объёмом коротких. Такие ошибки видны только на карте разностей.",
    ], title="Как читать морфометрику", collapsible=True)
    quality = f"""
<div class="grid-2e">
 {hk.card("Доли валидных сеток", '<div id="qualityRates" class="plot s"></div>')}
 {hk.card("Распределение показателя качества", '<select id="qualitySelect" aria-label="Показатель" style="max-width:340px"></select><div id="qualityBox" class="plot s"></div><div id="qualityAbout" class="small muted"></div>')}
</div>"""
    quality_note = hk.note([
        "Сетка считается валидной, если она замкнута, образует одну компоненту, имеет положительный объём, не содержит вырожденных "
        "граней и самопересечений. Для реальных шипиков доли посчитаны по встроенным в атлас запаянным сеткам. Для сгенерированных "
        "взят отчёт генератора о каждой сырой сетке.",
        "Шероховатость — это средний угол между нормалями соседних треугольников. У гладкой поверхности он мал, а рябь и ступеньки "
        "его увеличивают. Доля игольчатых треугольников показывает, подходит ли сетка для дальнейших расчётов. Такие треугольники "
        "портят оценку кривизны и площади.",
    ], title="Как читать качество поверхности", collapsible=True)
    trav = f"""
<div class="grid-2">
 {hk.card("Сетка на выбранном шаге", '<div class="row"><select id="travBase" aria-label="Базовый шипик" style="max-width:420px"></select><select id="travDir" aria-label="Направление" style="max-width:300px"></select></div><div class="row" style="margin:10px 0"><input type="range" id="travStep" min="0" max="4" step="1" style="max-width:320px"><span id="travStepText" class="mono"></span><button id="travPlay">Пройти</button></div><div id="travView" class="plot m"></div>')}
 {hk.card("Признак вдоль направления", '<select id="travMetric" aria-label="Признак" style="max-width:340px"></select><div id="travCurve" class="plot m"></div><div id="travAbout" class="small muted"></div>')}
</div>"""
    trav_note = hk.note([
        "Латентный код реального шипика сдвигается вдоль одного из главных направлений, по которым различаются коды реальных "
        "шипиков. По каждому сдвинутому коду декодер строит поверхность. Сдвиг измеряется в стандартных отклонениях вдоль этого "
        "направления, поэтому сдвиг на ±2σ ещё остаётся в пределах обычных для данных значений.",
        "Справа показано, как выбранный признак меняется вдоль пути, на фоне диапазона от 10-го до 90-го перцентиля реальных "
        "шипиков. Плавная монотонная кривая значит, что направление кодирует понятное свойство формы. Резкие скачки или выход за "
        "реальный диапазон при малых сдвигах указывают на области, где модель работает ненадёжно.",
    ], title="Как читать латентные направления", collapsible=True)
    return (head + hk.tiles(tiles) + summary + intro
            + hk.section("Карта форм", "Реальные и сгенерированные шипики в общем пространстве. Щелчок по точке открывает её сетку.", workspace + map_note, anchor="sec-map")
            + hk.section("Морфометрические признаки", "Насколько распределения признаков и связи между ними у моделей совпадают с реальными.", morph + morph_note, anchor="sec-morph")
            + gallery_section()
            + hk.section("Качество поверхности", "Пригодность сеток для анализа: замкнутость, связность, гладкость.", quality + quality_note, anchor="sec-quality")
            + hk.section("Латентные направления", "Как меняется форма шипика при движении по главным направлениям латентного пространства VAE.", trav + trav_note, anchor="sec-traversal")
            + f'<footer class="page">{hk.esc(meta["footer"])}</footer>')


SUMMARY_JS = r"""
(function () {
  const rows = PAYLOAD.summary, t = $('summaryTable');
  if (!rows.length) { t.innerHTML = '<tr><td>Сгенерированных наборов нет.</td></tr>'; return; }
  const pill = (v, good, bad, invert) => v === null || v === undefined ? '—' :
    `<span class="pill ${(invert ? v > good : v < good) ? 'good' : (invert ? v > bad : v < bad) ? 'warn' : 'bad'}">${fmt(v, 2)}</span>`;
  let h = '<tr><th>Модель</th><th>Валидных сеток</th><th>Среднее W₁/σ</th><th>Худший признак</th><th>Макс. |ΔR|</th><th>Покрытие</th><th>1-NNA</th><th>Близость к обучающим</th></tr>';
  for (const r of rows) {
    h += `<tr><td><span class="swatch" style="background:${r.color};margin-right:8px"></span>${r.name}</td><td>${r.valid === null ? '—' : (100 * r.valid).toFixed(0) + ' %'}</td>`
      + `<td>${pill(r.w1, 0.2, 0.5)}</td><td>${r.worst || '—'}</td><td>${pill(r.max_gap, 0.15, 0.3)}</td><td>${pill(r.coverage, 0.5, 0.3, true)}</td>`
      + `<td>${pill(r.one_nna === null ? null : Math.abs(r.one_nna - 0.5) + 0.5, 0.6, 0.8)}</td><td>${pill(r.memorization_ratio, 2, 0.8, true)}</td></tr>`;
  }
  t.innerHTML = h;
})();
"""


def render_page(title: str, meta: dict, sources: list, summary: list, payload: Mapping[str, Any]) -> str:
    page_js = ASSET.read_text(encoding="utf-8") + "\n" + SUMMARY_JS + "\n" + gallery_js()
    return hk.page(title, body_html(meta, sources, summary), payload, page_js)
