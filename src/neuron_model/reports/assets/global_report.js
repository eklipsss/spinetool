// Global report (src/neuron_model/reports/global_report.py). Uses COMMON_JS: $, fmt, layout, axisTitle, PLOT_CONFIG, el.
const D = PAYLOAD;
const LEVELS = Object.keys(D.levels);
const LIMIT = 50;                                   // colour scale clipped at +-50 %
const GREY_NS = '#dcdce0', UNDER = '#faf0d4';
const DS_COLOR = {minnie65: P.series_2, h01: P.series_1, labid: P.series_3};
const dsName = d => D.dataset_ru[d] || d.replace(' vs ', ' и ');
const VAL_RU = {excitatory: 'возбуждающие', inhibitory: 'тормозные', unknown: 'нет данных', pyramidal: 'пирамидные', other_excitatory: 'другие возбуждающие',
  interneuron: 'интернейроны', apical: 'апикальный', basal: 'базальный', uncertain: 'неопределённый', healthy: 'здоровые', disease: 'болезнь',
  untreated: 'без лечения', human: 'человек', mouse: 'мышь', minnie65: 'Minnie65', h01: 'H01', labid: 'LabID'};
const vr = v => (v === null || v === undefined) ? '' : (VAL_RU[v] || v);
const factorName = f => D.factor_ru[f] || f;
const qtxt = q => (q === null || q === undefined) ? '—' : (q < 0.001 ? q.toExponential(1) : fmt(q, 3));
const sgn = (v, d) => (v >= 0 ? '+' : '') + fmt(v, d);
const metricLabel = m => (m.module ? '[' + m.module + '] ' : '') + m.name + (m.unit ? ', ' + m.unit : '');
const fillSelect = (sel, items, onChange) => {
  const s = $(sel); s.innerHTML = '';
  items.forEach(([v, t]) => s.append(el('option', {value: v, text: t})));
  s.onchange = onChange;
};

// ---------------------------------------------------------------- inventory
const INV_NAMES = {cell_type_binary: 'Тип клетки: тормозный / возбуждающий', cell_class: 'Класс клетки (унифицированный)', cell_type: 'Подробный тип клетки',
  layer: 'Слой (унифицированный)', compartment: 'Компартмент', health: 'Состояние здоровья', treatment: 'Лечение', group: 'Группа LabID'};
function drawInventory() {
  const col = $('sel-inv').value;
  const rows = D.inventory[col] || [];
  if (!rows.length) { $('plot-inv').innerHTML = '<div class="empty">Нет данных</div>'; return; }
  const values = [...new Set(rows.map(r => r.v))];
  const names = values.map(vr);
  const traces = [...new Set(rows.map(r => r.ds))].map(ds => ({type: 'bar', name: dsName(ds), x: names, marker: {color: DS_COLOR[ds] || P.reference},
    y: values.map(v => { const r = rows.find(q => q.ds === ds && q.v === v); return r ? r.units : 0; }),
    customdata: values.map(v => { const r = rows.find(q => q.ds === ds && q.v === v); return r ? r.rows : 0; }),
    hovertemplate: '%{x}: %{y} единиц, %{customdata} строк<extra>' + dsName(ds) + '</extra>'}));
  Plotly.newPlot('plot-inv', traces, layout({barmode: 'group', yaxis: {title: axisTitle('Единиц вывода')}, xaxis: {tickangle: values.length > 6 ? -40 : 0},
    legend: {orientation: 'h', y: 1.1}, margin: {b: 90}}), PLOT_CONFIG);
}
fillSelect('sel-inv', Object.keys(D.inventory).map(k => [k, INV_NAMES[k] || k]), drawInventory);
if (Object.keys(D.inventory).length) drawInventory(); else $('plot-inv').innerHTML = '<div class="empty">Нет таблиц состава</div>';

// ---------------------------------------------------------------- distributions
function distGroups(level, metric, by) {
  const rows = D.levels[level].dist.filter(r => r.metric === metric && r.n > 0 && r.hist_counts);
  return rows.filter(r => by === 'type' ? ('cell_type_binary' in r) : !('cell_type_binary' in r));
}
function drawDist() {
  const level = $('sel-dist-level').value, metric = $('sel-dist-metric').value, by = $('sel-dist-by').value;
  const m = D.levels[level].metrics.find(x => x.id === metric);
  const groups = distGroups(level, metric, by);
  if (!groups.length) { $('plot-dist').innerHTML = '<div class="empty">Нет данных для этой группировки</div>'; $('dist-table').innerHTML = ''; return; }
  const f = m.factor, dash = {inhibitory: 'dash', excitatory: 'solid', unknown: 'dot'};
  const label = r => dsName(r.dataset) + ('cell_type_binary' in r ? ', ' + ({inhibitory: 'тормозные', excitatory: 'возбуждающие', unknown: 'тип неизвестен'}[r.cell_type_binary] || r.cell_type_binary) : '');
  const traces = groups.map(r => {
    const e = r.hist_edges, c = r.hist_counts, tot = c.reduce((a, b) => a + b, 0) || 1;
    const x = [], y = [];
    c.forEach((v, i) => { const w = (e[i + 1] - e[i]) * f; x.push(e[i] * f, e[i + 1] * f); y.push(v / tot / w, v / tot / w); });
    return {type: 'scatter', mode: 'lines', name: label(r) + ' (n = ' + r.n + ')', x: x, y: y, line: {color: DS_COLOR[r.dataset] || P.reference, width: 2, dash: dash[r.cell_type_binary] || 'solid'},
      hovertemplate: '%{x:.3g}: %{y:.3g}<extra>' + label(r) + '</extra>'};
  });
  Plotly.newPlot('plot-dist', traces, layout({xaxis: {title: axisTitle(metricLabel(m))}, yaxis: {title: axisTitle('Плотность')}, legend: {orientation: 'h', y: 1.15}}), PLOT_CONFIG);
  const nm = level === 'spine' ? 'шипиков' : 'групп';
  $('dist-table').innerHTML = '<div style="overflow-x:auto;margin-top:10px"><table class="metrics"><thead><tr><th>Группа</th><th>' + nm + '</th><th>Единиц</th><th>Медиана</th><th>Квартили</th><th>Среднее</th><th>sd</th><th>ICC</th></tr></thead><tbody>' +
    groups.map(r => '<tr><td>' + label(r) + '</td><td>' + r.n + '</td><td>' + r.n_units + '</td><td>' + fmt(r.q50 * f) + '</td><td>' + fmt(r.q25 * f) + ' – ' + fmt(r.q75 * f) + '</td><td>' + fmt(r.mean * f) +
      '</td><td>' + fmt(r.sd * f) + '</td><td>' + fmt(r.icc, 2) + '</td></tr>').join('') + '</tbody></table></div>';
}
function fillDistMetrics() {
  const level = $('sel-dist-level').value;
  const have = new Set(D.levels[level].dist.map(r => r.metric));
  fillSelect('sel-dist-metric', D.levels[level].metrics.filter(m => have.has(m.id)).map(m => [m.id, metricLabel(m)]), drawDist);
  const hasType = D.levels[level].dist.some(r => 'cell_type_binary' in r);
  fillSelect('sel-dist-by', [['ds', 'Датасет']].concat(hasType ? [['type', 'Датасет и тип клетки (тормозные / возбуждающие)']] : []), drawDist);
  drawDist();
}
fillSelect('sel-dist-level', LEVELS.map(l => [l, D.level_ru[l]]), fillDistMetrics);
if (LEVELS.length) fillDistMetrics();

// ---------------------------------------------------------------- significance heatmap
function scopeOptions(level) { return [...new Set(D.levels[level].results.map(r => r.ds))]; }
function drawSig() {
  const level = $('sel-sig-level').value, scope = $('sel-sig-scope').value, factor = $('sel-sig-factor').value;
  const L = D.levels[level];
  const rows = L.results.filter(r => r.ds === scope && r.f === factor);
  const cids = [...new Set(rows.map(r => r.cid))].sort((a, b) => a - b);
  const metricsAll = L.metrics.filter(m => rows.some(r => r.m === m.id));
  const colInfo = cid => { const r = rows.find(q => q.cid === cid); return r; };
  const xlab = cids.map(cid => { const r = colInfo(cid); return vr(r.a) + ' / ' + vr(r.b) + (r.st ? ' [' + vr(r.st) + ']' : ''); });
  // warning block
  const first = rows[0], box = $('sig-warning');
  if (first) {
    const cf = first.cf ? JSON.parse(first.cf) : [];
    const warn = first.w ? '<p>' + first.w.split(' | ').join('</p><p>') + '</p>' : '';
    box.innerHTML = first.causal ? (warn ? '<div class="note" style="margin:0 0 10px"><h3>Оговорки</h3>' + warn + '</div>' : '')
      : '<div class="note pink" style="margin:0 0 10px"><h3>Описательно, смешано</h3><p>Это сравнение не проверяет причинное влияние фактора. Смешано с: <b>' + cf.join('; ') + '</b>.</p>' + warn + '</div>';
  } else box.innerHTML = '';
  const skipped = L.comparisons.filter(c => c.ds === scope && c.f === factor && !c.valid);
  if (skipped.length) box.innerHTML += '<p class="small muted">Не проверялось: ' + skipped.map(c => vr(c.a) + (c.b ? ' / ' + vr(c.b) : '') + (c.st ? ' [' + vr(c.st) + ']' : '') + ' — ' + c.reason).join('; ') + '.</p>';
  if (!rows.length) { $('plot-sig').innerHTML = '<div class="empty">Нет проведённых тестов для этого сочетания</div>'; return; }
  const y = metricsAll.map(metricLabel);
  const zSig = [], zNs = [], zUn = [], hover = [], ann = [];
  metricsAll.forEach((m, i) => {
    const rs = [], rn = [], ru = [], rh = [];
    cids.forEach((cid, j) => {
      const r = rows.find(q => q.cid === cid && q.m === m.id);
      if (!r || r.eff === null || r.lab === 'not_tested') { rs.push(null); rn.push(null); ru.push(null); rh.push('не проверялось'); return; }
      const absK = r.kind === 'abs';
      const shown = absK ? r.eff * (m.abs ? 1 : m.factor) : 100 * r.eff;
      const colour = Math.max(-LIMIT, Math.min(LIMIT, absK ? 100 * r.eff : 100 * r.eff));
      const sig = r.lab === 'significant', under = r.lab === 'underpowered';
      rs.push(sig ? colour : null); rn.push(r.lab === 'not_significant' ? 0 : null); ru.push(under ? 0 : null);
      const txt = absK ? sgn(shown, 2) : sgn(shown, 0);
      const strong = sig && Math.abs(colour) > 0.6 * LIMIT;
      ann.push({x: xlab[j], y: y[i], text: txt + (sig ? ' *' : (under ? ' ?' : '')), showarrow: false, font: {size: 11, color: strong ? '#ffffff' : (sig ? P.ink : P.ink_muted)}});
      const u = m.unit && !absK ? '' : '';
      const f = (v) => v === null ? '—' : fmt(absK ? v * (m.abs ? 1 : m.factor) : 100 * v, absK ? 2 : 1) + (absK ? '' : ' %');
      rh.push('<b>' + m.name + '</b><br>' + xlab[j] + '<br>эффект: ' + f(r.eff) + ' (95 %: ' + f(r.lo) + ' … ' + f(r.hi) + ')' + (absK ? ' — абсолютная разность' : '') +
        '<br>минимально обнаружимый: ' + f(r.mde) + ', порог значимости: ' + f(r.mg) +
        '<br>единиц: ' + fmt(r.na) + ' / ' + fmt(r.nb) + (r.ma !== null ? '<br>медианы: ' + fmt(r.ma * m.factor) + ' / ' + fmt(r.mb * m.factor) + (m.unit ? ' ' + m.unit : '') : '') +
        '<br>p = ' + qtxt(r.p) + ', q = ' + qtxt(r.q) + '<br><b>' + (D.label_ru[r.lab] || r.lab) + '</b>' + (r.eq ? ' (эквивалентно: интервал внутри порога)' : '') +
        (r.causal ? '' : '<br><i>описательно, смешано</i>'));
    });
    zSig.push(rs); zNs.push(rn); zUn.push(ru); hover.push(rh);
  });
  const common = {type: 'heatmap', x: xlab, y: y, xgap: 2, ygap: 2, hoverinfo: 'text', text: hover};
  const traces = [
    Object.assign({}, common, {z: zUn, colorscale: [[0, UNDER], [1, UNDER]], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: zNs, colorscale: [[0, GREY_NS], [1, GREY_NS]], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: zSig, colorscale: THEME.diverging, zmin: -LIMIT, zmax: LIMIT,
      colorbar: {title: {text: '%', side: 'top'}, thickness: 12, len: 0.6, tickfont: {size: 11, color: P.ink_muted}}}),
  ];
  const maxLab = Math.max(...xlab.map(s => s.length));
  const top = xlab.length > 3 ? Math.min(220, 40 + maxLab * 3.6) : 50;
  const div = $('plot-sig');
  div.style.height = (top + 40 + 28 * y.length) + 'px';
  Plotly.newPlot('plot-sig', traces, layout({annotations: ann, margin: {l: 290, r: 20, t: top, b: 10},
    xaxis: {side: 'top', tickangle: xlab.length > 3 ? -35 : 0, tickfont: {size: 11.5, color: P.ink_secondary}, showgrid: false, zeroline: false},
    yaxis: {autorange: 'reversed', showgrid: false, zeroline: false, tickfont: {size: 12, color: P.ink}}}), PLOT_CONFIG);
}
function fillSigFactors() {
  const level = $('sel-sig-level').value, scope = $('sel-sig-scope').value;
  const fs = [...new Set(D.levels[level].results.filter(r => r.ds === scope).map(r => r.f))];
  fillSelect('sel-sig-factor', fs.map(f => [f, factorName(f)]), drawSig);
  drawSig();
}
function fillSigScopes() {
  const level = $('sel-sig-level').value;
  const order = ['minnie65', 'h01', 'labid'];
  const sc = scopeOptions(level).sort((a, b) => (a.includes(' vs ') - b.includes(' vs ')) || (order.indexOf(a) - order.indexOf(b)));
  fillSelect('sel-sig-scope', sc.map(s => [s, dsName(s) + (s.includes(' vs ') ? ' — описательно' : '')]), fillSigFactors);
  fillSigFactors();
}
fillSelect('sel-sig-level', LEVELS.map(l => [l, D.level_ru[l]]), fillSigScopes);
$('sig-legend').append(
  el('span', {html: '<i class="swatch" style="background:' + P.series_1 + '"></i> значимо, первая группа больше'}),
  el('span', {html: '<i class="swatch" style="background:' + P.series_2 + '"></i> значимо, первая группа меньше'}),
  el('span', {html: '<i class="swatch" style="background:' + GREY_NS + '"></i> не значимо, мощности достаточно'}),
  el('span', {html: '<i class="swatch" style="background:' + UNDER + ';border:1px solid #d9c98f"></i> не значимо, мощности не хватает (?)'}));
if (LEVELS.length) fillSigScopes();
