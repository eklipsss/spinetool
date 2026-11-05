// Global report (src/neuron_model/reports/global_report.py). Uses COMMON_JS: $, fmt, layout, axisTitle, PLOT_CONFIG, el.
const D = PAYLOAD;
const LEVELS = Object.keys(D.levels);
let LEVEL = LEVELS[0];                               // tab: spine / trunk / placement
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
  const level = LEVEL, metric = $('sel-dist-metric').value, by = $('sel-dist-by').value;
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
  const nm = level === 'spine' ? 'шипиков' : 'групп (нейрон × компартмент)';
  $('dist-table').innerHTML = '<div style="overflow-x:auto;margin-top:10px"><table class="metrics"><thead><tr><th>Группа</th><th>' + nm + '</th><th>Единиц</th><th>Медиана</th><th>Квартили</th><th>Среднее</th><th>sd</th><th>ICC</th></tr></thead><tbody>' +
    groups.map(r => '<tr><td>' + label(r) + '</td><td>' + r.n + '</td><td>' + r.n_units + '</td><td>' + fmt(r.q50 * f) + '</td><td>' + fmt(r.q25 * f) + ' – ' + fmt(r.q75 * f) + '</td><td>' + fmt(r.mean * f) +
      '</td><td>' + fmt(r.sd * f) + '</td><td>' + fmt(r.icc, 2) + '</td></tr>').join('') + '</tbody></table></div>';
}
function fillDistMetrics() {
  const level = LEVEL;
  const have = new Set(D.levels[level].dist.map(r => r.metric));
  fillSelect('sel-dist-metric', D.levels[level].metrics.filter(m => have.has(m.id)).map(m => [m.id, metricLabel(m)]), drawDist);
  const hasType = D.levels[level].dist.some(r => 'cell_type_binary' in r);
  fillSelect('sel-dist-by', [['ds', 'Датасет']].concat(hasType ? [['type', 'Датасет и тип клетки (тормозные / возбуждающие)']] : []), drawDist);
  drawDist();
}

// ---------------------------------------------------------------- significance heatmap
function scopeOptions(level) { return [...new Set(D.levels[level].results.map(r => r.ds))]; }
function drawSig() {
  const level = LEVEL, scope = $('sel-sig-scope').value, factor = $('sel-sig-factor').value;
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
  drawOmni(level, scope, factor);
  fillPairs(level, scope, factor);
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
  const level = LEVEL, scope = $('sel-sig-scope').value;
  const fs = [...new Set(D.levels[level].results.filter(r => r.ds === scope).map(r => r.f))];
  fillSelect('sel-sig-factor', fs.map(f => [f, factorName(f)]), drawSig);
  drawSig();
}
function fillSigScopes() {
  const level = LEVEL;
  const order = ['minnie65', 'h01', 'labid'];
  const sc = scopeOptions(level).sort((a, b) => (a.includes(' vs ') - b.includes(' vs ')) || (order.indexOf(a) - order.indexOf(b)));
  fillSelect('sel-sig-scope', sc.map(s => [s, dsName(s) + (s.includes(' vs ') ? ' — описательно' : '')]), fillSigFactors);
  fillSigFactors();
}
$('sig-legend').append(
  el('span', {html: '<i class="swatch" style="background:' + P.series_1 + '"></i> значимо, первая группа больше'}),
  el('span', {html: '<i class="swatch" style="background:' + P.series_2 + '"></i> значимо, первая группа меньше'}),
  el('span', {html: '<i class="swatch" style="background:' + GREY_NS + '"></i> не значимо, мощности достаточно'}),
  el('span', {html: '<i class="swatch" style="background:' + UNDER + ';border:1px solid #d9c98f"></i> не значимо, мощности не хватает (?)'}));

// ---------------------------------------------------------------- omnibus (factors with >= 3 levels)
function drawOmni(level, scope, factor) {
  const rows = (D.levels[level].omnibus || []).filter(r => r.dataset === scope && r.factor === factor);
  const box = $('omni');
  if (!rows.length) { box.innerHTML = ''; return; }
  const mt = Object.fromEntries(D.levels[level].metrics.map(m => [m.id, m]));
  box.innerHTML = '<h3 style="margin:16px 0 6px">Общий тест: различается ли метрика хоть между какими-то уровнями</h3>' +
    '<div style="overflow-x:auto"><table class="metrics"><thead><tr><th>Метрика</th><th>Уровней</th><th>Единиц</th><th>p</th><th>q</th><th>Вывод</th></tr></thead><tbody>' +
    rows.map(r => '<tr><td>' + (mt[r.metric] ? metricLabel(mt[r.metric]) : r.metric) + '</td><td>' + r.n_types + '</td><td>' + r.n_neurons + '</td><td>' + qtxt(r.p) + '</td><td>' + qtxt(r.q_bh) +
      '</td><td>' + (r.significant ? '<b>значимо</b>' : 'не значимо') + '</td></tr>').join('') + '</tbody></table></div>';
}

// ---------------------------------------------------------------- pair matrix of the levels of one factor
let PAIR_CTX = null;
function fillPairs(level, scope, factor) {
  const rows = D.levels[level].results.filter(r => r.ds === scope && r.f === factor && !r.st);
  const levels = [...new Set(rows.flatMap(r => [r.a, r.b]))];
  if (levels.length < 3) { PAIR_CTX = null; $('pairs-card').style.display = 'none'; return; }   // all pairs are on the heatmap
  $('pairs-card').style.display = '';
  PAIR_CTX = {level: level, rows: rows, levels: levels.sort()};
  const have = new Set(rows.map(r => r.m));
  const prev = $('sel-pair-metric').value;
  fillSelect('sel-pair-metric', D.levels[level].metrics.filter(m => have.has(m.id)).map(m => [m.id, metricLabel(m)]), drawPairs);
  if (prev && have.has(prev)) $('sel-pair-metric').value = prev;
  drawPairs();
}
function drawPairs() {
  if (!PAIR_CTX) return;
  const metric = $('sel-pair-metric').value, L = PAIR_CTX.levels;
  const m = D.levels[PAIR_CTX.level].metrics.find(x => x.id === metric);
  const z = L.map(() => L.map(() => null)), zNs = L.map(() => L.map(() => null)), zUn = L.map(() => L.map(() => null));
  const txt = L.map(() => L.map(() => '')), hov = L.map(() => L.map(() => ''));
  PAIR_CTX.rows.filter(r => r.m === metric).forEach(r => {
    const i = L.indexOf(r.a), j = L.indexOf(r.b);
    if (i < 0 || j < 0 || r.eff === null || r.lab === 'not_tested') return;
    const absK = r.kind === 'abs';
    // the stored row is "a vs b"; the mirrored cell is "b vs a": relative 1/(1+e) - 1, absolute -e
    [[i, j, r.eff], [j, i, absK ? -r.eff : 1 / (1 + r.eff) - 1]].forEach(([p, q, e]) => {
      const v = absK ? e * (m.abs ? 1 : m.factor) : 100 * e;
      const sig = r.lab === 'significant';
      if (sig) z[p][q] = Math.max(-LIMIT, Math.min(LIMIT, 100 * e)); else if (r.lab === 'underpowered') zUn[p][q] = 0; else zNs[p][q] = 0;
      txt[p][q] = (absK ? sgn(v, 2) : sgn(v, 0)) + (sig ? ' *' : (r.lab === 'underpowered' ? ' ?' : ''));
      hov[p][q] = '<b>' + vr(L[p]) + ' против ' + vr(L[q]) + '</b><br>' + m.name + ': ' + (absK ? sgn(v, 2) : sgn(v, 1) + ' %') + '<br>q = ' + qtxt(r.q) + '<br>' + (D.label_ru[r.lab] || r.lab);
    });
  });
  const names = L.map(vr);
  const common = {type: 'heatmap', x: names, y: names, xgap: 2, ygap: 2, hoverinfo: 'text', text: hov};
  const ann = [];
  L.forEach((_, p) => L.forEach((__, q) => { if (txt[p][q]) ann.push({x: names[q], y: names[p], text: txt[p][q], showarrow: false,
    font: {size: 11, color: z[p][q] !== null && Math.abs(z[p][q]) > 0.6 * LIMIT ? '#ffffff' : (z[p][q] !== null ? P.ink : P.ink_muted)}}); }));
  const div = $('plot-pairs');
  div.style.height = (120 + 34 * L.length) + 'px';
  Plotly.newPlot('plot-pairs', [
    Object.assign({}, common, {z: zUn, colorscale: [[0, UNDER], [1, UNDER]], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: zNs, colorscale: [[0, GREY_NS], [1, GREY_NS]], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: z, colorscale: THEME.diverging, zmin: -LIMIT, zmax: LIMIT, colorbar: {title: {text: '%', side: 'top'}, thickness: 12, len: 0.6}})],
    layout({annotations: ann, margin: {l: 120, r: 20, t: 90, b: 10}, xaxis: {side: 'top', tickangle: -35, showgrid: false, zeroline: false, title: axisTitle('вторая группа')},
      yaxis: {autorange: 'reversed', showgrid: false, zeroline: false, title: axisTitle('первая группа')}}), PLOT_CONFIG);
  $('pair-note').textContent = 'Строка — первая группа, столбец — вторая; цвет только у значимых различий (q < 0,05), «?» — мощности не хватает.';
}

// ---------------------------------------------------------------- Minnie65 vs H01: the same effect?
const VER_COLOR = {consistent: P.series_2, same_sign_different_size: P.series_1, opposite: P.series_1, minnie65_only: P.reference, h01_only: P.reference, neither: '#c8c8cc'};
function fillCons() {
  const S = D.levels[LEVEL].consistency_summary || [];
  if (!S.length) { $('sel-cons').innerHTML = ''; $('cons-summary').innerHTML = ''; $('cons-table').innerHTML = '';
    $('plot-cons').innerHTML = '<div class="empty">Нет сравнений, допустимых в обоих датасетах</div>'; return; }
  fillSelect('sel-cons', S.map((r, i) => [i, factorName(r.f) + ': ' + vr(r.a) + ' / ' + vr(r.b)]), drawCons);
  drawCons();
}
function drawCons() {
  const S = D.levels[LEVEL].consistency_summary, s = S[+$('sel-cons').value];
  const mt = Object.fromEntries(D.levels[LEVEL].metrics.map(m => [m.id, m]));
  const rows = D.levels[LEVEL].consistency.filter(r => r.f === s.f && r.a === s.a && r.b === s.b);
  const pct = v => v === null ? '—' : sgn(100 * v, 1) + ' %';
  const val = (r, v) => r.kind === 'abs' ? (v === null ? '—' : sgn(v * (mt[r.m] && !mt[r.m].abs ? mt[r.m].factor : 1), 3)) : pct(v);
  const pp = (r, v) => r.kind === 'abs' ? val(r, v) : (v === null ? '—' : sgn(100 * v, 1) + ' п. п.');
  $('cons-summary').innerHTML = '<p class="small" style="margin:0 0 8px">Метрик ' + s.n + '; значимо в Minnie65 ' + s.nm + ', в H01 ' + s.nh + ', в обоих ' + s.both +
    ' (знак совпал у ' + s.same_both + '); знак совпал у ' + fmt(100 * s.same_all, 0) + ' % всех метрик; разность эффектов значима у ' + s.ndiff +
    '. Ранговая корреляция эффектов: ' + (s.rho === null ? '—' : fmt(s.rho, 2) + ' (95 %: ' + fmt(s.rho_lo, 2) + ' … ' + fmt(s.rho_hi, 2) + ')') + '.</p>';
  const rel = rows.filter(r => r.kind === 'rel');
  if (rel.length) {
    const lim = Math.max(10, ...rel.flatMap(r => [r.em_lo, r.em_hi, r.eh_lo, r.eh_hi].filter(v => v !== null).map(v => Math.abs(100 * v)))) * 1.05;
    const traces = Object.keys(VER_COLOR).map(v => {
      const g = rel.filter(r => r.ver === v);
      return {type: 'scatter', mode: 'markers+text', name: D.verdict_ru[v] || v, x: g.map(r => 100 * r.em), y: g.map(r => 100 * r.eh),
        text: g.map(r => (r.pm || r.ph) ? (mt[r.m] ? mt[r.m].name : r.m) : ''), textposition: 'top center', textfont: {size: 10.5, color: P.ink_secondary},
        meta: g.map(r => mt[r.m] ? mt[r.m].name : r.m),
        marker: {size: 10, color: VER_COLOR[v], line: {color: '#ffffff', width: 2}},
        error_x: {type: 'data', symmetric: false, array: g.map(r => 100 * (r.em_hi - r.em)), arrayminus: g.map(r => 100 * (r.em - r.em_lo)), color: VER_COLOR[v], thickness: 1.2, width: 0},
        error_y: {type: 'data', symmetric: false, array: g.map(r => 100 * (r.eh_hi - r.eh)), arrayminus: g.map(r => 100 * (r.eh - r.eh_lo)), color: VER_COLOR[v], thickness: 1.2, width: 0},
        customdata: g.map(r => [pct(r.em), pct(r.eh), pp(r, r.d), qtxt(r.q)]),
        hovertemplate: '<b>%{meta}</b><br>Minnie65: %{customdata[0]}<br>H01: %{customdata[1]}<br>разность: %{customdata[2]}, q = %{customdata[3]}<extra>' + (D.verdict_ru[v] || v) + '</extra>'};
    }).filter(t => t.x.length);
    traces.unshift({type: 'scatter', mode: 'lines', x: [-lim, lim], y: [-lim, lim], line: {color: P.reference, dash: 'dot', width: 1}, hoverinfo: 'skip', showlegend: false});
    Plotly.newPlot('plot-cons', traces, layout({xaxis: {title: axisTitle('Эффект в Minnie65, %'), range: [-lim, lim], zeroline: true},
      yaxis: {title: axisTitle('Эффект в H01, %'), range: [-lim, lim], zeroline: true}, legend: {orientation: 'h', y: 1.12}}), PLOT_CONFIG);
  } else $('plot-cons').innerHTML = '<div class="empty">У этих метрик абсолютные эффекты: смотрите таблицу</div>';
  $('cons-table').innerHTML = '<div style="overflow-x:auto;margin-top:10px"><table class="metrics"><thead><tr><th>Метрика</th><th>Minnie65 (95 %)</th><th>H01 (95 %)</th><th>Разность (95 %)</th><th>q</th><th>Вывод</th></tr></thead><tbody>' +
    rows.map(r => '<tr><td>' + (mt[r.m] ? metricLabel(mt[r.m]) : r.m) + '</td><td>' + val(r, r.em) + ' (' + val(r, r.em_lo) + ' … ' + val(r, r.em_hi) + ')' + (r.pm ? ' *' : '') +
      '</td><td>' + val(r, r.eh) + ' (' + val(r, r.eh_lo) + ' … ' + val(r, r.eh_hi) + ')' + (r.ph ? ' *' : '') + '</td><td>' + pp(r, r.d) + ' (' + pp(r, r.d_lo) + ' … ' + pp(r, r.d_hi) + ')</td><td>' + qtxt(r.q) +
      '</td><td>' + (D.verdict_ru[r.ver] || r.ver) + '</td></tr>').join('') + '</tbody></table><p class="small muted">* — эффект значим в этом датасете (тепловая карта, q < 0,05).</p></div>';
}

// ---------------------------------------------------------------- level tabs (spine / trunk / placement)
function setLevel(l) {
  LEVEL = l;
  document.querySelectorAll('#level-tabs button').forEach(b => b.classList.toggle('on', b.dataset.level === l));
  const mod = {spine: 'S', trunk: 'T', placement: 'P'}[l];
  document.querySelectorAll('tr[data-mod]').forEach(tr => { tr.style.display = tr.dataset.mod === mod ? '' : 'none'; });
  fillDistMetrics(); fillSigScopes(); fillCons();
}
LEVELS.forEach(l => { const b = el('button', {text: D.level_ru[l] || l, 'data-level': l}); b.onclick = () => setLevel(l); $('level-tabs').append(b); });
if (LEVELS.length) setLevel(LEVELS[0]);
