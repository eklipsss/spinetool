// Morphology by condition (src/neuron_model/reports/morphology_report.py). Uses COMMON_JS: $, fmt, layout, axisTitle, PLOT_CONFIG, el.
const D = PAYLOAD;
const METRICS = D.metrics;
const LIMIT = 50;  // colour scale clipped at +-50 %
const GREY = P.muted_points;
const label = m => m.name + (m.unit ? ', ' + m.unit : '');
const pct = v => (v === null || v === undefined) ? '—' : (v >= 0 ? '+' : '') + fmt(100 * v, 1) + ' %';
const qtxt = q => (q === null || q === undefined) ? '—' : (q < 0.001 ? q.toExponential(1) : fmt(q, 3));

function fillSelect(id, onChange) {
  const sel = $(id);
  if (!sel) return;
  METRICS.forEach((m, i) => sel.append(el('option', {value: String(i), text: label(m)})));
  sel.addEventListener('change', () => onChange(METRICS[+sel.value]));
  onChange(METRICS[0]);
}

// ---------------------------------------------------------------- overview heatmap
function drawOverview() {
  const cols = D.columns;
  if (!cols.length) { $('overview').innerHTML = '<div class="empty">Нет сравнений</div>'; return; }
  const y = METRICS.map(label);
  const x = cols.map(c => c.label);
  const zSig = [], zNon = [], text = [], hover = [];
  METRICS.forEach((m, i) => {
    const rs = [], rn = [], rt = [], rh = [];
    cols.forEach(c => {
      const cell = D.cells[c.key][i];
      const rel = cell.rel;
      const v = rel === null ? null : Math.max(-LIMIT, Math.min(LIMIT, 100 * rel));
      rs.push(cell.sig ? v : null);
      rn.push(!cell.sig && rel !== null ? 0 : null);
      rt.push(rel === null ? '' : pct(rel) + (cell.sig ? ' *' : ''));
      const unit = m.unit ? ' ' + m.unit : '';
      const effect = c.kind === 'compartment'
        ? 'доля нейронов «первый больше»: ' + fmt(cell.share, 2) + '<br>нейронов с обоими: ' + fmt(cell.na)
        : 'медианы: ' + fmt(cell.ma) + unit + ' / ' + fmt(cell.mb) + unit + '<br>нейронов: ' + fmt(cell.na) + ' / ' + fmt(cell.nb) + '<br>дельта Клиффа: ' + fmt(cell.delta, 2);
      rh.push(rel === null ? 'нет данных' :
        '<b>' + m.name + '</b><br>' + c.label.replace('<br>', ', ') + '<br>разность: ' + pct(rel) + '<br>' + effect +
        '<br>p = ' + qtxt(cell.p) + ', q = ' + qtxt(cell.q) + '<br>' + (cell.sig ? '<b>значимо</b>' : 'не значимо'));
    });
    zSig.push(rs); zNon.push(rn); text.push(rt); hover.push(rh);
  });
  const common = {type: 'heatmap', x: x, y: y, xgap: 2, ygap: 2, hoverinfo: 'text', text: hover};
  const traces = [
    Object.assign({}, common, {z: zNon, colorscale: [[0, '#f3f3f5'], [1, '#f3f3f5']], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: zSig, colorscale: THEME.diverging, zmin: -LIMIT, zmax: LIMIT,
      colorbar: {title: {text: '%', side: 'top'}, thickness: 12, len: 0.8, tickfont: {size: 11, color: P.ink_muted}}}),
  ];
  const annotations = [];
  METRICS.forEach((m, i) => cols.forEach((c, j) => {
    const cell = D.cells[c.key][i];
    if (cell.rel === null) return;
    const strong = cell.sig && Math.abs(100 * cell.rel) > 0.6 * LIMIT;
    annotations.push({x: x[j], y: y[i], text: text[i][j], showarrow: false,
      font: {size: 11.5, color: strong ? '#ffffff' : (cell.sig ? P.ink : P.ink_muted)}});
  }));
  Plotly.newPlot('overview', traces, layout({annotations: annotations, margin: {l: 230, r: 20, t: 70, b: 10},
    xaxis: {side: 'top', tickfont: {size: 11.5, color: P.ink_secondary}, showgrid: false, zeroline: false},
    yaxis: {autorange: 'reversed', showgrid: false, zeroline: false, tickfont: {size: 12, color: P.ink}}}), PLOT_CONFIG);
}

// ---------------------------------------------------------------- binary: boxes per group (all + compartments)
function drawBinary(m) {
  const data = (D.dist.by_binary || {})[m.id];
  if (!data) { $('plot-binary').innerHTML = '<div class="empty">Нет данных</div>'; return; }
  const groups = Object.keys(data);
  const groupLabel = g => g === 'all' ? 'все шипики' : (D.compartment_ru[g] || g);
  const traces = [['inhibitory', P.series_1], ['excitatory', P.series_2]].map(([b, color]) => {
    const xs = [], ys = [];
    groups.forEach(g => (data[g][b] || []).forEach(v => { if (v !== null) { xs.push(groupLabel(g)); ys.push(v); } }));
    return {type: 'box', name: D.binary_ru[b], x: xs, y: ys, marker: {color: color, size: 4, opacity: 0.55}, line: {color: color, width: 1.5},
      fillcolor: 'rgba(0,0,0,0)', boxpoints: 'all', jitter: 0.35, pointpos: 0, hovertemplate: '%{y:.3g}<extra>' + D.binary_ru[b] + '</extra>'};
  });
  Plotly.newPlot('plot-binary', traces, layout({boxmode: 'group', yaxis: {title: axisTitle(label(m))}, legend: {orientation: 'h', y: 1.08}}), PLOT_CONFIG);
}

// ---------------------------------------------------------------- detailed types: matrix + boxes
function drawTypes(m) {
  const mat = (D.matrices || {})[m.id];
  const types = D.types.map(t => t.type);
  if (!mat || !types.length) {
    $('plot-type-matrix').innerHTML = '<div class="empty">Нет сравнений типов</div>';
    $('plot-type-box').innerHTML = '';
    return;
  }
  const n = types.length;
  const zSig = [], zNon = [], hover = [], annotations = [];
  for (let i = 0; i < n; i++) {
    const rs = [], rn = [], rh = [];
    for (let j = 0; j < n; j++) {
      const rel = mat.rel[i][j], sig = mat.sig[i][j];
      const v = rel === null ? null : Math.max(-LIMIT, Math.min(LIMIT, 100 * rel));
      rs.push(sig ? v : null);
      rn.push(!sig && rel !== null ? 0 : null);
      rh.push(i === j || rel === null ? types[i] : '<b>' + types[i] + ' / ' + types[j] + '</b><br>' + m.name + ': ' + pct(rel) + '<br>q = ' + qtxt(mat.q[i][j]) + '<br>' + (sig ? '<b>значимо</b>' : 'не значимо'));
      if (rel !== null && i !== j) annotations.push({x: types[j], y: types[i], text: (rel >= 0 ? '+' : '') + Math.round(100 * rel), showarrow: false,
        font: {size: 10, color: sig && Math.abs(100 * rel) > 0.6 * LIMIT ? '#ffffff' : (sig ? P.ink : P.ink_muted)}});
    }
    zSig.push(rs); zNon.push(rn); hover.push(rh);
  }
  const common = {type: 'heatmap', x: types, y: types, xgap: 2, ygap: 2, hoverinfo: 'text', text: hover};
  Plotly.newPlot('plot-type-matrix', [
    Object.assign({}, common, {z: zNon, colorscale: [[0, '#f3f3f5'], [1, '#f3f3f5']], showscale: false, zmin: 0, zmax: 1}),
    Object.assign({}, common, {z: zSig, colorscale: THEME.diverging, zmin: -LIMIT, zmax: LIMIT, colorbar: {title: {text: '%'}, thickness: 10, len: 0.8}}),
  ], layout({annotations: annotations, margin: {l: 70, r: 10, t: 10, b: 70}, xaxis: {showgrid: false, zeroline: false, tickangle: -45},
    yaxis: {autorange: 'reversed', showgrid: false, zeroline: false}}), PLOT_CONFIG);

  const byType = (D.dist.by_type || {})[m.id] || {};
  const traces = D.types.map(t => {
    const color = t.binary === 'inhibitory' ? P.series_1 : P.series_2;
    return {type: 'box', name: t.type + ' (' + t.n + ')', y: (byType[t.type] || []).filter(v => v !== null), marker: {color: color, size: 3.5, opacity: 0.5},
      line: {color: color, width: 1.5}, fillcolor: 'rgba(0,0,0,0)', boxpoints: 'all', jitter: 0.4, pointpos: 0, showlegend: false,
      hovertemplate: '%{y:.3g}<extra>' + t.type + '</extra>'};
  });
  Plotly.newPlot('plot-type-box', traces, layout({yaxis: {title: axisTitle(label(m))}, xaxis: {tickangle: -45}, margin: {b: 80}}), PLOT_CONFIG);
  const om = (D.omnibus || {})[m.id];
  $('omnibus-line').textContent = om ? 'Общий тест по всем типам (' + m.name + '): q = ' + qtxt(om.q) + (om.sig ? ' — типы различаются значимо.' : ' — значимого различия нет.') +
    ' Розовые ящики — тормозные типы, зелёные — возбуждающие; в скобках число нейронов.' : '';
}

// ---------------------------------------------------------------- compartments: paired differences
function drawComp(m) {
  const pairs = D.dist.paired || {};
  const keys = Object.keys(pairs).filter(k => (pairs[k][m.id] || []).length);
  if (!keys.length) { $('plot-comp').innerHTML = '<div class="empty">Нет нейронов с двумя компартментами</div>'; return; }
  const name = k => { const [a, b] = k.split('_'); return (D.compartment_ru[a] || a) + ' / ' + (D.compartment_ru[b] || b); };
  const traces = keys.map((k, i) => ({type: 'box', name: name(k) + ' (' + pairs[k][m.id].length + ')', y: pairs[k][m.id].filter(v => v !== null && Math.abs(v) < 400),
    marker: {color: P.reference, size: 3.5, opacity: 0.5}, line: {color: P.ink_secondary, width: 1.5}, fillcolor: 'rgba(0,0,0,0)',
    boxpoints: 'all', jitter: 0.4, pointpos: 0, showlegend: false, hovertemplate: '%{y:.1f} %<extra></extra>'}));
  Plotly.newPlot('plot-comp', traces, layout({yaxis: {title: axisTitle('Разность ' + m.name.toLowerCase() + ' внутри нейрона, %'), zeroline: true, zerolinecolor: P.ink_muted, zerolinewidth: 1.5}}), PLOT_CONFIG);
}

drawOverview();
fillSelect('sel-binary', drawBinary);
fillSelect('sel-types', drawTypes);
fillSelect('sel-comp', drawComp);
