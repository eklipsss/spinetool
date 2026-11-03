// Conditional generation report (src/neuron_model/reports/conditional_report.py). Uses COMMON_JS: $, fmt, layout, axisTitle, PLOT_CONFIG, el.
const D = PAYLOAD;
const R = D.results;
const M = D.metrics;
const MI = Object.fromEntries(M.map(m => [m.id, m]));
const VR = D.value_ru;
const PINK = P.series_1, GREEN = P.series_2, GREY = P.reference;
const esc = s => String(s === null || s === undefined ? '' : s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const pctOf = lr => (lr === null || lr === undefined) ? null : 100 * (Math.exp(lr) - 1);
const fp = v => (v === null || v === undefined) ? '—' : (v >= 0 ? '+' : '') + fmt(v, 1) + ' %';
const qtxt = q => (q === null || q === undefined) ? '—' : (q < 0.001 ? q.toExponential(1) : fmt(q, 3));
const f2 = v => (v === null || v === undefined) ? '—' : fmt(v, 2);
const pill = (cls, text) => '<span class="pill ' + cls + '">' + esc(text) + '</span>';
const setTitle = (v, l) => (VR[v] || v) + (l === 'default' ? '' : ', ' + l);
const valueColor = v => v === 'inhibitory' ? PINK : (v === 'excitatory' ? GREEN : P.series_3);
const mname = id => (MI[id] ? MI[id].name : id);

function table(id, cols, rows) {
  const host = $(id);
  if (!host) return;
  if (!rows.length) { host.innerHTML = '<div class="empty">Нет данных</div>'; return; }
  host.innerHTML = '<table class="metrics"><thead><tr>' + cols.map(c => '<th>' + c.h + '</th>').join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => '<td>' + c.f(r) + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
}
function fillSelect(id, options, onChange) {
  const sel = $(id);
  if (!sel) return;
  options.forEach(o => sel.append(el('option', {value: o.value, text: o.text})));
  sel.addEventListener('change', () => onChange(sel.value));
  if (options.length <= 1) sel.parentElement.style.display = 'none';
  if (options.length) onChange(options[0].value);
}
function noData(id, text) { $(id).innerHTML = '<div class="empty">' + esc(text || 'Нет данных') + '</div>'; }
const lineShape = (x, color, dash) => ({type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: {color: color, width: 2, dash: dash || 'solid'}});

// ----------------------------------------------------------------------------------------------- A
function drawA(label) {
  const e = (R.effects || {})[label];
  if (!e) { noData('plot-forest', 'Нет наборов обоих типов с одной меткой'); return; }
  const rows = e.concordance;
  const idx = rows.map((_, i) => i);
  const mk = (suffix, color, shift, name) => {
    const lr = rows.map(r => r['lr_' + suffix]), lo = rows.map(r => r['lo_' + suffix]), hi = rows.map(r => r['hi_' + suffix]);
    const x = lr.map(pctOf);
    return {type: 'scatter', mode: 'markers', name: name, x: x, y: idx.map(i => i + shift),
      error_x: {type: 'data', symmetric: false, array: x.map((v, i) => v === null ? 0 : pctOf(hi[i]) - v), arrayminus: x.map((v, i) => v === null ? 0 : v - pctOf(lo[i])), color: color, thickness: 1.6, width: 4},
      marker: {color: color, size: 9, symbol: rows.map(r => (suffix === 'real' ? r.real_sig : r.gen_sig) ? 'circle' : 'circle-open'), line: {color: color, width: 2}},
      text: rows.map((r, i) => '<b>' + mname(r.metric) + '</b><br>' + name + ': ' + fp(x[i]) + ' [' + fp(pctOf(lo[i])) + '; ' + fp(pctOf(hi[i])) + ']<br>q = ' + qtxt(r[suffix + '_q']) + '<br>статус: ' + D.status_ru[r.status]),
      hoverinfo: 'text'};
  };
  Plotly.newPlot('plot-forest', [mk('real', GREY, -0.17, 'реальные (нейрон — единица)'), mk('gen', PINK, 0.17, 'сгенерированные')],
    layout({yaxis: {tickvals: idx, ticktext: rows.map(r => mname(r.metric)), autorange: 'reversed', showgrid: true, zeroline: false},
      xaxis: {title: axisTitle('изменение медианы: ' + (VR[e.pair[0]] || e.pair[0]) + ' относительно ' + (VR[e.pair[1]] || e.pair[1]) + ', %'), zeroline: true, zerolinecolor: P.ink_muted, zerolinewidth: 1.5},
      legend: {orientation: 'h', y: 1.06}, margin: {l: 150}}), PLOT_CONFIG);
  table('table-a', [
    {h: 'Метрика', f: r => esc(mname(r.metric))},
    {h: 'Реальные, %', f: r => fp(pctOf(r.lr_real)) + ' <span class="muted">[' + fp(pctOf(r.lo_real)) + '; ' + fp(pctOf(r.hi_real)) + ']</span>'},
    {h: 'q (реал.)', f: r => qtxt(r.real_q)},
    {h: 'Сгенерированные, %', f: r => fp(pctOf(r.lr_gen)) + ' <span class="muted">[' + fp(pctOf(r.lo_gen)) + '; ' + fp(pctOf(r.hi_gen)) + ']</span>'},
    {h: 'q (ген.)', f: r => qtxt(r.gen_q)},
    {h: 'Мин. обнаружимый, %', f: r => r.mde === null ? '—' : fmt(100 * (Math.exp(r.mde) - 1), 1)},
    {h: 'Δ ln r [интервал]', f: r => r.delta === null || r.delta === undefined ? '—' : fmt(r.delta, 3) + ' <span class="muted">[' + fmt(r.delta_lo, 3) + '; ' + fmt(r.delta_hi, 3) + ']</span>'},
    {h: 'Итог', f: r => pill(D.status_class[r.status], D.status_ru[r.status])},
  ], rows);
}

// ----------------------------------------------------------------------------------------------- B
let setB = null, metricB = null;
const own = key => R.same_type[key].by_pool[R.same_type[key].own];
function drawBBars() {
  const res = own(setB), rows = res.table.filter(r => r.log_ratio !== undefined && r.log_ratio !== null);
  const idx = rows.map((_, i) => i), x = rows.map(r => pctOf(r.log_ratio));
  const shapes = rows.map((r, i) => ({type: 'rect', xref: 'x', yref: 'y', x0: pctOf(-r.delta), x1: pctOf(r.delta), y0: i - 0.42, y1: i + 0.42, fillcolor: P.brand_green, opacity: 0.22, line: {width: 0}}));
  Plotly.newPlot('plot-b-bars', [{type: 'scatter', mode: 'markers', x: x, y: idx, marker: {size: 10, color: rows.map(r => r.q_boot < 0.05 ? PINK : GREEN)},
    error_x: {type: 'data', symmetric: false, array: rows.map((r, i) => pctOf(r.d_hi) - x[i]), arrayminus: rows.map((r, i) => x[i] - pctOf(r.d_lo)), color: P.ink_secondary, thickness: 1.6, width: 4},
    hovertext: rows.map((r, i) => '<b>' + mname(r.metric) + '</b><br>разность медиан: ' + fp(x[i]) + ' [' + fp(pctOf(r.d_lo)) + '; ' + fp(pctOf(r.d_hi)) + ']<br>p = ' + qtxt(r.p_boot) + ', q = ' + qtxt(r.q_boot) +
      '<br>наименьшая граница, для которой доказана эквивалентность: ' + fmt(r.min_margin_sd, 2) + 'σ<br>' + D.verdict_ru[r.verdict]), hoverinfo: 'text'}],
    layout({shapes: shapes, yaxis: {tickvals: idx, ticktext: rows.map(r => mname(r.metric)), autorange: 'reversed'}, margin: {l: 150}, showlegend: false,
      xaxis: {title: axisTitle('разность медиан (набор − реальные), %; зелёная полоса — граница эквивалентности'), zeroline: true, zerolinecolor: P.ink_muted, zerolinewidth: 1.5}}), PLOT_CONFIG);
}
function drawBRef() {
  const res = own(setB), ref = res.reference[metricB], row = res.table.find(r => r.metric === metricB);
  if (!ref || !row) { noData('plot-b-ref'); return; }
  Plotly.newPlot('plot-b-ref', [{type: 'histogram', x: ref, marker: {color: GREY}, nbinsx: 40, hovertemplate: '%{x:.3f}<extra>реальные половины</extra>'}],
    layout({shapes: [lineShape(row.w1_sd, PINK), lineShape(row.ref_q95, P.ink_secondary, 'dash')], showlegend: false,
      xaxis: {title: axisTitle('W₁/σ: «' + mname(metricB) + '», реальная половина против другой половины')}, yaxis: {title: axisTitle('повторений')},
      annotations: [{x: row.w1_sd, y: 1, yref: 'paper', text: 'набор, p = ' + qtxt(row.p_floor), showarrow: false, xanchor: 'left', font: {color: PINK, size: 12}}]}), PLOT_CONFIG);
}
function drawBMv() {
  const res = own(setB), mv = res.mv;
  const hist = (id, ref, obs, q95, p, title) => {
    if (!ref || obs === undefined || obs === null) { noData(id, 'Мало полных векторов метрик'); return; }
    Plotly.newPlot(id, [{type: 'histogram', x: ref, marker: {color: GREY}, nbinsx: 35, hovertemplate: '%{x:.4f}<extra>шум</extra>'}],
      layout({shapes: [lineShape(obs, PINK), lineShape(q95, P.ink_secondary, 'dash')], showlegend: false, xaxis: {title: axisTitle(title)}, yaxis: {title: axisTitle('повторений')},
        annotations: [{x: obs, y: 1, yref: 'paper', text: 'p = ' + qtxt(p), showarrow: false, xanchor: 'left', font: {color: PINK, size: 12}}]}), PLOT_CONFIG);
  };
  hist('plot-b-ed', res.reference._ed, mv.ed, mv.ed_q95, mv.p_ed, 'энергетическое расстояние (полных векторов: ' + mv.n_complete + ')');
  hist('plot-b-corr', res.reference._corr, mv.corr_rms, mv.corr_q95, mv.p_corr, 'RMS-разность матриц Спирмена');
}
function tableB() {
  const res = own(setB);
  table('table-b', [
    {h: 'Метрика', f: r => esc(mname(r.metric)) + (r.base_independent ? ' <span class="muted" title="не зависит от поиска основания">●</span>' : '')},
    {h: 'n ген.', f: r => r.n_gen},
    {h: 'Медиана (ген. / реал.)', f: r => r.median_gen === undefined ? '—' : fmt(r.median_gen * MI[r.metric].factor) + ' / ' + fmt(r.median_real * MI[r.metric].factor) + (MI[r.metric].unit ? ' ' + MI[r.metric].unit : '')},
    {h: 'Разность медиан, % [95 % ИИ]', f: r => r.log_ratio === undefined ? '—' : fp(pctOf(r.log_ratio)) + ' <span class="muted">[' + fp(pctOf(r.d_lo)) + '; ' + fp(pctOf(r.d_hi)) + ']</span>'},
    {h: 'p (бутстрэп нейронов)', f: r => qtxt(r.p_boot)}, {h: 'q', f: r => qtxt(r.q_boot)},
    {h: 'Граница ±, % (0,2σ)', f: r => r.delta === undefined ? '—' : fmt(100 * (Math.exp(r.delta) - 1), 1)},
    {h: 'q TOST', f: r => qtxt(r.q_tost)},
    {h: 'Наименьшая граница, σ', f: r => r.min_margin_sd === undefined ? '—' : fmt(r.min_margin_sd, 2)},
    {h: 'Вердикт', f: r => pill(r.verdict === 'equivalent' ? 'good' : (r.verdict === 'differs' ? 'bad' : 'warn'), D.verdict_ru[r.verdict])},
    {h: 'W₁/σ (описательно)', f: r => r.w1_sd === undefined ? '—' : fmt(r.w1_sd, 3) + ' <span class="muted">(шум ' + fmt(r.ref_q50, 3) + ' / ' + fmt(r.ref_q95, 3) + ')</span>'},
    {h: 'в шуме реальных', f: r => r.within_noise === undefined ? '—' : (r.within_noise ? 'да' : 'нет')},
  ], res.table);
}
function drawBCross() {
  const pools = Object.keys(R.pools);
  const z = [], text = [], ann = [];
  D.sets.forEach((s, i) => {
    const rz = [], rt = [];
    pools.forEach((p, j) => {
      const r = R.same_type[s.key].by_pool[p];
      const mv = r ? r.mv : null;
      const v = mv && mv.ed !== undefined && mv.ed !== null ? mv.ed / mv.ed_q95 : null;
      rz.push(v === null ? null : Math.log10(v));
      rt.push(v === null ? '' : fmt(v, 2) + '<br>p = ' + qtxt(mv.p_ed));
    });
    z.push(rz); text.push(rt);
  });
  Plotly.newPlot('plot-b-cross', [{type: 'heatmap', x: pools.map(p => 'реальные: ' + (VR[p] || p)), y: D.sets.map(s => s.title), z: z, text: text, texttemplate: '%{text}', hoverinfo: 'skip',
    colorscale: THEME.sequential, zmin: -0.3, zmax: 1.5, xgap: 2, ygap: 2, colorbar: {title: {text: 'lg(расстояние / шум)', side: 'top'}, thickness: 12, len: 0.8}}],
    layout({margin: {l: 200, t: 30}, xaxis: {side: 'top', showgrid: false}, yaxis: {autorange: 'reversed', showgrid: false}}), PLOT_CONFIG);
}
function drawB() { drawBBars(); drawBRef(); drawBMv(); tableB(); }

// ----------------------------------------------------------------------------------------------- C
function drawC(key) {
  const c = (R.classifier || {})[key];
  if (!c) { noData('plot-c-frac', 'Классификатор не построен'); noData('plot-c-auc'); return; }
  const rows = Object.values(c.sets);
  const x = rows.map(r => setTitle(r.value, r.label));
  Plotly.newPlot('plot-c-frac', [
    {type: 'bar', name: 'набор', x: x, y: rows.map(r => 100 * r.frac_pred_a), marker: {color: rows.map(r => valueColor(r.value))},
      error_y: {type: 'data', symmetric: false, array: rows.map(r => 100 * (r.hi - r.frac_pred_a)), arrayminus: rows.map(r => 100 * (r.frac_pred_a - r.lo)), color: P.ink_secondary, thickness: 1.5},
      hovertemplate: '%{y:.1f} %<extra>%{x}</extra>'},
    {type: 'scatter', mode: 'markers', name: 'ожидание по реальным', x: rows.filter(r => r.expected !== undefined).map(r => setTitle(r.value, r.label)), y: rows.filter(r => r.expected !== undefined).map(r => 100 * r.expected),
      marker: {symbol: 'diamond', size: 13, color: '#ffffff', line: {color: P.ink, width: 2}},
      error_y: {type: 'data', symmetric: false, array: rows.filter(r => r.expected !== undefined).map(r => 100 * (r.expected_ci[1] - r.expected)), arrayminus: rows.filter(r => r.expected !== undefined).map(r => 100 * (r.expected - r.expected_ci[0])), color: P.ink, thickness: 1.5},
      hovertemplate: '%{y:.1f} %<extra>реальные</extra>'}],
    layout({yaxis: {title: axisTitle('названо тормозными, %'), range: [0, 100]}, legend: {orientation: 'h', y: 1.1}, xaxis: {tickangle: -20}, margin: {b: 90}}), PLOT_CONFIG);
  const pts = [{name: 'реальные', y: 'реальные', auc: c.auc, lo: c.auc_ci[0], hi: c.auc_ci[1], color: GREY}];
  Object.entries(c.gen_auc || {}).forEach(([lab, g]) => pts.push({name: 'сгенерированные ' + lab, y: 'сгенерированные' + (lab === 'default' ? '' : ', ' + lab), auc: g.auc, lo: g.ci[0], hi: g.ci[1], color: PINK}));
  Plotly.newPlot('plot-c-auc', [{type: 'scatter', mode: 'markers', x: pts.map(p => p.auc), y: pts.map(p => p.y), marker: {size: 11, color: pts.map(p => p.color)},
    error_x: {type: 'data', symmetric: false, array: pts.map(p => p.hi - p.auc), arrayminus: pts.map(p => p.auc - p.lo), color: P.ink_secondary, thickness: 1.6},
    hovertemplate: 'AUC %{x:.3f}<extra>%{y}</extra>'}],
    layout({xaxis: {title: axisTitle('AUC «тормозные против возбуждающих»'), range: [0.45, 1.0]}, yaxis: {autorange: 'reversed'}, margin: {l: 190}, showlegend: false,
      shapes: [{type: 'line', xref: 'x', yref: 'paper', x0: 0.5, x1: 0.5, y0: 0, y1: 1, line: {color: P.ink_muted, dash: 'dot', width: 1}}]}), PLOT_CONFIG);
  table('table-c', [
    {h: 'Набор', f: r => esc(setTitle(r.value, r.label))}, {h: 'n', f: r => r.n + (r.n_total > r.n ? ' <span class="muted">из ' + r.n_total + '</span>' : '')},
    {h: 'Названо тормозными, %', f: r => fmt(100 * r.frac_pred_a, 1) + ' <span class="muted">[' + fmt(100 * r.lo, 1) + '; ' + fmt(100 * r.hi, 1) + ']</span>'},
    {h: 'Ожидание по реальным, %', f: r => r.expected === undefined ? '—' : fmt(100 * r.expected, 1) + ' <span class="muted">[' + fmt(100 * r.expected_ci[0], 1) + '; ' + fmt(100 * r.expected_ci[1], 1) + ']</span>'},
    {h: 'Согласуется', f: r => r.within_expected === undefined ? '—' : pill(r.within_expected ? 'good' : 'bad', r.within_expected ? 'да' : 'нет')},
    {h: 'Эффективная доля тормозных, %', f: r => r.effective_share_a === undefined ? '—' : fmt(100 * r.effective_share_a, 0) + ' <span class="muted">[' + fmt(100 * r.effective_share_a_ci[0], 0) + '; ' + fmt(100 * r.effective_share_a_ci[1], 0) + ']</span>'},
  ], rows);
}

// ----------------------------------------------------------------------------------------------- D
function drawDAuc() {
  const rows = D.sets.filter(s => (R.discriminability || {})[s.key] && R.discriminability[s.key].logistic).map(s => ({title: s.title, d: R.discriminability[s.key].logistic}));
  if (!rows.length) { noData('plot-d-auc'); return; }
  const shapes = rows.length ? [{type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 0, y1: 0, line: {width: 0}}] : [];
  Plotly.newPlot('plot-d-auc', [
    {type: 'bar', name: 'шум реальных половин (95-й процентиль)', y: rows.map(r => r.title), x: rows.map(r => r.d.ref_q95), orientation: 'h', marker: {color: P.muted_points}, hovertemplate: '%{x:.3f}<extra>шум 95 %</extra>'},
    {type: 'scatter', mode: 'markers', name: 'набор', y: rows.map(r => r.title), x: rows.map(r => r.d.auc), marker: {size: 12, color: PINK},
      text: rows.map(r => 'AUC ' + fmt(r.d.auc, 3) + '<br>p = ' + qtxt(r.d.p)), hoverinfo: 'text'}],
    layout({xaxis: {title: axisTitle('AUC «реальный / сгенерированный»'), range: [0.4, 1.0]}, yaxis: {autorange: 'reversed'}, margin: {l: 200}, legend: {orientation: 'h', y: 1.1}, bargap: 0.5}), PLOT_CONFIG);
}
function drawD(key) {
  const d = ((R.discriminability || {})[key] || {}).logistic;
  if (!d) { noData('plot-d-ref', 'Мало полных векторов'); }
  else {
    Plotly.newPlot('plot-d-ref', [{type: 'histogram', x: d.ref, nbinsx: 25, marker: {color: GREY}, hovertemplate: '%{x:.3f}<extra>реальные половины</extra>'}],
      layout({shapes: [lineShape(d.auc, PINK), lineShape(d.ref_q95, P.ink_secondary, 'dash')], showlegend: false, xaxis: {title: axisTitle('AUC: реальная половина против другой')},
        annotations: [{x: d.auc, y: 1, yref: 'paper', text: 'набор, p = ' + qtxt(d.p), showarrow: false, xanchor: 'left', font: {color: PINK, size: 12}}]}), PLOT_CONFIG);
  }
  const res = (R.same_type[key] || {}).by_pool ? own(key) : null;
  const ref = res ? res.reference : null;
  const ids = M.map(m => m.name);
  const heat = (id, z, diverging) => {
    if (!z) { noData(id); return; }
    Plotly.newPlot(id, [{type: 'heatmap', x: ids, y: ids, z: z, zmin: -1, zmax: 1, colorscale: diverging ? THEME.diverging : THEME.diverging, xgap: 1, ygap: 1,
      hovertemplate: '%{y} × %{x}: %{z:.2f}<extra></extra>', colorbar: {thickness: 10, len: 0.8}}],
      layout({margin: {l: 110, b: 110, t: 10}, xaxis: {tickangle: -45, showgrid: false}, yaxis: {autorange: 'reversed', showgrid: false}}), PLOT_CONFIG);
  };
  heat('plot-d-real', ref ? ref._corr_real : null);
  heat('plot-d-gen', ref ? ref._corr_gen : null);
  const diff = ref && ref._corr_real && ref._corr_gen ? ref._corr_gen.map((row, i) => row.map((v, j) => v - ref._corr_real[i][j])) : null;
  heat('plot-d-diff', diff);
}

// ----------------------------------------------------------------------------------------------- E
function drawE(metric) {
  const labels = Object.keys(R.effects || {}).filter(l => D.weights[l] !== undefined && D.weights[l] !== null && isFinite(D.weights[l])).sort((a, b) => D.weights[a] - D.weights[b]);
  if (!labels.length) { noData('plot-e-metric'); return; }
  const ws = labels.map(l => D.weights[l]);
  const rows = labels.map(l => R.effects[l].generated.find(r => r.metric === metric));
  const real = (R.real_effect || []).find(r => r.metric === metric);
  const x = rows.map(r => r ? pctOf(r.log_ratio) : null);
  const shapes = real ? [{type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: pctOf(real.lr_lo), y1: pctOf(real.lr_hi), fillcolor: P.muted_points, opacity: 0.45, line: {width: 0}}] : [];
  Plotly.newPlot('plot-e-metric', [{type: 'scatter', mode: 'lines+markers', x: ws, y: x, marker: {color: PINK, size: 9}, line: {color: PINK},
    error_y: {type: 'data', symmetric: false, array: rows.map((r, i) => r ? pctOf(r.lr_hi) - x[i] : 0), arrayminus: rows.map((r, i) => r ? x[i] - pctOf(r.lr_lo) : 0), color: PINK, thickness: 1.5},
    hovertemplate: 'w = %{x}: %{y:.1f} %<extra>сгенерированные</extra>'}],
    layout({shapes: shapes, showlegend: false, xaxis: {title: axisTitle('вес guidance w'), tickvals: ws}, yaxis: {title: axisTitle('изменение медианы, %'), zeroline: true, zerolinecolor: P.ink_muted},
      annotations: real ? [{xref: 'paper', x: 0.99, yref: 'y', y: pctOf(real.lr_hi), text: 'серая полоса — реальный интервал (бутстрэп нейронов)', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: {size: 11, color: P.ink_muted}}] : []}), PLOT_CONFIG);
}
function drawEAux() {
  const dose = R.dose || [];
  if (!dose.length) { noData('plot-e-beta'); noData('plot-e-ed'); return; }
  const w = dose.map(d => d.w);
  Plotly.newPlot('plot-e-beta', [{type: 'scatter', mode: 'lines+markers', x: w, y: dose.map(d => d.beta), line: {color: PINK}, marker: {color: PINK, size: 9},
    error_y: {type: 'data', symmetric: false, array: dose.map(d => d.beta_hi - d.beta), arrayminus: dose.map(d => d.beta - d.beta_lo), color: PINK, thickness: 1.5}, hovertemplate: 'w = %{x}: β = %{y:.2f}<extra></extra>'}],
    layout({shapes: [{type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 1, y1: 1, line: {color: P.ink_secondary, dash: 'dash', width: 1.5}}], showlegend: false,
      xaxis: {title: axisTitle('вес guidance w'), tickvals: w}, yaxis: {title: axisTitle('амплитуда эффекта β (1 = реальная)'), zeroline: true}}), PLOT_CONFIG);
  const red = R.real_ed;
  const shapes = red ? [{type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: red[1], y1: red[2], fillcolor: P.muted_points, opacity: 0.45, line: {width: 0}},
    {type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: red[0], y1: red[0], line: {color: GREY, width: 2}}] : [];
  Plotly.newPlot('plot-e-ed', [{type: 'scatter', mode: 'lines+markers', x: w, y: dose.map(d => d.ed_gen), line: {color: PINK}, marker: {color: PINK, size: dose.map(d => d.closest ? 16 : 9), symbol: dose.map(d => d.closest ? 'diamond' : 'circle')},
    hovertemplate: 'w = %{x}: %{y:.3f}<extra>сгенерированные</extra>'}],
    layout({shapes: shapes, showlegend: false, xaxis: {title: axisTitle('вес guidance w'), tickvals: w}, yaxis: {title: axisTitle('энергетическое расстояние между типами')},
      annotations: red ? [{xref: 'paper', x: 0.99, yref: 'y', y: red[2], text: 'серая полоса — реальные типы (межквартильный интервал)', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: {size: 11, color: P.ink_muted}}] : []}), PLOT_CONFIG);
}

// ----------------------------------------------------------------------------------------------- F, G
function tableF() {
  const rate = (r, k) => r[k + '_rate'] === undefined ? '—' : fmt(100 * r[k + '_rate'], 1) + ' % <span class="muted">[' + fmt(100 * r[k + '_lo'], 1) + '; ' + fmt(100 * r[k + '_hi'], 1) + ']</span>';
  table('table-f', [
    {h: 'Набор', f: r => esc(setTitle(r.value, r.label))}, {h: 'Образцов', f: r => r.n},
    {h: 'Меш извлечён', f: r => rate(r, 'mesh')}, {h: 'Основание найдено', f: r => rate(r, 'base')}, {h: 'Меш валиден', f: r => rate(r, 'valid')},
    {h: 'Полный вектор метрик', f: r => rate(r, 'complete')},
  ], R.validity || []);
}
function tableG() {
  const uncond = D.sets.filter(s => s.value === 'unconditional');
  if (!uncond.length) { $('table-g').innerHTML = '<div class="empty">Набор без условия не задан</div>'; return; }
  const rows = [];
  uncond.forEach(s => Object.entries(R.same_type[s.key].by_pool).forEach(([p, r]) => rows.push({set: s.title, pool: p, mv: r.mv, n: r.n_real_spines, nn: r.n_real_neurons, t: r.table})));
  table('table-g', [
    {h: 'Набор', f: r => esc(r.set)}, {h: 'Реальные', f: r => esc(VR[r.pool] || r.pool) + ' <span class="muted">(' + r.nn + ' нейронов)</span>'},
    {h: 'Энергетическое расстояние', f: r => r.mv.ed === undefined ? '—' : fmt(r.mv.ed, 4)}, {h: 'Шум 95 %', f: r => r.mv.ed_q95 === undefined ? '—' : fmt(r.mv.ed_q95, 4)},
    {h: 'Расстояние / шум', f: r => r.mv.ed === undefined ? '—' : fmt(r.mv.ed / r.mv.ed_q95, 2)}, {h: 'p', f: r => qtxt(r.mv.p_ed)},
    {h: 'Метрик эквивалентно', f: r => r.t.filter(x => x.verdict === 'equivalent').length + ' из ' + r.t.length},
  ], rows);
  const closest = rows.filter(r => r.mv.ed !== undefined).sort((a, b) => a.mv.ed / a.mv.ed_q95 - b.mv.ed / b.mv.ed_q95)[0];
  const cb = R.class_balance;
  $('note-g').textContent = (closest ? 'Ближе всего набор без условия к реальным: ' + (VR[closest.pool] || closest.pool) + '. ' : '') +
    (cb ? 'Реальная доля тормозных: ' + fmt(100 * (cb.neurons.inhibitory || 0) / ((cb.neurons.inhibitory || 0) + (cb.neurons.excitatory || 0)), 1) + ' % нейронов, ' +
      fmt(100 * (cb.spines.inhibitory || 0) / ((cb.spines.inhibitory || 0) + (cb.spines.excitatory || 0)), 1) + ' % шипиков; эффективная доля тормозных в наборе без условия — в разделе C.' : '');
}

// ----------------------------------------------------------------------------------------------- wiring
fillSelect('sel-a', Object.keys(R.effects || {}).map(l => ({value: l, text: l === 'default' ? 'один набор' : (R.effects[l].pair.length && (R.pairs || []).length > 1 ? l : 'метка ' + l)})), drawA);
if (!Object.keys(R.effects || {}).length) { ['plot-forest'].forEach(id => noData(id, 'Нужны наборы обоих типов с одной меткой')); }
fillSelect('sel-b-set', D.sets.map(s => ({value: s.key, text: s.title + ' → реальные: ' + (VR[s.own] || s.own)})), v => { setB = v; if (metricB) drawB(); });
fillSelect('sel-b-metric', M.map(m => ({value: m.id, text: m.name})), v => { metricB = v; if (setB) drawB(); });
drawBCross();
fillSelect('sel-c', Object.keys(R.classifier || {}).map(k => ({value: k, text: k.replace('logistic/', 'логистическая: ').replace('gbm/', 'градиентный бустинг: ').replace('all', 'все метрики').replace('base_independent', 'метрики, не зависящие от основания')})), drawC);
if (!Object.keys(R.classifier || {}).length) { noData('plot-c-frac', 'Нужны наборы обоих типов и реальные данные обоих типов'); noData('plot-c-auc'); }
drawDAuc();
fillSelect('sel-d', D.sets.map(s => ({value: s.key, text: s.title})), drawD);
if ($('sel-e')) { fillSelect('sel-e', M.map(m => ({value: m.id, text: m.name})), drawE); drawEAux(); }
tableF();
tableG();

// ----------------------------------------------------------------------------------------------- C (detailed types)
function drawCMulti() {
  const t = R.type_classifier;
  if (!t || !$('plot-c-conf')) return;
  const cls = t.classes;
  Plotly.newPlot('plot-c-conf', [{type: 'heatmap', x: cls, y: cls.map((c, i) => c + ' (' + t.n_neurons[i] + ')'), z: t.confusion.map(r => r.map(v => 100 * v)), text: t.confusion.map(r => r.map(v => fmt(100 * v, 0))),
    texttemplate: '%{text}', hovertemplate: 'истинный %{y} → предсказан %{x}: %{z:.1f} %<extra></extra>', colorscale: THEME.sequential, zmin: 0, zmax: 100, xgap: 2, ygap: 2, colorbar: {title: {text: '%', side: 'top'}, thickness: 12, len: 0.8}}],
    layout({margin: {l: 130, b: 70, t: 20}, xaxis: {title: axisTitle('предсказанный тип'), tickangle: -30, showgrid: false}, yaxis: {autorange: 'reversed', showgrid: false}}), PLOT_CONFIG);
  const rows = Object.values(t.sets).filter(r => r.recovery !== undefined);
  const x = rows.map(r => setTitle(r.value, r.label));
  Plotly.newPlot('plot-c-recover', [
    {type: 'bar', name: 'набор', x: x, y: rows.map(r => 100 * r.recovery), marker: {color: PINK},
      error_y: {type: 'data', symmetric: false, array: rows.map(r => 100 * (r.hi - r.recovery)), arrayminus: rows.map(r => 100 * (r.recovery - r.lo)), color: P.ink_secondary, thickness: 1.5}, hovertemplate: '%{y:.1f} %<extra>%{x}</extra>'},
    {type: 'scatter', mode: 'markers', name: 'ожидание: доля верно названных у реальных', x: x, y: rows.map(r => 100 * r.expected), marker: {symbol: 'diamond', size: 13, color: '#ffffff', line: {color: P.ink, width: 2}},
      error_y: {type: 'data', symmetric: false, array: rows.map(r => 100 * (r.expected_ci[1] - r.expected)), arrayminus: rows.map(r => 100 * (r.expected - r.expected_ci[0])), color: P.ink, thickness: 1.5}}],
    layout({yaxis: {title: axisTitle('названо запрошенным типом, %'), range: [0, 100]}, legend: {orientation: 'h', y: 1.1}, xaxis: {tickangle: -20}, margin: {b: 90}}), PLOT_CONFIG);
  table('table-c-multi', [
    {h: 'Набор', f: r => esc(setTitle(r.value, r.label))}, {h: 'n', f: r => r.n},
    {h: 'Верно названо, %', f: r => r.recovery === undefined ? '—' : fmt(100 * r.recovery, 1) + ' <span class="muted">[' + fmt(100 * r.lo, 1) + '; ' + fmt(100 * r.hi, 1) + ']</span>'},
    {h: 'Ожидание по реальным, %', f: r => r.expected === undefined ? '—' : fmt(100 * r.expected, 1) + ' <span class="muted">[' + fmt(100 * r.expected_ci[0], 1) + '; ' + fmt(100 * r.expected_ci[1], 1) + ']</span>'},
    {h: 'Итог', f: r => r.within_expected === undefined ? '—' : pill(r.underpowered ? 'warn' : (r.within_expected ? 'good' : 'bad'), r.underpowered ? 'мало нейронов реального типа' : (r.within_expected ? 'как у реальных' : 'расходится'))},
    {h: 'Чаще всего названо', f: r => { const e = Object.entries(r.pred_share).sort((a, b) => b[1] - a[1])[0]; return esc(e[0]) + ' ' + fmt(100 * e[1], 0) + ' %'; }},
  ], Object.values(t.sets));
}
drawCMulti();
