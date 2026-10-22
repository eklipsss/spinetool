// Report of generated branches: page logic. Data: PAYLOAD (see scripts/neuron_model/build_branch_report.py), helpers: COMMON_JS.
const D = PAYLOAD, B = D.branches, state = {i: 0, centerline: false, anchors: false};
const GREY = P.mesh_real, PINK = P.brand_pink, GREEN = '#3f8a2a';
const um = v => (v === null || v === undefined) ? null : v;

// ---------------------------------------------------------------- viewer
function fillSelect() {
  const sel = $('branchSelect');
  B.forEach((b, k) => sel.add(new Option(`Ветвь ${b.index}: ${fmt(b.length_um, 1)} мкм, шипиков ${b.placed}`, k)));
  sel.onchange = () => { state.i = +sel.value; drawAll(); };
  $('showCenter').onchange = e => { state.centerline = e.target.checked; drawViewer(); };
  $('showAnchors').onchange = e => { state.anchors = e.target.checked; drawViewer(); };
}
function drawViewer() {
  const b = B[state.i], trunk = unpackMesh(b.mesh_trunk), traces = [meshTrace(trunk, GREY, 'ствол')], meshes = [trunk];
  if (b.mesh_spines) { const sp = unpackMesh(b.mesh_spines); meshes.push(sp); traces.push(meshTrace(sp, PINK, 'шипики')); }
  if (state.centerline) traces.push({type: 'scatter3d', mode: 'lines', x: b.center.x, y: b.center.y, z: b.center.z, line: {color: P.ink, width: 3}, name: 'центральная линия', hoverinfo: 'skip'});
  if (state.anchors) {
    const a = b.spines.filter(s => s.status === 'placed');
    traces.push({type: 'scatter3d', mode: 'markers', x: a.map(s => s.ax), y: a.map(s => s.ay), z: a.map(s => s.az), marker: {size: 4, color: GREEN}, name: 'основания шипиков',
      text: a.map(s => `шипик ${s.k}, s = ${fmt(s.s_um, 2)} мкм, погружение ${fmt(s.delta_nm, 0)} нм`), hoverinfo: 'text'});
  }
  const sc = scene3d(cubeRange(meshes), {aspectmode: 'data'});
  for (const a of ['xaxis', 'yaxis', 'zaxis']) { delete sc[a].range; sc[a].showticklabels = true; sc[a].title = {text: a[0] + ', мкм', font: {size: 11, color: P.ink_secondary}}; }
  Plotly.react('view', traces, layout({margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: 0.01}), scene: sc, uirevision: 'b' + state.i}), PLOT_CONFIG);
}
function drawStats() {
  const b = B[state.i];
  const rows = [['Статус', b.status === 'ok' ? 'собрана, все проверки пройдены' : 'собрана, есть замечания: ' + (b.reason || '—')],
    ['Длина, мкм', fmt(b.length_um, 1)], ['Медианный радиус, нм', fmt(b.radius_median_nm, 0)], ['Извилистость (путь / хорда)', fmt(b.tortuosity, 2)],
    ['Запрошено шипиков', b.placements], ['Поставлено', b.placed], ['Не удалось поставить', b.dropped], ['Плотность, шипиков на мкм', fmt(b.density, 2)],
    ['Замкнута', b.watertight ? 'да' : 'нет'], ['Компонент', b.n_components], ['Род поверхности', b.genus], ['Граней в сетке', b.n_faces === null ? '—' : b.n_faces.toLocaleString('ru-RU')],
    ['Площадь, мкм²', fmt(b.area_um2, 1)], ['Объём, мкм³', fmt(b.volume_um3, 2)], ['Время сборки, с', fmt(b.seconds, 1)]];
  $('branchStats').innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td style="text-align:right">${r[1]}</td></tr>`).join('');
  $('viewNote').textContent = b.shown_faces < b.n_faces ? `Для показа сетка упрощена до ${b.shown_faces.toLocaleString('ru-RU')} граней из ${b.n_faces.toLocaleString('ru-RU')}.` : '';
}
function drawProfile() {
  const b = B[state.i], placed = b.spines.filter(s => s.status === 'placed'), dropped = b.spines.filter(s => s.status !== 'placed');
  const traces = [{type: 'scatter', mode: 'lines', x: b.profile.s_um, y: b.profile.r_nm, line: {color: P.ink_secondary, width: 2}, name: 'радиус ствола', hovertemplate: 's = %{x:.1f} мкм<br>r = %{y:.0f} нм<extra></extra>'},
    {type: 'scatter', mode: 'markers', x: placed.map(s => s.s_um), y: placed.map(s => s.r_nm), marker: {color: GREEN, size: 8, line: {color: '#fff', width: 1}}, name: 'поставленные',
      text: placed.map(s => `шипик ${s.k}: погружение ${fmt(s.delta_nm, 0)} нм, угол ${fmt(s.theta_deg, 0)}°`), hoverinfo: 'text'}];
  if (dropped.length) traces.push({type: 'scatter', mode: 'markers', x: dropped.map(s => s.s_um), y: dropped.map(s => s.r_nm), marker: {color: PINK, size: 9, symbol: 'x'}, name: 'не удалось поставить',
    text: dropped.map(s => `шипик ${s.k}: ${s.reasons || '—'}`), hoverinfo: 'text'});
  Plotly.react('profile', traces, layout({showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: -0.25}), xaxis: {title: axisTitle('Положение вдоль ветви, мкм')}, yaxis: {title: axisTitle('Радиус ствола, нм')}, margin: {b: 90}}), PLOT_CONFIG);
}
function drawAll() { drawViewer(); drawStats(); drawProfile(); }

// ---------------------------------------------------------------- all branches
function hist(id, values, title, xtitle, opts) {
  opts = opts || {};
  const v = values.filter(x => x !== null && Number.isFinite(x));
  const traces = [{type: 'histogram', x: v, marker: {color: P.brand_pink}, nbinsx: opts.bins || 20, hovertemplate: '%{x}: %{y}<extra></extra>'}];
  const lay = {showlegend: false, bargap: 0.06, xaxis: {title: axisTitle(xtitle)}, yaxis: {title: axisTitle('Число')}, margin: {t: 20}};
  if (opts.uniform && v.length) {  // expected counts of a uniform distribution
    const lo = opts.range[0], hi = opts.range[1], nb = opts.bins; traces[0].xbins = {start: lo, end: hi, size: (hi - lo) / nb}; traces[0].autobinx = false;
    lay.shapes = [{type: 'line', xref: 'x', yref: 'y', x0: lo, x1: hi, y0: v.length / nb, y1: v.length / nb, line: {color: P.ink, width: 1.5, dash: 'dot'}}];
  }
  Plotly.react(id, traces, layout(lay), PLOT_CONFIG);
}
function bars(id, counts, color) {
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!keys.length) { $(id).innerHTML = '<div class="empty">Таких случаев не было</div>'; return; }
  Plotly.react(id, [{type: 'bar', orientation: 'h', y: keys.map(k => D.reason_names[k] || k), x: keys.map(k => counts[k]), marker: {color: color}, text: keys.map(k => counts[k]), textposition: 'outside',
    hovertemplate: '%{y}: %{x}<extra></extra>'}], layout({showlegend: false, yaxis: {autorange: 'reversed', automargin: true}, xaxis: {title: axisTitle('Число случаев')}, margin: {l: 20, t: 20, r: 40}}), PLOT_CONFIG);
}
function drawGlobal() {
  const t = D.table;
  Plotly.react('lenSpines', [{type: 'scatter', mode: 'markers', x: t.map(r => r.length_um), y: t.map(r => r.placed), marker: {color: P.brand_pink, size: 9, line: {color: '#fff', width: 1}},
    text: t.map(r => `ветвь ${r.index}: ${fmt(r.density, 2)} шипика на мкм`), hoverinfo: 'text'}], layout({showlegend: false, xaxis: {title: axisTitle('Длина ветви, мкм')}, yaxis: {title: axisTitle('Поставлено шипиков')}, margin: {t: 20}}), PLOT_CONFIG);
  hist('densityHist', t.map(r => r.density), '', 'Плотность, шипиков на мкм', {bins: 15});
  const s = D.placed_spines;
  hist('deltaHist', s.delta_nm, '', 'Глубина погружения шипика, нм', {bins: 25});
  hist('uHist', s.u, '', 'Положение вдоль ветви, доля длины (0 — начало, 1 — конец)', {bins: 10, uniform: true, range: [0, 1]});
  hist('thetaHist', s.theta_deg, '', 'Угол вокруг оси, градусы', {bins: 12, uniform: true, range: [0, 360]});
  bars('attemptReasons', D.failure_reasons, P.brand_pink);
  bars('poolReasons', D.pool.rejected_reasons, P.series_1 || P.brand_pink);
  const tb = $('tableBody');
  tb.innerHTML = t.map(r => `<tr><td>${r.index}</td><td>${fmt(r.length_um, 1)}</td><td>${r.placements}</td><td>${r.placed}</td><td>${r.dropped}</td><td>${fmt(r.density, 2)}</td><td>${r.watertight ? 'да' : 'нет'}</td><td>${r.n_components}</td><td>${fmt(r.seconds, 1)}</td><td>${r.status === 'ok' ? 'ок' : (r.reason || r.status)}</td></tr>`).join('');
}
function drawEvaluation() {
  const E = D.evaluation; if (!E) return;
  const names = E.metrics;
  const sel = $('evalMetric'); names.forEach(m => sel.add(new Option(m.name, m.key)));
  const draw = () => {
    const m = names.find(x => x.key === sel.value);
    const tr = ['real', 'generated'].map(kind => ({type: 'box', name: kind === 'real' ? 'реальные ветви' : 'собранные ветви', x: E.values[kind][m.key].filter(v => v !== null), orientation: 'h', boxpoints: 'outliers',
      marker: {color: kind === 'real' ? GREY : PINK, size: 3}, line: {color: kind === 'real' ? GREY : PINK, width: 1.5}, fillcolor: 'rgba(0,0,0,0)'}));
    Plotly.react('evalPlot', tr, layout({showlegend: false, xaxis: {title: axisTitle(m.name)}, yaxis: {autorange: 'reversed'}, margin: {l: 130, t: 20}}), PLOT_CONFIG);
    $('evalAbout').textContent = `Реальные ветви: ${E.values.real[m.key].filter(v => v !== null).length}, собранные: ${E.values.generated[m.key].filter(v => v !== null).length}. ` + (m.about || '');
  };
  sel.onchange = draw; draw();
}
fillSelect(); drawAll(); drawGlobal(); drawEvaluation();
