// Report of generated branches: page logic. Data: PAYLOAD (see scripts/neuron_model/build_branch_report.py), helpers: COMMON_JS.
const D = PAYLOAD, B = D.branches, state = {i: 0, centerline: false, anchors: false};
const GREY = P.mesh_real, PINK = P.brand_pink, GREEN = '#3f8a2a';
const um = v => (v === null || v === undefined) ? null : v;

// ---------------------------------------------------------------- viewer
state.view = 'gen';
function fillSelect() {
  const sel = $('branchSelect');
  B.forEach((b, k) => sel.add(new Option(`Ветвь ${b.index}: ${fmt(b.length_um, 1)} мкм, шипиков ${b.placed}`, k)));
  sel.onchange = () => { state.i = +sel.value; drawAll(); };
  $('showCenter').onchange = e => { state.centerline = e.target.checked; drawViewer(); };
  $('showAnchors').onchange = e => { state.anchors = e.target.checked; drawViewer(); };
  for (const r of document.querySelectorAll('input[name=viewMode]')) r.onchange = e => { state.view = e.target.value; drawViewer(); };
}
function generatedTraces(b, scene) {
  const trunk = unpackMesh(b.mesh_trunk), meshes = [trunk], traces = [Object.assign(meshTrace(trunk, GREY, 'ствол'), {scene: scene})];
  if (b.mesh_spines) { const sp = unpackMesh(b.mesh_spines); meshes.push(sp); traces.push(Object.assign(meshTrace(sp, PINK, 'шипики'), {scene: scene})); }
  if (state.centerline) traces.push({type: 'scatter3d', mode: 'lines', x: b.center.x, y: b.center.y, z: b.center.z, line: {color: P.ink, width: 3}, name: 'центральная линия', hoverinfo: 'skip', scene: scene});
  if (state.anchors) {
    const a = b.spines.filter(s => s.status === 'placed');
    traces.push({type: 'scatter3d', mode: 'markers', x: a.map(s => s.ax), y: a.map(s => s.ay), z: a.map(s => s.az), marker: {size: 4, color: GREEN}, name: 'основания шипиков', scene: scene,
      text: a.map(s => `шипик ${s.k}, s = ${fmt(s.s_um, 2)} мкм, погружение ${fmt(s.delta_nm, 0)} нм`), hoverinfo: 'text'});
  }
  return {traces: traces, meshes: meshes};
}
function realTraces(b, scene) {
  const m = unpackMesh(b.real.mesh), sp = b.real.spines;
  const traces = [Object.assign(meshTrace(m, GREY, 'ствол реальной ветви (круглая трубка по измеренной оси)', 0.9), {scene: scene}),
    {type: 'scatter3d', mode: 'markers', x: sp.map(s => s.ax), y: sp.map(s => s.ay), z: sp.map(s => s.az), marker: {size: 5, color: PINK, symbol: 'diamond'}, name: 'места крепления реальных шипиков', scene: scene,
     text: sp.map((s, k) => `шипик ${k + 1}, s = ${fmt(s.s_um, 2)} мкм`), hoverinfo: 'text'}];
  return {traces: traces, meshes: [m]};
}
function sceneFor(meshes, domain) {
  const sc = scene3d(cubeRange(meshes), {aspectmode: 'data', domain: domain});
  if (domain) sc.camera = {eye: {x: 0.95, y: 0.7, z: 0.7}, up: {x: 0, y: 1, z: 0}};  // two stacked scenes: come closer
  for (const a of ['xaxis', 'yaxis', 'zaxis']) { delete sc[a].range; sc[a].showticklabels = true; sc[a].title = {text: a[0] + ', мкм', font: {size: 11, color: P.ink_secondary}}; }
  return sc;
}
function syncCameras() {
  const gd = $('view'); if (gd._synced) return; gd._synced = true; let busy = false;
  gd.on('plotly_relayout', ev => {
    if (busy) return; const k = Object.keys(ev).find(x => /^scene2?\.camera$/.test(x)); if (!k) return;
    busy = true; Plotly.relayout(gd, {[k === 'scene.camera' ? 'scene2.camera' : 'scene.camera']: ev[k]}).finally(() => busy = false);
  });
}
function drawViewer() {
  const b = B[state.i], hasReal = !!b.real, mode = hasReal ? state.view : 'gen';
  $('viewModeBox').style.display = hasReal ? '' : 'none';
  let traces = [], lay = {margin: {l: 0, r: 0, t: 18, b: 0}, showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: 0.01}), uirevision: 'b' + state.i + mode, annotations: []};
  if (mode === 'gen') { const g = generatedTraces(b, 'scene'); traces = g.traces; lay.scene = sceneFor(g.meshes); }
  else if (mode === 'real') { const r = realTraces(b, 'scene'); traces = r.traces; lay.scene = sceneFor(r.meshes); }
  else {
    const g = generatedTraces(b, 'scene'), r = realTraces(b, 'scene2'); traces = g.traces.concat(r.traces);
    lay.scene = sceneFor(g.meshes, {x: [0, 1], y: [0.52, 1]}); lay.scene2 = sceneFor(r.meshes, {x: [0, 1], y: [0, 0.48]});
    lay.annotations = [{text: 'Собранная ветвь', x: 0.01, y: 1.0, xref: 'paper', yref: 'paper', showarrow: false, xanchor: 'left', font: {size: 12, color: P.ink_secondary}},
      {text: 'Реальная ветвь с теми же условиями', x: 0.01, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false, xanchor: 'left', font: {size: 12, color: P.ink_secondary}}];
  }
  $('view').style.height = mode === 'both' ? '900px' : '';
  Plotly.react('view', traces, layout(lay), PLOT_CONFIG);
  if (mode === 'both') syncCameras();
  $('viewNote').textContent = (b.shown_faces < b.n_faces ? `Сетка собранной ветви упрощена для показа до ${b.shown_faces.toLocaleString('ru-RU')} граней из ${b.n_faces.toLocaleString('ru-RU')}. ` : '')
    + (hasReal ? 'У реальной ветви показан ствол в виде круглой трубки по измеренной оси и радиусу, а шипики отмечены ромбами в местах крепления: сетки реальных шипиков здесь не показаны. ' : 'Реальная ветвь с теми же условиями недоступна для этого набора. ');
}
function drawConditions() {
  const b = B[state.i];
  $('condTable').innerHTML = (b.conditions || []).map(r => `<tr><td>${r[0]}</td><td style="text-align:right">${r[1]}</td></tr>`).join('') || '<tr><td>Условия набора недоступны</td></tr>';
  $('realSource').textContent = b.real ? `Реальная ветвь сравнения: нейрон ${b.real.neuron}, ветвь ${b.real.key.split('/').slice(2).join('/')}.` : '';
}
function drawStats() {
  const b = B[state.i], g = b.stats, r = b.real ? b.real.stats : null;
  const row = (name, gv, rv, d) => `<tr><td>${name}</td><td style="text-align:right">${gv === null || gv === undefined ? '—' : fmt(gv, d)}</td><td style="text-align:right">${!r ? '—' : (rv === null || rv === undefined ? '—' : fmt(rv, d))}</td></tr>`;
  const rows = [['Длина, мкм', g.length_um, r && r.length_um, 1], ['Средний радиус, нм', g.radius_mean_nm, r && r.radius_mean_nm, 0], ['Разброс радиуса (CV)', g.radius_cv, r && r.radius_cv, 3],
    ['Извилистость (путь / хорда)', g.tortuosity, r && r.tortuosity, 2], ['Кривизна, среднеквадратичная, 1/нм', g.kappa_rms, r && r.kappa_rms, 5],
    ['Число шипиков', g.n_spines, r && r.n_spines, 0], ['Плотность, шипиков на мкм', g.density, r && r.density, 2], ['Медианный интервал между шипиками, нм', g.gap_median_nm, r && r.gap_median_nm, 0]];
  const fixed = [['Статус сборки', b.status === 'ok' ? 'все проверки пройдены' : 'есть замечания: ' + (b.reason || '—')], ['Запрошено шипиков', b.placements], ['Не удалось поставить', b.dropped],
    ['Замкнута', b.watertight ? 'да' : 'нет'], ['Компонент', b.n_components], ['Род поверхности', b.genus], ['Граней в сетке', b.n_faces === null ? '—' : b.n_faces.toLocaleString('ru-RU')],
    ['Площадь, мкм²', fmt(b.area_um2, 1)], ['Объём, мкм³', fmt(b.volume_um3, 2)], ['Время сборки, с', fmt(b.seconds, 1)]];
  $('branchStats').innerHTML = '<tr><th></th><th style="text-align:right">Собранная</th><th style="text-align:right">Реальная</th></tr>'
    + rows.map(x => row(x[0], x[1], x[2], x[3])).join('') + fixed.map(x => `<tr><td>${x[0]}</td><td style="text-align:right" colspan="2">${x[1]}</td></tr>`).join('');
  $('statsNote').textContent = b.real && !b.real.eligible_for_spines ? 'У реальной ветви число шипиков не определено: она не годится для обучения модели P.' : '';
}
function drawProfile() {
  const b = B[state.i], placed = b.spines.filter(s => s.status === 'placed'), dropped = b.spines.filter(s => s.status !== 'placed');
  const traces = [{type: 'scatter', mode: 'lines', x: b.profile.s_um, y: b.profile.r_nm, line: {color: P.ink_secondary, width: 2}, name: 'радиус собранной ветви', hovertemplate: 's = %{x:.1f} мкм<br>r = %{y:.0f} нм<extra></extra>'},
    {type: 'scatter', mode: 'markers', x: placed.map(s => s.s_um), y: placed.map(s => s.r_nm), marker: {color: GREEN, size: 8, line: {color: '#fff', width: 1}}, name: 'поставленные',
      text: placed.map(s => `шипик ${s.k}: погружение ${fmt(s.delta_nm, 0)} нм, угол ${fmt(s.theta_deg, 0)}°`), hoverinfo: 'text'}];
  if (dropped.length) traces.push({type: 'scatter', mode: 'markers', x: dropped.map(s => s.s_um), y: dropped.map(s => s.r_nm), marker: {color: PINK, size: 9, symbol: 'x'}, name: 'не удалось поставить',
    text: dropped.map(s => `шипик ${s.k}: ${s.reasons || '—'}`), hoverinfo: 'text'});
  if (b.real) {
    traces.push({type: 'scatter', mode: 'lines', x: b.real.profile.s_um, y: b.real.profile.r_nm, line: {color: GREY, width: 2, dash: 'dot'}, name: 'радиус реальной ветви', hoverinfo: 'skip'},
      {type: 'scatter', mode: 'markers', x: b.real.spines.map(s => s.s_um), y: b.real.spines.map(s => s.r_nm), marker: {color: 'rgba(0,0,0,0)', size: 9, symbol: 'diamond', line: {color: PINK, width: 2}}, name: 'шипики реальной ветви',
        text: b.real.spines.map((s, k) => `реальный шипик ${k + 1}, угол ${fmt(s.theta_deg, 0)}°`), hoverinfo: 'text'});
  }
  Plotly.react('profile', traces, layout({showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: -0.28}), xaxis: {title: axisTitle('Положение вдоль ветви, мкм')}, yaxis: {title: axisTitle('Радиус ствола, нм')}, margin: {b: 100}}), PLOT_CONFIG);
}
function drawAll() { drawViewer(); drawConditions(); drawStats(); drawProfile(); }

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
  bars('poolReasons', D.pool.rejected_reasons, P.brand_pink);
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
