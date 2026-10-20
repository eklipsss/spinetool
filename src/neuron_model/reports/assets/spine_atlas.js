// Spine-generator atlas: page logic. Data: PAYLOAD (see scripts/neuron_model/build_spine_atlas.py), helpers: COMMON_JS.
const D = PAYLOAD, PTS = D.points, N = PTS.source.length;
const SRC = D.sources.map((s, i) => Object.assign({index: i}, s)), SRC_BY_ID = Object.fromEntries(SRC.map((s, i) => [s.id, Object.assign({index: i}, s)]));
const METRICS = Object.keys(D.metric_info), QUAL = Object.keys(D.quality_info);
const SYMBOLS = ['circle', 'diamond', 'triangle-up', 'square'];
// scattergl needs WebGL; without it (old GPU drivers, remote desktops) the map falls back to SVG
const SCATTER = (() => { try { const c = document.createElement('canvas'); return (c.getContext('webgl') || c.getContext('experimental-webgl')) ? 'scattergl' : 'scatter'; } catch (e) { return 'scatter'; } })();
const state = {map: Object.keys(PTS.maps)[0], color: 'source', enabled: Object.fromEntries(SRC.map(s => [s.id, true])),
  cellType: 'all', compartment: 'all', onlyValid: false, current: null, pinned: null, metric: METRICS[0],
  model: (SRC.find(s => s.kind === 'generated') || {}).id, tr: {base: 0, dir: 0, step: null, metric: METRICS[0]}};

const disp = (m, v) => (v === null || v === undefined) ? null : v * D.metric_info[m].factor;
const unit = m => D.metric_info[m].unit ? ', ' + D.metric_info[m].unit : '';
const realIdx = [...Array(N).keys()].filter(i => SRC[PTS.source[i]].kind === 'real');
const sortedReal = Object.fromEntries(METRICS.map(m => [m, realIdx.map(i => PTS.metrics[m][i]).filter(v => v !== null).sort((a, b) => a - b)]));
function percentile(m, v) {
  const a = sortedReal[m]; if (v === null || !a.length) return null;
  let lo = 0, hi = a.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
  return 100 * lo / a.length;
}

// ---------------------------------------------------------------- sidebar controls
function buildControls() {
  const mapSel = $('mapSelect');
  for (const [k, v] of Object.entries(PTS.maps)) mapSel.add(new Option(v.name, k));
  const colorSel = $('colorSelect');
  colorSel.add(new Option('Источник', 'source'));
  colorSel.add(new Option('Тип клетки (возб. / торм.)', 'ctb'));
  colorSel.add(new Option('Компартмент', 'compartment'));
  const g1 = el('optgroup', {label: 'Морфометрика'});
  for (const m of METRICS) g1.append(new Option(D.metric_info[m].name + unit(m), 'm:' + m));
  const g2 = el('optgroup', {label: 'Качество поверхности'});
  for (const q of QUAL) g2.append(new Option(D.quality_info[q].name, 'q:' + q));
  colorSel.append(g1, g2);
  const box = $('sources');
  for (const s of SRC) {
    const cb = el('input', {type: 'checkbox', id: 'src_' + s.id}); cb.checked = true;
    cb.addEventListener('change', () => { state.enabled[s.id] = cb.checked; drawMap(); });
    const lab = el('label', {class: 'check', for: 'src_' + s.id}, [cb, swatch(s.color), document.createTextNode(s.name), el('span', {class: 'count', text: s.n.toLocaleString('ru-RU')})]);
    box.append(lab);
  }
  const ct = $('cellTypeSelect'), cp = $('compartmentSelect');
  ct.add(new Option('Все', 'all')); cp.add(new Option('Все', 'all'));
  for (const v of D.categories.cell_type) ct.add(new Option(v, v));
  for (const v of D.categories.compartment) cp.add(new Option(v, v));
  mapSel.onchange = () => { state.map = mapSel.value; drawMap(true); };
  colorSel.onchange = () => { state.color = colorSel.value; drawMap(); };
  ct.onchange = () => { state.cellType = ct.value; drawMap(); };
  cp.onchange = () => { state.compartment = cp.value; drawMap(); };
  $('onlyValid').onchange = e => { state.onlyValid = e.target.checked; drawMap(); };
}

function visible(i) {
  const s = SRC[PTS.source[i]];
  if (!state.enabled[s.id] || PTS.maps[state.map].xy[i] === null) return false;
  if (s.kind === 'real') {
    if (state.cellType !== 'all' && PTS.cell_type[i] !== state.cellType) return false;
    if (state.compartment !== 'all' && PTS.compartment[i] !== state.compartment) return false;
  } else if (state.onlyValid && PTS.valid[i] === false) return false;
  return true;
}

// ---------------------------------------------------------------- map
const CAT_COLORS = {ctb: {excitatory: P.series_1, inhibitory: P.series_2}, compartment: {apical: P.series_1, basal: P.series_2, uncertain: P.series_3}};
const CAT_NAMES = {excitatory: 'возбуждающие', inhibitory: 'тормозные', apical: 'апикальные', basal: 'базальные', uncertain: 'не определён'};
function pointColor(i) {
  if (state.color === 'source') return SRC[PTS.source[i]].color;
  if (state.color === 'ctb' || state.color === 'compartment') {
    const v = state.color === 'ctb' ? PTS.ctb[i] : PTS.compartment[i];
    return CAT_COLORS[state.color][v] || P.muted_points;
  }
  return null;
}
function colorValue(i) {
  const [kind, name] = state.color.split(':');
  if (kind === 'm') { const v = disp(name, PTS.metrics[name][i]); return (v !== null && D.metric_info[name].log && v > 0) ? Math.log10(v) : v; }
  if (kind === 'q') return PTS.quality[name][i];
  return null;
}
function hover(i) {
  const s = SRC[PTS.source[i]];
  const parts = [`<b>${s.name}</b>`, PTS.key[i]];
  if (s.kind === 'real') parts.push(`${PTS.cell_type[i] || 'тип неизвестен'} · ${PTS.compartment[i] || 'компартмент неизвестен'}`);
  else parts.push(PTS.valid[i] === false ? 'сетка невалидна' : 'сетка валидна');
  for (const m of ['Length', 'Volume']) if (PTS.metrics[m]) parts.push(`${D.metric_info[m].name}: ${fmt(disp(m, PTS.metrics[m][i]))}${unit(m).replace(', ', ' ')}`);
  if (PTS.mesh[i] >= 0) parts.push('<i>есть 3D-сетка</i>');
  return parts.join('<br>');
}
function drawMap(resetView) {
  const ids = [...Array(N).keys()].filter(visible);
  const xy = PTS.maps[state.map].xy, continuous = state.color.includes(':');
  const traces = [];
  for (const s of SRC) {
    const sid = ids.filter(i => PTS.source[i] === s.index);
    if (!sid.length) continue;
    const marker = {size: s.kind === 'real' ? 5 : 6, symbol: SYMBOLS[s.index % SYMBOLS.length], opacity: s.kind === 'real' ? 0.55 : 0.8,
      line: {width: 0}};
    if (continuous) {
      marker.color = sid.map(colorValue); marker.colorscale = THEME.sequential;
      const all = ids.map(colorValue).filter(v => v !== null && Number.isFinite(v)).sort((a, b) => a - b);
      marker.cmin = all[Math.floor(all.length * 0.02)]; marker.cmax = all[Math.floor(all.length * 0.98)];
      if (traces.length === 0) {
        const [kind, name] = state.color.split(':');
        const t = kind === 'm' ? D.metric_info[name].name + (D.metric_info[name].log ? ', lg' : '') + unit(name).replace(', ', ' ') : D.quality_info[name].name;
        marker.colorbar = {title: {text: t, side: 'right', font: {size: 11, color: P.ink_secondary}}, thickness: 10, len: 0.7, outlinewidth: 0, tickfont: {size: 10, color: P.ink_muted}};
        marker.showscale = true;
      }
    } else marker.color = sid.map(pointColor);
    traces.push({type: SCATTER, mode: 'markers', name: s.name, x: sid.map(i => xy[i][0]), y: sid.map(i => xy[i][1]),
      customdata: sid, text: sid.map(hover), hovertemplate: '%{text}<extra></extra>', marker: marker, showlegend: false});
  }
  for (const [which, idx, color] of [['current', state.current, P.ink], ['pinned', state.pinned, P.series_2]]) {
    if (idx === null || xy[idx] === null) continue;
    traces.push({type: 'scatter', mode: 'markers', x: [xy[idx][0]], y: [xy[idx][1]], hoverinfo: 'skip', showlegend: false,
      marker: {size: 16, color: 'rgba(0,0,0,0)', line: {width: 2.5, color: color}}});
  }
  const lay = layout({margin: {l: 36, r: 12, t: 8, b: 36}, uirevision: resetView ? Math.random() : state.map,
    xaxis: {title: axisTitle(PTS.maps[state.map].axis + ' 1'), showticklabels: false, zeroline: false},
    yaxis: {title: axisTitle(PTS.maps[state.map].axis + ' 2'), showticklabels: false, zeroline: false, scaleanchor: 'x'}});
  Plotly.react('map', traces, lay, PLOT_CONFIG);
  mapLegend(ids);
}
function mapLegend(ids) {
  const box = $('mapLegend'); box.replaceChildren();
  const add = (color, text, symbol) => box.append(el('span', {}, [swatch(color), document.createTextNode(text)]));
  if (state.color === 'source') for (const s of SRC) { const n = ids.filter(i => PTS.source[i] === s.index).length; if (n) add(s.color, `${s.name}: ${n.toLocaleString('ru-RU')}`); }
  else if (state.color === 'ctb' || state.color === 'compartment') {
    for (const [k, c] of Object.entries(CAT_COLORS[state.color])) add(c, CAT_NAMES[k] || k);
    add(P.muted_points, 'нет данных (в том числе все сгенерированные)');
  }
  const sym = SRC.map((s, i) => `${['●', '◆', '▲', '■'][i % 4]} ${s.name}`).join('   ');
  box.append(el('span', {class: 'muted', text: 'Форма маркера: ' + sym}));
  $('mapCounts').textContent = `На карте ${ids.length.toLocaleString('ru-RU')} шипиков. Нажмите на точку, чтобы открыть её сетку и морфометрику.`;
}

// ---------------------------------------------------------------- viewer
function nearestWithMesh(i) {
  const xy = PTS.maps[state.map].xy; let best = null, bd = Infinity;
  for (let j = 0; j < N; j++) {
    if (PTS.mesh[j] < 0 || PTS.source[j] !== PTS.source[i] || xy[j] === null) continue;
    const d = (xy[j][0] - xy[i][0]) ** 2 + (xy[j][1] - xy[i][1]) ** 2; if (d < bd) { bd = d; best = j; }
  }
  return best;
}
function select(i) {
  state.current = i; drawMap(); drawViewer(); drawMetricTable();
}
function drawViewer() {
  const box = $('viewerNote'), shown = [];
  for (const [idx, label] of [[state.current, 'выбранный'], [state.pinned, 'закреплённый']]) {
    if (idx === null) continue;
    let j = idx, note = '';
    if (PTS.mesh[idx] < 0) { j = nearestWithMesh(idx); note = j === null ? '' : ' (сетка ближайшего на карте шипика того же источника)'; }
    if (j === null) continue;
    shown.push({idx: j, label: label, note: note});
  }
  if (!shown.length) { Plotly.purge('viewer'); $('viewer').innerHTML = '<div class="empty">Выберите точку на карте</div>'; box.textContent = ''; return; }
  $('viewer').innerHTML = '';
  const meshes = shown.map(s => unpackMesh(D.meshes[PTS.mesh[s.idx]]));
  const range = cubeRange(meshes);
  const traces = [], lay = layout({margin: {l: 0, r: 0, t: 26, b: 0}, showlegend: false, annotations: []});
  shown.forEach((s, k) => {
    const src = SRC[PTS.source[s.idx]], scene = k === 0 ? 'scene' : 'scene2';
    traces.push(Object.assign(meshTrace(meshes[k], src.kind === 'real' ? P.mesh_real : P.brand_pink, src.name), {scene: scene}));
    lay[scene] = scene3d(range, {domain: {x: shown.length === 1 ? [0, 1] : [k / 2 + 0.01, (k + 1) / 2 - 0.01], y: [0, 1]}});
    lay.annotations.push({text: `${src.name} · ${s.label}${s.note ? '*' : ''}`, x: shown.length === 1 ? 0.5 : (k + 0.5) / 2, y: 1.04, xref: 'paper', yref: 'paper', showarrow: false, font: {size: 12, color: P.ink_secondary}});
  });
  Plotly.react('viewer', traces, lay, PLOT_CONFIG);
  syncCameras('viewer', shown.length);
  box.textContent = shown.map(s => s.note ? `* ${s.label}: ${PTS.key[s.idx]}${s.note}` : '').filter(Boolean).join(' ');
}
function syncCameras(id, n) {
  const gd = $(id); if (n < 2 || gd._synced) return; gd._synced = true; let busy = false;
  gd.on('plotly_relayout', ev => {
    if (busy) return; const k = Object.keys(ev).find(x => /^scene2?\.camera$/.test(x)); if (!k) return;
    busy = true; Plotly.relayout(gd, {[k === 'scene.camera' ? 'scene2.camera' : 'scene.camera']: ev[k]}).finally(() => busy = false);
  });
}
function drawMetricTable() {
  const cols = [state.current, state.pinned].filter(i => i !== null);
  const t = $('metricTable');
  if (!cols.length) { t.innerHTML = ''; return; }
  const head = '<tr><th>Признак</th>' + cols.map(i => `<th>${SRC[PTS.source[i]].name}</th><th>перцентиль среди реальных</th>`).join('') + '</tr>';
  const rows = METRICS.map(m => '<tr><td title="' + D.metric_info[m].about + '">' + D.metric_info[m].name + unit(m) + '</td>' + cols.map(i => {
    const v = PTS.metrics[m][i], p = percentile(m, v);
    const bar = p === null ? '' : `<div class="bar" style="width:90px;display:inline-block;vertical-align:middle;margin-left:8px"><i style="width:${p.toFixed(0)}%;background:${(p < 5 || p > 95) ? P.bad : P.series_1}"></i></div>`;
    return `<td>${fmt(disp(m, v))}</td><td>${p === null ? '—' : p.toFixed(0)}${bar}</td>`;
  }).join('') + '</tr>');
  const qrows = QUAL.map(q => '<tr><td title="' + D.quality_info[q].about + '">' + D.quality_info[q].name + (D.quality_info[q].unit ? ', ' + D.quality_info[q].unit : '') + '</td>' + cols.map(i => `<td>${fmt(PTS.quality[q][i])}</td><td></td>`).join('') + '</tr>');
  t.innerHTML = head + rows.join('') + qrows.join('');
  $('keyLine').textContent = cols.map(i => PTS.key[i]).join('  |  ');
}

// ---------------------------------------------------------------- morphometric distributions
function quantile(a, q) { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]; }
function drawDistribution() {
  const m = state.metric, info = D.metric_info[m];
  const tr = v => (v !== null && info.log) ? (v > 0 ? Math.log10(v) : null) : v;
  const bySrc = SRC.map(s => [...Array(N).keys()].filter(i => PTS.source[i] === s.index).map(i => tr(disp(m, PTS.metrics[m][i]))).filter(v => v !== null && Number.isFinite(v)));
  const ref = bySrc[SRC.findIndex(s => s.kind === 'real')] || bySrc.flat();
  const lo = quantile(ref, 0.005), hi = quantile(ref, 0.995), nb = 40, w = (hi - lo) / nb;
  const traces = [];
  SRC.forEach((s, k) => {
    const v = bySrc[k]; if (!v.length || !(w > 0)) return;
    const counts = new Array(nb).fill(0); let out = 0;
    for (const x of v) { const b = Math.floor((x - lo) / w); if (b >= 0 && b < nb) counts[b]++; else out++; }
    const dens = counts.map(c => c / v.length / w);
    const xs = counts.map((_, b) => lo + (b + 0.5) * w);
    traces.push({type: 'scatter', mode: 'lines', line: {shape: 'hvh', width: s.kind === 'real' ? 0 : 2, color: s.color}, fill: s.kind === 'real' ? 'tozeroy' : 'none',
      fillcolor: s.kind === 'real' ? 'rgba(142,145,150,0.25)' : undefined, x: xs, y: dens, name: `${s.name} (n=${v.length}${out ? ', вне диапазона ' + out : ''})`,
      hovertemplate: `${s.name}<br>%{x:.3g}: плотность %{y:.3g}<extra></extra>`});
  });
  const xt = info.name + (info.log ? ' (log₁₀' + (info.unit ? ', ' + info.unit : '') + ')' : unit(m));
  Plotly.react('dist', traces, layout({showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: -0.22}),
    xaxis: {title: axisTitle(xt)}, yaxis: {title: axisTitle('Плотность')}, margin: {b: 70}}), PLOT_CONFIG);
  $('distAbout').textContent = `${info.name}: ${info.about}.`;
}
function drawCompareTable() {
  const gens = SRC.filter(s => s.kind === 'generated');
  const t = $('compareTable');
  if (!gens.length) { t.innerHTML = '<tr><td>Сгенерированных наборов нет.</td></tr>'; return; }
  const pill = v => v === null ? '—' : `<span class="pill ${v < 0.2 ? 'good' : v < 0.5 ? 'warn' : 'bad'}">${fmt(v, 2)}</span>`;
  let h = '<tr><th>Признак</th>' + gens.map(s => `<th>${s.name}: W₁/σ</th><th>KS</th><th>медиана, ген. / реал.</th>`).join('') + '</tr>';
  for (const m of METRICS) {
    h += `<tr><td>${D.metric_info[m].name}${unit(m)}</td>` + gens.map(s => {
      const c = (D.comparison[s.id] || {}).metrics?.[m] || {};
      return `<td>${pill(c.w1_rel ?? null)}</td><td>${fmt(c.ks ?? null, 2)}</td><td>${fmt(disp(m, c.median_gen ?? null))} / ${fmt(disp(m, c.median_real ?? null))}</td>`;
    }).join('') + '</tr>';
  }
  t.innerHTML = h;
}
function drawCorr() {
  const s = state.model, c = s && D.comparison[s] && D.comparison[s].corr;
  if (!c) { $('corr').innerHTML = '<div class="empty">Нет данных для корреляций</div>'; return; }
  const names = METRICS.map(m => D.metric_info[m].name);
  const delta = c.gen.map((r, a) => r.map((v, b) => v - c.real[a][b]));
  const hm = (z, scene, title, zmin, zmax, cs) => ({type: 'heatmap', z: z, x: names, y: names, xaxis: scene[0], yaxis: scene[1], colorscale: cs, zmin: zmin, zmax: zmax, zmid: 0,
    showscale: scene[0] !== 'x', colorbar: {thickness: 10, len: 0.8, x: 1.0, outlinewidth: 0, tickfont: {size: 10, color: P.ink_muted}, title: {text: 'R / ΔR', side: 'right', font: {size: 11, color: P.ink_secondary}}},
    hovertemplate: `${title}<br>%{y} × %{x}: %{z:.2f}<extra></extra>`, xgap: 1, ygap: 1});
  const lay = layout({margin: {l: 130, r: 30, t: 30, b: 110},
    xaxis: {domain: [0, 0.43], tickangle: -45, tickfont: {size: 10}}, yaxis: {autorange: 'reversed', tickfont: {size: 10}},
    xaxis2: {domain: [0.55, 0.98], tickangle: -45, tickfont: {size: 10}, anchor: 'y2'}, yaxis2: {autorange: 'reversed', anchor: 'x2', showticklabels: false},
    annotations: [{text: 'R, реальные', x: 0.215, y: 1.0, yanchor: 'bottom', xref: 'paper', yref: 'paper', showarrow: false, font: {size: 11, color: P.ink_secondary}},
      {text: `ΔR, ${SRC_BY_ID[s].name} − реальные`, x: 0.765, y: 1.0, yanchor: 'bottom', xref: 'paper', yref: 'paper', showarrow: false, font: {size: 11, color: P.ink_secondary}}]});
  Plotly.react('corr', [hm(c.real, ['x', 'y'], 'Реальные', -1, 1, THEME.diverging), hm(delta, ['x2', 'y2'], 'Разность', -1, 1, THEME.diverging)], lay, PLOT_CONFIG);
}

// ---------------------------------------------------------------- quality
function drawQuality() {
  const keys = [['valid', 'Валидная сетка'], ['watertight', 'Замкнутая'], ['single', 'Одна компонента'], ['attachment', 'Основание найдено']];
  const traces = SRC.map(s => {
    const q = D.quality_rates[s.id] || {};
    return {type: 'bar', orientation: 'h', name: s.name, y: keys.map(k => k[1]), x: keys.map(k => q[k[0]] === undefined || q[k[0]] === null ? null : 100 * q[k[0]]),
      marker: {color: s.color}, text: keys.map(k => q[k[0]] === undefined || q[k[0]] === null ? 'нет данных' : (100 * q[k[0]]).toFixed(0) + ' %'),
      textposition: 'outside', textfont: {size: 11, color: P.ink_secondary}, hovertemplate: `${s.name}<br>%{y}: %{x:.1f} %<extra></extra>`};
  });
  Plotly.react('qualityRates', traces, layout({barmode: 'group', bargap: 0.35, bargroupgap: 0.12, showlegend: true,
    legend: Object.assign({}, THEME.legend, {orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0}), xaxis: {range: [0, 112], title: axisTitle('Доля, %')}, yaxis: {autorange: 'reversed'}, margin: {l: 130, t: 30}}), PLOT_CONFIG);
  const q = $('qualitySelect').value;
  const box = SRC.map(s => {
    const v = [...Array(N).keys()].filter(i => PTS.source[i] === s.index).map(i => PTS.quality[q][i]).filter(x => x !== null && Number.isFinite(x));
    return {type: 'box', name: s.name, x: v, marker: {color: s.color, size: 3}, line: {color: s.color, width: 1.5}, fillcolor: 'rgba(0,0,0,0)', boxpoints: 'outliers', orientation: 'h',
      hovertemplate: `${s.name}: %{x:.3g}<extra></extra>`};
  });
  Plotly.react('qualityBox', box, layout({showlegend: false, xaxis: {title: axisTitle(D.quality_info[q].name + (D.quality_info[q].unit ? ', ' + D.quality_info[q].unit : ''))}, yaxis: {autorange: 'reversed'}, margin: {l: 130}}), PLOT_CONFIG);
  $('qualityAbout').textContent = `${D.quality_info[q].name}: ${D.quality_info[q].about}.`;
}

// ---------------------------------------------------------------- latent traversal
function drawTraversal() {
  const T = D.traversal;
  if (!T || !T.base.length) { $('travView').innerHTML = '<div class="empty">Проходы по латентным направлениям не построены</div>'; return; }
  const s = state.tr, frames = T.frames[s.base][s.dir], steps = T.steps;
  if (s.step === null) s.step = steps.indexOf(0) >= 0 ? steps.indexOf(0) : Math.floor(steps.length / 2);
  $('travStep').max = steps.length - 1; $('travStep').value = s.step;
  const f = frames[s.step];
  $('travStepText').textContent = `${steps[s.step] > 0 ? '+' : ''}${fmt(steps[s.step], 1)} σ`;
  if (f.mesh < 0) { $('travView').innerHTML = '<div class="empty">Для этого шага поверхность не извлечена</div>'; }
  else {
    $('travView').innerHTML = '';
    const all = frames.filter(x => x.mesh >= 0).map(x => unpackMesh(D.meshes[x.mesh]));
    const m = unpackMesh(D.meshes[f.mesh]);
    const lay = layout({margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false, uirevision: 'trav', scene: scene3d(cubeRange(all), {uirevision: 'trav'})});
    Plotly.react('travView', [meshTrace(m, f.valid ? P.brand_pink : P.warning, 'шаг')], lay, PLOT_CONFIG);
  }
  const met = s.metric, info = D.metric_info[met];
  const y = frames.map(x => disp(met, x.metrics[met] ?? null));
  const r = sortedReal[met].map(v => disp(met, v));
  const p10 = quantile(r, 0.1), p90 = quantile(r, 0.9), p50 = quantile(r, 0.5);
  const traces = [];
  if (p10 !== null) traces.push({type: 'scatter', x: [steps[0], steps[steps.length - 1], steps[steps.length - 1], steps[0]], y: [p10, p10, p90, p90], fill: 'toself',
    fillcolor: 'rgba(142,145,150,0.16)', line: {width: 0}, hoverinfo: 'skip', name: 'реальные шипики, 10–90 %'});
  if (p50 !== null) traces.push({type: 'scatter', mode: 'lines', x: [steps[0], steps[steps.length - 1]], y: [p50, p50], line: {color: P.reference, width: 1, dash: 'dot'}, hoverinfo: 'skip', name: 'медиана реальных'});
  traces.push({type: 'scatter', mode: 'lines+markers', x: steps, y: y, line: {color: P.series_1, width: 2}, marker: {size: 8, color: frames.map(x => x.valid ? P.series_1 : P.warning)},
    name: info.name, hovertemplate: '%{x:+.1f} σ: %{y:.3g}<extra></extra>'});
  traces.push({type: 'scatter', mode: 'markers', x: [steps[s.step]], y: [y[s.step]], marker: {size: 15, color: 'rgba(0,0,0,0)', line: {width: 2.5, color: P.ink}}, hoverinfo: 'skip', showlegend: false});
  Plotly.react('travCurve', traces, layout({showlegend: true, legend: Object.assign({}, THEME.legend, {orientation: 'h', y: -0.25}),
    xaxis: {title: axisTitle('Сдвиг вдоль направления, σ'), zeroline: false}, yaxis: {title: axisTitle(info.name + unit(met))}, margin: {b: 80}}), PLOT_CONFIG);
  const d = T.directions[s.dir];
  $('travAbout').textContent = `Направление ${s.dir + 1} объясняет ${(100 * d.explained).toFixed(0)} % разброса латентных кодов реальных шипиков. Базовый шипик: ${T.base[s.base]}. Оранжевым отмечены шаги, где сетка невалидна.`;
}
function buildTraversalControls() {
  const T = D.traversal; if (!T) return;
  const b = $('travBase'), d = $('travDir'), m = $('travMetric');
  T.base.forEach((k, i) => b.add(new Option(`Шипик ${i + 1}: ${k}`, i)));
  T.directions.forEach((x, i) => d.add(new Option(`Направление ${i + 1} (${(100 * x.explained).toFixed(0)} % разброса)`, i)));
  for (const k of METRICS) m.add(new Option(D.metric_info[k].name, k));
  b.onchange = () => { state.tr.base = +b.value; drawTraversal(); };
  d.onchange = () => { state.tr.dir = +d.value; drawTraversal(); };
  m.onchange = () => { state.tr.metric = m.value; drawTraversal(); };
  $('travStep').oninput = e => { state.tr.step = +e.target.value; drawTraversal(); };
  let timer = null;
  $('travPlay').onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $('travPlay').textContent = 'Пройти'; return; }
    $('travPlay').textContent = 'Остановить'; let dir = 1;
    timer = setInterval(() => { const n = T.steps.length; let k = state.tr.step + dir; if (k >= n || k < 0) { dir = -dir; k = state.tr.step + dir; } state.tr.step = k; drawTraversal(); }, 650);
  };
}

// ---------------------------------------------------------------- init
buildControls();
const msel = $('metricSelect'); for (const m of METRICS) msel.add(new Option(D.metric_info[m].name + unit(m), m));
msel.onchange = () => { state.metric = msel.value; drawDistribution(); };
const csel = $('corrModel'); for (const s of SRC.filter(s => s.kind === 'generated')) csel.add(new Option(s.name, s.id));
csel.onchange = () => { state.model = csel.value; drawCorr(); };
const qsel = $('qualitySelect'); for (const q of QUAL) qsel.add(new Option(D.quality_info[q].name, q));
qsel.onchange = drawQuality;
$('pin').onclick = () => { if (state.current === null) return; state.pinned = state.current; drawMap(); drawViewer(); drawMetricTable(); };
$('unpin').onclick = () => { state.pinned = null; $('viewer')._synced = false; drawMap(); drawViewer(); drawMetricTable(); };
buildTraversalControls();
state.current = (() => { const i = [...Array(N).keys()].find(j => PTS.mesh[j] >= 0 && SRC[PTS.source[j]].kind === 'real'); return i === undefined ? null : i; })();
drawMap(true); drawMetricTable(); drawViewer(); drawDistribution(); drawCompareTable(); drawCorr(); drawQuality(); drawTraversal();
$('map').on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p && p.customdata !== undefined) select(p.customdata); });
