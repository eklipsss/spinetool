// Neuron compartment viewer (scripts/neuron_model/view_neuron_compartments.py). Uses COMMON_JS of html_kit only.
// Coordinates are um relative to the soma centre (neurons) or to the branch centroid (close-ups).

const LABEL_RU = {apical: 'апикальный', basal: 'базальный', axon_like: 'аксон', uncertain: 'не определён',
  inhibitory_dendrite: 'дендрит тормозного нейрона', not_applicable: 'нет метки'};
// one fixed colour per compartment (palette roles only): apical pink, basal / interneuron dendrite green (never in the
// same neuron), axon violet, uncertain neutral grey, no label light grey
const LABEL_COLOR = {apical: P.series_1, basal: P.series_2, inhibitory_dendrite: P.series_2, axon_like: P.series_3,
  uncertain: P.reference, not_applicable: P.muted_points};
const LABEL_ORDER = ['apical', 'basal', 'inhibitory_dendrite', 'axon_like', 'uncertain', 'not_applicable'];
const KIND_RU = ['пресинапс (аксон)', 'постсинапс на стволе', 'постсинапс на шипике', 'постсинапс на соме'];
const state = {neuron: 0, mode: 'ours', showPre: true, showPost: true, showArrows: true, example: 0};

function branchGroups(n, keyOf) {
  const xyz = unpackArray(n.xyz), groups = new Map();
  for (let b = 0; b < n.branch_limb.length; b++) {
    const key = keyOf(b);
    if (!groups.has(key)) groups.set(key, {x: [], y: [], z: [], text: [], c: []});
    const g = groups.get(key);
    const r = n.branch_radius[b];
    const label = `лимб ${n.branch_limb[b]}, ветвь ${n.branch_id[b]}<br>радиус ветви ${r === null ? '—' : fmt(r) + ' нм'}`;
    for (let i = n.offsets[b]; i < n.offsets[b + 1]; i++) {
      g.x.push(xyz[3*i]); g.y.push(xyz[3*i+1]); g.z.push(xyz[3*i+2]); g.text.push(label); g.c.push(r === null ? NaN : r);
    }
    g.x.push(null); g.y.push(null); g.z.push(null); g.text.push(''); g.c.push(NaN);
  }
  return groups;
}

function neuronTraces(n) {
  const traces = [];
  const limbLabel = {};
  for (const l of n.limbs) limbLabel[l.limb] = state.mode === 'table' ? (l.table || 'not_applicable') : l.ours;
  if (state.mode === 'radius') {
    const g = branchGroups(n, () => 'all').get('all');
    traces.push({type: 'scatter3d', mode: 'lines', x: g.x, y: g.y, z: g.z, text: g.text, hovertemplate: '%{text}<extra></extra>', name: 'радиус ветви',
      line: {width: 4, color: g.c, colorscale: THEME.sequential, cmin: 80, cmax: 450,
             colorbar: {title: {text: 'Радиус, нм', font: {size: 12}}, thickness: 12, len: 0.6}}, showlegend: false});
  } else {
    const groups = branchGroups(n, b => limbLabel[n.branch_limb[b]] || 'not_applicable');
    for (const key of LABEL_ORDER) {
      if (!groups.has(key)) continue;
      const g = groups.get(key);
      traces.push({type: 'scatter3d', mode: 'lines', x: g.x, y: g.y, z: g.z, text: g.text, name: LABEL_RU[key],
        hovertemplate: `${LABEL_RU[key]}<br>%{text}<extra></extra>`, line: {width: key === 'not_applicable' ? 2 : 4, color: LABEL_COLOR[key]}});
    }
  }
  traces.push(Object.assign(meshTrace(unpackMesh(n.soma), P.mesh_real, 'сома', 1), {showlegend: true}));
  if (state.showArrows) {
    for (const a of n.arrows) {
      const lab = limbLabel[a.limb] || 'not_applicable';
      traces.push({type: 'scatter3d', mode: 'lines', x: [a.from[0], a.to[0]], y: [a.from[1], a.to[1]], z: [a.from[2], a.to[2]], showlegend: false,
        line: {width: 9, color: state.mode === 'radius' ? P.ink_secondary : LABEL_COLOR[lab]}, hovertemplate: `начальное направление лимба ${a.limb}<extra></extra>`});
    }
  }
  const pia = n.pia.map(v => v * 80);
  traces.push({type: 'scatter3d', mode: 'lines+text', x: [0, pia[0]], y: [0, pia[1]], z: [0, pia[2]], text: ['', 'к пиа'],
    textfont: {color: P.ink, size: 12}, textposition: 'top center', line: {width: 5, color: P.ink, dash: 'dash'}, name: 'направление на пиа (80 мкм)'});
  const pxyz = unpackArray(n.points.xyz);
  const sel = k => {
    const o = {x: [], y: [], z: [], text: []};
    n.points.kind.forEach((kind, i) => { if (k(kind)) { o.x.push(pxyz[3*i]); o.y.push(pxyz[3*i+1]); o.z.push(pxyz[3*i+2]); o.text.push(KIND_RU[kind] + (n.points.limb.length ? `, лимб ${n.points.limb[i]}` : '')); } });
    return o;
  };
  if (state.showPre) {
    const o = sel(k => k === 0);
    traces.push({type: 'scatter3d', mode: 'markers', x: o.x, y: o.y, z: o.z, text: o.text, hovertemplate: '%{text}<extra></extra>',
      name: `пресинапсы (${o.x.length})`, marker: {size: 2.6, symbol: 'diamond', color: P.ink}});
  }
  if (state.showPost) {
    const o = sel(k => k >= 1);
    traces.push({type: 'scatter3d', mode: 'markers', x: o.x, y: o.y, z: o.z, text: o.text, hovertemplate: '%{text}<extra></extra>',
      name: `постсинапсы (${o.x.length})`, marker: {size: 3.2, symbol: 'circle-open', color: P.ink_secondary, line: {width: 1.5}}});
  }
  return traces;
}

function drawNeuron() {
  const n = PAYLOAD.neurons[state.neuron];
  const scene = {aspectmode: 'data', xaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid},
    yaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid},
    zaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid},
    camera: {eye: {x: 0, y: 0, z: 2.2}, up: {x: 0, y: -1, z: 0}}};
  Plotly.react('plot-neuron', neuronTraces(n), layout({scene: scene, margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: n.neuron_id,
    legend: Object.assign({}, THEME.legend, {x: 0.01, y: 0.99})}), PLOT_CONFIG);
  $('neuron-why').textContent = n.why || '';
  $('neuron-title').textContent = `${n.neuron_id} · ${n.cell_type || 'тип неизвестен'}`;
  const tb = $('limb-table');
  tb.innerHTML = '';
  const head = el('tr', {}, ['Лимб', 'Наша разметка', 'Таблица коллеги', 'Радиус ветвей, нм', 'Кабель, мм', 'Ветвей', 'cos к пиа', 'Пресинапсов', 'Постсинапсов']
    .map(t => el('th', {text: t})));
  tb.append(head);
  for (const l of n.limbs) {
    const lab = key => { const s = el('span', {class: 'row'}, [swatch(LABEL_COLOR[key] || P.muted_points), document.createTextNode(LABEL_RU[key] || '—')]); s.style.display = 'inline-flex'; return s; };
    tb.append(el('tr', {}, [el('td', {text: String(l.limb)}), el('td', {}, [lab(l.ours)]), el('td', {}, [l.table ? lab(l.table) : document.createTextNode('нет в таблице')]),
      el('td', {text: fmt(l.radius_nm)}), el('td', {text: fmt(l.cable_mm)}), el('td', {text: fmt(l.n_branches)}), el('td', {text: fmt(l.cos, 2)}),
      el('td', {text: String(l.pre)}), el('td', {text: String(l.post)})]));
  }
}

function drawExample() {
  const ex = PAYLOAD.examples[state.example];
  if (!ex) return;
  const traces = [Object.assign(meshTrace(unpackMesh(ex.mesh), P.mesh_real, 'сетка ветви', 0.45), {showlegend: true})];
  ex.spines.forEach((m, i) => traces.push(Object.assign(meshTrace(unpackMesh(m), P.brand_pink, 'шипики сегментации', 1),
    {showlegend: i === 0, legendgroup: 'spines'})));
  const pxyz = unpackArray(ex.points.xyz);
  for (const [cond, name, marker] of [[k => k === 0, 'пресинапсы', {size: 3.5, symbol: 'diamond', color: P.ink}],
                                        [k => k >= 1, 'постсинапсы', {size: 4.5, symbol: 'circle-open', color: P.ink_secondary, line: {width: 2}}]]) {
    const o = {x: [], y: [], z: []};
    ex.points.kind.forEach((k, i) => { if (cond(k)) { o.x.push(pxyz[3*i]); o.y.push(pxyz[3*i+1]); o.z.push(pxyz[3*i+2]); } });
    if (o.x.length) traces.push({type: 'scatter3d', mode: 'markers', x: o.x, y: o.y, z: o.z, name: `${name} (${o.x.length})`, marker: marker, hoverinfo: 'skip'});
  }
  const scene = {aspectmode: 'data', xaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid},
    yaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid},
    zaxis: {showticklabels: false, title: {text: ''}, showbackground: false, gridcolor: P.grid}};
  Plotly.react('plot-example', traces, layout({scene: scene, margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: `${ex.neuron_id}-${ex.limb}-${ex.branch}`,
    legend: Object.assign({}, THEME.legend, {x: 0.01, y: 0.99})}), PLOT_CONFIG);
  const i = ex.info || {};
  $('example-why').textContent = `${ex.title}. Нейрон ${ex.neuron_id} (${i.cell_type || 'тип неизвестен'}), лимб ${ex.limb}, ветвь ${ex.branch}: ` +
    `радиус объединённой ветви ${i.radius_nm === null || i.radius_nm === undefined ? '—' : fmt(i.radius_nm) + ' нм'}, файлов шипиков ${ex.n_spine_files}` +
    (ex.spines.length < ex.n_spine_files ? ` (показано ${ex.spines.length})` : '') + `, разметка лимба: ${LABEL_RU[i.label] || '—'}.`;
}

function setup() {
  const sel = $('neuron-select');
  PAYLOAD.neurons.forEach((n, i) => sel.append(el('option', {value: String(i), text: `${n.group} · ${n.neuron_id} · ${n.cell_type || '?'}`})));
  sel.addEventListener('change', () => { state.neuron = +sel.value; drawNeuron(); });
  for (const b of document.querySelectorAll('#mode-seg button')) {
    b.addEventListener('click', () => {
      state.mode = b.dataset.mode;
      for (const o of document.querySelectorAll('#mode-seg button')) o.setAttribute('aria-pressed', String(o === b));
      drawNeuron();
    });
  }
  for (const [id, key] of [['show-pre', 'showPre'], ['show-post', 'showPost'], ['show-arrows', 'showArrows']]) {
    $(id).addEventListener('change', e => { state[key] = e.target.checked; drawNeuron(); });
  }
  const leg = $('label-legend');
  for (const key of LABEL_ORDER) leg.append(el('span', {class: 'row'}, [swatch(LABEL_COLOR[key]), document.createTextNode(LABEL_RU[key])]));
  const es = $('example-select');
  PAYLOAD.examples.forEach((e, i) => es.append(el('option', {value: String(i), text: `${e.title} · ${e.neuron_id} / лимб ${e.limb} / ветвь ${e.branch}`})));
  es.addEventListener('change', () => { state.example = +es.value; drawExample(); });
  if (PAYLOAD.neurons.length) drawNeuron();
  if (PAYLOAD.examples.length) drawExample(); else $('sec-examples').style.display = 'none';
}
setup();
