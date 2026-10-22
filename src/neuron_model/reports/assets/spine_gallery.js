// Gallery of spines: one row per source (real first), one column per real spine with its nearest valid generated neighbours.
// Runs inside the atlas page script (uses PAYLOAD-derived globals D, PTS, SRC, N, P, select, fmt, disp). No WebGL: each mesh is
// drawn on a 2-D canvas (orthographic view, painter's algorithm, Lambert shading), so 24+ meshes do not exhaust browser GL contexts.
(function () {
  const root = $('gallery'); if (!root) return;
  const map = PTS.maps.latent || PTS.maps[Object.keys(PTS.maps)[0]], XY = map.xy;
  const rows = [...SRC].sort((a, b) => (a.kind === 'real' ? 0 : 1) - (b.kind === 'real' ? 0 : 1));
  const G = {k: 5, angle: 30, seed: 1, cols: null};
  const W = 360, H = 270, PAD = 12, LIGHT = (() => { const v = [-0.4, 0.6, 0.7], n = Math.hypot(...v); return v.map(x => x / n); })();
  const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const rgbOf = css => {
    const c = document.createElement('canvas').getContext('2d'); c.fillStyle = css; const v = c.fillStyle;
    if (v[0] === '#') return [1, 3, 5].map(i => parseInt(v.slice(i, i + 2), 16));
    const m = v.match(/[\d.]+/g); return [+m[0], +m[1], +m[2]];
  };
  const d2 = (a, b) => (XY[a][0] - XY[b][0]) ** 2 + (XY[a][1] - XY[b][1]) ** 2;
  const candidates = src => { const c = []; for (let i = 0; i < N; i++) if (PTS.source[i] === src.index && PTS.mesh[i] >= 0 && XY[i] && PTS.valid[i] !== false) c.push(i); return c; };
  const realSrc = rows.find(s => s.kind === 'real'), genSrc = rows.filter(s => s.kind !== 'real');

  // k real spines far apart on the latent map (from a random pool); then, per column, the nearest unused valid generated spine of every model
  function newColumns() {
    const rand = rng(G.seed), c = candidates(realSrc);
    for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; }
    const pool = c.slice(0, 60), anchors = pool.length ? [pool[0]] : [];
    const d = pool.map(i => d2(i, pool[0]));
    while (anchors.length < Math.min(G.k, pool.length)) {
      let b = 0; for (let t = 1; t < pool.length; t++) if (d[t] > d[b]) b = t;
      anchors.push(pool[b]); for (let t = 0; t < pool.length; t++) d[t] = Math.min(d[t], d2(pool[t], pool[b]));
    }
    const used = new Set(), cols = anchors.map(a => ({real: a, gen: {}}));
    for (const s of genSrc) {
      const cand = candidates(s), taken = new Set();
      for (const col of cols) {
        let best = null; for (const i of cand) if (!taken.has(i) && (best === null || d2(i, col.real) < d2(best, col.real))) best = i;
        col.gen[s.id] = best; if (best !== null) taken.add(best);
      }
    }
    G.cols = cols;
  }

  function draw() {
    const rowsData = rows.map(src => ({src: src, ids: G.cols.map(c => src.kind === 'real' ? c.real : c.gen[src.id])}));
    const meshes = {};
    let rxz = 1e-9, ymin = Infinity, ymax = -Infinity;
    for (const r of rowsData) for (const i of r.ids) {
      if (i === null || i === undefined) continue;
      const m = unpackMesh(D.meshes[PTS.mesh[i]]); meshes[i] = m;
      for (let t = 0; t < m.x.length; t++) { rxz = Math.max(rxz, Math.hypot(m.x[t], m.z[t])); ymin = Math.min(ymin, m.y[t]); ymax = Math.max(ymax, m.y[t]); }
    }
    const s = Math.min((W / 2 - PAD) / rxz, (H - 2 * PAD) / Math.max(ymax - ymin, 1e-9));
    const a = G.angle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    root.replaceChildren();
    for (const r of rowsData) {
      const base = rgbOf(r.src.kind === 'real' ? P.mesh_real : r.src.color);
      root.append(el('div', {class: 'gal-label'}, [swatch(r.src.color), el('b', {text: r.src.name})]));
      const line = el('div', {class: 'gal-row'});
      for (const i of r.ids) {
        if (i === null || i === undefined) { line.append(el('div', {class: 'gal-fig'})); continue; }
        const m = meshes[i], cv = el('canvas', {width: W, height: H, class: 'gal-cell', title: 'Нажмите, чтобы открыть в блоке «Выбранный шипик»'}), ctx = cv.getContext('2d');
        const nv = m.x.length, X = new Float32Array(nv), Y = new Float32Array(nv), Z = new Float32Array(nv);
        for (let t = 0; t < nv; t++) { X[t] = m.x[t] * ca + m.z[t] * sa; Y[t] = m.y[t]; Z[t] = -m.x[t] * sa + m.z[t] * ca; }
        const nf = m.i.length, order = new Array(nf), depth = new Float32Array(nf);
        for (let f = 0; f < nf; f++) { order[f] = f; depth[f] = (Z[m.i[f]] + Z[m.j[f]] + Z[m.k[f]]) / 3; }
        order.sort((p, q) => depth[p] - depth[q]);           // far first, near last (camera on the +z side)
        ctx.lineJoin = 'round'; ctx.lineWidth = 0.7;
        for (const f of order) {
          const p = m.i[f], q = m.j[f], u = m.k[f];
          const ex = X[q] - X[p], ey = Y[q] - Y[p], ez = Z[q] - Z[p], fx = X[u] - X[p], fy = Y[u] - Y[p], fz = Z[u] - Z[p];
          const nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx, nl = Math.hypot(nx, ny, nz) || 1;
          const k = 0.32 + 0.68 * Math.abs((nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / nl);
          const col = `rgb(${Math.round(base[0] * k)},${Math.round(base[1] * k)},${Math.round(base[2] * k)})`;
          ctx.fillStyle = col; ctx.strokeStyle = col; ctx.beginPath();
          ctx.moveTo(W / 2 + X[p] * s, H - PAD - (Y[p] - ymin) * s); ctx.lineTo(W / 2 + X[q] * s, H - PAD - (Y[q] - ymin) * s); ctx.lineTo(W / 2 + X[u] * s, H - PAD - (Y[u] - ymin) * s);
          ctx.closePath(); ctx.fill(); ctx.stroke();
        }
        cv.addEventListener('click', () => { select(i); $('viewer').scrollIntoView({behavior: 'smooth', block: 'center'}); });
        const len = PTS.metrics.Length ? disp('Length', PTS.metrics.Length[i]) : null;
        line.append(el('figure', {class: 'gal-fig'}, [cv, el('figcaption', {class: 'small muted', text: len === null ? '' : `длина ${fmt(len, 2)} ${D.metric_info.Length.unit}`})]));
      }
      root.append(line);
    }
  }

  const ang = $('galAngle'), btn = $('galNew');
  ang.value = String(G.angle); $('galAngleText').textContent = G.angle + '°';
  ang.oninput = () => { G.angle = +ang.value; $('galAngleText').textContent = G.angle + '°'; draw(); };
  btn.onclick = () => { G.seed += 1; newColumns(); draw(); };
  newColumns(); draw();
})();
