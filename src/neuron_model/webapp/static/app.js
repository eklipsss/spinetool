// Page of the branch generator: form from /api/options, POST /api/generate, Three.js viewers, metric tables,
// comparison with real quantiles (Plotly), downloads. The API is in nm; the page shows um (factor of each metric).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = { branch: 0xfd89c3, spine: 0x7fbc41, highlight: 0x3f8a2a, anchor: 0xd6418f, local: 0xfd89c3 };
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };
const $ = (sel) => document.querySelector(sel);
const state = { options: null, result: null, branch: null, spine: null, selected: null, anchors: null, center: null };

// ---------------------------------------------------------------- helpers
function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return v.toLocaleString("ru-RU");
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2).replace(".", ",");
  return Number(v.toPrecision(digits)).toLocaleString("ru-RU", { maximumFractionDigits: 6 });
}
function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c !== null && c !== undefined) e.append(c);
  return e;
}
function label(field, value) {
  const l = state.options.labels[field] || {};
  return l[value] || value;
}
function showNotice(node, items) {
  node.innerHTML = "";
  if (!items || !items.length) { node.classList.add("hidden"); return; }
  node.append(el("ul", {}, items.map((t) => el("li", { text: t }))));
  node.classList.remove("hidden");
}

// ---------------------------------------------------------------- OFF parser
function parseOFF(text) {
  const tokens = text.replace(/#[^\n]*/g, " ").trim().split(/\s+/);
  let i = 0;
  const head = tokens[i++];
  if (!head.startsWith("OFF")) throw new Error("не OFF-файл");
  if (head.length > 3) tokens.splice(i, 0, head.slice(3)); // "OFF8 12 0" variant
  const nv = parseInt(tokens[i++], 10), nf = parseInt(tokens[i++], 10);
  i++; // edges
  const pos = new Float32Array(nv * 3);
  for (let k = 0; k < nv * 3; k++) pos[k] = parseFloat(tokens[i++]);
  const idx = [];
  for (let f = 0; f < nf; f++) {
    const n = parseInt(tokens[i++], 10);
    const poly = [];
    for (let k = 0; k < n; k++) poly.push(parseInt(tokens[i++], 10));
    for (let k = 1; k + 1 < n; k++) idx.push(poly[0], poly[k], poly[k + 1]);
  }
  return { positions: pos, indices: nv > 65535 ? new Uint32Array(idx) : new Uint16Array(idx) };
}

// nm -> um around ``center`` (nm): large absolute coordinates lose precision in float32 otherwise
function geometryFrom(off, center) {
  const p = off.positions.slice();
  for (let k = 0; k < p.length; k += 3) {
    p[k] = (p[k] - center[0]) / 1000; p[k + 1] = (p[k + 1] - center[1]) / 1000; p[k + 2] = (p[k + 2] - center[2]) / 1000;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  g.setIndex(new THREE.BufferAttribute(off.indices, 1));
  g.computeVertexNormals();
  g.computeBoundingBox();
  return g;
}
function bboxCenter(off) {
  const p = off.positions, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < p.length; k += 3) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[k + d]); hi[d] = Math.max(hi[d], p[k + d]); }
  return lo.map((v, d) => 0.5 * (v + hi[d]));
}

// ---------------------------------------------------------------- viewer
function createViewer(container) {
  container.innerHTML = "";
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xf7f7f8);
  container.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 10000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9aa0, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  camera.add(key);
  key.position.set(1, 1, 2);
  scene.add(camera);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const groups = { main: new THREE.Group(), overlay: new THREE.Group(), markers: new THREE.Group() };
  Object.values(groups).forEach((g) => scene.add(g));
  let home = null;

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
    camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();
  (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

  function clear(group) { while (group.children.length) { const c = group.children.pop(); c.geometry?.dispose(); c.material?.dispose(); } }
  function fit(box) {
    const size = box.getSize(new THREE.Vector3()), mid = box.getCenter(new THREE.Vector3());
    const r = Math.max(size.length() * 0.5, 1e-3);
    camera.near = r / 1000; camera.far = r * 100; camera.updateProjectionMatrix();
    camera.position.copy(mid.clone().add(new THREE.Vector3(0.35, 0.25, 1).normalize().multiplyScalar(r * 2.6)));
    controls.target.copy(mid); controls.update();
    home = { pos: camera.position.clone(), target: mid.clone() };
  }
  return {
    groups, camera, controls,
    setMesh(geometry, color) {
      clear(groups.main); clear(groups.overlay); clear(groups.markers);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0.0, side: THREE.DoubleSide }));
      groups.main.add(mesh);
      fit(geometry.boundingBox);
    },
    overlay(geometry, color) {
      clear(groups.overlay);
      if (!geometry) return;
      const m = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness: 0.5, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
      groups.overlay.add(m);
    },
    markers(points, radius, color) {
      clear(groups.markers);
      // drawn through the surface: the anchors lie under the spine bases
      const geo = new THREE.SphereGeometry(radius, 10, 8), mat = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.8 });
      const inst = new THREE.InstancedMesh(geo, mat, Math.max(points.length, 1));
      inst.count = points.length;
      inst.renderOrder = 2;
      const mtx = new THREE.Matrix4();
      points.forEach((p, k) => { mtx.makeTranslation(p[0], p[1], p[2]); inst.setMatrixAt(k, mtx); });
      groups.markers.add(inst);
    },
    focus(point, distance) {
      const t = new THREE.Vector3(...point);
      const dir = camera.position.clone().sub(controls.target).normalize();
      controls.target.copy(t);
      camera.position.copy(t.clone().add(dir.multiplyScalar(distance)));
    },
    reset() { if (home) { camera.position.copy(home.pos); controls.target.copy(home.target); } },
  };
}

// ---------------------------------------------------------------- form
async function loadOptions() {
  const res = await fetch("/api/options");
  const opt = await res.json();
  state.options = opt;
  const form = $("#request-form");
  for (const field of ["species", "cell_type_binary", "compartment"]) {
    const sel = form.elements[field];
    sel.innerHTML = "";
    opt.categories[field].forEach((v) => sel.append(el("option", { value: v, text: label(field, v) })));
    sel.value = opt.defaults[field];
  }
  fillSpineTypes();
  const lim = opt.limits;
  form.elements.length_um.min = lim.min_length_nm / 1000; form.elements.length_um.max = lim.max_length_nm / 1000;
  form.elements.length_um.value = opt.defaults.length_nm / 1000;
  form.elements.soma_um.min = 0; form.elements.soma_um.max = lim.max_soma_distance_nm / 1000;
  form.elements.soma_unknown.checked = opt.defaults.soma_distance_start_nm === null;
  form.elements.soma_um.value = opt.defaults.soma_distance_start_nm === null ? "" : opt.defaults.soma_distance_start_nm / 1000;
  form.elements.is_terminal.checked = !!opt.defaults.is_terminal;
  form.elements.seed.max = lim.seed_max; form.elements.seed.value = opt.defaults.seed;
  $("#limits-hint").textContent = `Длина от ${fmt(lim.min_length_nm / 1000)} до ${fmt(lim.max_length_nm / 1000)} мкм; расстояние до сомы до ${fmt(lim.max_soma_distance_nm / 1000)} мкм; не больше ${lim.max_spines} шипиков на ветвь.`;
  const b = opt.bundle;
  $("#bundle-meta").innerHTML = "";
  $("#bundle-meta").append(el("div", { text: `Набор моделей: ${b.bundle_id}` }), el("div", { text: `T0: ${b.t0_run} · P0: ${b.p0_run}` }),
    el("div", { text: `шипики: ${Object.entries(opt.spine_library).map(([k, v]) => `${k} (${v.n_spines})`).join(", ")}` }));
  form.addEventListener("change", onFormChange);
  onFormChange();
  applyHashRequest();
}

function fillSpineTypes() {
  const form = $("#request-form"), opt = state.options, sel = form.elements.spine_cell_type, keep = sel.value;
  const binary = form.elements.cell_type_binary.value;
  sel.innerHTML = "";
  sel.append(el("option", { value: "", text: "по типу клетки (общая библиотека)" }));
  (opt.categories.spine_cell_type || []).filter((t) => !opt.spine_cell_type_binary[t] || opt.spine_cell_type_binary[t] === binary)
    .forEach((t) => sel.append(el("option", { value: t, text: t })));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "";
}

function onFormChange(ev) {
  const form = $("#request-form");
  if (!ev || ev.target.name === "cell_type_binary") fillSpineTypes();
  form.elements.soma_um.disabled = form.elements.soma_unknown.checked;
  const g = `${form.elements.species.value}|${form.elements.cell_type_binary.value}|${form.elements.compartment.value}`;
  const groups = state.options.trained_groups || [];
  showNotice($("#form-warnings"), groups.length && !groups.includes(g)
    ? ["Такого сочетания условий не было среди отдельных групп при обучении модели ствола: будут использованы общие параметры."] : []);
}

function readRequest() {
  const f = $("#request-form").elements;
  return {
    species: f.species.value, cell_type_binary: f.cell_type_binary.value, compartment: f.compartment.value,
    spine_cell_type: f.spine_cell_type.value || null,
    length_nm: Math.round(parseFloat(f.length_um.value) * 1000),
    soma_distance_start_nm: f.soma_unknown.checked || f.soma_um.value === "" ? null : Math.round(parseFloat(f.soma_um.value) * 1000),
    is_terminal: f.is_terminal.checked, seed: parseInt(f.seed.value, 10),
  };
}

// shareable link: the request is kept in the URL hash
function applyHashRequest() {
  if (!location.hash.startsWith("#q=")) return;
  try {
    const q = JSON.parse(decodeURIComponent(location.hash.slice(3)));
    const f = $("#request-form").elements;
    for (const k of ["species", "cell_type_binary", "compartment"]) if (q[k]) f[k].value = q[k];
    fillSpineTypes();
    if (q.spine_cell_type) f.spine_cell_type.value = q.spine_cell_type;
    if (q.length_nm) f.length_um.value = q.length_nm / 1000;
    f.soma_unknown.checked = q.soma_distance_start_nm === null;
    if (q.soma_distance_start_nm !== null && q.soma_distance_start_nm !== undefined) f.soma_um.value = q.soma_distance_start_nm / 1000;
    f.is_terminal.checked = !!q.is_terminal;
    if (Number.isInteger(q.seed)) f.seed.value = q.seed;
    onFormChange();
    generate();
  } catch (e) { /* malformed link: keep the defaults */ }
}

async function generate(ev) {
  ev?.preventDefault();
  const request = readRequest();
  showNotice($("#form-errors"), []);
  $("#generate-btn").disabled = true; $("#busy").classList.remove("hidden");
  try {
    const res = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    const data = await res.json();
    if (!res.ok) {
      const names = { length_nm: "длина", soma_distance_start_nm: "расстояние до сомы", seed: "зерно", species: "вид", cell_type_binary: "тип клетки", compartment: "компартмент", spine_cell_type: "подробный тип", generation: "генерация", server: "сервер" };
      showNotice($("#form-errors"), (data.errors || [{ field: "", message: data.detail || res.statusText }]).map((e) => `${names[e.field] || e.field}: ${e.message}`));
      return;
    }
    history.replaceState(null, "", "#q=" + encodeURIComponent(JSON.stringify(data.request)));
    await showResult(data);
  } catch (e) {
    showNotice($("#form-errors"), [`ошибка связи с сервером: ${e.message}`]);
  } finally {
    $("#generate-btn").disabled = false; $("#busy").classList.add("hidden");
  }
}

// ---------------------------------------------------------------- result
const rowOf = (rows, key) => rows.find((r) => r.key === key) || {};

async function showResult(data) {
  state.result = data; state.selected = null;
  $("#empty-state").classList.add("hidden"); $("#result").classList.remove("hidden");
  const t = data.trunk, p = data.placement;
  const tiles = [
    { label: "Длина", value: `${fmt(rowOf(t, "length_nm").display)} мкм`, note: `извилистость ${fmt(rowOf(t, "tortuosity").display)}` },
    { label: "Медианный радиус", value: `${fmt(rowOf(t, "radius_median_nm").display, 3)} мкм`, note: `сужение ${fmt(rowOf(t, "taper_ratio").display)}` },
    { label: "Шипиков на сетке", value: fmt(rowOf(p, "n_placed").value), note: `по модели P: ${fmt(rowOf(p, "n_requested").value)}`, accent: "green" },
    { label: "Плотность", value: `${fmt(rowOf(p, "density_placed_per_um").display, 3)} мкм⁻¹`, note: "поставленных шипиков", accent: "green" },
    { label: "Библиотека шипиков", value: data.library.used_key, note: data.library.fallback ? `запрошена ${data.library.requested_keys[0]}` : `${data.library.pool_size} шипиков` },
    { label: "Сетка", value: data.status === "ok" ? "замкнута" : "с замечаниями", note: `${fmt(data.timings_s.total, 2)} с на генерацию` },
  ];
  const tilesNode = $("#tiles"); tilesNode.innerHTML = "";
  tiles.forEach((x) => tilesNode.append(el("div", { class: "tile" + (x.accent === "green" ? " green" : "") }, [el("div", { class: "label", text: x.label }), el("div", { class: "value", text: x.value }), el("div", { class: "note", text: x.note })])));
  showNotice($("#result-warnings"), data.warnings);

  metricTable($("#trunk-table"), t);
  metricTable($("#placement-table"), p.concat(data.branch_mesh));
  const fail = $("#failure-table"); fail.innerHTML = "";
  if (data.failure_reasons.length) {
    fail.append(el("h3", { text: "Неудачные попытки поставить шипик" }));
    const tb = el("table", { class: "data" }, [el("tr", {}, [el("th", { text: "Причина" }), el("th", { text: "Попыток" })])]);
    data.failure_reasons.forEach((r) => tb.append(el("tr", {}, [el("td", { text: r.name }), el("td", { class: "num", text: fmt(r.count) })])));
    fail.append(tb);
  }
  spineTable(data);
  comparePlots(data);
  downloads(data);
  resetSpineViewer();
  await loadBranch(data);
}

function metricTable(node, rows) {
  node.innerHTML = "";
  const tb = el("table", { class: "data" }, [el("tr", {}, [el("th", { text: "Характеристика" }), el("th", { text: "Значение" }), el("th", { text: "Единица" }), el("th", { text: "Смысл" })])]);
  rows.forEach((r) => tb.append(el("tr", {}, [el("td", { text: r.name }), el("td", { class: "num", text: fmt(r.display) }), el("td", { text: r.unit || "" }), el("td", { class: "about", text: r.about || "" })])));
  node.append(tb);
}

const SPINE_TABLE_COLUMNS = ["index", "status", "s_nm", "u", "theta", "radius_at_s_nm", "immersion_nm", "base_radius_nm", "library_key", "spine_id", "Length", "Volume", "Area", "JunctionArea", "OpenAngle", "ConvexHullRatio"];

function spineTable(data) {
  const cols = SPINE_TABLE_COLUMNS.map((k) => data.spine_columns.find((c) => c.key === k)).filter(Boolean);
  const table = $("#spine-table"); table.innerHTML = "";
  table.append(el("thead", {}, el("tr", {}, cols.map((c) => el("th", { text: c.unit ? `${c.name}, ${c.unit}` : c.name, title: c.about || "" })))));
  const body = el("tbody");
  data.spines.forEach((s) => {
    const tr = el("tr", { class: s.status === "placed" ? "" : "dropped", "data-index": s.index }, cols.map((c) => {
      if (c.key === "status") return el("td", {}, el("span", { class: "badge " + (s.status === "placed" ? "ok" : "no"), text: s.status === "placed" ? "поставлен" : "не поставлен" }));
      const v = s[c.key];
      return el("td", { class: typeof v === "number" ? "num" : "", text: typeof v === "number" ? fmt(v * c.factor) : (v ?? "—") });
    }));
    tr.addEventListener("click", () => selectSpine(s.index));
    body.append(tr);
  });
  table.append(body);
}

async function fetchOFF(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  return parseOFF(await res.text());
}

async function loadBranch(data) {
  if (!state.branch) state.branch = createViewer($("#branch-viewer"));
  const off = await fetchOFF(`/api/results/${data.id}/branch.off`);
  state.center = bboxCenter(off);
  state.branch.setMesh(geometryFrom(off, state.center), COLORS.branch);
  state.anchors = data.spines.filter((s) => s.status === "placed" && s.anchor_nm).map((s) => s.anchor_nm.map((v, d) => (v - state.center[d]) / 1000));
  drawAnchors();
}

function drawAnchors() {
  if (!state.branch) return;
  const radius = 0.06;
  state.branch.markers($("#show-anchors").checked ? state.anchors || [] : [], radius, COLORS.anchor);
}

function resetSpineViewer() {
  $("#spine-title").textContent = "Шипик";
  $("#spine-downloads").innerHTML = "";
  if (state.spine) { state.spine.groups.main.clear(); }
  if (!state.spine) $("#spine-viewer").innerHTML = '<p class="placeholder">Выберите шипик в таблице «Шипики»</p>';
}

async function selectSpine(index) {
  const data = state.result, s = data.spines.find((x) => x.index === index);
  document.querySelectorAll("#spine-table tbody tr").forEach((tr) => tr.classList.toggle("selected", Number(tr.dataset.index) === index));
  state.selected = index;
  if (s.status !== "placed") {
    state.branch.overlay(null);
    if (s.anchor_nm) state.branch.focus(s.anchor_nm.map((v, d) => (v - state.center[d]) / 1000), 5);
    $("#spine-title").textContent = `Положение ${index}: шипик не поставлен`;
    return;
  }
  const [glob, loc] = await Promise.all([fetchOFF(`/api/results/${data.id}/spines/${index}.off?frame=global`), fetchOFF(`/api/results/${data.id}/spines/${index}.off?frame=local`)]);
  state.branch.overlay(geometryFrom(glob, state.center), COLORS.highlight);
  state.branch.focus(s.anchor_nm.map((v, d) => (v - state.center[d]) / 1000), 5);
  if (!state.spine) state.spine = createViewer($("#spine-viewer"));
  state.spine.setMesh(geometryFrom(loc, [0, 0, 0]), COLORS.local);
  $("#spine-title").textContent = `Шипик ${index} · ${s.library_key} #${s.spine_id}`;
  const dl = $("#spine-downloads"); dl.innerHTML = "";
  dl.append(el("a", { class: "button small", href: `/api/results/${data.id}/spines/${index}.off?frame=local`, download: `spine_${index}_local.off`, text: "OFF, локальная" }),
    el("a", { class: "button small", href: `/api/results/${data.id}/spines/${index}.off?frame=global`, download: `spine_${index}_global.off`, text: "OFF, на ветви" }));
}

function comparePlots(data) {
  const node = $("#compare-plots"); node.innerHTML = "";
  const ref = data.reference;
  if (!ref || !ref.available) {
    $("#compare-lead").textContent = "В этом наборе моделей нет справочных данных о реальных ветвях.";
    return;
  }
  const groupText = ref.group === "all" ? "все реальные ветви (для выбранной группы условий данных мало)" : `реальные ветви группы ${ref.group}`;
  $("#compare-lead").textContent = `Сгенерированная ветвь на фоне: ${groupText}; ветвей для ствола ${ref.n_branches_T}, для шипиков ${ref.n_branches_P}.`;
  ref.rows.forEach((r) => {
    const div = el("div", { class: "plot" });
    node.append(div);
    const [q5, q25, q50, q75, q95] = r.quantiles;
    const title = r.unit ? `${r.name}, ${r.unit}` : r.name;
    Plotly.newPlot(div, [
      { type: "box", orientation: "h", y: [title], q1: [q25], median: [q50], q3: [q75], lowerfence: [q5], upperfence: [q95], name: "реальные ветви",
        fillcolor: "rgba(142,145,150,0.25)", line: { color: "#8e9196" }, hoverinfo: "x" },
      { type: "scatter", mode: "markers", y: [title], x: [r.generated], name: "сгенерированная", marker: { symbol: "diamond", size: 13, color: "#d6418f", line: { color: "#1d1d1f", width: 1 } },
        hovertemplate: `сгенерированная: %{x:.3g} ${r.unit}<extra></extra>` },
    ], {
      margin: { l: 10, r: 10, t: 28, b: 30 }, title: { text: title, font: { size: 13 }, x: 0.02 }, showlegend: false,
      paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: { family: "-apple-system, Segoe UI, Inter, Roboto, sans-serif", size: 11, color: "#4a4a50" },
      xaxis: { gridcolor: "#efeff2", zeroline: false }, yaxis: { showticklabels: false },
    }, PLOT_CONFIG);
  });
}

function downloads(data) {
  const node = $("#downloads"); node.innerHTML = "";
  const base = `/api/results/${data.id}`;
  [["download.zip", "Всё одним архивом (.zip)", "main"], ["branch.off", "Сетка ветви (.off)"], ["trunk.off", "Ствол без шипиков (.off)"], ["metrics.json", "Метрики (.json)"]]
    .forEach(([path, text, cls]) => node.append(el("a", { class: "button" + (cls ? " " + cls : ""), href: `${base}/${path}`, download: `${data.id}_${path}`, text })));
  const req = new Blob([JSON.stringify({ bundle_id: data.bundle_id, ...data.request }, null, 2)], { type: "application/json" });
  node.append(el("a", { class: "button", href: URL.createObjectURL(req), download: `request_${data.id}.json`, text: "Параметры запроса (.json)" }));
  node.append(el("button", { class: "ghost", type: "button", text: "Скопировать ссылку на эти параметры", onclick: () => navigator.clipboard?.writeText(location.href) }));
}

// ---------------------------------------------------------------- wiring
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== b.dataset.tab));
  if (b.dataset.tab === "compare") document.querySelectorAll("#compare-plots .plot").forEach((d) => Plotly.Plots.resize(d));
}));
$("#request-form").addEventListener("submit", generate);
$("#random-seed").addEventListener("click", () => { $("#request-form").elements.seed.value = Math.floor(Math.random() * 1e6); });
$("#show-anchors").addEventListener("change", drawAnchors);
$("#reset-view").addEventListener("click", () => { state.branch?.reset(); state.branch?.overlay(null); });
loadOptions().catch((e) => { $("#bundle-meta").textContent = `не удалось загрузить параметры: ${e.message}`; });
