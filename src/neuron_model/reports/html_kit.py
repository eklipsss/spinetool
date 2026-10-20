"""Building blocks of the project's self-contained HTML materials (one visual style for all of them).

Rules and rationale: ``docs/neuron-model/html-materials-style-guide.md``. In short:

- one file, opens offline: plotly.js and all data are inlined (no CDN, no server);
- white surface, near-black text, the project's pink + green as the data colours (validated for colour-vision
  deficiency and contrast on white, ``PALETTE`` below); the light brand pink only for large surfaces (meshes, tints);
- every colour, font and spacing comes from the CSS custom properties of :data:`CSS` and the plotly template of
  :func:`plotly_template` - a page never hard-codes a colour;
- explanations are connected prose in :func:`note` blocks, never scattered footnotes.

Data travel to the page as one JSON document (``<script type="application/json" id="payload">``); numeric arrays and
meshes are packed as base64 (:func:`pack_array`, :func:`pack_mesh`) and unpacked by :data:`COMMON_JS`.
"""

from __future__ import annotations

import base64
import html
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# palette (validated: scripts/validate_palette.js of the dataviz guide, light mode, surface #ffffff, --pairs all)
# ---------------------------------------------------------------------------

PALETTE: Dict[str, str] = {
    # surfaces and ink
    "surface": "#ffffff",
    "surface_alt": "#f7f7f8",       # sidebar, table stripes
    "border": "#e6e6ea",
    "grid": "#efeff2",
    "ink": "#1d1d1f",               # main text
    "ink_secondary": "#4a4a50",     # secondary text, axis titles
    "ink_muted": "#7a7a82",         # captions, ticks
    # data series (categorical, fixed order; all-pairs CVD dE >= 10.7, normal-vision dE >= 17.2, contrast >= 3:1)
    "series_1": "#d6418f",          # pink
    "series_2": "#3f8a2a",          # green
    "series_3": "#8a6bd1",          # violet - only when a third series is unavoidable
    "reference": "#8e9196",         # real data / the reference set (neutral, not a series hue)
    "muted_points": "#d4d5d9",      # context points behind a selection
    # brand surfaces (not for thin marks: contrast < 3:1 on white)
    "brand_pink": "#fd89c3",
    "brand_green": "#7fbc41",
    "tint_pink": "#fde0ef",
    "tint_green": "#e6f5d0",
    "mesh_real": "#c9cbd0",
    # status (reserved meaning, always with a word)
    "good": "#3f8a2a",
    "warning": "#b7791f",
    "bad": "#c0392b",
}

SERIES: Tuple[str, ...] = (PALETTE["series_1"], PALETTE["series_2"], PALETTE["series_3"])

# sequential = one hue light -> dark (pink arm of PiYG); diverging = green <-> neutral grey <-> pink
SEQUENTIAL: List[List[Any]] = [[0.0, "#fde0ef"], [0.25, "#f1b6da"], [0.5, "#de77ae"], [0.75, "#c51b7d"], [1.0, "#8e0152"]]
DIVERGING: List[List[Any]] = [[0.0, "#276419"], [0.2, "#4d9221"], [0.4, "#b8e186"], [0.5, "#f0efec"], [0.6, "#f1b6da"], [0.8, "#c51b7d"], [1.0, "#8e0152"]]

FONT_STACK = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace'


CSS = """
:root{
  --surface:%(surface)s;--surface-alt:%(surface_alt)s;--border:%(border)s;--grid:%(grid)s;
  --ink:%(ink)s;--ink-2:%(ink_secondary)s;--ink-3:%(ink_muted)s;
  --s1:%(series_1)s;--s2:%(series_2)s;--s3:%(series_3)s;--ref:%(reference)s;
  --brand-pink:%(brand_pink)s;--brand-green:%(brand_green)s;--tint-pink:%(tint_pink)s;--tint-green:%(tint_green)s;
  --good:%(good)s;--warning:%(warning)s;--bad:%(bad)s;
  --radius:14px;--radius-s:9px;--gap:16px;
  --font:%(font)s;--mono:%(mono)s;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 var(--font);-webkit-font-smoothing:antialiased}
a{color:var(--s1);text-decoration:none;border-bottom:1px solid var(--tint-pink)}a:hover{border-color:var(--s1)}
main{max-width:1680px;margin:0 auto;padding:28px 28px 56px}
header.page{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;padding-bottom:20px;border-bottom:1px solid var(--border);margin-bottom:24px}
header.page h1{font-size:28px;line-height:1.2;letter-spacing:-.02em;margin:0 0 6px;font-weight:650}
header.page .sub{color:var(--ink-2);font-size:14px;max-width:880px}
header.page .meta{color:var(--ink-3);font-size:12.5px;text-align:right;line-height:1.5}
.brand-bar{height:4px;border-radius:4px;background:linear-gradient(90deg,var(--brand-pink) 0 50%%,var(--brand-green) 50%% 100%%);width:64px;margin-bottom:14px}
h2{font-size:20px;letter-spacing:-.01em;margin:0 0 4px;font-weight:650}
h3{font-size:15px;margin:0 0 8px;font-weight:650}
section.block{margin-top:36px}
section.block>.lead{color:var(--ink-2);margin:2px 0 14px;max-width:980px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
.tile .label{font-size:12.5px;color:var(--ink-3)}
.tile .value{font-size:26px;font-weight:650;letter-spacing:-.02em;margin:2px 0;font-variant-numeric:tabular-nums}
.tile .tile-note{font-size:12.5px;color:var(--ink-2)}
.tile.accent{border-color:var(--tint-pink);box-shadow:inset 3px 0 0 var(--brand-pink)}
.tile.accent-green{border-color:var(--tint-green);box-shadow:inset 3px 0 0 var(--brand-green)}
.workspace{display:grid;grid-template-columns:290px minmax(0,1fr);gap:var(--gap);align-items:start}
.sidebar{position:sticky;top:14px;max-height:calc(100vh - 28px);overflow:auto;display:flex;flex-direction:column;gap:12px}
.panel{background:var(--surface-alt);border-radius:var(--radius);padding:14px}
.panel h3{font-size:13px;text-transform:none;color:var(--ink-2);margin-bottom:10px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;min-width:0}
.card>.card-title{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:13px 16px 0;font-weight:650;font-size:14px}
.card>.card-title .hint{font-weight:400;color:var(--ink-3);font-size:12.5px}
.card>.card-body{padding:10px 16px 14px}
.grid-2{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:var(--gap);align-items:start}
.grid-2e{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--gap);align-items:start}
.plot{width:100%%;height:520px}
.plot.s{height:340px}.plot.m{height:420px}.plot.l{height:620px}
label.field{display:block;font-size:12.5px;color:var(--ink-2);margin:10px 0 5px}
select,input[type=search],input[type=number],button{font:inherit;font-size:13.5px;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-s);padding:7px 10px}
select,input[type=search]{width:100%%}
button{cursor:pointer;transition:background .15s,border-color .15s}
button:hover{border-color:var(--brand-pink);background:var(--tint-pink)}
button.primary{background:var(--s1);border-color:var(--s1);color:#fff}button.primary:hover{background:#b8327a}
button[aria-pressed=true]{background:var(--tint-pink);border-color:var(--s1)}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:var(--radius-s);overflow:hidden}
.seg button{border:0;border-radius:0;border-right:1px solid var(--border)}.seg button:last-child{border-right:0}
input[type=range]{width:100%%;accent-color:var(--s1)}
input[type=checkbox]{accent-color:var(--s1);width:15px;height:15px}
:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.check{display:flex;align-items:center;gap:8px;font-size:13.5px;margin:6px 0;cursor:pointer}
.check .count{margin-left:auto;color:var(--ink-3);font-size:12px;font-variant-numeric:tabular-nums}
.swatch{display:inline-block;width:10px;height:10px;border-radius:50%%;flex:none}
.legend{display:flex;gap:6px 18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);padding:6px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--brand-green);border-radius:var(--radius-s);padding:14px 18px;margin-top:14px;max-width:1080px}
.note p{margin:0 0 9px;color:var(--ink-2)}.note p:last-child{margin-bottom:0}.note b{color:var(--ink)}
.note.pink{border-left-color:var(--brand-pink)}
details.note summary{cursor:pointer;font-weight:650;color:var(--ink)}details.note[open] summary{margin-bottom:10px}
table.metrics{width:100%%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
table.metrics th{font-weight:600;color:var(--ink-2);text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
table.metrics th:first-child,table.metrics td:first-child{text-align:left}
table.metrics td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}
table.metrics tr:hover td{background:var(--surface-alt)}
.pill{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:999px;background:var(--surface-alt);color:var(--ink-2)}
.pill.good{background:var(--tint-green);color:#2b5f1d}.pill.bad{background:#fbe3e0;color:var(--bad)}.pill.warn{background:#fbf0dc;color:#7a5414}
.bar{position:relative;height:6px;background:var(--grid);border-radius:6px;overflow:hidden}
.bar>i{position:absolute;inset:0 auto 0 0;background:var(--s1);border-radius:6px}
.muted{color:var(--ink-3)}.small{font-size:12.5px}.mono{font-family:var(--mono);font-size:12px}
.empty{display:flex;align-items:center;justify-content:center;color:var(--ink-3);font-size:13.5px;height:100%%;text-align:center;padding:20px}
footer.page{margin-top:40px;padding-top:16px;border-top:1px solid var(--border);color:var(--ink-3);font-size:12.5px}
@media(max-width:1180px){.grid-2,.grid-2e{grid-template-columns:1fr}}
@media(max-width:860px){main{padding:16px}.workspace{grid-template-columns:1fr}.sidebar{position:static;max-height:none}header.page .meta{text-align:left}}
""" % {**PALETTE, "font": FONT_STACK, "mono": MONO_STACK}


def plotly_template() -> Dict[str, Any]:
    """Layout defaults every figure of a page starts from (passed to the page as ``payload.theme``)."""
    p = PALETTE
    axis = {"gridcolor": p["grid"], "zerolinecolor": p["border"], "linecolor": p["border"], "tickcolor": p["border"],
            "tickfont": {"color": p["ink_muted"], "size": 11}, "title": {"font": {"color": p["ink_secondary"], "size": 12}}}
    return {
        "font": {"family": FONT_STACK, "color": p["ink"], "size": 12},
        "paper_bgcolor": p["surface"], "plot_bgcolor": p["surface"], "colorway": list(SERIES),
        "xaxis": axis, "yaxis": axis,
        "hoverlabel": {"bgcolor": p["surface"], "bordercolor": p["border"], "font": {"family": FONT_STACK, "color": p["ink"], "size": 12}},
        "legend": {"font": {"color": p["ink_secondary"], "size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 56, "r": 16, "t": 12, "b": 48},
        "sequential": SEQUENTIAL, "diverging": DIVERGING, "palette": p, "series": list(SERIES),
    }


# ---------------------------------------------------------------------------
# HTML components (all text escaped; ``html_*`` arguments are trusted markup built by the page code)
# ---------------------------------------------------------------------------


def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def header(title: str, subtitle: str = "", meta_lines: Sequence[str] = ()) -> str:
    meta = "<br>".join(esc(m) for m in meta_lines)
    return (f'<header class="page"><div><div class="brand-bar"></div><h1>{esc(title)}</h1>'
            f'<div class="sub">{esc(subtitle)}</div></div><div class="meta">{meta}</div></header>')


def tiles(items: Iterable[Mapping[str, Any]]) -> str:
    """items: {label, value, note?, accent?: 'pink'|'green'}"""
    out = []
    for it in items:
        cls = {"pink": " accent", "green": " accent-green"}.get(it.get("accent") or "", "")
        note = f'<div class="tile-note">{esc(it["note"])}</div>' if it.get("note") else ""
        out.append(f'<div class="tile{cls}"><div class="label">{esc(it["label"])}</div><div class="value">{esc(it["value"])}</div>{note}</div>')
    return '<div class="tiles">' + "".join(out) + "</div>"


def note(paragraphs: Sequence[str], *, title: Optional[str] = None, collapsible: bool = False, tone: str = "green") -> str:
    """An explanation block: connected paragraphs (each a full sentence or more). ``**bold**`` marks key facts."""
    def fmt(p: str) -> str:
        parts = esc(p).split("**")
        return "".join(f"<b>{x}</b>" if i % 2 else x for i, x in enumerate(parts))

    body = "".join(f"<p>{fmt(p)}</p>" for p in paragraphs)
    cls = "note" + (" pink" if tone == "pink" else "")
    if collapsible:
        return f'<details class="{cls}"><summary>{esc(title or "Как читать")}</summary>{body}</details>'
    head = f"<h3>{esc(title)}</h3>" if title else ""
    return f'<div class="{cls}">{head}{body}</div>'


def section(title: str, lead: str, html_inner: str, *, anchor: Optional[str] = None) -> str:
    """``anchor``: id of the section - use a ``sec-`` prefix: an id equal to a plot div id makes plotly draw into the
    section and wipe it."""
    aid = f' id="{esc(anchor)}"' if anchor else ""
    return f'<section class="block"{aid}><h2>{esc(title)}</h2><p class="lead">{esc(lead)}</p>{html_inner}</section>'


def card(title: str, html_inner: str, *, hint: str = "") -> str:
    h = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    return f'<div class="card"><div class="card-title"><span>{esc(title)}</span>{h}</div><div class="card-body">{html_inner}</div></div>'


# ---------------------------------------------------------------------------
# data packing
# ---------------------------------------------------------------------------


def pack_array(a: np.ndarray, dtype: str = "float32") -> Dict[str, Any]:
    """{dtype, shape, b64} - little-endian raw bytes; unpacked in the page by ``unpackArray``."""
    arr = np.ascontiguousarray(np.asarray(a).astype(np.dtype(dtype).newbyteorder("<")))
    return {"dtype": dtype, "shape": list(arr.shape), "b64": base64.b64encode(arr.tobytes()).decode("ascii")}


def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation for DISPLAY only (no topology guarantee): vertices are merged on a cubic grid whose
    cell is searched so that at most ``max_faces`` non-degenerate faces remain. Meshes already small are returned as is."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= max_faces:
        return v, f
    extent = float(np.max(v.max(0) - v.min(0))) or 1.0
    lo, hi = extent / 2000.0, extent / 8.0
    best = (v, f)
    for _ in range(18):
        cell = math.sqrt(lo * hi)
        key = np.floor((v - v.min(0)) / cell).astype(np.int64)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        sums = np.zeros((counts.size, 3))
        np.add.at(sums, inverse, v)
        nv = sums / counts[:, None]
        nf = inverse[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        first = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)[1]  # drop duplicates, keep orientation
        nf = nf[np.sort(first)]
        if len(nf) > max_faces:
            lo = cell
        else:
            hi = cell
            best = (nv, nf)
        if hi / lo < 1.08:
            break
    used = np.unique(best[1])
    remap = -np.ones(len(best[0]), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return best[0][used], remap[best[1]]


def pack_mesh(vertices: np.ndarray, faces: np.ndarray, *, max_faces: int = 2500) -> Dict[str, Any]:
    """Display mesh: decimated, vertices quantised to int16 over their box (~0.05 nm on a 3 um spine), faces
    uint16/uint32. ``unpackMesh`` in the page returns {x, y, z, i, j, k} for a plotly ``mesh3d``."""
    v, f = decimate(vertices, faces, max_faces)
    lo, hi = v.min(0), v.max(0)
    scale = np.maximum(hi - lo, 1e-9) / 65000.0
    q = np.round((v - lo) / scale - 32500.0).astype(np.int16)
    fdt = "uint16" if len(v) < 65535 else "uint32"
    return {"v": pack_array(q, "int16"), "f": pack_array(f, fdt), "lo": [float(x) for x in lo], "scale": [float(x) for x in scale], "nf": int(len(f))}


def _sanitize(o: Any) -> Any:
    """NaN / inf -> None everywhere (JSON.parse in the browser rejects NaN), numpy -> python."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_, np.ndarray)):
        return _sanitize(_json_default(o))
    return o


def payload_script(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_sanitize(payload), ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False).replace("</", "<\\/")
    return f'<script type="application/json" id="payload">{text}</script>'


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def clean_floats(values: Iterable[Any], digits: int = 6) -> List[Optional[float]]:
    """JSON-safe list: NaN / inf -> null, rounded to ``digits`` significant digits."""
    out = []
    for x in values:
        try:
            v = float(x)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(None if not math.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


COMMON_JS = r"""
'use strict';
const $ = id => document.getElementById(id);
const PAYLOAD = JSON.parse($('payload').textContent);
const THEME = PAYLOAD.theme, P = THEME.palette;
const NF = new Intl.NumberFormat('ru-RU', {maximumSignificantDigits: 3});
const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? '—' :
  (d === undefined ? NF.format(v) : Number(v).toLocaleString('ru-RU', {minimumFractionDigits: d, maximumFractionDigits: d}));
function unpackArray(a) {
  const bin = atob(a.b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  const T = {float32: Float32Array, float64: Float64Array, int16: Int16Array, int32: Int32Array, uint16: Uint16Array, uint32: Uint32Array, uint8: Uint8Array}[a.dtype];
  return new T(u.buffer);
}
const _meshCache = new Map();
function unpackMesh(m) {
  if (_meshCache.has(m)) return _meshCache.get(m);
  const q = unpackArray(m.v), f = unpackArray(m.f), n = q.length / 3;
  const x = new Array(n), y = new Array(n), z = new Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = m.lo[0] + (q[3*i] + 32500) * m.scale[0]; y[i] = m.lo[1] + (q[3*i+1] + 32500) * m.scale[1]; z[i] = m.lo[2] + (q[3*i+2] + 32500) * m.scale[2];
  }
  const nf = f.length / 3, I = new Array(nf), J = new Array(nf), K = new Array(nf);
  for (let i = 0; i < nf; i++) { I[i] = f[3*i]; J[i] = f[3*i+1]; K[i] = f[3*i+2]; }
  const out = {x, y, z, i: I, j: J, k: K};
  _meshCache.set(m, out);
  return out;
}
function layout(extra) {
  const base = {font: THEME.font, paper_bgcolor: THEME.paper_bgcolor, plot_bgcolor: THEME.plot_bgcolor, colorway: THEME.colorway,
    hoverlabel: THEME.hoverlabel, legend: THEME.legend, margin: Object.assign({}, THEME.margin), hovermode: 'closest',
    xaxis: Object.assign({}, THEME.xaxis), yaxis: Object.assign({}, THEME.yaxis)};
  for (const [k, v] of Object.entries(extra || {})) {
    if ((k === 'xaxis' || k === 'yaxis' || k === 'margin') && v && typeof v === 'object') base[k] = Object.assign({}, base[k], v);
    else base[k] = v;
  }
  return base;
}
function axisTitle(text) { return {text: text, font: {color: P.ink_secondary, size: 12}}; }
const PLOT_CONFIG = {responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian']};
function scene3d(range, extra) {
  const ax = r => ({range: r, showticklabels: false, title: {text: ''}, showgrid: true, gridcolor: P.grid, zeroline: false, showbackground: false, showspikes: false});
  return Object.assign({aspectmode: 'cube', xaxis: ax(range[0]), yaxis: ax(range[1]), zaxis: ax(range[2]),
    camera: {eye: {x: 1.55, y: 0.95, z: 1.05}, up: {x: 0, y: 1, z: 0}}}, extra || {});
}
function meshTrace(mesh, color, name, opacity) {
  return Object.assign({type: 'mesh3d', color: color, opacity: opacity === undefined ? 1 : opacity, name: name, flatshading: false, hoverinfo: 'skip',
    lighting: {ambient: 0.55, diffuse: 0.75, specular: 0.15, roughness: 0.6, fresnel: 0.1}, lightposition: {x: 800, y: 2000, z: 1500}}, mesh);
}
function cubeRange(meshes, pad) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const m of meshes) for (const [a, d] of [[m.x, 0], [m.y, 1], [m.z, 2]]) for (const v of a) { if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
  const r = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * (0.5 + (pad || 0.06));
  return lo.map((v, d) => [(v + hi[d]) / 2 - r, (v + hi[d]) / 2 + r]);
}
function swatch(color) { const s = document.createElement('span'); s.className = 'swatch'; s.style.background = color; return s; }
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) { if (k === 'text') e.textContent = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v); }
  for (const c of (children || [])) e.append(c);
  return e;
}
"""


def plotly_js() -> str:
    """The plotly.js bundle shipped with the installed ``plotly`` package (inlined: the page works offline)."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def page(title: str, body_html: str, payload: Mapping[str, Any], page_js: str, *, inline_plotly: bool = True) -> str:
    """The whole document. ``payload`` gets the plotly theme under ``theme``; ``page_js`` runs after :data:`COMMON_JS`."""
    data = dict(payload)
    data["theme"] = plotly_template()
    lib = f"<script>{plotly_js()}</script>" if inline_plotly else '<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>'
    return ("<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{CSS}</style>{lib}</head><body><main>{body_html}</main>"
            f"{payload_script(data)}<script>{COMMON_JS}\n{page_js}</script></body></html>")
