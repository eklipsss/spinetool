import base64
import json

import numpy as np
import pytest
import trimesh

from src.neuron_model.reports import html_kit as hk


def test_pack_mesh_decimates_and_quantises_to_sub_nanometre():
    m = trimesh.creation.icosphere(subdivisions=5, radius=1000.0)
    v, f = hk.decimate(m.vertices, m.faces, 2000)
    assert len(f) <= 2000 and f.max() < len(v)
    p = hk.pack_mesh(m.vertices, m.faces, max_faces=2000)
    q = np.frombuffer(base64.b64decode(p["v"]["b64"]), dtype="<i2").reshape(-1, 3).astype(float)
    rec = np.array(p["lo"]) + (q + 32500.0) * np.array(p["scale"])
    assert np.abs(rec - v).max() < 0.1
    assert np.linalg.norm(rec, axis=1).min() > 990.0  # still a sphere of radius ~1000 nm


def test_page_is_self_contained_and_payload_is_valid_json():
    html = hk.page("Тест", hk.header("Тест", "подзаголовок") + hk.note(["Первый абзац с **важным**."]), {"x": [1.0, float("nan")]}, "console.log(1);", inline_plotly=False)
    start = html.index('id="payload">') + len('id="payload">')
    payload = json.loads(html[start:html.index("</script>", start)])
    assert payload["x"] == [1.0, None] and "theme" in payload
    assert "<b>важным</b>" in html and "var(--s1)" in html


def test_clean_floats():
    assert hk.clean_floats([1.23456789, None, float("inf"), "x"]) == [1.23457, None, None, None]


def test_atlas_latent_scores_detect_copies():
    from src.neuron_model.spine_generation.atlas import latent_scores

    rng = np.random.default_rng(0)
    train, real = rng.normal(size=(300, 8)), rng.normal(size=(200, 8))
    honest = latent_scores(rng.normal(size=(200, 8)), real, train)
    copies = latent_scores(train[:200] + 1e-3, real, train)
    assert 0.35 < honest["one_nna"] < 0.65 and honest["memorization_ratio"] == pytest.approx(1.0, abs=0.25)
    assert copies["memorization_ratio"] < 0.05
