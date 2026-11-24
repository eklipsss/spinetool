from __future__ import annotations

import numpy as np
import pandas as pd

from src.neuron_model.labid_group_effects import Comparison, compare, unit_table


def _morph(rng, effect=0.0):
    rows = []
    for group, n_img, shift in (("Wt", 20, 0.0), ("Ab", 20, effect)):
        for img in range(n_img):
            u = rng.normal(0, 0.15)                       # image (cell) effect: spines of one image are correlated
            for crop in (1, 2):
                for s in range(8):
                    rows.append({"neuron_id": f"{group}_{img + 1}_{crop}", "Volume": float(np.exp(17 + shift + u + rng.normal(0, 0.4)))})
    return pd.DataFrame(rows)


def test_units_are_source_images_and_a_real_effect_is_found():
    rng = np.random.default_rng(0)
    m = _morph(rng, effect=-0.3)
    u = unit_table(m, None, metrics=("Volume",))
    # crops _1 and _2 of one image are one unit; Wt images 4+5 and 8+9 share one stack (metadata._LABID_SOURCE_MERGED)
    assert (u["group"] == "Ab").sum() == 20 and (u["group"] == "Wt").sum() == 18
    r = compare(u, Comparison("Ab vs Wt", "Ab", "Wt"), metrics=("Volume",), n_perm=2000, n_boot=500).iloc[0]
    assert r.significant and -35 < r.effect_pct < -18


def test_no_effect_is_rarely_significant():
    hits = 0
    for seed in range(20):
        u = unit_table(_morph(np.random.default_rng(seed), effect=0.0), None, metrics=("Volume",))
        hits += bool(compare(u, Comparison("Ab vs Wt", "Ab", "Wt"), metrics=("Volume",), n_perm=500, n_boot=100).iloc[0].significant)
    assert hits <= 3
