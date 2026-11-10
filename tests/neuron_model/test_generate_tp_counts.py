"""``generate_tp.py --counts real|real_positioned``: N is taken from the source branch, not from the count model."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/neuron_model/generate_tp.py"


def _module():
    spec = importlib.util.spec_from_file_location("generate_tp_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_counts_columns_and_missing():
    mod = _module()
    rows = pd.DataFrame({"n_spines_in_range": [5, 0, 12], "n_with_position": [4, 0, 11]}).iloc[[2, 0]]
    assert mod.real_counts(rows, "real").tolist() == [12, 5]
    assert mod.real_counts(rows, "real_positioned").tolist() == [11, 4]
    assert mod.real_counts(rows, "real").dtype == np.int64
    with pytest.raises(ValueError):
        mod.real_counts(pd.DataFrame({"n_spines_in_range": [3, None], "n_with_position": [3, 1]}), "real")
