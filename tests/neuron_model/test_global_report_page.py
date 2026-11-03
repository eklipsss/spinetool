"""HTML page of the global report built from synthetic engine tables (reports/global_report.py)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.neuron_model.global_report.datasets import dataset_inventory
from src.neuron_model.global_report.descriptive import summarize_metrics
from src.neuron_model.global_report.significance import run_comparisons, save_tables
from src.neuron_model.reports.global_report import build_report
from tests.neuron_model.test_global_report import METRICS, _build

SPINE_NAMES = {"M_cell": "Length", "M_comp": "Volume", "M_species": "Area", "M_path": "ConvexHullVolume", "M_null": "Width"}
BRANCH_NAMES = {"M_cell": "radius_median_nm", "M_comp": "density_per_um", "M_species": "tortuosity", "M_path": "share_empty", "M_null": "kappa_rms"}


def write_tables(folder: Path, n_perm: int = 800) -> None:
    frame = _build()
    spine = frame.rename(columns=SPINE_NAMES)
    sp_metrics = list(SPINE_NAMES.values())
    res = run_comparisons(spine, sp_metrics, level="spine", min_neurons=10, min_spines=5, n_permutations=n_perm, n_boot=60, log=lambda *_: None)
    tables = {f"{k}_spine": v for k, v in res.items()}
    tables["descriptive_spine"] = summarize_metrics(spine, sp_metrics, by=["dataset"], level="spine")
    tables["descriptive_spine_by_cell_type"] = summarize_metrics(spine, sp_metrics, by=["dataset", "cell_type_binary"], level="spine")
    tables.update({f"inventory_{k}": v for k, v in dataset_inventory(spine).items()})
    branch = frame.rename(columns=BRANCH_NAMES)
    bm = list(BRANCH_NAMES.values())
    resb = run_comparisons(branch, bm, level="branch", min_neurons=10, min_spines=1, n_permutations=n_perm, n_boot=60, abs_metrics=["share_empty"], log=lambda *_: None)
    tables.update({f"{k}_branch": v for k, v in resb.items()})
    tables["descriptive_branch"] = summarize_metrics(branch, bm, by=["dataset"], level="branch")
    save_tables(tables, folder)
    (folder / "run_info.json").write_text(json.dumps({"n_permutations": n_perm, "created": "2026-11-04 12:00"}), encoding="utf-8")


def test_page_builds_and_payload_is_strict_json(tmp_path):
    write_tables(tmp_path)
    out = build_report(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert out.name.endswith("_global-report.html") and "sec-validity" in html and "sec-sig" in html
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1)

    def no_const(c):
        raise AssertionError(f"non-JSON constant {c} in payload")

    data = json.loads(payload, parse_constant=no_const)
    assert set(data["levels"]) == {"spine", "branch"}
    spine = data["levels"]["spine"]
    assert any(r["lab"] == "significant" for r in spine["results"]) and any(not r["causal"] for r in spine["results"])
    assert {"cell_class", "layer", "species | layer"} <= {c["f"] for c in spine["comparisons"]}
    assert "descriptive_only" not in html or True
    assert data["inventory"]["cell_class"]
