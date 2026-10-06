"""NAS-oriented I/O of the preprocessing pipeline (no CGAL, synthetic rows only):
table parts + compaction, table-based eligibility, neuron chunks, sqlite neuron filter,
packed point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.neuron_model.spine_preprocessing as sp
from src.neuron_model.pointcloud_io import (
    available_pointclouds,
    legacy_variant_path,
    load_pointcloud,
    load_points,
    unpack_pointclouds,
    write_pointclouds,
)


def _record(tmp_path: Path, neuron: str = "n1", spine: str = "000") -> sp.SpineRecord:
    branch = tmp_path / "raw" / "minnie65" / neuron / "limb_000" / "branch_000"
    return sp.SpineRecord(
        dataset="minnie65",
        neuron_id=neuron,
        limb_id="000",
        branch_id="000",
        spine_id=spine,
        source_path=branch / "spines" / f"spine_{spine}.off",
        branch_path=branch,
        branch_skeleton_path=branch / "branch_skeleton.npy",
        dataset_root=tmp_path / "raw" / "minnie65",
    )


def _row(record: sp.SpineRecord, **values):
    return {**sp._base_row(record), **values}


def _cfg(tmp_path: Path, **kw) -> sp.SpinePreprocessingConfig:
    return sp.SpinePreprocessingConfig(raw_data_root=tmp_path / "raw", output_root=tmp_path / "out", **kw).normalized()


# --- table parts / compaction -------------------------------------------------------


def test_parts_are_merged_on_read_and_later_rows_win(tmp_path):
    path = tmp_path / "t.parquet"
    a, b = _record(tmp_path, spine="000"), _record(tmp_path, spine="001")
    sp._write_table([_row(a, status="failed"), _row(b, status="success")], path)
    sp._append_table_part(path, [_row(a, status="success", value=1.5)])
    rows = sp._read_table_rows(path)
    assert rows[a.key]["status"] == "success" and rows[a.key]["value"] == 1.5
    assert rows[b.key]["status"] == "success"
    frame = sp._read_table_frame(path)
    assert len(frame) == 2 and set(frame["status"]) == {"success"}


def test_compaction_rewrites_main_table_once_and_removes_parts(tmp_path):
    cfg = _cfg(tmp_path)
    spec = sp.STAGES[sp.STAGE_QC]
    root = tmp_path / "out" / "minnie65"
    root.mkdir(parents=True)
    a, b = _record(tmp_path, spine="000"), _record(tmp_path, spine="001")
    sp._append_table_part(root / spec.table, [_row(a, status="valid")])
    sp._append_table_part(root / spec.table, [_row(b, status="invalid"), _row(a, status="needs_review")])
    sp._append_errors(root, [{"stage": "x", "message": "boom"}])
    sp._compact_stage_table(spec, cfg, root, a.dataset_root)
    table = pd.read_parquet(root / spec.table)
    assert len(table) == 2 and dict(zip(table["spine_id"], table["status"])) == {"000": "needs_review", "001": "invalid"}
    assert not sp._table_part_paths(root / spec.table) and not (root / f"{spec.table}.parts").exists()
    assert (root / f"{spec.table}.manifest.json").exists()
    assert len(pd.read_parquet(root / "errors.parquet")) == 1 and not sp._table_part_paths(root / "errors.parquet")


def test_table_cache_tracks_appends_and_deduplicates(tmp_path):
    path = tmp_path / "t.parquet"
    a = _record(tmp_path)
    sp._write_table([_row(a, status="failed")], path)
    cache = sp._TableCache()
    assert len(cache.frame(path)) == 1
    cache.append(path, [_row(a, status="success")])
    assert len(cache.frame(path)) == 2  # raw, both versions
    dedup = cache.deduplicated(path)
    assert len(dedup) == 1 and dedup.loc[0, "status"] == "success"
    assert sp._rows_for_keys(cache.frame(path), [a.key])[a.key]["status"] == "success"


# --- eligibility from tables (no file system) ------------------------------------------


def _context(tmp_path, tables):
    root = tmp_path / "out" / "minnie65"
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        sp._write_table(rows, root / name)
    return {"root": root, "_lock": __import__("threading").Lock(), "cache": None}


def test_table_eligibility_matches_upstream_states_without_touching_files(tmp_path, monkeypatch):
    ok, ambiguous, failed_seal, not_valid = (_record(tmp_path, spine=s) for s in ("000", "001", "002", "003"))
    context = _context(
        tmp_path,
        {
            "sealed_spines.parquet": [
                _row(ok, status="success", attachment_ambiguous=False, n_attachment_cap_faces=12),
                _row(ambiguous, status="needs_review", attachment_ambiguous=True, n_attachment_cap_faces=12),
                _row(failed_seal, status="failed"),
                _row(not_valid, status="success", attachment_ambiguous=False, n_attachment_cap_faces=3),
            ],
            "false_spines.parquet": [
                _row(ok, status="valid"), _row(ambiguous, status="valid"), _row(not_valid, status="invalid"),
            ],
            "oriented_spines.parquet": [_row(ok, status="success"), _row(ambiguous, status="skipped")],
            "mesh_quality.parquet": [_row(ok, status="needs_review")],
        },
    )
    calls = {"n": 0}
    original = Path.exists

    def counting_exists(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "exists", counting_exists)
    cfg = _cfg(tmp_path)
    assert sp._false_spine_eligibility(ok, cfg, context) is None
    assert sp._false_spine_eligibility(failed_seal, cfg, context) == "missing_sealed_mesh"
    assert sp._orient_eligibility(ok, cfg, context) is None
    assert sp._orient_eligibility(ambiguous, cfg, context) == "attachment_ambiguous"
    assert sp._orient_eligibility(not_valid, cfg, context) == "not_valid_after_false_spine_detection"
    assert sp._qc_eligibility(ok, cfg, context) is None
    assert sp._qc_eligibility(ambiguous, cfg, context) == "missing_local_sealed_mesh"
    assert sp._training_eligibility(ok, cfg, context) is None
    assert sp._training_eligibility(not_valid, cfg, context) == "not_valid_after_false_spine_detection"
    assert sp._training_eligibility(ok, _cfg(tmp_path, allow_needs_review=False), context) == "mesh_quality_needs_review"
    # only the per-table existence checks of the bulk reads, nothing per spine
    assert calls["n"] <= 4


def test_verify_mode_falls_back_to_file_checks(tmp_path):
    record = _record(tmp_path)
    context = _context(tmp_path, {"sealed_spines.parquet": [_row(record, status="success")]})
    cfg = _cfg(tmp_path, verify_outputs_on_resume=True)
    assert sp._false_spine_eligibility(record, cfg, context) == "missing_sealed_mesh"  # table says yes, file absent
    path = sp.sealed_mesh_path(record, cfg)
    path.parent.mkdir(parents=True)
    path.write_text("OFF\n0 0 0\n")
    assert sp._false_spine_eligibility(record, cfg, context) is None
    assert sp._parallel_exists([path, path.with_name("nope.off")], threads=4) == [True, False]


# --- chunks / journal -------------------------------------------------------------------


def test_neuron_chunks_keep_whole_neurons(tmp_path):
    records = [_record(tmp_path, neuron=n, spine=s) for n in ("a", "b", "c") for s in ("000", "001")]
    chunks = sp._neuron_chunks(records, 2)
    assert [[r.neuron_id for r in chunk] for chunk in chunks] == [["a", "a", "b", "b"], ["c", "c"]]
    assert sp._neuron_chunks(records, 0) == [records]
    assert sp._neuron_chunks([], 2) == []


def test_state_store_neuron_filter(tmp_path):
    store = sp.ProcessingStateStore(tmp_path / "state.sqlite")
    records = [_record(tmp_path, neuron=n) for n in ("a", "b")]
    for record in records:
        store.mark(record, "stage", "success", config_hash="h")
    full = store.get_all_states("stage")
    only_a = store.get_all_states("stage", neurons=[("minnie65", "a")])
    assert len(full) == 2 and list(only_a) == [records[0].key] and only_a[records[0].key][0] == "success"
    store.close()


# --- packed point clouds -------------------------------------------------------------------


def _clouds(rng):
    return {
        (n, v): {
            "points": rng.normal(size=(n, 3)).astype(np.float32),
            "normals": rng.normal(size=(n, 3)).astype(np.float32),
            "face_indices": rng.integers(0, 100, n).astype(np.int32),
            "is_attachment_cap": rng.random(n) < 0.1,
            "seed": np.int64(v),
            "sampling_seed": np.int64(40 + v),
        }
        for n in (16, 32)
        for v in (1, 2)
    }


def test_packed_pointclouds_roundtrip_and_unpack(tmp_path):
    clouds = _clouds(np.random.default_rng(0))
    path = tmp_path / "pointclouds.npz"
    write_pointclouds(path, clouds)
    assert not path.with_name("pointclouds.npz.tmp").exists()
    assert available_pointclouds(path) == [(16, 1), (16, 2), (32, 1), (32, 2)]
    for (n, v), fields in clouds.items():
        loaded = load_pointcloud(path, n, v)
        assert set(loaded) == set(fields) and all(np.array_equal(loaded[k], fields[k]) for k in fields)
        assert np.array_equal(load_points(path, n, v), fields["points"])
    with pytest.raises(KeyError):
        load_pointcloud(path, 64, 1)
    files = unpack_pointclouds(path, tmp_path / "legacy")
    assert sorted(f.name for f in files) == ["pointcloud_16_1.npz", "pointcloud_16_2.npz", "pointcloud_32_1.npz", "pointcloud_32_2.npz"]
    # the legacy reader resolves other sizes/variants from any per-variant file of the spine
    legacy_first = tmp_path / "legacy" / "pointcloud_16_1.npz"
    assert np.array_equal(load_points(legacy_first, 32, 2), clouds[(32, 2)]["points"])
    assert legacy_variant_path(legacy_first, 2).name == "pointcloud_16_2.npz"
