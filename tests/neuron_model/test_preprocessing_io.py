"""NAS-oriented I/O of the preprocessing pipeline (no CGAL, synthetic rows only):
table parts + compaction, table-based eligibility, neuron chunks, sqlite neuron filter,
packed point clouds."""

from __future__ import annotations

import json
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


# --- discover_spines: narrowed glob -------------------------------------------------


def _make_raw_tree(root: Path) -> None:
    """``<root>/minnie65/<neuron>/limb_<L>/branch_<B>/spines/spine_<S>.off``, 2 neurons x
    2 limbs x 2 branches x 2 spines, so filters at every level have >1 candidate to narrow."""
    dataset_root = root / "minnie65"
    for neuron in ("n1", "n2"):
        for limb in ("000", "001"):
            for branch in ("000", "001"):
                spines_dir = dataset_root / neuron / f"limb_{limb}" / f"branch_{branch}" / "spines"
                spines_dir.mkdir(parents=True)
                for spine in ("000", "001"):
                    (spines_dir / f"spine_{spine}.off").write_text("OFF\n0 0 0\n")


def test_discover_spines_glob_matches_filters_at_every_level(tmp_path):
    _make_raw_tree(tmp_path)
    base = dict(raw_data_root=tmp_path, dataset="minnie65")

    all_records = sp.discover_spines(sp.SpinePreprocessingConfig(**base))
    assert len(all_records) == 16

    by_neuron = sp.discover_spines(sp.SpinePreprocessingConfig(**base, neuron_id="n1"))
    assert len(by_neuron) == 8 and {r.neuron_id for r in by_neuron} == {"n1"}

    by_limb = sp.discover_spines(sp.SpinePreprocessingConfig(**base, neuron_id="n1", limb_id="limb_000"))
    assert len(by_limb) == 4 and {r.limb_id for r in by_limb} == {"000"}  # accepts with or without the "limb_" prefix

    one = sp.discover_spines(sp.SpinePreprocessingConfig(**base, neuron_id="n2", limb_id="001", branch_id="000", spine_id="001"))
    assert [r.key for r in one] == [("minnie65", "n2", "001", "000", "001")]

    assert sp.discover_spines(sp.SpinePreprocessingConfig(**base, neuron_id="does-not-exist")) == []


def test_discover_spines_neuron_filter_does_not_list_other_neurons(tmp_path, monkeypatch):
    """The glob for one --neuron-id must not enumerate the other neuron directories at all -
    the whole point of narrowing it (see discover_spines): on a slow NAS, listing every
    neuron before filtering is what made a single-neuron run take 15+ minutes."""
    _make_raw_tree(tmp_path)
    listed = []
    original = Path.iterdir

    def tracking_iterdir(self):
        listed.append(self)
        return original(self)

    monkeypatch.setattr(Path, "iterdir", tracking_iterdir)
    records = sp.discover_spines(sp.SpinePreprocessingConfig(raw_data_root=tmp_path, dataset="minnie65", neuron_id="n1"))
    assert len(records) == 8
    assert not any(p.name == "n2" for p in listed)


def test_full_pipeline_shares_one_table_cache_even_for_a_single_chunk(tmp_path, monkeypatch):
    """A single-neuron (single-chunk) run must still share one _TableCache across every
    stage + manifest call - without it, each stage used to re-read its own table AND every
    upstream table its eligibility needs from scratch (observed: redundant full reads of
    the same NAS-hosted parquet across seal/false-spine/orient within one test run)."""
    seen_caches = []

    def fake_run_stage(stage, cfg, chunk, *, executor=None, compact=True, cache=None):
        seen_caches.append(("stage", stage, cache))
        return pd.DataFrame()

    def fake_run_manifest_stage(cfg, chunk, *, cache=None):
        seen_caches.append(("manifest", None, cache))
        return pd.DataFrame()

    def fake_compact(cfg, records, *, cache=None):
        seen_caches.append(("compact", None, cache))

    monkeypatch.setattr(sp, "discover_spines", lambda cfg: [_record(tmp_path)])
    monkeypatch.setattr(sp, "run_stage", fake_run_stage)
    def fake_run_fused_stages(cfg, chunk, *, stages, executor=None, compact=True, cache=None, staging_dir=None):
        seen_caches.append(("fused", None, cache))
        return {stage: pd.DataFrame() for stage in stages}

    monkeypatch.setattr(sp, "run_manifest_stage", fake_run_manifest_stage)
    monkeypatch.setattr(sp, "run_fused_stages", fake_run_fused_stages)
    monkeypatch.setattr(sp, "compact_stage_tables", fake_compact)

    cfg = _cfg(tmp_path, workers=1, neurons_per_chunk=25)  # 1 neuron -> a single chunk
    sp.run_full_spine_preprocessing_pipeline(cfg)

    assert seen_caches, "no stage/manifest calls recorded"
    caches = {cache for _, _, cache in seen_caches}
    assert len(caches) == 1 and None not in caches, f"expected one shared, non-None cache, got {caches}"
    assert {kind for kind, _, _ in seen_caches} == {"stage", "fused", "manifest", "compact"}


# --- winding numbers: numba kernel == numpy reference --------------------------------


def test_numba_winding_numbers_match_numpy_off_the_surface():
    import trimesh

    from src.neuron_model.spine_sampling import warm_up_winding_kernel, winding_numbers

    if not warm_up_winding_kernel():
        pytest.skip("numba not installed")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    rng = np.random.default_rng(0)
    surface = mesh.sample(500)
    normals = mesh.face_normals[mesh.nearest.on_surface(surface)[2]]
    points = np.vstack([
        rng.normal(size=(500, 3)) * 15.0,          # inside and outside
        surface + normals * 0.05,                   # just outside
        surface - normals * 0.05,                   # just inside
    ])
    triangles = np.asarray(mesh.triangles, dtype=float)
    reference = winding_numbers(points, triangles, backend="numpy")
    fast = winding_numbers(points, triangles, backend="numba")
    assert np.max(np.abs(reference - fast)) < 1e-10
    assert np.array_equal(reference > 0.5, fast > 0.5)
    assert np.all(fast[500:1000] < 0.5) and np.all(fast[1000:] > 0.5)
    with pytest.raises(ValueError):
        winding_numbers(points, triangles, backend="cuda")


# --- fused stages 5-9 / staging ----------------------------------------------------------


def test_fused_worker_gates_downstream_stages_on_the_qc_status_it_produced(tmp_path, monkeypatch):
    calls = []

    def fake_stage_worker(stage, record, cfg):
        calls.append(stage)
        row = {"status": "invalid"} if stage == sp.STAGE_QC else {"status": sp.STATUS_SUCCESS}
        return {"row": row, "error": None, "started": 0.0, "ended": 0.0, "timings": {}}

    monkeypatch.setattr(sp, "_execute_stage_worker", fake_stage_worker)
    record, cfg = _record(tmp_path), _cfg(tmp_path)
    plan = [
        (sp.STAGE_QC, sp._ACTION_RUN),
        (sp.STAGE_POINTCLOUD, sp._ACTION_GATED_RUN),
        (sp.STAGE_SDF, sp._ACTION_GATED_DONE),
        (sp.STAGE_METADATA, sp._ACTION_RUN),
    ]
    result = sp._execute_fused_worker(plan, record, cfg)
    assert calls == [sp.STAGE_QC, sp.STAGE_METADATA]  # gated point cloud not run, done SDF untouched
    assert result["quality"] == "invalid"
    assert result["outcomes"][sp.STAGE_POINTCLOUD] == {"skip_reason": "mesh_quality_invalid"}
    assert sp.STAGE_SDF not in result["outcomes"]
    assert sp._quality_reason("needs_review", cfg) is None
    assert sp._quality_reason("needs_review", _cfg(tmp_path, allow_needs_review=False)) == "mesh_quality_needs_review"


def test_spine_io_scope_memoises_reads_and_redirects_writes_to_staging(tmp_path):
    final_root, staging_dir = tmp_path / "final", tmp_path / "staging"
    target = final_root / "n1" / "spine_000" / "quality.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"a": 1}')
    with sp._spine_io_scope((str(final_root), str(staging_dir))):
        assert sp._read_json(target) == {"a": 1}
        target.write_text('{"a": 999}')  # memoised: the NAS file is not read again
        assert sp._read_json(target) == {"a": 1}
        sp._update_json_atomic(target, {"b": 2})
        staged = staging_dir / "n1" / "spine_000" / "quality.json"
        assert staged.exists() and json.loads(staged.read_text()) == {"a": 1, "b": 2}
        assert sp._read_json(target) == {"a": 1, "b": 2}
        assert sp._staged_path(tmp_path / "elsewhere.json") == tmp_path / "elsewhere.json"
    assert json.loads(target.read_text()) == {"a": 999}  # final file untouched until moved
    assert sp._staged_path(target) == target  # no scope -> no redirect


def test_move_staged_tree_moves_files_and_cleans_up(tmp_path):
    final_root = tmp_path / "final"
    (final_root / "n1" / "spine_000").mkdir(parents=True)
    staged = tmp_path / "staging" / "run_chunk_0001" / "minnie65"
    (staged / "n1" / "spine_000").mkdir(parents=True)
    (staged / sp._STAGING_MARKER).write_text(str(final_root))
    (staged / "n1" / "spine_000" / "pointclouds.npz").write_bytes(b"data")
    (staged / "n1" / "spine_001").mkdir()
    (staged / "n1" / "spine_001" / "metadata.json").write_text("{}")  # target dir does not exist yet
    (staged / "n1" / "spine_000" / "x.json.tmp").write_text("partial")
    assert sp._staged_root_dirs(tmp_path / "staging") == [staged]
    n_files, n_bytes, _ = sp._move_staged_tree(staged, threads=2)
    assert n_files == 2 and n_bytes == 6
    assert (final_root / "n1" / "spine_000" / "pointclouds.npz").read_bytes() == b"data"
    assert (final_root / "n1" / "spine_001" / "metadata.json").read_text() == "{}"
    assert not (final_root / "n1" / "spine_000" / "x.json.tmp").exists()
    assert not staged.exists() and sp._staged_root_dirs(tmp_path / "staging") == []


def test_append_after_parts_dir_was_removed_in_the_same_process(tmp_path):
    """Regression: a prior call in the same process (e.g. an earlier notebook cell) can
    compact and delete a table's parts dir; _ensure_dir's per-process cache must not make
    the next append() believe that directory still exists."""
    path = tmp_path / "errors.parquet"
    record = _record(tmp_path)
    sp._append_table_part(path, [_row(record, status="failed")])
    parts_dir = sp._table_parts_dir(path)
    assert str(parts_dir) not in sp._CREATED_DIRS  # no longer cached as existing
    assert parts_dir.is_dir()
    sp._remove_table_parts(path)  # simulates compaction: deletes the dir
    assert not parts_dir.exists()
    # a later call in the SAME process (same _CREATED_DIRS) must recreate it, not crash
    sp._append_table_part(path, [_row(record, status="needs_review")])
    assert len(sp._read_table_rows(path)) == 1


# --- journal: local redirection + rebuild from tables --------------------------------


def test_journal_path_defaults_to_nas_then_journal_root_then_staging_root(tmp_path):
    root = tmp_path / "preprocessed"
    assert sp._journal_path(root, _cfg(tmp_path)) == root / "processing_state.sqlite"

    staged = _cfg(tmp_path, staging_root=tmp_path / "staging")
    via_staging = sp._journal_path(root, staged)
    assert str(via_staging).startswith(str(tmp_path / "staging" / "journal"))
    assert via_staging.name == "processing_state.sqlite"

    both = _cfg(tmp_path, staging_root=tmp_path / "staging", journal_root=tmp_path / "journal_elsewhere")
    via_journal_root = sp._journal_path(root, both)
    assert str(via_journal_root).startswith(str(tmp_path / "journal_elsewhere"))

    # same root -> same path every time (stable, so a later rebuild lands on the same file)
    assert sp._journal_path(root, staged) == via_staging
    # a different root must not collide
    assert sp._journal_path(tmp_path / "other" / "preprocessed", staged) != via_staging


def test_rebuild_journal_from_tables_recovers_resume_decisions(tmp_path):
    root = tmp_path / "preprocessed"
    root.mkdir()
    cfg = _cfg(tmp_path)
    a, b, c = (_record(tmp_path, spine=s) for s in ("000", "001", "002"))
    sp._write_table(
        [
            _row(a, status="success", config_hash="h1"),
            _row(b, status="needs_review", config_hash="h1"),
            _row(c, status="failed", config_hash="h1", error="boom"),
        ],
        root / "sealed_spines.parquet",
    )
    # a real (working) journal already present - must be moved aside, not merged with
    old_store = sp.ProcessingStateStore(root / "processing_state.sqlite")
    old_store.mark_key(a.key, sp.STAGE_SEAL, "success", config_hash="stale-hash")
    old_store.close()

    n = sp.rebuild_journal_from_tables(root)
    assert n == 3
    assert list(root.glob("processing_state.sqlite.bak-*"))  # old file preserved, not deleted

    store = sp.ProcessingStateStore(root / "processing_state.sqlite")
    states = store.get_all_states(sp.STAGE_SEAL)
    store.close()
    assert states[a.key] == ("success", sp.ALGORITHM_VERSION, "h1")  # NOT the stale "stale-hash" row
    assert states[b.key][0] == "needs_review" and states[c.key][0] == "failed"
    # matches what run_stage's own resume logic would decide
    assert sp._status_from_state(states[a.key], "h1") == "success"
    assert sp._status_from_state(states[a.key], "different-hash") is None


def test_rebuild_journal_from_tables_to_an_explicit_local_path(tmp_path):
    root = tmp_path / "preprocessed"
    root.mkdir()
    record = _record(tmp_path)
    sp._write_table([_row(record, status="success", config_hash="h1")], root / "sealed_spines.parquet")
    target = tmp_path / "local_journal" / "processing_state.sqlite"
    n = sp.rebuild_journal_from_tables(root, db_path=target)
    assert n == 1 and target.exists() and not (root / "processing_state.sqlite").exists()
    store = sp.ProcessingStateStore(target)
    assert store.get_all_states(sp.STAGE_SEAL)[record.key][0] == "success"
    store.close()


def test_rebuild_journal_resolves_domain_status_to_the_real_journal_state(tmp_path):
    """Regression: a stage's table "status" can be a domain/business value (false-spine's
    "valid"/"invalid", QC's "valid"/"needs_review"/"invalid") that differs from the
    pipeline/journal state run_stage actually recorded (always "success" unless the
    worker itself failed or the status is literally one of the reserved journal words) -
    see _state_from_row, which strips the "_state" override before a row reaches a table."""
    root = tmp_path / "preprocessed"
    root.mkdir()
    valid, invalid, reviewed = (_record(tmp_path, spine=s) for s in ("000", "001", "002"))
    sp._write_table(
        [
            _row(valid, status="valid", config_hash="h1"),      # domain "valid" -> journal "success"
            _row(invalid, status="invalid", config_hash="h1"),  # domain "invalid" -> journal "success" too
            _row(reviewed, status="needs_review", config_hash="h1"),  # already a reserved word -> kept as-is
        ],
        root / "false_spines.parquet",
    )
    sp.rebuild_journal_from_tables(root)
    store = sp.ProcessingStateStore(root / "processing_state.sqlite")
    states = store.get_all_states(sp.STAGE_FALSE_SPINE)
    store.close()
    assert states[valid.key][0] == sp.STATUS_SUCCESS
    assert states[invalid.key][0] == sp.STATUS_SUCCESS
    assert states[reviewed.key][0] == sp.STATUS_NEEDS_REVIEW
    # exactly what run_stage's resume check needs to treat all three as already done
    for record in (valid, invalid, reviewed):
        assert sp._status_from_state(states[record.key], "h1") in {sp.STATUS_SUCCESS, sp.STATUS_NEEDS_REVIEW}


def test_rebuild_journal_skips_rows_missing_status_or_config_hash(tmp_path):
    root = tmp_path / "preprocessed"
    root.mkdir()
    a, b = _record(tmp_path, spine="000"), _record(tmp_path, spine="001")
    sp._write_table(
        [_row(a, status="success", config_hash="h1"), _row(b, status="success", config_hash=None)],
        root / "sealed_spines.parquet",
    )
    n = sp.rebuild_journal_from_tables(root)
    assert n == 1  # the row with no config_hash can't be trusted - skipped, not guessed
    store = sp.ProcessingStateStore(root / "processing_state.sqlite")
    states = store.get_all_states(sp.STAGE_SEAL)
    store.close()
    assert list(states) == [a.key]
