from __future__ import annotations

import json
import random
import time
from pathlib import Path

from fastapi.testclient import TestClient

from data_pipeline.pool_allocator import (
    AllocatorConfig,
    AllocatorPaths,
    RunOptions,
    _next_prefixed_id,
    _artist_cap_ok,
    _create_pool_from_asset,
    _has_duration_capacity,
    _license_matches,
    _primary_genre_matches,
    _territory_matches,
    calculate_style_score,
    edit_distance,
    list_assignments,
    load_allocator_config,
    normalize_manifest_row,
    run_pool_allocation,
)
from gui.backend import main as backend_main


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _base_row(source_id: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "freesound",
        "source_id": source_id,
        "download_status": "downloaded",
        "licence_class": "commercial_training_collective_v1",
        "license_normalized": "commercial_training_collective_v1",
        "territory": "AU",
        "record_label": "Example Label",
        "rights_holder": "Example Rights Holder",
        "artist_primary": f"artist_{source_id}",
        "artist_ids": [f"artist_{source_id}"],
        "primary_genre": "Children's Music",
        "secondary_genre": "Reggae",
        "style_tags": ["ukulele", "upbeat"],
        "duration_seconds": 240.0,
        "title": f"Song {source_id}",
        "language": "en",
        "content_hash": f"hash_{source_id}",
    }
    row.update(overrides)
    return row


def test_licence_and_territory_mismatch_reject_candidate() -> None:
    asset = normalize_manifest_row(_base_row(1), 0)
    pool = {
        "licence_class": "wrong_licence",
        "territory": "US",
        "primary_genre": "Children's Music",
    }
    assert not _license_matches(asset, pool)
    assert not _territory_matches(asset, pool)
    assert _primary_genre_matches(asset, pool)


def test_pool_full_rejects_assignment() -> None:
    config = AllocatorConfig()
    asset = normalize_manifest_row(_base_row(1, duration_seconds=500.0), 0)
    pool = {
        "current_duration_seconds": 17_700.0,
        "pool_duration_cap_seconds": config.max_pool_duration_seconds,
    }
    assert not _has_duration_capacity(asset, pool, config)


def test_artist_cap_rejects_assignment_above_limit() -> None:
    config = AllocatorConfig()
    asset = normalize_manifest_row(_base_row(1, duration_seconds=200.0, artist_ids=["artist_a"], artist_primary="artist_a"), 0)
    pool = {
        "artist_duration_seconds": {"artist_a": 1_700.0},
    }
    assert not _artist_cap_ok(asset, pool, config)


def test_style_filter_rejects_low_similarity() -> None:
    asset = normalize_manifest_row(_base_row(1, style_tags=["ukulele", "family"], secondary_genre="Reggae", language="en"), 0)
    pool = {
        "style_profile": {
            "style_tokens": [
                "secondary:metal",
                "style:distorted guitar",
                "mood:aggressive",
                "lang:de",
            ]
        }
    }
    assert calculate_style_score(asset, pool) < 0.2


def test_normalize_manifest_row_uses_local_meta_duration(tmp_path: Path, monkeypatch) -> None:
    meta_path = tmp_path / "data" / "freesound_meta" / "1.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"duration": 71.9006}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    asset = normalize_manifest_row(
        _base_row(1, duration_seconds=None, local_meta_path="data/freesound_meta/1.json"),
        0,
    )

    assert asset["duration_seconds"] == 71.9006


def test_normalize_manifest_row_collapses_niche_genre_labels() -> None:
    asset = normalize_manifest_row(
        _base_row(
            31,
            primary_genre="Angry",
            secondary_genre="Angry",
            title="Buddy Bark.wav",
            style_tags=["angry", "bark", "buddy", "dog"],
        ),
        0,
    )

    assert asset["primary_genre"] == "Sound Effects"
    assert asset["secondary_genre"] == ""
    assert asset["style_tags"] == ["animal"]
    assert asset["metadata_style_summary"] == "Sound Effects | animal"


def test_normalize_manifest_row_collapses_hardstyle_to_electronic() -> None:
    asset = normalize_manifest_row(
        _base_row(
            32,
            primary_genre="Hardstyle",
            title="B-BEAT001.wav",
            style_tags=["hardstyle"],
        ),
        0,
    )

    assert asset["primary_genre"] == "Electronic"
    assert asset["style_tags"] == ["electronic"]


def test_new_pool_id_generation_enforces_uniqueness_and_distance() -> None:
    config = AllocatorConfig()
    registry = {"pools": [], "assets": [], "assignments": [], "duplicates": [], "runs": []}
    asset = normalize_manifest_row(_base_row(1), 0)
    pool_one = _create_pool_from_asset(asset, registry, config, random.Random(1))
    registry["pools"].append(pool_one)
    pool_two = _create_pool_from_asset(asset, registry, config, random.Random(2))

    assert pool_one["pool_id"] != pool_two["pool_id"]
    assert pool_one["checksum"] == pool_one["pool_id"].split(":")[-1]
    assert pool_two["checksum"] == pool_two["pool_id"].split(":")[-1]
    assert edit_distance(pool_one["pool_code"].replace("-", ""), pool_two["pool_code"].replace("-", "")) >= config.min_pool_code_edit_distance


def test_next_prefixed_id_handles_assignment_id_rows() -> None:
    rows = [
        {"assignment_id": "assign_000001"},
        {"assignment_id": "assign_000009"},
    ]
    assert _next_prefixed_id(rows, "assign") == "assign_000010"


def test_exact_duplicate_returns_existing_assignment_and_no_new_pool(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(1, isrc="AUABC2400012", content_hash="same_hash"),
            _base_row(2, isrc="AUABC2400012", content_hash="same_hash", title="Song 2"),
        ],
    )

    result = run_pool_allocation(paths, config=load_allocator_config())
    assignments = list_assignments(paths, limit=10)
    pools_payload = json.loads(paths.pools_path.read_text(encoding="utf-8"))

    assert result["counts"]["new_pool_created"] == 1
    assert result["counts"]["duplicate_found"] == 1
    assert len(pools_payload) == 1
    assert assignments[0]["assignment_status"] == "duplicate_found"
    assert assignments[1]["assignment_status"] == "new_pool_created"


def test_run_creates_registry_files_and_manifest_bindings(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(11, isrc="AUABC2400011"),
            _base_row(12, isrc="AUABC2400012", style_tags=["ukulele", "family", "upbeat"]),
        ],
    )

    result = run_pool_allocation(paths, config=load_allocator_config())
    manifest_rows = [json.loads(line) for line in paths.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["processed_assets"] == 2
    assert paths.assets_path.exists()
    assert paths.pools_path.exists()
    assert paths.assignments_path.exists()
    assert paths.duplicates_path.exists()
    assert paths.runs_path.exists()
    assert paths.cara_manifest_path.exists()
    assert paths.cara_manifest_csv_path.exists()
    assert all(row.get("cara_source_asset_id") for row in manifest_rows)
    assert all(row.get("cara_source_pool_assignment_id") for row in manifest_rows)


def test_rerun_skips_already_allocated_assets_and_only_processes_new_rows(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(41, isrc="AUABC2400041"),
            _base_row(42, isrc="AUABC2400042", style_tags=["ukulele", "family", "upbeat"]),
        ],
    )

    first = run_pool_allocation(paths, config=load_allocator_config())
    assert first["processed_assets"] == 2

    _write_manifest(
        paths.manifest_path,
        [
            _base_row(41, isrc="AUABC2400041"),
            _base_row(42, isrc="AUABC2400042", style_tags=["ukulele", "family", "upbeat"]),
            _base_row(43, isrc="AUABC2400043", style_tags=["family", "upbeat"]),
        ],
    )

    second = run_pool_allocation(paths, config=load_allocator_config())
    assignments = list_assignments(paths, limit=20)

    assert second["total_assets"] == 3
    assert second["processed_assets"] == 3
    assert len([item for item in assignments if item["assignment_status"] in {"assigned", "new_pool_created"}]) == 3
    assert len(paths.assignments_path.read_text(encoding="utf-8").splitlines()) == 3


def test_incomplete_rights_metadata_becomes_review_required(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(21, record_label="", rights_holder="", isrc="AUABC2400021"),
        ],
    )

    result = run_pool_allocation(paths, config=load_allocator_config())
    assignments = list_assignments(paths, limit=10)

    assert result["counts"]["review_required"] == 1
    assert assignments[0]["assignment_status"] == "review_required"
    assert assignments[0]["review_required"] is True


def test_relaxed_metadata_mode_allocates_missing_rights_rows(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(22, record_label="", rights_holder="", isrc="AUABC2400022"),
            _base_row(23, record_label="", rights_holder="", isrc="AUABC2400023", style_tags=["upbeat", "family"]),
        ],
    )

    result = run_pool_allocation(
        paths,
        config=load_allocator_config(),
        options=RunOptions(only_downloaded=False, allow_relaxed_metadata=True),
    )
    assignments = list_assignments(paths, limit=10)

    assert result["counts"]["review_required"] == 0
    assert result["counts"]["new_pool_created"] >= 1
    assert assignments[0]["assignment_status"] in {"assigned", "new_pool_created"}


def test_relaxed_metadata_mode_reuses_compatible_pool(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(
                221,
                record_label="",
                rights_holder="",
                artist_primary="artist_a",
                artist_ids=["artist_a"],
                primary_genre="Percussion Drums",
                secondary_genre="Percussion Drums",
                style_tags=[],
                title="Drum loop A",
            ),
            _base_row(
                222,
                record_label="",
                rights_holder="",
                artist_primary="artist_b",
                artist_ids=["artist_b"],
                primary_genre="Percussion Drums",
                secondary_genre="Percussion Drums",
                style_tags=[],
                title="Drum loop B",
            ),
        ],
    )

    result = run_pool_allocation(
        paths,
        config=load_allocator_config(),
        options=RunOptions(allow_relaxed_metadata=True),
    )
    assignments = list_assignments(paths, limit=10)
    pools_payload = json.loads(paths.pools_path.read_text(encoding="utf-8"))

    assert result["counts"]["new_pool_created"] == 1
    assert result["counts"]["assigned"] == 1
    assert len(pools_payload) == 1
    assert {assignment["assignment_status"] for assignment in assignments[:2]} == {"assigned", "new_pool_created"}


def test_relaxed_rerun_reprocesses_previous_review(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(24, record_label="", rights_holder="", isrc="AUABC2400024"),
        ],
    )

    first = run_pool_allocation(paths, config=load_allocator_config())
    second = run_pool_allocation(
        paths,
        config=load_allocator_config(),
        options=RunOptions(allow_relaxed_metadata=True),
    )
    assignments = list_assignments(paths, limit=10)

    assert first["counts"]["review_required"] == 1
    assert second["counts"]["review_required"] == 0
    assert assignments[0]["assignment_status"] == "new_pool_created"


def test_cara_manifest_only_contains_processed_training_rows(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(41, isrc="AUABC2400041", download_status="downloaded", subset_role="music_train_candidate"),
            _base_row(
                42,
                isrc="AUABC2400042",
                download_status="not_downloaded",
                subset_role="",
                title="Not downloaded row",
            ),
        ],
    )

    run_pool_allocation(
        paths,
        config=load_allocator_config(),
        options=RunOptions(subset_role="music_train_candidate", only_downloaded=True),
    )

    cara_rows = [json.loads(line) for line in paths.cara_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in paths.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(source_rows) == 2
    assert len(cara_rows) == 1
    assert cara_rows[0]["title"] == "Song 41"
    assert cara_rows[0]["cara_source_pool_assignment_status"] in {"assigned", "new_pool_created"}


def test_start_fresh_run_rebuilds_allocator_outputs(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(51, isrc="AUABC2400051"),
            _base_row(52, isrc="AUABC2400052"),
        ],
    )

    first = run_pool_allocation(paths, config=load_allocator_config())
    first_assignments = [json.loads(line) for line in paths.assignments_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    _write_manifest(
        paths.manifest_path,
        [
            _base_row(53, isrc="AUABC2400053"),
        ],
    )
    second = run_pool_allocation(
        paths,
        config=load_allocator_config(),
        options=RunOptions(start_fresh=True),
    )
    second_assignments = [json.loads(line) for line in paths.assignments_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert first["processed_assets"] == 2
    assert len(first_assignments) == 2
    assert second["processed_assets"] == 1
    assert len(second_assignments) == 1
    assert second_assignments[0]["asset_id"] == "asset_000001"


def test_zero_duration_registry_rebuilds_on_new_engine(tmp_path: Path, monkeypatch) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(61, duration_seconds=None, local_meta_path="data/freesound_meta/61.json"),
        ],
    )
    meta_dir = tmp_path / "data" / "freesound_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "61.json").write_text(json.dumps({"duration": 42.5}), encoding="utf-8")

    paths.registry_dir.mkdir(parents=True, exist_ok=True)
    paths.assets_path.write_text(
        json.dumps({"asset_id": "asset_000001", "source_key": "freesound:61", "duration_seconds": 0.0}) + "\n",
        encoding="utf-8",
    )
    paths.pools_path.write_text(
        json.dumps([{"pool_id": "CARA:AUD:1:TEST-TEST-TEST:AA", "current_duration_seconds": 0.0, "asset_count": 1}]),
        encoding="utf-8",
    )
    paths.assignments_path.write_text(
        json.dumps(
            {
                "assignment_id": "assign_000001",
                "asset_id": "asset_000001",
                "pool_id": "CARA:AUD:1:TEST-TEST-TEST:AA",
                "assignment_status": "new_pool_created",
                "allocation_engine_version": "0.2.1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    result = run_pool_allocation(paths, config=load_allocator_config())
    pools_payload = json.loads(paths.pools_path.read_text(encoding="utf-8"))

    assert result["processed_assets"] == 1
    assert len(pools_payload) == 1
    assert pools_payload[0]["current_duration_seconds"] == 42.5


def test_duplicate_assignment_ids_force_rebuild(tmp_path: Path) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(71, isrc="AUABC2400071"),
            _base_row(72, isrc="AUABC2400072"),
        ],
    )
    paths.registry_dir.mkdir(parents=True, exist_ok=True)
    paths.assignments_path.write_text(
        "\n".join(
            [
                json.dumps({"assignment_id": "assign_000001", "asset_id": "asset_000001", "assignment_status": "assigned", "allocation_engine_version": "0.2.1"}),
                json.dumps({"assignment_id": "assign_000001", "asset_id": "asset_000002", "assignment_status": "assigned", "allocation_engine_version": "0.2.1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_pool_allocation(paths, config=load_allocator_config())
    assignments = [json.loads(line) for line in paths.assignments_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["processed_assets"] == 2
    assert len(assignments) == 2
    assert len({row["assignment_id"] for row in assignments}) == 2


def test_api_run_status_and_outputs(tmp_path: Path, monkeypatch) -> None:
    paths = AllocatorPaths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _base_row(31, isrc="AUABC2400031"),
            _base_row(32, isrc="AUABC2400032"),
        ],
    )

    monkeypatch.setattr(backend_main, "ROOT", tmp_path)
    with TestClient(backend_main.app) as client:
        start = client.post(
            "/api/data/pool-allocation/run",
            json={"subset_role": None, "only_downloaded": True, "allow_relaxed_metadata": True, "limit": None},
        )
        assert start.status_code == 200

        deadline = time.time() + 10
        latest_status = None
        while time.time() < deadline:
            latest_status = client.get("/api/data/pool-allocation/run-status")
            assert latest_status.status_code == 200
            payload = latest_status.json()
            if not payload["job"]["running"]:
                break
            time.sleep(0.1)

        assert latest_status is not None
        payload = latest_status.json()
        assert payload["job"]["running"] is False
        assert payload["summary"]["assignment_count"] >= 2

        pools = client.get("/api/data/pool-allocation/pools")
        assignments = client.get("/api/data/pool-allocation/assignments")
        review_queue = client.get("/api/data/pool-allocation/review-queue")

        assert pools.status_code == 200
        assert assignments.status_code == 200
        assert review_queue.status_code == 200
        assert isinstance(pools.json(), list)
        assert isinstance(assignments.json(), list)
        assert isinstance(review_queue.json(), list)
