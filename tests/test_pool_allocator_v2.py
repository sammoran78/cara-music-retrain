from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.pool_allocator_v2 import (
    AllocatorV2Config,
    AllocatorV2Paths,
    RunOptionsV2,
    list_pools_v2,
    run_pool_allocation_v2,
)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(source_id: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "freesound",
        "source_id": source_id,
        "subset_role": "music_train_candidate",
        "download_status": "downloaded",
        "license_normalized": "cc0",
        "licence_class": "cc0",
        "territory": "GLOBAL",
        "artist_primary": f"artist_{source_id}",
        "artist_ids": [f"artist_{source_id}"],
        "primary_genre": "Percussion Drums",
        "secondary_genre": "Percussion Drums",
        "style_tags": ["drum", "loop"],
        "title": f"Sound {source_id}",
        "duration_seconds": 900.0,
        "content_hash": f"hash_{source_id}",
    }
    row.update(overrides)
    return row


def test_v2_spills_into_new_pool_when_four_hour_cap_fills(tmp_path: Path) -> None:
    paths = AllocatorV2Paths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            *[_row(source_id, duration_seconds=1200.0) for source_id in range(1, 13)],
            _row(13, duration_seconds=1200.0),
        ],
    )

    result = run_pool_allocation_v2(
        paths,
        options=RunOptionsV2(subset_role="music_train_candidate", only_downloaded=True),
        config=AllocatorV2Config(),
    )
    pools = list_pools_v2(paths)

    assert result["status"] == "completed"
    assert result["pool_count"] == 2
    assert sorted(round(pool["current_duration_seconds"], 1) for pool in pools) == [1200.0, 14400.0]
    assert all(pool["current_duration_seconds"] <= 14_400 for pool in pools)


def test_v2_artist_concentrated_pool_is_explicit_exception(tmp_path: Path) -> None:
    paths = AllocatorV2Paths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _row(source_id, duration_seconds=1800.0, artist_primary="Jovica", artist_ids=["Jovica"])
            for source_id in range(10, 15)
        ],
    )

    result = run_pool_allocation_v2(
        paths,
        options=RunOptionsV2(subset_role="music_train_candidate", only_downloaded=True),
        config=AllocatorV2Config(artist_exception_min_duration_seconds=7200, artist_exception_min_pool_pressure=3),
    )
    pools = list_pools_v2(paths)
    assignments = [json.loads(line) for line in paths.assignments_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["status"] == "completed"
    assert len(pools) == 1
    assert pools[0]["pool_type"] == "artist_concentrated_pool"
    assert pools[0]["top_artist_share"] > 0.1
    assert any("ARTIST_CATALOGUE_EXCEPTION" in assignment["reason_codes"] for assignment in assignments)


def test_v2_collapses_micro_sound_effect_styles_into_broad_family(tmp_path: Path) -> None:
    paths = AllocatorV2Paths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _row(31, primary_genre="Angry", secondary_genre="Angry", title="Buddy Bark.wav", style_tags=["angry", "bark", "buddy", "dog"]),
            _row(32, primary_genre="Throw", secondary_genre="Throw", title="Door slam.wav", style_tags=["door", "slam", "impact"]),
        ],
    )

    result = run_pool_allocation_v2(
        paths,
        options=RunOptionsV2(subset_role="music_train_candidate", only_downloaded=True),
        config=AllocatorV2Config(),
    )
    pools = list_pools_v2(paths)

    assert result["pool_count"] == 1
    assert pools[0]["pool_family"] == "Sound Effects"
    assert set(pools[0]["included_primary_genres"]) == {"Sound Effects"}


def test_v2_writes_training_manifest_with_standard_cara_fields(tmp_path: Path) -> None:
    paths = AllocatorV2Paths.from_root(tmp_path)
    _write_manifest(
        paths.manifest_path,
        [
            _row(41, primary_genre="Acoustic/Folk", style_tags=["guitar", "acoustic"], duration_seconds=120.0),
            _row(42, primary_genre="Jazz Blues", style_tags=["jazz", "swing"], duration_seconds=180.0),
        ],
    )

    result = run_pool_allocation_v2(
        paths,
        options=RunOptionsV2(subset_role="music_train_candidate", only_downloaded=True),
        config=AllocatorV2Config(),
    )
    training_rows = [json.loads(line) for line in paths.cara_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["training_manifest_rows"] == 2
    assert len(training_rows) == 2
    assert all(row["cara_source_pool_id"].startswith("CARA:AUD:1:") for row in training_rows)
    assert all(row["cara_pool_allocator_version"] == "v2" for row in training_rows)
    assert {row["primary_genre"] for row in training_rows} == {"Acoustic/Folk", "Jazz/Blues"}
