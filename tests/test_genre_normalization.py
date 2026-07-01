from __future__ import annotations

import importlib.util
from pathlib import Path

from data_pipeline.genre_normalization import normalize_genre_fields, normalize_genre_label, normalize_pool_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalizer = _load_module("genre_label_normalizer_test", "data_pipeline/13_normalize_genre_labels.py")


def test_normalize_genre_label_prefers_slash_canonical_forms():
    assert normalize_genre_label("Percussion Drums") == "Percussion/Drums"
    assert normalize_genre_label("Voice Vocal") == "Voice/Vocal"
    assert normalize_genre_label("Jazz Blues") == "Jazz/Blues"
    assert normalize_genre_label("Hip Hop Beats") == "Hip-Hop/Beats"
    assert normalize_genre_label("Rock Metal") == "Rock/Metal"


def test_normalize_genre_fields_updates_pools_and_summaries():
    row = {
        "cara_tier1": "Jazz Blues",
        "primary_genre": "Percussion Drums",
        "cara_primary_pool": "Freesound-CC-BY-Jazz Blues",
        "cara_candidate_pools_json": ["Freesound-CC-BY-Jazz Blues", "Freesound-CC0-Voice Vocal"],
        "metadata_style_summary": "Percussion Drums | drum loop, acoustic",
        "broad_style_tokens": ["family:percussion/beats", "genre:percussion drums"],
    }

    normalized = normalize_genre_fields(row)

    assert normalized["cara_tier1"] == "Jazz/Blues"
    assert normalized["primary_genre"] == "Percussion/Drums"
    assert normalized["cara_primary_pool"] == "Freesound-CC-BY-Jazz/Blues"
    assert normalized["cara_candidate_pools_json"] == ["Freesound-CC-BY-Jazz/Blues", "Freesound-CC0-Voice/Vocal"]
    assert normalized["metadata_style_summary"] == "Percussion/Drums | drum loop, acoustic"
    assert normalized["broad_style_tokens"] == ["family:percussion/beats", "genre:percussion/drums"]


def test_normalize_rows_can_target_subset_only():
    rows = [
        {"source_id": "1", "include_in_subset": True, "cara_tier1": "Jazz Blues"},
        {"source_id": "2", "include_in_subset": False, "cara_tier1": "Voice Vocal"},
    ]

    normalized, report = normalizer.normalize_rows(rows, scope="subset", subset_role="music_train_candidate")

    assert normalized[0]["cara_tier1"] == "Jazz/Blues"
    assert normalized[1]["cara_tier1"] == "Voice Vocal"
    assert report["rows_in_scope"] == 1
    assert report["rows_changed"] == 1


def test_normalize_pool_name_handles_license_hyphens():
    assert normalize_pool_name("Freesound-CC-BY-Jazz Blues") == "Freesound-CC-BY-Jazz/Blues"
    assert normalize_pool_name("Freesound-CC0-Percussion Drums") == "Freesound-CC0-Percussion/Drums"
