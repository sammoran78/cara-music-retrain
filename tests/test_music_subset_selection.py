from __future__ import annotations

import csv
import importlib.util
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = _load_module("music_subset_selector_test", "data_pipeline/03b_select_music_subset.py")
replenish = _load_module("music_subset_replenish_test", "data_pipeline/03c_replenish_music_subset.py")


def test_canonicalize_tier_name_handles_space_and_slash_variants():
    assert selector.canonicalize_tier_name("Jazz Blues") == "Jazz/Blues"
    assert selector.canonicalize_tier_name("Hip Hop Beats") == "Hip-Hop/Beats"
    assert selector.canonicalize_tier_name("World Traditional") == "World/Traditional"
    assert selector.canonicalize_tier_name("Percussion Drums") == "Percussion/Drums"


def test_replenish_expansion_uses_seed_rows_from_requested_genres(tmp_path: Path):
    seed_csv = tmp_path / "seed.csv"
    with seed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "source_id",
                "raw_id",
                "title",
                "title_stem",
                "author",
                "license_raw",
                "license_normalized",
                "license_display",
                "url",
                "genre_tier1",
                "genre_tier2",
                "primary_pool",
                "candidate_pools",
                "auto_label_score",
                "auto_label_confidence",
                "auto_label_bucket",
                "matched_keywords",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source": "freesound",
                "source_id": "900001",
                "raw_id": "900001",
                "title": "folk_guitar.wav",
                "title_stem": "folk_guitar",
                "author": "tester",
                "license_raw": "cc0",
                "license_normalized": "cc0",
                "license_display": "CC0",
                "url": "https://freesound.org/people/tester/sounds/900001/",
                "genre_tier1": "Acoustic/Folk",
                "genre_tier2": "Acoustic/Folk",
                "primary_pool": "Freesound-CC0-Acoustic/Folk",
                "candidate_pools": '["Freesound-CC0-Acoustic/Folk"]',
                "auto_label_score": "5",
                "auto_label_confidence": "0.9",
                "auto_label_bucket": "high",
                "matched_keywords": '{"Acoustic/Folk":["guitar"]}',
            }
        )
        writer.writerow(
            {
                "source": "freesound",
                "source_id": "900002",
                "raw_id": "900002",
                "title": "jazz_sax.wav",
                "title_stem": "jazz_sax",
                "author": "tester",
                "license_raw": "cc-by",
                "license_normalized": "cc-by",
                "license_display": "CC-BY",
                "url": "https://freesound.org/people/tester/sounds/900002/",
                "genre_tier1": "Jazz/Blues",
                "genre_tier2": "Jazz/Blues",
                "primary_pool": "Freesound-CC-BY-Jazz/Blues",
                "candidate_pools": '["Freesound-CC-BY-Jazz/Blues"]',
                "auto_label_score": "4",
                "auto_label_confidence": "0.85",
                "auto_label_bucket": "medium",
                "matched_keywords": '{"Jazz/Blues":["sax"]}',
            }
        )

    args = Namespace(
        include_tiers="Electronic,Percussion/Drums,Acoustic/Folk,Classical/Orchestral,Jazz/Blues,Ambient/Drone,Experimental/Noise,Voice/Vocal",
        expansion_include_tiers="Acoustic/Folk,Jazz/Blues,World/Traditional,Hip-Hop/Beats",
        exclude_tiers="Sound Effects,Field Recording,Unclassified",
        allowed_licenses="cc0,cc-by,sampling+",
        min_bucket="medium",
        base_target_size=2,
        target_working_downloads=4,
        seed_labels_csv=str(seed_csv),
        block_authors_with_unavailable_count=3,
    )

    existing_rows = [
        {
            "source": "freesound",
            "source_id": "101",
            "cara_tier1": "Electronic",
            "cara_primary_pool": "Freesound-CC0-Electronic",
            "license_normalized": "cc0",
            "cara_auto_label_bucket": "high",
        },
        {
            "source": "freesound",
            "source_id": "102",
            "cara_tier1": "Percussion/Drums",
            "cara_primary_pool": "Freesound-CC0-Percussion/Drums",
            "license_normalized": "cc0",
            "cara_auto_label_bucket": "high",
        },
    ]

    selected, summary = replenish.select_replacements(
        rows=existing_rows,
        existing_candidate_ids={"101", "102"},
        blocked_ids=set(),
        replacements_needed=2,
        args=args,
    )

    assert len(selected) == 2
    assert {row["source_id"] for row in selected} == {"900001", "900002"}
    assert summary["selected_tier_counts"] == {"Acoustic/Folk": 1, "Jazz/Blues": 1}
