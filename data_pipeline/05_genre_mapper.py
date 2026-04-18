from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows

TIER1_GENRES = [
    "Electronic",
    "Acoustic/Folk",
    "Jazz",
    "Classical/Orchestral",
    "Rock/Metal",
    "Hip-Hop/Beats",
    "Ambient/Drone",
    "Percussion/Drums",
    "Sound Effects",
    "Field Recording",
    "Voice/Vocal",
    "World/Traditional",
    "Experimental/Noise",
    "Unclassified",
]

TAG_RULES = {
    "Electronic": {"techno", "house", "synth", "edm", "electronic", "idm", "trance", "drumandbass", "dnb"},
    "Acoustic/Folk": {"folk", "acoustic", "guitar", "singer-songwriter", "banjo", "ukulele"},
    "Jazz": {"jazz", "bebop", "swing", "sax", "improv", "blues-jazz"},
    "Classical/Orchestral": {"classical", "orchestral", "strings", "piano", "chamber", "symphony"},
    "Rock/Metal": {"rock", "metal", "punk", "grunge", "guitar-riff"},
    "Hip-Hop/Beats": {"hiphop", "hip-hop", "rap", "beats", "boom-bap", "trap"},
    "Ambient/Drone": {"ambient", "drone", "pad", "atmospheric", "soundscape"},
    "Percussion/Drums": {"drums", "percussion", "snare", "kick", "cymbal", "tom"},
    "Sound Effects": {"sfx", "sound-effect", "impact", "whoosh", "ui", "fx"},
    "Field Recording": {"field-recording", "nature", "birds", "rain", "city", "ambience"},
    "Voice/Vocal": {"voice", "vocal", "speech", "choir", "spoken-word"},
    "World/Traditional": {"world", "traditional", "ethnic", "tribal", "celtic"},
    "Experimental/Noise": {"experimental", "noise", "glitch", "avant-garde", "abstract"},
}

GENRE_TOP_TO_TIER1 = {
    "electronic": "Electronic",
    "folk": "Acoustic/Folk",
    "jazz": "Jazz",
    "classical": "Classical/Orchestral",
    "rock": "Rock/Metal",
    "hip-hop": "Hip-Hop/Beats",
    "ambient": "Ambient/Drone",
    "instrumental": "Acoustic/Folk",
    "experimental": "Experimental/Noise",
    "spoken": "Voice/Vocal",
    "old-time / historic": "World/Traditional",
}

OUTPUT_COLUMNS = [
    "source",
    "source_id",
    "filename",
    "tags",
    "description",
    "license",
    "duration_s",
    "bpm",
    "key",
    "mood_tags",
    "genre_inferred",
    "acoustic_electronic",
    "voice_instrumental",
    "essentia_available",
    "fma_genre_top",
    "fma_genre_ids",
    "artist",
    "album",
    "title",
    "genre_tier1",
    "genre_tier2",
    "primary_pool",
    "candidate_pools",
]


def _parse_list_cell(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _slug(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def infer_tier1_from_row(row: dict[str, str]) -> tuple[str, Counter[str]]:
    scores: Counter[str] = Counter()
    tags = [_slug(tag) for tag in _parse_list_cell(row.get("tags", ""))]
    mood_tags = [_slug(tag) for tag in _parse_list_cell(row.get("mood_tags", ""))]
    genre_inferred = _slug(row.get("genre_inferred", ""))
    fma_genre_top = _slug(row.get("fma_genre_top", ""))
    description = _slug(row.get("description", ""))

    for tier1, keywords in TAG_RULES.items():
        for token in tags + mood_tags:
            if token in keywords:
                scores[tier1] += 3
        for keyword in keywords:
            if keyword in description:
                scores[tier1] += 1
            if genre_inferred and keyword in genre_inferred:
                scores[tier1] += 2

    if fma_genre_top:
        mapped = GENRE_TOP_TO_TIER1.get(fma_genre_top)
        if mapped:
            scores[mapped] += 5

    if row.get("voice_instrumental", "").lower() == "voice":
        scores["Voice/Vocal"] += 2
    if row.get("acoustic_electronic", "").lower() == "acoustic":
        scores["Acoustic/Folk"] += 1
    if row.get("acoustic_electronic", "").lower() == "electronic":
        scores["Electronic"] += 1

    if not scores:
        return "Unclassified", scores
    return scores.most_common(1)[0][0], scores


def infer_tier2(row: dict[str, str], tier1: str) -> str:
    tags = [_slug(tag) for tag in _parse_list_cell(row.get("tags", ""))]
    fma_top = row.get("fma_genre_top", "").strip()
    genre_inferred = row.get("genre_inferred", "").strip()
    for candidate in [genre_inferred, fma_top]:
        if candidate:
            return candidate
    if tags:
        return tags[0].title()
    return tier1


def map_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    mapped: list[dict[str, str]] = []
    for row in rows:
        tier1, scores = infer_tier1_from_row(row)
        tier2 = infer_tier2(row, tier1)
        candidate_pools = [name for name, _score in scores.most_common(3)] or [tier1]
        primary_pool = tier2
        mapped.append(
            {
                **row,
                "genre_tier1": tier1,
                "genre_tier2": tier2,
                "primary_pool": primary_pool,
                "candidate_pools": json.dumps(candidate_pools),
            }
        )
    return mapped


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    tier1_counts = Counter(row["genre_tier1"] for row in rows)
    tier2_counts = Counter(row["genre_tier2"] for row in rows)
    summary = {
        "total_rows": len(rows),
        "tier1_counts": dict(tier1_counts),
        "tier2_top_20": dict(tier2_counts.most_common(20)),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/attribution_manifest.jsonl")
    parser.add_argument("--output", default="data/genre_mapped.csv")
    parser.add_argument("--summary-output", default="data/genre_mapping_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.input).endswith(".jsonl"):
        rows = load_manifest_rows(Path(args.input))
    else:
        rows = load_rows(Path(args.input))
    mapped = map_rows(rows)
    write_rows(Path(args.output), mapped)
    write_summary(Path(args.summary_output), mapped)
    if str(args.input).endswith(".jsonl"):
        manifest_lookup = {str(row["source_id"]): row for row in mapped}
        manifest_rows = load_manifest_rows(Path(args.input))
        for row in manifest_rows:
            mapped_row = manifest_lookup.get(str(row.get("source_id", "")))
            if not mapped_row:
                continue
            row["genre_tier1"] = mapped_row["genre_tier1"]
            row["genre_tier2"] = mapped_row["genre_tier2"]
            row["candidate_pools"] = json.loads(mapped_row["candidate_pools"])
        save_manifest_rows(manifest_rows, Path(args.input))
    print(json.dumps({"total_rows": len(mapped), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
