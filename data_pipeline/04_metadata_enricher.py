from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


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
]


def _normalise_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [stripped]
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [str(value)]


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_freesound_metadata(meta_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        analysis = payload.get("analysis") or {}
        music = _safe_get(analysis, "music")
        rhythm = _safe_get(analysis, "rhythm")
        tonal = _safe_get(analysis, "tonal")
        moods = _safe_get(analysis, "moods_mirex")
        voice = _safe_get(analysis, "voice_instrumental")

        row = {
            "source": "freesound",
            "source_id": str(payload.get("id", meta_path.stem)),
            "filename": str(payload.get("filename") or payload.get("name") or ""),
            "tags": _stringify(_normalise_list(payload.get("tags"))),
            "description": _stringify(payload.get("description", "")),
            "license": _stringify(payload.get("license", "")),
            "duration_s": _stringify(payload.get("duration", "")),
            "bpm": _stringify(_safe_get(rhythm, "bpm", default="")),
            "key": _stringify(_safe_get(tonal, "key_key", default="")),
            "mood_tags": _stringify(sorted([key for key, value in moods.items() if value] if isinstance(moods, dict) else [])),
            "genre_inferred": _stringify(_safe_get(music, "genre_dortmund", "value", default="")),
            "acoustic_electronic": _stringify(_safe_get(music, "acoustic", default="")),
            "voice_instrumental": _stringify(_safe_get(voice, "value", default="")),
            "essentia_available": "true" if analysis else "false",
            "fma_genre_top": "",
            "fma_genre_ids": "",
            "artist": _stringify(payload.get("username", "")),
            "album": "",
            "title": _stringify(payload.get("name", "")),
        }
        rows.append(row)
    return rows


def load_genre_lookup(genres_path: Path) -> dict[str, str]:
    if not genres_path.exists():
        return {}
    with genres_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("genre_id") or row.get("genreid") or row.get("id")): str(row.get("title") or "") for row in reader}


def load_fma_metadata(tracks_path: Path, genres_path: Path) -> list[dict[str, str]]:
    genre_lookup = load_genre_lookup(genres_path)
    if not tracks_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with tracks_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for track in reader:
            genre_ids = _normalise_list(track.get("genres") or track.get("track_genres") or "")
            genre_titles = [genre_lookup.get(genre_id, genre_id) for genre_id in genre_ids]
            row = {
                "source": "fma",
                "source_id": str(track.get("track_id") or track.get("id") or ""),
                "filename": str(track.get("filename") or ""),
                "tags": _stringify(_normalise_list(track.get("tags") or "")),
                "description": _stringify(track.get("description") or ""),
                "license": _stringify(track.get("license") or ""),
                "duration_s": _stringify(track.get("duration") or track.get("track_duration") or ""),
                "bpm": _stringify(track.get("bpm") or ""),
                "key": _stringify(track.get("key") or ""),
                "mood_tags": "[]",
                "genre_inferred": "",
                "acoustic_electronic": "",
                "voice_instrumental": "",
                "essentia_available": "false",
                "fma_genre_top": _stringify(track.get("genre_top") or ""),
                "fma_genre_ids": _stringify(genre_titles),
                "artist": _stringify(track.get("artist_name") or track.get("artist") or ""),
                "album": _stringify(track.get("album_title") or track.get("album") or ""),
                "title": _stringify(track.get("title") or track.get("track_title") or ""),
            }
            rows.append(row)
    return rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compute_coverage(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    total = len(rows)
    coverage: dict[str, dict[str, int]] = {}
    for column in OUTPUT_COLUMNS:
        non_empty = sum(1 for row in rows if str(row.get(column, "")).strip() not in {"", "[]"})
        coverage[column] = {
            "non_empty": non_empty,
            "empty": total - non_empty,
            "total": total,
        }
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freesound-meta-dir", default="data/freesound_meta")
    parser.add_argument("--fma-tracks", default="data/fma_meta/tracks_filtered.csv")
    parser.add_argument("--fma-genres", default="data/fma_meta/genres.csv")
    parser.add_argument("--output", default="data/enriched_metadata.csv")
    parser.add_argument("--coverage-output", default="data/enriched_metadata_coverage.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freesound_rows = load_freesound_metadata(Path(args.freesound_meta_dir))
    fma_rows = load_fma_metadata(Path(args.fma_tracks), Path(args.fma_genres))
    rows = freesound_rows + fma_rows
    write_rows(rows, Path(args.output))
    coverage = compute_coverage(rows)
    coverage_path = Path(args.coverage_output)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps({"total_rows": len(rows), "coverage_output": str(coverage_path)}, indent=2))


if __name__ == "__main__":
    main()
