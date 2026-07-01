from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.attribution_utils import (
    display_license,
    infer_audio_extension,
    load_csv_rows,
    normalize_license,
    strip_audio_extension,
)
from data_pipeline.genre_normalization import normalize_genre_label
from data_pipeline.manifest_utils import index_manifest_rows, load_manifest_rows, save_manifest_rows

TAG_RULES = {
    "Electronic": {"techno", "house", "synth", "edm", "electronic", "idm", "trance", "drumandbass", "dnb"},
    "Acoustic/Folk": {"folk", "acoustic", "guitar", "singer songwriter", "banjo", "ukulele"},
    "Jazz Blues": {"jazz", "bebop", "swing", "sax", "improv", "blues", "blues jazz"},
    "Classical/Orchestral": {"classical", "orchestral", "strings", "string", "piano", "violin", "violins", "chamber", "symphony"},
    "Rock Metal": {"rock", "metal", "punk", "grunge", "guitar riff"},
    "Hip Hop Beats": {"hiphop", "hip hop", "rap", "beats", "boom bap", "trap"},
    "Ambient/Drone": {"ambient", "drone", "pad", "atmospheric", "soundscape"},
    "Percussion Drums": {"drums", "drum", "percussion", "snare", "kick", "cymbal", "tom", "hihat", "hat"},
    "Sound Effects": {"sfx", "sound effect", "impact", "whoosh", "ui", "fx", "effect"},
    "Field Recording": {"field recording", "nature", "birds", "rain", "city", "ambience", "ocean", "sea", "ferry", "boat", "woods", "beach", "walking", "subway", "platform", "footsteps", "recorded"},
    "Voice Vocal": {"voice", "vocal", "speech", "choir", "spoken word"},
    "World Traditional": {"world", "traditional", "ethnic", "tribal", "celtic"},
    "Experimental/Noise": {"experimental", "noise", "glitch", "avant garde", "abstract"},
}

GENRE_PRIORITY = {
    "Percussion Drums": 6,
    "Field Recording": 5,
    "Voice Vocal": 5,
    "Experimental/Noise": 4,
    "Ambient/Drone": 4,
    "Electronic": 4,
    "Classical/Orchestral": 3,
    "Jazz Blues": 3,
    "Hip Hop Beats": 3,
    "Acoustic/Folk": 2,
    "Rock Metal": 2,
    "Sound Effects": 2,
    "World Traditional": 1,
}

STYLE_IGNORE_TAGS = {
    "ableton",
    "ableton live",
    "ableton-live",
    "awesome",
    "live",
    "mono",
    "stereo",
    "set",
    "sample",
    "sound",
    "sounds",
}

STYLE_GROUPS: list[tuple[str, set[str]]] = [
    ("drum one-shot", {"snare", "kick", "cymbal", "hat", "hihat", "tom", "clap", "perc", "percussion", "drum"}),
    ("drum loop", {"drumloop", "loop", "beat", "beats", "groove", "bpm"}),
    ("acoustic", {"acoustic", "wood", "organic", "natural"}),
    ("electronic", {"electro", "electronic", "synth", "synthetic", "digital", "bitcrushed", "bit crushed"}),
    ("ambient", {"ambient", "drone", "atmospheric", "soundscape", "texture"}),
    ("noise", {"noise", "glitch", "abstract", "distorted", "harsh"}),
    ("melodic", {"melodic", "pitched", "tonal", "tuned"}),
    ("field ambience", {"field", "ferry", "ocean", "sea", "woods", "beach", "subway", "city", "nature", "ambience", "recording"}),
    ("voice", {"voice", "vocal", "speech", "spoken"}),
    ("piano", {"piano", "keys", "keyboard"}),
    ("guitar", {"guitar"}),
    ("bells", {"bell", "bells", "chime", "chiming", "tinkling"}),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    parser = argparse.ArgumentParser(description="Reconcile the attribution manifest against local downloads and sidecar metadata.")
    parser.add_argument("--manifest", default=freesound_cfg.get("attribution_manifest_path", "data/attribution_manifest.jsonl"))
    parser.add_argument("--progress", default=freesound_cfg.get("progress_path", "data/download_progress.json"))
    parser.add_argument("--audio-dir", default=freesound_cfg.get("output_dir", "data/freesound"))
    parser.add_argument("--meta-dir", default=freesound_cfg.get("meta_dir", "data/freesound_meta"))
    parser.add_argument("--attribution-csv", default=freesound_cfg.get("attribution_csv_local_path", "data/freesound_dataset_attribution.csv"))
    parser.add_argument("--default-subset-role", default="music_train_candidate")
    parser.add_argument("--auto-subset", action="store_true")
    return parser.parse_args()


def _safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").split())


def _word_set(*values: Any) -> set[str]:
    words: set[str] = set()
    for value in values:
        words.update(re.findall(r"[a-z0-9]+", _slug(value)))
    return words


def _matches_keyword(keyword: str, *, text: str, words: set[str]) -> bool:
    normalized = _slug(keyword)
    if not normalized:
        return False
    if " " in normalized:
        return normalized in text
    return normalized in words


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _audio_file_index(audio_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for file_path in sorted(audio_dir.rglob("*")):
        if file_path.is_file():
            indexed[file_path.stem] = file_path
    return indexed


def _meta_file_index(meta_dir: Path) -> dict[str, Path]:
    return {file_path.stem: file_path for file_path in sorted(meta_dir.glob("*.json")) if file_path.is_file()}


def _load_attribution_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_csv_rows(str(path))
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("id") or "").strip()
        if source_id:
            lookup[source_id] = row
    return lookup


def _infer_genres_and_summary(meta_payload: dict[str, Any]) -> tuple[str | None, str | None, list[str], str | None]:
    title = str(meta_payload.get("name") or "").strip()
    tags = [str(item).strip() for item in meta_payload.get("tags", []) if str(item).strip()]
    description = str(meta_payload.get("description") or "").strip()
    title_slug = _slug(title)
    tokens = [_slug(tag) for tag in tags]
    description_slug = _slug(description)
    title_words = _word_set(title)
    description_words = _word_set(description)
    word_space = _word_set(title, description, *tags)
    scores = Counter()
    for genre, keywords in TAG_RULES.items():
        for token in tokens:
            if token in keywords:
                scores[genre] += 3
        for keyword in keywords:
            if _matches_keyword(keyword, text=title_slug, words=title_words):
                scores[genre] += 2
            if _matches_keyword(keyword, text=description_slug, words=description_words):
                scores[genre] += 1
    # Domain-specific tie breakers so broad acoustic tags do not beat obvious drum/field cues.
    token_space = " ".join(tokens + [title_slug, description_slug])
    if any(_matches_keyword(word, text=token_space, words=word_space) for word in ["drum", "snare", "kick", "cymbal", "tom", "hihat", "hat", "percussion"]):
        scores["Percussion Drums"] += 4
    if any(_matches_keyword(word, text=token_space, words=word_space) for word in ["ferry", "ocean", "beach", "woods", "subway", "walking", "footsteps", "heartbeat"]):
        scores["Field Recording"] += 4
    if any(_matches_keyword(word, text=token_space, words=word_space) for word in ["voice", "vocal", "speech", "spoken"]):
        scores["Voice Vocal"] += 4
    if any(_matches_keyword(word, text=token_space, words=word_space) for word in ["piano", "violin", "violins", "orchestral", "strings"]):
        scores["Classical/Orchestral"] += 3
    if any(_matches_keyword(word, text=token_space, words=word_space) for word in ["guitar", "banjo", "ukulele", "folk"]) and not scores["Percussion Drums"]:
        scores["Acoustic/Folk"] += 2
    if scores:
        primary_genre = max(scores.items(), key=lambda item: (item[1], GENRE_PRIORITY.get(item[0], 0), item[0]))[0]
    else:
        primary_genre = None
    if primary_genre is None:
        if "heartbeat" in title_slug or "heartbeat" in description_slug:
            primary_genre = "Field Recording"
        elif "loop" in title_slug:
            primary_genre = "Electronic"
    canonical_style_tags: list[str] = []
    for label, keywords in STYLE_GROUPS:
        if any(_matches_keyword(keyword, text=token_space, words=word_space) for keyword in keywords):
            canonical_style_tags.append(label)
    if not canonical_style_tags:
        for tag in tags:
            normalized = _slug(tag)
            if not normalized or normalized in STYLE_IGNORE_TAGS:
                continue
            if len(canonical_style_tags) >= 4:
                break
            canonical_style_tags.append(tag)
    canonical_style_tags = list(dict.fromkeys(canonical_style_tags))

    secondary_genre = primary_genre
    if primary_genre == "Percussion Drums":
        if "drum loop" in canonical_style_tags:
            secondary_genre = "Drum Loop"
        elif "drum one-shot" in canonical_style_tags and "acoustic" in canonical_style_tags:
            secondary_genre = "Acoustic Percussion"
        elif "drum one-shot" in canonical_style_tags and "electronic" in canonical_style_tags:
            secondary_genre = "Electronic Percussion"
        else:
            secondary_genre = "Percussion Drums"
    elif primary_genre == "Experimental/Noise":
        if "noise" in canonical_style_tags and "melodic" in canonical_style_tags:
            secondary_genre = "Melodic Noise"
        elif "ambient" in canonical_style_tags:
            secondary_genre = "Noise Texture"
        else:
            secondary_genre = "Experimental/Noise"
    elif primary_genre == "Field Recording":
        secondary_genre = "Environmental Recording"
    elif primary_genre == "Classical/Orchestral" and "piano" in canonical_style_tags:
        secondary_genre = "Piano"
    elif primary_genre == "Acoustic/Folk" and "guitar" in canonical_style_tags:
        secondary_genre = "Guitar"

    primary_genre = normalize_genre_label(primary_genre, preserve_unknown=True) if primary_genre else None
    secondary_genre = normalize_genre_label(secondary_genre, preserve_unknown=True) if secondary_genre else None

    summary_parts: list[str] = []
    if primary_genre:
        summary_parts.append(primary_genre)
    if canonical_style_tags:
        summary_parts.append(", ".join(canonical_style_tags[:4]))
    elif description:
        summary_parts.append(description[:180])
    metadata_style_summary = " | ".join(part for part in summary_parts if part) or None
    return primary_genre, secondary_genre, canonical_style_tags, metadata_style_summary


def _build_row_from_meta(
    source_id: str,
    meta_payload: dict[str, Any],
    *,
    audio_path: Path | None,
    meta_path: Path | None,
    attribution_row: dict[str, Any] | None,
    default_subset_role: str | None,
    auto_subset: bool,
) -> dict[str, Any]:
    source_license = str((attribution_row or {}).get("license") or meta_payload.get("license") or "").strip()
    normalized_license = normalize_license(source_license)
    title = str((attribution_row or {}).get("title") or meta_payload.get("name") or (audio_path.name if audio_path else source_id)).strip()
    author = str((attribution_row or {}).get("author") or meta_payload.get("username") or "").strip()
    source_url = str((attribution_row or {}).get("url") or meta_payload.get("url") or f"https://freesound.org/sounds/{source_id}/").strip()
    primary_genre, secondary_genre, style_tags, metadata_style_summary = _infer_genres_and_summary(meta_payload)
    row: dict[str, Any] = {
        "source": "freesound",
        "source_id": source_id,
        "raw_id": int(source_id) if source_id.isdigit() else source_id,
        "url_sound_id": int(source_id) if source_id.isdigit() else None,
        "id_matches_url": True,
        "title": title,
        "title_stem": strip_audio_extension(title),
        "author": author,
        "license_raw": source_license,
        "license_normalized": normalized_license,
        "license_display": display_license(normalized_license),
        "licence_class": normalized_license,
        "license_class": normalized_license,
        "url": source_url,
        "file_extension": (audio_path.suffix.lower() if audio_path else infer_audio_extension(title, str(meta_payload.get("url") or ""))),
        "original_training_dataset": "Freesound",
        "original_training_manifest": "reconciled-local-freesound-downloads",
        "originally_in_stable_audio_open_small": True,
        "api_enrichment_status": "reconciled_from_local_meta",
        "api_last_checked_utc": utc_now(),
        "api_current_name": str(meta_payload.get("name") or "").strip() or None,
        "api_current_license_raw": str(meta_payload.get("license") or "").strip() or None,
        "api_current_license_normalized": normalized_license or None,
        "api_current_tags_json": meta_payload.get("tags") if isinstance(meta_payload.get("tags"), list) else [],
        "api_current_description": str(meta_payload.get("description") or "").strip() or None,
        "api_current_duration_s": meta_payload.get("duration"),
        "api_current_samplerate": meta_payload.get("samplerate"),
        "api_current_channels": meta_payload.get("channels"),
        "api_analysis_available": bool(meta_payload.get("analysis")),
        "api_bpm": None,
        "api_key": None,
        "api_voice_instrumental": None,
        "api_genre_inferred": primary_genre,
        "api_error_message": None,
        "cara_label_status": "unlabeled",
        "cara_label_source": None,
        "cara_label_updated_utc": None,
        "cara_tier1": primary_genre,
        "cara_tier2": secondary_genre,
        "cara_primary_pool": None,
        "cara_candidate_pools_json": [],
        "cara_soft_targets_json": [],
        "cara_family_codeword": None,
        "cara_codeword": None,
        "cara_auto_label_score": None,
        "cara_auto_label_confidence": None,
        "cara_auto_label_bucket": None,
        "cara_matched_keywords_json": {},
        "primary_genre": primary_genre,
        "secondary_genre": secondary_genre,
        "style_tags": style_tags,
        "metadata_style_summary": metadata_style_summary,
        "include_in_subset": bool(auto_subset and default_subset_role),
        "subset_role": default_subset_role if auto_subset and default_subset_role else None,
        "subset_note": "Reconciled from local Freesound downloads" if auto_subset and default_subset_role else None,
        "local_audio_path": _relative_to_root(audio_path) if audio_path else None,
        "local_meta_path": _relative_to_root(meta_path) if meta_path else None,
        "local_sidecar_path": None,
        "download_status": "downloaded" if audio_path else "metadata_only",
        "content_fingerprint": None,
        "manifest_notes": "Auto-created during manifest reconciliation from local download artifacts.",
    }
    return row
def _apply_meta_updates(row: dict[str, Any], meta_payload: dict[str, Any], attribution_row: dict[str, Any] | None = None) -> None:
    attribution_license = str((attribution_row or {}).get("license") or "").strip()
    sidecar_license = str(meta_payload.get("license") or "").strip()
    normalized_license = normalize_license(attribution_license or sidecar_license) if (attribution_license or sidecar_license) else ""
    primary_genre, secondary_genre, style_tags, metadata_style_summary = _infer_genres_and_summary(meta_payload)
    force_style_refresh = True
    row["api_enrichment_status"] = "reconciled_from_local_meta"
    row["api_last_checked_utc"] = utc_now()
    row["api_current_name"] = str(meta_payload.get("name") or "").strip() or row.get("api_current_name")
    row["api_current_license_raw"] = sidecar_license or row.get("api_current_license_raw")
    row["api_current_license_normalized"] = normalized_license or row.get("api_current_license_normalized")
    row["api_current_tags_json"] = meta_payload.get("tags") if isinstance(meta_payload.get("tags"), list) else row.get("api_current_tags_json", [])
    row["api_current_description"] = str(meta_payload.get("description") or "").strip() or row.get("api_current_description")
    row["api_current_duration_s"] = meta_payload.get("duration") if meta_payload.get("duration") is not None else row.get("api_current_duration_s")
    row["api_current_samplerate"] = meta_payload.get("samplerate") if meta_payload.get("samplerate") is not None else row.get("api_current_samplerate")
    row["api_current_channels"] = meta_payload.get("channels") if meta_payload.get("channels") is not None else row.get("api_current_channels")
    row["api_analysis_available"] = bool(meta_payload.get("analysis"))
    if primary_genre and (force_style_refresh or not row.get("api_genre_inferred")):
        row["api_genre_inferred"] = primary_genre
    if not row.get("title"):
        row["title"] = str(meta_payload.get("name") or "").strip() or row.get("title")
    if not row.get("title_stem"):
        row["title_stem"] = strip_audio_extension(str(row.get("title") or ""))
    if not row.get("author"):
        row["author"] = str(meta_payload.get("username") or "").strip() or row.get("author")
    if attribution_row and attribution_row.get("author"):
        row["author"] = str(attribution_row.get("author") or "").strip() or row.get("author")
    if not row.get("url"):
        row["url"] = str(meta_payload.get("url") or f"https://freesound.org/sounds/{row.get('source_id')}/").strip()
    if attribution_row and attribution_row.get("title"):
        row["title"] = str(attribution_row.get("title") or "").strip() or row.get("title")
        row["title_stem"] = strip_audio_extension(str(row.get("title") or ""))
    if attribution_row and attribution_row.get("url"):
        row["url"] = str(attribution_row.get("url") or "").strip() or row.get("url")
    canonical_raw_license = attribution_license or row.get("license_raw") or sidecar_license
    if canonical_raw_license:
        row["license_raw"] = canonical_raw_license
    if normalized_license:
        row["license_normalized"] = normalized_license
        row["license_display"] = display_license(normalized_license)
        row["licence_class"] = normalized_license
        row["license_class"] = normalized_license
    if primary_genre and (force_style_refresh or not row.get("primary_genre")):
        row["primary_genre"] = primary_genre
    if secondary_genre and (force_style_refresh or not row.get("secondary_genre")):
        row["secondary_genre"] = secondary_genre
    if style_tags and (force_style_refresh or not row.get("style_tags")):
        row["style_tags"] = style_tags
    if metadata_style_summary and (force_style_refresh or not row.get("metadata_style_summary")):
        row["metadata_style_summary"] = metadata_style_summary
    if primary_genre and (force_style_refresh or not row.get("cara_tier1")):
        row["cara_tier1"] = primary_genre
    if secondary_genre and (force_style_refresh or not row.get("cara_tier2")):
        row["cara_tier2"] = secondary_genre


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)
    audio_dir = Path(args.audio_dir)
    meta_dir = Path(args.meta_dir)
    attribution_csv = Path(args.attribution_csv)
    auto_subset = bool(args.auto_subset)
    default_subset_role = (args.default_subset_role or "").strip() or None

    rows = load_manifest_rows(manifest_path)
    manifest_by_id = index_manifest_rows(rows)
    progress = _safe_load_json(progress_path, {"completed_ids": [], "metadata_only_ids": [], "unavailable_ids": []})
    completed_ids = {str(item) for item in progress.get("completed_ids", [])}
    metadata_only_ids = {str(item) for item in progress.get("metadata_only_ids", [])}
    unavailable_ids = {str(item) for item in progress.get("unavailable_ids", [])}
    audio_by_id = _audio_file_index(audio_dir)
    meta_by_id = _meta_file_index(meta_dir)
    attribution_lookup = _load_attribution_lookup(attribution_csv)

    summary = Counter()

    for source_id, row in manifest_by_id.items():
        if str(row.get("source") or "freesound") != "freesound":
            continue
        audio_path = audio_by_id.get(source_id)
        meta_path = meta_by_id.get(source_id)
        meta_payload = _safe_load_json(meta_path, {}) if meta_path else {}
        attribution_row = attribution_lookup.get(source_id)

        if audio_path:
            next_audio_path = _relative_to_root(audio_path)
            if row.get("local_audio_path") != next_audio_path:
                row["local_audio_path"] = next_audio_path
                summary["audio_path_repaired"] += 1
            if row.get("download_status") != "downloaded":
                row["download_status"] = "downloaded"
                summary["downloaded_status_repaired"] += 1
            if auto_subset and default_subset_role and not row.get("subset_role"):
                row["include_in_subset"] = True
                row["subset_role"] = default_subset_role
                row["subset_note"] = "Marked as subset candidate during manifest reconciliation."
                summary["subset_backfilled_existing"] += 1
        elif source_id in metadata_only_ids:
            row["download_status"] = "metadata_only"
        elif source_id in unavailable_ids:
            row["download_status"] = "unavailable"

        if meta_path:
            next_meta_path = _relative_to_root(meta_path)
            if row.get("local_meta_path") != next_meta_path:
                row["local_meta_path"] = next_meta_path
                summary["meta_path_repaired"] += 1
            if isinstance(meta_payload, dict) and meta_payload:
                _apply_meta_updates(row, meta_payload, attribution_row=attribution_row)

        if not row.get("file_extension") and audio_path:
            row["file_extension"] = audio_path.suffix.lower()
        if not row.get("title_stem") and row.get("title"):
            row["title_stem"] = strip_audio_extension(str(row.get("title") or ""))

    appended_ids: list[str] = []
    for source_id, audio_path in audio_by_id.items():
        if source_id in manifest_by_id:
            continue
        meta_path = meta_by_id.get(source_id)
        meta_payload = _safe_load_json(meta_path, {}) if meta_path else {}
        row = _build_row_from_meta(
            source_id,
            meta_payload if isinstance(meta_payload, dict) else {},
            audio_path=audio_path,
            meta_path=meta_path,
            attribution_row=attribution_lookup.get(source_id),
            default_subset_role=default_subset_role,
            auto_subset=auto_subset,
        )
        rows.append(row)
        manifest_by_id[source_id] = row
        appended_ids.append(source_id)

    summary["rows_appended"] = len(appended_ids)
    summary["final_manifest_rows"] = len(rows)
    summary["final_downloaded_rows"] = sum(1 for row in rows if str(row.get("download_status") or "") == "downloaded")
    summary["final_subset_rows"] = sum(1 for row in rows if bool(row.get("include_in_subset")) and str(row.get("subset_role") or "") == (default_subset_role or ""))

    rows.sort(key=lambda row: (str(row.get("source") or ""), str(row.get("source_id") or "")))
    save_manifest_rows(rows, manifest_path)

    print(json.dumps(
        {
            "manifest_path": str(manifest_path),
            "audio_dir": str(audio_dir),
            "meta_dir": str(meta_dir),
            "completed_ids": len(completed_ids),
            "audio_files_found": len(audio_by_id),
            "meta_files_found": len(meta_by_id),
            "summary": dict(summary),
            "appended_sample": appended_ids[:20],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
