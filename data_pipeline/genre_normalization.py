from __future__ import annotations

import json
from typing import Any


CANONICAL_GENRE_LABELS = {
    "Ambient/Drone",
    "Experimental/Noise",
    "Hip-Hop/Beats",
    "Classical/Orchestral",
    "Acoustic/Folk",
    "World/Traditional",
    "Field Recording",
    "Sound Effects",
    "Percussion/Drums",
    "Voice/Vocal",
    "Electronic",
    "Jazz/Blues",
    "Rock/Metal",
    "Children's Music",
    "Unclassified",
}

GENRE_LABEL_ALIASES = {
    "ambient drone": "Ambient/Drone",
    "ambient/drone": "Ambient/Drone",
    "experimental noise": "Experimental/Noise",
    "experimental/noise": "Experimental/Noise",
    "hip hop beats": "Hip-Hop/Beats",
    "hip-hop beats": "Hip-Hop/Beats",
    "hip-hop/beats": "Hip-Hop/Beats",
    "classical orchestral": "Classical/Orchestral",
    "classical/orchestral": "Classical/Orchestral",
    "acoustic folk": "Acoustic/Folk",
    "acoustic/folk": "Acoustic/Folk",
    "world traditional": "World/Traditional",
    "world/traditional": "World/Traditional",
    "field recording": "Field Recording",
    "sound effects": "Sound Effects",
    "percussion drums": "Percussion/Drums",
    "percussion/drums": "Percussion/Drums",
    "voice vocal": "Voice/Vocal",
    "voice/vocal": "Voice/Vocal",
    "jazz blues": "Jazz/Blues",
    "jazz/blues": "Jazz/Blues",
    "rock metal": "Rock/Metal",
    "rock/metal": "Rock/Metal",
    "kids music": "Children's Music",
    "children music": "Children's Music",
    "childrens music": "Children's Music",
    "children s music": "Children's Music",
    "children's music": "Children's Music",
    "unclassified": "Unclassified",
    "electronic": "Electronic",
}

GENRE_FIELDS = (
    "cara_tier1",
    "cara_tier2",
    "genre_tier1",
    "genre_tier2",
    "primary_genre",
    "secondary_genre",
    "api_genre_inferred",
)

POOL_NAME_FIELDS = (
    "cara_primary_pool",
    "primary_pool",
)

POOL_LIST_FIELDS = (
    "cara_candidate_pools_json",
    "candidate_pools",
)

SUMMARY_FIELDS = (
    "metadata_style_summary",
    "broad_style_summary",
)


def normalize_genre_label(value: Any, default: str = "", preserve_unknown: bool = True) -> str:
    text = " ".join(str(value or "").replace("_", " ").split()).strip()
    if not text:
        return default
    normalized = GENRE_LABEL_ALIASES.get(text.lower())
    if normalized:
        return normalized
    if text in CANONICAL_GENRE_LABELS:
        return text
    return text if preserve_unknown else default


def normalize_pool_name(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    pool_name = value.strip()
    for alias, canonical in sorted(GENRE_LABEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        suffix = alias.title()
        if pool_name.lower().endswith(f"-{alias}"):
            return f"{pool_name[:-(len(alias))]}{canonical}"
        if pool_name.endswith(f"-{suffix}"):
            return f"{pool_name[:-(len(suffix))]}{canonical}"
    return pool_name


def normalize_pool_list(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_pool_name(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return normalize_pool_name(value)
    if not isinstance(decoded, list):
        return value
    return json.dumps([normalize_pool_name(item) for item in decoded], ensure_ascii=False)


def normalize_genre_summary(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    for separator in (" | ", " :: "):
        if separator in value:
            prefix, suffix = value.split(separator, 1)
            normalized = normalize_genre_label(prefix, preserve_unknown=True)
            if normalized != prefix:
                return f"{normalized}{separator}{suffix}"
            return value
    return normalize_genre_label(value, preserve_unknown=True)


def normalize_style_tokens(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized_tokens: list[Any] = []
    for token in value:
        if isinstance(token, str) and token.startswith("genre:"):
            genre = token.removeprefix("genre:")
            normalized_tokens.append(f"genre:{normalize_genre_label(genre, preserve_unknown=True).lower()}")
        else:
            normalized_tokens.append(token)
    return normalized_tokens


def normalize_genre_fields(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    for field in GENRE_FIELDS:
        if field in updated:
            updated[field] = normalize_genre_label(updated.get(field), preserve_unknown=True)
    for field in POOL_NAME_FIELDS:
        if field in updated:
            updated[field] = normalize_pool_name(updated.get(field))
    for field in POOL_LIST_FIELDS:
        if field in updated:
            updated[field] = normalize_pool_list(updated.get(field))
    for field in SUMMARY_FIELDS:
        if field in updated:
            updated[field] = normalize_genre_summary(updated.get(field))
    for field in ("style_tokens", "broad_style_tokens"):
        if field in updated:
            updated[field] = normalize_style_tokens(updated.get(field))
    return updated
