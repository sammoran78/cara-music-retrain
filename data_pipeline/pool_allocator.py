from __future__ import annotations

import json
import random
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_project_config
from data_pipeline.genre_normalization import CANONICAL_GENRE_LABELS, normalize_genre_label
from data_pipeline.manifest_utils import export_manifest_csv, load_manifest_rows, save_manifest_rows

POOL_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
POOL_CODE_GROUPS = (4, 4, 4)
DEFAULT_REGISTRY_VERSION = "2026.05.15"
DEFAULT_ENGINE_VERSION = "0.2.2"
CANONICAL_GENRES = set(CANONICAL_GENRE_LABELS)
GENRE_ALIASES = {
    "kids music": "Children's Music",
    "children music": "Children's Music",
    "childrens music": "Children's Music",
    "children s music": "Children's Music",
    "children songs": "Children's Music",
    "ambient drone": "Ambient/Drone",
    "experimental noise": "Experimental/Noise",
    "hip hop beats": "Hip-Hop/Beats",
    "classical orchestral": "Classical/Orchestral",
    "acoustic folk": "Acoustic/Folk",
    "world traditional": "World/Traditional",
    "field recording": "Field Recording",
    "sound effects": "Sound Effects",
}
GENRE_KEYWORDS = {
    "Electronic": {"electronic", "techno", "house", "trance", "edm", "hardstyle", "synth", "synthetic", "idm"},
    "Acoustic/Folk": {"acoustic", "folk", "guitar", "banjo", "ukulele", "mandolin"},
    "Jazz/Blues": {"jazz", "blues", "sax", "swing", "bebop", "double bass"},
    "Classical/Orchestral": {"classical", "orchestral", "piano", "violin", "cello", "strings", "double bass"},
    "Rock/Metal": {"rock", "metal", "punk", "grunge"},
    "Hip-Hop/Beats": {"hip hop", "hiphop", "rap", "trap", "boom bap"},
    "Ambient/Drone": {"ambient", "drone", "atmospheric", "soundscape", "pad"},
    "Percussion/Drums": {"drum", "drums", "percussion", "kick", "snare", "cymbal", "tom", "hihat", "hat", "beat"},
    "Sound Effects": {"throw", "toilet", "vomit", "door", "engine", "radio", "bark", "dog", "impact", "slam", "bang", "machine", "fx", "sfx"},
    "Field Recording": {"field", "ambience", "environment", "nature", "ocean", "beach", "city", "heartbeat", "footsteps"},
    "Voice/Vocal": {"voice", "vocal", "speech", "female", "male", "breath", "gasp", "teasing", "sultry"},
    "World/Traditional": {"world", "traditional", "ethnic", "tribal", "celtic"},
    "Experimental/Noise": {"experimental", "noise", "glitch", "feedback", "static", "crackle", "analog", "distorted"},
}
STYLE_IGNORE_TAGS = {
    "ableton",
    "ableton live",
    "ableton-live",
    "awesome",
    "live",
    "mono",
    "stereo",
    "sample",
    "sound",
    "sounds",
    "buddy",
}
STYLE_GROUPS: list[tuple[str, set[str]]] = [
    ("drum one-shot", {"snare", "kick", "cymbal", "hat", "hihat", "tom", "clap", "perc", "percussion", "drum"}),
    ("drum loop", {"drumloop", "loop", "beat", "beats", "groove", "bpm"}),
    ("acoustic", {"acoustic", "organic", "natural", "wood"}),
    ("electronic", {"electronic", "hardstyle", "synth", "synthetic", "digital"}),
    ("ambient", {"ambient", "drone", "atmospheric", "soundscape", "pad"}),
    ("noise", {"noise", "glitch", "feedback", "static", "crackle", "distorted"}),
    ("field ambience", {"field", "ambience", "environment", "nature", "ocean", "beach", "city", "footsteps", "heartbeat"}),
    ("voice", {"voice", "vocal", "speech", "female", "male", "breath", "gasp"}),
    ("animal", {"bark", "dog", "cat", "bird", "roar"}),
    ("mechanical", {"engine", "radio", "machine", "motor"}),
    ("impact", {"throw", "slam", "bang", "hit", "punch", "crash", "vomit", "toilet"}),
    ("piano", {"piano", "keys", "keyboard"}),
]

STATUS_ASSIGNED = "assigned"
STATUS_NEW_POOL = "new_pool_created"
STATUS_DUPLICATE = "duplicate_found"
STATUS_REVIEW = "review_required"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"
CHECKPOINT_MANIFEST_EVERY = 25
POOL_ASSIGNED_STATUSES = {STATUS_ASSIGNED, STATUS_NEW_POOL}
PROCESSED_ASSIGNMENT_STATUSES = {
    STATUS_ASSIGNED,
    STATUS_NEW_POOL,
    STATUS_DUPLICATE,
}
POOL_MANIFEST_FIELDS = (
    "cara_source_asset_id",
    "cara_source_pool_id",
    "cara_source_pool_assignment_status",
    "cara_source_pool_reason_codes",
    "cara_source_pool_review_required",
    "cara_source_pool_assignment_id",
    "cara_source_pool_last_assigned_utc",
)
_LOCAL_META_CACHE: dict[str, dict[str, Any] | None] = {}


@dataclass
class AllocatorConfig:
    max_pool_duration_seconds: int = 18_000
    max_artist_duration_seconds: int = 1_800
    min_style_score: float = 0.2
    min_assignment_score: float = 0.6
    min_pool_code_edit_distance: int = 5
    repair_threshold: int = 2
    fuzzy_duration_tolerance_seconds: float = 2.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "licence_match": 0.15,
            "territory_match": 0.1,
            "record_label_match": 0.25,
            "rights_holder_match": 0.25,
            "relaxed_metadata_compatibility": 0.15,
            "primary_genre_match": 0.2,
            "style_similarity": 0.2,
            "pool_balance": 0.1,
        }
    )
    penalties: dict[str, float] = field(
        default_factory=lambda: {
            "artist_concentration_near_limit": 0.2,
            "metadata_conflict": 0.3,
        }
    )


@dataclass
class AllocatorPaths:
    root: Path
    manifest_path: Path
    cara_manifest_path: Path
    cara_manifest_csv_path: Path
    progress_path: Path
    registry_dir: Path
    assets_path: Path
    pools_path: Path
    assignments_path: Path
    duplicates_path: Path
    runs_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "AllocatorPaths":
        registry_dir = root / "registry" / "pool_allocator"
        manifest_path = root / "data" / "attribution_manifest.jsonl"
        cara_manifest_path = root / "data" / "cara_pool_manifest.jsonl"
        return cls(
            root=root,
            manifest_path=manifest_path,
            cara_manifest_path=cara_manifest_path,
            cara_manifest_csv_path=cara_manifest_path.with_suffix(".csv"),
            progress_path=registry_dir / "progress.json",
            registry_dir=registry_dir,
            assets_path=registry_dir / "assets.jsonl",
            pools_path=registry_dir / "pools.json",
            assignments_path=registry_dir / "assignments.jsonl",
            duplicates_path=registry_dir / "duplicates.jsonl",
            runs_path=registry_dir / "runs.jsonl",
        )


@dataclass
class RunOptions:
    subset_role: str | None = None
    only_downloaded: bool = True
    limit: int | None = None
    allow_relaxed_metadata: bool = False
    start_fresh: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_allocator_config() -> AllocatorConfig:
    raw = load_project_config().get("pool_allocator", {})
    if not isinstance(raw, dict):
        return AllocatorConfig()
    return AllocatorConfig(
        max_pool_duration_seconds=int(raw.get("max_pool_duration_seconds", 18_000)),
        max_artist_duration_seconds=int(raw.get("max_artist_duration_seconds", 1_800)),
        min_style_score=float(raw.get("min_style_score", 0.2)),
        min_assignment_score=float(raw.get("min_assignment_score", 0.6)),
        min_pool_code_edit_distance=int(raw.get("min_pool_code_edit_distance", 5)),
        repair_threshold=int(raw.get("repair_threshold", 2)),
        fuzzy_duration_tolerance_seconds=float(raw.get("fuzzy_duration_tolerance_seconds", 2.0)),
        weights=dict(raw.get("weights", {}))
        if isinstance(raw.get("weights"), dict)
        else AllocatorConfig().weights,
        penalties=dict(raw.get("penalties", {}))
        if isinstance(raw.get("penalties"), dict)
        else AllocatorConfig().penalties,
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _empty_registry() -> dict[str, list[dict[str, Any]]]:
    return {"assets": [], "pools": [], "assignments": [], "duplicates": [], "runs": []}


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _empty_counts() -> dict[str, int]:
    return {
        STATUS_ASSIGNED: 0,
        STATUS_NEW_POOL: 0,
        STATUS_DUPLICATE: 0,
        STATUS_REVIEW: 0,
        STATUS_REJECTED: 0,
        STATUS_UNRESOLVED: 0,
    }


def _default_progress_state(options: RunOptions | None = None) -> dict[str, Any]:
    return {
        "status": "idle",
        "run_id": None,
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "current_phase": None,
        "current_asset": None,
        "current_asset_title": None,
        "current_pool_id": None,
        "processed_assets": 0,
        "total_assets": 0,
        "percent_complete": 0.0,
        "counts": _empty_counts(),
        "options": asdict(options) if options else None,
        "activity_log": [],
    }


def _append_progress_activity(
    progress: dict[str, Any],
    message: str,
    *,
    phase: str = "job",
    level: str = "info",
    asset_id: str | None = None,
    source_key: str | None = None,
    pool_id: str | None = None,
) -> None:
    activity_log = list(progress.get("activity_log", []))[-119:]
    activity_log.append(
        {
            "ts": utc_now(),
            "phase": phase,
            "level": level,
            "asset_id": asset_id,
            "source_key": source_key,
            "pool_id": pool_id,
            "message": message,
        }
    )
    progress["activity_log"] = activity_log


def _write_progress_state(paths: AllocatorPaths, progress: dict[str, Any]) -> None:
    _write_json(paths.progress_path, progress)


def read_progress_state(paths: AllocatorPaths) -> dict[str, Any]:
    return _load_json(paths.progress_path, _default_progress_state())


def _sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_sanitize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_sanitize_text(v) for v in value.values())
    return str(value).strip()


def _normalize_phrase(value: Any) -> str:
    text = _sanitize_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _title_case_or_empty(value: Any) -> str:
    text = _sanitize_text(value)
    return text if text else ""


def _word_set(*values: Any) -> set[str]:
    words: set[str] = set()
    for value in values:
        words.update(re.findall(r"[a-z0-9]+", _normalize_phrase(value)))
    return words


def _matches_keyword(keyword: str, *, text: str, words: set[str]) -> bool:
    normalized = _normalize_phrase(keyword)
    if not normalized:
        return False
    if " " in normalized:
        return normalized in text
    return normalized in words


def _normalize_genre(value: Any) -> str:
    return normalize_genre_label(value, preserve_unknown=True)


def _canonical_genre_from_value(value: Any) -> str:
    genre = _normalize_genre(value)
    return genre if genre in CANONICAL_GENRES else ""


def _infer_canonical_genre(row: dict[str, Any]) -> str:
    for key in ("primary_genre", "api_genre_inferred", "cara_tier1", "cara_tier2", "secondary_genre"):
        canonical = _canonical_genre_from_value(row.get(key))
        if canonical:
            return canonical

    text_values = [
        row.get("primary_genre"),
        row.get("secondary_genre"),
        row.get("cara_tier1"),
        row.get("cara_tier2"),
        row.get("title"),
        row.get("metadata_style_summary"),
        row.get("api_current_description"),
        row.get("style_tags"),
        row.get("api_current_tags_json"),
    ]
    text = " ".join(_normalize_phrase(value) for value in text_values if _sanitize_text(value))
    words = _word_set(*text_values)
    scores = Counter()
    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if _matches_keyword(keyword, text=text, words=words):
                scores[genre] += 1
    if not scores:
        return ""
    best_score = max(scores.values())
    candidates = sorted(genre for genre, score in scores.items() if score == best_score)
    return candidates[0] if candidates else ""


def _canonical_style_tags(row: dict[str, Any], primary_genre: str) -> list[str]:
    raw_tags = _normalize_string_list(row.get("style_tags") or row.get("api_current_tags_json"))
    text_values = [row.get("title"), row.get("metadata_style_summary"), row.get("api_current_description"), raw_tags]
    text = " ".join(_normalize_phrase(value) for value in text_values if _sanitize_text(value))
    words = _word_set(*text_values)

    tags: list[str] = []
    for label, keywords in STYLE_GROUPS:
        if any(_matches_keyword(keyword, text=text, words=words) for keyword in keywords):
            tags.append(label)

    if primary_genre == "Sound Effects" and "animal" in tags:
        tags = [tag for tag in tags if tag not in {"voice"}]
    if primary_genre == "Voice/Vocal" and "voice" not in tags:
        tags.insert(0, "voice")

    return list(dict.fromkeys(tags[:4]))


def _normalize_territory(value: Any) -> str:
    text = _sanitize_text(value).upper()
    return text if text else ""


def _normalize_licence_class(row: dict[str, Any]) -> str:
    for key in ("licence_class", "license_class", "license_normalized", "api_current_license_normalized", "license_raw"):
        value = _sanitize_text(row.get(key))
        if value:
            return value
    return ""


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                raw_items = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                raw_items = [part.strip() for part in raw.split(",")]
        else:
            raw_items = [part.strip() for part in raw.split(",")]
    else:
        raw_items = [value]
    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        normalized = _sanitize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_manifest_path(value: Any) -> Path | None:
    text = _sanitize_text(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_local_meta(value: Any) -> dict[str, Any] | None:
    path = _resolve_manifest_path(value)
    if path is None:
        return None
    key = str(path)
    if key in _LOCAL_META_CACHE:
        return _LOCAL_META_CACHE[key]
    if not path.exists():
        _LOCAL_META_CACHE[key] = None
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    _LOCAL_META_CACHE[key] = payload if isinstance(payload, dict) else None
    return _LOCAL_META_CACHE[key]


def _duration_from_row_or_sidecar(row: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "api_current_duration_s", "duration", "preview_duration"):
        duration = _safe_float(row.get(key))
        if duration is not None:
            return duration
    meta_payload = _read_local_meta(row.get("local_meta_path"))
    if meta_payload:
        for key in ("duration", "duration_seconds"):
            duration = _safe_float(meta_payload.get(key))
            if duration is not None:
                return duration
    return None


def _language_from_value(value: Any) -> str:
    text = _sanitize_text(value).lower()
    if not text:
        return ""
    mapping = {"english": "en"}
    return mapping.get(text, text)


def _version_title_from_row(row: dict[str, Any]) -> str:
    for key in ("version_title", "mix_name", "version", "subtitle"):
        value = _sanitize_text(row.get(key))
        if value:
            return value
    return ""


def _bpm_bucket(bpm: float | None) -> str:
    if bpm is None:
        return ""
    if bpm < 80:
        return "bpm_slow"
    if bpm < 120:
        return "bpm_mid"
    return "bpm_fast"


def _is_unknown_artist(artist_ids: list[str], artist_primary: str) -> bool:
    return not artist_ids and not artist_primary


def _base_style_tokens(
    secondary_genre: str,
    style_tags: list[str],
    mood_tags: list[str],
    instrumentation_tags: list[str],
    version_title: str,
    bpm: float | None,
    language: str,
) -> list[str]:
    tokens: list[str] = []
    if secondary_genre:
        tokens.append(f"secondary:{_normalize_phrase(secondary_genre)}")
    for item in style_tags:
        normalized = _normalize_phrase(item)
        if normalized:
            tokens.append(f"style:{normalized}")
    for item in mood_tags:
        normalized = _normalize_phrase(item)
        if normalized:
            tokens.append(f"mood:{normalized}")
    for item in instrumentation_tags:
        normalized = _normalize_phrase(item)
        if normalized:
            tokens.append(f"inst:{normalized}")
    version_bits = _normalize_phrase(version_title)
    for bit in version_bits.split():
        if bit:
            tokens.append(f"version:{bit}")
    bucket = _bpm_bucket(bpm)
    if bucket:
        tokens.append(bucket)
    if language:
        tokens.append(f"lang:{_normalize_phrase(language)}")
    return list(dict.fromkeys(tokens))


def _build_style_summary(primary_genre: str, tokens: list[str]) -> str:
    parts = [primary_genre] if primary_genre else []
    human_tokens = [
        token.split(":", 1)[1] if ":" in token else token.replace("_", " ")
        for token in tokens[:8]
        if not token.startswith("lang:")
    ]
    if human_tokens:
        parts.append(", ".join(human_tokens))
    return " | ".join(part for part in parts if part)


def jaccard_similarity(left: list[str], right: list[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _pool_checksum(code: str) -> str:
    compact = code.replace("-", "")
    total = sum((idx + 1) * POOL_ALPHABET.index(char) for idx, char in enumerate(compact) if char in POOL_ALPHABET)
    return f"{POOL_ALPHABET[total % len(POOL_ALPHABET)]}{POOL_ALPHABET[(total // len(POOL_ALPHABET)) % len(POOL_ALPHABET)]}"


def _generate_pool_code(existing_codes: list[str], config: AllocatorConfig, rng: random.Random) -> str:
    total_length = sum(POOL_CODE_GROUPS)
    while True:
        chars = [rng.choice(POOL_ALPHABET) for _ in range(total_length)]
        pieces: list[str] = []
        cursor = 0
        for size in POOL_CODE_GROUPS:
            pieces.append("".join(chars[cursor : cursor + size]))
            cursor += size
        code = "-".join(pieces)
        if all(edit_distance(code.replace("-", ""), existing.replace("-", "")) >= config.min_pool_code_edit_distance for existing in existing_codes):
            return code


def _next_prefixed_id(rows: list[dict[str, Any]], prefix: str) -> str:
    highest = 0
    for row in rows:
        candidate_values = []
        direct = _sanitize_text(row.get(f"{prefix}_id") or row.get("id"))
        if direct:
            candidate_values.append(direct)
        for key, raw_value in row.items():
            if key == "id" or key.endswith("_id"):
                value = _sanitize_text(raw_value)
                if value:
                    candidate_values.append(value)
        for value in candidate_values:
            if not value.startswith(f"{prefix}_"):
                continue
            try:
                highest = max(highest, int(value.split("_", 1)[1]))
            except ValueError:
                continue
    return f"{prefix}_{highest + 1:06d}"


def _pool_lookup_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _sanitize_text(row.get(key))
        if value:
            return value
    return ""


def normalize_manifest_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    source = _sanitize_text(row.get("source")) or "unknown"
    source_id = _sanitize_text(row.get("source_id")) or _sanitize_text(row.get("raw_id")) or f"row_{index}"
    artist_primary = _pool_lookup_value(row, "artist_primary", "author", "artist")
    artist_ids = _normalize_string_list(row.get("artist_ids"))
    if not artist_ids and artist_primary:
        artist_ids = [artist_primary]
    primary_genre = _infer_canonical_genre(row)
    secondary_genre = _canonical_genre_from_value(row.get("secondary_genre") or row.get("cara_tier1")) or primary_genre
    if secondary_genre == primary_genre:
        secondary_genre = ""
    style_tags = _canonical_style_tags(row, primary_genre) or _normalize_string_list(row.get("style_tags") or row.get("api_current_tags_json"))
    mood_tags = _normalize_string_list(row.get("mood_tags"))
    instrumentation_tags = _normalize_string_list(row.get("instrumentation_tags"))
    version_title = _version_title_from_row(row)
    duration = _duration_from_row_or_sidecar(row)
    bpm = _safe_float(row.get("bpm"))
    if bpm is None:
        bpm = _safe_float(row.get("api_bpm"))
    language = _language_from_value(row.get("language"))
    style_tokens = _base_style_tokens(
        secondary_genre=secondary_genre,
        style_tags=style_tags,
        mood_tags=mood_tags,
        instrumentation_tags=instrumentation_tags,
        version_title=version_title,
        bpm=bpm,
        language=language,
    )
    metadata_style_summary = _build_style_summary(primary_genre, style_tokens) if (primary_genre or style_tokens) else _sanitize_text(row.get("metadata_style_summary"))
    record_label = _pool_lookup_value(row, "record_label")
    rights_holder = _pool_lookup_value(row, "rights_holder", "publisher")
    territory = _normalize_territory(row.get("territory")) or "GLOBAL"
    licence_class = _normalize_licence_class(row)
    content_hash = _pool_lookup_value(row, "content_hash")
    if not content_hash and row.get("content_fingerprint"):
        content_hash = _sanitize_text(row.get("content_fingerprint"))
    fingerprint = _pool_lookup_value(row, "audio_fingerprint", "content_fingerprint")
    return {
        "source_key": f"{source}:{source_id}",
        "source": source,
        "source_id": source_id,
        "source_file_path": _pool_lookup_value(row, "local_audio_path"),
        "local_meta_path": _pool_lookup_value(row, "local_meta_path"),
        "duration_seconds": duration or 0.0,
        "isrc": _pool_lookup_value(row, "isrc"),
        "iswc": _pool_lookup_value(row, "iswc"),
        "title": _pool_lookup_value(row, "title"),
        "version_title": version_title,
        "artist_primary": artist_primary,
        "artist_ids": artist_ids,
        "featured_artists": _normalize_string_list(row.get("featured_artists")),
        "record_label": record_label,
        "publisher": _pool_lookup_value(row, "publisher"),
        "rights_holder": rights_holder,
        "licence_class": licence_class,
        "territory": territory,
        "primary_genre": primary_genre,
        "secondary_genre": secondary_genre,
        "style_tags": style_tags,
        "mood_tags": mood_tags,
        "instrumentation_tags": instrumentation_tags,
        "metadata_style_summary": metadata_style_summary,
        "style_tokens": style_tokens,
        "bpm": bpm,
        "language": language,
        "audio_fingerprint": fingerprint,
        "content_hash": content_hash,
        "manifest_source": "attribution_manifest.jsonl",
        "download_status": _pool_lookup_value(row, "download_status"),
    }


def _review_result(asset: dict[str, Any], reason_codes: list[str], message: str | None = None) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "assignment_status": STATUS_REVIEW,
        "pool_id": None,
        "pool_was_created": False,
        "reason_codes": reason_codes,
        "review_required": True,
        "message": message,
    }


def _rejected_result(asset: dict[str, Any], reason_codes: list[str], message: str | None = None) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "assignment_status": STATUS_REJECTED,
        "pool_id": None,
        "pool_was_created": False,
        "reason_codes": reason_codes,
        "review_required": False,
        "message": message,
    }


def _duplicate_result(asset: dict[str, Any], existing_assignment: dict[str, Any] | None, reason_codes: list[str], duplicate_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "assignment_status": STATUS_DUPLICATE,
        "pool_id": existing_assignment.get("pool_id") if existing_assignment else None,
        "pool_was_created": False,
        "reason_codes": reason_codes,
        "review_required": bool(duplicate_record.get("review_required")),
        "duplicate_record": duplicate_record,
    }


def _load_registry(paths: AllocatorPaths) -> dict[str, list[dict[str, Any]]]:
    registry = _empty_registry()
    registry["assets"] = _load_jsonl(paths.assets_path)
    registry["pools"] = _load_json(paths.pools_path, [])
    registry["assignments"] = _load_jsonl(paths.assignments_path)
    registry["duplicates"] = _load_jsonl(paths.duplicates_path)
    registry["runs"] = _load_jsonl(paths.runs_path)
    return registry


def _persist_registry(paths: AllocatorPaths, registry: dict[str, list[dict[str, Any]]]) -> None:
    _write_jsonl(paths.assets_path, registry["assets"])
    _write_json(paths.pools_path, registry["pools"])
    _write_jsonl(paths.assignments_path, registry["assignments"])
    _write_jsonl(paths.duplicates_path, registry["duplicates"])
    _write_jsonl(paths.runs_path, registry["runs"])


def _clear_pool_manifest_fields(row: dict[str, Any]) -> None:
    for field in POOL_MANIFEST_FIELDS:
        row.pop(field, None)


def _is_cara_manifest_training_row(row: dict[str, Any]) -> bool:
    status = _sanitize_text(row.get("cara_source_pool_assignment_status"))
    if status not in POOL_ASSIGNED_STATUSES:
        return False
    return bool(_sanitize_text(row.get("cara_source_pool_id")))


def _cara_manifest_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in manifest_rows if _is_cara_manifest_training_row(row)]


def _save_manifests(manifest_rows: list[dict[str, Any]], paths: AllocatorPaths) -> None:
    cara_manifest_rows = _cara_manifest_rows(manifest_rows)
    save_manifest_rows(manifest_rows, paths.manifest_path, export_csv=False)
    export_manifest_csv(manifest_rows, paths.manifest_path.with_suffix(".csv"))
    save_manifest_rows(cara_manifest_rows, paths.cara_manifest_path, export_csv=False)
    export_manifest_csv(cara_manifest_rows, paths.cara_manifest_csv_path)


def _persist_checkpoint(
    *,
    paths: AllocatorPaths,
    registry: dict[str, list[dict[str, Any]]],
    run_summary: dict[str, Any],
    progress_state: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    write_manifests: bool = False,
) -> None:
    registry["runs"][-1] = run_summary
    _persist_registry(paths, registry)
    _write_progress_state(paths, progress_state)
    if write_manifests:
        _save_manifests(manifest_rows, paths)


def _candidate_manifest_rows(
    manifest_rows: list[dict[str, Any]],
    options: RunOptions,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(manifest_rows):
        if options.subset_role and _sanitize_text(row.get("subset_role")) != options.subset_role:
            continue
        if options.only_downloaded and _sanitize_text(row.get("download_status")) != "downloaded":
            continue
        rows.append((index, row))
        if options.limit is not None and len(rows) >= options.limit:
            break
    return rows


def _existing_assignment_for_asset(assignments: list[dict[str, Any]], asset_id: str) -> dict[str, Any] | None:
    for assignment in reversed(assignments):
        if assignment.get("asset_id") == asset_id and assignment.get("assignment_status") in {
            STATUS_ASSIGNED,
            STATUS_NEW_POOL,
            STATUS_DUPLICATE,
        }:
            return assignment
    return None


def _is_current_assignment(assignment: dict[str, Any] | None) -> bool:
    if not assignment:
        return False
    return bool(
        _sanitize_text(assignment.get("assignment_status")) in PROCESSED_ASSIGNMENT_STATUSES
        and _sanitize_text(assignment.get("allocation_engine_version")) == DEFAULT_ENGINE_VERSION
    )


def _duplicate_match_reason(asset: dict[str, Any], other: dict[str, Any]) -> tuple[bool, str]:
    if asset.get("isrc") and asset.get("isrc") == other.get("isrc"):
        return True, "DUPLICATE_ISRC_FOUND"
    if asset.get("iswc") and asset.get("iswc") == other.get("iswc"):
        return True, "DUPLICATE_ISWC_FOUND"
    if asset.get("audio_fingerprint") and asset.get("audio_fingerprint") == other.get("audio_fingerprint"):
        return True, "DUPLICATE_FINGERPRINT_FOUND"
    if asset.get("content_hash") and asset.get("content_hash") == other.get("content_hash"):
        return True, "DUPLICATE_CONTENT_HASH_FOUND"
    return False, ""


def _is_fuzzy_duplicate(asset: dict[str, Any], other: dict[str, Any], config: AllocatorConfig) -> bool:
    title_match = _normalize_phrase(asset.get("title")) == _normalize_phrase(other.get("title"))
    artist_match = _normalize_phrase(asset.get("artist_primary")) == _normalize_phrase(other.get("artist_primary"))
    left_duration = float(asset.get("duration_seconds") or 0.0)
    right_duration = float(other.get("duration_seconds") or 0.0)
    duration_match = abs(left_duration - right_duration) <= config.fuzzy_duration_tolerance_seconds
    return bool(title_match and artist_match and duration_match and (title_match or artist_match))


def _find_duplicate(asset: dict[str, Any], registry: dict[str, list[dict[str, Any]]], config: AllocatorConfig) -> tuple[str | None, dict[str, Any] | None]:
    for other in registry["assets"]:
        if other.get("source_key") == asset.get("source_key"):
            continue
        matched, reason = _duplicate_match_reason(asset, other)
        if matched:
            return reason, other
    for other in registry["assets"]:
        if other.get("source_key") == asset.get("source_key"):
            continue
        if _is_fuzzy_duplicate(asset, other, config):
            return "POTENTIAL_DUPLICATE_FUZZY_MATCH", other
    return None, None


def _license_matches(asset: dict[str, Any], pool: dict[str, Any]) -> bool:
    return asset.get("licence_class") == pool.get("licence_class")


def _territory_matches(asset: dict[str, Any], pool: dict[str, Any]) -> bool:
    return asset.get("territory") == pool.get("territory")


def _rights_or_label_matches(asset: dict[str, Any], pool: dict[str, Any], allow_relaxed_metadata: bool = False) -> bool:
    asset_label = _normalize_phrase(asset.get("record_label"))
    asset_rights = _normalize_phrase(asset.get("rights_holder"))
    pool_label = _normalize_phrase(pool.get("record_label"))
    pool_rights = _normalize_phrase(pool.get("rights_holder_group"))
    if not asset_label and not asset_rights:
        return allow_relaxed_metadata and not pool_label and not pool_rights
    return bool((asset_label and asset_label == pool_label) or (asset_rights and asset_rights == pool_rights))


def _has_duration_capacity(asset: dict[str, Any], pool: dict[str, Any], config: AllocatorConfig) -> bool:
    return float(pool.get("current_duration_seconds", 0.0)) + float(asset.get("duration_seconds", 0.0)) <= config.max_pool_duration_seconds


def _artist_cap_ok(asset: dict[str, Any], pool: dict[str, Any], config: AllocatorConfig) -> bool:
    artist_map = dict(pool.get("artist_duration_seconds", {}))
    duration = float(asset.get("duration_seconds", 0.0))
    artist_ids = list(asset.get("artist_ids") or [])
    if _is_unknown_artist(artist_ids, _sanitize_text(asset.get("artist_primary"))):
        current = float(artist_map.get("__unknown__", 0.0))
        return current + duration <= config.max_artist_duration_seconds
    artist_key = artist_ids[0] if artist_ids else _sanitize_text(asset.get("artist_primary"))
    current = float(artist_map.get(artist_key, 0.0))
    return current + duration <= config.max_artist_duration_seconds


def _primary_genre_matches(asset: dict[str, Any], pool: dict[str, Any]) -> bool:
    return _normalize_phrase(asset.get("primary_genre")) == _normalize_phrase(pool.get("primary_genre"))


def calculate_style_score(asset: dict[str, Any], pool: dict[str, Any]) -> float:
    pool_tokens = _normalize_string_list(pool.get("style_profile", {}).get("style_tokens", []))
    return jaccard_similarity(asset.get("style_tokens", []), pool_tokens)


def calculate_candidate_score(
    asset: dict[str, Any],
    pool: dict[str, Any],
    style_score: float,
    config: AllocatorConfig,
    *,
    allow_relaxed_metadata: bool = False,
) -> float:
    weights = config.weights
    penalties = config.penalties
    licence_match = 1.0 if _license_matches(asset, pool) else 0.0
    territory_match = 1.0 if _territory_matches(asset, pool) else 0.0
    label_match = 1.0 if _normalize_phrase(asset.get("record_label")) == _normalize_phrase(pool.get("record_label")) and asset.get("record_label") else 0.0
    rights_match = 1.0 if _normalize_phrase(asset.get("rights_holder")) == _normalize_phrase(pool.get("rights_holder_group")) and asset.get("rights_holder") else 0.0
    asset_missing_rights = not _sanitize_text(asset.get("record_label")) and not _sanitize_text(asset.get("rights_holder"))
    pool_missing_rights = not _sanitize_text(pool.get("record_label")) and not _sanitize_text(pool.get("rights_holder_group"))
    relaxed_metadata_bonus = 1.0 if allow_relaxed_metadata and asset_missing_rights and pool_missing_rights else 0.0
    genre_match = 1.0 if _primary_genre_matches(asset, pool) else 0.0
    capacity_ratio = float(pool.get("current_duration_seconds", 0.0)) / max(float(pool.get("pool_duration_cap_seconds", 1.0)), 1.0)
    pool_balance = max(0.0, 1.0 - capacity_ratio)
    score = (
        licence_match * weights.get("licence_match", 0.15)
        + territory_match * weights.get("territory_match", 0.1)
        + label_match * weights.get("record_label_match", 0.25)
        + rights_match * weights.get("rights_holder_match", 0.25)
        + relaxed_metadata_bonus * weights.get("relaxed_metadata_compatibility", 0.15)
        + genre_match * weights.get("primary_genre_match", 0.2)
        + style_score * weights.get("style_similarity", 0.2)
        + pool_balance * weights.get("pool_balance", 0.1)
    )
    artist_ids = list(asset.get("artist_ids") or [])
    artist_key = artist_ids[0] if artist_ids else (_sanitize_text(asset.get("artist_primary")) or "__unknown__")
    artist_duration = float(pool.get("artist_duration_seconds", {}).get(artist_key, 0.0))
    if artist_duration >= config.max_artist_duration_seconds * 0.8:
        score -= penalties.get("artist_concentration_near_limit", 0.2)
    if asset_missing_rights and not allow_relaxed_metadata:
        score -= penalties.get("metadata_conflict", 0.3)
    return score


def _build_reason_codes(base: list[str], status: str) -> list[str]:
    codes = list(base)
    if status == STATUS_NEW_POOL:
        codes.extend(["NO_COMPATIBLE_POOL", "NEW_POOL_CREATED"])
    return list(dict.fromkeys(codes))


def _create_pool_from_asset(asset: dict[str, Any], registry: dict[str, list[dict[str, Any]]], config: AllocatorConfig, rng: random.Random) -> dict[str, Any]:
    existing_codes = [str(pool.get("pool_code") or "") for pool in registry["pools"] if pool.get("pool_code")]
    code = _generate_pool_code(existing_codes, config, rng)
    checksum = _pool_checksum(code)
    pool_id = f"CARA:AUD:1:{code}:{checksum}"
    now = utc_now()
    return {
        "pool_id": pool_id,
        "modality": "audio",
        "schema_version": "1",
        "pool_code": code,
        "checksum": checksum,
        "pool_duration_cap_seconds": config.max_pool_duration_seconds,
        "current_duration_seconds": 0.0,
        "asset_count": 0,
        "licence_class": asset.get("licence_class"),
        "territory": asset.get("territory"),
        "rights_holder_group": asset.get("rights_holder"),
        "record_label": asset.get("record_label"),
        "primary_genre": asset.get("primary_genre"),
        "style_profile": {
            "secondary_genres": [asset.get("secondary_genre")] if asset.get("secondary_genre") else [],
            "style_tags": asset.get("style_tags", []),
            "style_summary": asset.get("metadata_style_summary", ""),
            "style_tokens": asset.get("style_tokens", []),
        },
        "artist_duration_seconds": {},
        "asset_ids": [],
        "created_at": now,
        "updated_at": now,
        "pool_status": "active",
        "repair_threshold": config.repair_threshold,
    }


def _update_pool_with_asset(pool: dict[str, Any], asset: dict[str, Any]) -> None:
    duration = float(asset.get("duration_seconds", 0.0))
    pool["current_duration_seconds"] = float(pool.get("current_duration_seconds", 0.0)) + duration
    pool["asset_count"] = int(pool.get("asset_count", 0)) + 1
    asset_ids = list(pool.get("asset_ids", []))
    if asset["asset_id"] not in asset_ids:
        asset_ids.append(asset["asset_id"])
    pool["asset_ids"] = asset_ids
    artist_map = dict(pool.get("artist_duration_seconds", {}))
    artist_ids = list(asset.get("artist_ids") or [])
    if _is_unknown_artist(artist_ids, _sanitize_text(asset.get("artist_primary"))):
        artist_key = "__unknown__"
    else:
        artist_key = artist_ids[0] if artist_ids else _sanitize_text(asset.get("artist_primary"))
    artist_map[artist_key] = float(artist_map.get(artist_key, 0.0)) + duration
    pool["artist_duration_seconds"] = artist_map
    pool["updated_at"] = utc_now()


def _assignment_row(asset: dict[str, Any], pool: dict[str, Any], assignment_id: str, status: str, reason_codes: list[str], private_score: float | None, review_required: bool = False) -> dict[str, Any]:
    return {
        "assignment_id": assignment_id,
        "asset_id": asset["asset_id"],
        "pool_id": pool["pool_id"],
        "assigned_at": utc_now(),
        "registry_version": DEFAULT_REGISTRY_VERSION,
        "allocation_engine_version": DEFAULT_ENGINE_VERSION,
        "assignment_method": "automatic",
        "assignment_status": status,
        "reason_codes": reason_codes,
        "private_score": private_score,
        "review_required": review_required,
        "pool_was_created": status == STATUS_NEW_POOL,
    }


def _result_from_assignment(asset: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    status = _sanitize_text(assignment.get("assignment_status")) or STATUS_UNRESOLVED
    return {
        "asset_id": asset.get("asset_id"),
        "assignment_id": assignment.get("assignment_id"),
        "assignment_status": status,
        "pool_id": assignment.get("pool_id"),
        "reason_codes": assignment.get("reason_codes", []),
        "review_required": bool(assignment.get("review_required")),
        "pool_was_created": bool(assignment.get("pool_was_created") or status == STATUS_NEW_POOL),
    }


def _manifest_updates_from_asset(asset: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "isrc": asset.get("isrc") or None,
        "iswc": asset.get("iswc") or None,
        "record_label": asset.get("record_label") or None,
        "rights_holder": asset.get("rights_holder") or None,
        "territory": asset.get("territory") or None,
        "licence_class": asset.get("licence_class") or None,
        "artist_ids": asset.get("artist_ids") or [],
        "featured_artists": asset.get("featured_artists") or [],
        "primary_genre": asset.get("primary_genre") or None,
        "secondary_genre": asset.get("secondary_genre") or None,
        "style_tags": asset.get("style_tags") or [],
        "metadata_style_summary": asset.get("metadata_style_summary") or None,
        "language": asset.get("language") or None,
        "cara_source_asset_id": asset.get("asset_id"),
        "cara_source_pool_id": result.get("pool_id"),
        "cara_source_pool_assignment_status": result.get("assignment_status"),
        "cara_source_pool_reason_codes": result.get("reason_codes", []),
        "cara_source_pool_review_required": bool(result.get("review_required")),
        "cara_source_pool_assignment_id": result.get("assignment_id"),
        "cara_source_pool_last_assigned_utc": utc_now(),
    }


def _top_artist_share(pool: dict[str, Any], config: AllocatorConfig) -> float:
    artist_map = dict(pool.get("artist_duration_seconds", {}))
    if not artist_map:
        return 0.0
    return max(float(value) for value in artist_map.values()) / max(float(config.max_pool_duration_seconds), 1.0)


def summarize_registry(paths: AllocatorPaths, config: AllocatorConfig | None = None) -> dict[str, Any]:
    config = config or load_allocator_config()
    registry = _load_registry(paths)
    assignments = registry["assignments"]
    counts = Counter(str(assignment.get("assignment_status") or STATUS_UNRESOLVED) for assignment in assignments)
    for status in [STATUS_ASSIGNED, STATUS_NEW_POOL, STATUS_DUPLICATE, STATUS_REVIEW, STATUS_REJECTED, STATUS_UNRESOLVED]:
        counts.setdefault(status, 0)
    latest_run = registry["runs"][-1] if registry["runs"] else None
    return {
        "counts": dict(counts),
        "pool_count": len(registry["pools"]),
        "asset_count": len(registry["assets"]),
        "assignment_count": len(assignments),
        "duplicate_count": len(registry["duplicates"]),
        "review_count": sum(1 for item in assignments if item.get("review_required")),
        "latest_run": latest_run,
        "manifest_paths": {
            "source_manifest_path": str(paths.manifest_path),
            "cara_pool_manifest_path": str(paths.cara_manifest_path),
            "cara_pool_manifest_csv_path": str(paths.cara_manifest_csv_path),
        },
        "rules": {
            "max_pool_duration_seconds": config.max_pool_duration_seconds,
            "max_artist_duration_seconds": config.max_artist_duration_seconds,
            "repair_threshold": config.repair_threshold,
            "min_pool_code_edit_distance": config.min_pool_code_edit_distance,
        },
    }


def list_pools(paths: AllocatorPaths, config: AllocatorConfig | None = None) -> list[dict[str, Any]]:
    config = config or load_allocator_config()
    registry = _load_registry(paths)
    rows: list[dict[str, Any]] = []
    for pool in registry["pools"]:
        current_duration = float(pool.get("current_duration_seconds", 0.0))
        rows.append(
            {
                "pool_id": pool.get("pool_id"),
                "licence_class": pool.get("licence_class"),
                "territory": pool.get("territory"),
                "record_label": pool.get("record_label"),
                "rights_holder_group": pool.get("rights_holder_group"),
                "primary_genre": pool.get("primary_genre"),
                "asset_count": int(pool.get("asset_count", 0)),
                "current_duration_seconds": current_duration,
                "remaining_capacity_seconds": max(config.max_pool_duration_seconds - current_duration, 0.0),
                "top_artist_share": _top_artist_share(pool, config),
                "style_summary": pool.get("style_profile", {}).get("style_summary"),
                "updated_at": pool.get("updated_at"),
            }
        )
    rows.sort(key=lambda row: row["updated_at"] or "", reverse=True)
    return rows


def list_assignments(paths: AllocatorPaths, limit: int = 200) -> list[dict[str, Any]]:
    registry = _load_registry(paths)
    rows: list[dict[str, Any]] = []
    for assignment in reversed(registry["assignments"][-limit:]):
        rows.append(
            {
                "assignment_id": assignment.get("assignment_id"),
                "asset_id": assignment.get("asset_id"),
                "pool_id": assignment.get("pool_id"),
                "assignment_status": assignment.get("assignment_status"),
                "reason_codes": assignment.get("reason_codes", []),
                "review_required": bool(assignment.get("review_required")),
                "assigned_at": assignment.get("assigned_at"),
                "pool_was_created": bool(assignment.get("pool_was_created")),
            }
        )
    return rows


def list_review_queue(paths: AllocatorPaths, limit: int = 200) -> list[dict[str, Any]]:
    registry = _load_registry(paths)
    reviews = [
        {
            "assignment_id": assignment.get("assignment_id"),
            "asset_id": assignment.get("asset_id"),
            "pool_id": assignment.get("pool_id"),
            "assignment_status": assignment.get("assignment_status"),
            "reason_codes": assignment.get("reason_codes", []),
            "assigned_at": assignment.get("assigned_at"),
        }
        for assignment in reversed(registry["assignments"])
        if assignment.get("review_required")
    ]
    return reviews[:limit]


def _empty_run_summary(run_id: str, options: RunOptions) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": utc_now(),
        "finished_at": None,
        "options": asdict(options),
        "processed_assets": 0,
        "total_assets": 0,
        "counts": _empty_counts(),
        "status": "running",
    }


def _registry_requires_rebuild(registry: dict[str, list[dict[str, Any]]]) -> bool:
    versions = {
        _sanitize_text(assignment.get("allocation_engine_version"))
        for assignment in registry["assignments"]
        if assignment.get("allocation_engine_version")
    }
    if any(version and version != DEFAULT_ENGINE_VERSION for version in versions):
        return True
    assignment_ids = [
        _sanitize_text(assignment.get("assignment_id"))
        for assignment in registry.get("assignments", [])
        if assignment.get("assignment_id")
    ]
    if assignment_ids and len(set(assignment_ids)) != len(assignment_ids):
        return True
    pools = registry.get("pools", [])
    if pools and all(float(pool.get("current_duration_seconds", 0.0)) <= 0.0 for pool in pools):
        assets = registry.get("assets", [])
        if assets and all(float(asset.get("duration_seconds", 0.0)) <= 0.0 for asset in assets):
            return True
    return False


def _reset_allocator_outputs(
    paths: AllocatorPaths,
    manifest_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    for row in manifest_rows:
        _clear_pool_manifest_fields(row)
    registry = _empty_registry()
    _persist_registry(paths, registry)
    _write_progress_state(paths, _default_progress_state())
    _save_manifests(manifest_rows, paths)
    return registry, manifest_rows


def reset_allocator_state(paths: AllocatorPaths) -> dict[str, Any]:
    existing = _load_registry(paths)
    manifest_rows = load_manifest_rows(paths.manifest_path)
    _reset_allocator_outputs(paths, manifest_rows)
    return {
        "status": "reset",
        "cleared_assets": len(existing["assets"]),
        "cleared_pools": len(existing["pools"]),
        "cleared_assignments": len(existing["assignments"]),
        "manifest_rows": len(manifest_rows),
    }


def run_pool_allocation(
    paths: AllocatorPaths,
    options: RunOptions | None = None,
    config: AllocatorConfig | None = None,
    *,
    progress_callback: Any = None,
    stop_requested: Any = None,
) -> dict[str, Any]:
    options = options or RunOptions()
    config = config or load_allocator_config()
    registry = _load_registry(paths)
    manifest_rows = load_manifest_rows(paths.manifest_path)
    if options.start_fresh or _registry_requires_rebuild(registry):
        registry, manifest_rows = _reset_allocator_outputs(paths, manifest_rows)
    run_id = _next_prefixed_id([{"run_id": row.get("run_id")} for row in registry["runs"]], "run")
    run_summary = _empty_run_summary(run_id, options)
    registry["runs"].append(run_summary)
    candidate_rows_all = _candidate_manifest_rows(manifest_rows, options)

    existing_assets_by_source = {str(asset.get("source_key")): asset for asset in registry["assets"]}
    existing_asset_index_by_source = {
        str(asset.get("source_key")): idx for idx, asset in enumerate(registry["assets"]) if asset.get("source_key")
    }
    existing_assignments_by_asset = {
        str(assignment.get("asset_id")): assignment
        for assignment in registry["assignments"]
        if assignment.get("asset_id")
    }

    candidate_rows: list[tuple[int, dict[str, Any]]] = []
    baseline_processed = 0
    baseline_counts = _empty_counts()
    for index, row in candidate_rows_all:
        source_id = _sanitize_text(row.get("source_id") or row.get("raw_id"))
        source = _sanitize_text(row.get("source")) or "unknown"
        source_key = f"{source}:{source_id}"
        existing_asset = existing_assets_by_source.get(source_key)
        existing_assignment = (
            existing_assignments_by_asset.get(str(existing_asset.get("asset_id"))) if existing_asset else None
        )
        if _is_current_assignment(existing_assignment):
            row.update(_manifest_updates_from_asset(existing_asset, _result_from_assignment(existing_asset, existing_assignment)))
            baseline_processed += 1
            status = _sanitize_text(existing_assignment.get("assignment_status")) or STATUS_UNRESOLVED
            baseline_counts[status] = int(baseline_counts.get(status, 0)) + 1
            continue
        candidate_rows.append((index, row))

    run_summary["total_assets"] = len(candidate_rows_all)
    run_summary["processed_assets"] = baseline_processed
    run_summary["counts"] = dict(baseline_counts)

    progress_state = _default_progress_state(options)
    progress_state.update(
        {
            "status": "running",
            "run_id": run_id,
            "started_at": run_summary["started_at"],
            "updated_at": run_summary["started_at"],
            "processed_assets": baseline_processed,
            "total_assets": len(candidate_rows_all),
            "percent_complete": round((baseline_processed / len(candidate_rows_all)) * 100, 2) if candidate_rows_all else 100.0,
            "counts": dict(run_summary["counts"]),
        }
    )
    _append_progress_activity(
        progress_state,
        (
            f"Pool allocation run started for {len(candidate_rows_all)} candidate assets"
            if baseline_processed == 0
            else f"Pool allocation run resumed with {baseline_processed} already processed and {len(candidate_rows)} pending"
        ),
        phase="job",
    )
    _write_progress_state(paths, progress_state)
    if callable(progress_callback):
        progress_callback(dict(progress_state))

    rng = random.Random(run_id)
    processed = baseline_processed

    def publish_progress(*, phase: str, message: str, level: str = "info", asset: dict[str, Any] | None = None, pool_id: str | None = None, checkpoint_manifests: bool = False) -> None:
        progress_state["updated_at"] = utc_now()
        progress_state["current_phase"] = phase
        progress_state["current_asset"] = asset.get("asset_id") if asset else None
        progress_state["current_asset_title"] = asset.get("title") if asset else None
        progress_state["current_pool_id"] = pool_id
        progress_state["processed_assets"] = processed
        progress_state["counts"] = dict(run_summary["counts"])
        progress_state["percent_complete"] = round((processed / len(candidate_rows_all)) * 100, 2) if candidate_rows_all else 100.0
        _append_progress_activity(
            progress_state,
            message,
            phase=phase,
            level=level,
            asset_id=asset.get("asset_id") if asset else None,
            source_key=asset.get("source_key") if asset else None,
            pool_id=pool_id,
        )
        _persist_checkpoint(
            paths=paths,
            registry=registry,
            run_summary=run_summary,
            progress_state=progress_state,
            manifest_rows=manifest_rows,
            write_manifests=checkpoint_manifests,
        )
        if callable(progress_callback):
            progress_callback(dict(progress_state))

    if not candidate_rows:
        run_summary["finished_at"] = utc_now()
        run_summary["status"] = "completed"
        progress_state["status"] = "completed"
        progress_state["finished_at"] = run_summary["finished_at"]
        publish_progress(
            phase="completed",
            message=f"Completed pool allocation run: {processed} / {len(candidate_rows_all)} assets already allocated",
            checkpoint_manifests=True,
        )
        return run_summary

    for index, row in candidate_rows:
        if callable(stop_requested) and stop_requested():
            run_summary["processed_assets"] = processed
            run_summary["finished_at"] = utc_now()
            run_summary["status"] = "paused"
            progress_state["status"] = "paused"
            progress_state["finished_at"] = run_summary["finished_at"]
            publish_progress(
                phase="paused",
                message=f"Pause requested after {processed} / {len(candidate_rows_all)} assets",
                level="warn",
                checkpoint_manifests=True,
            )
            return run_summary

        source_id = _sanitize_text(row.get("source_id") or row.get("raw_id"))
        source = _sanitize_text(row.get("source")) or "unknown"
        source_key = f"{source}:{source_id}"

        asset = normalize_manifest_row(row, index)
        existing_asset = existing_assets_by_source.get(source_key)
        existing_asset_index = existing_asset_index_by_source.get(source_key)
        progress_state["current_asset_title"] = asset.get("title")
        progress_state["current_phase"] = "processing"
        progress_state["current_asset"] = asset.get("asset_id")
        progress_state["updated_at"] = utc_now()
        if callable(progress_callback):
            progress_callback(dict(progress_state))
        if existing_asset:
            assignment = existing_assignments_by_asset.get(str(existing_asset.get("asset_id")))
            if assignment and assignment.get("assignment_status") in {
                STATUS_ASSIGNED,
                STATUS_NEW_POOL,
                STATUS_DUPLICATE,
            } and _sanitize_text(assignment.get("allocation_engine_version")) == DEFAULT_ENGINE_VERSION:
                result = {
                    **_result_from_assignment(existing_asset, assignment),
                }
                update = _manifest_updates_from_asset(existing_asset, result)
                row.update(update)
                processed += 1
                run_summary["processed_assets"] = processed
                publish_progress(
                    phase="reuse",
                    message=f"Reused existing assignment for {source_key}",
                    asset=existing_asset,
                    pool_id=result.get("pool_id"),
                    checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
                )
                continue
            asset = {**existing_asset, **asset}
            asset["asset_id"] = existing_asset.get("asset_id")
        else:
            asset["asset_id"] = _next_prefixed_id([{"asset_id": item.get("asset_id")} for item in registry["assets"]], "asset")

        duplicate_reason, duplicate_asset = _find_duplicate(asset, registry, config)

        if duplicate_reason == "POTENTIAL_DUPLICATE_FUZZY_MATCH":
            duplicate_record = {
                "duplicate_id": _next_prefixed_id([{"duplicate_id": item.get("duplicate_id")} for item in registry["duplicates"]], "duplicate"),
                "asset_id": asset["asset_id"],
                "duplicate_of_asset_id": duplicate_asset.get("asset_id") if duplicate_asset else None,
                "reason_code": duplicate_reason,
                "detected_at": utc_now(),
                "review_required": True,
            }
            registry["duplicates"].append(duplicate_record)
            if existing_asset_index is None:
                registry["assets"].append(asset)
                existing_asset_index = len(registry["assets"]) - 1
                existing_asset_index_by_source[source_key] = existing_asset_index
            else:
                registry["assets"][existing_asset_index] = asset
            assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
            assignment = {
                "assignment_id": assignment_id,
                "asset_id": asset["asset_id"],
                "pool_id": None,
                "assigned_at": utc_now(),
                "registry_version": DEFAULT_REGISTRY_VERSION,
                "allocation_engine_version": DEFAULT_ENGINE_VERSION,
                "assignment_method": "automatic",
                "assignment_status": STATUS_REVIEW,
                "reason_codes": ["POTENTIAL_DUPLICATE_MATCH", "REVIEW_REQUIRED"],
                "private_score": None,
                "review_required": True,
                "pool_was_created": False,
            }
            registry["assignments"].append(assignment)
            result = {
                "asset_id": asset["asset_id"],
                "assignment_id": assignment_id,
                "assignment_status": STATUS_REVIEW,
                "pool_id": None,
                "reason_codes": assignment["reason_codes"],
                "review_required": True,
                "pool_was_created": False,
            }
            row.update(_manifest_updates_from_asset(asset, result))
            run_summary["counts"][STATUS_REVIEW] += 1
            existing_assets_by_source[source_key] = asset
            existing_assignments_by_asset[asset["asset_id"]] = assignment
            processed += 1
            run_summary["processed_assets"] = processed
            publish_progress(
                phase="duplicate-review",
                message=f"Flagged fuzzy duplicate for review: {source_key}",
                level="warn",
                asset=asset,
                checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
            )
            continue

        if duplicate_reason and duplicate_asset:
            existing_assignment = _existing_assignment_for_asset(registry["assignments"], str(duplicate_asset.get("asset_id")))
            duplicate_record = {
                "duplicate_id": _next_prefixed_id([{"duplicate_id": item.get("duplicate_id")} for item in registry["duplicates"]], "duplicate"),
                "asset_id": asset["asset_id"],
                "duplicate_of_asset_id": duplicate_asset.get("asset_id"),
                "reason_code": duplicate_reason,
                "detected_at": utc_now(),
                "review_required": False,
            }
            registry["duplicates"].append(duplicate_record)
            if existing_asset_index is None:
                registry["assets"].append(asset)
                existing_asset_index = len(registry["assets"]) - 1
                existing_asset_index_by_source[source_key] = existing_asset_index
            else:
                registry["assets"][existing_asset_index] = asset
            assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
            assignment = {
                "assignment_id": assignment_id,
                "asset_id": asset["asset_id"],
                "pool_id": existing_assignment.get("pool_id") if existing_assignment else None,
                "assigned_at": utc_now(),
                "registry_version": DEFAULT_REGISTRY_VERSION,
                "allocation_engine_version": DEFAULT_ENGINE_VERSION,
                "assignment_method": "automatic",
                "assignment_status": STATUS_DUPLICATE,
                "reason_codes": [duplicate_reason],
                "private_score": None,
                "review_required": False,
                "pool_was_created": False,
            }
            registry["assignments"].append(assignment)
            result = {
                "asset_id": asset["asset_id"],
                "assignment_id": assignment_id,
                "assignment_status": STATUS_DUPLICATE,
                "pool_id": assignment["pool_id"],
                "reason_codes": assignment["reason_codes"],
                "review_required": False,
                "pool_was_created": False,
            }
            row.update(_manifest_updates_from_asset(asset, result))
            run_summary["counts"][STATUS_DUPLICATE] += 1
            existing_assets_by_source[source_key] = asset
            existing_assignments_by_asset[asset["asset_id"]] = assignment
            processed += 1
            run_summary["processed_assets"] = processed
            publish_progress(
                phase="duplicate",
                message=f"Resolved exact duplicate for {source_key}",
                asset=asset,
                pool_id=assignment.get("pool_id"),
                checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
            )
            continue

        if not asset.get("licence_class"):
            if existing_asset_index is None:
                registry["assets"].append(asset)
                existing_asset_index = len(registry["assets"]) - 1
                existing_asset_index_by_source[source_key] = existing_asset_index
            else:
                registry["assets"][existing_asset_index] = asset
            assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
            assignment = {
                "assignment_id": assignment_id,
                "asset_id": asset["asset_id"],
                "pool_id": None,
                "assigned_at": utc_now(),
                "registry_version": DEFAULT_REGISTRY_VERSION,
                "allocation_engine_version": DEFAULT_ENGINE_VERSION,
                "assignment_method": "automatic",
                "assignment_status": STATUS_REVIEW,
                "reason_codes": ["LICENCE_MISSING", "REVIEW_REQUIRED"],
                "private_score": None,
                "review_required": True,
                "pool_was_created": False,
            }
            registry["assignments"].append(assignment)
            result = {
                "asset_id": asset["asset_id"],
                "assignment_id": assignment_id,
                "assignment_status": STATUS_REVIEW,
                "pool_id": None,
                "reason_codes": assignment["reason_codes"],
                "review_required": True,
                "pool_was_created": False,
            }
            row.update(_manifest_updates_from_asset(asset, result))
            run_summary["counts"][STATUS_REVIEW] += 1
            existing_assets_by_source[source_key] = asset
            existing_assignments_by_asset[asset["asset_id"]] = assignment
            processed += 1
            run_summary["processed_assets"] = processed
            publish_progress(
                phase="review",
                message=f"Licence metadata missing for {source_key}",
                level="warn",
                asset=asset,
                checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
            )
            continue

        if not asset.get("record_label") and not asset.get("rights_holder") and not options.allow_relaxed_metadata:
            if existing_asset_index is None:
                registry["assets"].append(asset)
                existing_asset_index = len(registry["assets"]) - 1
                existing_asset_index_by_source[source_key] = existing_asset_index
            else:
                registry["assets"][existing_asset_index] = asset
            assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
            assignment = {
                "assignment_id": assignment_id,
                "asset_id": asset["asset_id"],
                "pool_id": None,
                "assigned_at": utc_now(),
                "registry_version": DEFAULT_REGISTRY_VERSION,
                "allocation_engine_version": DEFAULT_ENGINE_VERSION,
                "assignment_method": "automatic",
                "assignment_status": STATUS_REVIEW,
                "reason_codes": ["RIGHTSHOLDER_MISSING", "REVIEW_REQUIRED"],
                "private_score": None,
                "review_required": True,
                "pool_was_created": False,
            }
            registry["assignments"].append(assignment)
            result = {
                "asset_id": asset["asset_id"],
                "assignment_id": assignment_id,
                "assignment_status": STATUS_REVIEW,
                "pool_id": None,
                "reason_codes": assignment["reason_codes"],
                "review_required": True,
                "pool_was_created": False,
            }
            row.update(_manifest_updates_from_asset(asset, result))
            run_summary["counts"][STATUS_REVIEW] += 1
            existing_assets_by_source[source_key] = asset
            existing_assignments_by_asset[asset["asset_id"]] = assignment
            processed += 1
            run_summary["processed_assets"] = processed
            publish_progress(
                phase="review",
                message=f"Rights metadata missing for {source_key}",
                level="warn",
                asset=asset,
                checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
            )
            continue

        if not asset.get("primary_genre"):
            if existing_asset_index is None:
                registry["assets"].append(asset)
                existing_asset_index = len(registry["assets"]) - 1
                existing_asset_index_by_source[source_key] = existing_asset_index
            else:
                registry["assets"][existing_asset_index] = asset
            assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
            assignment = {
                "assignment_id": assignment_id,
                "asset_id": asset["asset_id"],
                "pool_id": None,
                "assigned_at": utc_now(),
                "registry_version": DEFAULT_REGISTRY_VERSION,
                "allocation_engine_version": DEFAULT_ENGINE_VERSION,
                "assignment_method": "automatic",
                "assignment_status": STATUS_REVIEW,
                "reason_codes": ["PRIMARY_GENRE_MISSING", "REVIEW_REQUIRED"],
                "private_score": None,
                "review_required": True,
                "pool_was_created": False,
            }
            registry["assignments"].append(assignment)
            result = {
                "asset_id": asset["asset_id"],
                "assignment_id": assignment_id,
                "assignment_status": STATUS_REVIEW,
                "pool_id": None,
                "reason_codes": assignment["reason_codes"],
                "review_required": True,
                "pool_was_created": False,
            }
            row.update(_manifest_updates_from_asset(asset, result))
            run_summary["counts"][STATUS_REVIEW] += 1
            existing_assets_by_source[source_key] = asset
            existing_assignments_by_asset[asset["asset_id"]] = assignment
            processed += 1
            run_summary["processed_assets"] = processed
            publish_progress(
                phase="review",
                message=f"Primary genre missing for {source_key}",
                level="warn",
                asset=asset,
                checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
            )
            continue

        candidate_pools = [pool for pool in registry["pools"] if pool.get("modality") == "audio"]
        valid_candidates: list[dict[str, Any]] = []
        for pool in candidate_pools:
            if not _license_matches(asset, pool):
                continue
            if not _territory_matches(asset, pool):
                continue
            if not _rights_or_label_matches(asset, pool, allow_relaxed_metadata=options.allow_relaxed_metadata):
                continue
            if not _has_duration_capacity(asset, pool, config):
                continue
            if not _artist_cap_ok(asset, pool, config):
                continue
            if not _primary_genre_matches(asset, pool):
                continue
            style_score = calculate_style_score(asset, pool)
            if style_score < config.min_style_score:
                continue
            total_score = calculate_candidate_score(
                asset,
                pool,
                style_score,
                config,
                allow_relaxed_metadata=options.allow_relaxed_metadata,
            )
            valid_candidates.append({"pool": pool, "score": total_score})

        selected_pool: dict[str, Any] | None = None
        assignment_status = STATUS_ASSIGNED
        private_score: float | None = None
        reason_codes = [
            "LICENCE_MATCH",
            "TERRITORY_MATCH",
            "RIGHTSHOLDER_OR_LABEL_MATCH",
            "PRIMARY_GENRE_MATCH",
            "STYLE_COMPATIBLE",
            "POOL_CAPACITY_AVAILABLE",
            "ARTIST_CAP_OK",
        ]

        if valid_candidates:
            best = max(valid_candidates, key=lambda item: item["score"])
            if float(best["score"]) >= config.min_assignment_score:
                selected_pool = best["pool"]
                private_score = float(best["score"])

        if selected_pool is None:
            selected_pool = _create_pool_from_asset(asset, registry, config, rng)
            registry["pools"].append(selected_pool)
            assignment_status = STATUS_NEW_POOL
            private_score = None
            reason_codes = ["NO_COMPATIBLE_POOL", "NEW_POOL_CREATED"]

        _update_pool_with_asset(selected_pool, asset)
        if existing_asset_index is None:
            registry["assets"].append(asset)
            existing_asset_index = len(registry["assets"]) - 1
            existing_asset_index_by_source[source_key] = existing_asset_index
        else:
            registry["assets"][existing_asset_index] = asset
        assignment_id = _next_prefixed_id([{"assignment_id": item.get("assignment_id")} for item in registry["assignments"]], "assign")
        assignment = _assignment_row(
            asset=asset,
            pool=selected_pool,
            assignment_id=assignment_id,
            status=assignment_status,
            reason_codes=reason_codes,
            private_score=private_score,
            review_required=False,
        )
        registry["assignments"].append(assignment)
        result = {
            "asset_id": asset["asset_id"],
            "assignment_id": assignment_id,
            "assignment_status": assignment_status,
            "pool_id": selected_pool["pool_id"],
            "reason_codes": reason_codes,
            "review_required": False,
            "pool_was_created": assignment_status == STATUS_NEW_POOL,
        }
        row.update(_manifest_updates_from_asset(asset, result))
        run_summary["counts"][assignment_status] += 1
        existing_assets_by_source[source_key] = asset
        existing_assignments_by_asset[asset["asset_id"]] = assignment
        processed += 1
        run_summary["processed_assets"] = processed
        publish_progress(
            phase="assigned" if assignment_status == STATUS_ASSIGNED else "new-pool",
            message=(
                f"Assigned {source_key} to {selected_pool['pool_id']}"
                if assignment_status == STATUS_ASSIGNED
                else f"Created {selected_pool['pool_id']} and assigned {source_key}"
            ),
            asset=asset,
            pool_id=selected_pool["pool_id"],
            checkpoint_manifests=processed % CHECKPOINT_MANIFEST_EVERY == 0,
        )

    run_summary["processed_assets"] = processed
    run_summary["finished_at"] = utc_now()
    run_summary["status"] = "completed"
    progress_state["status"] = "completed"
    progress_state["finished_at"] = run_summary["finished_at"]
    publish_progress(
        phase="completed",
        message=f"Completed pool allocation run: {processed} / {len(candidate_rows_all)} assets processed",
        checkpoint_manifests=True,
    )
    return run_summary
