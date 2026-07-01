from __future__ import annotations

import json
import math
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_project_config
from data_pipeline.manifest_utils import export_manifest_csv, save_manifest_rows
from data_pipeline.pool_allocator import (
    POOL_ASSIGNED_STATUSES,
    STATUS_ASSIGNED,
    STATUS_DUPLICATE,
    STATUS_NEW_POOL,
    STATUS_REJECTED,
    STATUS_REVIEW,
    STATUS_UNRESOLVED,
    _duplicate_match_reason,
    _generate_pool_code,
    _is_fuzzy_duplicate,
    _next_prefixed_id,
    _normalize_phrase,
    _pool_checksum,
    _sanitize_text,
    _write_json,
    _write_jsonl,
    edit_distance,
    normalize_manifest_row,
)

POOL_V2_REGISTRY_VERSION = "2026.05.21"
POOL_V2_ENGINE_VERSION = "0.3.0-v2"
POOL_V2_SCHEMA_VERSION = "2"

POOL_V2_MANIFEST_FIELDS = (
    "cara_v2_source_asset_id",
    "cara_v2_source_pool_id",
    "cara_v2_source_pool_assignment_status",
    "cara_v2_source_pool_reason_codes",
    "cara_v2_source_pool_review_required",
    "cara_v2_source_pool_assignment_id",
    "cara_v2_source_pool_last_assigned_utc",
)

POOL_FAMILY_BY_GENRE = {
    "Ambient/Drone": "Atmosphere/Field",
    "Field Recording": "Atmosphere/Field",
    "Percussion/Drums": "Percussion/Beats",
    "Percussion Drums": "Percussion/Beats",
    "Hip-Hop/Beats": "Percussion/Beats",
    "Acoustic/Folk": "Acoustic/Jazz/World",
    "Jazz/Blues": "Acoustic/Jazz/World",
    "Jazz Blues": "Acoustic/Jazz/World",
    "World/Traditional": "Acoustic/Jazz/World",
    "Classical/Orchestral": "Tonal/Orchestral",
    "Electronic": "Produced/Electronic",
    "Rock/Metal": "Produced/Electronic",
    "Rock Metal": "Produced/Electronic",
    "Experimental/Noise": "Experimental/Noise",
    "Voice/Vocal": "Voice/Vocal",
    "Voice Vocal": "Voice/Vocal",
    "Sound Effects": "Sound Effects",
}


@dataclass
class AllocatorV2Config:
    max_pool_duration_seconds: int = 14_400
    general_artist_cap_seconds: int = 1_440
    target_pool_count: int = 60
    min_pool_code_edit_distance: int = 5
    repair_threshold: int = 2
    fuzzy_duration_tolerance_seconds: float = 2.0
    artist_exception_min_duration_seconds: int = 7_200
    artist_exception_min_pool_pressure: int = 3
    include_progress_completed_ids: bool = True
    family_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class AllocatorV2Paths:
    root: Path
    manifest_path: Path
    cara_manifest_path: Path
    cara_manifest_csv_path: Path
    download_progress_path: Path
    registry_dir: Path
    assets_path: Path
    pools_path: Path
    assignments_path: Path
    duplicates_path: Path
    runs_path: Path
    progress_path: Path
    plan_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "AllocatorV2Paths":
        registry_dir = root / "registry" / "pool_allocator_v2"
        manifest_path = root / "data" / "attribution_manifest.jsonl"
        cara_manifest_path = root / "data" / "cara_pool_manifest_v2.jsonl"
        return cls(
            root=root,
            manifest_path=manifest_path,
            cara_manifest_path=cara_manifest_path,
            cara_manifest_csv_path=cara_manifest_path.with_suffix(".csv"),
            download_progress_path=root / "data" / "download_progress.json",
            registry_dir=registry_dir,
            assets_path=registry_dir / "assets.jsonl",
            pools_path=registry_dir / "pools.json",
            assignments_path=registry_dir / "assignments.jsonl",
            duplicates_path=registry_dir / "duplicates.jsonl",
            runs_path=registry_dir / "runs.jsonl",
            progress_path=registry_dir / "progress.json",
            plan_path=registry_dir / "plan.json",
        )


@dataclass
class RunOptionsV2:
    subset_role: str | None = None
    only_downloaded: bool = True
    limit: int | None = None
    allow_relaxed_metadata: bool = True
    start_fresh: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_allocator_v2_config() -> AllocatorV2Config:
    raw = load_project_config().get("pool_allocator_v2", {})
    if not isinstance(raw, dict):
        return AllocatorV2Config()
    default = AllocatorV2Config()
    return AllocatorV2Config(
        max_pool_duration_seconds=int(raw.get("max_pool_duration_seconds", default.max_pool_duration_seconds)),
        general_artist_cap_seconds=int(raw.get("general_artist_cap_seconds", default.general_artist_cap_seconds)),
        target_pool_count=int(raw.get("target_pool_count", default.target_pool_count)),
        min_pool_code_edit_distance=int(raw.get("min_pool_code_edit_distance", default.min_pool_code_edit_distance)),
        repair_threshold=int(raw.get("repair_threshold", default.repair_threshold)),
        fuzzy_duration_tolerance_seconds=float(raw.get("fuzzy_duration_tolerance_seconds", default.fuzzy_duration_tolerance_seconds)),
        artist_exception_min_duration_seconds=int(raw.get("artist_exception_min_duration_seconds", default.artist_exception_min_duration_seconds)),
        artist_exception_min_pool_pressure=int(raw.get("artist_exception_min_pool_pressure", default.artist_exception_min_pool_pressure)),
        include_progress_completed_ids=bool(raw.get("include_progress_completed_ids", default.include_progress_completed_ids)),
        family_aliases=dict(raw.get("family_aliases", {})) if isinstance(raw.get("family_aliases"), dict) else {},
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
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _load_manifest_rows_lenient(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, errors
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "preview": line[:240],
                    }
                )
    return rows, errors


def _empty_counts() -> dict[str, int]:
    return {
        STATUS_ASSIGNED: 0,
        STATUS_NEW_POOL: 0,
        STATUS_DUPLICATE: 0,
        STATUS_REVIEW: 0,
        STATUS_REJECTED: 0,
        STATUS_UNRESOLVED: 0,
    }


def _default_progress_state(options: RunOptionsV2 | None = None) -> dict[str, Any]:
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


def read_progress_state_v2(paths: AllocatorV2Paths) -> dict[str, Any]:
    return _load_json(paths.progress_path, _default_progress_state())


def _write_progress_state(paths: AllocatorV2Paths, progress: dict[str, Any]) -> None:
    _write_json(paths.progress_path, progress)


def _load_registry(paths: AllocatorV2Paths) -> dict[str, list[dict[str, Any]]]:
    return {
        "assets": _load_jsonl(paths.assets_path),
        "pools": _load_json(paths.pools_path, []),
        "assignments": _load_jsonl(paths.assignments_path),
        "duplicates": _load_jsonl(paths.duplicates_path),
        "runs": _load_jsonl(paths.runs_path),
    }


def _persist_registry(paths: AllocatorV2Paths, registry: dict[str, list[dict[str, Any]]]) -> None:
    _write_jsonl(paths.assets_path, registry["assets"])
    _write_json(paths.pools_path, registry["pools"])
    _write_jsonl(paths.assignments_path, registry["assignments"])
    _write_jsonl(paths.duplicates_path, registry["duplicates"])
    _write_jsonl(paths.runs_path, registry["runs"])


def _downloaded_source_ids(paths: AllocatorV2Paths) -> set[str]:
    if not paths.download_progress_path.exists():
        return set()
    try:
        payload = json.loads(paths.download_progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in payload.get("completed_ids", []) or []}


def _candidate_manifest_rows_v2(
    manifest_rows: list[dict[str, Any]],
    options: RunOptionsV2,
    paths: AllocatorV2Paths,
    config: AllocatorV2Config,
) -> list[tuple[int, dict[str, Any]]]:
    completed_ids = _downloaded_source_ids(paths) if config.include_progress_completed_ids else set()
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(manifest_rows):
        if options.subset_role and _sanitize_text(row.get("subset_role")) != options.subset_role:
            continue
        source_id = _sanitize_text(row.get("source_id") or row.get("raw_id"))
        if options.only_downloaded:
            is_downloaded = _sanitize_text(row.get("download_status")) == "downloaded"
            if completed_ids:
                is_downloaded = is_downloaded or source_id in completed_ids
            if not is_downloaded:
                continue
        rows.append((index, row))
        if options.limit is not None and len(rows) >= options.limit:
            break
    return rows


def _duration_from_default_sidecar(asset: dict[str, Any], paths: AllocatorV2Paths) -> float:
    duration = float(asset.get("duration_seconds") or 0.0)
    if duration > 0:
        return duration
    source_id = _sanitize_text(asset.get("source_id"))
    if not source_id:
        return 0.0
    meta_path = paths.root / "data" / "freesound_meta" / f"{source_id}.json"
    if not meta_path.exists():
        return 0.0
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    for key in ("duration", "duration_seconds"):
        try:
            value = float(payload.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


def _default_audio_path(asset: dict[str, Any], paths: AllocatorV2Paths) -> str:
    existing = _sanitize_text(asset.get("source_file_path"))
    if existing:
        return existing
    source_id = _sanitize_text(asset.get("source_id"))
    if not source_id:
        return ""
    audio_dir = paths.root / "data" / "freesound"
    for path in audio_dir.glob(f"{source_id}.*"):
        if path.is_file():
            return str(path)
    return ""


def _normalize_asset_v2(row: dict[str, Any], index: int, paths: AllocatorV2Paths) -> dict[str, Any]:
    asset = normalize_manifest_row(row, index)
    asset["duration_seconds"] = _duration_from_default_sidecar(asset, paths)
    asset["source_file_path"] = _default_audio_path(asset, paths)
    asset["pool_family"] = _pool_family_for_asset(asset)
    asset["broad_style_tokens"] = _broad_style_tokens(asset)
    asset["broad_style_summary"] = _broad_style_summary(asset)
    return asset


def _pool_family_for_asset(asset: dict[str, Any]) -> str:
    genre = _sanitize_text(asset.get("primary_genre"))
    return POOL_FAMILY_BY_GENRE.get(genre, "Mixed/Unclassified")


def _broad_style_tokens(asset: dict[str, Any]) -> list[str]:
    family = _pool_family_for_asset(asset)
    tokens = [f"family:{family.lower()}"]
    genre = _sanitize_text(asset.get("primary_genre"))
    if genre:
        tokens.append(f"genre:{genre.lower()}")
    for tag in asset.get("style_tags") or []:
        text = _sanitize_text(tag).lower()
        if text in {"ableton", "ableton-live", "awesome", "live", "mono", "stereo", "sample", "sound"}:
            continue
        if text:
            tokens.append(f"style:{text}")
    language = _sanitize_text(asset.get("language"))
    if language:
        tokens.append(f"lang:{language.lower()}")
    return list(dict.fromkeys(tokens[:8]))


def _broad_style_summary(asset: dict[str, Any]) -> str:
    family = _pool_family_for_asset(asset)
    genre = _sanitize_text(asset.get("primary_genre"))
    tags = [_sanitize_text(tag) for tag in asset.get("style_tags") or [] if _sanitize_text(tag)]
    if genre and genre != family:
        return f"{family} | {genre}"
    if tags:
        return f"{family} | {', '.join(tags[:3])}"
    return family


def _artist_key(asset: dict[str, Any]) -> str:
    artist_ids = list(asset.get("artist_ids") or [])
    if artist_ids:
        return _sanitize_text(artist_ids[0])
    return _sanitize_text(asset.get("artist_primary")) or "__unknown__"


def _rights_key(asset: dict[str, Any], allow_relaxed_metadata: bool) -> str:
    label = _sanitize_text(asset.get("record_label"))
    rights = _sanitize_text(asset.get("rights_holder"))
    if label:
        return f"label:{label}"
    if rights:
        return f"rights:{rights}"
    return "__missing_rights_relaxed__" if allow_relaxed_metadata else ""


def _group_key(asset: dict[str, Any], allow_relaxed_metadata: bool) -> tuple[str, str, str, str]:
    return (
        _sanitize_text(asset.get("licence_class")),
        _sanitize_text(asset.get("territory")) or "GLOBAL",
        _rights_key(asset, allow_relaxed_metadata),
        _sanitize_text(asset.get("pool_family")) or "Mixed/Unclassified",
    )


def _pool_group_id(group_key: tuple[str, str, str, str], *, artist_key: str | None = None) -> str:
    parts = list(group_key)
    if artist_key:
        parts.append(f"artist:{artist_key}")
    return " | ".join(parts)


def _next_counter_id(counters: dict[str, int], prefix: str) -> str:
    value = counters.get(prefix, 1)
    counters[prefix] = value + 1
    return f"{prefix}_{value:06d}"


def _artist_exception_artists(
    assets_by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    config: AllocatorV2Config,
) -> dict[tuple[str, str, str, str], set[str]]:
    exceptions: dict[tuple[str, str, str, str], set[str]] = {}
    for group_key, assets in assets_by_group.items():
        durations: dict[str, float] = defaultdict(float)
        max_single_duration: dict[str, float] = defaultdict(float)
        for asset in assets:
            key = _artist_key(asset)
            duration = float(asset.get("duration_seconds") or 0.0)
            durations[key] += duration
            max_single_duration[key] = max(max_single_duration[key], duration)
        selected: set[str] = set()
        for artist_key, duration in durations.items():
            if artist_key == "__unknown__":
                continue
            pressure = math.ceil(duration / max(float(config.general_artist_cap_seconds), 1.0))
            if (
                duration >= config.artist_exception_min_duration_seconds
                and pressure >= config.artist_exception_min_pool_pressure
            ):
                selected.add(artist_key)
                continue
            if max_single_duration[artist_key] > config.general_artist_cap_seconds:
                selected.add(artist_key)
        if selected:
            exceptions[group_key] = selected
    return exceptions


def _create_pool(
    *,
    registry: dict[str, list[dict[str, Any]]],
    config: AllocatorV2Config,
    rng: random.Random,
    group_key: tuple[str, str, str, str],
    pool_type: str,
    spillover_index: int,
    artist_key: str | None = None,
) -> dict[str, Any]:
    existing_codes = [str(pool.get("pool_code") or "") for pool in registry["pools"] if pool.get("pool_code")]
    code = _generate_pool_code(existing_codes, config, rng)  # type: ignore[arg-type]
    checksum = _pool_checksum(code)
    pool_id = f"CARA:AUD:1:{code}:{checksum}"
    licence_class, territory, rights, family = group_key
    now = utc_now()
    return {
        "pool_id": pool_id,
        "modality": "audio",
        "schema_version": POOL_V2_SCHEMA_VERSION,
        "allocation_engine_version": POOL_V2_ENGINE_VERSION,
        "pool_code": code,
        "checksum": checksum,
        "pool_type": pool_type,
        "pool_family": family,
        "pool_group_id": _pool_group_id(group_key, artist_key=artist_key if pool_type == "artist_concentrated_pool" else None),
        "spillover_index": spillover_index,
        "pool_duration_cap_seconds": config.max_pool_duration_seconds,
        "general_artist_cap_seconds": None if pool_type == "artist_concentrated_pool" else config.general_artist_cap_seconds,
        "current_duration_seconds": 0.0,
        "asset_count": 0,
        "licence_class": licence_class,
        "territory": territory,
        "rights_holder_group": "" if rights.startswith("__") or rights.startswith("label:") else rights.removeprefix("rights:"),
        "record_label": rights.removeprefix("label:") if rights.startswith("label:") else "",
        "primary_genre": family,
        "included_primary_genres": [],
        "style_profile": {
            "style_summary": family,
            "style_tokens": [f"family:{family.lower()}"],
            "style_tags": [],
        },
        "artist_exception_key": artist_key if pool_type == "artist_concentrated_pool" else None,
        "artist_duration_seconds": {},
        "asset_ids": [],
        "created_at": now,
        "updated_at": now,
        "pool_status": "active",
        "repair_threshold": config.repair_threshold,
    }


def _pool_has_capacity(pool: dict[str, Any], asset: dict[str, Any], config: AllocatorV2Config) -> bool:
    return float(pool.get("current_duration_seconds") or 0.0) + float(asset.get("duration_seconds") or 0.0) <= config.max_pool_duration_seconds


def _general_artist_cap_ok(pool: dict[str, Any], asset: dict[str, Any], config: AllocatorV2Config) -> bool:
    if pool.get("pool_type") == "artist_concentrated_pool":
        return True
    artist_map = dict(pool.get("artist_duration_seconds", {}))
    key = _artist_key(asset)
    current = float(artist_map.get(key, 0.0))
    return current + float(asset.get("duration_seconds") or 0.0) <= config.general_artist_cap_seconds


def _update_pool(pool: dict[str, Any], asset: dict[str, Any]) -> None:
    duration = float(asset.get("duration_seconds") or 0.0)
    pool["current_duration_seconds"] = float(pool.get("current_duration_seconds") or 0.0) + duration
    pool["asset_count"] = int(pool.get("asset_count") or 0) + 1
    asset_ids = list(pool.get("asset_ids") or [])
    if asset["asset_id"] not in asset_ids:
        asset_ids.append(asset["asset_id"])
    pool["asset_ids"] = asset_ids
    artist_map = dict(pool.get("artist_duration_seconds") or {})
    artist_map[_artist_key(asset)] = float(artist_map.get(_artist_key(asset), 0.0)) + duration
    pool["artist_duration_seconds"] = artist_map
    genres = list(pool.get("included_primary_genres") or [])
    genre = _sanitize_text(asset.get("primary_genre"))
    if genre and genre not in genres:
        genres.append(genre)
    pool["included_primary_genres"] = sorted(genres)
    tags = list(pool.get("style_profile", {}).get("style_tags") or [])
    for tag in asset.get("style_tags") or []:
        text = _sanitize_text(tag)
        if text and text not in tags:
            tags.append(text)
    pool["style_profile"]["style_tags"] = tags[:10]
    if pool["included_primary_genres"]:
        pool["style_profile"]["style_summary"] = f"{pool.get('pool_family')} | {', '.join(pool['included_primary_genres'][:4])}"
    pool["updated_at"] = utc_now()


def _assignment_row(
    *,
    asset: dict[str, Any],
    pool: dict[str, Any] | None,
    assignment_id: str,
    status: str,
    reason_codes: list[str],
    review_required: bool = False,
) -> dict[str, Any]:
    return {
        "assignment_id": assignment_id,
        "asset_id": asset["asset_id"],
        "pool_id": pool.get("pool_id") if pool else None,
        "assigned_at": utc_now(),
        "registry_version": POOL_V2_REGISTRY_VERSION,
        "allocation_engine_version": POOL_V2_ENGINE_VERSION,
        "assignment_method": "automatic_v2_planned",
        "assignment_status": status,
        "reason_codes": reason_codes,
        "review_required": review_required,
        "pool_was_created": status == STATUS_NEW_POOL,
        "pool_type": pool.get("pool_type") if pool else None,
        "pool_family": pool.get("pool_family") if pool else asset.get("pool_family"),
    }


def _manifest_updates_from_assignment(asset: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    reason_codes = assignment.get("reason_codes", [])
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
        "metadata_style_summary": asset.get("metadata_style_summary") or asset.get("broad_style_summary") or None,
        "language": asset.get("language") or None,
        "cara_source_asset_id": asset.get("asset_id"),
        "cara_source_pool_id": assignment.get("pool_id"),
        "cara_source_pool_assignment_status": assignment.get("assignment_status"),
        "cara_source_pool_reason_codes": reason_codes,
        "cara_source_pool_review_required": bool(assignment.get("review_required")),
        "cara_source_pool_assignment_id": assignment.get("assignment_id"),
        "cara_source_pool_last_assigned_utc": assignment.get("assigned_at"),
        "cara_pool_allocator_version": "v2",
        "cara_pool_family": asset.get("pool_family"),
        "cara_pool_broad_style_summary": asset.get("broad_style_summary"),
    }


def _source_manifest_v2_updates(asset: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "cara_v2_source_asset_id": asset.get("asset_id"),
        "cara_v2_source_pool_id": assignment.get("pool_id"),
        "cara_v2_source_pool_assignment_status": assignment.get("assignment_status"),
        "cara_v2_source_pool_reason_codes": assignment.get("reason_codes", []),
        "cara_v2_source_pool_review_required": bool(assignment.get("review_required")),
        "cara_v2_source_pool_assignment_id": assignment.get("assignment_id"),
        "cara_v2_source_pool_last_assigned_utc": assignment.get("assigned_at"),
        "cara_v2_pool_family": asset.get("pool_family"),
    }


def _is_training_assignment(assignment: dict[str, Any]) -> bool:
    return _sanitize_text(assignment.get("assignment_status")) in POOL_ASSIGNED_STATUSES and bool(_sanitize_text(assignment.get("pool_id")))


def _save_outputs(
    paths: AllocatorV2Paths,
    manifest_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> None:
    save_manifest_rows(manifest_rows, paths.manifest_path, export_csv=False)
    export_manifest_csv(manifest_rows, paths.manifest_path.with_suffix(".csv"))
    save_manifest_rows(training_rows, paths.cara_manifest_path, export_csv=False)
    export_manifest_csv(training_rows, paths.cara_manifest_csv_path)


def _build_registry_plan(
    *,
    assets: list[dict[str, Any]],
    registry: dict[str, list[dict[str, Any]]],
    config: AllocatorV2Config,
    rng: random.Random,
    options: RunOptionsV2,
    id_counters: dict[str, int],
) -> None:
    assets_by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        assets_by_group[_group_key(asset, options.allow_relaxed_metadata)].append(asset)
    exception_artists = _artist_exception_artists(assets_by_group, config)
    plan_summary: dict[str, Any] = {
        "target_pool_count": config.target_pool_count,
        "pool_duration_cap_seconds": config.max_pool_duration_seconds,
        "general_artist_cap_seconds": config.general_artist_cap_seconds,
        "groups": [],
    }

    for group_key, group_assets in sorted(assets_by_group.items(), key=lambda item: _pool_group_id(item[0])):
        artists = exception_artists.get(group_key, set())
        planned_groups: list[tuple[str, str | None, list[dict[str, Any]]]] = []
        if artists:
            for artist in sorted(artists):
                artist_assets = [asset for asset in group_assets if _artist_key(asset) == artist]
                if artist_assets:
                    planned_groups.append(("artist_concentrated_pool", artist, artist_assets))
        general_assets = [asset for asset in group_assets if _artist_key(asset) not in artists]
        if general_assets:
            planned_groups.append(("general_pool", None, general_assets))

        for pool_type, artist, pool_assets in planned_groups:
            pool_bins: list[dict[str, Any]] = []
            for asset in sorted(pool_assets, key=lambda item: float(item.get("duration_seconds") or 0.0), reverse=True):
                duration = float(asset.get("duration_seconds") or 0.0)
                if duration <= 0:
                    assignment_id = _next_counter_id(id_counters, "assign")
                    registry["assignments"].append(
                        _assignment_row(
                            asset=asset,
                            pool=None,
                            assignment_id=assignment_id,
                            status=STATUS_REVIEW,
                            reason_codes=["DURATION_MISSING", "REVIEW_REQUIRED"],
                            review_required=True,
                        )
                    )
                    continue
                if duration > config.max_pool_duration_seconds:
                    assignment_id = _next_counter_id(id_counters, "assign")
                    registry["assignments"].append(
                        _assignment_row(
                            asset=asset,
                            pool=None,
                            assignment_id=assignment_id,
                            status=STATUS_REJECTED,
                            reason_codes=["ASSET_EXCEEDS_POOL_CAP"],
                            review_required=False,
                        )
                    )
                    continue

                eligible = [
                    pool
                    for pool in pool_bins
                    if _pool_has_capacity(pool, asset, config) and _general_artist_cap_ok(pool, asset, config)
                ]
                if eligible:
                    selected_pool = max(eligible, key=lambda pool: float(pool.get("current_duration_seconds") or 0.0))
                    status = STATUS_ASSIGNED
                    reason_codes = [
                        "V2_PLANNED_POOL_MAP",
                        "LICENCE_MATCH",
                        "TERRITORY_MATCH",
                        "BROAD_STYLE_FAMILY_MATCH",
                        "POOL_CAPACITY_AVAILABLE",
                    ]
                else:
                    selected_pool = _create_pool(
                        registry=registry,
                        config=config,
                        rng=rng,
                        group_key=group_key,
                        pool_type=pool_type,
                        spillover_index=len(pool_bins) + 1,
                        artist_key=artist,
                    )
                    registry["pools"].append(selected_pool)
                    pool_bins.append(selected_pool)
                    status = STATUS_NEW_POOL
                    reason_codes = [
                        "V2_PLANNED_POOL_MAP",
                        "POOL_SPILLOVER_CREATED" if len(pool_bins) > 1 else "NEW_POOL_CREATED",
                        "LICENCE_MATCH",
                        "TERRITORY_MATCH",
                        "BROAD_STYLE_FAMILY_MATCH",
                    ]
                if pool_type == "artist_concentrated_pool":
                    reason_codes.append("ARTIST_CATALOGUE_EXCEPTION")
                else:
                    reason_codes.append("ARTIST_CAP_OK")
                _update_pool(selected_pool, asset)
                assignment_id = _next_counter_id(id_counters, "assign")
                registry["assignments"].append(
                    _assignment_row(
                        asset=asset,
                        pool=selected_pool,
                        assignment_id=assignment_id,
                        status=status,
                        reason_codes=list(dict.fromkeys(reason_codes)),
                    )
                )
            plan_summary["groups"].append(
                {
                    "group_id": _pool_group_id(group_key, artist_key=artist),
                    "pool_type": pool_type,
                    "asset_count": len(pool_assets),
                    "duration_seconds": round(sum(float(asset.get("duration_seconds") or 0.0) for asset in pool_assets), 4),
                    "pool_count": len(pool_bins),
                }
            )
    registry["plan"] = [plan_summary]


def _exact_duplicate_keys(asset: dict[str, Any]) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    for field, reason in (
        ("isrc", "DUPLICATE_ISRC_FOUND"),
        ("iswc", "DUPLICATE_ISWC_FOUND"),
        ("audio_fingerprint", "DUPLICATE_FINGERPRINT_FOUND"),
        ("content_hash", "DUPLICATE_CONTENT_HASH_FOUND"),
    ):
        value = _sanitize_text(asset.get(field))
        if value:
            keys.append((field, value, reason))
    return keys


def _fuzzy_duplicate_key(asset: dict[str, Any], config: AllocatorV2Config) -> tuple[str, str, int] | None:
    title = _normalize_phrase(asset.get("title"))
    artist = _normalize_phrase(asset.get("artist_primary"))
    duration = float(asset.get("duration_seconds") or 0.0)
    if not title or not artist or duration <= 0:
        return None
    bucket = int(round(duration / max(float(config.fuzzy_duration_tolerance_seconds), 1.0)))
    return title, artist, bucket


def reset_allocator_v2_state(paths: AllocatorV2Paths) -> dict[str, Any]:
    existing = _load_registry(paths)
    for path in (
        paths.assets_path,
        paths.assignments_path,
        paths.duplicates_path,
        paths.runs_path,
        paths.progress_path,
        paths.pools_path,
        paths.plan_path,
        paths.cara_manifest_path,
        paths.cara_manifest_csv_path,
    ):
        if path.exists():
            path.unlink()
    return {
        "status": "reset",
        "cleared_assets": len(existing["assets"]),
        "cleared_pools": len(existing["pools"]),
        "cleared_assignments": len(existing["assignments"]),
    }


def run_pool_allocation_v2(
    paths: AllocatorV2Paths,
    options: RunOptionsV2 | None = None,
    config: AllocatorV2Config | None = None,
    *,
    progress_callback: Any = None,
    stop_requested: Any = None,
) -> dict[str, Any]:
    options = options or RunOptionsV2()
    config = config or load_allocator_v2_config()
    previous_registry = _load_registry(paths)
    manifest_rows, manifest_errors = _load_manifest_rows_lenient(paths.manifest_path)
    candidate_rows = _candidate_manifest_rows_v2(manifest_rows, options, paths, config)
    run_id = _next_prefixed_id([{"run_id": row.get("run_id")} for row in previous_registry["runs"]], "run")
    started_at = utc_now()
    run_summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "options": asdict(options),
        "processed_assets": 0,
        "total_assets": len(candidate_rows),
        "counts": _empty_counts(),
        "status": "running",
        "allocation_engine_version": POOL_V2_ENGINE_VERSION,
        "manifest_errors": manifest_errors,
    }
    progress_state = _default_progress_state(options)
    progress_state.update(
        {
            "status": "running",
            "run_id": run_id,
            "started_at": started_at,
            "updated_at": started_at,
            "total_assets": len(candidate_rows),
            "counts": dict(run_summary["counts"]),
        }
    )
    _append_progress_activity(
        progress_state,
        f"v2 planning run started for {len(candidate_rows)} candidate assets",
        phase="planning",
    )
    _write_progress_state(paths, progress_state)
    if callable(progress_callback):
        progress_callback(dict(progress_state))

    registry: dict[str, list[dict[str, Any]]] = {
        "assets": [],
        "pools": [],
        "assignments": [],
        "duplicates": [],
        "runs": [*previous_registry["runs"], run_summary],
        "plan": [],
    }
    accepted_assets: list[dict[str, Any]] = []
    asset_by_id: dict[str, dict[str, Any]] = {}
    row_by_asset_id: dict[str, dict[str, Any]] = {}
    exact_duplicate_index: dict[tuple[str, str], dict[str, Any]] = {}
    fuzzy_duplicate_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    id_counters = {"asset": 1, "assign": 1, "duplicate": 1}

    processed = 0
    for row_index, row in candidate_rows:
        if callable(stop_requested) and stop_requested():
            run_summary["processed_assets"] = processed
            run_summary["finished_at"] = utc_now()
            run_summary["status"] = "paused"
            progress_state["status"] = "paused"
            progress_state["finished_at"] = run_summary["finished_at"]
            _append_progress_activity(
                progress_state,
                f"v2 planning paused after {processed} / {len(candidate_rows)} assets",
                phase="paused",
                level="warn",
            )
            registry["runs"][-1] = run_summary
            _persist_registry(paths, registry)
            _write_progress_state(paths, progress_state)
            return run_summary

        asset = _normalize_asset_v2(row, row_index, paths)
        asset["asset_id"] = _next_counter_id(id_counters, "asset")
        processed += 1
        progress_state["processed_assets"] = processed
        progress_state["percent_complete"] = round((processed / len(candidate_rows)) * 50.0, 2) if candidate_rows else 50.0
        progress_state["current_asset"] = asset["asset_id"]
        progress_state["current_asset_title"] = asset.get("title")
        progress_state["current_phase"] = "normalizing"
        progress_state["updated_at"] = utc_now()
        if processed % 250 == 0 and callable(progress_callback):
            progress_callback(dict(progress_state))

        registry["assets"].append(asset)
        asset_by_id[asset["asset_id"]] = asset
        row_by_asset_id[asset["asset_id"]] = row

        if not asset.get("licence_class"):
            assignment_id = _next_counter_id(id_counters, "assign")
            registry["assignments"].append(
                _assignment_row(
                    asset=asset,
                    pool=None,
                    assignment_id=assignment_id,
                    status=STATUS_REVIEW,
                    reason_codes=["LICENCE_MISSING", "REVIEW_REQUIRED"],
                    review_required=True,
                )
            )
            continue
        if not asset.get("pool_family") or asset.get("pool_family") == "Mixed/Unclassified":
            # Mixed/Unclassified is still allocatable in relaxed mode so long as the
            # licence is known; otherwise v2 would create another review backlog.
            asset["pool_family"] = "Mixed/Unclassified"
        if not _rights_key(asset, options.allow_relaxed_metadata):
            assignment_id = _next_counter_id(id_counters, "assign")
            registry["assignments"].append(
                _assignment_row(
                    asset=asset,
                    pool=None,
                    assignment_id=assignment_id,
                    status=STATUS_REVIEW,
                    reason_codes=["RIGHTSHOLDER_MISSING", "REVIEW_REQUIRED"],
                    review_required=True,
                )
            )
            continue
        duplicate_reason = None
        duplicate_asset = None
        for field, value, reason in _exact_duplicate_keys(asset):
            duplicate_asset = exact_duplicate_index.get((field, value))
            if duplicate_asset:
                duplicate_reason = reason
                break
        fuzzy_key = _fuzzy_duplicate_key(asset, config)
        if duplicate_reason is None and fuzzy_key is not None:
            duplicate_asset = fuzzy_duplicate_index.get(fuzzy_key)
            if duplicate_asset and _is_fuzzy_duplicate(asset, duplicate_asset, config):  # type: ignore[arg-type]
                duplicate_reason = "POTENTIAL_DUPLICATE_FUZZY_MATCH"
        if duplicate_reason == "POTENTIAL_DUPLICATE_FUZZY_MATCH":
            duplicate_id = _next_counter_id(id_counters, "duplicate")
            registry["duplicates"].append(
                {
                    "duplicate_id": duplicate_id,
                    "asset_id": asset["asset_id"],
                    "duplicate_of_asset_id": duplicate_asset.get("asset_id") if duplicate_asset else None,
                    "reason_code": duplicate_reason,
                    "detected_at": utc_now(),
                    "review_required": True,
                }
            )
            assignment_id = _next_counter_id(id_counters, "assign")
            registry["assignments"].append(
                _assignment_row(
                    asset=asset,
                    pool=None,
                    assignment_id=assignment_id,
                    status=STATUS_REVIEW,
                    reason_codes=["POTENTIAL_DUPLICATE_MATCH", "REVIEW_REQUIRED"],
                    review_required=True,
                )
            )
            continue
        if duplicate_reason and duplicate_asset:
            duplicate_id = _next_counter_id(id_counters, "duplicate")
            registry["duplicates"].append(
                {
                    "duplicate_id": duplicate_id,
                    "asset_id": asset["asset_id"],
                    "duplicate_of_asset_id": duplicate_asset.get("asset_id"),
                    "reason_code": duplicate_reason,
                    "detected_at": utc_now(),
                    "review_required": False,
                }
            )
            assignment_id = _next_counter_id(id_counters, "assign")
            registry["assignments"].append(
                _assignment_row(
                    asset=asset,
                    pool=None,
                    assignment_id=assignment_id,
                    status=STATUS_DUPLICATE,
                    reason_codes=[duplicate_reason],
                    review_required=False,
                )
            )
            continue
        accepted_assets.append(asset)
        for field, value, _reason in _exact_duplicate_keys(asset):
            exact_duplicate_index.setdefault((field, value), asset)
        if fuzzy_key is not None:
            fuzzy_duplicate_index.setdefault(fuzzy_key, asset)

    _append_progress_activity(
        progress_state,
        f"v2 normalized {processed} assets; building broad pool map",
        phase="planning",
    )
    if callable(progress_callback):
        progress_callback(dict(progress_state))

    _build_registry_plan(
        assets=accepted_assets,
        registry=registry,
        config=config,
        rng=random.Random(run_id),
        options=options,
        id_counters=id_counters,
    )

    assignment_by_asset_id = {
        str(assignment.get("asset_id")): assignment
        for assignment in registry["assignments"]
        if assignment.get("asset_id")
    }
    training_rows: list[dict[str, Any]] = []
    for assignment in registry["assignments"]:
        asset = asset_by_id.get(str(assignment.get("asset_id")))
        if not asset:
            continue
        row = row_by_asset_id.get(asset["asset_id"])
        if row is not None:
            row.update(_source_manifest_v2_updates(asset, assignment))
        if _is_training_assignment(assignment):
            training_row = dict(row or {})
            training_row.update(_manifest_updates_from_assignment(asset, assignment))
            training_rows.append(training_row)

    counts = Counter(str(assignment.get("assignment_status") or STATUS_UNRESOLVED) for assignment in registry["assignments"])
    for key in _empty_counts():
        counts.setdefault(key, 0)
    run_summary["counts"] = dict(counts)
    run_summary["processed_assets"] = len(candidate_rows)
    run_summary["finished_at"] = utc_now()
    run_summary["status"] = "completed"
    run_summary["pool_count"] = len(registry["pools"])
    run_summary["training_manifest_rows"] = len(training_rows)
    registry["runs"][-1] = run_summary

    _persist_registry(paths, registry)
    _write_json(paths.plan_path, registry["plan"][0] if registry.get("plan") else {})
    _save_outputs(paths, manifest_rows, training_rows)

    progress_state.update(
        {
            "status": "completed",
            "finished_at": run_summary["finished_at"],
            "updated_at": run_summary["finished_at"],
            "processed_assets": len(candidate_rows),
            "total_assets": len(candidate_rows),
            "percent_complete": 100.0 if candidate_rows else 0.0,
            "counts": dict(run_summary["counts"]),
            "current_phase": "completed",
            "current_asset": None,
            "current_asset_title": None,
            "current_pool_id": None,
        }
    )
    _append_progress_activity(
        progress_state,
        f"v2 completed: {len(training_rows)} training rows across {len(registry['pools'])} registered pools",
        phase="completed",
    )
    _write_progress_state(paths, progress_state)
    if callable(progress_callback):
        progress_callback(dict(progress_state))
    return run_summary


def summarize_registry_v2(paths: AllocatorV2Paths, config: AllocatorV2Config | None = None) -> dict[str, Any]:
    config = config or load_allocator_v2_config()
    registry = _load_registry(paths)
    assignments = registry["assignments"]
    counts = Counter(str(assignment.get("assignment_status") or STATUS_UNRESOLVED) for assignment in assignments)
    for status in _empty_counts():
        counts.setdefault(status, 0)
    latest_run = registry["runs"][-1] if registry["runs"] else None
    manifest_rows, _ = _load_manifest_rows_lenient(paths.manifest_path)
    options = RunOptionsV2(subset_role="music_train_candidate", only_downloaded=True, allow_relaxed_metadata=True)
    candidate_count = len(_candidate_manifest_rows_v2(manifest_rows, options, paths, config))
    return {
        "engine_version": "v2",
        "allocation_engine_version": POOL_V2_ENGINE_VERSION,
        "counts": dict(counts),
        "pool_count": len(registry["pools"]),
        "asset_count": len(registry["assets"]),
        "candidate_asset_count": candidate_count,
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
            "max_artist_duration_seconds": config.general_artist_cap_seconds,
            "repair_threshold": config.repair_threshold,
            "min_pool_code_edit_distance": config.min_pool_code_edit_distance,
            "target_pool_count": config.target_pool_count,
            "artist_exception_min_duration_seconds": config.artist_exception_min_duration_seconds,
        },
    }


def _top_artist_share(pool: dict[str, Any]) -> float:
    artist_map = dict(pool.get("artist_duration_seconds") or {})
    if not artist_map:
        return 0.0
    return max(float(value) for value in artist_map.values()) / max(float(pool.get("pool_duration_cap_seconds") or 1.0), 1.0)


def list_pools_v2(paths: AllocatorV2Paths) -> list[dict[str, Any]]:
    registry = _load_registry(paths)
    rows: list[dict[str, Any]] = []
    for pool in registry["pools"]:
        current_duration = float(pool.get("current_duration_seconds") or 0.0)
        cap = float(pool.get("pool_duration_cap_seconds") or 0.0)
        rows.append(
            {
                "pool_id": pool.get("pool_id"),
                "licence_class": pool.get("licence_class"),
                "territory": pool.get("territory"),
                "record_label": pool.get("record_label"),
                "rights_holder_group": pool.get("rights_holder_group"),
                "primary_genre": pool.get("primary_genre"),
                "pool_family": pool.get("pool_family"),
                "pool_type": pool.get("pool_type"),
                "spillover_index": pool.get("spillover_index"),
                "included_primary_genres": pool.get("included_primary_genres", []),
                "asset_count": int(pool.get("asset_count") or 0),
                "current_duration_seconds": current_duration,
                "remaining_capacity_seconds": max(cap - current_duration, 0.0),
                "top_artist_share": _top_artist_share(pool),
                "style_summary": pool.get("style_profile", {}).get("style_summary"),
                "updated_at": pool.get("updated_at"),
            }
        )
    rows.sort(key=lambda row: (str(row.get("pool_family") or ""), str(row.get("pool_id") or "")))
    return rows


def list_assignments_v2(paths: AllocatorV2Paths, limit: int = 200) -> list[dict[str, Any]]:
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
                "pool_type": assignment.get("pool_type"),
                "pool_family": assignment.get("pool_family"),
            }
        )
    return rows


def list_review_queue_v2(paths: AllocatorV2Paths, limit: int = 200) -> list[dict[str, Any]]:
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
