from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.freesound_api import FreesoundClient, FreesoundRateLimitError, safe_suffix_from_metadata
from data_pipeline.manifest_utils import (
    coerce_source_id,
    load_manifest_rows,
    merge_manifest_updates,
    save_manifest_rows,
)


DEFAULT_METADATA_FIELDS = [
    "id",
    "name",
    "username",
    "license",
    "duration",
    "samplerate",
    "channels",
    "type",
    "tags",
    "description",
    "url",
    "homepage",
    "original_filename",
    "filename",
    "filesize",
]


def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_unavailable(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sound_id", "reason"])
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _classify_failure(exc: Exception) -> tuple[bool, str]:
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, FreesoundRateLimitError) or "rate limited" in lowered or "429" in lowered:
        return True, "temporary_rate_limited"
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True, "temporary_network"
    if isinstance(exc, requests.HTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403, 404, 410}:
            return False, f"permanent_http_{status_code}"
        if status_code is not None:
            return True, f"temporary_http_{status_code}"
    if "timed out" in lowered or "connection" in lowered or "temporar" in lowered:
        return True, "temporary_network"
    return False, "permanent_error"


def _save_download(response, target_path: Path, progress_callback: Optional[Callable[[int], None]] = None) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                if progress_callback is not None:
                    progress_callback(len(chunk))


def _load_cached_metadata(meta_dir: Path, sound_id: int) -> dict[str, Any] | None:
    metadata_path = meta_dir / f"{sound_id}.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_target_path(output_dir: Path, sound_id: int, metadata: dict[str, Any]) -> Path:
    suffix = safe_suffix_from_metadata(metadata)
    return output_dir / f"{sound_id}{suffix}"


def _audio_file_is_usable(target_path: Path, metadata: dict[str, Any]) -> bool:
    if not target_path.exists() or not target_path.is_file() or target_path.stat().st_size <= 0:
        return False
    expected_size = metadata.get("filesize")
    if expected_size in (None, "", 0, "0"):
        return True
    try:
        return target_path.stat().st_size == int(expected_size)
    except Exception:
        return True


def _row_matches_subset_mode(row: dict[str, Any], subset_mode: str, subset_role: str | None) -> bool:
    if row.get("source") != "freesound":
        return False
    if subset_mode == "confirmed_only":
        return row.get("prefilter_status") == "confirmed"
    if subset_mode == "subset_role":
        if not row.get("include_in_subset"):
            return False
        if subset_role:
            return str(row.get("subset_role") or "") == subset_role
        return True
    if subset_mode == "all_freesound":
        return True
    raise ValueError(f"Unknown subset_mode: {subset_mode}")


def _candidate_rows(manifest_rows: list[dict[str, Any]], subset_mode: str, subset_role: str | None = None) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if not _row_matches_subset_mode(row, subset_mode=subset_mode, subset_role=subset_role):
            continue
        source_id = str(row.get("source_id") or "").strip()
        if not source_id or source_id in deduped:
            continue
        deduped[source_id] = row
    return list(deduped.values())


def _flush_manifest_updates(
    manifest_path: Path,
    pending_updates: dict[str, dict[str, Any]],
) -> None:
    """Merge per-row download deltas into whatever is currently on disk.

    Loads the manifest fresh at flush time so that concurrent edits (e.g. by
    ``03b_select_music_subset.py``) are preserved. Only the fields that the
    downloader explicitly tracked as changed are written; everything else on
    every row is left untouched.
    """
    if not pending_updates:
        return
    current_rows = load_manifest_rows(manifest_path)
    updated_rows, _ = merge_manifest_updates(current_rows, pending_updates)
    save_manifest_rows(updated_rows, manifest_path)
    pending_updates.clear()


def _flush_state(
    progress_path: Path,
    manifest_path: Path,
    completed_ids: set[int],
    metadata_only_ids: set[int],
    unavailable_ids: set[int],
    pending_updates: dict[str, dict[str, Any]],
    activity_log: list[dict[str, Any]] | None = None,
    unavailable_reasons: dict[str, Any] | None = None,
) -> None:
    progress = {
        "completed_ids": sorted(completed_ids),
        "metadata_only_ids": sorted(metadata_only_ids),
        "unavailable_ids": sorted(unavailable_ids),
        "active_batch": None,
    }
    if activity_log is not None:
        progress["activity_log"] = activity_log[-120:]
    if unavailable_reasons is not None:
        progress["unavailable_reasons"] = unavailable_reasons
    _write_json(progress_path, progress)
    _flush_manifest_updates(manifest_path, pending_updates)


def download_freesound_subset(
    manifest_path: Path,
    output_dir: Path,
    meta_dir: Path,
    unavailable_log: Path,
    progress_path: Path,
    limit: int | None = None,
    skip_audio: bool = False,
    fetch_analysis: bool = False,
    require_confirmed: bool = True,
    subset_mode: str = "confirmed_only",
    subset_role: str | None = None,
    bulk_metadata_batch_size: int = 25,
    manifest_write_every: int = 25,
    cancel_check: "Callable[[], bool] | None" = None,
) -> dict[str, int]:
    config = load_pipeline_config()
    client = FreesoundClient(config)
    freesound_cfg = config.get("freesound", {})
    manifest_rows = load_manifest_rows(manifest_path)
    batch_size = max(int(freesound_cfg.get("bulk_metadata_batch_size", bulk_metadata_batch_size) or bulk_metadata_batch_size), 1)
    write_every = max(int(freesound_cfg.get("manifest_write_every", manifest_write_every) or manifest_write_every), 1)
    max_download_workers = max(int(freesound_cfg.get("max_concurrent_downloads", 4) or 1), 1)
    effective_subset_mode = subset_mode
    if require_confirmed and subset_mode == "confirmed_only":
        effective_subset_mode = "confirmed_only"
    all_candidate_rows = _candidate_rows(manifest_rows, subset_mode=effective_subset_mode, subset_role=subset_role)

    progress = _load_json(progress_path, {"completed_ids": [], "metadata_only_ids": [], "unavailable_ids": [], "unavailable_reasons": {}})
    completed_ids = set(progress.get("completed_ids", []))
    metadata_only_ids = set(progress.get("metadata_only_ids", []))
    unavailable_ids = set(progress.get("unavailable_ids", []))
    unavailable_reasons: dict[str, Any] = dict(progress.get("unavailable_reasons", {}) or {})

    candidate_rows = [
        row for row in all_candidate_rows
        if int(row["source_id"]) not in completed_ids
        and int(row["source_id"]) not in unavailable_ids
        and int(row["source_id"]) not in metadata_only_ids
    ]
    if limit is not None:
        candidate_rows = candidate_rows[:limit]

    downloaded = 0
    metadata_only = 0
    unavailable_rows: list[dict[str, str]] = []
    cached_metadata_hits = 0
    bulk_metadata_calls = 0
    single_metadata_fallback_calls = 0
    existing_audio_skips = 0
    skipped_non_confirmed = sum(1 for row in manifest_rows if row.get("source") == "freesound" and row.get("prefilter_status") != "confirmed") if effective_subset_mode == "confirmed_only" else 0
    pending_rows = candidate_rows
    progress_lock = threading.Lock()
    active_download_bytes: dict[int, int] = {}
    active_download_expected_bytes: dict[int, int] = {}
    activity_log: list[dict[str, Any]] = list(progress.get("activity_log", []))[-120:]
    last_progress_write_at = 0.0

    def _log_activity(message: str, phase: str | None = None, sound_id: int | None = None, level: str = "info") -> None:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "level": level,
            "message": message,
        }
        if phase:
            entry["phase"] = phase
        if sound_id is not None:
            entry["sound_id"] = sound_id
        activity_log.append(entry)
        del activity_log[:-120]

    metadata_cache: dict[int, dict[str, Any]] = {}
    missing_ids: list[int] = []
    for row in pending_rows:
        sound_id = int(row["source_id"])
        cached_metadata = _load_cached_metadata(meta_dir, sound_id)
        if cached_metadata is not None:
            metadata_cache[sound_id] = cached_metadata
            cached_metadata_hits += 1
        else:
            missing_ids.append(sound_id)

    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start:start + batch_size]
        if not batch_ids:
            continue
        try:
            batch_payload = client.fetch_sounds_bulk(batch_ids, fields=DEFAULT_METADATA_FIELDS)
            bulk_metadata_calls += 1
            _log_activity(f"Fetched bulk metadata for {len(batch_payload)} / {len(batch_ids)} items", phase="metadata")
        except Exception:
            _log_activity(f"Bulk metadata fetch failed for {len(batch_ids)} items; falling back to per-item metadata", phase="metadata", level="warn")
            batch_payload = {}
        for sound_id in batch_ids:
            metadata = batch_payload.get(sound_id)
            if metadata is not None:
                metadata_cache[sound_id] = metadata

    processed_since_flush = 0
    cancelled = False

    # Collect per-row field deltas instead of mutating the in-memory manifest.
    # At flush time these deltas are merged into a freshly-loaded manifest so
    # concurrent edits from other tools (e.g. 03b_select_music_subset) are not
    # overwritten by a stale in-memory snapshot.
    pending_row_updates: dict[str, dict[str, Any]] = {}

    def _stage(sound_id: int, **fields: Any) -> None:
        key = coerce_source_id(sound_id)
        if not key:
            return
        bucket = pending_row_updates.setdefault(key, {})
        bucket.update(fields)

    def _write_progress_snapshot() -> None:
        _write_json(progress_path, {
            "completed_ids": sorted(completed_ids),
            "metadata_only_ids": sorted(metadata_only_ids),
            "unavailable_ids": sorted(unavailable_ids),
            "unavailable_reasons": unavailable_reasons,
            "active_batch": active_batch,
            "activity_log": activity_log,
        })

    def _write_progress_snapshot_throttled() -> None:
        nonlocal last_progress_write_at
        now = time.time()
        if now - last_progress_write_at < 1.0:
            return
        last_progress_write_at = now
        _write_progress_snapshot()

    active_batch: dict[str, Any] | None = {
        "phase": "metadata",
        "started_at": time.time(),
        "updated_at": time.time(),
        "requested": len(candidate_rows),
        "total_in_batch": len(candidate_rows),
        "completed_in_batch": 0,
        "active_ids": [],
        "message": "Preparing metadata",
    }
    _log_activity(f"Starting batch with {len(candidate_rows)} candidate items", phase="start")
    _write_progress_snapshot()

    # Items whose metadata is ready and still need an actual audio download.
    pending_downloads: list[tuple[int, Path]] = []

    # --- Phase 1 (serial): metadata fetch + write, plus classification ---
    for row in candidate_rows:
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        sound_id = int(row["source_id"])
        if active_batch is not None:
            active_batch.update({
                "phase": "metadata",
                "updated_at": time.time(),
                "current_id": sound_id,
                "active_ids": [sound_id],
                "message": f"Processing metadata for {sound_id}",
            })
            _write_progress_snapshot()
        if sound_id in completed_ids or sound_id in unavailable_ids or sound_id in metadata_only_ids:
            continue
        try:
            metadata = metadata_cache.get(sound_id)
            if metadata is None:
                _log_activity(f"Fetching single metadata for {sound_id}", phase="metadata", sound_id=sound_id)
                metadata = client.fetch_sound(sound_id, fields=DEFAULT_METADATA_FIELDS)
                single_metadata_fallback_calls += 1
            analysis = metadata.get("analysis", {}) if isinstance(metadata, dict) else {}
            if fetch_analysis and not analysis:
                try:
                    analysis = client.fetch_analysis(sound_id)
                except Exception:
                    analysis = {}
                metadata["analysis"] = analysis
            _write_json(meta_dir / f"{sound_id}.json", metadata)

            if skip_audio:
                _log_activity(f"Saved metadata only for {sound_id}", phase="metadata", sound_id=sound_id)
                metadata_only_ids.add(sound_id)
                metadata_only += 1
                _stage(
                    sound_id,
                    download_status="metadata_only",
                    local_meta_path=str(meta_dir / f"{sound_id}.json"),
                )
                processed_since_flush += 1
                _write_progress_snapshot()
                if processed_since_flush >= write_every:
                    _flush_manifest_updates(manifest_path, pending_row_updates)
                    processed_since_flush = 0
                if active_batch is not None:
                    active_batch["completed_in_batch"] = int(active_batch.get("completed_in_batch", 0)) + 1
            else:
                target_path = _resolve_target_path(output_dir, sound_id, metadata)
                if _audio_file_is_usable(target_path, metadata):
                    _log_activity(f"Audio already exists for {sound_id}; marking complete", phase="metadata", sound_id=sound_id)
                    existing_audio_skips += 1
                    completed_ids.add(sound_id)
                    downloaded += 1
                    _stage(
                        sound_id,
                        download_status="downloaded",
                        local_audio_path=str(target_path),
                        local_meta_path=str(meta_dir / f"{sound_id}.json"),
                    )
                    processed_since_flush += 1
                    _write_progress_snapshot()
                    if processed_since_flush >= write_every:
                        _flush_manifest_updates(manifest_path, pending_row_updates)
                        processed_since_flush = 0
                    if active_batch is not None:
                        active_batch["completed_in_batch"] = int(active_batch.get("completed_in_batch", 0)) + 1
                else:
                    _log_activity(f"Queued audio download for {sound_id}", phase="metadata", sound_id=sound_id)
                    pending_downloads.append((sound_id, target_path))
        except Exception as exc:
            retryable, category = _classify_failure(exc)
            level = "warn" if retryable else "error"
            _log_activity(f"Metadata failed for {sound_id}: {exc}", phase="metadata", sound_id=sound_id, level=level)
            unavailable_rows.append({"sound_id": str(sound_id), "reason": f"{category}: {exc}"})
            if retryable:
                unavailable_reasons.pop(str(sound_id), None)
            else:
                unavailable_ids.add(sound_id)
                unavailable_reasons[str(sound_id)] = {"category": category, "reason": str(exc), "ts": time.time(), "retryable": False}
                _stage(
                    sound_id,
                    download_status="unavailable",
                    manifest_notes=str(exc),
                )
            processed_since_flush += 1
            _write_progress_snapshot()
            if processed_since_flush >= write_every:
                _flush_manifest_updates(manifest_path, pending_row_updates)
                processed_since_flush = 0
            if active_batch is not None:
                active_batch["completed_in_batch"] = int(active_batch.get("completed_in_batch", 0)) + 1

    # --- Phase 2 (parallel): actual audio downloads, up to max_download_workers ---
    # Honors freesound.max_concurrent_downloads. The FreesoundClient's session
    # and internal rate limiter tolerate concurrent use (the per-minute quota is
    # still enforced by _throttle on each thread; the usage-file count may be
    # slightly racy under heavy concurrency but is recorded best-effort).
    def _download_job(sound_id: int, target_path: Path) -> None:
        _log_activity(f"Requesting original audio for {sound_id}", phase="audio_download", sound_id=sound_id)
        response = client.download_original(sound_id)
        expected_size = 0
        metadata = metadata_cache.get(sound_id) or {}
        try:
            expected_size = int(metadata.get("filesize") or 0)
        except Exception:
            expected_size = 0
        with progress_lock:
            active_download_bytes[sound_id] = 0
            active_download_expected_bytes[sound_id] = expected_size
            if active_batch is not None:
                active_batch["downloaded_bytes"] = sum(active_download_bytes.values())
                active_batch["expected_bytes"] = sum(active_download_expected_bytes.values())
                active_batch["download_progress_by_id"] = {str(k): v for k, v in active_download_bytes.items()}
                active_batch["expected_bytes_by_id"] = {str(k): v for k, v in active_download_expected_bytes.items()}
                active_batch["updated_at"] = time.time()
                active_batch["message"] = f"Started audio stream for {sound_id}"
            _log_activity(f"Started audio stream for {sound_id} ({expected_size or 'unknown'} bytes expected)", phase="audio_download", sound_id=sound_id)
            _write_progress_snapshot_throttled()

        def on_chunk(chunk_size: int) -> None:
            with progress_lock:
                active_download_bytes[sound_id] = active_download_bytes.get(sound_id, 0) + chunk_size
                if active_batch is not None:
                    active_batch["downloaded_bytes"] = sum(active_download_bytes.values())
                    active_batch["expected_bytes"] = sum(active_download_expected_bytes.values())
                    active_batch["download_progress_by_id"] = {str(k): v for k, v in active_download_bytes.items()}
                    active_batch["expected_bytes_by_id"] = {str(k): v for k, v in active_download_expected_bytes.items()}
                    active_batch["updated_at"] = time.time()
                    active_batch["message"] = f"Downloading audio for {sound_id}"
                _write_progress_snapshot_throttled()

        _save_download(response, target_path, progress_callback=on_chunk)
        _log_activity(f"Finished writing audio for {sound_id}", phase="audio_download", sound_id=sound_id)

    if pending_downloads and not cancelled:
        workers = min(max_download_workers, len(pending_downloads))
        active_download_ids = {sid for sid, _ in pending_downloads}
        if active_batch is not None:
            active_batch.update({
                "phase": "audio_download",
                "updated_at": time.time(),
                "total_audio_downloads": len(pending_downloads),
                "completed_audio_downloads": 0,
                "active_downloads": len(active_download_ids),
                "active_ids": sorted(active_download_ids),
                "message": f"Downloading {len(pending_downloads)} audio files with {workers} workers",
            })
            _write_progress_snapshot()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fs-dl") as pool:
            futures = {
                pool.submit(_download_job, sid, tp): (sid, tp)
                for sid, tp in pending_downloads
            }
            for future in as_completed(futures):
                sid, target_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    retryable, category = _classify_failure(exc)
                    level = "warn" if retryable else "error"
                    _log_activity(f"Audio download failed for {sid}: {exc}", phase="audio_download", sound_id=sid, level=level)
                    unavailable_rows.append({"sound_id": str(sid), "reason": f"{category}: {exc}"})
                    if retryable:
                        unavailable_reasons.pop(str(sid), None)
                    else:
                        unavailable_ids.add(sid)
                        unavailable_reasons[str(sid)] = {"category": category, "reason": str(exc), "ts": time.time(), "retryable": False}
                        _stage(
                            sid,
                            download_status="unavailable",
                            manifest_notes=str(exc),
                        )
                else:
                    _log_activity(f"Audio download complete for {sid}", phase="audio_download", sound_id=sid)
                    completed_ids.add(sid)
                    downloaded += 1
                    _stage(
                        sid,
                        download_status="downloaded",
                        local_audio_path=str(target_path),
                        local_meta_path=str(meta_dir / f"{sid}.json"),
                    )

                with progress_lock:
                    active_download_ids.discard(sid)
                    active_download_bytes.pop(sid, None)
                    active_download_expected_bytes.pop(sid, None)
                    if active_batch is not None:
                        active_batch["updated_at"] = time.time()
                        active_batch["completed_in_batch"] = int(active_batch.get("completed_in_batch", 0)) + 1
                        active_batch["completed_audio_downloads"] = int(active_batch.get("completed_audio_downloads", 0)) + 1
                        active_batch["active_downloads"] = len(active_download_ids)
                        active_batch["active_ids"] = sorted(active_download_ids)
                        active_batch["current_id"] = sid
                        active_batch["downloaded_bytes"] = sum(active_download_bytes.values())
                        active_batch["expected_bytes"] = sum(active_download_expected_bytes.values())
                        active_batch["download_progress_by_id"] = {str(k): v for k, v in active_download_bytes.items()}
                        active_batch["expected_bytes_by_id"] = {str(k): v for k, v in active_download_expected_bytes.items()}
                        active_batch["message"] = f"Finished audio download for {sid}"
                processed_since_flush += 1
                _write_progress_snapshot()
                if processed_since_flush >= write_every:
                    _flush_manifest_updates(manifest_path, pending_row_updates)
                    processed_since_flush = 0
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    # Let already-submitted futures drain; no new ones will be added.

    if unavailable_rows:
        _append_unavailable(unavailable_log, unavailable_rows)

    _log_activity(
        f"Batch finished: downloaded {downloaded}, metadata_only {metadata_only}, unavailable {len(unavailable_rows)}",
        phase="finish",
    )
    _flush_state(
        progress_path=progress_path,
        manifest_path=manifest_path,
        completed_ids=completed_ids,
        metadata_only_ids=metadata_only_ids,
        unavailable_ids=unavailable_ids,
        pending_updates=pending_row_updates,
        activity_log=activity_log,
        unavailable_reasons=unavailable_reasons,
    )

    return {
        "subset_mode": effective_subset_mode,
        "subset_role": subset_role or "",
        "cancelled": bool(cancelled),
        "requested": len(candidate_rows),
        "downloaded": downloaded,
        "metadata_only": metadata_only,
        "unavailable": len(unavailable_rows),
        "cached_metadata_hits": cached_metadata_hits,
        "bulk_metadata_calls": bulk_metadata_calls,
        "single_metadata_fallback_calls": single_metadata_fallback_calls,
        "downloads_skipped_existing": existing_audio_skips,
        "skipped_non_confirmed": skipped_non_confirmed,
        "api_requests_used_today": int(client.get_usage_snapshot().get("daily_count", 0)),
    }


def parse_args() -> argparse.Namespace:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=freesound_cfg.get("attribution_manifest_path", "data/attribution_manifest.jsonl"))
    parser.add_argument("--output-dir", default=freesound_cfg.get("output_dir", "data/freesound"))
    parser.add_argument("--meta-dir", default=freesound_cfg.get("meta_dir", "data/freesound_meta"))
    parser.add_argument("--unavailable-log", default=freesound_cfg.get("unavailable_log", "data/unavailable_freesound.csv"))
    parser.add_argument("--progress-path", default=freesound_cfg.get("progress_path", "data/download_progress.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--fetch-analysis", action="store_true")
    parser.add_argument("--no-require-confirmed", action="store_true")
    parser.add_argument("--subset-mode", default="confirmed_only", choices=["confirmed_only", "subset_role", "all_freesound"])
    parser.add_argument("--subset-role", default=None)
    parser.add_argument("--bulk-metadata-batch-size", type=int, default=int(freesound_cfg.get("bulk_metadata_batch_size", 25)))
    parser.add_argument("--manifest-write-every", type=int, default=int(freesound_cfg.get("manifest_write_every", 25)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = download_freesound_subset(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        meta_dir=Path(args.meta_dir),
        unavailable_log=Path(args.unavailable_log),
        progress_path=Path(args.progress_path),
        limit=args.limit,
        skip_audio=args.skip_audio,
        fetch_analysis=args.fetch_analysis,
        require_confirmed=not args.no_require_confirmed,
        subset_mode=args.subset_mode,
        subset_role=args.subset_role,
        bulk_metadata_batch_size=args.bulk_metadata_batch_size,
        manifest_write_every=args.manifest_write_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
