from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.freesound_api import FreesoundClient, FreesoundRateLimitError

DEFAULT_ATTRIBUTION_CSV_URL = "https://info.stability.ai/hubfs/freesound_dataset_attribution2%20(1).csv?hsLang=en"
DEFAULT_ALLOWED_LICENSES = [
    "creative commons 0",
    "cc0",
    "attribution",
    "cc-by",
    "cc by",
    "sampling+",
    "cc sampling+",
    "creative commons sampling+",
]
OUTPUT_FIELDS = [
    "sound_id",
    "source_url",
    "name",
    "username",
    "license",
    "duration",
    "samplerate",
    "channels",
    "type",
    "tags",
    "description",
    "genre_inferred",
    "bpm",
    "key",
    "acoustic_electronic",
    "voice_instrumental",
    "music_proxy_pass",
    "duration_pass",
    "license_pass",
    "analysis_available",
    "reason",
]


def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _fetch_csv_text(csv_source: str) -> str:
    if csv_source.startswith("http://") or csv_source.startswith("https://"):
        response = requests.get(csv_source, timeout=120)
        response.raise_for_status()
        return response.text
    return Path(csv_source).read_text(encoding="utf-8")


def _extract_sound_id(row: dict[str, str]) -> int | None:
    candidate_keys = ["id", "sound_id", "freesound_id", "soundid"]
    for key in candidate_keys:
        value = row.get(key)
        if value:
            digits = "".join(ch for ch in str(value) if ch in "0123456789")
            if digits:
                return int(digits)
    for value in row.values():
        digits = "".join(ch for ch in str(value) if ch in "0123456789")
        if digits and len(digits) >= 2:
            return int(digits)
    return None


def _normalise_license(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _license_passes(license_value: str, allowed_licenses: list[str]) -> bool:
    normalised = _normalise_license(license_value)
    if "creativecommons.org/licenses/by/" in normalised or "/licenses/by/" in normalised:
        return True
    if "creativecommons.org/publicdomain/zero/" in normalised or "publicdomain/zero/" in normalised:
        return True
    if "creativecommons.org/licenses/sampling+" in normalised or "/licenses/sampling+" in normalised:
        return True
    return any(token in normalised for token in allowed_licenses)


def _safe_get(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _music_proxy_passes(metadata: dict[str, Any], analysis: dict[str, Any], minimum_duration_s: float) -> tuple[bool, bool]:
    duration = float(metadata.get("duration") or 0.0)
    duration_pass = duration >= minimum_duration_s
    music = analysis.get("music") if isinstance(analysis, dict) else {}
    rhythm = analysis.get("rhythm") if isinstance(analysis, dict) else {}
    tonal = analysis.get("tonal") if isinstance(analysis, dict) else {}
    voice = analysis.get("voice_instrumental") if isinstance(analysis, dict) else {}

    tag_text = " ".join(str(item).lower() for item in metadata.get("tags", []))
    desc_text = " ".join(
        str(metadata.get(key, "")).lower() for key in ["name", "description", "type"]
    )
    keyword_hit = any(keyword in f"{tag_text} {desc_text}" for keyword in [
        "music",
        "song",
        "instrument",
        "drum",
        "piano",
        "guitar",
        "synth",
        "bass",
        "jazz",
        "rock",
        "techno",
        "orchestra",
    ])
    analysis_hit = any(
        [
            bool(_safe_get(music or {}, "genre_dortmund", "value", default="")),
            bool(_safe_get(rhythm or {}, "bpm", default="")),
            bool(_safe_get(tonal or {}, "key_key", default="")),
            bool(_safe_get(voice or {}, "value", default="")),
        ]
    )
    return duration_pass and (analysis_hit or keyword_hit), duration_pass


def _row_from_payload(metadata: dict[str, Any], analysis: dict[str, Any], reason: str, allowed_licenses: list[str], minimum_duration_s: float) -> dict[str, str]:
    license_value = str(metadata.get("license") or "")
    music_proxy_pass, duration_pass = _music_proxy_passes(metadata, analysis, minimum_duration_s)
    license_pass = _license_passes(license_value, allowed_licenses)
    return {
        "sound_id": str(metadata.get("id") or ""),
        "source_url": str(metadata.get("url") or metadata.get("homepage") or ""),
        "name": str(metadata.get("name") or ""),
        "username": str(metadata.get("username") or ""),
        "license": license_value,
        "duration": str(metadata.get("duration") or ""),
        "samplerate": str(metadata.get("samplerate") or ""),
        "channels": str(metadata.get("channels") or ""),
        "type": str(metadata.get("type") or ""),
        "tags": json.dumps(metadata.get("tags") or [], ensure_ascii=False),
        "description": str(metadata.get("description") or ""),
        "genre_inferred": str(_safe_get(analysis, "music", "genre_dortmund", "value", default="")),
        "bpm": str(_safe_get(analysis, "rhythm", "bpm", default="")),
        "key": str(_safe_get(analysis, "tonal", "key_key", default="")),
        "acoustic_electronic": str(_safe_get(analysis, "music", "acoustic", default="")),
        "voice_instrumental": str(_safe_get(analysis, "voice_instrumental", "value", default="")),
        "music_proxy_pass": "true" if music_proxy_pass else "false",
        "duration_pass": "true" if duration_pass else "false",
        "license_pass": "true" if license_pass else "false",
        "analysis_available": "true" if analysis else "false",
        "reason": reason,
    }


def _print_progress(processed: int, confirmed: int, rejected: int, goal: int) -> None:
    safe_goal = max(goal, 1)
    ratio = min(processed / safe_goal, 1.0)
    filled = int(ratio * 24)
    bar = f"{'#' * filled}{'-' * (24 - filled)}"
    sys.stdout.write(
        f"\r[{bar}] {processed}/{goal} processed | {confirmed} confirmed | {rejected} rejected"
    )
    sys.stdout.flush()


def prefilter_freesound_sources(
    csv_source: str,
    output_csv: Path,
    progress_path: Path,
    rejected_csv: Path,
    limit: int | None = None,
    minimum_duration_s: float = 30.0,
    allowed_licenses: list[str] | None = None,
) -> dict[str, int]:
    config = load_pipeline_config()
    client = FreesoundClient(config)
    allowed = [_normalise_license(item) for item in (allowed_licenses or DEFAULT_ALLOWED_LICENSES)]

    rows = list(csv.DictReader(_fetch_csv_text(csv_source).splitlines()))

    progress = _load_json(progress_path, {"processed_ids": [], "confirmed_ids": [], "rejected_ids": []})
    processed_ids = set(progress.get("processed_ids", []))
    confirmed_ids = set(progress.get("confirmed_ids", []))
    rejected_ids = set(progress.get("rejected_ids", []))

    confirmed_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    newly_processed = 0
    candidate_total = len(rows)
    remaining_candidates = sum(1 for source_row in rows if (_extract_sound_id(source_row) is not None and _extract_sound_id(source_row) not in processed_ids))
    stopped_due_to_rate_limit = False
    progress_goal = limit if limit is not None else remaining_candidates

    if progress_goal > 0:
        _print_progress(0, 0, 0, progress_goal)

    for source_row in rows:
        if limit is not None and newly_processed >= limit:
            break
        sound_id = _extract_sound_id(source_row)
        if sound_id is None or sound_id in processed_ids:
            continue
        try:
            metadata = client.fetch_sound(sound_id)
            try:
                analysis = client.fetch_analysis(sound_id)
            except Exception:
                analysis = {}
            processed_ids.add(sound_id)
            newly_processed += 1
            candidate_row = _row_from_payload(metadata, analysis, reason="confirmed", allowed_licenses=allowed, minimum_duration_s=minimum_duration_s)
            passes = (
                candidate_row["license_pass"] == "true"
                and candidate_row["duration_pass"] == "true"
                and candidate_row["music_proxy_pass"] == "true"
            )
            if passes:
                confirmed_ids.add(sound_id)
                confirmed_rows.append(candidate_row)
                _append_rows(output_csv, [candidate_row])
            else:
                rejected_ids.add(sound_id)
                reason = []
                if candidate_row["license_pass"] != "true":
                    reason.append("license")
                if candidate_row["duration_pass"] != "true":
                    reason.append("duration")
                if candidate_row["music_proxy_pass"] != "true":
                    reason.append("music_proxy")
                candidate_row["reason"] = ",".join(reason) or "rejected"
                rejected_rows.append(candidate_row)
                _append_rows(rejected_csv, [candidate_row])
            _print_progress(newly_processed, len(confirmed_rows), len(rejected_rows), progress_goal)
        except FreesoundRateLimitError:
            stopped_due_to_rate_limit = True
            break
        except Exception as exc:
            processed_ids.add(sound_id)
            newly_processed += 1
            rejected_ids.add(sound_id)
            rejected_row = {
                "sound_id": str(sound_id),
                "source_url": "",
                "name": "",
                "username": "",
                "license": "",
                "duration": "",
                "samplerate": "",
                "channels": "",
                "type": "",
                "tags": "[]",
                "description": "",
                "genre_inferred": "",
                "bpm": "",
                "key": "",
                "acoustic_electronic": "",
                "voice_instrumental": "",
                "music_proxy_pass": "false",
                "duration_pass": "false",
                "license_pass": "false",
                "analysis_available": "false",
                "reason": str(exc),
            }
            rejected_rows.append(rejected_row)
            _append_rows(rejected_csv, [rejected_row])
            _print_progress(newly_processed, len(confirmed_rows), len(rejected_rows), progress_goal)
        _write_json(
            progress_path,
            {
                "processed_ids": sorted(processed_ids),
                "confirmed_ids": sorted(confirmed_ids),
                "rejected_ids": sorted(rejected_ids),
            },
        )

    if progress_goal > 0:
        sys.stdout.write("\n")

    return {
        "requested": limit if limit is not None else remaining_candidates,
        "processed": newly_processed,
        "confirmed": len(confirmed_rows),
        "rejected": len(rejected_rows),
        "cumulative_processed": len(processed_ids),
        "cumulative_confirmed": len(confirmed_ids),
        "cumulative_rejected": len(rejected_ids),
        "candidate_total": candidate_total,
        "remaining_candidates": max(remaining_candidates - newly_processed, 0),
        "stopped_due_to_rate_limit": 1 if stopped_due_to_rate_limit else 0,
    }


def parse_args() -> argparse.Namespace:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-source", default=freesound_cfg.get("attribution_csv_url", DEFAULT_ATTRIBUTION_CSV_URL))
    parser.add_argument("--output-csv", default=freesound_cfg.get("filtered_sources_csv", "data/freesound_filtered_sources.csv"))
    parser.add_argument("--rejected-csv", default=freesound_cfg.get("rejected_sources_csv", "data/freesound_rejected_sources.csv"))
    parser.add_argument("--progress-path", default=freesound_cfg.get("prefilter_progress_path", "data/freesound_prefilter_progress.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--minimum-duration", type=float, default=float(freesound_cfg.get("minimum_duration_seconds", 30.0)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prefilter_freesound_sources(
        csv_source=args.csv_source,
        output_csv=Path(args.output_csv),
        progress_path=Path(args.progress_path),
        rejected_csv=Path(args.rejected_csv),
        limit=args.limit,
        minimum_duration_s=args.minimum_duration,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
