from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.freesound_api import FreesoundClient, safe_suffix_from_metadata
from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows


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


def _save_download(response, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_freesound_subset(
    manifest_path: Path,
    output_dir: Path,
    meta_dir: Path,
    unavailable_log: Path,
    progress_path: Path,
    limit: int | None = None,
    skip_audio: bool = False,
) -> dict[str, int]:
    config = load_pipeline_config()
    client = FreesoundClient(config)
    manifest_rows = load_manifest_rows(manifest_path)
    candidate_rows = [row for row in manifest_rows if row.get("source") == "freesound"]
    if limit is not None:
        candidate_rows = candidate_rows[:limit]

    progress = _load_json(progress_path, {"completed_ids": [], "metadata_only_ids": [], "unavailable_ids": []})
    completed_ids = set(progress.get("completed_ids", []))
    metadata_only_ids = set(progress.get("metadata_only_ids", []))
    unavailable_ids = set(progress.get("unavailable_ids", []))

    downloaded = 0
    metadata_only = 0
    unavailable_rows: list[dict[str, str]] = []

    for row in candidate_rows:
        sound_id = int(row["source_id"])
        if sound_id in completed_ids or sound_id in unavailable_ids:
            continue
        try:
            metadata = client.fetch_sound(sound_id)
            try:
                analysis = client.fetch_analysis(sound_id)
            except Exception:
                analysis = {}
            metadata["analysis"] = analysis
            _write_json(meta_dir / f"{sound_id}.json", metadata)

            if skip_audio:
                metadata_only_ids.add(sound_id)
                metadata_only += 1
                row["download_status"] = "metadata_only"
            else:
                suffix = safe_suffix_from_metadata(metadata)
                response = client.download_original(sound_id)
                target_path = output_dir / f"{sound_id}{suffix}"
                _save_download(response, target_path)
                completed_ids.add(sound_id)
                downloaded += 1
                row["download_status"] = "downloaded"
                row["local_audio_path"] = str(target_path)
        except Exception as exc:
            unavailable_ids.add(sound_id)
            unavailable_rows.append({"sound_id": str(sound_id), "reason": str(exc)})
            row["download_status"] = "unavailable"
            row["manifest_notes"] = str(exc)

        progress = {
            "completed_ids": sorted(completed_ids),
            "metadata_only_ids": sorted(metadata_only_ids),
            "unavailable_ids": sorted(unavailable_ids),
        }
        _write_json(progress_path, progress)
        save_manifest_rows(manifest_rows, manifest_path)

    if unavailable_rows:
        _append_unavailable(unavailable_log, unavailable_rows)

    return {
        "requested": len(candidate_rows),
        "downloaded": downloaded,
        "metadata_only": metadata_only,
        "unavailable": len(unavailable_rows),
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
