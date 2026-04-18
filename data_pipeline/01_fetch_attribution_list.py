from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.attribution_utils import (
    canonical_source_id,
    display_license,
    extract_sound_id_from_url,
    infer_audio_extension,
    load_csv_rows,
    normalize_license,
    strip_audio_extension,
)
from data_pipeline.manifest_utils import export_manifest_csv, get_manifest_paths, save_manifest_rows

ATTRIBUTION_URL = "https://info.stability.ai/hubfs/freesound_dataset_attribution2%20(1).csv?hsLang=en"
EXPECTED_FREESOUND_COUNT = 472618
EXPECTED_FMA_COUNT = 13874

def build_master_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    master_rows: list[dict[str, object]] = []
    for row in rows:
        source_id = canonical_source_id(row.get("id"), row.get("url"))
        url_sound_id = extract_sound_id_from_url(row.get("url", ""))
        license_normalized = normalize_license(row.get("license"))
        master_rows.append(
            {
                "source": "freesound",
                "source_id": source_id,
                "raw_id": int(row.get("id")) if str(row.get("id", "")).isdigit() else (row.get("id") or "").strip() or None,
                "url_sound_id": url_sound_id,
                "id_matches_url": (
                    True
                    if source_id and url_sound_id is not None and source_id == str(url_sound_id)
                    else False
                    if source_id and url_sound_id is not None
                    else None
                ),
                "title": (row.get("title") or "").strip(),
                "title_stem": strip_audio_extension(row.get("title", "")),
                "author": (row.get("author") or "").strip(),
                "license_raw": (row.get("license") or "").strip(),
                "license_normalized": license_normalized,
                "license_display": display_license(license_normalized),
                "url": (row.get("url") or "").strip(),
                "file_extension": infer_audio_extension(row.get("title", ""), row.get("url", "")),
                "original_training_dataset": "Freesound",
                "original_training_manifest": "stable-audio-open-small-freesound-attribution",
                "originally_in_stable_audio_open_small": True,
                "api_enrichment_status": "pending",
                "api_last_checked_utc": None,
                "api_current_name": None,
                "api_current_license_raw": None,
                "api_current_license_normalized": None,
                "api_current_tags_json": [],
                "api_current_description": None,
                "api_current_duration_s": None,
                "api_current_samplerate": None,
                "api_current_channels": None,
                "api_analysis_available": None,
                "api_bpm": None,
                "api_key": None,
                "api_voice_instrumental": None,
                "api_genre_inferred": None,
                "api_error_message": None,
                "cara_label_status": "unlabeled",
                "cara_label_source": None,
                "cara_label_updated_utc": None,
                "cara_tier1": None,
                "cara_tier2": None,
                "cara_primary_pool": None,
                "cara_candidate_pools_json": [],
                "cara_soft_targets_json": [],
                "cara_family_codeword": None,
                "cara_codeword": None,
                "cara_auto_label_score": None,
                "cara_auto_label_confidence": None,
                "cara_auto_label_bucket": None,
                "cara_matched_keywords_json": {},
                "include_in_subset": False,
                "subset_role": None,
                "subset_note": None,
                "local_audio_path": None,
                "local_sidecar_path": None,
                "download_status": "not_downloaded",
                "content_fingerprint": None,
                "manifest_notes": None,
            }
        )
    return master_rows


def parse_attribution_ids(master_rows: list[dict[str, str]]) -> dict[str, list[int]]:
    freesound_ids = sorted({int(row["source_id"]) for row in master_rows if row.get("source_id", "").isdigit()})
    return {"freesound_ids": freesound_ids, "fma_ids": []}


def build_report(master_rows: list[dict[str, str]]) -> dict[str, object]:
    unique_ids = {row["source_id"] for row in master_rows if row.get("source_id")}
    license_counts = Counter(row["license_normalized"] for row in master_rows)
    raw_license_counts = Counter(row["license_raw"] for row in master_rows)
    extension_counts = Counter(row["file_extension"] or "<none>" for row in master_rows)
    mismatch_rows = sum(1 for row in master_rows if row.get("id_matches_url") == "false")
    missing_ids = sum(1 for row in master_rows if not row.get("source_id"))
    return {
        "total_rows": len(master_rows),
        "freesound_count": len(unique_ids),
        "fma_count": 0,
        "expected_freesound_count": EXPECTED_FREESOUND_COUNT,
        "expected_fma_count": EXPECTED_FMA_COUNT,
        "freesound_delta": len(unique_ids) - EXPECTED_FREESOUND_COUNT,
        "fma_delta": -EXPECTED_FMA_COUNT,
        "duplicate_rows": len(master_rows) - len(unique_ids),
        "rows_missing_sound_id": missing_ids,
        "rows_with_id_url_mismatch": mismatch_rows,
        "license_normalized_counts": dict(sorted(license_counts.items())),
        "license_raw_top_10": dict(raw_license_counts.most_common(10)),
        "file_extension_top_10": dict(extension_counts.most_common(10)),
    }


def parse_args() -> argparse.Namespace:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    manifest_jsonl_path, manifest_csv_path = get_manifest_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=freesound_cfg.get("attribution_csv_url", ATTRIBUTION_URL))
    parser.add_argument("--output", default=freesound_cfg.get("attribution_list_path", "data/attribution_list.json"))
    parser.add_argument("--manifest-output", default=str(manifest_jsonl_path))
    parser.add_argument("--manifest-csv-output", default=str(manifest_csv_path))
    parser.add_argument("--report-output", default="data/attribution_master_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.url)
    if args.limit is not None:
        rows = rows[: args.limit]
    master_rows = build_master_rows(rows)
    data = parse_attribution_ids(master_rows)
    report = build_report(master_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    save_manifest_rows(master_rows, Path(args.manifest_output), export_csv=False)
    export_manifest_csv(master_rows, Path(args.manifest_csv_output))

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
