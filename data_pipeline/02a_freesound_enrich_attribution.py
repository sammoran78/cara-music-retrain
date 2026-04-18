#!/usr/bin/env python3
"""
Freesound attribution enrichment crawler.

This script treats the Stability attribution CSV as the canonical membership list
and enriches it with current Freesound API metadata. It does NOT filter or reject
rows based on music heuristics or license changes.

Core principle: Historical inclusion != Current metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_pipeline_config
from data_pipeline.freesound_api import FreesoundClient, FreesoundRateLimitError
from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows, index_manifest_rows

DEFAULT_ATTRIBUTION_CSV_URL = "https://info.stability.ai/hubfs/freesound_dataset_attribution2%20(1).csv?hsLang=en"

# Output column definitions
OUTPUT_COLUMNS = [
    # Provenance / source columns
    "attribution_row_index",
    "attribution_source_csv",
    "attribution_crawl_timestamp_utc",
    "in_stability_attribution_list",
    "source_dataset",
    "freesound_sound_id",
    "original_attribution_url",
    "original_attribution_name",
    "original_attribution_username",
    "original_attribution_license",
    "original_attribution_raw_json",
    # Current API state columns
    "api_checked_at_utc",
    "api_status",
    "api_error_message",
    "current_exists",
    "current_sound_url",
    "current_name",
    "current_username",
    "current_license",
    "current_created",
    "current_type",
    "current_duration",
    "current_samplerate",
    "current_channels",
    "current_num_downloads",
    "current_avg_rating",
    "current_num_ratings",
    "current_pack",
    "current_tags_json",
    "current_description",
    "current_geotag_json",
    "current_previews_json",
    "current_images_json",
    # Analysis / descriptors columns
    "analysis_available",
    "genre_inferred",
    "bpm",
    "key",
    "acoustic_electronic",
    "voice_instrumental",
    "onset_count",
    "note_name",
    "note_midi",
    "pitch",
    "brightness",
    "loudness",
    "warmness_or_warmth_if_available",
    "spectral_centroid_if_available",
    "analysis_raw_json",
    # Research / audit columns
    "current_license_normalized",
    "current_license_family",
    "current_license_matches_stability_allowed_set",
    "duration_ge_30s",
    "currently_accessible_for_metadata",
    "changed_from_original_attribution_license",
    "changed_from_original_attribution_name",
    "changed_from_original_attribution_username",
    "deleted_or_unavailable_since_attribution",
    "notes",
    # Optional ethics / AI preference columns
    "ai_training_preference_raw_json",
    "ai_training_preference_summary",
    "ai_training_preference_allows_open_model",
    "ai_training_preference_allows_commercial",
    "ai_training_preference_requires_disclosure",
    "ai_training_preference_present",
]


def normalize_license(license_str: str | None) -> str:
    """Normalize license string to canonical form."""
    if not license_str:
        return "unknown"
    
    license_lower = license_str.lower().strip()
    
    # CC0 / Public Domain
    if any(term in license_lower for term in ["cc0", "publicdomain", "public domain", "zero"]):
        return "cc0"
    
    # CC-BY variants
    if "by-nc-sa" in license_lower:
        return "cc-by-nc-sa"
    elif "by-nc-nd" in license_lower:
        return "cc-by-nc-nd"
    elif "by-nc" in license_lower:
        return "cc-by-nc"
    elif "by-sa" in license_lower:
        return "cc-by-sa"
    elif "by-nd" in license_lower:
        return "cc-by-nd"
    elif ("by" in license_lower and "cc" in license_lower) or "creativecommons.org/licenses/by" in license_lower:
        return "cc-by"
    
    # Sampling+
    if "sampling" in license_lower:
        return "sampling+"
    
    return "unknown"


def license_family(normalized_license: str) -> str:
    """Get license family from normalized license."""
    if normalized_license == "cc0":
        return "cc0"
    elif normalized_license == "cc-by":
        return "cc-by"
    elif normalized_license.startswith("cc-by-nc"):
        return "cc-by-nc"
    elif normalized_license == "sampling+":
        return "sampling+"
    else:
        return "unknown"


def extract_sound_id(row: dict[str, str]) -> int | None:
    """Extract Freesound sound ID from attribution row."""
    # First try explicit ID columns
    id_columns = ["id", "sound_id", "freesound_id", "soundid", "ID", "Sound_ID"]
    for col in id_columns:
        if col in row and row[col]:
            try:
                # Only accept pure numeric values
                if row[col].strip().isdigit():
                    return int(row[col].strip())
            except (ValueError, AttributeError):
                continue
    
    # Next try to extract from Freesound URL
    url_columns = ["url", "source_url", "link", "freesound_url", "URL"]
    for col in url_columns:
        if col in row and row[col]:
            url = row[col].strip()
            # Match freesound.org/sounds/123456/ pattern
            match = re.search(r'freesound\.org/sounds/(\d+)', url)
            if match:
                return int(match.group(1))
    
    # Do NOT use arbitrary digit extraction - too risky
    return None


def safe_json_dump(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return "{}"


def safe_get(obj: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Safely get nested dictionary value."""
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def enrich_row(
    row: dict[str, str],
    row_index: int,
    sound_id: int,
    client: FreesoundClient,
    csv_source: str,
    crawl_timestamp: str,
    sleep_seconds: float = 0.0
) -> dict[str, str]:
    """Enrich a single attribution row with current API data."""
    
    # Initialize output row with provenance data
    output = {col: "" for col in OUTPUT_COLUMNS}
    
    # Provenance columns
    output["attribution_row_index"] = str(row_index)
    output["attribution_source_csv"] = csv_source
    output["attribution_crawl_timestamp_utc"] = crawl_timestamp
    output["in_stability_attribution_list"] = "true"
    output["source_dataset"] = "freesound"
    output["freesound_sound_id"] = str(sound_id)
    
    # Original attribution data
    output["original_attribution_url"] = row.get("url", row.get("source_url", ""))
    output["original_attribution_name"] = row.get("name", row.get("title", ""))
    output["original_attribution_username"] = row.get("username", row.get("user", ""))
    output["original_attribution_license"] = row.get("license", "")
    output["original_attribution_raw_json"] = safe_json_dump(row)
    
    # Initialize API status
    output["api_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    
    # Fetch current metadata
    try:
        metadata = client.fetch_sound(sound_id)
        output["api_status"] = "ok"
        output["current_exists"] = "true"
        
        # Extract current metadata
        output["current_sound_url"] = metadata.get("url", "")
        output["current_name"] = metadata.get("name", "")
        output["current_username"] = metadata.get("username", "")
        output["current_license"] = metadata.get("license", "")
        output["current_created"] = metadata.get("created", "")
        output["current_type"] = metadata.get("type", "")
        output["current_duration"] = str(metadata.get("duration", ""))
        output["current_samplerate"] = str(metadata.get("samplerate", ""))
        output["current_channels"] = str(metadata.get("channels", ""))
        output["current_num_downloads"] = str(metadata.get("num_downloads", ""))
        output["current_avg_rating"] = str(metadata.get("avg_rating", ""))
        output["current_num_ratings"] = str(metadata.get("num_ratings", ""))
        output["current_pack"] = metadata.get("pack", "")
        output["current_tags_json"] = safe_json_dump(metadata.get("tags", []))
        output["current_description"] = metadata.get("description", "")
        output["current_geotag_json"] = safe_json_dump(metadata.get("geotag", {}))
        output["current_previews_json"] = safe_json_dump(metadata.get("previews", {}))
        output["current_images_json"] = safe_json_dump(metadata.get("images", {}))
        
        # Normalize license
        output["current_license_normalized"] = normalize_license(metadata.get("license"))
        output["current_license_family"] = license_family(output["current_license_normalized"])
        
        # Research flags
        output["current_license_matches_stability_allowed_set"] = (
            "true" if output["current_license_family"] in ["cc0", "cc-by", "sampling+"] else "false"
        )
        output["duration_ge_30s"] = "true" if float(metadata.get("duration", 0)) >= 30.0 else "false"
        output["currently_accessible_for_metadata"] = "true"
        
        # Change detection
        orig_license = normalize_license(output["original_attribution_license"])
        output["changed_from_original_attribution_license"] = (
            "true" if orig_license != "unknown" and orig_license != output["current_license_normalized"] else "false"
        )
        output["changed_from_original_attribution_name"] = (
            "true" if output["original_attribution_name"] and 
            output["original_attribution_name"] != output["current_name"] else "false"
        )
        output["changed_from_original_attribution_username"] = (
            "true" if output["original_attribution_username"] and 
            output["original_attribution_username"] != output["current_username"] else "false"
        )
        output["deleted_or_unavailable_since_attribution"] = "false"
        
        # Sleep between API calls
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        
        # Fetch analysis data
        try:
            analysis = client.fetch_analysis(sound_id)
            output["analysis_available"] = "true"
            
            # Extract analysis fields
            output["genre_inferred"] = safe_get(analysis, "genre", default="")
            output["bpm"] = str(safe_get(analysis, "rhythm", "bpm", default=""))
            output["key"] = safe_get(analysis, "tonal", "key_key", default="")
            output["acoustic_electronic"] = safe_get(analysis, "music", "acoustic", default="")
            output["voice_instrumental"] = safe_get(analysis, "voice_instrumental", "value", default="")
            output["onset_count"] = str(safe_get(analysis, "rhythm", "onset_count", default=""))
            output["note_name"] = safe_get(analysis, "tonal", "note_name", default="")
            output["note_midi"] = str(safe_get(analysis, "tonal", "note_midi", default=""))
            output["pitch"] = str(safe_get(analysis, "lowlevel", "pitch", "mean", default=""))
            output["brightness"] = str(safe_get(analysis, "highlevel", "timbre", "brightness", default=""))
            output["loudness"] = str(safe_get(analysis, "lowlevel", "loudness", "mean", default=""))
            
            # Try different paths for warmth
            warmth = (safe_get(analysis, "highlevel", "timbre", "warmth", default="") or
                     safe_get(analysis, "highlevel", "timbre", "warmness", default=""))
            output["warmness_or_warmth_if_available"] = str(warmth)
            
            output["spectral_centroid_if_available"] = str(
                safe_get(analysis, "lowlevel", "spectral_centroid", "mean", default="")
            )
            
            # Store raw analysis for future use
            output["analysis_raw_json"] = safe_json_dump(analysis)
            
        except Exception as e:
            output["analysis_available"] = "false"
            output["notes"] = f"Analysis fetch failed: {str(e)}"
        
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            output["api_status"] = "not_found"
            output["current_exists"] = "false"
            output["deleted_or_unavailable_since_attribution"] = "true"
        elif e.response.status_code == 403:
            output["api_status"] = "forbidden"
        else:
            output["api_status"] = "api_error"
            output["api_error_message"] = str(e)
    except FreesoundRateLimitError:
        output["api_status"] = "rate_limited"
        raise  # Re-raise to handle at higher level
    except Exception as e:
        output["api_status"] = "parse_error"
        output["api_error_message"] = str(e)
    
    return output


def load_progress(progress_path: Path) -> dict[str, Any]:
    """Load progress from JSON file."""
    if progress_path.exists():
        with progress_path.open("r") as f:
            return json.load(f)
    return {
        "processed_ids": [],
        "successful_ids": [],
        "failed_ids": [],
        "not_found_ids": [],
        "last_processed_at_utc": None,
        "counters": {
            "ok": 0,
            "not_found": 0,
            "forbidden": 0,
            "rate_limited": 0,
            "api_error": 0,
            "parse_error": 0
        }
    }


def save_progress(progress_path: Path, progress: dict[str, Any]) -> None:
    """Save progress to JSON file."""
    progress["last_processed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with progress_path.open("w") as f:
        json.dump(progress, f, indent=2)


def write_csv_header(csv_path: Path) -> None:
    """Write CSV header if file doesn't exist."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()


def append_csv_row(csv_path: Path, row: dict[str, str]) -> None:
    """Append a row to CSV file."""
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(row)


def write_error_row(errors_path: Path, row_index: int, row: dict[str, str], error: str) -> None:
    """Write error row to errors CSV."""
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = errors_path.exists()
    
    with errors_path.open("a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write("row_index,error,raw_row_json\n")
        
        error_data = {
            "row_index": row_index,
            "error": error,
            "raw_row_json": safe_json_dump(row)
        }
        f.write(f"{row_index},{error},{safe_json_dump(row)}\n")


def load_attribution_csv(csv_source: str) -> list[dict[str, str]]:
    """Load attribution CSV from URL or file path."""
    if csv_source.startswith("http"):
        response = requests.get(csv_source, timeout=300)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        reader = csv.DictReader(lines)
        return list(reader)
    else:
        with Path(csv_source).open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def apply_enrichment_to_manifest_row(row: dict[str, Any], enriched: dict[str, str]) -> None:
    row["api_last_checked_utc"] = enriched.get("api_checked_at_utc") or None
    row["api_enrichment_status"] = enriched.get("api_status") or "unknown"
    row["api_current_name"] = enriched.get("current_name") or None
    row["api_current_license_raw"] = enriched.get("current_license") or None
    row["api_current_license_normalized"] = enriched.get("current_license_normalized") or None
    row["api_current_tags_json"] = _parse_json_field(enriched.get("current_tags_json", "[]"), default=[])
    row["api_current_description"] = enriched.get("current_description") or None
    row["api_current_duration_s"] = _to_float(enriched.get("current_duration"))
    row["api_current_samplerate"] = _to_int(enriched.get("current_samplerate"))
    row["api_current_channels"] = _to_int(enriched.get("current_channels"))
    row["api_analysis_available"] = _to_bool(enriched.get("analysis_available"))
    row["api_bpm"] = _to_float(enriched.get("bpm"))
    row["api_key"] = enriched.get("key") or None
    row["api_voice_instrumental"] = enriched.get("voice_instrumental") or None
    row["api_genre_inferred"] = enriched.get("genre_inferred") or None
    row["api_error_message"] = enriched.get("api_error_message") or None


def _parse_json_field(value: str, default: Any):
    text = (value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _to_int(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: str | None) -> bool | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def main():
    parser = argparse.ArgumentParser(description="Enrich Freesound attribution with current metadata")
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl",
                       help="Central JSONL manifest path")
    parser.add_argument("--csv-source", default=DEFAULT_ATTRIBUTION_CSV_URL,
                       help="Attribution CSV source (URL or path)")
    parser.add_argument("--output-csv", default="data/freesound_attribution_enriched.csv",
                       help="Output enriched CSV path")
    parser.add_argument("--errors-csv", default="data/freesound_attribution_enriched_errors.csv",
                       help="Errors CSV path")
    parser.add_argument("--progress-path", default="data/freesound_attribution_enriched_progress.json",
                       help="Progress JSON path")
    parser.add_argument("--summary-path", default="data/freesound_attribution_enriched_summary.json",
                       help="Summary JSON path")
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N rows")
    parser.add_argument("--sleep-seconds", type=float, default=0.5,
                       help="Sleep seconds between API calls")
    parser.add_argument("--refresh-existing", action="store_true",
                       help="Re-process already processed IDs")
    parser.add_argument("--write-raw-json", action="store_true",
                       help="Write raw JSON responses (increases file size)")
    
    args = parser.parse_args()
    
    # Load configuration and create client
    config = load_pipeline_config()
    client = FreesoundClient(config)
    manifest_rows = load_manifest_rows(Path(args.manifest))
    manifest_by_id = index_manifest_rows(manifest_rows)
    
    # Load progress
    progress_path = Path(args.progress_path)
    progress = load_progress(progress_path)
    processed_ids = set(progress["processed_ids"])
    
    # Setup output paths
    output_path = Path(args.output_csv)
    errors_path = Path(args.errors_csv)
    write_csv_header(output_path)
    
    # Load attribution CSV
    print(f"Loading attribution CSV from: {args.csv_source}")
    rows = load_attribution_csv(args.csv_source)
    total_rows = len(rows)
    print(f"Loaded {total_rows} rows")
    
    # Process tracking
    crawl_timestamp = datetime.now(timezone.utc).isoformat()
    rows_with_valid_id = 0
    processed_this_run = 0
    
    # Apply offset and limit
    start_idx = args.offset
    end_idx = start_idx + args.limit if args.limit else total_rows
    
    print(f"\nProcessing rows {start_idx} to {end_idx}")
    print("=" * 60)
    
    try:
        for idx in range(start_idx, min(end_idx, total_rows)):
            row = rows[idx]
            
            # Extract sound ID
            sound_id = extract_sound_id(row)
            if sound_id is None:
                write_error_row(errors_path, idx, row, "No valid Freesound ID found")
                continue
            
            rows_with_valid_id += 1
            
            # Skip if already processed (unless refresh requested)
            if sound_id in processed_ids and not args.refresh_existing:
                continue
            
            # Progress display
            if processed_this_run % 10 == 0:
                print(f"\rProcessed: {processed_this_run} | "
                      f"OK: {progress['counters']['ok']} | "
                      f"Not Found: {progress['counters']['not_found']} | "
                      f"Errors: {progress['counters']['api_error'] + progress['counters']['parse_error']}",
                      end="", flush=True)
            
            try:
                # Enrich row
                enriched = enrich_row(
                    row, idx, sound_id, client,
                    args.csv_source, crawl_timestamp,
                    args.sleep_seconds
                )
                
                # Write to CSV
                append_csv_row(output_path, enriched)
                manifest_row = manifest_by_id.get(str(sound_id))
                if manifest_row is not None:
                    apply_enrichment_to_manifest_row(manifest_row, enriched)
                    save_manifest_rows(manifest_rows, Path(args.manifest))
                
                # Update progress
                processed_ids.add(sound_id)
                progress["processed_ids"] = sorted(processed_ids)
                
                status = enriched["api_status"]
                progress["counters"][status] = progress["counters"].get(status, 0) + 1
                
                if status == "ok":
                    progress["successful_ids"].append(sound_id)
                elif status == "not_found":
                    progress["not_found_ids"].append(sound_id)
                else:
                    progress["failed_ids"].append(sound_id)
                
                processed_this_run += 1
                
                # Save progress periodically
                if processed_this_run % 50 == 0:
                    save_progress(progress_path, progress)
                
            except FreesoundRateLimitError:
                print(f"\n\nRate limited. Saving progress and stopping.")
                save_progress(progress_path, progress)
                break
            
            except KeyboardInterrupt:
                print(f"\n\nInterrupted. Saving progress.")
                save_progress(progress_path, progress)
                raise
            
            except Exception as e:
                print(f"\n\nError processing sound {sound_id}: {e}")
                write_error_row(errors_path, idx, row, str(e))
                progress["failed_ids"].append(sound_id)
                manifest_row = manifest_by_id.get(str(sound_id))
                if manifest_row is not None:
                    manifest_row["api_enrichment_status"] = "parse_error"
                    manifest_row["api_error_message"] = str(e)
                    save_manifest_rows(manifest_rows, Path(args.manifest))
    
    finally:
        # Save final progress
        save_progress(progress_path, progress)
        
        # Generate summary
        summary = {
            "total_input_rows": total_rows,
            "rows_with_valid_sound_id": rows_with_valid_id,
            "processed_this_run": processed_this_run,
            "total_processed": len(processed_ids),
            "api_status_counts": progress["counters"],
            "crawl_timestamp": crawl_timestamp,
            "csv_source": args.csv_source
        }
        
        # Calculate additional stats if we have data
        if progress["successful_ids"]:
            # Would need to read back the CSV to calculate these properly
            # For now, just include the counts we have
            summary["analysis_available_count"] = "See CSV for details"
            summary["license_family_counts"] = "See CSV for details"
        
        # Write summary
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n\nEnrichment complete!")
        print(f"Total processed: {len(processed_ids)}")
        print(f"This run: {processed_this_run}")
        print(f"Output: {output_path}")
        print(f"Summary: {summary_path}")
        if errors_path.exists():
            print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()
