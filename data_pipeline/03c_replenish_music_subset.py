from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.attribution_utils import display_license, infer_audio_extension, strip_audio_extension
from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows

DEFAULT_TARGET_WORKING_DOWNLOADS = 25000
DEFAULT_BASE_TARGET_SIZE = 20000
DEFAULT_SUBSET_ROLE = "music_train_candidate"
DEFAULT_SUBSET_NOTE = "balanced_music_replenish_v2"
DEFAULT_MANIFEST = "data/attribution_manifest.jsonl"
DEFAULT_PROGRESS = "data/download_progress.json"
DEFAULT_REPORT = "data/music_subset_replenish_report.json"
DEFAULT_CSV = "data/music_subset_replenish_additions.csv"
DEFAULT_SEED_LABELS_CSV = "data/attribution_seed_labels.csv"


selector_path = Path(__file__).with_name("03b_select_music_subset.py")
selector_spec = importlib.util.spec_from_file_location("music_subset_selector", selector_path)
if selector_spec is None or selector_spec.loader is None:
    raise RuntimeError(f"Unable to load selector module from {selector_path}")
selector = importlib.util.module_from_spec(selector_spec)
selector_spec.loader.exec_module(selector)


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def id_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("source_id") or "").strip()


def is_current_role_candidate(row: dict[str, Any], subset_role: str) -> bool:
    return bool(row.get("include_in_subset")) and str(row.get("subset_role") or "") == subset_role


def is_downloaded(row: dict[str, Any], completed_ids: set[str]) -> bool:
    source_id = row_id(row)
    return source_id in completed_ids or str(row.get("download_status") or "") == "downloaded"


def is_unsuccessful(row: dict[str, Any], unavailable_ids: set[str], metadata_only_ids: set[str], remove_metadata_only: bool) -> bool:
    source_id = row_id(row)
    status = str(row.get("download_status") or "")
    if source_id in unavailable_ids or status == "unavailable":
        return True
    if remove_metadata_only and (source_id in metadata_only_ids or status == "metadata_only"):
        return True
    return False


def blocked_authors_from_rows(rows: list[dict[str, Any]], blocked_ids: set[str], threshold: int) -> set[str]:
    unavailable_author_counts = Counter(
        str(row.get("author") or "").strip().lower()
        for row in rows
        if row_id(row) in blocked_ids and str(row.get("author") or "").strip()
    )
    return {
        author
        for author, count in unavailable_author_counts.items()
        if count >= max(1, int(threshold))
    }


def _bucket_rank(value: Any) -> int:
    return selector.BUCKET_RANK.get(str(value or "none"), 0)


def _seed_jsonish(value: str, default: Any) -> Any:
    text = (value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _seed_row_to_manifest_row(seed_row: dict[str, str]) -> dict[str, Any]:
    source_id = str(seed_row.get("source_id") or "").strip()
    tier1 = selector.canonicalize_tier_name(seed_row.get("genre_tier1") or "Unclassified")
    tier2 = selector.canonicalize_tier_name(seed_row.get("genre_tier2") or tier1)
    license_normalized = str(seed_row.get("license_normalized") or "").strip()
    title = str(seed_row.get("title") or "").strip()
    return {
        "source": "freesound",
        "source_id": source_id,
        "raw_id": int(source_id) if source_id.isdigit() else source_id,
        "url_sound_id": int(source_id) if source_id.isdigit() else None,
        "id_matches_url": True,
        "title": title,
        "title_stem": strip_audio_extension(title),
        "author": str(seed_row.get("author") or "").strip(),
        "license_raw": str(seed_row.get("license_raw") or "").strip(),
        "license_normalized": license_normalized,
        "license_display": display_license(license_normalized),
        "licence_class": license_normalized,
        "license_class": license_normalized,
        "url": str(seed_row.get("url") or "").strip(),
        "file_extension": infer_audio_extension(title, str(seed_row.get("url") or "")),
        "original_training_dataset": "Freesound",
        "original_training_manifest": "attribution_seed_labels",
        "originally_in_stable_audio_open_small": True,
        "api_enrichment_status": "seed_imported",
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
        "api_genre_inferred": tier1,
        "api_error_message": None,
        "cara_label_status": "seed_imported",
        "cara_label_source": "attribution_seed_labels",
        "cara_label_updated_utc": None,
        "cara_tier1": tier1,
        "cara_tier2": tier2,
        "cara_primary_pool": str(seed_row.get("primary_pool") or "").strip() or None,
        "cara_candidate_pools_json": _seed_jsonish(seed_row.get("candidate_pools", ""), []),
        "cara_soft_targets_json": [],
        "cara_family_codeword": None,
        "cara_codeword": None,
        "cara_auto_label_score": selector.numeric(seed_row.get("auto_label_score"), 0.0),
        "cara_auto_label_confidence": selector.numeric(seed_row.get("auto_label_confidence"), 0.0),
        "cara_auto_label_bucket": str(seed_row.get("auto_label_bucket") or "").strip() or None,
        "cara_matched_keywords_json": _seed_jsonish(seed_row.get("matched_keywords", ""), {}),
        "primary_genre": tier1,
        "secondary_genre": tier2,
        "style_tags": [],
        "metadata_style_summary": f"{tier1} | seed expansion candidate",
        "include_in_subset": False,
        "subset_role": None,
        "subset_note": None,
        "local_audio_path": None,
        "local_meta_path": None,
        "local_sidecar_path": None,
        "download_status": "not_downloaded",
        "content_fingerprint": None,
        "manifest_notes": "Appended from attribution_seed_labels.csv during 25k subset expansion.",
        "prefilter_status": None,
    }


def load_seed_expansion_rows(
    seed_csv_path: Path,
    existing_ids: set[str],
    blocked_ids: set[str],
    blocked_authors: set[str],
    include_tiers: list[str],
    allowed_licenses: list[str],
    min_bucket: str,
) -> list[dict[str, Any]]:
    if not seed_csv_path.exists():
        return []
    include_tier_set = {selector.canonicalize_tier_name(tier) for tier in include_tiers}
    allowed_license_set = {str(item).strip() for item in allowed_licenses if str(item).strip()}
    rows: list[dict[str, Any]] = []
    with seed_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for seed_row in reader:
            source_id = str(seed_row.get("source_id") or "").strip()
            if not source_id or source_id in existing_ids or source_id in blocked_ids:
                continue
            author = str(seed_row.get("author") or "").strip().lower()
            if author and author in blocked_authors:
                continue
            tier = selector.canonicalize_tier_name(seed_row.get("genre_tier1") or "Unclassified")
            if include_tier_set and tier not in include_tier_set:
                continue
            license_value = str(seed_row.get("license_normalized") or "").strip()
            if allowed_license_set and license_value not in allowed_license_set:
                continue
            if _bucket_rank(seed_row.get("auto_label_bucket")) < _bucket_rank(min_bucket):
                continue
            if not str(seed_row.get("primary_pool") or "").strip():
                continue
            rows.append(_seed_row_to_manifest_row(seed_row))
    return rows


def select_replacements(
    rows: list[dict[str, Any]],
    existing_candidate_ids: set[str],
    blocked_ids: set[str],
    replacements_needed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    include_tiers = selector.parse_csv_list(args.include_tiers)
    exclude_tiers = selector.parse_csv_list(args.exclude_tiers)
    expansion_include_tiers = selector.parse_csv_list(args.expansion_include_tiers)
    allowed_licenses = selector.parse_csv_list(args.allowed_licenses)
    base_target_size = max(0, min(int(args.base_target_size), int(args.target_working_downloads)))
    selectable_rows = [
        row for row in rows
        if row_id(row) not in existing_candidate_ids
        and row_id(row) not in blocked_ids
    ]
    unavailable_author_counts = Counter(
        str(row.get("author") or "").strip().lower()
        for row in rows
        if row_id(row) in blocked_ids and str(row.get("author") or "").strip()
    )
    blocked_authors = blocked_authors_from_rows(rows, blocked_ids, args.block_authors_with_unavailable_count)
    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "requested_replacements": replacements_needed,
        "base_target_size": base_target_size,
        "current_candidate_count": len(existing_candidate_ids),
    }

    general_needed = 0
    if len(existing_candidate_ids) < base_target_size:
        general_needed = min(replacements_needed, base_target_size - len(existing_candidate_ids))
    expansion_needed = max(replacements_needed - general_needed, 0)

    if general_needed > 0:
        general_selected, general_summary = selector.select_subset(
            rows=selectable_rows,
            target_size=general_needed,
            include_tiers=include_tiers,
            exclude_tiers=exclude_tiers,
            min_bucket=args.min_bucket,
            allowed_licenses=allowed_licenses,
            tier_weights=selector.DEFAULT_TIER_WEIGHTS,
            tier_cap_fractions=selector.DEFAULT_TIER_CAP_FRACTIONS,
            pool_cap_fraction=selector.DEFAULT_POOL_CAP_FRACTION,
            license_cap_fraction=selector.DEFAULT_LICENSE_CAP_FRACTION,
        )
        selected.extend(general_selected)
        summary["general_selection_summary"] = general_summary

    selected_ids = {row_id(row) for row in selected if row_id(row)}
    expansion_pool = [
        row for row in selectable_rows
        if row_id(row) not in selected_ids
        and str(row.get("author") or "").strip().lower() not in blocked_authors
    ]
    seed_rows = load_seed_expansion_rows(
        seed_csv_path=Path(args.seed_labels_csv),
        existing_ids={row_id(row) for row in rows} | existing_candidate_ids | blocked_ids | selected_ids,
        blocked_ids=blocked_ids,
        blocked_authors=blocked_authors,
        include_tiers=expansion_include_tiers,
        allowed_licenses=allowed_licenses,
        min_bucket=args.min_bucket,
    )
    expansion_pool.extend(seed_rows)

    if expansion_needed > 0:
        expansion_selected, expansion_summary = selector.select_subset(
            rows=expansion_pool,
            target_size=expansion_needed,
            include_tiers=expansion_include_tiers,
            exclude_tiers=exclude_tiers,
            min_bucket=args.min_bucket,
            allowed_licenses=allowed_licenses,
            tier_weights=selector.DEFAULT_EXPANSION_TIER_WEIGHTS,
            tier_cap_fractions={},
            pool_cap_fraction=1.0,
            license_cap_fraction=1.0,
        )
        selected.extend(expansion_selected)
        summary["expansion_selection_summary"] = expansion_summary
        summary["seed_expansion_candidate_count"] = len(seed_rows)

    summary["selected_count"] = len(selected)
    summary["blocked_seed_authors"] = dict(sorted(unavailable_author_counts.items()))
    summary["blocked_seed_author_threshold"] = int(args.block_authors_with_unavailable_count)
    summary["selected_tier_counts"] = dict(
        Counter(
            selector.canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
            for row in selected
        )
    )
    return selected, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_id", "title", "author", "license_normalized", "cara_tier1", "cara_tier2", "cara_primary_pool", "cara_auto_label_bucket", "cara_auto_label_score", "cara_auto_label_confidence", "prefilter_status", "api_current_duration_s", "selection_score", "url"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload = {column: row.get(column, "") for column in columns}
            payload["selection_score"] = row.get("_selection_score", selector.row_score(row))
            writer.writerow(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--progress-path", default=DEFAULT_PROGRESS)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV)
    parser.add_argument("--target-working-downloads", type=int, default=DEFAULT_TARGET_WORKING_DOWNLOADS)
    parser.add_argument("--base-target-size", type=int, default=DEFAULT_BASE_TARGET_SIZE)
    parser.add_argument("--subset-role", default=DEFAULT_SUBSET_ROLE)
    parser.add_argument("--subset-note", default=DEFAULT_SUBSET_NOTE)
    parser.add_argument("--min-bucket", default="medium", choices=["none", "low", "medium", "high"])
    parser.add_argument("--include-tiers", default=",".join(selector.DEFAULT_INCLUDE_TIERS))
    parser.add_argument("--expansion-include-tiers", default=",".join(selector.DEFAULT_EXPANSION_INCLUDE_TIERS))
    parser.add_argument("--exclude-tiers", default=",".join(selector.DEFAULT_EXCLUDE_TIERS))
    parser.add_argument("--allowed-licenses", default="cc0,cc-by,sampling+")
    parser.add_argument("--seed-labels-csv", default=DEFAULT_SEED_LABELS_CSV)
    parser.add_argument("--block-authors-with-unavailable-count", type=int, default=3)
    parser.add_argument("--extra-buffer", type=int, default=0)
    parser.add_argument("--remove-metadata-only", action="store_true")
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    progress = load_progress(Path(args.progress_path))
    rows = load_manifest_rows(manifest_path)

    completed_ids = id_set(progress.get("completed_ids"))
    unavailable_ids = id_set(progress.get("unavailable_ids"))
    metadata_only_ids = id_set(progress.get("metadata_only_ids"))

    current_candidate_rows = [row for row in rows if is_current_role_candidate(row, args.subset_role)]
    current_candidate_ids = {row_id(row) for row in current_candidate_rows if row_id(row)}
    downloaded_candidate_ids = {row_id(row) for row in current_candidate_rows if is_downloaded(row, completed_ids)}
    unsuccessful_candidate_ids = {
        row_id(row) for row in current_candidate_rows
        if is_unsuccessful(row, unavailable_ids, metadata_only_ids, args.remove_metadata_only)
    }
    blocked_authors = blocked_authors_from_rows(rows, unsuccessful_candidate_ids | unavailable_ids, args.block_authors_with_unavailable_count)
    blocked_pending_candidate_ids = {
        row_id(row) for row in current_candidate_rows
        if row_id(row) not in downloaded_candidate_ids
        and row_id(row) not in unsuccessful_candidate_ids
        and str(row.get("author") or "").strip().lower() in blocked_authors
    }
    unsuccessful_candidate_ids |= blocked_pending_candidate_ids
    pending_candidate_ids = current_candidate_ids - downloaded_candidate_ids - unsuccessful_candidate_ids
    replacements_needed = max(args.target_working_downloads - len(downloaded_candidate_ids) - len(pending_candidate_ids), 0) + max(args.extra_buffer, 0)

    replacements, selection_summary = select_replacements(
        rows=rows,
        existing_candidate_ids=current_candidate_ids,
        blocked_ids=unsuccessful_candidate_ids | unavailable_ids | metadata_only_ids,
        replacements_needed=replacements_needed,
        args=args,
    )
    replacement_ids = {row_id(row) for row in replacements if row_id(row)}

    if args.update_manifest:
        manifest_by_id = {row_id(row): row for row in rows if row_id(row)}
        appended_from_seed = 0
        for row in replacements:
            source_id = row_id(row)
            if source_id and source_id not in manifest_by_id:
                rows.append(dict(row))
                manifest_by_id[source_id] = rows[-1]
                appended_from_seed += 1
        for row in rows:
            source_id = row_id(row)
            if source_id in unsuccessful_candidate_ids:
                row["include_in_subset"] = False
                if str(row.get("subset_role") or "") == args.subset_role:
                    row["subset_role"] = ""
                row["subset_note"] = "unsuccessful_download_removed_from_subset"
            elif source_id in replacement_ids:
                row["include_in_subset"] = True
                row["subset_role"] = args.subset_role
                row["subset_note"] = args.subset_note
        save_manifest_rows(rows, Path(args.output_manifest) if args.output_manifest else manifest_path)
        selection_summary["appended_from_seed"] = appended_from_seed

    write_csv(Path(args.csv_output), replacements)
    report = {
        "dry_run": not args.update_manifest,
        "target_working_downloads": args.target_working_downloads,
        "subset_role": args.subset_role,
        "current_candidates": len(current_candidate_ids),
        "downloaded_candidates": len(downloaded_candidate_ids),
        "pending_candidates": len(pending_candidate_ids),
        "unsuccessful_candidates_removed": len(unsuccessful_candidate_ids),
        "blocked_pending_candidates_removed": len(blocked_pending_candidate_ids),
        "blocked_pending_authors": sorted(blocked_authors),
        "replacements_needed": replacements_needed,
        "replacements_selected": len(replacements),
        "shortfall_after_replenish": max(replacements_needed - len(replacements), 0),
        "metadata_only_removed": args.remove_metadata_only,
        "updated_manifest": str(Path(args.output_manifest) if args.output_manifest else manifest_path) if args.update_manifest else "",
        "csv_output": args.csv_output,
        "selection_summary": selection_summary,
    }
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
