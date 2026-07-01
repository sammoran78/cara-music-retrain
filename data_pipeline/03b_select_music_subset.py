from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.genre_normalization import GENRE_LABEL_ALIASES, normalize_genre_label
from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows

DEFAULT_TARGET_SIZE = 25000
DEFAULT_BASE_TARGET_SIZE = 20000
DEFAULT_INCLUDE_TIERS = [
    "Electronic",
    "Percussion/Drums",
    "Acoustic/Folk",
    "Classical/Orchestral",
    "Jazz/Blues",
    "Ambient/Drone",
    "Experimental/Noise",
    "Voice/Vocal",
]
DEFAULT_EXPANSION_INCLUDE_TIERS = [
    "Acoustic/Folk",
    "Jazz/Blues",
    "World/Traditional",
    "Hip-Hop/Beats",
]
DEFAULT_EXCLUDE_TIERS = [
    "Sound Effects",
    "Field Recording",
    "Unclassified",
]
DEFAULT_TIER_WEIGHTS = {
    "Electronic": 0.31,
    "Percussion/Drums": 0.25,
    "Acoustic/Folk": 0.18,
    "Classical/Orchestral": 0.11,
    "Jazz/Blues": 0.08,
    "Ambient/Drone": 0.04,
    "Experimental/Noise": 0.02,
    "Voice/Vocal": 0.01,
}
DEFAULT_TIER_CAP_FRACTIONS = {
    "Ambient/Drone": 0.07,
    "Experimental/Noise": 0.04,
    "Voice/Vocal": 0.025,
}
DEFAULT_EXPANSION_TIER_WEIGHTS = {
    "Acoustic/Folk": 0.45,
    "Jazz/Blues": 0.30,
    "Hip-Hop/Beats": 0.15,
    "World/Traditional": 0.10,
}
DEFAULT_LICENSE_CAP_FRACTION = 0.45
DEFAULT_POOL_CAP_FRACTION = 0.2
OUTPUT_COLUMNS = [
    "source_id",
    "title",
    "author",
    "license_normalized",
    "cara_tier1",
    "cara_tier2",
    "cara_primary_pool",
    "cara_auto_label_bucket",
    "cara_auto_label_score",
    "cara_auto_label_confidence",
    "prefilter_status",
    "api_current_duration_s",
    "score",
    "selection_reason",
    "url",
]


BUCKET_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}

TIER_NAME_ALIASES = GENRE_LABEL_ALIASES


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def canonicalize_tier_name(value: Any) -> str:
    return normalize_genre_label(value, default="Unclassified", preserve_unknown=True)


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    return load_manifest_rows(manifest_path)


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def row_score(row: dict[str, Any]) -> float:
    bucket = str(row.get("cara_auto_label_bucket") or "none")
    confidence = numeric(row.get("cara_auto_label_confidence"), 0.0)
    auto_score = numeric(row.get("cara_auto_label_score"), 0.0)
    duration = numeric(row.get("api_current_duration_s"), 0.0)
    prefilter = str(row.get("prefilter_status") or "")
    tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
    keyword_matches = row.get("cara_matched_keywords_json") or {}
    keyword_bonus = 0.0
    if isinstance(keyword_matches, dict):
        keyword_bonus = min(sum(len(values) for values in keyword_matches.values()), 8)
    duration_bonus = 0.0
    if duration >= 30:
        duration_bonus += 2.0
    elif duration >= 10:
        duration_bonus += 1.0
    elif duration > 0:
        duration_bonus -= 1.5
    base = BUCKET_RANK.get(bucket, 0) * 20.0
    if prefilter == "confirmed":
        base += 12.0
    elif prefilter == "rejected":
        base -= 25.0
    if row.get("include_in_subset"):
        base += 3.0
    tier_bias = {
        "Electronic": 5.0,
        "Percussion/Drums": 5.0,
        "Acoustic/Folk": 4.0,
        "Classical/Orchestral": 4.0,
        "Jazz/Blues": 3.0,
        "Ambient/Drone": -4.0,
        "Experimental/Noise": -7.0,
        "Voice/Vocal": -3.0,
    }.get(tier, 0.0)
    return round(base + confidence * 40.0 + auto_score * 3.0 + keyword_bonus + duration_bonus + tier_bias, 4)


def is_candidate(
    row: dict[str, Any],
    include_tiers: set[str],
    exclude_tiers: set[str],
    min_bucket: str,
    allowed_licenses: set[str],
) -> bool:
    if row.get("source") != "freesound":
        return False
    tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
    if tier in exclude_tiers:
        return False
    if include_tiers and tier not in include_tiers:
        return False
    if BUCKET_RANK.get(str(row.get("cara_auto_label_bucket") or "none"), 0) < BUCKET_RANK[min_bucket]:
        return False
    if not row.get("cara_primary_pool"):
        return False
    license_value = str(row.get("license_normalized") or "")
    if allowed_licenses and license_value not in allowed_licenses:
        return False
    if str(row.get("prefilter_status") or "") == "rejected":
        return False
    return True


def tier_target_map(target_size: int, tiers: list[str], tier_weights: dict[str, float]) -> dict[str, int]:
    tiers = [canonicalize_tier_name(tier) for tier in tiers]
    total_weight = sum(max(tier_weights.get(tier, 0.0), 0.0) for tier in tiers)
    if total_weight <= 0:
        total_weight = float(len(tiers) or 1)
    provisional: dict[str, int] = {}
    running_total = 0
    for tier in tiers:
        weight = max(tier_weights.get(tier, 0.0), 0.0)
        amount = int(math.floor(target_size * (weight / total_weight))) if total_weight else 0
        provisional[tier] = amount
        running_total += amount
    remainder = max(target_size - running_total, 0)
    ranked = sorted(tiers, key=lambda tier: tier_weights.get(tier, 0.0), reverse=True)
    for tier in ranked:
        if remainder <= 0:
            break
        provisional[tier] += 1
        remainder -= 1
    return provisional


def select_subset(
    rows: list[dict[str, Any]],
    target_size: int,
    include_tiers: list[str],
    exclude_tiers: list[str],
    min_bucket: str,
    allowed_licenses: list[str],
    tier_weights: dict[str, float],
    tier_cap_fractions: dict[str, float],
    pool_cap_fraction: float,
    license_cap_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    include_tiers = [canonicalize_tier_name(tier) for tier in include_tiers]
    exclude_tiers = [canonicalize_tier_name(tier) for tier in exclude_tiers]
    tier_weights = {canonicalize_tier_name(tier): weight for tier, weight in tier_weights.items()}
    tier_cap_fractions = {canonicalize_tier_name(tier): fraction for tier, fraction in tier_cap_fractions.items()}
    include_tier_set = set(include_tiers)
    exclude_tier_set = set(exclude_tiers)
    allowed_license_set = set(allowed_licenses)
    candidates = [
        row for row in rows
        if is_candidate(row, include_tier_set, exclude_tier_set, min_bucket, allowed_license_set)
    ]
    for row in candidates:
        row["_selection_score"] = row_score(row)
    candidates.sort(
        key=lambda row: (
            -float(row["_selection_score"]),
            -numeric(row.get("cara_auto_label_confidence"), 0.0),
            -numeric(row.get("cara_auto_label_score"), 0.0),
            str(row.get("source_id") or ""),
        )
    )

    tier_targets = tier_target_map(target_size, include_tiers, tier_weights)
    tier_caps = {
        tier: max(tier_targets.get(tier, 0), int(target_size * tier_cap_fractions.get(tier, 1.0)))
        for tier in include_tiers
    }
    pool_cap = max(1, int(target_size * pool_cap_fraction))
    license_cap = max(1, int(target_size * license_cap_fraction))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    tier_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    deferred_tier_target: list[dict[str, Any]] = []
    deferred_soft_caps: list[dict[str, Any]] = []

    def _select_row(row: dict[str, Any]) -> None:
        source_id = str(row.get("source_id") or "")
        tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
        pool = str(row.get("cara_primary_pool") or "")
        license_value = str(row.get("license_normalized") or "")
        selected.append(row)
        selected_ids.add(source_id)
        tier_counts[tier] += 1
        if pool:
            pool_counts[pool] += 1
        if license_value:
            license_counts[license_value] += 1

    for row in candidates:
        source_id = str(row.get("source_id") or "")
        tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
        pool = str(row.get("cara_primary_pool") or "")
        license_value = str(row.get("license_normalized") or "")
        if source_id in selected_ids:
            rejected_by_reason["duplicate_source_id"] += 1
            continue
        if tier_counts[tier] >= tier_targets.get(tier, 0):
            deferred_tier_target.append(row)
            continue
        if tier_counts[tier] >= tier_caps.get(tier, target_size):
            rejected_by_reason[f"tier_cap:{tier}"] += 1
            continue
        if pool and pool_counts[pool] >= pool_cap:
            deferred_soft_caps.append(row)
            continue
        if license_value and license_counts[license_value] >= license_cap:
            deferred_soft_caps.append(row)
            continue
        _select_row(row)
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        for row in deferred_tier_target:
            if len(selected) >= target_size:
                break
            source_id = str(row.get("source_id") or "")
            if source_id in selected_ids:
                continue
            tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
            pool = str(row.get("cara_primary_pool") or "")
            license_value = str(row.get("license_normalized") or "")
            if tier_counts[tier] >= tier_caps.get(tier, target_size):
                continue
            if pool and pool_counts[pool] >= pool_cap:
                continue
            if license_value and license_counts[license_value] >= license_cap:
                continue
            _select_row(row)

    if len(selected) < target_size:
        for row in deferred_soft_caps + deferred_tier_target + candidates:
            if len(selected) >= target_size:
                break
            source_id = str(row.get("source_id") or "")
            if source_id in selected_ids:
                continue
            tier = canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
            if tier_counts[tier] >= tier_caps.get(tier, target_size):
                continue
            _select_row(row)

    summary = {
        "target_size": target_size,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "min_bucket": min_bucket,
        "tier_targets": tier_targets,
        "selected_tier_counts": dict(tier_counts),
        "selected_license_counts": dict(sorted(license_counts.items())),
        "selected_pool_counts_top20": dict(pool_counts.most_common(20)),
        "rejected_by_reason_top20": dict(rejected_by_reason.most_common(20)),
        "selected_prefilter_counts": dict(Counter(str(row.get("prefilter_status") or "") for row in selected)),
    }
    return selected, summary


def select_subset_with_expansion(
    rows: list[dict[str, Any]],
    target_size: int,
    include_tiers: list[str],
    exclude_tiers: list[str],
    min_bucket: str,
    allowed_licenses: list[str],
    tier_weights: dict[str, float],
    tier_cap_fractions: dict[str, float],
    pool_cap_fraction: float,
    license_cap_fraction: float,
    base_target_size: int,
    expansion_include_tiers: list[str],
    expansion_tier_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_base_target = max(0, min(target_size, int(base_target_size)))
    if target_size <= normalized_base_target:
        return select_subset(
            rows=rows,
            target_size=target_size,
            include_tiers=include_tiers,
            exclude_tiers=exclude_tiers,
            min_bucket=min_bucket,
            allowed_licenses=allowed_licenses,
            tier_weights=tier_weights,
            tier_cap_fractions=tier_cap_fractions,
            pool_cap_fraction=pool_cap_fraction,
            license_cap_fraction=license_cap_fraction,
        )

    base_selected, base_summary = select_subset(
        rows=rows,
        target_size=normalized_base_target,
        include_tiers=include_tiers,
        exclude_tiers=exclude_tiers,
        min_bucket=min_bucket,
        allowed_licenses=allowed_licenses,
        tier_weights=tier_weights,
        tier_cap_fractions=tier_cap_fractions,
        pool_cap_fraction=pool_cap_fraction,
        license_cap_fraction=license_cap_fraction,
    )
    base_ids = {str(row.get("source_id") or "") for row in base_selected if str(row.get("source_id") or "")}
    expansion_target = max(int(target_size) - len(base_selected), 0)
    remaining_rows = [row for row in rows if str(row.get("source_id") or "") not in base_ids]
    expansion_selected, expansion_summary = select_subset(
        rows=remaining_rows,
        target_size=expansion_target,
        include_tiers=expansion_include_tiers,
        exclude_tiers=exclude_tiers,
        min_bucket=min_bucket,
        allowed_licenses=allowed_licenses,
        tier_weights=expansion_tier_weights,
        tier_cap_fractions={},
        pool_cap_fraction=1.0,
        license_cap_fraction=1.0,
    )

    combined = base_selected + expansion_selected
    combined_tiers = Counter(
        canonicalize_tier_name(row.get("cara_tier1") or row.get("genre_tier1") or "Unclassified")
        for row in combined
    )
    combined_licenses = Counter(str(row.get("license_normalized") or "") for row in combined)
    summary = {
        "target_size": target_size,
        "base_target_size": normalized_base_target,
        "expansion_target_size": expansion_target,
        "candidate_count": base_summary.get("candidate_count", 0),
        "selected_count": len(combined),
        "selected_tier_counts": dict(combined_tiers),
        "selected_license_counts": dict(sorted(combined_licenses.items())),
        "base_selection_summary": base_summary,
        "expansion_selection_summary": expansion_summary,
        "expansion_include_tiers": [canonicalize_tier_name(tier) for tier in expansion_include_tiers],
    }
    return combined, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            payload = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            writer.writerow(payload)


def update_manifest(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    subset_role: str,
    subset_note: str,
) -> dict[str, int]:
    selected_updates = 0
    deselected_updates = 0
    for row in rows:
        if row.get("source") != "freesound":
            continue
        source_id = str(row.get("source_id") or "")
        if source_id in selected_ids:
            row["include_in_subset"] = True
            row["subset_role"] = subset_role
            row["subset_note"] = subset_note
            selected_updates += 1
        elif row.get("include_in_subset"):
            row["include_in_subset"] = False
            if row.get("subset_role") == subset_role:
                row["subset_role"] = ""
            if row.get("subset_note") == subset_note:
                row["subset_note"] = ""
            deselected_updates += 1
    return {"selected_updates": selected_updates, "deselected_updates": deselected_updates}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--output-csv", default="data/music_subset_candidates.csv")
    parser.add_argument("--summary-output", default="data/music_subset_selection_summary.json")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--min-bucket", default="medium", choices=["none", "low", "medium", "high"])
    parser.add_argument("--include-tiers", default=",".join(DEFAULT_INCLUDE_TIERS))
    parser.add_argument("--exclude-tiers", default=",".join(DEFAULT_EXCLUDE_TIERS))
    parser.add_argument("--allowed-licenses", default="cc0,cc-by,sampling+")
    parser.add_argument("--subset-role", default="music_train_candidate")
    parser.add_argument("--subset-note", default="balanced_music_selector_v1")
    parser.add_argument("--base-target-size", type=int, default=DEFAULT_BASE_TARGET_SIZE)
    parser.add_argument("--expansion-include-tiers", default=",".join(DEFAULT_EXPANSION_INCLUDE_TIERS))
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = load_rows(manifest_path)
    include_tiers = parse_csv_list(args.include_tiers)
    exclude_tiers = parse_csv_list(args.exclude_tiers)
    allowed_licenses = parse_csv_list(args.allowed_licenses)
    expansion_include_tiers = parse_csv_list(args.expansion_include_tiers)
    selected, summary = select_subset_with_expansion(
        rows=rows,
        target_size=args.target_size,
        include_tiers=include_tiers,
        exclude_tiers=exclude_tiers,
        min_bucket=args.min_bucket,
        allowed_licenses=allowed_licenses,
        tier_weights=DEFAULT_TIER_WEIGHTS,
        tier_cap_fractions=DEFAULT_TIER_CAP_FRACTIONS,
        pool_cap_fraction=DEFAULT_POOL_CAP_FRACTION,
        license_cap_fraction=DEFAULT_LICENSE_CAP_FRACTION,
        base_target_size=args.base_target_size,
        expansion_include_tiers=expansion_include_tiers,
        expansion_tier_weights=DEFAULT_EXPANSION_TIER_WEIGHTS,
    )
    for row in selected:
        row["score"] = row.get("_selection_score")
        row["selection_reason"] = f"tier={row.get('cara_tier1','')} bucket={row.get('cara_auto_label_bucket','')} pool={row.get('cara_primary_pool','')}"
    write_csv(Path(args.output_csv), selected)
    if args.update_manifest:
        selected_ids = {str(row.get("source_id") or "") for row in selected}
        update_summary = update_manifest(rows, selected_ids, args.subset_role, args.subset_note)
        output_manifest = Path(args.output_manifest) if args.output_manifest else manifest_path
        save_manifest_rows(rows, output_manifest)
        summary.update(update_summary)
        summary["updated_manifest"] = str(output_manifest)
    summary["output_csv"] = args.output_csv
    summary["include_tiers"] = [canonicalize_tier_name(tier) for tier in include_tiers]
    summary["exclude_tiers"] = [canonicalize_tier_name(tier) for tier in exclude_tiers]
    summary["allowed_licenses"] = allowed_licenses
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
