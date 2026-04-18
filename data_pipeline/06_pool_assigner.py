from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows

OUTPUT_COLUMNS = [
    "source",
    "source_id",
    "filename",
    "genre_tier1",
    "genre_tier2",
    "primary_pool",
    "all_pools",
    "pool_merge_applied",
]


def _parse_candidate_pools(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def _build_pool_name(source: str, license: str, genre: str) -> str:
    """Build pool name following Source-License-Genre convention."""
    # Normalize license names
    license_map = {
        "cc0": "CC0",
        "cc-by": "CC-BY",
        "cc-by-sa": "CC-BY-SA",
        "cc-by-nc": "CC-BY-NC",
        "cc-sampling+": "CC-Sampling+",
        "ccby": "CC-BY",
        "ccbysa": "CC-BY-SA",
        "ccbync": "CC-BY-NC",
        "sampling+": "CC-Sampling+",
    }
    
    # Normalize source names
    source_map = {
        "freesound": "Freesound",
        "fma": "FMA",
        "free music archive": "FMA",
    }
    
    normalized_source = source_map.get(source.lower(), source)
    normalized_license = license_map.get(license.lower().replace(" ", "").replace("_", "-"), license)
    
    return f"{normalized_source}-{normalized_license}-{genre}"


def _nearest_pool(pool_name: str, tier1: str, pool_counts: Counter[str], tier1_to_pools: dict[str, list[str]]) -> str:
    candidates = [candidate for candidate in tier1_to_pools.get(tier1, []) if candidate != pool_name]
    if candidates:
        return max(candidates, key=lambda name: pool_counts[name])
    if pool_counts:
        return max(pool_counts, key=pool_counts.get)
    return pool_name


def assign_pools(rows: list[dict[str, str]], minimum_pool_size: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    # Build pool names using source-license-genre convention
    for row in rows:
        source = row.get("source", "Unknown")
        license = row.get("license", "Unknown")
        genre = row.get("genre_tier2", row.get("primary_pool", "Unclassified"))
        row["primary_pool"] = _build_pool_name(source, license, genre)
    
    pool_counts = Counter(row["primary_pool"] for row in rows)
    tier1_to_pools: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tier1_to_pools[row["genre_tier1"]].append(row["primary_pool"])

    merge_map: dict[str, str] = {}
    for pool_name, count in pool_counts.items():
        if count < minimum_pool_size:
            sample_row = next(row for row in rows if row["primary_pool"] == pool_name)
            merge_map[pool_name] = _nearest_pool(pool_name, sample_row["genre_tier1"], pool_counts, tier1_to_pools)

    assigned_rows: list[dict[str, str]] = []
    for row in rows:
        original_pool = row["primary_pool"]
        final_pool = merge_map.get(original_pool, original_pool)
        candidate_pools = _parse_candidate_pools(row.get("candidate_pools", ""))
        all_pools = [merge_map.get(pool, pool) for pool in candidate_pools] or [final_pool]
        unique_pools = list(dict.fromkeys([final_pool] + all_pools))
        assigned_rows.append(
            {
                "source": row["source"],
                "source_id": row["source_id"],
                "filename": row["filename"],
                "genre_tier1": row["genre_tier1"],
                "genre_tier2": row["genre_tier2"],
                "primary_pool": final_pool,
                "all_pools": json.dumps(unique_pools),
                "pool_merge_applied": "true" if final_pool != original_pool else "false",
            }
        )
    return assigned_rows, merge_map


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, str]], merge_map: dict[str, str]) -> None:
    pool_sizes = Counter(row["primary_pool"] for row in rows)
    report = {
        "pool_sizes": dict(pool_sizes),
        "merge_map": merge_map,
        "num_merges": len(merge_map),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/genre_mapped.csv")
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--output", default="data/pool_assignments.csv")
    parser.add_argument("--report-output", default="data/pool_assignment_report.json")
    parser.add_argument("--minimum-pool-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    assigned_rows, merge_map = assign_pools(rows, minimum_pool_size=args.minimum_pool_size)
    write_rows(Path(args.output), assigned_rows)
    write_report(Path(args.report_output), assigned_rows, merge_map)
    manifest_rows = load_manifest_rows(Path(args.manifest))
    assigned_lookup = {row["source_id"]: row for row in assigned_rows}
    for row in manifest_rows:
        assigned = assigned_lookup.get(str(row.get("source_id", "")))
        if not assigned:
            continue
        row["cara_primary_pool"] = assigned["primary_pool"]
        row["all_pools"] = json.loads(assigned["all_pools"])
        row["pool_merge_applied"] = assigned["pool_merge_applied"] == "true"
    save_manifest_rows(manifest_rows, Path(args.manifest))
    print(json.dumps({"total_rows": len(assigned_rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
