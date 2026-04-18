from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest_utils import load_manifest_rows

FIELDNAMES = [
    "source",
    "source_id",
    "filename",
    "filepath",
    "content_fingerprint",
    "codeword",
    "pool_name",
    "family_codeword",
    "family_name",
    "genre_tier1",
    "genre_tier2",
    "soft_target_1_cw",
    "soft_target_1_prob",
    "soft_target_2_cw",
    "soft_target_2_prob",
    "soft_target_3_cw",
    "soft_target_3_prob",
    "license",
    "duration_s",
    "bpm",
    "key",
    "original_tags",
    "download_status",
]


def _parse_json(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return []
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def load_hierarchy(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pools(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    return {str(entry.get("pool_name") or entry.get("name") or entry.get("genre_tier2") or ""): entry for entry in entries}


def build_master_rows(
    genre_rows: dict[str, dict[str, str]],
    assignment_rows: dict[str, dict[str, str]],
    soft_rows: dict[str, dict[str, str]],
    fingerprint_rows: dict[str, dict[str, str]],
    pools: dict[str, dict[str, str]],
    hierarchy: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_id, assignment in assignment_rows.items():
        genre = genre_rows.get(source_id, {})
        soft = soft_rows.get(source_id, {})
        fingerprint = fingerprint_rows.get(source_id, {})
        pool_entry = pools.get(assignment["primary_pool"], {})
        codeword = pool_entry.get("codeword", "")
        family_codeword = hierarchy.get(codeword, {}).get("parent", "")
        family_name = hierarchy.get(family_codeword, {}).get("name", "")
        targets = _parse_json(soft.get("soft_targets", "[]"))
        converted = []
        for target in targets[:3]:
            target_entry = pools.get(target.get("pool_id", ""), {})
            converted.append((target_entry.get("codeword", ""), str(target.get("probability", 0))))
        while len(converted) < 3:
            converted.append(("", "0"))
        rows.append(
            {
                "source": assignment.get("source", ""),
                "source_id": source_id,
                "filename": assignment.get("filename", genre.get("filename", "")),
                "filepath": fingerprint.get("filepath", ""),
                "content_fingerprint": fingerprint.get("fingerprint_hash", ""),
                "codeword": codeword,
                "pool_name": assignment.get("primary_pool", ""),
                "family_codeword": family_codeword,
                "family_name": family_name,
                "genre_tier1": genre.get("genre_tier1", ""),
                "genre_tier2": genre.get("genre_tier2", ""),
                "soft_target_1_cw": converted[0][0],
                "soft_target_1_prob": converted[0][1],
                "soft_target_2_cw": converted[1][0],
                "soft_target_2_prob": converted[1][1],
                "soft_target_3_cw": converted[2][0],
                "soft_target_3_prob": converted[2][1],
                "license": genre.get("license", ""),
                "duration_s": genre.get("duration_s", ""),
                "bpm": genre.get("bpm", ""),
                "key": genre.get("key", ""),
                "original_tags": genre.get("tags", ""),
                "download_status": "available" if fingerprint.get("filepath") else "metadata_only",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--genre-mapped", default="data/genre_mapped.csv")
    parser.add_argument("--pool-assignments", default="data/pool_assignments.csv")
    parser.add_argument("--soft-targets", default="data/soft_targets.csv")
    parser.add_argument("--fingerprints", default="data/fingerprints.csv")
    parser.add_argument("--pools", default="registry/pools.json")
    parser.add_argument("--hierarchy", default="registry/hierarchy.json")
    parser.add_argument("--output", default="master_registry.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest_rows(Path(args.manifest))
    rows = []
    for row in manifest_rows:
        soft_targets = list(row.get("cara_soft_targets_json") or [])
        while len(soft_targets) < 3:
            soft_targets.append({"codeword": "", "probability": 0})
        rows.append(
            {
                "source": row.get("source", ""),
                "source_id": row.get("source_id", ""),
                "filename": row.get("title", "") or row.get("filename", ""),
                "filepath": row.get("local_audio_path", "") or "",
                "content_fingerprint": row.get("content_fingerprint", "") or "",
                "codeword": row.get("cara_codeword", "") or "",
                "pool_name": row.get("cara_primary_pool", "") or "",
                "family_codeword": row.get("cara_family_codeword", "") or "",
                "family_name": row.get("family_name", "") or "",
                "genre_tier1": row.get("cara_tier1", "") or row.get("genre_tier1", "") or "",
                "genre_tier2": row.get("cara_tier2", "") or row.get("genre_tier2", "") or "",
                "soft_target_1_cw": soft_targets[0].get("codeword", ""),
                "soft_target_1_prob": str(soft_targets[0].get("probability", 0)),
                "soft_target_2_cw": soft_targets[1].get("codeword", ""),
                "soft_target_2_prob": str(soft_targets[1].get("probability", 0)),
                "soft_target_3_cw": soft_targets[2].get("codeword", ""),
                "soft_target_3_prob": str(soft_targets[2].get("probability", 0)),
                "license": row.get("api_current_license_raw", "") or row.get("license_raw", "") or "",
                "duration_s": str(row.get("api_current_duration_s", "") or ""),
                "bpm": str(row.get("api_bpm", "") or ""),
                "key": row.get("api_key", "") or "",
                "original_tags": json.dumps(row.get("api_current_tags_json", []) or [], ensure_ascii=False),
                "download_status": row.get("download_status", "") or "",
            }
        )
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"total_rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
