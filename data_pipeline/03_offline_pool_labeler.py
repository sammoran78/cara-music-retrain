from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.attribution_utils import display_license, normalise_text, tokenize_text
from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows


OUTPUT_COLUMNS = [
    "source",
    "source_id",
    "raw_id",
    "title",
    "title_stem",
    "author",
    "license_raw",
    "license_normalized",
    "license_display",
    "url",
    "genre_tier1",
    "genre_tier2",
    "primary_pool",
    "candidate_pools",
    "auto_label_score",
    "auto_label_confidence",
    "auto_label_bucket",
    "matched_keywords",
]


def load_ontology(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _keyword_score(search_blob: str, token_set: set[str], keyword: str, weight: int) -> tuple[int, str | None]:
    normalised = normalise_text(keyword)
    if not normalised:
        return 0, None
    if " " in normalised:
        if normalised in search_blob:
            return weight + 1, keyword
        return 0, None
    if normalised in token_set:
        return weight, keyword
    if normalised in search_blob:
        return 1, keyword
    return 0, None


def _slug(value: str) -> str:
    return "-".join(normalise_text(value).split())


def score_row(row: dict[str, str], ontology: dict[str, Any]) -> dict[str, Any]:
    search_blob = normalise_text(" ".join([row.get("title_stem", ""), row.get("title", ""), row.get("author", ""), row.get("url", "")]))
    token_set = set(tokenize_text(row.get("title_stem", ""), row.get("title", ""), row.get("author", ""), row.get("url", "")))
    rule_scores: list[dict[str, Any]] = []

    for rule in ontology.get("genres", []):
        score = 0
        matched: list[str] = []
        penalties: list[str] = []

        for keyword in rule.get("keywords", []):
            delta, match = _keyword_score(search_blob, token_set, keyword, weight=2)
            score += delta
            if match:
                matched.append(match)

        for keyword in rule.get("boost_keywords", []):
            delta, match = _keyword_score(search_blob, token_set, keyword, weight=4)
            score += delta
            if match:
                matched.append(match)

        for keyword in rule.get("anti_keywords", []):
            delta, match = _keyword_score(search_blob, token_set, keyword, weight=2)
            if delta > 0:
                score -= delta
                penalties.append(keyword)

        rule_scores.append(
            {
                "tier1": rule["tier1"],
                "score": score,
                "matched": sorted(dict.fromkeys(matched)),
                "penalties": penalties,
            }
        )

    ranked = sorted(rule_scores, key=lambda item: (-item["score"], item["tier1"]))
    positive = [item for item in ranked if item["score"] > 0]
    if not positive:
        return {
            "genre_tier1": "Unclassified",
            "genre_tier2": "Unclassified",
            "primary_pool": "",
            "candidate_pools": [],
            "auto_label_score": 0,
            "auto_label_confidence": 0.0,
            "auto_label_bucket": "none",
            "matched_keywords": {},
        }

    top1 = positive[0]
    top2_score = positive[1]["score"] if len(positive) > 1 else 0
    total_positive = sum(item["score"] for item in positive)
    base_confidence = top1["score"] / max(total_positive, 1)
    margin = (top1["score"] - top2_score) / max(top1["score"], 1)
    confidence = round(min(1.0, 0.65 * base_confidence + 0.35 * max(margin, 0.0)), 4)

    thresholds = ontology.get("confidence_thresholds", {})
    if confidence >= float(thresholds.get("high", 0.72)) and top1["score"] >= 4:
        bucket = "high"
    elif confidence >= float(thresholds.get("medium", 0.45)) and top1["score"] >= 2:
        bucket = "medium"
    else:
        bucket = "low"

    license_display = row.get("license_display") or display_license(row.get("license_normalized", "unknown"))
    candidate_pools = [
        f"{ontology.get('source', 'Freesound')}-{license_display}-{item['tier1']}"
        for item in positive[:3]
        if row.get("license_normalized") in set(ontology.get("allowed_licenses", []))
    ]
    primary_pool = candidate_pools[0] if candidate_pools else ""

    matched_keywords = {item["tier1"]: item["matched"] for item in positive[:3] if item["matched"]}
    return {
        "genre_tier1": top1["tier1"],
        "genre_tier2": top1["tier1"],
        "primary_pool": primary_pool,
        "candidate_pools": candidate_pools,
        "auto_label_score": top1["score"],
        "auto_label_confidence": confidence,
        "auto_label_bucket": bucket,
        "matched_keywords": matched_keywords,
    }


def label_rows(rows: list[dict[str, str]], ontology: dict[str, Any]) -> list[dict[str, str]]:
    labeled_rows: list[dict[str, str]] = []
    for row in rows:
        scored = score_row(row, ontology)
        labeled_rows.append(
            {
                "source": row.get("source", "freesound"),
                "source_id": row.get("source_id", ""),
                "raw_id": row.get("raw_id", ""),
                "title": row.get("title", ""),
                "title_stem": row.get("title_stem", ""),
                "author": row.get("author", ""),
                "license_raw": row.get("license_raw", ""),
                "license_normalized": row.get("license_normalized", ""),
                "license_display": row.get("license_display", ""),
                "url": row.get("url", ""),
                "genre_tier1": scored["genre_tier1"],
                "genre_tier2": scored["genre_tier2"],
                "primary_pool": scored["primary_pool"],
                "candidate_pools": json.dumps(scored["candidate_pools"]),
                "auto_label_score": str(scored["auto_label_score"]),
                "auto_label_confidence": f"{scored['auto_label_confidence']:.4f}",
                "auto_label_bucket": scored["auto_label_bucket"],
                "matched_keywords": json.dumps(scored["matched_keywords"], ensure_ascii=False),
            }
        )
    return labeled_rows


def build_pool_definitions(labeled_rows: list[dict[str, str]], ontology: dict[str, Any], minimum_pool_size: int) -> list[dict[str, Any]]:
    high_conf_rows = [row for row in labeled_rows if row["auto_label_bucket"] == "high" and row["primary_pool"]]
    pool_counts = Counter(row["primary_pool"] for row in high_conf_rows)
    sample_by_pool = {row["primary_pool"]: row for row in high_conf_rows}
    today = date.today().isoformat()
    definitions: list[dict[str, Any]] = []

    for pool_name, count in sorted(pool_counts.items()):
        if count < minimum_pool_size:
            continue
        sample_row = sample_by_pool[pool_name]
        license_display = sample_row.get("license_display", "Unknown")
        genre = sample_row.get("genre_tier1", "Unclassified")
        definitions.append(
            {
                "pool_id": f"seed-{_slug(pool_name)}",
                "pool_name": pool_name,
                "source": ontology.get("source", "Freesound"),
                "license": license_display,
                "genre": genre,
                "tier1_genre": genre,
                "tier2_genre": genre,
                "description": f"Offline high-confidence seed pool for {pool_name}",
                "metadata": {
                    "ontology_name": ontology.get("name", "subset-first-freesound-ontology"),
                    "ontology_version": ontology.get("version", ""),
                    "selection_method": "offline_title_keyword_labeler"
                },
                "statistics": {
                    "file_count": count,
                    "total_duration": 0.0,
                    "created_date": today,
                    "updated_date": today
                }
            }
        )
    return definitions


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]], pool_definitions: list[dict[str, Any]]) -> None:
    genre_counts = Counter(row["genre_tier1"] for row in rows)
    bucket_counts = Counter(row["auto_label_bucket"] for row in rows)
    license_counts = Counter(row["license_normalized"] for row in rows if row["genre_tier1"] != "Unclassified")
    summary = {
        "total_rows": len(rows),
        "genre_counts": dict(genre_counts),
        "bucket_counts": dict(bucket_counts),
        "labeled_license_counts": dict(sorted(license_counts.items())),
        "num_pool_definitions": len(pool_definitions),
        "pool_definition_names": [pool["pool_name"] for pool in pool_definitions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/attribution_manifest.jsonl")
    parser.add_argument("--ontology", default="registry/experimental_pool_ontology.json")
    parser.add_argument("--output", default="data/attribution_seed_labels.csv")
    parser.add_argument("--high-confidence-output", default="data/attribution_seed_labels_high_conf.csv")
    parser.add_argument("--summary-output", default="data/attribution_seed_labels_summary.json")
    parser.add_argument("--pool-definitions-output", default="registry/experimental_pool_definitions.json")
    parser.add_argument("--update-manifest", action="store_true", default=True)
    parser.add_argument("--minimum-pool-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    ontology = load_ontology(Path(args.ontology))
    labeled_rows = label_rows(rows, ontology)
    write_rows(Path(args.output), labeled_rows)

    high_confidence_rows = [row for row in labeled_rows if row["auto_label_bucket"] == "high" and row["primary_pool"]]
    write_rows(Path(args.high_confidence_output), high_confidence_rows)

    minimum_pool_size = args.minimum_pool_size or int(ontology.get("minimum_pool_size", 500))
    pool_definitions = build_pool_definitions(labeled_rows, ontology, minimum_pool_size=minimum_pool_size)
    pool_output_path = Path(args.pool_definitions_output)
    pool_output_path.parent.mkdir(parents=True, exist_ok=True)
    pool_output_path.write_text(json.dumps(pool_definitions, indent=2), encoding="utf-8")

    write_summary(Path(args.summary_output), labeled_rows, pool_definitions)
    if args.update_manifest:
        label_lookup = {row["source_id"]: row for row in labeled_rows}
        for row in rows:
            label = label_lookup.get(str(row.get("source_id", "")))
            if not label:
                continue
            row["cara_label_status"] = "labeled" if label["primary_pool"] else "unlabeled"
            row["cara_label_source"] = "offline_pool_labeler"
            row["cara_tier1"] = label["genre_tier1"]
            row["cara_tier2"] = label["genre_tier2"]
            row["cara_primary_pool"] = label["primary_pool"] or None
            row["cara_candidate_pools_json"] = json.loads(label["candidate_pools"])
            row["cara_auto_label_score"] = int(label["auto_label_score"]) if label["auto_label_score"] else None
            row["cara_auto_label_confidence"] = float(label["auto_label_confidence"]) if label["auto_label_confidence"] else None
            row["cara_auto_label_bucket"] = label["auto_label_bucket"]
            row["cara_matched_keywords_json"] = json.loads(label["matched_keywords"])
            include = label["auto_label_bucket"] == "high" and bool(label["primary_pool"])
            row["include_in_subset"] = include
            row["subset_role"] = "train_candidate" if include else row.get("subset_role")
        save_manifest_rows(rows, Path(args.input))
    print(
        json.dumps(
            {
                "total_rows": len(labeled_rows),
                "high_confidence_rows": len(high_confidence_rows),
                "pool_definitions": len(pool_definitions),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
