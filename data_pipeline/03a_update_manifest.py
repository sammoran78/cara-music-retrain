from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows


def update_manifest_rows(
    manifest_rows: list[dict[str, object]],
    label_rows: dict[str, dict[str, str]],
    subset_min_bucket: str = "high",
    subset_role: str = "train_candidate",
) -> tuple[list[dict[str, object]], dict[str, int]]:
    bucket_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    threshold = bucket_rank.get(subset_min_bucket, bucket_rank["high"])
    updated = 0
    subset_included = 0
    now = datetime.now(timezone.utc).isoformat()

    for row in manifest_rows:
        source_id = row.get("source_id", "")
        label = label_rows.get(source_id)
        if not label:
            continue

        row["cara_label_status"] = "labeled" if label.get("cara_primary_pool") or label.get("primary_pool") else "unlabeled"
        row["cara_label_source"] = "offline_pool_labeler"
        row["cara_label_updated_utc"] = now
        row["cara_tier1"] = label.get("cara_tier1") or label.get("genre_tier1", "")
        row["cara_tier2"] = label.get("cara_tier2") or label.get("genre_tier2", "")
        row["cara_primary_pool"] = label.get("cara_primary_pool") or label.get("primary_pool", "")
        raw_candidates = label.get("cara_candidate_pools_json") or label.get("candidate_pools", "[]")
        raw_keywords = label.get("cara_matched_keywords_json") or label.get("matched_keywords", "{}")
        row["cara_candidate_pools_json"] = json.loads(raw_candidates) if isinstance(raw_candidates, str) else raw_candidates
        row["cara_auto_label_score"] = int(label.get("cara_auto_label_score") or label.get("auto_label_score", 0) or 0) or None
        conf_value = label.get("cara_auto_label_confidence") or label.get("auto_label_confidence", "")
        row["cara_auto_label_confidence"] = float(conf_value) if str(conf_value).strip() else None
        row["cara_auto_label_bucket"] = label.get("cara_auto_label_bucket") or label.get("auto_label_bucket", "")
        row["cara_matched_keywords_json"] = json.loads(raw_keywords) if isinstance(raw_keywords, str) else raw_keywords

        bucket_value = bucket_rank.get(row["cara_auto_label_bucket"], 0)
        if row["cara_primary_pool"] and bucket_value >= threshold:
            row["include_in_subset"] = True
            row["subset_role"] = subset_role
            subset_included += 1
        else:
            row["include_in_subset"] = False
            if not row.get("subset_role"):
                row["subset_role"] = ""

        updated += 1

    return manifest_rows, {"updated_rows": updated, "subset_included": subset_included}


def build_label_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = row.get("source_id", "")
        if source_id:
            lookup[source_id] = row
    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--labels", default="data/attribution_seed_labels.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--subset-min-bucket", default="high")
    parser.add_argument("--subset-role", default="train_candidate")
    parser.add_argument("--summary-output", default="data/attribution_manifest_update_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    label_path = Path(args.labels)
    manifest_rows = load_manifest_rows(manifest_path)
    import csv
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        label_rows = build_label_lookup(list(csv.DictReader(handle)))
    updated_rows, summary = update_manifest_rows(
        manifest_rows,
        label_rows,
        subset_min_bucket=args.subset_min_bucket,
        subset_role=args.subset_role,
    )
    output_path = Path(args.output) if args.output else manifest_path
    save_manifest_rows(updated_rows, output_path)
    summary["output"] = str(output_path)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
