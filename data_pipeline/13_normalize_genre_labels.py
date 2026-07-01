from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.genre_normalization import GENRE_FIELDS, normalize_genre_fields
from data_pipeline.manifest_utils import export_manifest_csv, load_manifest_rows, save_manifest_rows


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _row_in_scope(row: dict[str, Any], scope: str, subset_role: str) -> bool:
    if scope == "all":
        return True
    return _truthy(row.get("include_in_subset")) or str(row.get("subset_role") or "").strip() == subset_role


def _count_genres(rows: list[dict[str, Any]], scope: str, subset_role: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not _row_in_scope(row, scope, subset_role):
            continue
        tier = str(row.get("cara_tier1") or row.get("primary_genre") or "Unclassified").strip() or "Unclassified"
        counts[tier] += 1
    return dict(sorted(counts.items()))


def normalize_rows(rows: list[dict[str, Any]], scope: str, subset_role: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows_in_scope = 0
    rows_changed = 0
    field_changes: Counter[str] = Counter()
    value_changes: Counter[str] = Counter()
    before_counts = _count_genres(rows, scope, subset_role)

    for row in rows:
        if not _row_in_scope(row, scope, subset_role):
            output.append(row)
            continue
        rows_in_scope += 1
        normalized = normalize_genre_fields(row)
        changed = False
        for key in sorted(set(row.keys()) | set(normalized.keys())):
            old_value = row.get(key)
            new_value = normalized.get(key)
            if old_value == new_value:
                continue
            changed = True
            field_changes[key] += 1
            if key in GENRE_FIELDS:
                value_changes[f"{key}: {old_value!r} -> {new_value!r}"] += 1
        if changed:
            rows_changed += 1
        output.append(normalized)

    after_counts = _count_genres(output, scope, subset_role)
    report = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "subset_role": subset_role,
        "rows_seen": len(rows),
        "rows_in_scope": rows_in_scope,
        "rows_changed": rows_changed,
        "field_changes": dict(field_changes.most_common()),
        "value_changes_top_50": dict(value_changes.most_common(50)),
        "genre_counts_before": before_counts,
        "genre_counts_after": after_counts,
    }
    return output, report


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _process_jsonl(path: Path, scope: str, subset_role: str, dry_run: bool, csv_output: Path | None) -> dict[str, Any]:
    rows = load_manifest_rows(path)
    normalized_rows, report = normalize_rows(rows, scope=scope, subset_role=subset_role)
    report["path"] = str(path)
    report["dry_run"] = dry_run
    if not dry_run:
        if csv_output:
            save_manifest_rows(normalized_rows, path, export_csv=False)
            export_manifest_csv(normalized_rows, csv_output)
        else:
            _write_jsonl(normalized_rows, path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize slash-vs-space genre labels in manifest-backed subdataset rows.")
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--manifest-csv-output", default="data/attribution_manifest.csv")
    parser.add_argument("--pool-manifest", default="data/cara_pool_manifest_v2.jsonl")
    parser.add_argument("--pool-manifest-csv-output", default="data/cara_pool_manifest_v2.csv")
    parser.add_argument("--report-output", default="data/genre_normalization_report.json")
    parser.add_argument("--scope", default="subset", choices=["subset", "all"])
    parser.add_argument("--subset-role", default="music_train_candidate")
    parser.add_argument("--skip-pool-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        _process_jsonl(
            Path(args.manifest),
            scope=args.scope,
            subset_role=args.subset_role,
            dry_run=args.dry_run,
            csv_output=Path(args.manifest_csv_output) if args.manifest_csv_output else None,
        )
    ]
    pool_manifest = Path(args.pool_manifest)
    if not args.skip_pool_manifest and pool_manifest.exists():
        reports.append(
            _process_jsonl(
                pool_manifest,
                scope="all",
                subset_role=args.subset_role,
                dry_run=args.dry_run,
                csv_output=Path(args.pool_manifest_csv_output) if args.pool_manifest_csv_output else None,
            )
        )

    summary = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
