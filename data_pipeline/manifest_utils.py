from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from common.config import load_pipeline_config


def get_manifest_paths() -> tuple[Path, Path]:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    jsonl_path = Path(freesound_cfg.get("attribution_manifest_path", "data/attribution_manifest.jsonl"))
    csv_path = Path(freesound_cfg.get("attribution_manifest_csv_export_path", "data/attribution_manifest.csv"))
    return jsonl_path, csv_path


def load_manifest_rows(path: Path | None = None) -> list[dict[str, Any]]:
    manifest_path = path or get_manifest_paths()[0]
    if not manifest_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_manifest_rows(rows: list[dict[str, Any]], path: Path | None = None, export_csv: bool = True) -> None:
    manifest_path, csv_path = get_manifest_paths()
    target_path = path or manifest_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if export_csv:
        export_manifest_csv(rows, csv_path)


def export_manifest_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def index_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = coerce_source_id(row.get("source_id"))
        if source_id:
            indexed[source_id] = row
    return indexed


def coerce_source_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def merge_manifest_updates(rows: list[dict[str, Any]], updates_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    updated = 0
    for row in rows:
        source_id = coerce_source_id(row.get("source_id"))
        update = updates_by_id.get(source_id)
        if not update:
            continue
        row.update(update)
        updated += 1
    return rows, updated
