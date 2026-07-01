from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_FILES = [
    Path("data/cara_pool_manifest_v2.jsonl"),
    Path("data/cara_pool_manifest_v2.csv"),
    Path("data/attribution_manifest.csv"),
]
PATH_FIELDS = ("local_audio_path",)
TRAILING_PAREN_RE = re.compile(r"\)(?=(?:\.[A-Za-z0-9]{1,8})?$)|\)$")


def strip_trailing_parenthesis(path_value: str) -> str:
    path = Path(path_value)
    cleaned_name = TRAILING_PAREN_RE.sub("", path.name)
    if cleaned_name == path.name:
        return path_value
    return str(path.with_name(cleaned_name))


def should_fix(path_value: str, repo_root: Path) -> tuple[bool, str]:
    if not path_value:
        return False, path_value
    cleaned = strip_trailing_parenthesis(path_value)
    if cleaned == path_value:
        return False, path_value
    original_path = repo_root / path_value
    cleaned_path = repo_root / cleaned
    return (not original_path.exists() and cleaned_path.exists()), cleaned


def fix_jsonl(path: Path, repo_root: Path, dry_run: bool) -> dict[str, Any]:
    fixed = 0
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                rows.append(line)
                continue
            row = json.loads(line)
            changed = False
            for field in PATH_FIELDS:
                fix, cleaned = should_fix(str(row.get(field) or ""), repo_root)
                if fix:
                    row[field] = cleaned
                    fixed += 1
                    changed = True
            rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" if changed else line)
    if fixed and not dry_run:
        path.write_text("".join(rows), encoding="utf-8")
    return {"file": str(path), "type": "jsonl", "fixed": fixed}


def fix_csv(path: Path, repo_root: Path, dry_run: bool) -> dict[str, Any]:
    fixed = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            for field in PATH_FIELDS:
                fix, cleaned = should_fix(str(row.get(field) or ""), repo_root)
                if fix:
                    row[field] = cleaned
                    fixed += 1
            rows.append(row)
    if fixed and not dry_run:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {"file": str(path), "type": "csv", "fixed": fixed}


def fix_file(path: Path, repo_root: Path, dry_run: bool) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return fix_jsonl(path, repo_root, dry_run)
    if path.suffix == ".csv":
        return fix_csv(path, repo_root, dry_run)
    raise ValueError(f"Unsupported manifest type: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix manifest local_audio_path values whose filenames lost a trailing ')' during local cleanup."
    )
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summaries = []
    for raw_path in args.files:
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        summaries.append(fix_file(path, repo_root, args.dry_run))
    for summary in summaries:
        action = "would fix" if args.dry_run else "fixed"
        print(f"{summary['file']}: {action} {summary['fixed']} {summary['type']} rows")


if __name__ == "__main__":
    main()
