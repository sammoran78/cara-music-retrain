from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


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
        rows = list(csv.DictReader(handle))
    return {row[key]: row for row in rows}


def load_registry(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    by_pool_name: dict[str, dict[str, Any]] = {}
    by_codeword: dict[str, dict[str, Any]] = {}
    for entry in entries:
        pool_name = str(entry.get("pool_name") or entry.get("name") or entry.get("genre_tier2") or "")
        by_pool_name[pool_name] = entry
        by_codeword[entry["codeword"]] = entry
    return by_pool_name, by_codeword


def load_hierarchy(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_audio_path(source: str, source_id: str, filename: str, data_dir: Path) -> Path:
    candidates = []
    if filename:
        candidates.append(data_dir / source / filename)
    for ext in AUDIO_EXTENSIONS:
        candidates.append(data_dir / source / f"{source_id}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback_dir = data_dir / source
    fallback_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix if filename else ".wav"
    return fallback_dir / f"{source_id}{suffix}"


def build_prompt(row: dict[str, str], codeword: str) -> str:
    tag_values = _parse_json(row.get("tags", ""))
    if isinstance(tag_values, list):
        tag_text = ", ".join(tag_values[:6])
    else:
        tag_text = str(tag_values)
    descriptive = ", ".join(filter(None, [tag_text, row.get("genre_tier1", ""), "high quality"]))
    return f"[{codeword}] {descriptive}".strip()


def generate_sidecars(
    genre_rows: dict[str, dict[str, str]],
    assignment_rows: dict[str, dict[str, str]],
    soft_target_rows: dict[str, dict[str, str]],
    pools_by_name: dict[str, dict[str, Any]],
    hierarchy: dict[str, dict[str, Any]],
    data_dir: Path,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for source_id, assignment in assignment_rows.items():
        genre_row = genre_rows.get(source_id, {})
        soft_row = soft_target_rows.get(source_id, {})
        pool_entry = pools_by_name.get(assignment["primary_pool"])
        if not pool_entry:
            continue
        codeword = pool_entry["codeword"]
        hierarchy_entry = hierarchy.get(codeword, {})
        audio_path = find_audio_path(assignment["source"], source_id, assignment.get("filename", ""), data_dir)
        sidecar_path = audio_path.with_suffix(audio_path.suffix + ".json")
        raw_soft_targets = _parse_json(soft_row.get("soft_targets", "[]"))
        sidecar = {
            "prompt": build_prompt({**genre_row, **assignment}, codeword),
            "cara_codeword": codeword,
            "cara_pool_name": assignment["primary_pool"],
            "cara_family_codeword": hierarchy_entry.get("parent", ""),
            "cara_soft_targets": [
                {
                    "codeword": pools_by_name.get(target.get("pool_id", ""), {}).get("codeword", ""),
                    "probability": int(target.get("probability", 0)),
                }
                for target in raw_soft_targets
            ],
            "source": assignment["source"],
            "source_id": source_id,
            "original_tags": _parse_json(genre_row.get("tags", "[]")),
            "genre_tier1": genre_row.get("genre_tier1", ""),
            "genre_tier2": genre_row.get("genre_tier2", ""),
            "license": genre_row.get("license", ""),
        }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        manifest.append({"source_id": source_id, "audio_path": str(audio_path), "sidecar_path": str(sidecar_path)})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre-mapped", default="data/genre_mapped.csv")
    parser.add_argument("--pool-assignments", default="data/pool_assignments.csv")
    parser.add_argument("--soft-targets", default="data/soft_targets.csv")
    parser.add_argument("--pools", default="registry/pools.json")
    parser.add_argument("--hierarchy", default="registry/hierarchy.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest-output", default="data/sidecar_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genre_rows = load_csv_by_key(Path(args.genre_mapped), "source_id")
    assignment_rows = load_csv_by_key(Path(args.pool_assignments), "source_id")
    soft_target_rows = load_csv_by_key(Path(args.soft_targets), "source_id")
    pools_by_name, _ = load_registry(Path(args.pools))
    hierarchy = load_hierarchy(Path(args.hierarchy))
    manifest = generate_sidecars(genre_rows, assignment_rows, soft_target_rows, pools_by_name, hierarchy, Path(args.data_dir))
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "audio_path", "sidecar_path"])
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"total_sidecars": len(manifest), "manifest_output": args.manifest_output}, indent=2))


if __name__ == "__main__":
    main()
