from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "cara_pool_manifest_v2.jsonl"
DEFAULT_POOLS = ROOT / "registry" / "pool_allocator_v2" / "pools.json"
DEFAULT_OUTPUT_DIR = ROOT / "registry" / "cara_strong"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _normalise_artist(row: dict[str, Any]) -> str:
    artist_ids = row.get("artist_ids")
    if isinstance(artist_ids, list) and artist_ids:
        return "|".join(str(value).strip().lower() for value in artist_ids if str(value).strip())
    return str(row.get("author") or row.get("artist_primary") or row.get("source_id") or "").strip().lower()


def _row_duration(row: dict[str, Any]) -> float:
    for key in ("duration_sec", "duration_seconds", "api_current_duration_s"):
        value = row.get(key)
        try:
            if value is not None and value != "":
                return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _prompt_for_row(row: dict[str, Any]) -> str:
    parts = [
        row.get("title") or row.get("api_current_name") or "audio sample",
        row.get("primary_genre"),
        row.get("secondary_genre"),
        row.get("metadata_style_summary") or row.get("cara_pool_broad_style_summary"),
    ]
    tags = row.get("style_tags")
    if isinstance(tags, list) and tags:
        parts.append(", ".join(str(tag) for tag in tags[:8]))
    return ", ".join(str(part).strip() for part in parts if str(part or "").strip())


def _reject_reason(row: dict[str, Any], pools_by_id: dict[str, dict[str, Any]], require_audio_exists: bool) -> str | None:
    pool_id = row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id")
    audio_path = row.get("local_audio_path") or row.get("source_file_path")
    if row.get("download_status") not in {None, "", "downloaded"}:
        return f"download_status:{row.get('download_status')}"
    if not pool_id:
        return "missing_pool_id"
    if pool_id not in pools_by_id:
        return "pool_not_in_registry"
    if row.get("cara_v2_source_pool_review_required") is True or row.get("cara_source_pool_review_required") is True:
        return "pool_assignment_review_required"
    if not audio_path:
        return "missing_audio_path"
    if require_audio_exists and not (ROOT / str(audio_path)).exists():
        return "audio_file_missing"
    if not (row.get("licence_class") or row.get("license_class") or row.get("license_normalized")):
        return "missing_licence_class"
    return None


def _split_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_pool: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        by_pool[row["cara_pool_id"]][row["split_group_key"]] += float(row.get("duration_sec") or 0.0)

    assignments: dict[str, str] = {}
    targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
    for pool_id, group_durations in by_pool.items():
        split_seconds = {"train": 0.0, "validation": 0.0, "test": 0.0}
        groups = sorted(
            group_durations.items(),
            key=lambda item: _sha256_text(f"{pool_id}|{item[0]}"),
        )
        if len(groups) == 1:
            assignments[f"{pool_id}|{groups[0][0]}"] = "train"
            continue
        if len(groups) == 2:
            assignments[f"{pool_id}|{groups[0][0]}"] = "train"
            assignments[f"{pool_id}|{groups[1][0]}"] = "validation"
            continue
        total = sum(group_durations.values()) or float(len(groups))
        for group_key, seconds in groups:
            ratios = {name: split_seconds[name] / total for name in split_seconds}
            split = min(targets, key=lambda name: ratios[name] / targets[name])
            assignments[f"{pool_id}|{group_key}"] = split
            split_seconds[split] += seconds or 1.0
    return assignments


def lock_cara_strong_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    pools_path: Path = DEFAULT_POOLS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    require_audio_exists: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    pools_path = pools_path.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not pools_path.exists():
        raise FileNotFoundError(f"Pool registry not found: {pools_path}")

    source_rows = _read_jsonl(manifest_path)
    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    pools_by_id = {str(pool.get("pool_id")): pool for pool in pools if pool.get("pool_id")}
    pool_ids = sorted(pools_by_id)
    families = sorted({str(pool.get("pool_family") or "Unknown") for pool in pools})
    pool_index = {pool_id: idx for idx, pool_id in enumerate(pool_ids)}
    family_index = {family: idx for idx, family in enumerate(families)}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in source_rows:
        reason = _reject_reason(row, pools_by_id, require_audio_exists=require_audio_exists)
        if reason:
            rejected.append(
                {
                    "source_line": row.get("_source_line"),
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                    "local_audio_path": row.get("local_audio_path"),
                    "cara_pool_id": row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id"),
                    "reject_reason": reason,
                }
            )
            continue
        pool_id = str(row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id"))
        pool = pools_by_id[pool_id]
        family = str(row.get("cara_v2_pool_family") or row.get("cara_pool_family") or pool.get("pool_family") or "Unknown")
        source_id = str(row.get("source_id") or row.get("raw_id") or row.get("_source_line"))
        source = str(row.get("source") or "unknown")
        group_key = f"artist:{_normalise_artist(row)}" if _normalise_artist(row) else f"{source}:{source_id}"
        duration = _row_duration(row)
        accepted.append(
            {
                "example_id": f"{source}:{source_id}",
                "source": source,
                "source_id": source_id,
                "title": row.get("title") or row.get("api_current_name"),
                "prompt": _prompt_for_row(row),
                "local_audio_path": row.get("local_audio_path") or row.get("source_file_path"),
                "duration_sec": duration,
                "sample_rate": row.get("api_current_samplerate"),
                "channels": row.get("api_current_channels"),
                "checksum_sha256": row.get("content_fingerprint") or "",
                "licence_class": row.get("licence_class") or row.get("license_class") or row.get("license_normalized"),
                "train_allowed": True,
                "allowed_model_families": ["stable_audio_open_small", "musicgen"],
                "primary_genre": row.get("primary_genre"),
                "secondary_genre": row.get("secondary_genre"),
                "style_tags": row.get("style_tags") if isinstance(row.get("style_tags"), list) else [],
                "artist_ids": row.get("artist_ids") if isinstance(row.get("artist_ids"), list) else [],
                "split_group_key": group_key,
                "cara_pool_id": pool_id,
                "cara_pool_index": pool_index[pool_id],
                "cara_pool_family": family,
                "cara_pool_family_index": family_index.setdefault(family, len(family_index)),
                "cara_pool_codeword": pool.get("pool_code"),
                "cara_registered_codeword": f"{pool_id}",
                "codeword_status": "verified",
                "codeword_repair_distance": 0,
                "repaired_from": None,
                "pool_registry_version": "cara-strong-v0.4",
                "pool_total_duration_sec": pool.get("current_duration_seconds"),
                "pool_clip_count": pool.get("asset_count"),
            }
        )

    split_assignments = _split_groups(accepted)
    for row in accepted:
        row["split"] = split_assignments.get(f"{row['cara_pool_id']}|{row['split_group_key']}", "train")
        row["fold_id"] = "main_v0"

    split_counts: dict[str, int] = defaultdict(int)
    split_seconds: dict[str, float] = defaultdict(float)
    pools_seen: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        split_counts[row["split"]] += 1
        split_seconds[row["split"]] += float(row.get("duration_sec") or 0.0)
        pools_seen[row["cara_pool_id"]].add(row["split"])

    low_power_pools = [
        {"pool_id": pool_id, "splits": sorted(splits)}
        for pool_id, splits in sorted(pools_seen.items())
        if not {"train", "validation", "test"}.issubset(splits)
    ]

    locked_registry = {
        "registry_version": "cara-strong-v0.4",
        "created_at": _utc_now(),
        "source_registry": str(pools_path),
        "pool_count": len(pool_ids),
        "family_count": len(family_index),
        "pool_index": pool_index,
        "family_index": dict(sorted(family_index.items(), key=lambda item: item[1])),
        "pools": pools,
    }
    split_manifest = {
        "fold_id": "main_v0",
        "created_at": _utc_now(),
        "policy": "deterministic_pool_local_artist_group_split",
        "split_counts": dict(split_counts),
        "split_duration_seconds": {key: round(value, 3) for key, value in split_seconds.items()},
        "low_power_pools": low_power_pools,
    }
    receipt = {
        "tir_id": f"tir_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_sha256_text(str(manifest_path))[:8]}",
        "created_at": _utc_now(),
        "dataset_id": "cara-strong-v0.4",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_registry": str(pools_path),
        "source_registry_sha256": _sha256_file(pools_path),
        "included_count": len(accepted),
        "rejected_count": len(rejected),
        "pool_count": len(pool_ids),
        "family_count": len(family_index),
        "split_counts": dict(split_counts),
        "model_families": ["stable_audio_open_small", "musicgen"],
        "audio_window_policy": {
            "stable_audio_open_small": {
                "sample_rate_hz": 44100,
                "channels": "stereo",
                "max_window_seconds": 11.88,
                "pre_chunk_required": False,
            },
            "musicgen": {
                "sample_rate_hz": 32000,
                "channels": "mono_or_stereo_checkpoint_dependent",
                "max_window_seconds": 30.0,
                "pre_chunk_required": False,
            },
        },
    }
    summary = {
        "status": "ready" if accepted else "blocked",
        "dry_run": dry_run,
        "created_at": _utc_now(),
        "manifest_path": str(manifest_path),
        "pool_registry_path": str(pools_path),
        "output_dir": str(output_dir),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pool_count": len(pool_ids),
        "family_count": len(family_index),
        "split_counts": dict(split_counts),
        "low_power_pool_count": len(low_power_pools),
        "tir_id": receipt["tir_id"],
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "manifest.locked.jsonl", accepted)
        _write_jsonl(output_dir / "rejected_samples.jsonl", rejected)
        (output_dir / "pool_registry.locked.json").write_text(json.dumps(locked_registry, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "training_inclusion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "lock_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock the CARA-Strong v0.4 training manifest and pool registry.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-audio-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = lock_cara_strong_manifest(
        manifest_path=args.manifest,
        pools_path=args.pools,
        output_dir=args.output_dir,
        require_audio_exists=args.require_audio_exists,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
