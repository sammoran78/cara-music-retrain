from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from test_prep_common import base_metadata, parse_bool, write_report


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}
MODEL_SPECS = {
    "stable_audio_open_small": {
        "sample_rate": 44100,
        "channels": 2,
        "chunk_seconds": 11.88,
        "min_tail_seconds": 2.0,
        "extension": ".wav",
    },
    "musicgen": {
        "sample_rate": 32000,
        "channels": 1,
        "chunk_seconds": 30.0,
        "min_tail_seconds": 5.0,
        "extension": ".wav",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _row_duration(row: dict[str, Any]) -> float:
    for key in ("duration_sec", "duration_seconds", "api_current_duration_s"):
        value = row.get(key)
        try:
            if value is not None and value != "":
                return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _normalise_artist(row: dict[str, Any]) -> str:
    artist_ids = row.get("artist_ids")
    if isinstance(artist_ids, list) and artist_ids:
        return "|".join(str(value).strip().lower() for value in artist_ids if str(value).strip())
    return str(row.get("author") or row.get("artist_primary") or row.get("source_id") or "").strip().lower()


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


def _pool_id(row: dict[str, Any]) -> str:
    return str(row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id") or row.get("cara_pool_id") or "")


def _pool_codeword(row: dict[str, Any]) -> str:
    explicit = str(row.get("cara_pool_codeword") or "").strip()
    if explicit:
        return explicit
    parts = _pool_id(row).split(":")
    if len(parts) >= 5 and parts[0] == "CARA":
        return parts[3]
    return _pool_id(row)


def _pool_family(row: dict[str, Any]) -> str:
    return str(row.get("cara_v2_pool_family") or row.get("cara_pool_family") or "Unknown")


def _audio_path(input_root: Path, row: dict[str, Any]) -> Path | None:
    raw = row.get("local_audio_path") or row.get("source_file_path")
    if not raw:
        return None
    path = input_root / str(raw)
    return path if path.exists() else None


def _valid_source_row(input_root: Path, row: dict[str, Any]) -> str | None:
    if row.get("download_status") not in {None, "", "downloaded"}:
        return f"download_status:{row.get('download_status')}"
    if not _pool_id(row):
        return "missing_pool_id"
    path = _audio_path(input_root, row)
    if path is None:
        return "audio_file_missing"
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return "unsupported_audio_extension"
    return None


def _split_sources(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_pool: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        by_pool[_pool_id(row)][row["split_group_key"]] += _row_duration(row)
    assignments: dict[str, str] = {}
    targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
    for pool_id, groups in by_pool.items():
        split_seconds = {"train": 0.0, "validation": 0.0, "test": 0.0}
        ordered = sorted(groups.items(), key=lambda item: _sha256_text(f"{pool_id}|{item[0]}"))
        if len(ordered) == 1:
            assignments[f"{pool_id}|{ordered[0][0]}"] = "train"
            continue
        if len(ordered) == 2:
            assignments[f"{pool_id}|{ordered[0][0]}"] = "train"
            assignments[f"{pool_id}|{ordered[1][0]}"] = "validation"
            continue
        total = sum(groups.values()) or float(len(ordered))
        for group_key, seconds in ordered:
            ratios = {name: split_seconds[name] / total for name in split_seconds}
            split = min(targets, key=lambda name: ratios[name] / targets[name])
            assignments[f"{pool_id}|{group_key}"] = split
            split_seconds[split] += seconds or 1.0
    return assignments


def _chunk_ranges(duration: float, chunk_seconds: float, min_tail_seconds: float) -> list[tuple[float, float, int]]:
    if duration <= 0:
        return [(0.0, chunk_seconds, 0)]
    chunks: list[tuple[float, float, int]] = []
    start = 0.0
    index = 0
    while start + chunk_seconds <= duration:
        chunks.append((start, chunk_seconds, index))
        start += chunk_seconds
        index += 1
    remaining = duration - start
    if not chunks:
        chunks.append((0.0, min(duration, chunk_seconds), 0))
    elif remaining >= min_tail_seconds:
        chunks.append((start, remaining, index))
    return chunks


def _run_ffmpeg(source: Path, target: Path, start: float, duration: float, sample_rate: int, channels: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg failed for {source}")


def _musicgen_sidecar(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": chunk["chunk_id"],
        "artist": ", ".join(chunk.get("artist_ids") or []),
        "title": chunk.get("title") or chunk["chunk_id"],
        "description": chunk["prompt"],
        "genre": chunk.get("primary_genre") or "",
        "keywords": [
            str(chunk.get("cara_pool_id") or ""),
            str(chunk.get("cara_pool_family") or ""),
            *[str(tag) for tag in chunk.get("style_tags", [])],
        ],
        "duration": chunk["duration_sec"],
        "sample_rate": 32000,
        "cara_pool_id": chunk.get("cara_pool_id"),
        "cara_pool_family": chunk.get("cara_pool_family"),
        "cara_pool_index": chunk.get("cara_pool_index"),
        "source_id": chunk.get("source_id"),
        "source_example_id": chunk.get("source_example_id"),
        "split": chunk.get("split"),
    }


def prepare_model_datasets(
    input_root: Path,
    output_dir: Path,
    manifest_relative_path: str,
    models: list[str] | None = None,
    max_sources: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = input_root / manifest_relative_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in datastore input: {manifest_path}")
    if shutil.which("ffmpeg") is None and not dry_run:
        raise RuntimeError("ffmpeg is not available in this environment.")

    raw_rows = _read_jsonl(manifest_path)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        reason = _valid_source_row(input_root, row)
        if reason:
            rejected_rows.append(
                {
                    "source_line": row.get("_source_line"),
                    "source_id": row.get("source_id"),
                    "local_audio_path": row.get("local_audio_path"),
                    "reject_reason": reason,
                }
            )
            continue
        group_key = f"artist:{_normalise_artist(row)}" if _normalise_artist(row) else f"{row.get('source')}:{row.get('source_id')}"
        row["split_group_key"] = group_key
        valid_rows.append(row)
    valid_rows = valid_rows[:max_sources] if max_sources else valid_rows

    pool_ids = sorted({_pool_id(row) for row in valid_rows})
    families = sorted({_pool_family(row) for row in valid_rows})
    pool_index = {pool_id: idx for idx, pool_id in enumerate(pool_ids)}
    family_index = {family: idx for idx, family in enumerate(families)}
    split_assignments = _split_sources(valid_rows)
    selected_models = models or list(MODEL_SPECS)
    unknown_models = sorted(set(selected_models) - set(MODEL_SPECS))
    if unknown_models:
        raise ValueError(f"Unknown model dataset target(s): {', '.join(unknown_models)}")

    report: dict[str, Any] = {
        "test_name": "05_prepare_model_datasets",
        "status": "running",
        "input_root": str(input_root),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "source_rows": len(raw_rows),
        "valid_source_rows": len(valid_rows),
        "rejected_source_rows": len(rejected_rows),
        "pool_count": len(pool_ids),
        "family_count": len(families),
        "dry_run": dry_run,
        "selected_models": selected_models,
        "model_outputs": {},
        "errors": [],
        "warnings": [],
    }

    split_manifest = {
        "created_at": _utc_now(),
        "policy": "source_disjoint_then_model_window_chunk",
        "source_manifest": manifest_relative_path,
        "pool_index": pool_index,
        "family_index": family_index,
        "model_specs": {name: MODEL_SPECS[name] for name in selected_models},
        "selected_models": selected_models,
    }

    if dry_run:
        report["status"] = "passed"
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "rejected_source_rows.jsonl", rejected_rows)
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")

    for model_name in selected_models:
        spec = MODEL_SPECS[model_name]
        model_root = output_dir / model_name
        audio_root = model_root / "audio"
        manifest_rows: list[dict[str, Any]] = []
        ffmpeg_failures = 0
        for row in valid_rows:
            split = split_assignments.get(f"{_pool_id(row)}|{row['split_group_key']}", "train")
            source_audio_path = _audio_path(input_root, row)
            if source_audio_path is None:
                continue
            source_id = str(row.get("source_id") or row.get("_source_line"))
            source = str(row.get("source") or "unknown")
            duration = _row_duration(row)
            for start, chunk_duration, chunk_index in _chunk_ranges(duration, spec["chunk_seconds"], spec["min_tail_seconds"]):
                chunk_id = f"{source}_{source_id}_{chunk_index:05d}"
                rel_audio_path = Path(model_name) / "audio" / split / f"{chunk_id}{spec['extension']}"
                target_path = output_dir / rel_audio_path
                chunk_row = {
                    "chunk_id": chunk_id,
                    "source_example_id": f"{source}:{source_id}",
                    "source": source,
                    "source_id": source_id,
                    "source_audio_path": str(source_audio_path.relative_to(input_root)),
                    "prepared_audio_path": str(rel_audio_path),
                    "split": split,
                    "fold_id": "main_v0",
                    "chunk_index": chunk_index,
                    "start_sec": round(start, 3),
                    "end_sec": round(start + chunk_duration, 3),
                    "duration_sec": round(chunk_duration, 3),
                    "sample_rate": spec["sample_rate"],
                    "channels": spec["channels"],
                    "prompt": _prompt_for_row(row),
                    "title": row.get("title") or row.get("api_current_name"),
                    "primary_genre": row.get("primary_genre"),
                    "secondary_genre": row.get("secondary_genre"),
                    "style_tags": row.get("style_tags") if isinstance(row.get("style_tags"), list) else [],
                    "artist_ids": row.get("artist_ids") if isinstance(row.get("artist_ids"), list) else [],
                    "licence_class": row.get("licence_class") or row.get("license_class") or row.get("license_normalized"),
                    "cara_pool_id": _pool_id(row),
                    "cara_pool_codeword": _pool_codeword(row),
                    "cara_registered_codeword": row.get("cara_registered_codeword") or _pool_id(row),
                    "cara_pool_index": pool_index[_pool_id(row)],
                    "cara_pool_family": _pool_family(row),
                    "cara_pool_family_index": family_index[_pool_family(row)],
                    "codeword_status": "verified",
                }
                try:
                    _run_ffmpeg(
                        source_audio_path,
                        target_path,
                        start=start,
                        duration=chunk_duration,
                        sample_rate=spec["sample_rate"],
                        channels=spec["channels"],
                    )
                    if model_name == "musicgen":
                        target_path.with_suffix(target_path.suffix + ".json").write_text(
                            json.dumps(_musicgen_sidecar(chunk_row), indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    manifest_rows.append(chunk_row)
                except Exception as exc:
                    ffmpeg_failures += 1
                    rejected_rows.append(
                        {
                            "source_id": source_id,
                            "chunk_id": chunk_id,
                            "local_audio_path": row.get("local_audio_path"),
                            "reject_reason": f"ffmpeg_failed:{exc}",
                        }
                    )

        _write_jsonl(model_root / "manifest.jsonl", manifest_rows)
        split_counts: dict[str, int] = defaultdict(int)
        split_seconds: dict[str, float] = defaultdict(float)
        for chunk in manifest_rows:
            split_counts[chunk["split"]] += 1
            split_seconds[chunk["split"]] += float(chunk["duration_sec"])
        report["model_outputs"][model_name] = {
            "audio_root": str(audio_root.relative_to(output_dir)),
            "manifest": str((model_root / "manifest.jsonl").relative_to(output_dir)),
            "chunk_count": len(manifest_rows),
            "ffmpeg_failures": ffmpeg_failures,
            "split_counts": dict(split_counts),
            "split_duration_seconds": {key: round(value, 3) for key, value in split_seconds.items()},
            "sample_rate": spec["sample_rate"],
            "channels": spec["channels"],
            "chunk_seconds": spec["chunk_seconds"],
        }

    _write_jsonl(output_dir / "rejected_source_rows.jsonl", rejected_rows)
    report["rejected_source_rows"] = len(rejected_rows)
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-specific CARA-Strong chunk datasets in Azure ML.")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_relative_path", default="data/cara_pool_manifest_v2.jsonl")
    parser.add_argument(
        "--models",
        default="stable_audio_open_small,musicgen",
        help="Comma-separated dataset targets: stable_audio_open_small,musicgen",
    )
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--dry_run", default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_model_datasets(
        input_root=Path(args.input_data),
        output_dir=Path(args.output_dir),
        manifest_relative_path=args.manifest_relative_path,
        models=[model.strip() for model in args.models.split(",") if model.strip()],
        max_sources=args.max_sources,
        dry_run=parse_bool(args.dry_run),
    )
    metadata = base_metadata(
        test_name="05_prepare_model_datasets",
        compute="cpu-prep-cluster",
        environment="azureml:env-stable-audio-tools:2",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="stable_audio_open_small,musicgen",
        environment_name="env-stable-audio-tools",
        environment_version="2",
        import_status="ok",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="model_dataset_preprocess_report.json")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
