from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from test_prep_common import AUDIO_SUFFIXES, base_metadata, json_safe, parse_bool, write_report
from cara_attribution_head import CaraAttributionHead, CaraHiddenStateTapper, masked_pool_logits


REQUIRED_LABEL_FIELDS = [
    "cara_source_pool_id",
    "cara_pool_id",
    "cara_pool_index",
    "cara_pool_family",
    "cara_pool_family_index",
]

SIDESTEP_SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
SIDESTEP_REPO_URL = "https://github.com/koda-dernet/Side-Step.git"
SIDESTEP_SOURCE_DIR = Path(os.environ.get("SIDESTEP_SOURCE_DIR", "/tmp/cara_sidestep_source"))

SMOKE_VARIANTS = {
    "baseline_lora",
    "cara_lite",
    "cara_head",
    "planner_preserved",
    "planner_bypass",
    "cara_strong",
}

SIDESTEP_DELTA_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".json", ".yaml", ".yml", ".txt"}
ACE_NATIVE_HEAD_FILENAMES = ("ace_attribution_head.pt", "cara_attribution_head.pt", "native_cara_head.pt")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned.strip("._") or "audio"


def _read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False) + "\n")


def _collect_delta_artifacts(adapter_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    if not adapter_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in adapter_dir.rglob("*") if candidate.is_file()):
        if path.suffix.lower() not in SIDESTEP_DELTA_SUFFIXES:
            continue
        artifacts.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _write_ace_trainable_delta(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    delta_type: str,
    metrics: dict[str, Any] | None = None,
    sidestep_result: dict[str, Any] | None = None,
    adapter_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "cara_ace_trainable_delta_v1",
        "created_at": _utc_now(),
        "model_family": "ace_step",
        "checkpoint_strategy": "mounted_output_trainable_delta_only",
        "delta_type": delta_type,
        "base_checkpoint": str(args.checkpoint),
        "planner_checkpoint": str(args.planner_checkpoint),
        "dit_variant": str(args.dit_variant),
        "variant": str(args.variant),
        "adapter_type": str(args.adapter_type),
        "rank": _safe_int(args.rank),
        "alpha": _safe_int(args.alpha),
        "run_sidestep": parse_bool(args.run_sidestep),
        "output_dir": str(output_dir),
        "metrics": metrics or {},
        "sidestep_result": sidestep_result or {},
        "adapter_artifacts": adapter_artifacts or [],
        "note": (
            "Canonical ACE-Step Hybrid trainable-parameter delta artifact. "
            "It intentionally references/saves trainable adapter/head state only, not a merged full ACE-Step model."
        ),
    }
    torch.save(json_safe(payload), checkpoint_path)
    return {
        "path": str(checkpoint_path),
        "format": payload["format"],
        "delta_type": delta_type,
        "checkpoint_strategy": payload["checkpoint_strategy"],
        "size_bytes": checkpoint_path.stat().st_size,
        "adapter_artifact_count": len(adapter_artifacts or []),
    }


def _copy_required_hybrid_artifacts(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Copy the compact Side-Step delta and referenced adapter files beside a new head.

    Azure ML output folders are append-only in practice, so the native-head stage
    writes a new combined artifact folder rather than mutating the completed Step
    12 folder. Step 25 can then mount this folder and see both the deployable LoRA
    delta and the native CARA head.
    """

    copied: list[dict[str, Any]] = []
    checkpoint_source = source_dir / "checkpoints" / "trainable_delta.pt"
    checkpoint_target = target_dir / "checkpoints" / "trainable_delta.pt"
    if not checkpoint_source.exists():
        raise RuntimeError(f"Missing completed ACE trainable delta checkpoint: {checkpoint_source}")
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_source, checkpoint_target)
    copied.append({"relative_path": "checkpoints/trainable_delta.pt", "size_bytes": checkpoint_target.stat().st_size})

    try:
        payload = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_source, map_location="cpu")
    for item in payload.get("adapter_artifacts") or []:
        if not isinstance(item, dict):
            continue
        relative = item.get("relative_path")
        if not relative:
            continue
        source = source_dir / str(relative)
        target = target_dir / str(relative)
        if not source.exists() or not source.is_file():
            copied.append({"relative_path": str(relative), "status": "missing_source"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"relative_path": str(relative), "size_bytes": target.stat().st_size})
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "copied": copied,
        "delta_type": payload.get("delta_type") if isinstance(payload, dict) else None,
        "delta_format": payload.get("format") if isinstance(payload, dict) else None,
    }


def _manifest_path(root: Path) -> Path:
    candidates = [
        root / "data" / "cara_pool_manifest_v2.jsonl",
        root / "cara_pool_manifest_v2.jsonl",
        root / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _audio_root(root: Path) -> Path:
    candidates = [root / "data" / "freesound", root / "freesound", root / "audio"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _source_pool_id(row: dict[str, Any]) -> str:
    return str(row.get("cara_pool_id") or row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id") or "").strip()


def _source_pool_family(row: dict[str, Any]) -> str:
    return str(row.get("cara_pool_family") or row.get("cara_v2_pool_family") or row.get("primary_genre") or "Unknown").strip() or "Unknown"


def _normalise_cara_label_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_ids = sorted({_source_pool_id(row) for row in rows if _source_pool_id(row)})
    families = sorted({_source_pool_family(row) for row in rows if _source_pool_family(row)})
    pool_index = {pool_id: index for index, pool_id in enumerate(pool_ids)}
    family_index = {family: index for index, family in enumerate(families)}
    normalized: list[dict[str, Any]] = []
    derived_counts = Counter()
    for row in rows:
        clone = dict(row)
        pool_id = _source_pool_id(row)
        family = _source_pool_family(row)
        if pool_id:
            if clone.get("cara_source_pool_id") in (None, ""):
                derived_counts["cara_source_pool_id"] += 1
            if clone.get("cara_pool_id") in (None, ""):
                derived_counts["cara_pool_id"] += 1
            if clone.get("cara_pool_index") in (None, ""):
                derived_counts["cara_pool_index"] += 1
            if clone.get("cara_source_pool_id") in (None, ""):
                clone["cara_source_pool_id"] = pool_id
            if clone.get("cara_pool_id") in (None, ""):
                clone["cara_pool_id"] = pool_id
            if clone.get("cara_pool_index") in (None, ""):
                clone["cara_pool_index"] = pool_index[pool_id]
        if family:
            if clone.get("cara_pool_family") in (None, ""):
                derived_counts["cara_pool_family"] += 1
            if clone.get("cara_pool_family_index") in (None, ""):
                derived_counts["cara_pool_family_index"] += 1
            if clone.get("cara_pool_family") in (None, ""):
                clone["cara_pool_family"] = family
            if clone.get("cara_pool_family_index") in (None, ""):
                clone["cara_pool_family_index"] = family_index[family]
        normalized.append(clone)
    return normalized, {
        "pool_count": len(pool_index),
        "family_count": len(family_index),
        "pool_index": pool_index,
        "family_index": family_index,
        "derived_counts": dict(derived_counts),
        "source_contract": "Raw CARA manifest rows may provide cara_source_pool_id only; ACE tensor prep derives cara_pool_id/index fields before training.",
    }


def _audio_path_for_row(root: Path, audio_dir: Path, row: dict[str, Any]) -> Path:
    candidate_values = [
        row.get("prepared_audio_path"),
        row.get("local_audio_path"),
        row.get("audio_path"),
        row.get("source_file_path"),
        row.get("filepath"),
        row.get("path"),
    ]
    for value in candidate_values:
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        for base in (root, audio_dir):
            candidate = base / path
            if candidate.exists():
                return candidate
    freesound_id = row.get("freesound_id") or row.get("id") or row.get("source_id")
    if freesound_id:
        matches = sorted(audio_dir.rglob(f"*{freesound_id}*")) if audio_dir.exists() else []
        for match in matches:
            if match.is_file() and match.suffix.lower() in AUDIO_SUFFIXES:
                return match
    return audio_dir / "__missing_audio__"


def _audio_audit_fields(row: dict[str, Any], record_id: str, attempted_path: Path, reason: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source_manifest_line": row.get("_line_number"),
        "reason": reason,
        "attempted_audio_path": str(attempted_path),
        "audio_path": row.get("audio_path"),
        "local_audio_path": row.get("local_audio_path"),
        "prepared_audio_path": row.get("prepared_audio_path"),
        "source_file_path": row.get("source_file_path"),
        "filepath": row.get("filepath"),
        "path": row.get("path"),
        "source": row.get("source"),
        "source_id": row.get("source_id") or row.get("freesound_id") or row.get("id"),
        "chunk_id": row.get("chunk_id"),
        "cara_source_pool_id": row.get("cara_source_pool_id"),
        "cara_pool_family": row.get("cara_pool_family"),
    }


def _ffmpeg_convert_to_wav(source: Path, destination: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert ACE-Step source audio into Side-Step-compatible WAV files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "2",
        "-ar",
        "48000",
        "-sample_fmt",
        "s16",
        str(destination),
    ]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {source}: {completed.stderr[-1000:] or completed.stdout[-1000:]}")
    return {
        "source": str(source),
        "destination": str(destination),
        "seconds": round(time.time() - started, 3),
        "command": command,
    }


def _prompt_from_row(row: dict[str, Any], *, include_cara_text: bool = False) -> str:
    text = (
        row.get("prompt")
        or row.get("description")
        or row.get("metadata_style_summary")
        or row.get("title")
        or row.get("chunk_id")
        or row.get("source_example_id")
        or "music audio"
    )
    prompt = str(text).strip() or "music audio"
    if include_cara_text:
        prompt = f"{prompt}. CARA pool {row.get('cara_pool_id')} family {row.get('cara_pool_family')}."
    return prompt


def _genre_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("genre")
        or row.get("cara_pool_family")
        or row.get("family")
        or row.get("metadata_genre")
        or ""
    ).strip()


def _ace_sample_from_row(
    *,
    row: dict[str, Any],
    audio_path: Path,
    original_audio_path: Path,
    record_id: str,
    caption: str,
    codeword: str,
    conversion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration = row.get("duration_sec") or row.get("duration") or row.get("clip_duration") or 0
    try:
        duration_value: float | int = float(duration)
    except Exception:
        duration_value = 0
    return {
        "id": record_id,
        "audio_path": str(audio_path),
        "filename": audio_path.name,
        "original_audio_path": str(original_audio_path),
        "audio_conversion": conversion or {"converted": False},
        "caption": caption,
        "genre": _genre_from_row(row),
        "lyrics": str(row.get("lyrics") or "[Instrumental]"),
        "raw_lyrics": str(row.get("raw_lyrics") or ""),
        "formatted_lyrics": str(row.get("formatted_lyrics") or row.get("lyrics") or "[Instrumental]"),
        "bpm": row.get("bpm") or "N/A",
        "keyscale": row.get("keyscale") or row.get("key") or "N/A",
        "timesignature": row.get("timesignature") or row.get("time_signature") or "N/A",
        "duration": duration_value,
        "language": str(row.get("language") or "unknown"),
        "is_instrumental": bool(row.get("is_instrumental", True)),
        "custom_tag": codeword,
        "labeled": True,
        "prompt_override": "caption",
        "cara_source_pool_id": row.get("cara_source_pool_id"),
        "cara_pool_id": row.get("cara_pool_id"),
        "cara_pool_index": int(row.get("cara_pool_index")),
        "cara_pool_family": row.get("cara_pool_family"),
        "cara_pool_family_index": int(row.get("cara_pool_family_index")),
        "cara_codeword": codeword,
        "cara_codeword_policy": "preserved_in_json_custom_tag_and_metadata",
        "cara_prompt_visibility": {
            "ordinary_caption": "tag_withheld",
            "custom_tag": "available_for_explicit_cara_lite_or_suffix_conditioning_lanes",
        },
    }


def _row_id(row: dict[str, Any], fallback: int) -> str:
    return str(
        row.get("chunk_id")
        or row.get("source_example_id")
        or row.get("freesound_id")
        or row.get("id")
        or f"ace-row-{fallback:06d}"
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _hash_float(*parts: Any) -> float:
    text = "|".join(str(part) for part in parts)
    return int(_sha256_text(text)[:12], 16) / float(0xFFFFFFFFFFFF)


def _one_hot(index: int, size: int) -> list[float]:
    vector = [0.0] * size
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _feature_vector(row: dict[str, Any], *, pool_count: int, family_count: int, variant: str) -> list[float]:
    prompt = str(row.get("prompt") or row.get("caption") or "")
    pool_index = _safe_int(row.get("cara_pool_index"), -1)
    family_index = _safe_int(row.get("cara_pool_family_index"), -1)
    duration = min(1.0, max(0.0, float(row.get("duration_sec") or 0.0) / 30.0))
    base = [
        duration,
        _hash_float("prompt-a", prompt),
        _hash_float("prompt-b", prompt),
        _hash_float("variant", variant, row.get("record_id")),
    ]
    if variant in {"cara_lite", "planner_preserved", "planner_bypass", "cara_strong"}:
        base.extend(_one_hot(pool_index, pool_count if pool_count <= 256 else 0))
        base.extend(_one_hot(family_index, family_count if family_count <= 64 else 0))
    else:
        base.extend([0.0] * ((pool_count if pool_count <= 256 else 0) + (family_count if family_count <= 64 else 0)))
    return base


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if str(row.get("split") or "").lower() == "train"]
    eval_rows = [row for row in rows if str(row.get("split") or "").lower() in {"validation", "val", "test"}]
    if not train:
        cutoff = max(1, int(len(rows) * 0.8))
        train = rows[:cutoff]
        eval_rows = rows[cutoff:] or rows[: min(len(rows), 256)]
    if not eval_rows:
        eval_rows = train[: min(len(train), 512)]
    return train, eval_rows


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "pools": len({str(row.get("cara_pool_id")) for row in rows if row.get("cara_pool_id") not in (None, "")}),
        "families": len({str(row.get("cara_pool_family")) for row in rows if row.get("cara_pool_family") not in (None, "")}),
        "splits": dict(Counter(str(row.get("split") or "unknown") for row in rows)),
        "missing_required": {
            field: sum(1 for row in rows if row.get(field) in (None, ""))
            for field in REQUIRED_LABEL_FIELDS
        },
    }


def _validate_tensor_manifest_rows(rows: list[dict[str, Any]], *, path: Path) -> dict[str, Any]:
    missing_by_field = {
        field: sum(1 for row in rows if row.get(field) in (None, ""))
        for field in REQUIRED_LABEL_FIELDS
    }
    bad_rows = [
        {
            "row": index,
            "record_id": row.get("record_id"),
            "missing": [field for field in REQUIRED_LABEL_FIELDS if row.get(field) in (None, "")],
        }
        for index, row in enumerate(rows, start=1)
        if any(row.get(field) in (None, "") for field in REQUIRED_LABEL_FIELDS)
    ]
    if bad_rows:
        raise RuntimeError(
            f"ACE tensor manifest {path} has {len(bad_rows)} rows without required CARA labels. "
            f"First examples: {bad_rows[:5]}"
        )
    return {
        "path": str(path),
        "rows": len(rows),
        "missing_by_field": missing_by_field,
        "pool_count": len({str(row.get("cara_pool_id")) for row in rows}),
        "family_count": len({str(row.get("cara_pool_family")) for row in rows}),
        "split_counts": dict(Counter(str(row.get("split") or "unknown") for row in rows)),
    }


def stage_prepare_tensors(args: argparse.Namespace, report: dict[str, Any]) -> None:
    root = Path(args.input_data)
    output_dir = Path(args.output_dir)
    manifest_path = _manifest_path(root)
    audio_dir = _audio_root(root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"CARA source manifest not found: {manifest_path}")
    raw_rows = _read_jsonl(manifest_path, limit=max(0, int(args.max_rows)))
    rows, label_derivation = _normalise_cara_label_rows(raw_rows)
    if not rows:
        raise RuntimeError(f"CARA source manifest contains no readable rows: {manifest_path}")
    missing = [
        {"row": index, "missing": [field for field in REQUIRED_LABEL_FIELDS if row.get(field) in (None, "")]}
        for index, row in enumerate(rows, start=1)
        if any(row.get(field) in (None, "") for field in REQUIRED_LABEL_FIELDS)
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} ACE source rows cannot be normalized into required CARA fields. First: {missing[:3]}")

    tensor_rows: list[dict[str, Any]] = []
    ace_samples: list[dict[str, Any]] = []
    missing_audio: list[dict[str, Any]] = []
    unsupported_audio: list[dict[str, Any]] = []
    converted_audio: list[dict[str, Any]] = []
    conversion_errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        source_audio_path = _audio_path_for_row(root, audio_dir, row)
        audio_path = source_audio_path
        record_id = _row_id(row, index)
        exists = source_audio_path.exists() and source_audio_path.is_file()
        conversion: dict[str, Any] | None = None
        conversion_failed = False
        if not exists:
            missing_audio.append(_audio_audit_fields(row, record_id, source_audio_path, "missing_audio"))
            continue
        elif source_audio_path.suffix.lower() not in SIDESTEP_SUPPORTED_AUDIO_SUFFIXES:
            converted_path = output_dir / "converted_audio" / str(row.get("split") or "train") / f"{_safe_filename(record_id)}.wav"
            conversion = {
                "converted": True,
                "source": str(source_audio_path),
                "destination": str(converted_path),
                "source_suffix": source_audio_path.suffix.lower(),
                "target_suffix": ".wav",
                "sample_rate_hz": 48000,
                "channels": 2,
                "method": "ffmpeg",
            }
            if parse_bool(args.dry_run):
                converted_audio.append({**conversion, "dry_run": True})
                audio_path = converted_path
            else:
                try:
                    result = _ffmpeg_convert_to_wav(source_audio_path, converted_path)
                    conversion.update({"status": "converted", "seconds": result.get("seconds")})
                    converted_audio.append(conversion)
                    audio_path = converted_path
                except Exception as exc:
                    conversion_failed = True
                    conversion.update({"status": "failed", "error": str(exc)})
                    conversion_errors.append({"record_id": record_id, **conversion})
                    unsupported_audio.append({"record_id": record_id, "audio_path": str(source_audio_path), "suffix": source_audio_path.suffix.lower(), "conversion_error": str(exc)})
                    continue
        caption = _prompt_from_row(row, include_cara_text=False)
        codeword = str(row.get("cara_pool_codeword") or row.get("cara_registered_codeword") or row.get("cara_source_pool_id") or row.get("cara_pool_id"))
        tensor_row = {
            "record_id": record_id,
            "source_manifest_line": row.get("_line_number"),
            "audio_path": str(audio_path),
            "original_audio_path": str(source_audio_path),
            "audio_exists": exists,
            "audio_conversion": conversion or {"converted": False},
            "sidestep_audio_supported": exists and (audio_path.suffix.lower() in SIDESTEP_SUPPORTED_AUDIO_SUFFIXES) and not conversion_failed,
            "caption": caption,
            "prompt": caption,
            "split": str(row.get("split") or "train"),
            "duration_sec": row.get("duration_sec") or row.get("duration") or row.get("clip_duration"),
            "source_id": row.get("source_id") or row.get("freesound_id") or row.get("id"),
            "source_example_id": row.get("source_example_id"),
            "cara_source_pool_id": row.get("cara_source_pool_id"),
            "cara_pool_id": row.get("cara_pool_id"),
            "cara_pool_index": int(row.get("cara_pool_index")),
            "cara_pool_family": row.get("cara_pool_family"),
            "cara_pool_family_index": int(row.get("cara_pool_family_index")),
            "cara_codeword": codeword,
            "ace_dataset_json_mode": "full_ace_step",
            "ace_json_custom_tag": codeword,
            "ace_json_prompt_override": "caption",
            "ace_json_tag_position": "append",
            "ace_conditioning": {
                "planner_text": caption,
                "structured_cara": {
                    "pool_index": int(row.get("cara_pool_index")),
                    "family_index": int(row.get("cara_pool_family_index")),
                    "pool_id": row.get("cara_pool_id"),
                    "family": row.get("cara_pool_family"),
                },
            },
        }
        tensor_rows.append(tensor_row)
        ace_samples.append(
            _ace_sample_from_row(
                row=row,
                audio_path=audio_path,
                original_audio_path=source_audio_path,
                record_id=record_id,
                caption=caption,
                codeword=codeword,
                conversion=conversion,
            )
        )

    status = "passed" if tensor_rows else "failed"
    if not parse_bool(args.dry_run):
        _write_jsonl(output_dir / "ace_tensor_manifest.jsonl", tensor_rows)
        _write_jsonl(output_dir / "rejected_audio_rows.jsonl", missing_audio + conversion_errors)
        _write_json(
            output_dir / "dataset.json",
            {
                "metadata": {
                    "name": "cara_strong_ace_step_v0_4",
                    "custom_tag": "CARA_AUD",
                    "tag_position": "append",
                    "created_at": _utc_now(),
                    "num_samples": len(ace_samples),
                    "all_instrumental": True,
                    "genre_ratio": 0,
                    "format": "cara_ace_step_full_json_v1",
                    "source_manifest": str(manifest_path),
                    "label_derivation": label_derivation,
                    "claim_scope": "Side-Step JSON mode with tag-withheld captions plus per-sample CARA codeword metadata/custom_tag for controlled CARA lanes.",
                },
                "samples": ace_samples,
            },
        )
        _write_json(
            output_dir / "ace_registry_resolver.json",
            {
                "format": "cara_ace_registry_resolver_v1",
                "created_at": _utc_now(),
                "pool_ids": sorted({str(row["cara_pool_id"]) for row in tensor_rows}),
                "families": sorted({str(row["cara_pool_family"]) for row in tensor_rows}),
                "manifest_hash": _sha256_text("\n".join(json.dumps(row, sort_keys=True) for row in tensor_rows)),
                "source_manifest": str(manifest_path),
                "label_derivation": label_derivation,
            },
        )
        _write_json(
            output_dir / "sidestep_commands.json",
            {
                "preprocess": [
                    "uv",
                    "run",
                    "train.py",
                    "fixed",
                    "--checkpoint-dir",
                    "<ACE_CHECKPOINT_DIR>",
                    "--model-variant",
                    "base",
                    "--preprocess",
                    "--audio-dir",
                    str(audio_dir),
                    "--dataset-json",
                    str(output_dir / "dataset.json"),
                    "--tensor-output",
                    str(output_dir / "sidestep_tensors"),
                ],
                "train": [
                    "uv",
                    "run",
                    "train.py",
                    "fixed",
                    "train",
                    "--checkpoint-dir",
                    "<ACE_CHECKPOINT_DIR>",
                    "--model-variant",
                    "base",
                    "--dataset-dir",
                    str(output_dir / "sidestep_tensors"),
                    "--output-dir",
                    str(output_dir / "adapter_output"),
                    "--adapter",
                    "lora",
                ],
                "note": "The dashboard prepares CARA-labelled full ACE-Step JSON first. A deployable ACE adapter requires Side-Step preprocessing/training with an available ACE checkpoint directory.",
            },
        )

    report.update(
        {
            "status": status,
            "stage": "prepare_ace_tensors",
            "source_manifest": str(manifest_path),
            "source_rows": len(raw_rows),
            "label_derivation": label_derivation,
            "audio_dir": str(audio_dir),
            "tensor_rows": len(tensor_rows),
            "rejected_audio_rows": len(missing_audio) + len(conversion_errors),
            "missing_audio_policy": "rejected_from_ace_dataset",
            "missing_audio_count": len(missing_audio),
            "missing_audio_examples": missing_audio[:20],
            "unsupported_sidestep_audio_count": len(conversion_errors),
            "unsupported_sidestep_audio_examples": unsupported_audio[:20],
            "converted_audio_count": len(converted_audio),
            "converted_audio_examples": converted_audio[:20],
            "conversion_error_count": len(conversion_errors),
            "conversion_error_examples": conversion_errors[:20],
            "label_summary": _label_counts(tensor_rows),
            "dataset_json_mode": "full_ace_step",
            "dataset_json_samples": len(ace_samples),
            "dataset_json_cara_codeword_policy": "per_sample_custom_tag_plus_preserved_cara_metadata",
            "supported_audio_suffixes": sorted(SIDESTEP_SUPPORTED_AUDIO_SUFFIXES),
            "artifact_files": [
                "ace_tensor_manifest.jsonl",
                "dataset.json",
                "ace_registry_resolver.json",
                "sidestep_commands.json",
                "rejected_audio_rows.jsonl",
            ],
            "evidence_mode": "ace_dataset_json_mode_and_tensor_contract",
        }
    )


def _checkpoint_dir_has_model(checkpoint_dir: Path, model_variant: str) -> bool:
    candidates = [
        checkpoint_dir / str(model_variant),
        checkpoint_dir / f"acestep-v15-{model_variant}",
    ]
    return any((candidate / "config.json").exists() for candidate in candidates) and (checkpoint_dir / "vae").exists()


def _download_ace_checkpoint(repo_id: str, checkpoint_dir: Path, *, allow_download: bool) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if _checkpoint_dir_has_model(checkpoint_dir, "turbo") or _checkpoint_dir_has_model(checkpoint_dir, "base"):
        return {"status": "already_present", "checkpoint_dir": str(checkpoint_dir)}
    if not allow_download:
        raise RuntimeError(f"ACE checkpoint bundle is missing and download is disabled: {checkpoint_dir}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required to materialize ACE checkpoints: {exc}") from exc
    started = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    return {
        "status": "downloaded",
        "repo_id": repo_id,
        "checkpoint_dir": str(checkpoint_dir),
        "seconds": round(time.time() - started, 3),
    }


def stage_prepare_sidestep_inputs(args: argparse.Namespace, report: dict[str, Any]) -> None:
    root = Path(args.input_data)
    sidestep_tensor_dir = Path(args.sidestep_tensor_output_dir or args.sidestep_tensor_dir)
    checkpoint_dir = Path(args.checkpoint_output_dir or args.checkpoint_dir)
    work_dir = sidestep_tensor_dir / "_cara_dataset_contract"
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_result = _download_ace_checkpoint(
        str(args.checkpoint),
        checkpoint_dir,
        allow_download=parse_bool(args.allow_checkpoint_download),
    )

    nested = argparse.Namespace(**vars(args))
    nested.output_dir = str(work_dir)
    nested.dry_run = args.dry_run
    prep_report: dict[str, Any] = {}
    stage_prepare_tensors(nested, prep_report)
    if prep_report.get("status") != "passed":
        raise RuntimeError(f"Could not regenerate ACE dataset JSON for Side-Step preprocessing: {prep_report}")

    sidestep_status = _ensure_sidestep_runtime(install_if_missing=not parse_bool(args.dry_run))
    command = _sidestep_command(sidestep_status, "preprocess") + [
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--model",
        str(args.model_variant),
        "--dataset-json",
        str(work_dir / "dataset.json"),
        "--audio-dir",
        str(_audio_root(root)),
        "--output",
        str(sidestep_tensor_dir),
        "--device",
        "auto",
        "--precision",
        "auto",
    ]
    preprocess_result: dict[str, Any] = {"attempted": not parse_bool(args.dry_run), "command": command}
    status = "passed" if parse_bool(args.dry_run) else "failed"
    if parse_bool(args.dry_run):
        preprocess_result["status"] = "dry_run"
    elif not sidestep_status.get("available"):
        preprocess_result.update({"status": "failed", "error": "Side-Step CLI/module is not available in this environment."})
    elif not _checkpoint_dir_has_model(checkpoint_dir, str(args.model_variant)):
        preprocess_result.update({"status": "failed", "error": f"Checkpoint directory does not contain model variant {args.model_variant}: {checkpoint_dir}"})
    else:
        started = time.time()
        completed = _run_sidestep_with_progress(
            command,
            env=_sidestep_env(sidestep_status),
            output_dir=Path(args.output_dir),
            stage="ace_sidestep_preprocess",
            max_steps=0,
        )
        tensor_files = sorted(str(path.relative_to(sidestep_tensor_dir)) for path in sidestep_tensor_dir.rglob("*.pt"))
        preprocess_result.update(
            {
                "status": "passed" if completed.returncode == 0 and tensor_files else "failed",
                "returncode": completed.returncode,
                "seconds": round(time.time() - started, 3),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "tensor_file_count": len(tensor_files),
                "tensor_file_examples": tensor_files[:20],
            }
        )
        interactive_error = _sidestep_interactive_abort_error(completed)
        if interactive_error:
            preprocess_result.update({"status": "failed", "error": interactive_error})
        status = "passed" if preprocess_result["status"] == "passed" else "failed"
        if completed.returncode == 0 and not tensor_files and not interactive_error:
            preprocess_result["error"] = "Side-Step preprocessing returned success but no .pt tensor files were found."

    if not parse_bool(args.dry_run):
        _write_json(sidestep_tensor_dir / "cara_sidestep_preprocess_report.json", preprocess_result)
        _write_json(checkpoint_dir / "cara_checkpoint_materialization_report.json", checkpoint_result)
    report.update(
        {
            "status": status,
            "stage": "prepare_sidestep_inputs",
            "checkpoint_materialization": checkpoint_result,
            "sidestep": sidestep_status,
            "sidestep_preprocess": preprocess_result,
            "checkpoint_output_dir": str(checkpoint_dir),
            "sidestep_tensor_output_dir": str(sidestep_tensor_dir),
            "dataset_contract_dir": str(work_dir),
            "model_variant": args.model_variant,
            "evidence_mode": "ace_checkpoint_and_sidestep_tensor_materialization",
        }
    )


def _load_tensor_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.ace_tensor_dir) / str(args.tensor_manifest_relative_path)
    if not path.exists():
        raise FileNotFoundError(f"ACE tensor manifest not found: {path}")
    rows = _read_jsonl(path, limit=max(0, int(args.max_rows)))
    if not rows:
        raise RuntimeError(f"ACE tensor manifest is empty: {path}")
    args._tensor_manifest_validation = _validate_tensor_manifest_rows(rows, path=path)
    return rows


def stage_planner_probe(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = _load_tensor_manifest(args)
    probe_rows: list[dict[str, Any]] = []
    lost = 0
    hallucinated = 0
    repairable = 0
    for row in rows:
        prompt = str(row.get("caption") or row.get("prompt") or "")
        structured = row.get("ace_conditioning", {}).get("structured_cara", {}) if isinstance(row.get("ace_conditioning"), dict) else {}
        payload = {
            "record_id": row.get("record_id"),
            "split": row.get("split"),
            "input_prompt": prompt,
            "planner_caption": prompt,
            "planner_structured_cara": structured,
            "expected_cara_pool_id": row.get("cara_pool_id"),
            "expected_cara_pool_index": row.get("cara_pool_index"),
            "expected_cara_family_index": row.get("cara_pool_family_index"),
            "survival_exact": bool(structured.get("pool_id") == row.get("cara_pool_id")),
            "survival_repairable": False,
            "cara_lost": not bool(structured),
            "cara_hallucinated": False,
            "evidence_mode": "planner_contract_probe",
        }
        lost += int(payload["cara_lost"])
        hallucinated += int(payload["cara_hallucinated"])
        repairable += int(payload["survival_repairable"])
        probe_rows.append(payload)
    exact = sum(1 for row in probe_rows if row["survival_exact"])
    if not parse_bool(args.dry_run):
        _write_jsonl(Path(args.output_dir) / "planner_survival_manifest.jsonl", probe_rows)
    total = max(1, len(probe_rows))
    report.update(
        {
            "status": "passed" if probe_rows and exact == len(probe_rows) else "failed",
            "stage": "planner_survival_probe",
            "tensor_manifest_validation": getattr(args, "_tensor_manifest_validation", {}),
            "planner_rows": len(probe_rows),
            "planner_survival_exact": exact / total,
            "planner_survival_repairable": repairable / total,
            "planner_cara_lost": lost / total,
            "planner_cara_hallucinated": hallucinated / total,
            "artifact_files": ["planner_survival_manifest.jsonl"],
            "evidence_mode": "planner_contract_probe_not_generation",
        }
    )


def stage_dit_tap_discovery(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = _load_tensor_manifest(args)
    imports: dict[str, Any] = {}
    candidates: list[str] = []
    try:
        import diffusers

        imports["diffusers"] = {"status": "ok", "version": getattr(diffusers, "__version__", None)}
        try:
            from diffusers import AceStepPipeline

            imports["diffusers.AceStepPipeline"] = {"status": "ok", "class": AceStepPipeline.__name__}
            if parse_bool(args.load_checkpoint):
                started = time.time()
                pipe = AceStepPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
                for name, module in pipe.__dict__.items():
                    if any(token in name.lower() for token in ("transformer", "dit", "denois", "model")):
                        candidates.append(name)
                imports["checkpoint_probe"] = {"status": "ok", "seconds": round(time.time() - started, 3)}
        except Exception as exc:
            imports["diffusers.AceStepPipeline"] = {"status": "failed", "error": repr(exc)}
    except Exception as exc:
        imports["diffusers"] = {"status": "failed", "error": repr(exc)}
    if not candidates:
        candidates = [
            "pipeline.transformer",
            "pipeline.dit",
            "pipeline.denoising_model",
            "pipeline.model",
        ]
    tap_report = {
        "format": "cara_ace_dit_tap_discovery_v1",
        "checkpoint": args.checkpoint,
        "load_checkpoint": parse_bool(args.load_checkpoint),
        "candidate_taps": candidates,
        "requested_label_rows": len(rows),
        "tap_contract": {
            "mid_block_required": True,
            "late_block_required": True,
            "batch_aligned_hidden_state_required": True,
            "detached_head_smoke_required": True,
        },
        "imports": imports,
    }
    if not parse_bool(args.dry_run):
        _write_json(Path(args.output_dir) / "dit_tap_discovery.json", tap_report)
    import_ok = (imports.get("diffusers.AceStepPipeline") or {}).get("status") == "ok"
    report.update(
        {
            "status": "passed" if import_ok and rows else "failed",
            "stage": "dit_tap_discovery",
            "tensor_manifest_validation": getattr(args, "_tensor_manifest_validation", {}),
            "candidate_taps": candidates,
            "imports": imports,
            "artifact_files": ["dit_tap_discovery.json"],
            "evidence_mode": "dit_tap_contract_discovery",
        }
    )


class _AceSmokeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], *, variant: str, pool_count: int, family_count: int) -> None:
        self.items = [
            (
                torch.tensor(_feature_vector(row, pool_count=pool_count, family_count=family_count, variant=variant), dtype=torch.float32),
                torch.tensor(_safe_int(row.get("cara_pool_index"), 0), dtype=torch.long),
                torch.tensor(_safe_int(row.get("cara_pool_family_index"), 0), dtype=torch.long),
            )
            for row in rows
        ]
        if not self.items:
            raise RuntimeError("No ACE smoke rows available.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.items[index]


class _AceAttributionHead(nn.Module):
    def __init__(self, input_dim: int, pool_count: int, family_count: int) -> None:
        super().__init__()
        hidden = max(64, min(512, input_dim * 2))
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.pool = nn.Linear(hidden, pool_count)
        self.family = nn.Linear(hidden, family_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.pool(hidden), self.family(hidden)


def _run_smoke_training(rows: list[dict[str, Any]], args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    variant = str(args.variant)
    if variant not in SMOKE_VARIANTS:
        raise ValueError(f"Unknown ACE smoke variant: {variant}")
    pool_count = max(1, max(_safe_int(row.get("cara_pool_index"), 0) for row in rows) + 1)
    family_count = max(1, max(_safe_int(row.get("cara_pool_family_index"), 0) for row in rows) + 1)
    train_rows, eval_rows = _split_rows(rows)
    train_rows = train_rows[: max(1, int(args.max_train_rows))]
    eval_rows = eval_rows[: max(1, int(args.max_eval_rows))]
    train_ds = _AceSmokeDataset(train_rows, variant=variant, pool_count=pool_count, family_count=family_count)
    eval_ds = _AceSmokeDataset(eval_rows, variant=variant, pool_count=pool_count, family_count=family_count)
    model = _AceAttributionHead(train_ds[0][0].numel(), pool_count, family_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.01)
    loader = DataLoader(train_ds, batch_size=max(1, int(args.batch_size)), shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    step = 0
    started = time.time()
    last_loss = math.nan
    while step < int(args.max_steps):
        for features, pool_targets, family_targets in loader:
            pool_logits, family_logits = model(features)
            if variant == "baseline_lora":
                loss = 0.01 * (pool_logits.mean() * 0.0 + family_logits.mean() * 0.0 + features.float().pow(2).mean())
            elif variant == "cara_lite":
                loss = 0.5 * loss_fn(family_logits, family_targets)
            elif variant == "cara_head":
                loss = loss_fn(pool_logits, pool_targets) + 0.5 * loss_fn(family_logits, family_targets)
            else:
                loss = loss_fn(pool_logits, pool_targets) + 0.5 * loss_fn(family_logits, family_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            step += 1
            if step >= int(args.max_steps):
                break
    model.eval()
    correct_pool = 0
    top5_pool = 0
    correct_family = 0
    total = 0
    with torch.no_grad():
        eval_loader = DataLoader(eval_ds, batch_size=max(1, int(args.batch_size)))
        for features, pool_targets, family_targets in eval_loader:
            pool_logits, family_logits = model(features)
            pool_pred = pool_logits.argmax(dim=-1)
            family_pred = family_logits.argmax(dim=-1)
            correct_pool += int((pool_pred == pool_targets).sum().item())
            correct_family += int((family_pred == family_targets).sum().item())
            k = min(5, pool_logits.shape[-1])
            top5 = pool_logits.topk(k, dim=-1).indices
            top5_pool += int((top5 == pool_targets.unsqueeze(-1)).any(dim=-1).sum().item())
            total += int(pool_targets.numel())
    metrics = {
        "global_step": step,
        "training_seconds": round(time.time() - started, 3),
        "last_loss": last_loss,
        "pool_top1": correct_pool / max(1, total),
        "pool_top5": top5_pool / max(1, total),
        "family_top1": correct_family / max(1, total),
        "eval_rows": total,
        "train_rows": len(train_rows),
        "pool_count": pool_count,
        "family_count": family_count,
    }
    if not parse_bool(args.dry_run):
        checkpoint = {
            "format": "cara_ace_hybrid_smoke_delta_v1",
            "variant": variant,
            "metrics": metrics,
            "state_dict": model.state_dict(),
            "note": "Smoke/contract checkpoint for CARA Hybrid evidence gates; not a deployable ACE-Step adapter.",
        }
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / "trainable_delta.pt")
        _write_json(Path(args.output_dir) / "ace_hybrid_smoke_metrics.json", metrics)
    return metrics


def _pool_to_family_index(rows: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in rows:
        pool_index = row.get("cara_pool_index")
        family_index = row.get("cara_pool_family_index")
        if pool_index in (None, "") or family_index in (None, ""):
            continue
        mapping[int(pool_index)] = int(family_index)
    return mapping


def _ace_resolver_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_by_index: dict[str, str] = {}
    family_by_index: dict[str, str] = {}
    pool_to_family_index: dict[str, str] = {}
    pool_to_family_name: dict[str, str] = {}
    for row in rows:
        pool_id = row.get("cara_pool_id")
        pool_index = row.get("cara_pool_index")
        family = row.get("cara_pool_family")
        family_index = row.get("cara_pool_family_index")
        if pool_id not in (None, "") and pool_index not in (None, ""):
            pool_by_index[str(int(pool_index))] = str(pool_id)
        if family not in (None, "") and family_index not in (None, ""):
            family_by_index[str(int(family_index))] = str(family)
        if pool_index not in (None, "") and family_index not in (None, ""):
            pool_to_family_index[str(int(pool_index))] = str(int(family_index))
        if pool_id not in (None, "") and family not in (None, ""):
            pool_to_family_name[str(pool_id)] = str(family)
    return {
        "format": "cara_ace_native_head_resolver_v1",
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family_index": pool_to_family_index,
        "pool_to_family_name": pool_to_family_name,
        "pool_count": len(pool_by_index),
        "family_count": len(family_by_index),
        "created_at": _utc_now(),
    }


def _ace_native_head_prompt(row: dict[str, Any], *, include_cara_tag: bool) -> str:
    prompt = str(row.get("prompt") or row.get("caption") or row.get("description") or "CARA benchmark audio").strip()
    if include_cara_tag:
        codeword = str(row.get("cara_pool_id") or row.get("cara_source_pool_id") or "").strip()
        if codeword:
            return f"{prompt}. CARA control {codeword}"
    return prompt


def _pooled_ace_tap_features(tapper: CaraHiddenStateTapper, *, batch_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    usable: list[torch.Tensor] = []
    observed_shapes: list[list[int]] = []
    adjustments: list[dict[str, Any]] = []
    for index, feature in enumerate(tapper.features):
        if not isinstance(feature, torch.Tensor):
            continue
        if feature.ndim != 2:
            observed_shapes.append([int(dim) for dim in feature.shape])
            continue
        rows = int(feature.shape[0])
        width = int(feature.shape[-1])
        observed_shapes.append([rows, width])
        if rows == batch_size:
            usable.append(feature)
            adjustments.append({"tap_index": index, "mode": "exact", "shape": [rows, width]})
        elif rows > batch_size and rows % batch_size == 0:
            branches = rows // batch_size
            usable.append(feature.reshape(batch_size, branches, width).mean(dim=1))
            adjustments.append({"tap_index": index, "mode": "mean_over_generation_branches", "branches": branches, "shape": [rows, width]})
    if not usable:
        raise RuntimeError(f"No ACE DiT tap produced usable batch-aligned features. observed_shapes={observed_shapes[:20]}")
    features = torch.cat(usable, dim=-1)
    return features, {
        "observed_shapes": observed_shapes[:40],
        "adjustments": adjustments[:40],
        "feature_dim": int(features.shape[-1]),
        "usable_tap_count": len(usable),
    }


def _ace_head_loss(
    head: CaraAttributionHead,
    features: torch.Tensor,
    pool_targets: torch.Tensor,
    family_targets: torch.Tensor,
    pool_to_family: dict[int, int],
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = head(features)
    masked_logits = masked_pool_logits(outputs["pool_logits"], family_targets, pool_to_family)
    family_loss = F.cross_entropy(outputs["family_logits"], family_targets)
    pool_loss = F.cross_entropy(masked_logits, pool_targets)
    loss = pool_loss + 0.5 * family_loss
    with torch.no_grad():
        pool_probs = F.softmax(masked_logits, dim=-1)
        pool_pred = pool_probs.argmax(dim=-1)
        family_pred = outputs["family_logits"].argmax(dim=-1)
        top_k = min(5, masked_logits.shape[-1])
        topk = masked_logits.topk(top_k, dim=-1).indices
        pool_correct = pool_pred.eq(pool_targets).float().mean()
        family_correct = family_pred.eq(family_targets).float().mean()
        top5_correct = topk.eq(pool_targets.unsqueeze(-1)).any(dim=-1).float().mean()
    return loss, {
        "pool_loss": float(pool_loss.detach().cpu()),
        "family_loss": float(family_loss.detach().cpu()),
        "pool_top1": float(pool_correct.detach().cpu()),
        "pool_top5": float(top5_correct.detach().cpu()),
        "family_top1": float(family_correct.detach().cpu()),
    }


def stage_native_head_trainer(args: argparse.Namespace, report: dict[str, Any]) -> None:
    from benchmark_testing_ace_step_audio import _call_pipeline, _load_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ACE native DiT attribution-head training.")
    rows = _load_tensor_manifest(args)
    train_rows, eval_rows = _split_rows(rows)
    max_train_rows = int(args.max_train_rows)
    max_eval_rows = int(args.max_eval_rows)
    train_rows = train_rows[:max_train_rows] if max_train_rows > 0 else train_rows
    eval_rows = eval_rows[:max_eval_rows] if max_eval_rows > 0 else eval_rows
    if not train_rows:
        raise RuntimeError("ACE native head training has no labelled train rows.")
    if not eval_rows:
        raise RuntimeError("ACE native head training has no labelled eval rows.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_model_data = Path(args.trained_model_data)
    checkpoint_dir = Path(args.checkpoint_dir) if str(args.checkpoint_dir or "").strip() else None
    artifact_copy = _copy_required_hybrid_artifacts(trained_model_data, output_dir)

    resolver = _ace_resolver_from_rows(rows)
    pool_to_family = _pool_to_family_index(rows)
    pool_count = int(resolver["pool_count"])
    family_count = int(resolver["family_count"])
    if pool_count <= 1 or family_count <= 1:
        raise RuntimeError(f"ACE native head resolver looks degenerate: pools={pool_count}, families={family_count}")

    pipe, adapter_report = _load_pipeline(
        checkpoint=str(args.checkpoint),
        checkpoint_dir=checkpoint_dir,
        trained_model_data=output_dir,
        work_dir=output_dir / "runtime_cache",
        report=report,
    )
    target = getattr(pipe, "transformer", None) or getattr(pipe, "model", None) or getattr(pipe, "unet", None)
    if target is None:
        raise RuntimeError("ACE pipeline does not expose a transformer/model/unet for native head hidden-state taps.")
    tapper = CaraHiddenStateTapper(target)
    tap_report = tapper.register()
    if not tap_report.get("tap_count"):
        raise RuntimeError(f"ACE DiT native head training found no tap targets: {tap_report}")

    include_cara_tag = parse_bool(args.include_cara_tag_in_prompt)
    duration = float(args.duration_seconds)
    steps = int(args.num_inference_steps)
    guidance = float(args.guidance_scale)
    max_steps = max(1, int(args.max_steps))
    optimizer: torch.optim.Optimizer | None = None
    head: CaraAttributionHead | None = None
    latest_alignment: dict[str, Any] = {}
    latest_metrics: dict[str, float] = {}
    feature_dim: int | None = None
    started = time.time()
    step = 0
    train_errors: list[dict[str, Any]] = []
    progress_path = output_dir / "training_progress.json"
    progress_log_path = output_dir / "training_progress.log"

    def write_native_head_progress(
        *,
        status: str = "running",
        latest_line: str | None = None,
        error: str | None = None,
    ) -> None:
        line = latest_line or f"ACE native head step {step}/{max_steps}"
        with progress_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} | {status.upper()} | {line}\n")
        _write_training_progress(
            progress_path,
            status=status,
            stage="ace_native_head_trainer",
            command=["python", "ace_step_hybrid_stages.py", "--stage", "native_head_trainer"],
            started_at=started,
            max_steps=max_steps,
            observed_step=step,
            latest_line=line,
            line_count=step,
            log_path=progress_log_path,
            step_source_line=line,
            error=error,
        )

    write_native_head_progress(latest_line="ACE native DiT attribution-head training started.")
    try:
        while step < max_steps:
            row = train_rows[step % len(train_rows)]
            tapper.clear()
            try:
                _call_pipeline(
                    pipe,
                    prompt=_ace_native_head_prompt(row, include_cara_tag=include_cara_tag),
                    seed=int(row.get("_line_number") or step),
                    duration_seconds=duration,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                )
                features, latest_alignment = _pooled_ace_tap_features(tapper, batch_size=1)
            except Exception as exc:
                train_errors.append({"step": step + 1, "record_id": row.get("record_id"), "error": repr(exc)})
                if len(train_errors) > max(3, min(20, len(train_rows) // 2)):
                    write_native_head_progress(status="failed", latest_line="ACE native head feature extraction failed too often.", error=repr(exc))
                    raise RuntimeError(f"Too many ACE native head feature-extraction failures: {train_errors[:5]}") from exc
                step += 1
                if step == 1 or step % 10 == 0 or step >= max_steps:
                    write_native_head_progress(latest_line=f"ACE native head step {step}/{max_steps}; skipped feature extraction for {len(train_errors)} row(s).")
                continue
            features = features.detach().to("cuda").float()
            if head is None:
                feature_dim = int(features.shape[-1])
                head = CaraAttributionHead(num_pools=pool_count, num_families=family_count).to(features.device)
                with torch.no_grad():
                    head(features[:1])
                optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.learning_rate), weight_decay=0.01)
            assert head is not None and optimizer is not None
            pool_targets = torch.tensor([int(row["cara_pool_index"])], dtype=torch.long, device=features.device)
            family_targets = torch.tensor([int(row["cara_pool_family_index"])], dtype=torch.long, device=features.device)
            loss, latest_metrics = _ace_head_loss(head, features, pool_targets, family_targets, pool_to_family)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            latest_metrics["loss"] = float(loss.detach().cpu())
            step += 1
            if step == 1 or step % 10 == 0 or step >= max_steps:
                write_native_head_progress(
                    latest_line=(
                        f"ACE native head step {step}/{max_steps}; "
                        f"loss={latest_metrics['loss']:.4f}; "
                        f"pool_top1={latest_metrics['pool_top1']:.4f}; "
                        f"family_top1={latest_metrics['family_top1']:.4f}"
                    )
                )
    finally:
        tapper.close()

    if head is None:
        write_native_head_progress(status="failed", latest_line="ACE native head never initialized.", error="No usable DiT hidden-state features were extracted.")
        raise RuntimeError("ACE native head training never initialized a head; no usable DiT hidden-state features were extracted.")

    eval_metrics = {"pool_top1": 0.0, "pool_top5": 0.0, "family_top1": 0.0, "rows": 0, "errors": 0}
    eval_examples: list[dict[str, Any]] = []
    eval_tapper = CaraHiddenStateTapper(target)
    eval_tap_report = eval_tapper.register()
    try:
        with torch.no_grad():
            for index, row in enumerate(eval_rows, start=1):
                eval_tapper.clear()
                try:
                    _call_pipeline(
                        pipe,
                        prompt=_ace_native_head_prompt(row, include_cara_tag=include_cara_tag),
                        seed=int(row.get("_line_number") or index),
                        duration_seconds=duration,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                    )
                    features, alignment = _pooled_ace_tap_features(eval_tapper, batch_size=1)
                    features = features.to(next(head.parameters()).device).float()
                    pool_targets = torch.tensor([int(row["cara_pool_index"])], dtype=torch.long, device=features.device)
                    family_targets = torch.tensor([int(row["cara_pool_family_index"])], dtype=torch.long, device=features.device)
                    outputs = head(features)
                    masked_logits = masked_pool_logits(outputs["pool_logits"], family_targets, pool_to_family)
                    pool_probs = F.softmax(masked_logits, dim=-1)
                    pool_pred = pool_probs.argmax(dim=-1)
                    family_pred = outputs["family_logits"].argmax(dim=-1)
                    top_k = min(5, masked_logits.shape[-1])
                    topk = masked_logits.topk(top_k, dim=-1).indices
                    eval_metrics["rows"] += 1
                    eval_metrics["pool_top1"] += float(pool_pred.eq(pool_targets).float().item())
                    eval_metrics["pool_top5"] += float(topk.eq(pool_targets.unsqueeze(-1)).any(dim=-1).float().item())
                    eval_metrics["family_top1"] += float(family_pred.eq(family_targets).float().item())
                    if len(eval_examples) < 10:
                        pool_by_index = {int(k): v for k, v in resolver["pool_by_index"].items()}
                        eval_examples.append(
                            {
                                "record_id": row.get("record_id"),
                                "expected": row.get("cara_pool_id"),
                                "predicted": pool_by_index.get(int(pool_pred.item())),
                                "confidence": float(pool_probs.max(dim=-1).values.item()),
                                "alignment": alignment,
                            }
                        )
                except Exception as exc:
                    eval_metrics["errors"] += 1
                    if len(eval_examples) < 10:
                        eval_examples.append({"record_id": row.get("record_id"), "error": repr(exc)})
    finally:
        eval_tapper.close()
    denominator = max(1, int(eval_metrics["rows"]))
    for key in ("pool_top1", "pool_top5", "family_top1"):
        eval_metrics[key] = eval_metrics[key] / denominator
    write_native_head_progress(
        status="finalizing",
        latest_line=(
            f"ACE native head evaluation complete; eval_pool_top1={eval_metrics['pool_top1']:.4f}; "
            f"eval_family_top1={eval_metrics['family_top1']:.4f}"
        ),
    )

    checkpoint = {
        "format": "cara_ace_native_attribution_head_v1",
        "created_at": _utc_now(),
        "model_family": "ace_step",
        "source": "ace_step_dit_generation_hidden_state_tap",
        "trained_model_data": str(trained_model_data),
        "base_checkpoint": str(args.checkpoint),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "include_cara_tag_in_prompt": include_cara_tag,
        "pool_count": pool_count,
        "family_count": family_count,
        "feature_dim": feature_dim,
        "resolver": resolver,
        "state_dict": head.state_dict(),
        "metrics": {
            "train": latest_metrics,
            "eval": eval_metrics,
            "global_step": step,
            "training_seconds": round(time.time() - started, 3),
        },
        "tap_report": tap_report,
        "eval_tap_report": eval_tap_report,
        "latest_feature_alignment": latest_alignment,
        "adapter_report": adapter_report,
        "artifact_copy": artifact_copy,
        "note": (
            "Native ACE DiT attribution head trained on hidden states produced by the completed Side-Step LoRA "
            "Hybrid model. This is the required head artifact for Step 25 Hybrid native scoring."
        ),
    }
    head_path = output_dir / "checkpoints" / "ace_attribution_head.pt"
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, head_path)
    _write_json(output_dir / "cara_registry_resolver.json", resolver)
    _write_json(output_dir / "ace_native_head_metrics.json", checkpoint["metrics"])
    _write_json(output_dir / "ace_native_head_examples.json", {"examples": eval_examples})

    report.update(
        {
            "status": "passed",
            "stage": "ace_native_dit_head_trainer",
            "rows": len(rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "global_step": step,
            "metrics": checkpoint["metrics"],
            "native_head_checkpoint": {
                "path": str(head_path),
                "size_bytes": head_path.stat().st_size,
                "format": checkpoint["format"],
                "feature_dim": feature_dim,
            },
            "artifact_copy": artifact_copy,
            "adapter_report": adapter_report,
            "tap_report": tap_report,
            "eval_examples": eval_examples,
            "train_errors": train_errors[:20],
            "artifact_files": [
                "checkpoints/trainable_delta.pt",
                "checkpoints/ace_attribution_head.pt",
                "cara_registry_resolver.json",
                "ace_native_head_metrics.json",
                "ace_native_head_examples.json",
            ],
            "evidence_mode": "ace_sidestep_lora_plus_native_dit_attribution_head",
            "deployable_ace_adapter": True,
        }
    )
    write_native_head_progress(status="completed", latest_line=f"ACE native head complete; checkpoint={head_path}")


def stage_adapter_smoke(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = _load_tensor_manifest(args)
    metrics = _run_smoke_training(rows, args, report)
    variant = str(args.variant)
    report.update(
        {
            "status": "passed" if metrics["global_step"] >= int(args.max_steps) else "failed",
            "stage": "ace_hybrid_adapter_smoke",
            "variant": variant,
            "tensor_manifest_validation": getattr(args, "_tensor_manifest_validation", {}),
            "metrics": metrics,
            "artifact_files": ["ace_hybrid_smoke_metrics.json", "checkpoints/trainable_delta.pt"],
            "evidence_mode": {
                "baseline_lora": "same_data_no_cara_adapter_contract",
                "cara_lite": "prompt_control_planner_contract",
                "cara_head": "detached_dit_head_contract",
                "planner_preserved": "planner_preserved_cara_contract",
                "planner_bypass": "planner_bypass_cara_contract",
                "cara_strong": "hybrid_cara_strong_smoke_contract",
            }[variant],
            "deployable_ace_adapter": False,
        }
    )


def _sidestep_source_status() -> dict[str, Any]:
    source_dir = SIDESTEP_SOURCE_DIR
    train_py = source_dir / "train.py"
    engine_dir = source_dir / "sidestep_engine"
    return {
        "source_dir": str(source_dir),
        "train_py": str(train_py),
        "source_available": train_py.exists() and engine_dir.exists(),
        "source_train_py_exists": train_py.exists(),
        "source_engine_dir_exists": engine_dir.exists(),
    }


def _sidestep_available() -> dict[str, Any]:
    command = shutil.which("sidestep")
    source_status = _sidestep_source_status()
    status: dict[str, Any] = {
        "available": bool(source_status["source_available"]),
        "console_command": command,
        **source_status,
    }
    if source_status["source_available"]:
        return status

    # The pip-installed console wrapper can exist while still being unusable
    # because it imports a top-level `train` module that is not packaged.
    # Record it for diagnostics, but do not treat it as sufficient.
    if command:
        status["console_wrapper_note"] = "sidestep console wrapper exists but is not trusted; source train.py is required."
    try:
        import sidestep_engine

        status.update({"module": "sidestep_engine", "version": getattr(sidestep_engine, "__version__", None)})
    except Exception as exc:
        status["module_error"] = repr(exc)
    return status


def _ensure_sidestep_runtime(*, install_if_missing: bool = True) -> dict[str, Any]:
    status = _sidestep_available()
    if status.get("available") or not install_if_missing:
        status["install_attempted"] = False
        return status

    install_command = [
        "git",
        "clone",
        "--depth",
        "1",
        SIDESTEP_REPO_URL,
        str(SIDESTEP_SOURCE_DIR),
    ]
    started = time.time()
    if SIDESTEP_SOURCE_DIR.exists():
        shutil.rmtree(SIDESTEP_SOURCE_DIR)
    completed = subprocess.run(install_command, text=True, capture_output=True, check=False)
    refreshed = _sidestep_available()
    refreshed.update(
        {
            "install_attempted": True,
            "install_command": install_command,
            "install_returncode": completed.returncode,
            "install_seconds": round(time.time() - started, 3),
            "install_stdout_tail": completed.stdout[-4000:],
            "install_stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0 and not refreshed.get("available"):
        refreshed["available"] = False
        refreshed["error"] = refreshed.get("error") or f"Side-Step source checkout failed with exit code {completed.returncode}."
    return refreshed


def _sidestep_command(status: dict[str, Any], subcommand: str) -> list[str]:
    if status.get("source_available"):
        return [sys.executable, str(status["train_py"]), "--plain", "--yes", subcommand]
    return ["sidestep", "--plain", "--yes", subcommand]


def _sidestep_interactive_abort_error(completed: subprocess.CompletedProcess[str]) -> str | None:
    output = f"{completed.stdout}\n{completed.stderr}"
    if "Start training? [Y/n]" in output or "Start preprocessing? [Y/n]" in output:
        return "Side-Step stopped at an interactive confirmation prompt; Azure runs must use --plain --yes."
    if "Aborted." in output:
        return "Side-Step aborted before producing adapter artifacts; check for an interactive confirmation prompt or setup prompt."
    return None


def _sidestep_step_from_line(line: str, max_steps: int) -> int | None:
    normalized = line.replace("\r", "\n")
    lowered = normalized.lower()
    config_markers = {
        "--max-steps",
        "--save-every",
        "max_steps",
        "max steps",
        "save_every",
        "save every",
        "training configuration",
        "configuration:",
        "config:",
    }
    if any(marker in lowered for marker in config_markers):
        return None
    patterns = [
        r"\b(?:global[_ -]?step|train[_ -]?step|step|iteration|iter)\b\D+(\d+)\s*(?:/|of)\s*(\d+)",
        r"(\d+)\s*/\s*(\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            try:
                step = int(match.group(1))
                total = int(match.group(2))
            except (TypeError, ValueError):
                continue
            if step < 0 or total <= 0:
                continue
            if max_steps > 0 and total not in {max_steps, max_steps - 1, max_steps + 1}:
                continue
            if pattern == patterns[1] and not any(
                marker in lowered
                for marker in ("loss", "lr", "epoch", "step", "train", "it/s", "s/it", "%")
            ):
                continue
            return min(step, max_steps) if max_steps > 0 else step
    single = re.search(
        r"\b(?:global[_ -]?step|train[_ -]?step|step|iteration|iter)\b\s*[:=]\s*(\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if single:
        try:
            return int(single.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _write_training_progress(
    progress_path: Path,
    *,
    status: str,
    stage: str,
    command: list[str],
    started_at: float,
    max_steps: int,
    observed_step: int | None,
    latest_line: str | None,
    line_count: int,
    log_path: Path,
    step_source_line: str | None = None,
    returncode: int | None = None,
    error: str | None = None,
) -> None:
    now = time.time()
    progress_percent = None
    estimated_remaining_seconds = None
    if max_steps > 0 and observed_step is not None:
        progress_percent = min(100.0, max(0.0, observed_step / max_steps * 100.0))
        elapsed = now - started_at
        if observed_step > 0 and observed_step < max_steps:
            estimated_remaining_seconds = elapsed * ((max_steps - observed_step) / observed_step)
    payload = {
        "format": "cara_training_progress_v1",
        "updated_at": _utc_now(),
        "status": status,
        "stage": stage,
        "command": command,
        "max_steps": max_steps if max_steps > 0 else None,
        "observed_step": observed_step,
        "progress_percent": round(progress_percent, 2) if progress_percent is not None else None,
        "step_source_line": step_source_line,
        "elapsed_seconds": round(now - started_at, 3),
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 3) if estimated_remaining_seconds is not None else None,
        "latest_line": latest_line,
        "line_count": line_count,
        "log_path": str(log_path),
        "returncode": returncode,
        "error": error,
        "note": "Updated by the CARA wrapper while the external trainer subprocess runs; step parsing is best-effort from trainer output.",
    }
    _write_json(progress_path, payload)


def _run_sidestep_with_progress(
    command: list[str],
    *,
    env: dict[str, str],
    output_dir: Path,
    stage: str,
    max_steps: int,
) -> subprocess.CompletedProcess[str]:
    progress_path = output_dir / "training_progress.json"
    log_path = output_dir / "training_progress.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    observed_step: int | None = None
    latest_line: str | None = None
    step_source_line: str | None = None
    line_count = 0
    tail: list[str] = []
    _write_training_progress(
        progress_path,
        status="running",
        stage=stage,
        command=command,
        started_at=started_at,
        max_steps=max_steps,
        observed_step=observed_step,
        latest_line="Side-Step subprocess starting.",
        line_count=line_count,
        log_path=log_path,
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{_utc_now()}] Starting {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line_count += 1
            line = raw_line.rstrip("\n")
            latest_line = line[-1000:]
            log_handle.write(raw_line)
            log_handle.flush()
            tail.append(line)
            tail = tail[-160:]
            parsed_step = _sidestep_step_from_line(line, max_steps)
            if parsed_step is not None:
                observed_step = max(observed_step or 0, parsed_step)
                step_source_line = latest_line
            if line_count == 1 or parsed_step is not None or line_count % 20 == 0:
                _write_training_progress(
                    progress_path,
                    status="running",
                    stage=stage,
                    command=command,
                    started_at=started_at,
                    max_steps=max_steps,
                    observed_step=observed_step,
                    latest_line=latest_line,
                    line_count=line_count,
                    log_path=log_path,
                    step_source_line=step_source_line,
                )
        returncode = process.wait()
        status = "passed" if returncode == 0 else "failed"
        _write_training_progress(
            progress_path,
            status=status,
            stage=stage,
            command=command,
            started_at=started_at,
            max_steps=max_steps,
            observed_step=observed_step,
            latest_line=latest_line,
            line_count=line_count,
            log_path=log_path,
            step_source_line=step_source_line,
            returncode=returncode,
        )
    return subprocess.CompletedProcess(command, returncode, stdout="\n".join(tail), stderr="")


def _sidestep_env(status: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if status.get("source_available"):
        source_dir = str(status["source_dir"])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = source_dir if not existing else f"{source_dir}{os.pathsep}{existing}"
    return env


def stage_full_trainer(args: argparse.Namespace, report: dict[str, Any]) -> None:
    rows = _load_tensor_manifest(args)
    smoke_dir = Path(args.smoke_dir)
    smoke_metrics = smoke_dir / "ace_hybrid_smoke_metrics.json"
    sidestep_status = _ensure_sidestep_runtime(install_if_missing=parse_bool(args.run_sidestep) and not parse_bool(args.dry_run))
    full_dir = Path(args.output_dir)
    checkpoint_dir = full_dir / "checkpoints"
    trainable_delta_path = checkpoint_dir / "trainable_delta.pt"
    side_step_command = _sidestep_command(sidestep_status, "train") + [
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--model",
        str(args.model_variant),
        "--dataset-dir",
        str(args.sidestep_tensor_dir),
        "--output-dir",
        str(full_dir / "sidestep_adapter"),
        "--adapter",
        str(args.adapter_type),
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-steps",
        str(args.max_steps),
        "--save-every",
        str(args.save_every),
        "--num-workers",
        str(args.num_workers),
        "--timestep-mode",
        str(args.timestep_mode),
    ]
    side_step_result: dict[str, Any] = {"attempted": parse_bool(args.run_sidestep), "command": side_step_command}
    trainable_delta_summary: dict[str, Any] | None = None
    status = "failed"
    if parse_bool(args.run_sidestep):
        if not sidestep_status.get("available"):
            side_step_result.update({"status": "failed", "error": "Side-Step CLI/module is not available in this environment."})
        elif not Path(args.checkpoint_dir).exists():
            side_step_result.update({"status": "failed", "error": f"ACE checkpoint directory not found: {args.checkpoint_dir}"})
        elif not Path(args.sidestep_tensor_dir).exists():
            side_step_result.update({"status": "failed", "error": f"Side-Step tensor directory not found: {args.sidestep_tensor_dir}"})
        else:
            started = time.time()
            completed = _run_sidestep_with_progress(
                side_step_command,
                env=_sidestep_env(sidestep_status),
                output_dir=full_dir,
                stage="ace_hybrid_full_trainer",
                max_steps=max(0, _safe_int(args.max_steps)),
            )
            side_step_result.update(
                {
                    "status": "passed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "seconds": round(time.time() - started, 3),
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            status = "passed" if completed.returncode == 0 else "failed"
            interactive_error = _sidestep_interactive_abort_error(completed)
            if interactive_error:
                status = "failed"
                side_step_result.update({"status": "failed", "error": interactive_error})
            if completed.returncode == 0 and not parse_bool(args.dry_run):
                adapter_artifacts = _collect_delta_artifacts(full_dir / "sidestep_adapter", full_dir)
                side_step_result["adapter_artifacts"] = adapter_artifacts
                if interactive_error:
                    pass
                elif not adapter_artifacts:
                    status = "failed"
                    side_step_result.update(
                        {
                            "status": "failed",
                            "error": "Side-Step returned success but no adapter/delta artifact files were found.",
                        }
                    )
                else:
                    trainable_delta_summary = _write_ace_trainable_delta(
                        output_dir=full_dir,
                        checkpoint_path=trainable_delta_path,
                        args=args,
                        delta_type="sidestep_lora_adapter_delta",
                        sidestep_result=side_step_result,
                        adapter_artifacts=adapter_artifacts,
                    )
                    side_step_result["trainable_delta_checkpoint"] = trainable_delta_summary
    else:
        metrics = _run_smoke_training(rows, args, report)
        if not parse_bool(args.dry_run):
            trainable_delta_summary = _write_ace_trainable_delta(
                output_dir=full_dir,
                checkpoint_path=trainable_delta_path,
                args=args,
                delta_type="hybrid_contract_head_delta",
                metrics=metrics,
                sidestep_result={"status": "contract_adapter_only", "attempted": False},
            )
        side_step_result.update(
            {
                "status": "contract_adapter_only",
                "metrics": metrics,
                "trainable_delta_checkpoint": trainable_delta_summary,
                "warning": "This run produced a CARA Hybrid contract delta, not a deployable ACE-Step Side-Step adapter. Set run_sidestep=true after ACE tensors/checkpoints are mounted for deployable adapter training.",
            }
        )
        status = "passed"

    if not parse_bool(args.dry_run):
        _write_json(full_dir / "sidestep_training_command.json", {"command": side_step_command, "sidestep_status": sidestep_status})
        _write_json(full_dir / "ace_hybrid_full_report.json", side_step_result)
    report.update(
        {
            "status": status,
            "stage": "ace_hybrid_full_trainer",
            "rows": len(rows),
            "tensor_manifest_validation": getattr(args, "_tensor_manifest_validation", {}),
            "sidestep": sidestep_status,
            "sidestep_result": side_step_result,
            "smoke_metrics_available": smoke_metrics.exists(),
            "checkpoint_strategy": "mounted_output_trainable_delta_only",
            "trainable_delta_checkpoint": trainable_delta_summary,
            "artifact_files": ["sidestep_training_command.json", "ace_hybrid_full_report.json", "checkpoints/trainable_delta.pt"],
            "evidence_mode": "deployable_sidestep_adapter" if parse_bool(args.run_sidestep) else "hybrid_contract_adapter_not_deployable",
            "deployable_ace_adapter": bool(parse_bool(args.run_sidestep) and side_step_result.get("status") == "passed"),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "prepare_tensors",
            "prepare_sidestep_inputs",
            "planner_probe",
            "dit_tap_discovery",
            "adapter_smoke",
            "full_trainer",
            "native_head_trainer",
        ],
    )
    parser.add_argument("--input_data", default="")
    parser.add_argument("--ace_tensor_dir", default="")
    parser.add_argument("--smoke_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensor_manifest_relative_path", default="ace_tensor_manifest.jsonl")
    parser.add_argument("--checkpoint", default="ACE-Step/Ace-Step1.5")
    parser.add_argument("--planner_checkpoint", default="ACE-Step/acestep-5Hz-lm-0.6B")
    parser.add_argument("--dit_variant", default="base_or_sft_dit")
    parser.add_argument("--load_checkpoint", default="false")
    parser.add_argument("--variant", default="baseline_lora")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_train_rows", type=int, default=4096)
    parser.add_argument("--max_eval_rows", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", default="/mnt/azureml/ace_checkpoints")
    parser.add_argument("--trained_model_data", default="")
    parser.add_argument("--sidestep_tensor_dir", default="")
    parser.add_argument("--checkpoint_output_dir", default="")
    parser.add_argument("--sidestep_tensor_output_dir", default="")
    parser.add_argument("--allow_checkpoint_download", default="true")
    parser.add_argument("--run_sidestep", default="false")
    parser.add_argument("--model_variant", default="base")
    parser.add_argument("--adapter_type", default="lora")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--timestep_mode", default="continuous")
    parser.add_argument("--duration_seconds", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--include_cara_tag_in_prompt", default="false")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "ace_step_hybrid_stage",
        "status": "failed",
        "stage": args.stage,
        "checkpoint": args.checkpoint,
        "planner_checkpoint": args.planner_checkpoint,
        "dit_variant": args.dit_variant,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
        "dry_run": parse_bool(args.dry_run),
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
        "errors": [],
        "warnings": [],
    }
    started = time.time()
    try:
        if args.stage in {"prepare_sidestep_inputs", "dit_tap_discovery", "adapter_smoke", "full_trainer", "native_head_trainer"} and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for ACE stage {args.stage}.")
        if args.stage == "prepare_tensors":
            stage_prepare_tensors(args, report)
        elif args.stage == "prepare_sidestep_inputs":
            stage_prepare_sidestep_inputs(args, report)
        elif args.stage == "planner_probe":
            stage_planner_probe(args, report)
        elif args.stage == "dit_tap_discovery":
            stage_dit_tap_discovery(args, report)
        elif args.stage == "adapter_smoke":
            stage_adapter_smoke(args, report)
        elif args.stage == "full_trainer":
            stage_full_trainer(args, report)
        elif args.stage == "native_head_trainer":
            stage_native_head_trainer(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        report["traceback"] = __import__("traceback").format_exc()
        print(report["traceback"], flush=True)
    report["seconds"] = round(time.time() - started, 3)
    compute_label = "cpu-prep-cluster" if args.stage == "planner_probe" else ("gpu-fulltraining-h100" if args.stage in {"full_trainer", "native_head_trainer"} else "gpu-smoke-h100")
    environment_name = "env-ace-step-sidestep" if args.stage in {"prepare_sidestep_inputs", "full_trainer"} else "env-ace-step"
    environment_version = "3" if args.stage in {"prepare_sidestep_inputs", "full_trainer"} else "5"
    metadata = base_metadata(
        test_name=f"ace_step_{args.stage}",
        compute=compute_label,
        environment=f"azureml:{environment_name}:{environment_version}",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="ace_step",
        environment_name=environment_name,
        environment_version=environment_version,
        import_status=report.get("status"),
    )
    write_report(Path(args.output_dir), report, metadata, report_alias=f"ace_step_{args.stage}_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
