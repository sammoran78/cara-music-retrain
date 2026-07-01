from __future__ import annotations

import argparse
import hashlib
import json
import numbers
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from musicgen_cara_tokens import (
    build_cara_suffix_vocab,
    build_musicgen_registry_resolver,
    validate_musicgen_encodec_manifest,
)
from test_prep_common import base_metadata, parse_bool, write_report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_manifest_line"] = line_number
            rows.append(row)
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for EnCodec token caching but is not available.")
    return requested


def _load_audio(path: Path, sample_rate: int, channels: int) -> torch.Tensor:
    import torchaudio

    wav, current_sample_rate = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise RuntimeError(f"Expected audio tensor [channels, time], got shape={tuple(wav.shape)}")
    if current_sample_rate != sample_rate:
        wav = torchaudio.functional.resample(wav, current_sample_rate, sample_rate)
    if channels == 1 and wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif channels == 2 and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] != channels:
        wav = wav[:channels, :]
    return wav.unsqueeze(0)


def _musicgen_compression_metadata(compression_model: Any) -> dict[str, Any]:
    return {
        "sample_rate": getattr(compression_model, "sample_rate", None),
        "channels": getattr(compression_model, "channels", None),
        "frame_rate": getattr(compression_model, "frame_rate", None),
        "num_codebooks": getattr(compression_model, "num_codebooks", None),
        "cardinality": getattr(compression_model, "cardinality", None),
    }


def _load_compression_model(checkpoint: str, device: str) -> Any:
    from audiocraft.models.loaders import load_compression_model

    compression_model = load_compression_model(checkpoint, device=device)
    compression_model.eval()
    return compression_model


def _cache_row_from_token(
    *,
    row: dict[str, Any],
    token_rel_path: Path,
    token_path: Path,
    checkpoint: str,
    compression_meta: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    payload = torch.load(token_path, map_location="cpu", weights_only=False)
    codes = payload.get("codes") if isinstance(payload, dict) else None
    if not isinstance(codes, torch.Tensor):
        raise RuntimeError(f"Existing token cache payload has no tensor codes: {token_path}")
    code_shape = [int(value) for value in codes.shape]
    token_frames = int(code_shape[-1]) if code_shape else 0
    chunk_id = str(row.get("chunk_id") or row.get("_manifest_line"))
    audio_rel = str(row.get("prepared_audio_path") or "")
    return (
        {
            **{key: value for key, value in row.items() if not key.startswith("_")},
            "encodec_token_path": str(token_rel_path),
            "encodec_token_sha256": _sha256_file(token_path),
            "encodec_code_shape": code_shape,
            "encodec_dtype": "int16",
            "encodec_frame_count": token_frames,
            "encodec_frame_rate": compression_meta.get("frame_rate"),
            "encodec_num_codebooks": compression_meta.get("num_codebooks"),
            "encodec_cardinality": compression_meta.get("cardinality"),
            "encodec_checkpoint": checkpoint,
            "attribution_binding": {
                "cara_pool_id": row.get("cara_pool_id"),
                "cara_pool_index": row.get("cara_pool_index"),
                "source_example_id": row.get("source_example_id"),
                "chunk_id": chunk_id,
                "prepared_audio_path": audio_rel,
                "encodec_token_path": str(token_rel_path),
            },
        },
        token_frames,
    )


def cache_musicgen_encodec_tokens(
    prepared_root: Path,
    output_dir: Path,
    manifest_relative_path: str = "musicgen/manifest.jsonl",
    checkpoint: str = "facebook/musicgen-small",
    device: str = "auto",
    max_chunks: int | None = None,
    existing_cache_dir: Path | None = None,
    resume_existing: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = prepared_root / manifest_relative_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared MusicGen manifest not found: {manifest_path}")

    rows = _read_jsonl(manifest_path)
    if max_chunks:
        rows = rows[:max_chunks]

    report: dict[str, Any] = {
        "test_name": "06_cache_musicgen_encodec_tokens",
        "status": "running",
        "prepared_root": str(prepared_root),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "existing_cache_dir": str(existing_cache_dir) if existing_cache_dir else None,
        "checkpoint": checkpoint,
        "requested_device": device,
        "resolved_device": None,
        "source_chunk_count": len(rows),
        "cached_chunk_count": 0,
        "failed_chunk_count": 0,
        "split_counts": {},
        "split_token_frames": {},
        "compression_model": {},
        "dry_run": dry_run,
        "resume_existing": resume_existing,
        "resumed_token_count": 0,
        "errors": [],
        "warnings": [],
    }
    if dry_run:
        report["status"] = "passed"
        return report

    resolved_device = _resolve_device(device)
    report["resolved_device"] = resolved_device

    compression_model = _load_compression_model(checkpoint, resolved_device)
    compression_meta = _musicgen_compression_metadata(compression_model)
    report["compression_model"] = _json_safe(compression_meta)
    sample_rate = int(compression_meta.get("sample_rate") or 32000)
    channels = int(compression_meta.get("channels") or 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    token_root = output_dir / "tokens"
    cached_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = defaultdict(int)
    split_token_frames: dict[str, int] = defaultdict(int)
    resumed_token_count = 0

    for row in rows:
        split = str(row.get("split") or "train")
        audio_rel = str(row.get("prepared_audio_path") or "")
        chunk_id = str(row.get("chunk_id") or row.get("_manifest_line"))
        token_rel_path = Path("tokens") / split / f"{chunk_id}.pt"
        token_path = output_dir / token_rel_path
        audio_path = prepared_root / audio_rel
        if not audio_rel or not audio_path.exists():
            failed_rows.append(
                {
                    "chunk_id": chunk_id,
                    "prepared_audio_path": audio_rel,
                    "reject_reason": "prepared_audio_missing",
                }
            )
            continue

        try:
            existing_token_path = (
                existing_cache_dir / token_rel_path
                if existing_cache_dir is not None
                else token_path
            )
            if resume_existing and existing_token_path.exists() and existing_token_path.stat().st_size > 0:
                cache_row, token_frames = _cache_row_from_token(
                    row=row,
                    token_rel_path=token_rel_path,
                    token_path=existing_token_path,
                    checkpoint=checkpoint,
                    compression_meta=compression_meta,
                )
                resumed_token_count += 1
            else:
                wav = _load_audio(audio_path, sample_rate=sample_rate, channels=channels).to(resolved_device)
                with torch.no_grad():
                    codes, scale = compression_model.encode(wav)
                if scale is not None:
                    raise RuntimeError("MusicGen compression model unexpectedly returned a scale tensor.")
                codes = codes.detach().cpu().to(torch.int16)
                token_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "codes": codes,
                        "chunk_id": chunk_id,
                        "source_example_id": row.get("source_example_id"),
                        "cara_pool_id": row.get("cara_pool_id"),
                        "cara_pool_index": row.get("cara_pool_index"),
                        "cara_pool_family": row.get("cara_pool_family"),
                        "cara_pool_family_index": row.get("cara_pool_family_index"),
                        "split": split,
                        "prepared_audio_path": audio_rel,
                        "checkpoint": checkpoint,
                        "compression_model": _json_safe(compression_meta),
                        "created_at": _utc_now(),
                    },
                    token_path,
                )
                cache_row, token_frames = _cache_row_from_token(
                    row=row,
                    token_rel_path=token_rel_path,
                    token_path=token_path,
                    checkpoint=checkpoint,
                    compression_meta=compression_meta,
                )
            cached_rows.append(cache_row)
            split_counts[split] += 1
            split_token_frames[split] += token_frames
        except Exception as exc:
            failed_rows.append(
                {
                    "chunk_id": chunk_id,
                    "prepared_audio_path": audio_rel,
                    "reject_reason": f"encodec_failed:{exc}",
                }
            )

    _write_jsonl(output_dir / "manifest.encodec.jsonl", cached_rows)
    _write_jsonl(output_dir / "failed_encodec_rows.jsonl", failed_rows)
    if cached_rows:
        validate_musicgen_encodec_manifest(cached_rows)
        resolver = build_musicgen_registry_resolver(cached_rows, split_manifest_path=prepared_root / "split_manifest.json")
        suffix_vocab = build_cara_suffix_vocab(cached_rows)
        _write_json(output_dir / "cara_registry_resolver.json", resolver)
        _write_json(output_dir / "cara_suffix_vocab.json", suffix_vocab)
        report["cara_registry"] = {
            "registry_hash": resolver["registry_hash"],
            "pool_count": resolver["pool_count"],
            "family_count": resolver["family_count"],
            "suffix_vocab_size": suffix_vocab["size"],
            "suffix_vocab_hash": suffix_vocab["hash"],
        }
    _write_json(
        output_dir / "encodec_cache_summary.json",
        {
            "created_at": _utc_now(),
            "checkpoint": checkpoint,
            "prepared_manifest": manifest_relative_path,
            "compression_model": compression_meta,
            "source_chunk_count": len(rows),
            "cached_chunk_count": len(cached_rows),
            "failed_chunk_count": len(failed_rows),
            "resumed_token_count": resumed_token_count,
            "split_counts": dict(split_counts),
            "split_token_frames": dict(split_token_frames),
            "cara_registry_hash": report.get("cara_registry", {}).get("registry_hash"),
            "cara_suffix_vocab_hash": report.get("cara_registry", {}).get("suffix_vocab_hash"),
        },
    )

    report["cached_chunk_count"] = len(cached_rows)
    report["failed_chunk_count"] = len(failed_rows)
    report["resumed_token_count"] = resumed_token_count
    report["split_counts"] = dict(split_counts)
    report["split_token_frames"] = dict(split_token_frames)
    report["status"] = "passed" if not failed_rows else "warning"
    if failed_rows:
        report["warnings"].append(f"{len(failed_rows)} chunk(s) could not be tokenized; see failed_encodec_rows.jsonl.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache MusicGen EnCodec token targets for prepared CARA chunks.")
    parser.add_argument("--prepared_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_relative_path", default="musicgen/manifest.jsonl")
    parser.add_argument("--checkpoint", default="facebook/musicgen-small")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--existing_cache_dir", default=None)
    parser.add_argument("--resume_existing", default="true")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = cache_musicgen_encodec_tokens(
        prepared_root=Path(args.prepared_root),
        output_dir=Path(args.output_dir),
        manifest_relative_path=args.manifest_relative_path,
        checkpoint=args.checkpoint,
        device=args.device,
        max_chunks=args.max_chunks,
        existing_cache_dir=Path(args.existing_cache_dir) if args.existing_cache_dir else None,
        resume_existing=parse_bool(args.resume_existing),
        dry_run=parse_bool(args.dry_run),
    )
    metadata = base_metadata(
        test_name="06_cache_musicgen_encodec_tokens",
        compute="gpu-smoke-h100" if report.get("resolved_device") == "cuda" else "cpu-prep-cluster",
        environment="azureml:env-musicgen-audiocraft:3",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="musicgen",
        environment_name="env-musicgen-audiocraft",
        environment_version="3",
        import_status="ok",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="musicgen_encodec_cache_report.json")
    raise SystemExit(0 if report["status"] in {"passed", "warning"} else 1)


if __name__ == "__main__":
    main()
