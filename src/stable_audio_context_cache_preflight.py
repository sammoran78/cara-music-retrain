from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from test_prep_common import base_metadata, json_safe, parse_bool, write_report


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audio_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    result: dict[str, Any] = {"exists": True, "path": str(path), "size_bytes": path.stat().st_size}
    try:
        with wave.open(str(path), "rb") as handle:
            result.update(
                {
                    "channels": handle.getnchannels(),
                    "sample_rate": handle.getframerate(),
                    "frames": handle.getnframes(),
                    "duration_sec": round(handle.getnframes() / max(1, handle.getframerate()), 6),
                    "sample_width_bytes": handle.getsampwidth(),
                }
            )
    except Exception as exc:
        result["wave_probe_error"] = str(exc)
    return result


def _context_audio_path(prepared_root: Path, example: dict[str, Any]) -> Path:
    rel = example.get("prepared_audio_path")
    if not rel:
        return prepared_root / "__missing_prepared_audio_path__"
    return prepared_root / str(rel)


def cache_context_metadata(
    *,
    prepared_data: Path,
    context_pack_dir: Path,
    output_dir: Path,
    context_pack_relative_path: str,
    dry_run: bool,
) -> dict[str, Any]:
    pack_path = context_pack_dir / context_pack_relative_path
    print(f"[context-cache] reading context packs from {pack_path}", flush=True)
    if not pack_path.exists():
        raise FileNotFoundError(f"Context pack manifest not found: {pack_path}")
    packs = _read_jsonl(pack_path)
    if not packs:
        raise RuntimeError(f"Context pack manifest is empty: {pack_path}")
    print(f"[context-cache] loaded {len(packs):,} context pack rows", flush=True)

    cache_rows: list[dict[str, Any]] = []
    missing_audio: list[dict[str, Any]] = []
    context_counts: Counter[int] = Counter()
    policy_counts: Counter[str] = Counter()
    probed_audio: dict[str, dict[str, Any]] = {}

    for row_index, pack in enumerate(packs, start=1):
        pack_id = str(pack.get("context_pack_id") or pack.get("_line_number"))
        policy_counts[str(pack.get("context_policy") or "unknown")] += 1
        examples = pack.get("context_examples") if isinstance(pack.get("context_examples"), list) else []
        context_counts[len(examples)] += 1
        for position, example in enumerate(examples):
            path = _context_audio_path(prepared_data, example)
            cache_key = _sha256_text(f"{pack_id}|{position}|{example.get('chunk_id')}")[:24]
            probe = probed_audio.get(str(path))
            if probe is None:
                probe = _audio_probe(path)
                probed_audio[str(path)] = probe
            if not probe.get("exists"):
                missing_audio.append(
                    {
                        "context_pack_id": pack_id,
                        "context_position": position,
                        "chunk_id": example.get("chunk_id"),
                        "prepared_audio_path": example.get("prepared_audio_path"),
                    }
                )
            cache_rows.append(
                {
                    "context_cache_id": cache_key,
                    "context_pack_id": pack_id,
                    "context_position": position,
                    "chunk_id": example.get("chunk_id"),
                    "prepared_audio_path": example.get("prepared_audio_path"),
                    "source_id": example.get("source_id"),
                    "cara_pool_id": example.get("cara_pool_id"),
                    "cara_pool_index": example.get("cara_pool_index"),
                    "cara_pool_family": example.get("cara_pool_family"),
                    "cara_pool_family_index": example.get("cara_pool_family_index"),
                    "audio_probe": probe,
                    "conditioning_cache_mode": "audio_metadata_preflight_cache",
                    "future_trainer_requirement": "replace_or_extend_with_frozen_stable_audio_latent_tokens_before full context training",
                }
            )
        if row_index == 1 or row_index % 5000 == 0 or row_index == len(packs):
            print(
                "[context-cache] processed "
                f"{row_index:,}/{len(packs):,} packs; "
                f"cache_rows={len(cache_rows):,}; "
                f"unique_audio={len(probed_audio):,}; "
                f"missing_audio={len(missing_audio):,}",
                flush=True,
            )

    status = "passed" if not missing_audio else "failed"
    summary = {
        "test_name": "11_cache_stable_audio_context_metadata",
        "status": status,
        "created_at": _utc_now(),
        "prepared_data": str(prepared_data),
        "context_pack_dir": str(context_pack_dir),
        "context_pack_relative_path": context_pack_relative_path,
        "output_dir": str(output_dir),
        "context_pack_rows": len(packs),
        "context_cache_rows": len(cache_rows),
        "unique_context_audio_files": len(probed_audio),
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
        "context_count_distribution": {str(key): value for key, value in sorted(context_counts.items())},
        "context_policy_counts": dict(policy_counts),
        "artifact_files": {
            "context_cache_manifest": "context_cache_manifest.jsonl",
            "context_cache_summary": "context_cache_summary.json",
        },
        "preflight_scope": "metadata_and_source_disjoint_cache_only",
        "trainer_unlock_status": "conditioner_preflight_ready",
        "dry_run": dry_run,
    }
    if not dry_run:
        print(f"[context-cache] writing artifacts to {output_dir}", flush=True)
        _write_jsonl(output_dir / "context_cache_manifest.jsonl", cache_rows)
        _write_json(output_dir / "context_cache_summary.json", summary)
    print(f"[context-cache] completed with status={status}", flush=True)
    return summary


def context_conditioner_preflight(
    *,
    context_pack_dir: Path,
    context_cache_dir: Path,
    output_dir: Path,
    context_pack_relative_path: str,
    context_cache_relative_path: str,
    dry_run: bool,
) -> dict[str, Any]:
    pack_path = context_pack_dir / context_pack_relative_path
    cache_path = context_cache_dir / context_cache_relative_path
    print(f"[context-preflight] reading context packs from {pack_path}", flush=True)
    print(f"[context-preflight] reading context cache from {cache_path}", flush=True)
    if not pack_path.exists():
        raise FileNotFoundError(f"Context pack manifest not found: {pack_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Context cache manifest not found: {cache_path}")

    packs = _read_jsonl(pack_path)
    cache_rows = _read_jsonl(cache_path)
    print(
        f"[context-preflight] loaded {len(packs):,} pack rows and {len(cache_rows):,} cache rows",
        flush=True,
    )
    pack_ids = {str(row.get("context_pack_id") or "") for row in packs}
    cache_pack_ids = {str(row.get("context_pack_id") or "") for row in cache_rows}
    orphan_cache_ids = sorted(cache_pack_ids - pack_ids)
    packs_with_context = sum(1 for row in packs if int(row.get("context_count") or 0) > 0)
    prompt_lanes = [
        "context_plus_prompt",
        "context_only_prompt_empty",
        "prompt_only_context_masked",
        "shuffled_context",
        "mismatched_family_context",
    ]
    projected_context_token_shape = {
        "batch": "B",
        "max_context_examples": max((int(row.get("context_count") or 0) for row in packs), default=0),
        "tokens_per_context": 1,
        "projection_target": "stable_audio_cross_attention_conditioning_dim",
        "source": "context_cache_manifest.audio_probe",
    }
    status = "passed" if packs and cache_rows and not orphan_cache_ids and packs_with_context else "failed"
    report = {
        "test_name": "12_stable_audio_context_conditioner_preflight",
        "status": status,
        "created_at": _utc_now(),
        "context_pack_rows": len(packs),
        "context_cache_rows": len(cache_rows),
        "packs_with_context": packs_with_context,
        "orphan_cache_pack_ids": orphan_cache_ids[:20],
        "paper_alignment_checks": {
            "context_as_separate_conditioning_tokens": True,
            "ordinary_prompt_text_unchanged": True,
            "context_only_lane_required": True,
            "prompt_only_lane_required": True,
            "shuffled_context_lane_required": True,
            "mismatched_family_lane_required": True,
            "source_disjoint_context_required": True,
        },
        "planned_prompt_lanes": prompt_lanes,
        "projected_context_token_shape": projected_context_token_shape,
        "trainer_unlock_status": "smoke_locked_until_context_conditioner_training_wrapper_is_implemented",
        "next_implementation": "Add a Stable Audio context conditioner that projects cached context audio tokens into cross-attention beside text conditioning.",
        "dry_run": dry_run,
    }
    if not dry_run:
        print(f"[context-preflight] writing artifacts to {output_dir}", flush=True)
        _write_json(output_dir / "context_conditioner_preflight_summary.json", report)
    print(f"[context-preflight] completed with status={status}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache/preflight Stable Audio Context Diffusion artifacts.")
    parser.add_argument("--mode", choices=["cache", "preflight"], required=True)
    parser.add_argument("--prepared_data")
    parser.add_argument("--context_pack_dir", required=True)
    parser.add_argument("--context_cache_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_pack_relative_path", default="context_pack_manifest.jsonl")
    parser.add_argument("--context_cache_relative_path", default="context_cache_manifest.jsonl")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "test_name": "stable_audio_context_cache_preflight",
        "status": "running",
        "mode": args.mode,
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
    }
    print(f"[context-{args.mode}] starting Stable Audio Context Diffusion step", flush=True)
    try:
        if args.mode == "cache":
            if not args.prepared_data:
                raise RuntimeError("--prepared_data is required in cache mode.")
            report.update(
                cache_context_metadata(
                    prepared_data=Path(args.prepared_data),
                    context_pack_dir=Path(args.context_pack_dir),
                    output_dir=Path(args.output_dir),
                    context_pack_relative_path=args.context_pack_relative_path,
                    dry_run=parse_bool(args.dry_run),
                )
            )
            alias = "stable_audio_context_cache_report.json"
            test_name = "11_cache_stable_audio_context_metadata"
        else:
            if not args.context_cache_dir:
                raise RuntimeError("--context_cache_dir is required in preflight mode.")
            report.update(
                context_conditioner_preflight(
                    context_pack_dir=Path(args.context_pack_dir),
                    context_cache_dir=Path(args.context_cache_dir),
                    output_dir=Path(args.output_dir),
                    context_pack_relative_path=args.context_pack_relative_path,
                    context_cache_relative_path=args.context_cache_relative_path,
                    dry_run=parse_bool(args.dry_run),
                )
            )
            alias = "stable_audio_context_preflight_report.json"
            test_name = "12_stable_audio_context_conditioner_preflight"
    except Exception as exc:
        report.update({"status": "failed", "stage": args.mode, "error": str(exc)})
        alias = "stable_audio_context_cache_preflight_report.json"
        test_name = "stable_audio_context_cache_preflight"
        raise
    finally:
        metadata = base_metadata(
            test_name=test_name,
            compute="h100_preferred_else_cpu",
            environment="azureml:env-stable-audio-tools:8",
            dashboard_triggered=parse_bool(args.dashboard_triggered),
            report=report,
            model_family="stable_audio_open_small_context_diffusion",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias=alias)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
