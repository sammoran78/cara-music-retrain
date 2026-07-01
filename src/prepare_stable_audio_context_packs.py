from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("chunk_id") or row.get("_line_number") or ""),
            str(row.get("cara_pool_id") or ""),
            str(row.get("source_id") or ""),
            str(row.get("split") or ""),
        ]
    )


def _context_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "prepared_audio_path": row.get("prepared_audio_path"),
        "source_id": row.get("source_id"),
        "source_example_id": row.get("source_example_id"),
        "split": row.get("split"),
        "duration_sec": row.get("duration_sec"),
        "prompt": row.get("prompt"),
        "cara_pool_id": row.get("cara_pool_id"),
        "cara_pool_index": row.get("cara_pool_index"),
        "cara_pool_family": row.get("cara_pool_family"),
        "cara_pool_family_index": row.get("cara_pool_family_index"),
    }


def _stable_bucket(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _sha256_text(f"{seed}|bucket|{_stable_key(row)}"))


def _deterministic_pick(
    candidates: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    seed: str,
    limit: int,
    exclude_chunk_ids: set[str] | None = None,
    require_different_source: bool = True,
    require_different_pool: bool = False,
    require_different_family: bool = False,
) -> list[dict[str, Any]]:
    if not candidates or limit <= 0:
        return []
    target_source_id = str(target.get("source_id") or "")
    target_pool_id = str(target.get("cara_pool_id") or "")
    target_family = str(target.get("cara_pool_family") or "")
    excluded = exclude_chunk_ids or set()
    start = int(_sha256_text(f"{seed}|offset|{_stable_key(target)}")[:12], 16) % len(candidates)
    selected: list[dict[str, Any]] = []
    scanned = 0
    # One circular pass is enough: buckets are pre-sorted and the offset is target-specific.
    while scanned < len(candidates) and len(selected) < limit:
        candidate = candidates[(start + scanned) % len(candidates)]
        scanned += 1
        chunk_id = str(candidate.get("chunk_id") or "")
        if chunk_id in excluded:
            continue
        if require_different_source and str(candidate.get("source_id") or "") == target_source_id:
            continue
        if require_different_pool and str(candidate.get("cara_pool_id") or "") == target_pool_id:
            continue
        if require_different_family and str(candidate.get("cara_pool_family") or "") == target_family:
            continue
        selected.append(candidate)
    return selected


def _select_contexts(
    target: dict[str, Any],
    by_pool_split: dict[tuple[str, str], list[dict[str, Any]]],
    by_family_split: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    max_contexts: int,
    seed: str,
) -> tuple[list[dict[str, Any]], str]:
    split = str(target.get("split") or "train")
    source_id = str(target.get("source_id") or "")
    pool_id = str(target.get("cara_pool_id") or "")
    family = str(target.get("cara_pool_family") or "")

    same_pool_bucket = by_pool_split.get((pool_id, split), [])
    selected = _deterministic_pick(
        same_pool_bucket,
        target,
        seed=f"{seed}|same_pool",
        limit=max_contexts,
    )
    if len(selected) >= max_contexts:
        return selected, "same_pool_source_disjoint"

    already = {str(row.get("chunk_id") or "") for row in selected}
    same_family_bucket = by_family_split.get((family, split), [])
    selected.extend(
        _deterministic_pick(
            same_family_bucket,
            target,
            seed=f"{seed}|same_family",
            limit=max_contexts - len(selected),
            exclude_chunk_ids=already,
            require_different_pool=True,
        )
    )
    if selected:
        return selected, "same_pool_plus_family_fallback" if same_pool_bucket else "same_family_source_disjoint"
    return [], "no_source_disjoint_context"


def prepare_context_packs(
    *,
    input_data: Path,
    output_dir: Path,
    manifest_relative_path: str,
    max_contexts: int,
    seed: str,
    dry_run: bool,
) -> dict[str, Any]:
    manifest_path = input_data / manifest_relative_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared Stable Audio manifest not found: {manifest_path}")

    print(f"[context-packs] reading prepared manifest: {manifest_path}", flush=True)
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Prepared Stable Audio manifest is empty: {manifest_path}")
    print(f"[context-packs] loaded {len(rows):,} prepared rows", flush=True)

    required = {
        "chunk_id",
        "prepared_audio_path",
        "split",
        "source_id",
        "cara_pool_id",
        "cara_pool_index",
        "cara_pool_family",
        "cara_pool_family_index",
    }
    missing_rows = [
        {"line_number": row.get("_line_number"), "missing": sorted(key for key in required if row.get(key) in {None, ""})}
        for row in rows
        if any(row.get(key) in {None, ""} for key in required)
    ]
    if missing_rows:
        raise RuntimeError(f"{len(missing_rows)} prepared manifest rows are missing required CARA/context fields.")

    by_pool_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_family_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split"))
        by_pool_split[(str(row.get("cara_pool_id")), split)].append(row)
        by_family_split[(str(row.get("cara_pool_family")), split)].append(row)
        by_split[split].append(row)
    for key, bucket in list(by_pool_split.items()):
        by_pool_split[key] = _stable_bucket(bucket, f"{seed}|pool")
    for key, bucket in list(by_family_split.items()):
        by_family_split[key] = _stable_bucket(bucket, f"{seed}|family")
    for split, bucket in list(by_split.items()):
        by_split[split] = _stable_bucket(bucket, f"{seed}|split")
    print(
        "[context-packs] indexed "
        f"{len(by_pool_split):,} pool/split buckets, {len(by_family_split):,} family/split buckets",
        flush=True,
    )

    pack_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    pack_policy_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    violation_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        contexts, policy = _select_contexts(
            row,
            by_pool_split,
            by_family_split,
            max_contexts=max_contexts,
            seed=seed,
        )
        pack_policy_counts[policy] += 1
        split_counts[str(row.get("split"))] += 1
        pack_id = _sha256_text(f"context-pack|{seed}|{_stable_key(row)}")[:24]
        if any(str(ctx.get("source_id") or "") == str(row.get("source_id") or "") for ctx in contexts):
            violation_rows.append(
                {
                    "target_chunk_id": row.get("chunk_id"),
                    "target_source_id": row.get("source_id"),
                    "reason": "source_overlap",
                }
            )
        pack_rows.append(
            {
                "context_pack_id": pack_id,
                "target": _context_summary(row),
                "context_examples": [_context_summary(ctx) for ctx in contexts],
                "context_count": len(contexts),
                "context_policy": policy,
                "max_contexts": max_contexts,
                "selection_seed": seed,
                "conditioning_lane": "context_plus_prompt",
                "paper_alignment": {
                    "context_as_cross_attention_conditioning": True,
                    "prompt_text_unchanged": True,
                    "source_disjoint_context_required": True,
                },
            }
        )

        split_bucket = by_split.get(str(row.get("split")), rows)
        shuffled_candidates = _deterministic_pick(
            split_bucket,
            row,
            seed=f"{seed}|shuffled",
            limit=max_contexts,
            require_different_pool=True,
        )
        mismatched_family = _deterministic_pick(
            split_bucket,
            row,
            seed=f"{seed}|mismatched_family",
            limit=max_contexts,
            require_different_pool=True,
            require_different_family=True,
        )
        control_rows.append(
            {
                "context_pack_id": pack_id,
                "target_chunk_id": row.get("chunk_id"),
                "target_pool_id": row.get("cara_pool_id"),
                "target_family": row.get("cara_pool_family"),
                "controls": {
                    "prompt_only": {"context_examples": [], "context_masked": True},
                    "shuffled_context": {
                        "context_examples": [_context_summary(ctx) for ctx in shuffled_candidates[:max_contexts]],
                        "context_masked": False,
                    },
                    "mismatched_family_context": {
                        "context_examples": [_context_summary(ctx) for ctx in mismatched_family[:max_contexts]],
                        "context_masked": False,
                    },
                },
            }
        )
        if index == 1 or index % 5000 == 0 or index == len(rows):
            print(
                f"[context-packs] processed {index:,}/{len(rows):,} rows "
                f"({index / len(rows) * 100:.1f}%)",
                flush=True,
            )

    summary = {
        "test_name": "10_prepare_stable_audio_context_packs",
        "status": "passed" if not violation_rows else "failed",
        "created_at": _utc_now(),
        "input_data": str(input_data),
        "manifest_relative_path": manifest_relative_path,
        "manifest_sha256": _file_sha256(manifest_path),
        "output_dir": str(output_dir),
        "target_rows": len(rows),
        "context_pack_rows": len(pack_rows),
        "control_rows": len(control_rows),
        "max_contexts": max_contexts,
        "selection_seed": seed,
        "split_counts": dict(split_counts),
        "context_policy_counts": dict(pack_policy_counts),
        "source_disjoint_violations": len(violation_rows),
        "source_disjoint_violation_examples": violation_rows[:20],
        "artifact_files": {
            "context_pack_manifest": "context_pack_manifest.jsonl",
            "context_controls_manifest": "context_controls_manifest.jsonl",
            "context_pack_summary": "context_pack_summary.json",
        },
        "notes": [
            "Context examples are selected from the prepared Stable Audio chunk manifest, not raw files.",
            "This stage locks context/example relationships only; it does not train a context-conditioned model.",
        ],
        "dry_run": dry_run,
    }

    if not dry_run:
        print(f"[context-packs] writing artifacts to {output_dir}", flush=True)
        _write_jsonl(output_dir / "context_pack_manifest.jsonl", pack_rows)
        _write_jsonl(output_dir / "context_controls_manifest.jsonl", control_rows)
        _write_json(output_dir / "context_pack_summary.json", summary)
        if violation_rows:
            _write_jsonl(output_dir / "source_disjoint_violations.jsonl", violation_rows)
    print(f"[context-packs] completed with status={summary['status']}", flush=True)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare source-disjoint Stable Audio Context Diffusion packs.")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_relative_path", default="stable_audio_open_small/manifest.jsonl")
    parser.add_argument("--max_contexts", type=int, default=3)
    parser.add_argument("--selection_seed", default="cara-context-v1")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "test_name": "10_prepare_stable_audio_context_packs",
        "status": "running",
        "stage": "start",
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
    }
    try:
        report.update(
            prepare_context_packs(
                input_data=Path(args.input_data),
                output_dir=Path(args.output_dir),
                manifest_relative_path=args.manifest_relative_path,
                max_contexts=max(1, int(args.max_contexts)),
                seed=str(args.selection_seed),
                dry_run=parse_bool(args.dry_run),
            )
        )
    except Exception as exc:
        report.update({"status": "failed", "stage": "prepare_context_packs", "error": str(exc)})
        raise
    finally:
        metadata = base_metadata(
            test_name="10_prepare_stable_audio_context_packs",
            compute="h100_preferred_else_cpu",
            environment="azureml:env-stable-audio-tools:8",
            dashboard_triggered=parse_bool(args.dashboard_triggered),
            report=report,
            model_family="stable_audio_open_small_context_diffusion",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias="stable_audio_context_pack_report.json")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
