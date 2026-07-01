from __future__ import annotations

import argparse
import json
import math
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest_utils import load_manifest_rows

DEFAULT_MANIFEST = "data/attribution_manifest.jsonl"
DEFAULT_PROGRESS = "data/download_progress.json"
DEFAULT_REPORT = "data/pool_analytics_report.json"
DEFAULT_POOL_SECONDS = 5 * 60 * 60  # 5 hours
DEFAULT_MODEL_CLIP_SECONDS = {
    "model_a_11_88s": 11.88,
    "model_b_30s": 30.0,
}

# Tokens to suppress when building descriptor signatures: licenses, formats,
# generic noise words, and tier1 names (already encoded in the genre prefix).
DEFAULT_STOPWORDS = {
    "loop", "loops", "sample", "samples", "sound", "sounds", "audio",
    "music", "musical", "recording", "recorded", "clip", "track", "file",
    "wav", "mp3", "flac", "ogg", "aiff", "stereo", "mono",
    "freesound", "creative", "commons", "cc", "cc0", "cc-by", "cc-by-nc",
    "sampling", "sampling+", "public", "domain", "royalty", "free",
    "short", "long", "loud", "quiet", "high", "low", "clean", "dirty",
    "single", "multi", "one", "shot", "oneshot", "hit", "hits",
    "ui", "unclassified", "misc", "various", "other",
    "effects", "effect", "fx",
}
TIER_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value).strip() for value in payload.get("completed_ids", []) if str(value).strip()}


def is_downloaded(row: dict[str, Any], completed_ids: set[str]) -> bool:
    if str(row.get("download_status") or "") == "downloaded":
        return True
    return str(row.get("source_id") or "").strip() in completed_ids


def row_duration_seconds(row: dict[str, Any], meta_cache: dict[str, float]) -> float:
    direct = numeric(row.get("api_current_duration_s"), 0.0)
    if direct > 0:
        return direct
    meta_path = str(row.get("local_meta_path") or "").strip()
    if not meta_path:
        return 0.0
    if meta_path in meta_cache:
        return meta_cache[meta_path]
    duration = 0.0
    path = Path(meta_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            duration = numeric(payload.get("duration"), 0.0)
        except Exception:
            duration = 0.0
    meta_cache[meta_path] = duration
    return duration


def tier_tokens(tier: str) -> set[str]:
    return {tok for tok in TIER_TOKEN_RE.split(tier.lower()) if tok}


def row_tier(row: dict[str, Any]) -> str:
    return str(row.get("cara_tier1") or "Unclassified").strip() or "Unclassified"


def row_descriptor_tokens(row: dict[str, Any], sidecar_cache: dict[str, list[str]]) -> list[str]:
    tokens: list[str] = []
    sidecar_path = str(row.get("local_meta_path") or "").strip()
    if sidecar_path:
        if sidecar_path not in sidecar_cache:
            cached: list[str] = []
            path = Path(sidecar_path)
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    raw_tags = payload.get("tags")
                    if isinstance(raw_tags, list):
                        cached = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]
                except Exception:
                    cached = []
            sidecar_cache[sidecar_path] = cached
        tokens.extend(sidecar_cache[sidecar_path])
    raw_kw = row.get("cara_matched_keywords_json") or {}
    if isinstance(raw_kw, dict):
        for words in raw_kw.values():
            if isinstance(words, list):
                tokens.extend(str(word).strip().lower() for word in words if str(word).strip())
    return tokens


def clean_tokens(tokens: list[str], stopwords: set[str], tier_block: set[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in stopwords or token in tier_block:
            continue
        if len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return cleaned


def descriptor_signature(
    tokens: list[str],
    global_freq: Counter[str],
    descriptor_tag_count: int,
    vocabulary: set[str] | None,
) -> str:
    if not tokens:
        return "general"
    candidates = [tok for tok in tokens if (vocabulary is None or tok in vocabulary)]
    if not candidates:
        return "general"
    ranked = sorted(candidates, key=lambda token: (-global_freq.get(token, 0), token))
    chosen = ranked[:descriptor_tag_count]
    return "+".join(sorted(chosen))


def pretty_relationship(key: tuple[str, str]) -> str:
    tier, descriptor = key
    return f"{tier} :: {descriptor}"


def fmt_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{hours:,.2f}h"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if len(cell) > widths[index]:
                widths[index] = len(cell)
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return f"{line}\n{sep}\n{body}"


def analyze(
    rows: list[dict[str, Any]],
    completed_ids: set[str],
    pool_seconds: float,
    model_clip_seconds: dict[str, float],
    descriptor_tag_count: int = 1,
    vocabulary_size: int = 30,
    stopwords: set[str] | None = None,
) -> dict[str, Any]:
    stopwords = (stopwords or DEFAULT_STOPWORDS) | DEFAULT_STOPWORDS
    eligible_rows = [row for row in rows if is_downloaded(row, completed_ids)]
    meta_cache: dict[str, float] = {}
    sidecar_cache: dict[str, list[str]] = {}
    durations: dict[int, float] = {id(row): row_duration_seconds(row, meta_cache) for row in eligible_rows}
    rows_missing_duration = sum(1 for row in eligible_rows if durations[id(row)] <= 0)

    # Pass 1: gather per-row cleaned descriptor tokens and global frequency.
    row_tokens: dict[int, list[str]] = {}
    global_freq: Counter[str] = Counter()
    for row in eligible_rows:
        tier = row_tier(row)
        tier_block = tier_tokens(tier)
        cleaned = clean_tokens(
            row_descriptor_tokens(row, sidecar_cache),
            stopwords=stopwords,
            tier_block=tier_block,
        )
        row_tokens[id(row)] = cleaned
        for token in cleaned:
            global_freq[token] += 1

    # Build a curated vocabulary of the most common tokens so descriptor names
    # consolidate similar styles instead of fragmenting on rare combinations.
    vocabulary: set[str] | None = None
    if vocabulary_size and vocabulary_size > 0:
        vocabulary = {token for token, _ in global_freq.most_common(vocabulary_size)}

    # Pass 2: assign each row a (tier, descriptor) key.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        tier = row_tier(row)
        descriptor = descriptor_signature(
            row_tokens[id(row)],
            global_freq=global_freq,
            descriptor_tag_count=descriptor_tag_count,
            vocabulary=vocabulary,
        )
        grouped[(tier, descriptor)].append(row)

    relationships: list[dict[str, Any]] = []
    total_raw_seconds = 0.0
    for key, items in sorted(grouped.items(), key=lambda kv: -sum(durations[id(r)] for r in kv[1])):
        raw_seconds = sum(durations[id(row)] for row in items)
        total_raw_seconds += raw_seconds
        per_model: dict[str, dict[str, Any]] = {}
        for model_name, clip_len in model_clip_seconds.items():
            usable_clips = sum(int(durations[id(row)] // clip_len) for row in items)
            usable_seconds = usable_clips * clip_len
            full_pools = int(usable_seconds // pool_seconds)
            partial_seconds = usable_seconds - full_pools * pool_seconds
            partial_pool = 1 if partial_seconds > 0 else 0
            per_model[model_name] = {
                "clip_seconds": clip_len,
                "usable_clips": usable_clips,
                "usable_seconds": round(usable_seconds, 2),
                "usable_hours": round(usable_seconds / 3600.0, 3),
                "full_pools": full_pools,
                "partial_pool": partial_pool,
                "partial_pool_seconds": round(partial_seconds, 2),
                "total_pools_incl_partial": full_pools + partial_pool,
            }
        relationships.append({
            "tier": key[0],
            "descriptor": key[1],
            "label": pretty_relationship(key),
            "file_count": len(items),
            "raw_seconds": round(raw_seconds, 2),
            "raw_hours": round(raw_seconds / 3600.0, 3),
            "per_model": per_model,
        })

    model_summary: dict[str, dict[str, Any]] = {}
    for model_name, clip_len in model_clip_seconds.items():
        full_pools = sum(rel["per_model"][model_name]["full_pools"] for rel in relationships)
        partial_pools = sum(rel["per_model"][model_name]["partial_pool"] for rel in relationships)
        usable_clips = sum(rel["per_model"][model_name]["usable_clips"] for rel in relationships)
        usable_seconds = sum(rel["per_model"][model_name]["usable_seconds"] for rel in relationships)
        partial_seconds = sum(rel["per_model"][model_name]["partial_pool_seconds"] for rel in relationships)
        model_summary[model_name] = {
            "clip_seconds": clip_len,
            "usable_clips_total": usable_clips,
            "usable_hours_total": round(usable_seconds / 3600.0, 3),
            "full_5h_pools": full_pools,
            "partial_pools": partial_pools,
            "partial_pool_hours": round(partial_seconds / 3600.0, 3),
            "total_pools_incl_partial": full_pools + partial_pools,
        }

    return {
        "pool_seconds": pool_seconds,
        "pool_hours": pool_seconds / 3600.0,
        "descriptor_tag_count": descriptor_tag_count,
        "vocabulary_size": vocabulary_size,
        "vocabulary": sorted(vocabulary) if vocabulary else [],
        "top_global_descriptor_tokens": dict(global_freq.most_common(40)),
        "downloaded_files": len(eligible_rows),
        "downloaded_files_missing_duration": rows_missing_duration,
        "relationship_count": len(relationships),
        "raw_audio_hours": round(total_raw_seconds / 3600.0, 3),
        "model_clip_seconds": model_clip_seconds,
        "model_summary": model_summary,
        "relationships": relationships,
    }


def print_report(report: dict[str, Any], top_relationships: int) -> None:
    print(f"Eligible downloaded files: {report['downloaded_files']:,}")
    if report.get("downloaded_files_missing_duration"):
        print(f"  (missing duration metadata: {report['downloaded_files_missing_duration']:,})")
    print(f"Distinct relationships: {report['relationship_count']:,}")
    print(f"Raw audio: {report['raw_audio_hours']:,.2f}h")
    print(f"Pool size target: {report['pool_hours']:.2f}h\n")

    headers = ["Model", "Clip (s)", "Usable clips", "Usable hours", "Full 5h pools", "Partial pools", "Partial pool hrs", "Total pools"]
    table_rows: list[list[str]] = []
    for model_name, summary in report["model_summary"].items():
        table_rows.append([
            model_name,
            f"{summary['clip_seconds']:.2f}",
            f"{summary['usable_clips_total']:,}",
            f"{summary['usable_hours_total']:,.2f}",
            f"{summary['full_5h_pools']:,}",
            f"{summary['partial_pools']:,}",
            f"{summary['partial_pool_hours']:,.2f}",
            f"{summary['total_pools_incl_partial']:,}",
        ])
    print("Per-model pool counts (5-hour pools):")
    print(render_table(headers, table_rows))

    print("\nLikely relationship pool names (top by raw hours):")
    rel_headers = ["Relationship (tier :: primary_pool)", "Files", "Raw hours", "Pools (11.88s)", "Pools (30s)"]
    rel_rows: list[list[str]] = []
    for rel in report["relationships"][:top_relationships]:
        rel_rows.append([
            rel["label"],
            f"{rel['file_count']:,}",
            f"{rel['raw_hours']:,.2f}",
            f"{rel['per_model']['model_a_11_88s']['total_pools_incl_partial']:,}",
            f"{rel['per_model']['model_b_30s']['total_pools_incl_partial']:,}",
        ])
    print(render_table(rel_headers, rel_rows))
    if len(report["relationships"]) > top_relationships:
        print(f"... and {len(report['relationships']) - top_relationships} more relationships (see report JSON)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--progress-path", default=DEFAULT_PROGRESS)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--pool-hours", type=float, default=5.0)
    parser.add_argument("--top-relationships", type=int, default=25)
    parser.add_argument("--model-a-clip-seconds", type=float, default=11.88)
    parser.add_argument("--model-b-clip-seconds", type=float, default=30.0)
    parser.add_argument("--descriptor-tag-count", type=int, default=1,
                        help="Number of top tags combined into the descriptor portion of each pool name.")
    parser.add_argument("--vocabulary-size", type=int, default=30,
                        help="Restrict descriptors to the top-N most common tags. 0 disables (every tag eligible).")
    parser.add_argument("--extra-stopwords", default="",
                        help="Comma-separated extra tokens to ignore when building descriptor signatures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(Path(args.manifest))
    completed_ids = load_completed_ids(Path(args.progress_path))
    model_clip_seconds = {
        "model_a_11_88s": args.model_a_clip_seconds,
        "model_b_30s": args.model_b_clip_seconds,
    }
    extra_stopwords = {tok.strip().lower() for tok in args.extra_stopwords.split(",") if tok.strip()}
    report = analyze(
        rows=rows,
        completed_ids=completed_ids,
        pool_seconds=args.pool_hours * 3600.0,
        model_clip_seconds=model_clip_seconds,
        descriptor_tag_count=args.descriptor_tag_count,
        vocabulary_size=args.vocabulary_size,
        stopwords=DEFAULT_STOPWORDS | extra_stopwords,
    )
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report, top_relationships=args.top_relationships)
    print(f"\nFull report written to {args.report_output}")


if __name__ == "__main__":
    main()
