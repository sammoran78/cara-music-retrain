from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.manifest_utils import load_manifest_rows, save_manifest_rows

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def iterate_audio_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def compute_chromaprint(filepath: Path) -> str | None:
    try:
        result = subprocess.run(["fpcalc", "-raw", str(filepath)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            return line.split("=", 1)[1]
    return None


def fingerprint_similarity(fp1: str, fp2: str) -> float:
    ints1 = [int(x) for x in fp1.split(",") if x.strip()]
    ints2 = [int(x) for x in fp2.split(",") if x.strip()]
    min_len = min(len(ints1), len(ints2))
    if min_len == 0:
        return 0.0
    matching_bits = sum(32 - bin(a ^ b).count("1") for a, b in zip(ints1[:min_len], ints2[:min_len]))
    return matching_bits / (min_len * 32)


def fallback_hash(filepath: Path) -> str:
    digest = hashlib.sha256(filepath.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def extract_source_id(filepath: Path) -> str:
    return filepath.stem


def build_fingerprints(audio_files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filepath in audio_files:
        fingerprint = compute_chromaprint(filepath)
        fingerprint_hash = fingerprint if fingerprint else fallback_hash(filepath)
        rows.append({"source_id": extract_source_id(filepath), "filepath": str(filepath), "fingerprint_hash": fingerprint_hash})
    return rows


def find_duplicates(rows: list[dict[str, str]], threshold: float = 0.85) -> list[dict[str, str]]:
    duplicates: list[dict[str, str]] = []
    cluster_id = 1
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            left_fp = left["fingerprint_hash"]
            right_fp = right["fingerprint_hash"]
            if left_fp.startswith("sha256:") or right_fp.startswith("sha256:"):
                similarity = 1.0 if left_fp == right_fp else 0.0
            else:
                similarity = fingerprint_similarity(left_fp, right_fp)
            if similarity >= threshold:
                duplicates.append(
                    {
                        "duplicate_cluster_id": str(cluster_id),
                        "kept_file_id": left["source_id"],
                        "removed_file_ids": right["source_id"],
                        "similarity_score": f"{similarity:.4f}",
                        "same_uploader": "unknown",
                        "pool_memberships_before": "",
                        "pool_memberships_after": "",
                        "decision_reason": "near_duplicate_detected",
                    }
                )
                cluster_id += 1
    return duplicates


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="data/attribution_manifest.jsonl")
    parser.add_argument("--fingerprints-output", default="data/fingerprints.csv")
    parser.add_argument("--duplicates-output", default="data/duplicates_report.csv")
    parser.add_argument("--threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_files = iterate_audio_files(Path(args.data_dir))
    fingerprint_rows = build_fingerprints(audio_files)
    duplicate_rows = find_duplicates(fingerprint_rows, threshold=args.threshold)
    write_csv(Path(args.fingerprints_output), fingerprint_rows, ["source_id", "filepath", "fingerprint_hash"])
    write_csv(
        Path(args.duplicates_output),
        duplicate_rows,
        [
            "duplicate_cluster_id",
            "kept_file_id",
            "removed_file_ids",
            "similarity_score",
            "same_uploader",
            "pool_memberships_before",
            "pool_memberships_after",
            "decision_reason",
        ],
    )
    manifest_rows = load_manifest_rows(Path(args.manifest))
    fp_lookup = {row["source_id"]: row for row in fingerprint_rows}
    for row in manifest_rows:
        fp = fp_lookup.get(str(row.get("source_id", "")))
        if not fp:
            continue
        row["content_fingerprint"] = fp["fingerprint_hash"]
        row["local_audio_path"] = fp["filepath"]
        row["download_status"] = "available"
    save_manifest_rows(manifest_rows, Path(args.manifest))
    print(json.dumps({"fingerprints": len(fingerprint_rows), "duplicates": len(duplicate_rows)}, indent=2))


if __name__ == "__main__":
    main()
