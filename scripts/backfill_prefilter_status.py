from __future__ import annotations

import csv
import json
import time
from pathlib import Path

MANIFEST_PATH = Path("data/attribution_manifest.jsonl")
CONFIRMED_CSV = Path("data/old_freesound_filtered_sources.csv")
REJECTED_CSV = Path("data/old_freesound_rejected_sources.csv")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ids(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = str(row.get("sound_id", "")).strip()
            if sid:
                mapping[sid] = row.get("reason", "")
    return mapping


def main() -> None:
    log("Loading confirmed/rejected id sets from CSVs")
    confirmed_map = load_ids(CONFIRMED_CSV)
    rejected_map = load_ids(REJECTED_CSV)
    log(f"Confirmed ids: {len(confirmed_map)} | Rejected ids: {len(rejected_map)}")

    tmp_path = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    log(f"Streaming rewrite: {MANIFEST_PATH} -> {tmp_path}")

    updated_confirmed = 0
    updated_rejected = 0
    seen = 0
    start = time.time()

    with MANIFEST_PATH.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            seen += 1
            stripped = line.strip()
            if not stripped:
                dst.write(line)
                continue
            row = json.loads(stripped)
            if row.get("source") == "freesound":
                sid = str(row.get("source_id", "")).strip()
                if sid in confirmed_map:
                    row["prefilter_status"] = "confirmed"
                    row["prefilter_reason"] = confirmed_map[sid]
                    updated_confirmed += 1
                elif sid in rejected_map:
                    row["prefilter_status"] = "rejected"
                    row["prefilter_reason"] = rejected_map[sid]
                    updated_rejected += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            if seen % 50000 == 0:
                elapsed = time.time() - start
                log(
                    f"  processed {seen} rows ({elapsed:.1f}s) | "
                    f"confirmed_updates={updated_confirmed} rejected_updates={updated_rejected}"
                )

    log(f"Total rows processed: {seen} in {time.time() - start:.1f}s")
    log("Replacing manifest atomically")
    tmp_path.replace(MANIFEST_PATH)
    log(
        f"Done. confirmed_updates={updated_confirmed}, rejected_updates={updated_rejected}"
    )


if __name__ == "__main__":
    main()
