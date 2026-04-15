from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

ATTRIBUTION_URL = "https://info.stability.ai/attributions"
EXPECTED_FREESOUND_COUNT = 472618
EXPECTED_FMA_COUNT = 13874

FREESOUND_PATTERNS = [
    re.compile(r"freesound(?:\.org)?/(?:people/[^/]+/sounds/|sounds/)(\d+)", re.IGNORECASE),
    re.compile(r"\bfreesound\D{0,20}(\d{2,})\b", re.IGNORECASE),
]
FMA_PATTERNS = [
    re.compile(r"freemusicarchive(?:\.org)?/.{0,80}/track/(\d+)", re.IGNORECASE),
    re.compile(r"\bfma\D{0,20}(\d{2,})\b", re.IGNORECASE),
]


def fetch_attribution_page(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _collect_ids(text: str, patterns: list[re.Pattern[str]]) -> list[int]:
    ids: set[int] = set()
    for pattern in patterns:
        for match in pattern.findall(text):
            ids.add(int(match))
    return sorted(ids)


def parse_attribution_ids(page_text: str) -> dict[str, list[int]]:
    freesound_ids = _collect_ids(page_text, FREESOUND_PATTERNS)
    fma_ids = _collect_ids(page_text, FMA_PATTERNS)
    return {
        "freesound_ids": freesound_ids,
        "fma_ids": fma_ids,
    }


def build_report(data: dict[str, list[int]]) -> dict[str, int]:
    return {
        "freesound_count": len(data["freesound_ids"]),
        "fma_count": len(data["fma_ids"]),
        "expected_freesound_count": EXPECTED_FREESOUND_COUNT,
        "expected_fma_count": EXPECTED_FMA_COUNT,
        "freesound_delta": len(data["freesound_ids"]) - EXPECTED_FREESOUND_COUNT,
        "fma_delta": len(data["fma_ids"]) - EXPECTED_FMA_COUNT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=ATTRIBUTION_URL)
    parser.add_argument("--output", default="data/attribution_list.json")
    parser.add_argument("--report-output", default="data/attribution_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_text = fetch_attribution_page(args.url)
    data = parse_attribution_ids(page_text)
    report = build_report(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
