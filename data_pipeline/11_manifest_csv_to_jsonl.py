from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


JSONISH_FIELDS = {
    "api_current_tags_json",
    "cara_candidate_pools_json",
    "cara_soft_targets_json",
    "cara_matched_keywords_json",
}

BOOL_FIELDS = {
    "id_matches_url",
    "originally_in_stable_audio_open_small",
    "include_in_subset",
}

INT_FIELDS = {
    "source_id",
    "raw_id",
    "url_sound_id",
    "api_current_samplerate",
    "api_current_channels",
    "cara_auto_label_score",
}

FLOAT_FIELDS = {
    "api_current_duration_s",
    "api_bpm",
    "cara_auto_label_confidence",
}


def _parse_jsonish(value: str):
    text = (value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def _parse_bool(value: str):
    text = (value or "").strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return value


def _parse_int(value: str):
    text = (value or "").strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return value


def _parse_float(value: str):
    text = (value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return value


def convert_value(field: str, value: str):
    if field in JSONISH_FIELDS:
        return _parse_jsonish(value)
    if field in BOOL_FIELDS:
        return _parse_bool(value)
    if field in INT_FIELDS:
        return _parse_int(value)
    if field in FLOAT_FIELDS:
        return _parse_float(value)
    text = (value or "").strip()
    return text if text != "" else None


def convert_csv_to_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as in_handle, output_path.open("w", encoding="utf-8") as out_handle:
        reader = csv.DictReader(in_handle)
        for row in reader:
            converted = {field: convert_value(field, value) for field, value in row.items()}
            out_handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            row_count += 1
    return {"rows": row_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/attribution_manifest.csv")
    parser.add_argument("--output", default="data/attribution_manifest.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_csv_to_jsonl(Path(args.input), Path(args.output))
    summary["input"] = args.input
    summary["output"] = args.output
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
