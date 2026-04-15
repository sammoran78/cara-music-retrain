from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise_metrics(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    lines = ["# Evaluation Summary", ""]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="evaluation/metrics_latest.json")
    parser.add_argument("--output", default="evaluation/results_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarise_metrics(Path(args.metrics), Path(args.output))


if __name__ == "__main__":
    main()
