from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from test_prep_common import (
    audio_path_warnings,
    base_metadata,
    dataset_paths,
    list_audio_files,
    manifest_schema_warnings,
    parse_bool,
    write_report,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input_data", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--dashboard_triggered", default="false")
args = parser.parse_args()

root = Path(args.input_data)
audio_dir, manifest_path = dataset_paths(root)
report = {
    "test_name": "01_data_access_test",
    "input_root": str(root),
    "audio_dir": str(audio_dir),
    "audio_dir_exists": audio_dir.exists(),
    "manifest_path": str(manifest_path),
    "manifest_exists": manifest_path.exists(),
    "audio_file_count": 0,
    "audio_file_examples": [],
    "manifest_rows": None,
    "manifest_columns": [],
    "status": "failed",
    "errors": [],
    "warnings": [],
}

try:
    audio_files = list_audio_files(audio_dir)
    report["audio_file_count"] = len(audio_files)
    report["audio_file_examples"] = [str(path) for path in audio_files[:10]]
    report["warnings"].extend(audio_path_warnings(audio_dir))
    if not audio_dir.exists():
        report["errors"].append("Audio directory not found.")
    if manifest_path.exists():
        dataframe = pd.read_csv(manifest_path)
        report["manifest_rows"] = len(dataframe)
        report["manifest_columns"] = list(dataframe.columns)
        report["warnings"].extend(manifest_schema_warnings(report["manifest_columns"]))
    else:
        report["errors"].append("Manifest file not found.")
    if report["audio_dir_exists"] and report["manifest_exists"] and report["audio_file_count"] > 0:
        report["status"] = "passed"
except Exception as exc:
    report["errors"].append(repr(exc))

metadata = base_metadata(
    test_name=report["test_name"],
    compute="cpu-prep-cluster",
    environment="azureml://registries/azureml/environments/sklearn-1.5/versions/1",
    dashboard_triggered=parse_bool(args.dashboard_triggered),
    report=report,
)
write_report(Path(args.output_dir), report, metadata, report_alias="data_access_report.json")
raise SystemExit(0 if report["status"] == "passed" else 1)
