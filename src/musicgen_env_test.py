from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

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
    "test_name": "03_musicgen_env_test",
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_name": None,
    "import_status": "not_attempted",
    "audiocraft_import": "not_attempted",
    "transformers_import": "not_attempted",
    "transformers_torch_available": None,
    "audio_dir_exists": audio_dir.exists(),
    "manifest_exists": manifest_path.exists(),
    "audio_file_count": 0,
    "manifest_rows": None,
    "manifest_columns": [],
    "status": "failed",
    "errors": [],
    "warnings": [],
}
try:
    import audiocraft  # noqa: F401

    report["audiocraft_import"] = "ok"
    report["import_status"] = "ok"
except Exception as exc:
    report["audiocraft_import"] = "failed"
    report["import_status"] = "failed"
    report["errors"].append(f"AudioCraft import failed: {exc!r}")

try:
    import transformers
    from transformers.utils import is_torch_available

    report["transformers_version"] = transformers.__version__
    report["transformers_torch_available"] = bool(is_torch_available())
    _ = transformers.T5EncoderModel
    report["transformers_import"] = "ok"
    if not report["transformers_torch_available"]:
        report["errors"].append("Transformers does not consider PyTorch available; MusicGen T5 conditioner loading will fail.")
except Exception as exc:
    report["transformers_import"] = "failed"
    report["errors"].append(f"Transformers T5/PyTorch compatibility check failed: {exc!r}")

try:
    if report["cuda_available"] and report["cuda_device_count"] > 0:
        report["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        report["errors"].append("CUDA not available or no CUDA device detected.")
    audio_files = list_audio_files(audio_dir)
    report["audio_file_count"] = len(audio_files)
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
    if (
        report["audiocraft_import"] == "ok"
        and report["transformers_import"] == "ok"
        and report["transformers_torch_available"]
        and report["cuda_available"]
        and report["cuda_device_count"] > 0
        and report["audio_dir_exists"]
        and report["manifest_exists"]
        and report["audio_file_count"] > 0
    ):
        report["status"] = "passed"
except Exception as exc:
    report["errors"].append(repr(exc))

metadata = base_metadata(
    test_name=report["test_name"],
    compute="gpu-smoke-h100",
    environment="azureml:env-musicgen-audiocraft:3",
    dashboard_triggered=parse_bool(args.dashboard_triggered),
    report=report,
    model_family="musicgen",
    environment_name="env-musicgen-audiocraft",
    environment_version="3",
    import_status=report["audiocraft_import"],
)
write_report(Path(args.output_dir), report, metadata, report_alias="musicgen_env_test_report.json")
raise SystemExit(0 if report["status"] == "passed" else 1)
