from __future__ import annotations

import argparse
from pathlib import Path

import torch

from test_prep_common import base_metadata, parse_bool, write_report


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--dashboard_triggered", default="false")
args = parser.parse_args()

report = {
    "test_name": "02_gpu_sanity_test",
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_name": None,
    "status": "failed",
    "errors": [],
}
try:
    if report["cuda_available"] and report["cuda_device_count"] > 0:
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["status"] = "passed"
    else:
        report["errors"].append("CUDA not available or no CUDA device detected.")
except Exception as exc:
    report["errors"].append(repr(exc))

metadata = base_metadata(
    test_name=report["test_name"],
    compute="gpu-smoke-h100",
    environment="azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/10",
    dashboard_triggered=parse_bool(args.dashboard_triggered),
    report=report,
)
write_report(Path(args.output_dir), report, metadata, report_alias="gpu_sanity_report.json")
raise SystemExit(0 if report["status"] == "passed" else 1)

