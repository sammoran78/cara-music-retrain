from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from smoke_stable_audio_trainer import _ensure_cara_conditioner, _prepare_baseline_training_config, _prepare_hf_auth
from cara_attribution_head import CaraHiddenStateTapper
from test_prep_common import base_metadata, parse_bool, write_report


REQUIRED_DISTRIBUTIONS = [
    "stable-audio-tools",
    "auraloss",
    "descript-audiotools",
    "descript-audio-codec",
    "encodec",
    "inf-cl",
    "laion-clap",
    "matplotlib",
    "pytorch-lightning",
    "torchmetrics",
    "wandb",
    "webdataset",
]

REQUIRED_MODULES = [
    "auraloss",
    "audiotools",
    "dac",
    "encodec",
    "laion_clap",
    "matplotlib.backends.backend_agg",
    "prefigure",
    "pytorch_lightning",
    "torchmetrics",
    "wandb",
    "webdataset",
    "stable_audio_tools.training.factory",
    "stable_audio_tools.training.diffusion",
    "stable_audio_tools.training.autoencoders",
    "stable_audio_tools.training.losses",
    "stable_audio_tools.training.losses.semantic",
    "stable_audio_tools.training.metrics.fad_metrics",
    "cara_attribution_head",
]


def _check_distributions(report: dict[str, Any]) -> None:
    report["distribution_versions"] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            report["distribution_versions"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report["distribution_versions"][distribution] = None
            report["errors"].append(f"Missing Python distribution: {distribution}")


def _check_module_imports(report: dict[str, Any]) -> None:
    report["module_imports"] = {}
    for module_name in REQUIRED_MODULES:
        started = time.time()
        try:
            importlib.import_module(module_name)
            report["module_imports"][module_name] = {
                "status": "ok",
                "seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            report["module_imports"][module_name] = {
                "status": "failed",
                "error": repr(exc),
                "seconds": round(time.time() - started, 3),
            }
            report["errors"].append(f"Import failed for {module_name}: {exc!r}")


def _run_wrapper_check(args: argparse.Namespace, report: dict[str, Any]) -> None:
    from stable_audio_tools.models.pretrained import get_pretrained_model
    from stable_audio_tools.training import create_training_wrapper_from_config

    report["stage"] = "prepare_hf_auth"
    _prepare_hf_auth(report)

    report["stage"] = "load_pretrained_model"
    started = time.time()
    model, model_config = get_pretrained_model(args.checkpoint)
    report["pretrained_load_seconds"] = round(time.time() - started, 3)
    report["checkpoint"] = args.checkpoint
    report["model_type"] = model_config.get("model_type")
    report["sample_rate"] = model_config.get("sample_rate")
    report["sample_size"] = model_config.get("sample_size")
    report["audio_channels"] = model_config.get("audio_channels", 2)

    report["stage"] = "prepare_baseline_training_config"
    _prepare_baseline_training_config(model_config, report)
    _ensure_cara_conditioner(
        model_config,
        {
            "pool_count": 2,
            "family_count": 2,
        },
        report,
        enabled=True,
    )
    report["training_config_keys"] = sorted(model_config.get("training", {}).keys())

    report["stage"] = "create_training_wrapper"
    started = time.time()
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    report["training_wrapper_create_seconds"] = round(time.time() - started, 3)
    report["training_wrapper_class"] = type(training_wrapper).__name__

    report["stage"] = "discover_cara_hidden_state_taps"
    tapper = CaraHiddenStateTapper(model)
    try:
        report["cara_hidden_state_taps"] = tapper.register()
    finally:
        tapper.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--wrapper_check", default="true")
    parser.add_argument("--dashboard_triggered", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "07a_stable_audio_trainer_preflight",
        "status": "failed",
        "stage": "start",
        "checkpoint": args.checkpoint,
        "wrapper_check": parse_bool(args.wrapper_check),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "errors": [],
        "warnings": [],
    }

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        report["stage"] = "check_distributions"
        _check_distributions(report)
        report["stage"] = "check_module_imports"
        _check_module_imports(report)
        if report["errors"]:
            raise RuntimeError("Stable Audio trainer preflight import checks failed.")
        if parse_bool(args.wrapper_check):
            _run_wrapper_check(args, report)
        report["stage"] = "passed"
        report["status"] = "passed"
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])

    metadata = base_metadata(
        test_name=report["test_name"],
        compute="cpu-prep-cluster",
        environment="azureml:env-stable-audio-tools:8",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="stable_audio_open_small",
        environment_name="env-stable-audio-tools",
        environment_version="8",
        import_status="ok" if not report["errors"] else "failed",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="stable_audio_trainer_preflight_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
