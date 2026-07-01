from __future__ import annotations

import json
import numbers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = "CARA audio attribution survival"
PHASE = "test-prep"
AZURE_SUBSCRIPTION_ID = "2aa6790f-891f-4a9b-8b7c-476ac25b0f82"
RESOURCE_GROUP = "rg-cara-audio-training-aue"
REGION = "australiaeast"
WORKSPACE = "rg-cara-audio-training-aue"
DATASTORE = "ds_cara_raw_audio"
INPUT_PATH = "azureml://datastores/ds_cara_raw_audio/paths/test-audio/"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def dataset_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "data" / "freesound",
        root / "data" / "freesound_meta" / "test-manifest" / "tracks.csv",
    )


def list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    return sorted(path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def audio_path_warnings(audio_dir: Path) -> list[str]:
    if not audio_dir.exists():
        return []
    suspicious = sorted(
        str(path)
        for path in audio_dir.rglob("*")
        if path.is_file() and (path.name.endswith(")") or path.suffix.lower() not in AUDIO_SUFFIXES)
    )
    if not suspicious:
        return []
    examples = ", ".join(suspicious[:5])
    remainder = f" and {len(suspicious) - 5} more" if len(suspicious) > 5 else ""
    return [f"Audio folder contains unsupported or malformed filenames: {examples}{remainder}."]


def manifest_schema_warnings(columns: list[str]) -> list[str]:
    warnings = []
    stable_names = {"title", "local_audio_path", "source_file_path", "freesound_id", "subset_role"}
    unnamed_count = sum(str(column).startswith("Unnamed:") for column in columns)
    if columns and not stable_names.intersection(columns):
        warnings.append(
            "Manifest columns do not include expected stable field names. "
            "The CSV may be headerless or may have been exported without its schema row."
        )
    if unnamed_count >= 5:
        warnings.append(f"Manifest exposes {unnamed_count} unnamed columns; inspect the uploaded CSV export.")
    return warnings


def base_metadata(
    *,
    test_name: str,
    compute: str,
    environment: str,
    dashboard_triggered: bool,
    report: dict[str, Any],
    model_family: str | None = None,
    environment_name: str | None = None,
    environment_version: str | None = None,
    import_status: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "project": PROJECT,
        "phase": PHASE,
        "test_name": test_name,
        "azure_subscription_id": AZURE_SUBSCRIPTION_ID,
        "resource_group": RESOURCE_GROUP,
        "region": REGION,
        "workspace": WORKSPACE,
        "datastore": DATASTORE,
        "input_path": INPUT_PATH,
        "compute": compute,
        "environment": environment,
        "dashboard_triggered": dashboard_triggered,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": report.get("status", "failed"),
    }
    for key in ("gpu_name", "cuda_available", "cuda_device_count"):
        if key in report:
            metadata[key] = report[key]
    if model_family:
        metadata["model_family"] = model_family
    if environment_name:
        metadata["environment_name"] = environment_name
    if environment_version:
        metadata["environment_version"] = environment_version
    if import_status is not None:
        metadata["import_status"] = import_status
    return metadata


def try_log_mlflow(report: dict[str, Any], report_path: Path, metadata_path: Path) -> str | None:
    try:
        import mlflow

        for key in ("test_name", "status", "gpu_name", "import_status"):
            value = report.get(key)
            if value is not None:
                mlflow.log_param(key, value)
        for key in ("audio_file_count", "manifest_rows", "cuda_device_count"):
            value = report.get(key)
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        if "cuda_available" in report:
            mlflow.log_metric("cuda_available", int(bool(report["cuda_available"])))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metadata_path))
        return None
    except Exception as exc:  # MLflow is audit enrichment, never a test dependency.
        return repr(exc)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            tensor = value.detach().cpu()
            return tensor.item() if getattr(tensor, "ndim", 1) == 0 else tensor.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return value


def write_report(
    output_dir: Path,
    report: dict[str, Any],
    metadata: dict[str, Any],
    *,
    report_alias: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    metadata_path = output_dir / "metadata.json"
    safe_report = json_safe(report)
    safe_metadata = json_safe(metadata)
    report_text = json.dumps(safe_report, indent=2, sort_keys=True)
    metadata_text = json.dumps(safe_metadata, indent=2, sort_keys=True)
    report_path.write_text(report_text, encoding="utf-8")
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / report_alias).write_text(report_text, encoding="utf-8")
    metadata_path.write_text(metadata_text, encoding="utf-8")
    mlflow_error = try_log_mlflow(safe_report, report_path, metadata_path)
    if mlflow_error:
        report["mlflow_logging_error"] = mlflow_error
        safe_report = json_safe(report)
        report_text = json.dumps(safe_report, indent=2, sort_keys=True)
        report_path.write_text(report_text, encoding="utf-8")
        (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
        (output_dir / report_alias).write_text(report_text, encoding="utf-8")
    print(json.dumps(safe_report, indent=2, sort_keys=True))
