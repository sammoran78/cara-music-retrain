from __future__ import annotations

import argparse
import importlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from test_prep_common import base_metadata, parse_bool, write_report


REQUIRED_LABEL_FIELDS = [
    "cara_source_pool_id",
    "cara_pool_id",
    "cara_pool_index",
    "cara_pool_family",
    "cara_pool_family_index",
]

RAW_SOURCE_REQUIRED_FIELDS = [
    "cara_source_pool_id",
    "cara_pool_family",
]


def _import_status(module_name: str, attr_name: str | None = None) -> dict[str, Any]:
    started = time.time()
    try:
        module = importlib.import_module(module_name)
        attr_ok = True
        if attr_name:
            getattr(module, attr_name)
        return {
            "status": "ok",
            "version": getattr(module, "__version__", None),
            "attr": attr_name,
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "attr": attr_name,
            "error": repr(exc),
            "seconds": round(time.time() - started, 3),
        }


def _load_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _read_jsonl(path: Path, limit: int = 5000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _manifest_path(root: Path) -> Path:
    candidates = [
        root / "data" / "cara_pool_manifest_v2.jsonl",
        root / "cara_pool_manifest_v2.jsonl",
        root / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _audio_root(root: Path) -> Path:
    candidates = [
        root / "data" / "freesound",
        root / "freesound",
        root / "audio",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows = _normalise_cara_rows(rows)
    missing_by_field = {field: 0 for field in REQUIRED_LABEL_FIELDS}
    distinct: dict[str, set[str]] = {field: set() for field in REQUIRED_LABEL_FIELDS}
    raw_missing_by_field = {field: 0 for field in RAW_SOURCE_REQUIRED_FIELDS}
    for row in rows:
        for field in RAW_SOURCE_REQUIRED_FIELDS:
            value = row.get(field)
            if value in (None, ""):
                raw_missing_by_field[field] += 1
    for row in normalized_rows:
        for field in REQUIRED_LABEL_FIELDS:
            value = row.get(field)
            if value in (None, ""):
                missing_by_field[field] += 1
            else:
                distinct[field].add(str(value))
    return {
        "sampled_rows": len(rows),
        "required_fields": REQUIRED_LABEL_FIELDS,
        "missing_by_field": missing_by_field,
        "raw_source_required_fields": RAW_SOURCE_REQUIRED_FIELDS,
        "raw_source_missing_by_field": raw_missing_by_field,
        "distinct_counts": {field: len(values) for field, values in distinct.items()},
        "all_sampled_rows_have_required_labels": bool(rows) and all(count == 0 for count in missing_by_field.values()),
        "derived_label_contract": bool(rows) and all(count == 0 for count in missing_by_field.values()),
        "label_derivation": "cara_pool_id/cara_pool_index/family_index are derived from cara_source_pool_id and cara_pool_family when absent in the raw source manifest.",
    }


def _pool_id(row: dict[str, Any]) -> str:
    return str(row.get("cara_pool_id") or row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id") or "").strip()


def _pool_family(row: dict[str, Any]) -> str:
    return str(row.get("cara_pool_family") or row.get("cara_v2_pool_family") or row.get("primary_genre") or "Unknown").strip() or "Unknown"


def _normalise_cara_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool_ids = sorted({_pool_id(row) for row in rows if _pool_id(row)})
    families = sorted({_pool_family(row) for row in rows if _pool_family(row)})
    pool_index = {pool_id: index for index, pool_id in enumerate(pool_ids)}
    family_index = {family: index for index, family in enumerate(families)}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        pool_id = _pool_id(row)
        family = _pool_family(row)
        clone = dict(row)
        if pool_id:
            if clone.get("cara_source_pool_id") in (None, ""):
                clone["cara_source_pool_id"] = pool_id
            if clone.get("cara_pool_id") in (None, ""):
                clone["cara_pool_id"] = pool_id
            if clone.get("cara_pool_index") in (None, ""):
                clone["cara_pool_index"] = pool_index[pool_id]
        if family:
            if clone.get("cara_pool_family") in (None, ""):
                clone["cara_pool_family"] = family
            if clone.get("cara_pool_family_index") in (None, ""):
                clone["cara_pool_family_index"] = family_index[family]
        normalized.append(clone)
    return normalized


def _checkpoint_probe(checkpoint: str) -> dict[str, Any]:
    torch = _load_torch()
    if torch is None:
        return {"status": "failed", "checkpoint": checkpoint, "error": "torch import failed before checkpoint probe"}
    try:
        from diffusers import AceStepPipeline

        started = time.time()
        pipe = AceStepPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, local_files_only=False)
        return {
            "status": "ok",
            "checkpoint": checkpoint,
            "pipeline_class": pipe.__class__.__name__,
            "sample_rate": getattr(pipe, "sample_rate", None),
            "latents_per_second": getattr(pipe, "latents_per_second", None),
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {"status": "failed", "checkpoint": checkpoint, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="ACE-Step/Ace-Step1.5")
    parser.add_argument("--planner_checkpoint", default="ACE-Step/acestep-5Hz-lm-0.6B")
    parser.add_argument("--dit_variant", default="base_or_sft_dit")
    parser.add_argument("--load_checkpoint", default="false")
    parser.add_argument("--dashboard_triggered", default="false")
    args = parser.parse_args()

    root = Path(args.input_data)
    audio_dir = _audio_root(root)
    manifest_path = _manifest_path(root)
    rows = _read_jsonl(manifest_path)
    torch = _load_torch()
    imports = {
        "torch": _import_status("torch"),
        "diffusers": _import_status("diffusers"),
        "diffusers.AceStepPipeline": _import_status("diffusers", "AceStepPipeline"),
        "transformers": _import_status("transformers"),
        "peft": _import_status("peft"),
        "safetensors": _import_status("safetensors"),
        "soundfile": _import_status("soundfile"),
        "librosa": _import_status("librosa"),
        "acestep": _import_status("acestep"),
    }
    label_summary = _label_summary(rows)
    report: dict[str, Any] = {
        "test_name": "13_ace_step_env_preflight",
        "status": "failed",
        "checkpoint": args.checkpoint,
        "planner_checkpoint": args.planner_checkpoint,
        "dit_variant": args.dit_variant,
        "target_comparison_model": {
            "architecture": "hybrid_lm_planner_plus_dit_synthesizer",
            "planner_size": "0.6B",
            "planner_checkpoint": args.planner_checkpoint,
            "dit_variant": args.dit_variant,
            "comparison_goal": "Comparable-size ACE-Step v1.5 Hybrid CARA-Strong arm beside Diffusion, Context Diffusion, and MusicGen.",
        },
        "load_checkpoint": parse_bool(args.load_checkpoint),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "cuda_device_count": int(torch.cuda.device_count()) if torch is not None else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() else None,
        "imports": imports,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "input_root": str(root),
        "audio_dir": str(audio_dir),
        "audio_dir_exists": audio_dir.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_label_summary": label_summary,
        "checkpoint_probe": None,
        "architecture_contract": {
            "model_family": "ace_step",
            "sample_rate_hz": 48000,
            "latent_rate_hz": 25,
            "comparison_role": "hybrid 0.6B LM planner plus DiT synthesizer",
            "evidence_mode": "environment and source-manifest preflight only",
            "planner_survival_required": True,
            "dit_hidden_state_attribution_required": True,
        },
        "errors": [],
        "warnings": [],
    }

    if imports["torch"]["status"] != "ok" or torch is None:
        report["errors"].append("torch import failed; env-ace-step must include a working PyTorch CUDA build.")
    if not report["cuda_available"] or report["cuda_device_count"] < 1:
        report["errors"].append("CUDA device is required for ACE-Step preflight.")
    if imports["diffusers.AceStepPipeline"]["status"] != "ok":
        report["errors"].append("diffusers.AceStepPipeline import failed; ACE-Step DiT path is not available in this environment.")
    if imports["acestep"]["status"] != "ok":
        report["warnings"].append("acestep package import failed; this is acceptable for a Diffusers-first preflight but Side-Step/source-repo training probes will require it later.")
    if not report["ffmpeg_path"]:
        report["errors"].append("ffmpeg is not available on PATH.")
    if not report["manifest_exists"]:
        report["errors"].append("CARA source manifest was not found in the mounted input folder.")
    if not label_summary["all_sampled_rows_have_required_labels"]:
        report["errors"].append("Sampled manifest rows cannot be normalized into the required CARA pool/family label contract.")
    if not report["audio_dir_exists"]:
        report["errors"].append("Audio directory was not found in the mounted input folder.")

    if parse_bool(args.load_checkpoint):
        report["checkpoint_probe"] = _checkpoint_probe(args.checkpoint)
        if report["checkpoint_probe"]["status"] != "ok":
            report["errors"].append("ACE-Step checkpoint probe failed.")

    if not report["errors"]:
        report["status"] = "passed"

    metadata = base_metadata(
        test_name=report["test_name"],
        compute="gpu-smoke-h100",
        environment="azureml:env-ace-step:5",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="ace_step",
        environment_name="env-ace-step",
        environment_version="5",
        import_status=imports["diffusers.AceStepPipeline"]["status"],
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="ace_step_env_preflight_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
