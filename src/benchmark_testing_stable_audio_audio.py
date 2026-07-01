from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio

from smoke_stable_audio_trainer import (
    _configure_disk_safe_runtime_dirs,
    _ensure_cara_conditioner,
    _ensure_context_conditioner,
    _patch_loaded_model_cara_conditioners,
    _patch_loaded_model_context_conditioners,
    _prepare_hf_auth,
)
from test_prep_common import base_metadata, parse_bool, write_report


BASE_MODEL_ID = "base_stable_audio_open_small"
DIFFUSION_MODEL_ID = "diffusion_cara_strong_full_modest_arch"
CONTEXT_DIFFUSION_MODEL_ID = "context_diffusion_cara_strong_full"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _safe_id(value: Any) -> str:
    text = str(value or "item").strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return safe[:160] or hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _select_prompt_rows(
    rows: list[dict[str, Any]],
    *,
    suite_ids: list[str],
    seed_ids: list[int],
    max_prompts: int,
) -> list[dict[str, Any]]:
    suite_set = set(suite_ids)
    seed_set = set(seed_ids)
    selected = [
        row
        for row in rows
        if str(row.get("suite_id") or "") in suite_set and int(row.get("seed") or 0) in seed_set
    ]
    selected.sort(key=lambda row: (str(row.get("suite_id") or ""), str(row.get("prompt_id") or "")))
    if max_prompts > 0:
        selected = selected[:max_prompts]
    return selected


def _normalise_audio(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().to(torch.float32).cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > audio.shape[-1]:
        audio = audio.T
    peak = audio.abs().max().clamp_min(1e-8)
    return (audio / peak).clamp(-1, 1)


def _load_trainable_delta(trained_model_data: Path) -> dict[str, Any]:
    checkpoint_path = trained_model_data / "checkpoints" / "trainable_delta.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing trainable delta checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "cara_trainable_delta_v1":
        raise RuntimeError(f"Unexpected trainable delta format in {checkpoint_path}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Trainable delta checkpoint has no state_dict: {checkpoint_path}")
    return payload


def _candidate_delta_keys(name: str) -> list[str]:
    candidates = [name]
    prefixes = ("model.", "model.model.", "diffusion.", "diffusion_model.", "module.")
    for prefix in prefixes:
        if name.startswith(prefix):
            candidates.append(name[len(prefix) :])
    return list(dict.fromkeys(candidates))


def _critical_unmatched_delta_tensors(unmatched: list[str]) -> list[str]:
    critical_prefixes = (
        "diffusion.conditioner.conditioners.cara_",
        "conditioner.conditioners.cara_",
        "model.conditioner.conditioners.cara_",
        "model.model.conditioner.conditioners.cara_",
    )
    return [key for key in unmatched if key.startswith(critical_prefixes)]


def _apply_trainable_delta_to_model(model: torch.nn.Module, trained_model_data: Path) -> dict[str, Any]:
    payload = _load_trainable_delta(trained_model_data)
    delta_state = payload["state_dict"]
    model_state = model.state_dict()
    suffix_index: dict[str, list[str]] = {}
    for key in model_state:
        parts = key.split(".")
        for start in range(max(0, len(parts) - 4), len(parts)):
            suffix = ".".join(parts[start:])
            suffix_index.setdefault(suffix, []).append(key)

    matched: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for delta_key, tensor in delta_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        target_key = None
        for candidate in _candidate_delta_keys(str(delta_key)):
            if candidate in model_state:
                target_key = candidate
                break
        if target_key is None:
            parts = str(delta_key).split(".")
            for start in range(max(0, len(parts) - 4), len(parts)):
                suffix = ".".join(parts[start:])
                matches = suffix_index.get(suffix) or []
                if len(matches) == 1:
                    target_key = matches[0]
                    break
        if target_key is None:
            unmatched.append(str(delta_key))
            continue
        expected = model_state[target_key]
        if tuple(tensor.shape) != tuple(expected.shape):
            shape_mismatches.append(
                {
                    "delta_key": str(delta_key),
                    "target_key": target_key,
                    "delta_shape": list(tensor.shape),
                    "target_shape": list(expected.shape),
                }
            )
            continue
        matched[target_key] = tensor.to(dtype=expected.dtype)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=True)
    critical_unmatched = _critical_unmatched_delta_tensors(unmatched)
    return {
        "format": payload.get("format"),
        "base_checkpoint": payload.get("base_checkpoint"),
        "global_step": payload.get("global_step"),
        "trainable_tensor_count": len(delta_state),
        "applied_tensor_count": len(matched),
        "unmatched_tensor_count": len(unmatched),
        "critical_unmatched_tensor_count": len(critical_unmatched),
        "shape_mismatch_count": len(shape_mismatches),
        "unmatched_preview": unmatched[:20],
        "critical_unmatched_preview": critical_unmatched[:20],
        "shape_mismatch_preview": shape_mismatches[:10],
    }


def _resolver_from_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_by_index: dict[int, str] = {}
    family_by_index: dict[int, str] = {}
    pool_to_family_index: dict[int, int] = {}
    for row in rows:
        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        pool_id = expected.get("cara_pool_id")
        pool_index = expected.get("cara_pool_index")
        family = expected.get("cara_pool_family")
        family_index = expected.get("cara_pool_family_index")
        if pool_id not in (None, "") and pool_index not in (None, ""):
            pool_by_index[int(pool_index)] = str(pool_id)
        if family not in (None, "") and family_index not in (None, ""):
            family_by_index[int(family_index)] = str(family)
        if pool_index not in (None, "") and family_index not in (None, ""):
            pool_to_family_index[int(pool_index)] = int(family_index)
    return {
        "format": "cara_generation_resolver_from_prompt_manifest_v1",
        "pool_count": len(pool_by_index),
        "family_count": len(family_by_index),
        "pool_by_index": {str(key): pool_by_index[key] for key in sorted(pool_by_index)},
        "family_by_index": {str(key): family_by_index[key] for key in sorted(family_by_index)},
        "pool_to_family_index": {str(key): pool_to_family_index[key] for key in sorted(pool_to_family_index)},
    }


def _load_registry_resolver_for_generation(trained_model_data: Path, prompt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        trained_model_data / "cara_registry_resolver.json",
        trained_model_data / "work" / "cara_registry_resolver.json",
        trained_model_data / "outputs" / "cara_registry_resolver.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    try:
        for candidate in trained_model_data.rglob("cara_registry_resolver.json"):
            return json.loads(candidate.read_text(encoding="utf-8"))
    except OSError:
        pass
    return _resolver_from_prompt_rows(prompt_rows)


def _load_stable_audio_model(
    *,
    model_id: str,
    base_checkpoint: str,
    trained_model_data: Path,
    context_trained_model_data: Path | None,
    resolver: dict[str, Any] | None = None,
    report: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    _prepare_hf_auth(report)
    from stable_audio_tools.models.pretrained import get_pretrained_model

    model, model_config = get_pretrained_model(base_checkpoint)
    delta_report: dict[str, Any] | None = None
    selected_trained_model_data = trained_model_data
    if model_id == CONTEXT_DIFFUSION_MODEL_ID:
        selected_trained_model_data = context_trained_model_data or trained_model_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if model_id in {DIFFUSION_MODEL_ID, CONTEXT_DIFFUSION_MODEL_ID}:
        if resolver is None:
            resolver = _load_registry_resolver_for_generation(selected_trained_model_data, [])
        conditioner_report: dict[str, Any] = {}
        _ensure_cara_conditioner(model_config, resolver, conditioner_report, enabled=True)
        _patch_loaded_model_cara_conditioners(model, model_config, resolver, conditioner_report)
        if model_id == CONTEXT_DIFFUSION_MODEL_ID:
            _ensure_context_conditioner(model_config, resolver, conditioner_report, enabled=True)
            _patch_loaded_model_context_conditioners(model, model_config, resolver, conditioner_report)
        report.setdefault("stable_audio_model_reconstruction", {})[model_id] = conditioner_report
        delta_report = _apply_trainable_delta_to_model(model, selected_trained_model_data)
        if int(delta_report.get("applied_tensor_count") or 0) <= 0:
            raise RuntimeError(
                "Trainable delta checkpoint loaded but no tensors matched the Stable Audio model; "
                "refusing to generate the fine-tuned lane as a second base-model run."
            )
        if int(delta_report.get("critical_unmatched_tensor_count") or 0) > 0:
            raise RuntimeError(
                "Trainable delta checkpoint contains CARA/context conditioner tensors that were not reattached "
                "to the Stable Audio model before inference: "
                f"{delta_report.get('critical_unmatched_preview')}"
            )
    elif model_id != BASE_MODEL_ID:
        raise RuntimeError(f"Unsupported Stable Audio benchmark model id: {model_id}")
    model.eval()
    return model, model_config, {
        "model_id": model_id,
        "checkpoint": base_checkpoint,
        "device": device,
        "sample_rate": model_config.get("sample_rate"),
        "sample_size": model_config.get("sample_size"),
        "trainable_delta": delta_report,
        "trained_model_data": str(selected_trained_model_data) if delta_report else None,
    }


def _generate_one(
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    prompt: str,
    conditioning_metadata: dict[str, Any] | None = None,
    seed: int,
    steps: int,
    cfg_scale: float,
) -> tuple[torch.Tensor, int, int]:
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    device = next(model.parameters()).device
    sample_rate = int(model_config.get("sample_rate") or 44100)
    sample_size = int(model_config.get("sample_size") or 524288)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    row_conditioning = {
        "prompt": str(prompt or "CARA benchmark audio"),
        "seconds_start": 0,
        "seconds_total": sample_size / sample_rate,
    }
    for key, value in (conditioning_metadata or {}).items():
        if value not in (None, ""):
            row_conditioning[key] = value
    conditioning = [row_conditioning]
    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            conditioning=conditioning,
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sample_size=sample_size,
            sample_rate=sample_rate,
            device=str(device),
        )
    return _normalise_audio(output), sample_rate, sample_size


def _expected_from_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("expected"), dict):
        expected = row["expected"]
        return {
            "cara_pool_id": expected.get("cara_pool_id"),
            "cara_pool_index": expected.get("cara_pool_index"),
            "cara_pool_family": expected.get("cara_pool_family"),
            "cara_pool_family_index": expected.get("cara_pool_family_index"),
            "cara_pool_codeword": expected.get("cara_pool_codeword"),
        }
    return {
        "cara_pool_id": row.get("expected_cara_pool_id"),
        "cara_pool_index": row.get("expected_cara_pool_index"),
        "cara_pool_family": row.get("expected_cara_pool_family"),
        "cara_pool_family_index": row.get("expected_cara_pool_family_index"),
        "cara_pool_codeword": row.get("expected_cara_pool_codeword"),
    }


def _structured_conditioning_from_prompt_row(row: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    if model_id == BASE_MODEL_ID:
        return {}
    expected = _expected_from_prompt_row(row)
    leakage_policy = row.get("leakage_policy") if isinstance(row.get("leakage_policy"), dict) else {}
    condition = str(row.get("condition") or "").strip()
    if condition == "open_quality" or not leakage_policy.get("pool_accuracy_applicable", True):
        return {}
    if leakage_policy.get("expected_label_is_shuffled") or condition == "shuffled_label":
        return {}
    if expected.get("cara_pool_index") in (None, "") or expected.get("cara_pool_family_index") in (None, ""):
        return {}
    metadata: dict[str, Any] = {
        "cara_pool_index": int(expected["cara_pool_index"]),
        "cara_pool_family_index": int(expected["cara_pool_family_index"]),
    }
    if model_id == CONTEXT_DIFFUSION_MODEL_ID:
        metadata.update(
            {
                "cara_context_pool_index": int(row.get("cara_context_pool_index", expected["cara_pool_index"])),
                "cara_context_pool_family_index": int(
                    row.get("cara_context_pool_family_index", expected["cara_pool_family_index"])
                ),
                "cara_context_policy_index": int(row.get("cara_context_policy_index", 0)),
                "cara_context_count": int(row.get("cara_context_count", 1)),
            }
        )
    return metadata


def _metrics_from_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = Counter(str(row.get("model_id") or "unknown") for row in rows)
    by_suite = Counter(str(row.get("suite_id") or "unknown") for row in rows)
    labelled = [row for row in rows if row.get("expected", {}).get("cara_pool_id")]
    return {
        "format": "cara_audio_benchmark_metrics_v1",
        "created_at": _utc_now(),
        "generated_audio_count": len(rows),
        "labelled_audio_count": len(labelled),
        "by_model": dict(sorted(by_model.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "native_cara_metrics": {
            BASE_MODEL_ID: {"status": "not_applicable", "reason": "Base checkpoint has no native CARA output channel."},
            DIFFUSION_MODEL_ID: {
                "status": "pending_attribution_extractor",
                "reason": "Generated audio is saved; native hidden-state/probe attribution extraction is a follow-on scoring pass.",
            },
            CONTEXT_DIFFUSION_MODEL_ID: {
                "status": "pending_attribution_extractor",
                "reason": "Generated audio is saved; context native hidden-state attribution extraction is a follow-on scoring pass.",
            },
        },
        "repairability": {
            "status": "pending_attribution_extractor",
            "reason": "Repairability buckets require real predicted CARA IDs and are not inferred from expected labels.",
        },
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["runtime_dirs"] = _configure_disk_safe_runtime_dirs(output_dir)
    prompt_manifest_file = Path(args.prompt_manifest_file)
    trained_model_data = Path(args.trained_model_data)
    context_trained_model_data = Path(args.context_trained_model_data) if str(args.context_trained_model_data or "").strip() else None
    if not prompt_manifest_file.exists():
        raise RuntimeError(f"Locked prompt manifest input is missing: {prompt_manifest_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("Generated-audio benchmark is GPU-only; CUDA is not available.")

    model_ids = _split_csv(args.model_ids)
    suite_ids = _split_csv(args.suite_ids)
    seed_ids = [int(seed) for seed in _split_csv(args.seed_ids)]
    prompt_rows = _read_jsonl(prompt_manifest_file)
    selected_rows = _select_prompt_rows(
        prompt_rows,
        suite_ids=suite_ids,
        seed_ids=seed_ids,
        max_prompts=max(0, int(args.max_prompts)),
    )
    if not selected_rows:
        raise RuntimeError("No prompt rows matched the selected suites/seeds.")

    report.update(
        {
            "stage": "generate_audio",
            "prompt_manifest_file": str(prompt_manifest_file),
            "prompt_manifest_rows": len(prompt_rows),
            "selected_prompt_rows": len(selected_rows),
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seed_ids": seed_ids,
            "max_prompts": int(args.max_prompts),
            "scope": args.scope,
            "generation_steps": int(args.generation_steps),
            "cfg_scale": float(args.cfg_scale),
            "audio_output_policy": "WAV files are written by model_id/suite_id/prompt_id under audio/.",
        }
    )

    generation_rows: list[dict[str, Any]] = []
    model_load_reports: list[dict[str, Any]] = []
    for model_id in model_ids:
        selected_trained_model_data = context_trained_model_data if model_id == CONTEXT_DIFFUSION_MODEL_ID and context_trained_model_data is not None else trained_model_data
        resolver = _load_registry_resolver_for_generation(selected_trained_model_data, selected_rows)
        model, model_config, model_report = _load_stable_audio_model(
            model_id=model_id,
            base_checkpoint=args.base_checkpoint,
            trained_model_data=trained_model_data,
            context_trained_model_data=context_trained_model_data,
            resolver=resolver,
            report=report,
        )
        model_load_reports.append(model_report)
        for row_index, prompt_row in enumerate(selected_rows, start=1):
            prompt_id = _safe_id(prompt_row.get("prompt_id") or f"prompt-{row_index:04d}")
            suite_id = _safe_id(prompt_row.get("suite_id") or "suite")
            seed = int(prompt_row.get("seed") or 0)
            structured_conditioning = _structured_conditioning_from_prompt_row(prompt_row, model_id=model_id)
            audio, sample_rate, sample_size = _generate_one(
                model=model,
                model_config=model_config,
                prompt=str(prompt_row.get("prompt") or ""),
                conditioning_metadata=structured_conditioning,
                seed=seed,
                steps=int(args.generation_steps),
                cfg_scale=float(args.cfg_scale),
            )
            relative_audio_path = Path("audio") / _safe_id(model_id) / suite_id / f"{prompt_id}.wav"
            audio_path = output_dir / relative_audio_path
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(audio_path), audio, sample_rate)
            generation_rows.append(
                {
                    "format": "cara_generated_audio_row_v1",
                    "created_at": _utc_now(),
                    "model_id": model_id,
                    "suite_id": prompt_row.get("suite_id"),
                    "prompt_id": prompt_row.get("prompt_id"),
                    "seed": seed,
                    "prompt": prompt_row.get("prompt"),
                    "audio_path": str(relative_audio_path),
                    "sample_rate": sample_rate,
                    "sample_size": sample_size,
                    "generation_steps": int(args.generation_steps),
                    "cfg_scale": float(args.cfg_scale),
                    "expected": _expected_from_prompt_row(prompt_row),
                    "structured_conditioning": structured_conditioning,
                    "structured_conditioning_policy": {
                        "supplied": bool(structured_conditioning),
                        "source": "expected benchmark label fields",
                        "withheld_from_text_only": not bool((prompt_row.get("leakage_policy") or {}).get("prompt_contains_visible_cara")),
                        "shuffled_label_control_suppressed": bool(
                            (prompt_row.get("leakage_policy") or {}).get("expected_label_is_shuffled")
                            or str(prompt_row.get("condition") or "") == "shuffled_label"
                        ),
                    },
                    "native_cara_prediction": None if model_id == BASE_MODEL_ID else {"status": "pending_attribution_extractor"},
                    "external_probe_prediction": {"status": "pending_external_probe"},
                }
            )
            report["generated_audio_count"] = len(generation_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["model_loads"] = model_load_reports
    report["generated_audio_count"] = len(generation_rows)
    _write_jsonl(output_dir / "generation_manifest.jsonl", generation_rows)
    _write_json(output_dir / "benchmark_audio_metrics.json", _metrics_from_manifest(generation_rows))
    _write_json(
        output_dir / "benchmark_audio_plan.json",
        {
            "format": "cara_audio_benchmark_plan_v1",
            "created_at": _utc_now(),
            "scope": args.scope,
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seed_ids": seed_ids,
            "max_prompts": int(args.max_prompts),
            "prompt_manifest_file": str(prompt_manifest_file),
            "generation_manifest": "generation_manifest.jsonl",
            "metrics": "benchmark_audio_metrics.json",
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    )
    report["status"] = "passed"
    report["stage"] = "completed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_manifest_file", required=True)
    parser.add_argument("--trained_model_data", required=True)
    parser.add_argument("--context_trained_model_data", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_ids", required=True)
    parser.add_argument("--suite_ids", required=True)
    parser.add_argument("--seed_ids", default="0")
    parser.add_argument("--max_prompts", type=int, default=20)
    parser.add_argument("--scope", default="smoke")
    parser.add_argument("--base_checkpoint", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--generation_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "15_benchmark_testing_stable_audio_audio",
        "status": "failed",
        "stage": "initializing",
        "created_at": _utc_now(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
        "errors": [],
        "warnings": [],
    }
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        run(args, report)
    except Exception as exc:
        report["errors"].append(repr(exc))
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])

    metadata = base_metadata(
        test_name=report["test_name"],
        compute="gpu-smoke-h100",
        environment="azureml:env-stable-audio-tools:8",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="stable_audio_open_small",
        environment_name="env-stable-audio-tools",
        environment_version="8",
        import_status="ok" if not report["errors"] else "failed",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_stable_audio_audio_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
