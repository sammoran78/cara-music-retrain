from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio

from smoke_stable_audio_trainer import _configure_disk_safe_runtime_dirs, _prepare_hf_auth
from test_prep_common import base_metadata, parse_bool, write_report


BASE_MODEL_ID = "base_musicgen_small"
CARA_MODEL_ID = "musicgen_cara_strong_full"
PUBLIC_MUSICGEN_PREFIXES = ("facebook/musicgen-", "facebook/magnet-")


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
    peak = audio.abs().max().clamp_min(1e-8)
    return (audio / peak).clamp(-1, 1)


def _expected_from_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cara_pool_id": row.get("expected_cara_pool_id"),
        "cara_pool_index": row.get("expected_cara_pool_index"),
        "cara_pool_family": row.get("expected_cara_pool_family"),
        "cara_pool_family_index": row.get("expected_cara_pool_family_index"),
        "cara_pool_codeword": row.get("expected_cara_pool_codeword"),
    }


def _load_delta(trained_model_data: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [
        trained_model_data / "checkpoints" / "musicgen_lm_cara_delta.pt",
        trained_model_data / "musicgen_lm_cara_delta.pt",
    ]
    candidates.extend(trained_model_data.rglob("musicgen_lm_cara_delta.pt"))
    for checkpoint_path in candidates:
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or payload.get("format") != "musicgen_lm_cara_delta_v1":
                raise RuntimeError(f"Unexpected MusicGen delta format in {checkpoint_path}")
            state_dict = payload.get("state_dict")
            if not isinstance(state_dict, dict) or not state_dict:
                raise RuntimeError(f"MusicGen delta has no state_dict: {checkpoint_path}")
            return checkpoint_path, payload
    raise RuntimeError(f"Missing MusicGen CARA delta under {trained_model_data}")


def _apply_delta_to_lm(lm: torch.nn.Module, trained_model_data: Path) -> dict[str, Any]:
    checkpoint_path, payload = _load_delta(trained_model_data)
    delta_state = payload["state_dict"]
    model_state = lm.state_dict()
    matched: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    unmatched: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for delta_key, tensor in delta_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        key = str(delta_key)
        if key.startswith("cara_suffix_head."):
            ignored.append(key)
            continue
        candidates = [key]
        if key.startswith("lm."):
            candidates.append(key[3:])
        target_key = next((candidate for candidate in candidates if candidate in model_state), None)
        if target_key is None:
            unmatched.append(key)
            continue
        expected = model_state[target_key]
        if tuple(tensor.shape) != tuple(expected.shape):
            shape_mismatches.append(
                {
                    "delta_key": key,
                    "target_key": target_key,
                    "delta_shape": list(tensor.shape),
                    "target_shape": list(expected.shape),
                }
            )
            continue
        matched[target_key] = tensor.to(dtype=expected.dtype)
    if not matched:
        raise RuntimeError(
            "MusicGen CARA delta loaded but no LM tensors matched the base model; refusing to benchmark the CARA lane as base MusicGen."
        )
    model_state.update(matched)
    lm.load_state_dict(model_state, strict=True)
    return {
        "path": str(checkpoint_path),
        "format": payload.get("format"),
        "base_checkpoint": payload.get("base_checkpoint"),
        "variant": payload.get("variant"),
        "global_step": payload.get("global_step"),
        "trainable_tensor_count": len(delta_state),
        "applied_lm_tensor_count": len(matched),
        "ignored_suffix_head_tensor_count": len(ignored),
        "unmatched_tensor_count": len(unmatched),
        "shape_mismatch_count": len(shape_mismatches),
        "unmatched_preview": unmatched[:20],
        "shape_mismatch_preview": shape_mismatches[:10],
    }


def _copy_artifact_if_present(source_root: Path, output_dir: Path, filename: str) -> str | None:
    candidates = [source_root / filename]
    candidates.extend(source_root.rglob(filename))
    for candidate in candidates:
        if candidate.exists():
            target = output_dir / filename
            shutil.copy2(candidate, target)
            return filename
    return None


def _prepare_musicgen_hf_auth(report: dict[str, Any], *, base_checkpoint: str) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        report["hf_auth_source"] = "environment"
        return
    if str(base_checkpoint).startswith(PUBLIC_MUSICGEN_PREFIXES):
        report["hf_auth_source"] = "public_checkpoint_no_token"
        report.setdefault("warnings", []).append(
            f"HF token was not available, but {base_checkpoint} is a public MusicGen checkpoint; continuing without Key Vault auth."
        )
        return
    _prepare_hf_auth(report)


def _load_musicgen_model(
    *,
    model_id: str,
    base_checkpoint: str,
    trained_model_data: Path,
    report: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    _prepare_musicgen_hf_auth(report, base_checkpoint=base_checkpoint)
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained(base_checkpoint, device="cuda")
    delta_report = None
    if model_id == CARA_MODEL_ID:
        delta_report = _apply_delta_to_lm(model.lm, trained_model_data)
    elif model_id != BASE_MODEL_ID:
        raise RuntimeError(f"Unsupported MusicGen benchmark model id: {model_id}")
    return model, {
        "model_id": model_id,
        "checkpoint": base_checkpoint,
        "device": "cuda",
        "sample_rate": int(getattr(model.compression_model, "sample_rate", 32000) or 32000),
        "trainable_delta": delta_report,
    }


def _generate_one(model: Any, *, prompt: str, seed: int, duration_seconds: float, top_k: int, cfg_coef: float) -> tuple[torch.Tensor, int]:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model.set_generation_params(duration=float(duration_seconds), top_k=int(top_k), cfg_coef=float(cfg_coef))
    with torch.no_grad():
        audio = model.generate([str(prompt or "CARA benchmark audio")], progress=True)
    sample_rate = int(getattr(model.compression_model, "sample_rate", 32000) or 32000)
    return _normalise_audio(audio), sample_rate


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
            CARA_MODEL_ID: {
                "status": "pending_musicgen_suffix_scorer",
                "reason": "Generated audio is saved; MusicGen suffix/head attribution scoring runs in the follow-on scorer.",
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
    if not prompt_manifest_file.exists():
        raise RuntimeError(f"Locked prompt manifest input is missing: {prompt_manifest_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("MusicGen generated-audio benchmark is GPU-only; CUDA is not available.")

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

    copied = {
        "cara_registry_resolver": _copy_artifact_if_present(trained_model_data, output_dir, "cara_registry_resolver.json"),
        "cara_suffix_vocab": _copy_artifact_if_present(trained_model_data, output_dir, "cara_suffix_vocab.json"),
    }
    if CARA_MODEL_ID in model_ids and (not copied["cara_registry_resolver"] or not copied["cara_suffix_vocab"]):
        raise RuntimeError("MusicGen CARA benchmark requires cara_registry_resolver.json and cara_suffix_vocab.json from the full run.")

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
            "duration_seconds": float(args.duration_seconds),
            "top_k": int(args.top_k),
            "cfg_coef": float(args.cfg_coef),
            "registered_artifacts": copied,
            "audio_output_policy": "WAV files are written by model_id/suite_id/prompt_id under audio/.",
        }
    )

    generation_rows: list[dict[str, Any]] = []
    model_load_reports: list[dict[str, Any]] = []
    for model_id in model_ids:
        model, model_report = _load_musicgen_model(
            model_id=model_id,
            base_checkpoint=args.base_checkpoint,
            trained_model_data=trained_model_data,
            report=report,
        )
        model_load_reports.append(model_report)
        for row_index, prompt_row in enumerate(selected_rows, start=1):
            prompt_id = _safe_id(prompt_row.get("prompt_id") or f"prompt-{row_index:04d}")
            suite_id = _safe_id(prompt_row.get("suite_id") or "suite")
            seed = int(prompt_row.get("seed") or 0)
            audio, sample_rate = _generate_one(
                model,
                prompt=str(prompt_row.get("prompt") or ""),
                seed=seed,
                duration_seconds=float(args.duration_seconds),
                top_k=int(args.top_k),
                cfg_coef=float(args.cfg_coef),
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
                    "duration_seconds": float(args.duration_seconds),
                    "top_k": int(args.top_k),
                    "cfg_coef": float(args.cfg_coef),
                    "expected": _expected_from_prompt_row(prompt_row),
                    "native_cara_prediction": None if model_id == BASE_MODEL_ID else {"status": "pending_musicgen_suffix_scorer"},
                    "external_probe_prediction": {"status": "pending_external_probe"},
                }
            )
            report["generated_audio_count"] = len(generation_rows)
        del model
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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_ids", required=True)
    parser.add_argument("--suite_ids", required=True)
    parser.add_argument("--seed_ids", default="0")
    parser.add_argument("--max_prompts", type=int, default=20)
    parser.add_argument("--scope", default="smoke")
    parser.add_argument("--base_checkpoint", default="facebook/musicgen-small")
    parser.add_argument("--duration_seconds", type=float, default=12.0)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--cfg_coef", type=float, default=3.0)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "17_benchmark_testing_musicgen_audio",
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
        environment="azureml:env-musicgen-audiocraft:3",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="musicgen",
        environment_name="env-musicgen-audiocraft",
        environment_version="3",
        import_status="ok" if not report["errors"] else "failed",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_musicgen_audio_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
