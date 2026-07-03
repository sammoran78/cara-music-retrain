from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchaudio

from smoke_stable_audio_trainer import _configure_disk_safe_runtime_dirs
from test_prep_common import base_metadata, parse_bool, write_report


CARA_MODEL_ID = "hybrid_ace_step_cara_strong_full"
ACE_ADAPTER_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin"}
PUBLIC_ACE_PREFIXES = ("ACE-Step/", "ace-step/")


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


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
    return selected[:max_prompts] if max_prompts > 0 else selected


def _normalise_audio(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().to(torch.float32).cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    peak = audio.abs().max().clamp_min(1e-8)
    return (audio / peak).clamp(-1, 1)


def _preview_files(root: Path, *, limit: int = 40) -> list[str]:
    if not root.exists():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root)))
            if len(files) >= limit:
                break
    return files


def _checkpoint_layout(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {"provided": False}
    turbo = checkpoint_dir / "acestep-v15-turbo"
    base = checkpoint_dir / "acestep-v15-base"
    layout = {
        "provided": True,
        "path": str(checkpoint_dir),
        "exists": checkpoint_dir.exists(),
        "has_model_index_json": (checkpoint_dir / "model_index.json").exists(),
        "has_config_json": (checkpoint_dir / "config.json").exists(),
        "has_vae": (checkpoint_dir / "vae").exists(),
        "has_qwen_embedding": (checkpoint_dir / "Qwen3-Embedding-0.6B").exists(),
        "has_turbo_dit": (turbo / "config.json").exists(),
        "has_base_dit": (base / "config.json").exists(),
        "file_preview": _preview_files(checkpoint_dir),
    }
    layout["ace_bundle_layout"] = bool(
        layout["has_config_json"]
        and layout["has_vae"]
        and layout["has_qwen_embedding"]
        and (layout["has_turbo_dit"] or layout["has_base_dit"])
    )
    return layout


def _safe_link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _stage_ace_transformers_checkpoint(checkpoint_dir: Path, stage_dir: Path, layout: dict[str, Any]) -> Path:
    """Build a writable Transformers-compatible view of the ACE bundle.

    The public ACE-Step v1.5 bundle stores the custom model code and DiT weights
    under `acestep-v15-turbo/`, while the root config points to those modules as
    if they were root-level files. Azure mounts are read-only, so we create a
    tiny staged directory with symlinks instead of modifying the mounted bundle.
    """

    if not layout.get("ace_bundle_layout"):
        return checkpoint_dir
    variant = "acestep-v15-turbo" if layout.get("has_turbo_dit") else "acestep-v15-base"
    variant_dir = checkpoint_dir / variant
    if not variant_dir.exists():
        raise RuntimeError(f"ACE checkpoint layout selected {variant}, but the directory is missing: {variant_dir}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    for path in checkpoint_dir.iterdir():
        if path.name == ".cache":
            continue
        _safe_link_or_copy(path, stage_dir / path.name)

    for name in (
        "configuration_acestep_v15.py",
        "modeling_acestep_v15_turbo.py",
        "model.safetensors",
        "silence_latent.pt",
    ):
        source = variant_dir / name
        if source.exists():
            _safe_link_or_copy(source, stage_dir / name)

    staged_required = [
        stage_dir / "config.json",
        stage_dir / "configuration_acestep_v15.py",
        stage_dir / "modeling_acestep_v15_turbo.py",
        stage_dir / "model.safetensors",
        stage_dir / "vae",
        stage_dir / "Qwen3-Embedding-0.6B",
    ]
    missing = [str(path) for path in staged_required if not path.exists()]
    if missing:
        raise RuntimeError(f"Could not stage a Transformers-compatible ACE checkpoint view; missing: {missing}")
    return stage_dir


def _expected_from_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected")
    if isinstance(expected, dict):
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


def _find_adapter_artifacts(trained_model_data: Path) -> list[Path]:
    preferred_names = {
        "trainable_delta.pt",
        "adapter_model.safetensors",
        "pytorch_lora_weights.safetensors",
        "lora.safetensors",
    }
    candidates = [
        path
        for path in trained_model_data.rglob("*")
        if path.is_file() and path.suffix.lower() in ACE_ADAPTER_SUFFIXES
    ]
    candidates.sort(key=lambda path: (0 if path.name in preferred_names else 1, len(path.parts), str(path)))
    return candidates


def _read_trainable_delta(trained_model_data: Path) -> dict[str, Any]:
    delta_path = trained_model_data / "checkpoints" / "trainable_delta.pt"
    if not delta_path.exists():
        raise RuntimeError(f"Missing ACE-Step trainable delta checkpoint: {delta_path}")
    try:
        payload = torch.load(delta_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(delta_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"ACE-Step trainable delta is not a dictionary payload: {delta_path}")
    if payload.get("format") != "cara_ace_trainable_delta_v1":
        raise RuntimeError(f"Unexpected ACE-Step trainable delta format in {delta_path}: {payload.get('format')}")
    if payload.get("delta_type") != "sidestep_lora_adapter_delta":
        raise RuntimeError(
            "ACE-Step benchmark requires the deployable Side-Step LoRA adapter delta. "
            f"Found delta_type={payload.get('delta_type')!r}; refusing to benchmark a contract-only or base checkpoint lane."
        )
    return payload


def _adapter_candidates_from_delta(trained_model_data: Path, delta_payload: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for item in delta_payload.get("adapter_artifacts") or []:
        if not isinstance(item, dict):
            continue
        rel = item.get("relative_path")
        raw = item.get("path")
        if rel:
            candidates.append(trained_model_data / str(rel))
        elif raw:
            path = Path(str(raw))
            candidates.append(path if path.is_absolute() else trained_model_data / path)
    candidates.extend(_find_adapter_artifacts(trained_model_data))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if (
            key in seen
            or path.suffix.lower() not in ACE_ADAPTER_SUFFIXES
            or path.name == "trainable_delta.pt"
        ):
            continue
        seen.add(key)
        unique.append(path)
    unique.sort(key=lambda path: (0 if path.exists() else 1, len(path.parts), str(path)))
    return unique


def _candidate_adapter_roots(adapter_path: Path) -> list[Path]:
    roots = [adapter_path.parent]
    if adapter_path.parent.name in {"checkpoint", "checkpoints"}:
        roots.append(adapter_path.parent.parent)
    roots.append(adapter_path)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _prepare_ace_hf_auth(report: dict[str, Any], *, checkpoint: str) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        report["hf_auth_source"] = "environment"
        return

    vault_url = os.environ.get("KEY_VAULT_URL")
    secret_name = os.environ.get("HF_TOKEN_SECRET_NAME", "hf-token")
    if vault_url:
        try:
            from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
            from azure.keyvault.secrets import SecretClient

            credential = AzureMLOnBehalfOfCredential()
            secret_client = SecretClient(vault_url=vault_url, credential=credential)
            token = secret_client.get_secret(secret_name).value
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
                report["hf_auth_source"] = "workspace_key_vault"
                return
            report["hf_auth_source"] = "key_vault_empty"
        except Exception as exc:
            report["hf_auth_source"] = "key_vault_failed_nonfatal_for_public_checkpoint"
            report.setdefault("warnings", []).append(
                f"Could not retrieve HF token secret {secret_name!r} from Key Vault ({exc!r}); continuing because {checkpoint} is treated as a public ACE checkpoint."
            )
    else:
        report["hf_auth_source"] = "missing_token_public_checkpoint"

    if not str(checkpoint).startswith(PUBLIC_ACE_PREFIXES):
        raise RuntimeError(
            f"HF token is unavailable and checkpoint {checkpoint!r} is not in the public ACE-Step prefix allowlist."
        )


def _load_transformers_pipeline(checkpoint_ref: str) -> Any:
    raise RuntimeError(
        "ACE-Step audio benchmarking cannot use Transformers pipeline('text-to-audio') "
        "for the mounted ACE-Step 1.5 bundle. The bundle exposes a custom "
        "AceStepConditionGenerationModel that returns acoustic latents, while the "
        "benchmark needs the Diffusers AceStepPipeline so VAE decoding produces WAV audio. "
        f"Convert the checkpoint bundle to Diffusers format first: {checkpoint_ref}"
    )


def _convert_ace_bundle_to_diffusers(checkpoint_dir: Path, output_dir: Path, *, dit_config: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "model_index.json").exists():
        return {
            "status": "already_converted",
            "checkpoint_dir": str(checkpoint_dir),
            "dit_config": dit_config,
            "output_dir": str(output_dir),
            "output_preview": _preview_files(output_dir, limit=80),
        }

    started = time.time()
    try:
        from ace_step_diffusers_converter import convert_ace_step_weights
    except Exception as exc:
        raise RuntimeError(
            "ACE-Step Diffusers converter helper could not be imported. "
            "This benchmark requires the local src/ace_step_diffusers_converter.py helper."
        ) from exc

    convert_ace_step_weights(
        checkpoint_dir=str(checkpoint_dir),
        dit_config=dit_config,
        output_dir=str(output_dir),
        dtype_str="bf16",
    )
    if not (output_dir / "model_index.json").exists():
        raise RuntimeError(
            "ACE-Step checkpoint conversion finished without writing model_index.json; "
            f"converted output is not a Diffusers pipeline: {output_dir}"
        )
    return {
        "status": "converted",
        "checkpoint_dir": str(checkpoint_dir),
        "dit_config": dit_config,
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_preview": _preview_files(output_dir, limit=80),
    }


def _load_diffusers_pipeline(checkpoint_ref: str, report: dict[str, Any]) -> Any:
    report.setdefault("runtime_patches", []).append(_patch_ace_step_bool_mask_sort())
    from diffusers import AceStepPipeline

    pipe = AceStepPipeline.from_pretrained(checkpoint_ref, torch_dtype=torch.bfloat16, local_files_only=Path(checkpoint_ref).exists())
    return pipe.to("cuda")


def _mask_sort_key(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask.to(torch.int64)
    return mask


def _patch_ace_step_bool_mask_sort() -> dict[str, Any]:
    try:
        from diffusers.pipelines.ace_step import modeling_ace_step
    except Exception as exc:
        return {"name": "diffusers_ace_bool_mask_sort", "status": "not_available", "error": repr(exc)}

    if getattr(modeling_ace_step, "_cara_bool_mask_sort_patch", False):
        return {"name": "diffusers_ace_bool_mask_sort", "status": "already_applied"}

    original = getattr(modeling_ace_step, "_pack_sequences", None)
    if original is None:
        return {"name": "diffusers_ace_bool_mask_sort", "status": "missing_pack_sequences"}

    def _safe_pack_sequences(
        hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_cat = torch.cat([hidden1, hidden2], dim=1)
        mask_cat = torch.cat([mask1, mask2], dim=1)
        batch, length, width = hidden_cat.shape
        sort_idx = _mask_sort_key(mask_cat).argsort(dim=1, descending=True, stable=True)
        hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(batch, length, width))
        lengths = mask_cat.to(torch.int64).sum(dim=1)
        new_mask = torch.arange(length, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1)
        return hidden_left, new_mask

    modeling_ace_step._pack_sequences = _safe_pack_sequences
    modeling_ace_step._cara_bool_mask_sort_patch = True
    modeling_ace_step._cara_original_pack_sequences = original
    return {"name": "diffusers_ace_bool_mask_sort", "status": "applied"}


def _unwrap_peft_native_model(wrapped: Any) -> tuple[Any, str]:
    if hasattr(wrapped, "set_adapter"):
        try:
            wrapped.set_adapter("cara_hybrid")
        except Exception:
            pass
    if hasattr(wrapped, "get_base_model"):
        native = wrapped.get_base_model()
        if native is not None:
            return native, "get_base_model"
    base_model = getattr(wrapped, "base_model", None)
    native = getattr(base_model, "model", None)
    if native is not None:
        return native, "base_model.model"
    return wrapped, "wrapped"


def _load_adapter_into_pipeline(pipe: Any, adapter_candidates: list[Path]) -> dict[str, Any]:
    adapter_report: dict[str, Any] = {
        "candidate_count": len(adapter_candidates),
        "candidate_preview": [str(path) for path in adapter_candidates[:20]],
        "loaded": False,
    }
    load_errors: list[dict[str, Any]] = []
    if not adapter_candidates:
        raise RuntimeError("ACE-Step Hybrid benchmark requires a deployable adapter artifact; no adapter candidates were found.")

    for adapter_path in adapter_candidates:
        if not adapter_path.exists():
            load_errors.append({"path": str(adapter_path), "reason": "missing"})
            continue
        for adapter_root in _candidate_adapter_roots(adapter_path):
            try:
                if hasattr(pipe, "load_lora_weights") and adapter_path.suffix.lower() not in {".pt", ".pth"}:
                    pipe.load_lora_weights(str(adapter_path.parent), weight_name=adapter_path.name)
                    adapter_report.update(
                        {
                            "selected": str(adapter_path),
                            "loaded": True,
                            "load_method": "diffusers_load_lora_weights",
                            "load_errors": load_errors[:20],
                        }
                    )
                    return adapter_report

                for attr_name in ("model", "transformer", "unet"):
                    model = getattr(pipe, attr_name, None)
                    if model is None:
                        continue
                    if hasattr(model, "load_adapter"):
                        model.load_adapter(str(adapter_root), adapter_name="cara_hybrid")
                        if hasattr(model, "set_adapter"):
                            model.set_adapter("cara_hybrid")
                        adapter_report.update(
                            {
                                "selected": str(adapter_path),
                                "selected_root": str(adapter_root),
                                "selected_target": attr_name,
                                "loaded": True,
                                "load_method": "target_load_adapter",
                                "load_errors": load_errors[:20],
                            }
                        )
                        return adapter_report

                    from peft import PeftModel

                    wrapped = PeftModel.from_pretrained(model, str(adapter_root), adapter_name="cara_hybrid")
                    native_model, unwrap_method = _unwrap_peft_native_model(wrapped)
                    setattr(pipe, attr_name, native_model)
                    adapter_report.update(
                        {
                            "selected": str(adapter_path),
                            "selected_root": str(adapter_root),
                            "selected_target": attr_name,
                            "loaded": True,
                            "load_method": "peft_from_pretrained_native_forward",
                            "peft_unwrap_method": unwrap_method,
                            "forward_policy": "LoRA modules are injected, then the ACE transformer native forward signature is restored so PEFT does not pass text-model kwargs such as input_ids.",
                            "load_errors": load_errors[:20],
                        }
                    )
                    return adapter_report
            except Exception as exc:
                load_errors.append({"path": str(adapter_path), "root": str(adapter_root), "reason": repr(exc)})

    adapter_report["load_errors"] = load_errors[:30]
    raise RuntimeError(
        "ACE-Step Hybrid benchmark found the Side-Step trainable delta, but could not load any adapter artifact "
        "into the ACE pipeline; refusing to benchmark the base checkpoint as the Hybrid lane. "
        f"Adapter report: {adapter_report}"
    )


def _load_pipeline(
    *,
    checkpoint: str,
    checkpoint_dir: Path | None,
    trained_model_data: Path,
    work_dir: Path,
    report: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    _prepare_ace_hf_auth(report, checkpoint=checkpoint)
    delta_payload = _read_trainable_delta(trained_model_data)
    checkpoint_layout = _checkpoint_layout(checkpoint_dir)
    checkpoint_ref = checkpoint
    loader = "transformers_text_to_audio"
    if checkpoint_dir is not None and checkpoint_dir.exists():
        checkpoint_ref = str(checkpoint_dir)
        if (checkpoint_dir / "model_index.json").exists():
            loader = "diffusers_local_model_index"
        elif not (checkpoint_dir / "config.json").exists():
            raise RuntimeError(
                "Mounted ACE checkpoint_dir exists but is not a usable ACE checkpoint root. "
                f"Layout: {checkpoint_layout}"
            )
        else:
            dit_config = "acestep-v15-turbo" if checkpoint_layout.get("has_turbo_dit") else "acestep-v15-base"
            conversion = _convert_ace_bundle_to_diffusers(
                checkpoint_dir,
                work_dir / "ace_diffusers_checkpoint",
                dit_config=dit_config,
            )
            report["ace_diffusers_conversion"] = conversion
            checkpoint_ref = str(conversion["output_dir"])
            loader = "diffusers_converted_ace_bundle"

    report["checkpoint_resolution"] = {
        "requested_checkpoint": checkpoint,
        "selected_checkpoint_ref": checkpoint_ref,
        "loader": loader,
        "checkpoint_layout": checkpoint_layout,
        "selected_checkpoint_file_preview": _preview_files(Path(checkpoint_ref), limit=40) if Path(checkpoint_ref).exists() else [],
    }
    if loader in {"diffusers_local_model_index", "diffusers_converted_ace_bundle"}:
        pipe = _load_diffusers_pipeline(checkpoint_ref, report)
    else:
        pipe = _load_transformers_pipeline(checkpoint_ref)

    adapter_candidates = _adapter_candidates_from_delta(trained_model_data, delta_payload)
    adapter_report = _load_adapter_into_pipeline(pipe, adapter_candidates)
    adapter_report.update(
        {
            "trained_model_data": str(trained_model_data),
            "delta_type": delta_payload.get("delta_type"),
            "delta_format": delta_payload.get("format"),
            "loader": loader,
        }
    )
    return pipe, adapter_report


def _extract_audio_tensor(result: Any) -> torch.Tensor:
    candidates: list[Any] = []
    if isinstance(result, dict):
        for key in ("audio", "audios", "waveform", "waveforms"):
            if key in result:
                candidates.append(result[key])
    for attr in ("audios", "audio", "waveform", "waveforms"):
        if hasattr(result, attr):
            candidates.append(getattr(result, attr))
    if isinstance(result, (tuple, list)):
        candidates.extend(result)
    else:
        candidates.append(result)
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("audio", "audios", "waveform", "waveforms"):
                if key in candidate:
                    candidates.append(candidate[key])
            continue
        if isinstance(candidate, torch.Tensor):
            return _normalise_audio(candidate)
        try:
            import numpy as np

            if isinstance(candidate, np.ndarray):
                return _normalise_audio(torch.from_numpy(candidate))
            if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], np.ndarray):
                return _normalise_audio(torch.from_numpy(candidate[0]))
        except Exception:
            continue
        if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], torch.Tensor):
            return _normalise_audio(candidate[0])
    raise RuntimeError(f"ACE-Step pipeline returned no recognizable audio tensor: {type(result).__name__}")


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    finite = torch.isfinite(detached).all().item() if detached.numel() else True
    summary_tensor = detached.float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "finite": bool(finite),
        "mean": round(float(summary_tensor.mean().item()), 6) if summary_tensor.numel() else 0.0,
        "std": round(float(summary_tensor.std(unbiased=False).item()), 6) if summary_tensor.numel() else 0.0,
        "abs_max": round(float(summary_tensor.abs().max().item()), 6) if summary_tensor.numel() else 0.0,
    }


def _call_pipeline(
    pipe: Any,
    *,
    prompt: str,
    seed: int,
    duration_seconds: float,
    num_inference_steps: int,
    guidance_scale: float,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    latent_evidence: dict[str, Any] = {
        "status": "pending",
        "source": "ace_step_callback_on_step_end_latents",
        "note": "Continuous ACE-Step denoising latents are captured for audit; they are not a discrete CARA codeword by themselves.",
    }

    def _capture_latents(_pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        latents = callback_kwargs.get("latents")
        if isinstance(latents, torch.Tensor):
            latent_evidence.update(
                {
                    "status": "captured",
                    "step": int(step),
                    "timestep": float(timestep) if isinstance(timestep, (int, float)) else str(timestep),
                    "summary": _tensor_summary(latents),
                }
            )
        return {}

    attempts = [
        {
            "prompt": prompt,
            "audio_duration": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _capture_latents,
            "callback_on_step_end_tensor_inputs": ["latents"],
        },
        {
            "prompt": prompt,
            "duration": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _capture_latents,
            "callback_on_step_end_tensor_inputs": ["latents"],
        },
        {
            "prompt": prompt,
            "seconds": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": _capture_latents,
            "callback_on_step_end_tensor_inputs": ["latents"],
        },
        {
            "text": prompt,
            "audio_duration": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
        },
        {
            "prompt": prompt,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
        },
        {"prompt": prompt, "generator": generator},
        {"prompt": prompt},
        prompt,
    ]
    last_type_error: TypeError | None = None
    with torch.no_grad():
        for kwargs in attempts:
            try:
                if isinstance(kwargs, dict):
                    return {"audio": _extract_audio_tensor(pipe(**kwargs)), "latent_evidence": dict(latent_evidence)}
                return {"audio": _extract_audio_tensor(pipe(kwargs)), "latent_evidence": dict(latent_evidence)}
            except TypeError as exc:
                if "forward() got an unexpected keyword argument" in str(exc):
                    raise
                last_type_error = exc
                continue
    if last_type_error:
        raise last_type_error
    raise RuntimeError("ACE-Step pipeline call failed before returning audio.")


def _sample_rate(pipe: Any) -> int:
    for owner in (pipe, getattr(pipe, "vae", None), getattr(pipe, "audio_encoder", None)):
        if owner is None:
            continue
        for attr in ("sample_rate", "sampling_rate"):
            value = getattr(owner, attr, None)
            if value:
                return int(value)
    return 44100


def _metrics_from_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = Counter(str(row.get("model_id") or "unknown") for row in rows)
    by_suite = Counter(str(row.get("suite_id") or "unknown") for row in rows)
    labelled = [row for row in rows if row.get("expected", {}).get("cara_pool_id")]
    latent_captured = [
        row for row in rows if (row.get("latent_evidence") or {}).get("status") == "captured"
    ]
    return {
        "format": "cara_audio_benchmark_metrics_v1",
        "created_at": _utc_now(),
        "generated_audio_count": len(rows),
        "labelled_audio_count": len(labelled),
        "latent_evidence_count": len(latent_captured),
        "by_model": dict(sorted(by_model.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "native_cara_metrics": {
            CARA_MODEL_ID: {
                "status": "pending_ace_step_native_scorer",
                "reason": "Generated audio is saved; ACE-Step native DiT-head attribution scoring is a follow-on adapter.",
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
    checkpoint_dir = Path(args.checkpoint_dir) if str(args.checkpoint_dir or "").strip() else None
    if not prompt_manifest_file.exists():
        raise RuntimeError(f"Locked prompt manifest input is missing: {prompt_manifest_file}")
    if not trained_model_data.exists():
        raise RuntimeError(f"ACE-Step trained model input is missing: {trained_model_data}")
    if not torch.cuda.is_available():
        raise RuntimeError("ACE-Step generated-audio benchmark is GPU-only; CUDA is not available.")

    model_ids = _split_csv(args.model_ids)
    if any(model_id != CARA_MODEL_ID for model_id in model_ids):
        raise RuntimeError(f"Unsupported ACE-Step benchmark model ids: {model_ids}")
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

    pipe, adapter_report = _load_pipeline(
        checkpoint=str(args.checkpoint),
        checkpoint_dir=checkpoint_dir,
        trained_model_data=trained_model_data,
        work_dir=output_dir / "runtime_cache",
        report=report,
    )
    sample_rate = _sample_rate(pipe)
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
            "checkpoint": args.checkpoint,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "duration_seconds": float(args.duration_seconds),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "adapter": adapter_report,
            "sample_rate": sample_rate,
            "audio_output_policy": "WAV files are written by model_id/suite_id/prompt_id under audio/.",
        }
    )

    generation_rows: list[dict[str, Any]] = []
    for prompt_row in selected_rows:
        prompt_id = _safe_id(prompt_row.get("prompt_id"))
        suite_id = _safe_id(prompt_row.get("suite_id") or "suite")
        seed = int(prompt_row.get("seed") or 0)
        generation = _call_pipeline(
            pipe,
            prompt=str(prompt_row.get("prompt") or "CARA benchmark audio"),
            seed=seed,
            duration_seconds=float(args.duration_seconds),
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
        )
        audio = generation["audio"]
        latent_evidence = generation.get("latent_evidence") or {}
        relative_audio_path = Path("audio") / CARA_MODEL_ID / suite_id / f"{prompt_id}.wav"
        audio_path = output_dir / relative_audio_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(audio_path), audio, sample_rate)
        generation_rows.append(
            {
                "format": "cara_generated_audio_row_v1",
                "created_at": _utc_now(),
                "model_id": CARA_MODEL_ID,
                "suite_id": prompt_row.get("suite_id"),
                "prompt_id": prompt_row.get("prompt_id"),
                "seed": seed,
                "prompt": prompt_row.get("prompt"),
                "audio_path": str(relative_audio_path),
                "sample_rate": sample_rate,
                "duration_seconds": float(args.duration_seconds),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "expected": _expected_from_prompt_row(prompt_row),
                "latent_evidence": latent_evidence,
                "native_cara_prediction": {
                    "status": "pending_ace_step_native_scorer",
                    "reason": "ACE-Step generated-audio job captured continuous denoising latents, but a separate native ACE DiT-head scorer is required to decode CARA pool IDs.",
                },
                "external_probe_prediction": {"status": "pending_external_probe"},
            }
        )
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
    report["generated_audio_count"] = len(generation_rows)
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
    parser.add_argument("--checkpoint", default="ACE-Step/Ace-Step1.5")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--duration_seconds", type=float, default=12.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "24_benchmark_testing_ace_step_audio",
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
        environment="azureml:env-ace-step:5",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="ace_step",
        environment_name="env-ace-step",
        environment_version="5",
        import_status="ok" if not report["errors"] else "failed",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_ace_step_audio_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
