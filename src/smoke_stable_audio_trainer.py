from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from cara_attribution_head import (
    attach_cara_attribution_to_training_wrapper,
    build_cara_registry_resolver,
    validate_cara_manifest_labels,
    write_cara_registry_resolver,
)
from test_prep_common import base_metadata, parse_bool, write_report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _split_audio_relative_path(row: dict[str, Any], split: str) -> Path | None:
    prepared = Path(str(row.get("prepared_audio_path") or ""))
    try:
        return prepared.relative_to(f"stable_audio_open_small/audio/{split}")
    except ValueError:
        return None


def _codeword_from_pool_id(pool_id: Any) -> str | None:
    text = str(pool_id or "").strip()
    parts = text.split(":")
    if len(parts) >= 5 and parts[0] == "CARA":
        return parts[3]
    return text or None


_CONTEXT_POLICY_INDEX = {
    "same_pool_source_disjoint": 0,
    "same_pool_plus_family_fallback": 1,
    "same_family_source_disjoint": 2,
    "no_source_disjoint_context": 3,
    "unknown": 4,
}


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        if value in (None, ""):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _mode_int(values: list[Any], fallback: int) -> int:
    normalized = [_safe_int(value, fallback=fallback) for value in values if value not in (None, "")]
    if not normalized:
        return fallback
    return int(Counter(normalized).most_common(1)[0][0])


def _load_context_pack_rows(context_pack_dir: str | None, relative_path: str) -> list[dict[str, Any]]:
    if not context_pack_dir:
        return []
    path = Path(context_pack_dir) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Context pack manifest not found: {path}")
    return _read_jsonl(path)


def _enrich_manifest_rows_with_context(
    manifest_rows: list[dict[str, Any]],
    *,
    context_pack_dir: str | None,
    context_pack_relative_path: str,
    report: dict[str, Any],
) -> None:
    pack_rows = _load_context_pack_rows(context_pack_dir, context_pack_relative_path)
    if not pack_rows:
        raise RuntimeError("Context mode is enabled but no context pack rows were loaded.")

    pack_by_chunk_id: dict[str, dict[str, Any]] = {}
    policy_counts: Counter[str] = Counter()
    context_counts: Counter[int] = Counter()
    for pack in pack_rows:
        target = pack.get("target") if isinstance(pack.get("target"), dict) else {}
        chunk_id = str(target.get("chunk_id") or "").strip()
        if chunk_id:
            pack_by_chunk_id[chunk_id] = pack

    missing_context = 0
    source_overlap = 0
    for row in manifest_rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        target_pool = _safe_int(row.get("cara_pool_index"), fallback=0)
        target_family = _safe_int(row.get("cara_pool_family_index"), fallback=0)
        pack = pack_by_chunk_id.get(chunk_id)
        contexts = pack.get("context_examples") if isinstance(pack, dict) and isinstance(pack.get("context_examples"), list) else []
        policy = str(pack.get("context_policy") or "unknown") if isinstance(pack, dict) else "unknown"
        if pack is None:
            missing_context += 1
        if any(str(ctx.get("source_id") or "") == str(row.get("source_id") or "") for ctx in contexts if isinstance(ctx, dict)):
            source_overlap += 1

        context_pool = _mode_int([ctx.get("cara_pool_index") for ctx in contexts if isinstance(ctx, dict)], fallback=target_pool)
        context_family = _mode_int(
            [ctx.get("cara_pool_family_index") for ctx in contexts if isinstance(ctx, dict)],
            fallback=target_family,
        )
        context_count = len([ctx for ctx in contexts if isinstance(ctx, dict)])
        policy_index = _CONTEXT_POLICY_INDEX.get(policy, _CONTEXT_POLICY_INDEX["unknown"])

        row["cara_context_pool_index"] = context_pool
        row["cara_context_pool_family_index"] = context_family
        row["cara_context_policy_index"] = policy_index
        row["cara_context_count"] = context_count
        row["cara_context_policy"] = policy
        row["cara_context_pack_id"] = pack.get("context_pack_id") if isinstance(pack, dict) else None
        policy_counts[policy] += 1
        context_counts[context_count] += 1

    if source_overlap:
        raise RuntimeError(f"Context pack source-disjoint violation detected in {source_overlap} prepared rows.")

    report["context_diffusion_conditioning"] = {
        "enabled": True,
        "mode": "metadata_context_conditioning",
        "context_pack_dir": str(context_pack_dir),
        "context_pack_relative_path": context_pack_relative_path,
        "context_pack_rows": len(pack_rows),
        "manifest_rows_enriched": len(manifest_rows),
        "manifest_rows_missing_context_pack": missing_context,
        "context_policy_counts": dict(policy_counts),
        "context_count_distribution": {str(key): value for key, value in sorted(context_counts.items())},
        "conditioner_ids": [
            "cara_context_pool_index",
            "cara_context_pool_family_index",
            "cara_context_policy_index",
            "cara_context_count",
        ],
        "claim_scope": (
            "Context branch full run adds source-disjoint context-pack metadata as Stable Audio cross-attention "
            "int conditioners beside CARA-Strong pool/family conditioning. It is more context-aware than the "
            "original CARA-Strong run, but it is not yet raw audio-context latent cross-attention."
        ),
    }


def _prune_checkpoint_dir(checkpoint_dir: Path, keep_last_n: int) -> dict[str, Any]:
    keep_last_n = max(0, int(keep_last_n))
    if not checkpoint_dir.exists():
        return {"kept_periodic": [], "deleted": [], "last_checkpoint_present": False}
    periodic = sorted(
        [path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt"],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    keep = periodic[-keep_last_n:] if keep_last_n else []
    keep_names = {path.name for path in keep}
    deleted: list[str] = []
    for path in periodic:
        if path.name in keep_names:
            continue
        try:
            path.unlink()
            deleted.append(path.name)
        except OSError:
            pass
    return {
        "kept_periodic": [path.name for path in keep],
        "deleted": deleted,
        "last_checkpoint_present": (checkpoint_dir / "last.ckpt").exists(),
    }


def _configure_disk_safe_runtime_dirs(output_dir: Path) -> dict[str, str]:
    cache_root = output_dir / "runtime_cache"
    tmp_root = output_dir / "tmp"
    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TMPDIR": tmp_root,
        "TEMP": tmp_root,
        "TMP": tmp_root,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    configured: dict[str, str] = {}
    for key, path in paths.items():
        os.environ.setdefault(key, str(path))
        configured[key] = os.environ[key]
    return configured


def _save_trainable_delta_checkpoint(
    training_wrapper: torch.nn.Module,
    checkpoint_path: Path,
    *,
    global_step: int,
    base_checkpoint: str,
    variant: str,
    training_scope: str,
) -> dict[str, Any]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state: dict[str, torch.Tensor] = {}
    trainable_parameter_count = 0
    for name, parameter in training_wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
        trainable_state[name] = tensor
        trainable_parameter_count += int(parameter.numel())
    payload = {
        "format": "cara_trainable_delta_v1",
        "base_checkpoint": base_checkpoint,
        "variant": variant,
        "training_scope": training_scope,
        "global_step": int(global_step),
        "trainable_parameter_count": trainable_parameter_count,
        "state_dict": trainable_state,
    }
    torch.save(payload, checkpoint_path)
    return {
        "path": str(checkpoint_path),
        "format": "cara_trainable_delta_v1",
        "global_step": int(global_step),
        "trainable_tensor_count": len(trainable_state),
        "trainable_parameter_count": trainable_parameter_count,
        "size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 3),
    }


def _limit_audio_dir(
    source_audio_dir: Path,
    work_dir: Path,
    manifest_rows: list[dict[str, Any]],
    limit: int,
    *,
    split: str,
) -> Path:
    if limit <= 0:
        return source_audio_dir
    limited_root = work_dir / f"limited_audio_{split}"
    limited_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in manifest_rows:
        if copied >= limit:
            break
        rel = _split_audio_relative_path(row, split)
        if rel is None:
            continue
        source = source_audio_dir / rel
        if not source.exists():
            continue
        target = limited_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"No {split} audio files could be linked from {source_audio_dir}")
    return limited_root


def _metadata_prompt(row: dict[str, Any], variant: str) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        prompt = str(row.get("title") or row.get("chunk_id") or "CARA training audio").strip()
    if variant == "cara_lite":
        pool = str(row.get("cara_pool_id") or "").strip()
        if pool:
            return f"{prompt}. CARA_POOL: {pool}"
    return prompt


def _write_metadata_module(work_dir: Path, manifest_rows: list[dict[str, Any]], variant: str) -> Path:
    lookup: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        split = str(row.get("split") or "").strip()
        if split not in {"train", "validation", "test"}:
            continue
        rel_path = _split_audio_relative_path(row, split)
        prepared = Path(str(row.get("prepared_audio_path") or ""))
        rel = rel_path.as_posix() if rel_path is not None else prepared.name
        pool_id = row.get("cara_pool_id")
        codeword = row.get("cara_pool_codeword") or _codeword_from_pool_id(pool_id)
        registered_codeword = row.get("cara_registered_codeword") or pool_id
        payload = {
            "prompt": _metadata_prompt(row, variant),
            "seconds_total": float(row.get("duration_sec") or 0.0),
            "cara_pool_id": pool_id,
            "cara_pool_codeword": codeword,
            "cara_registered_codeword": registered_codeword,
            "cara_pool_index": row.get("cara_pool_index"),
            "cara_pool_family": row.get("cara_pool_family"),
            "cara_pool_family_index": row.get("cara_pool_family_index"),
            "cara_tag_text": registered_codeword,
            "source_id": row.get("source_id"),
            "chunk_id": row.get("chunk_id"),
            "split": split,
            "cara_context_pool_index": row.get("cara_context_pool_index"),
            "cara_context_pool_family_index": row.get("cara_context_pool_family_index"),
            "cara_context_policy_index": row.get("cara_context_policy_index"),
            "cara_context_count": row.get("cara_context_count"),
            "cara_context_policy": row.get("cara_context_policy"),
            "cara_context_pack_id": row.get("cara_context_pack_id"),
        }
        lookup[rel] = payload
        lookup[f"{split}/{rel}"] = payload
        lookup[prepared.name] = payload

    lookup_path = work_dir / "cara_metadata_lookup.json"
    module_path = work_dir / "cara_metadata.py"
    lookup_path.write_text(json.dumps(lookup, indent=2, sort_keys=True), encoding="utf-8")
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import json",
                "from pathlib import Path",
                "",
                f"_LOOKUP_PATH = Path({str(lookup_path)!r})",
                "_LOOKUP = None",
                "",
                "def _load_lookup():",
                "    global _LOOKUP",
                "    if _LOOKUP is None:",
                "        _LOOKUP = json.loads(_LOOKUP_PATH.read_text(encoding='utf-8'))",
                "    return _LOOKUP",
                "",
                "def get_custom_metadata(info, audio):",
                "    lookup = _load_lookup()",
                "    relpath = str(info.get('relpath') or '')",
                "    name = Path(relpath).name",
                "    row = lookup.get(relpath) or lookup.get(name) or {}",
                "    prompt = row.get('prompt') or relpath or name or 'CARA training audio'",
                "    metadata = {",
                "        'prompt': prompt,",
                "        'text': prompt,",
                "        'cara_pool_id': row.get('cara_pool_id'),",
                "        'cara_pool_codeword': row.get('cara_pool_codeword'),",
                "        'cara_registered_codeword': row.get('cara_registered_codeword'),",
                "        'cara_pool_index': row.get('cara_pool_index'),",
                "        'cara_pool_family': row.get('cara_pool_family'),",
                "        'cara_pool_family_index': row.get('cara_pool_family_index'),",
                "        'cara_context_pool_index': row.get('cara_context_pool_index'),",
                "        'cara_context_pool_family_index': row.get('cara_context_pool_family_index'),",
                "        'cara_context_policy_index': row.get('cara_context_policy_index'),",
                "        'cara_context_count': row.get('cara_context_count'),",
                "        'cara_context_policy': row.get('cara_context_policy'),",
                "        'cara_context_pack_id': row.get('cara_context_pack_id'),",
                "        'cara_structured_conditioning': {",
                "            'pool_id': row.get('cara_pool_id'),",
                "            'pool_codeword': row.get('cara_pool_codeword'),",
                "            'registered_codeword': row.get('cara_registered_codeword'),",
                "            'pool_index': row.get('cara_pool_index'),",
                "            'family': row.get('cara_pool_family'),",
                "            'family_index': row.get('cara_pool_family_index'),",
                "        },",
                "        'cara_context_conditioning': {",
                "            'pool_index': row.get('cara_context_pool_index'),",
                "            'family_index': row.get('cara_context_pool_family_index'),",
                "            'policy_index': row.get('cara_context_policy_index'),",
                "            'context_count': row.get('cara_context_count'),",
                "            'policy': row.get('cara_context_policy'),",
                "            'pack_id': row.get('cara_context_pack_id'),",
                "        },",
                "        'cara_tag_text': row.get('cara_tag_text'),",
                "        'source_id': row.get('source_id'),",
                "        'chunk_id': row.get('chunk_id'),",
                "    }",
                "    if row.get('seconds_total'):",
                "        metadata['seconds_total'] = row['seconds_total']",
                "    return metadata",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return module_path


def _prepare_hf_auth(report: dict[str, Any]) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        report["hf_auth_source"] = "environment"
        return

    vault_url = os.environ.get("KEY_VAULT_URL")
    secret_name = os.environ.get("HF_TOKEN_SECRET_NAME", "hf-token")
    if not vault_url:
        report["hf_auth_source"] = "missing"
        report["warnings"].append("HF_TOKEN is not set and KEY_VAULT_URL was not provided.")
        return

    try:
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
        from azure.keyvault.secrets import SecretClient

        credential = AzureMLOnBehalfOfCredential()
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        token = secret_client.get_secret(secret_name).value
    except Exception as exc:
        report["hf_auth_source"] = "key_vault_failed"
        raise RuntimeError(f"Unable to retrieve Hugging Face token secret '{secret_name}' from Azure Key Vault.") from exc

    if not token:
        report["hf_auth_source"] = "key_vault_empty"
        raise RuntimeError(f"Azure Key Vault secret '{secret_name}' is empty.")

    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    report["hf_auth_source"] = "azure_key_vault"


def _set_optimizer_lr(config: Any, learning_rate: float) -> None:
    if isinstance(config, dict):
        if "lr" in config and isinstance(config["lr"], (int, float)):
            config["lr"] = learning_rate
        for value in config.values():
            _set_optimizer_lr(value, learning_rate)
    elif isinstance(config, list):
        for value in config:
            _set_optimizer_lr(value, learning_rate)


def _prepare_baseline_training_config(model_config: dict[str, Any], report: dict[str, Any]) -> None:
    training_config = model_config.setdefault("training", {})
    removed_sections: dict[str, Any] = {}

    arc_config = training_config.get("arc")
    if isinstance(arc_config, dict):
        discriminator_ckpt = str(arc_config.get("discriminator_base_ckpt") or "")
        teacher_ckpt = str(arc_config.get("teacher_model_ckpt") or "")
        if discriminator_ckpt.startswith("/path/to/") or teacher_ckpt.startswith("/path/to/"):
            removed_sections["arc"] = {
                "reason": "Stable Audio Open Small config ships an ARC discriminator checkpoint placeholder.",
                "discriminator_base_ckpt": discriminator_ckpt or None,
                "teacher_model_ckpt": teacher_ckpt or None,
            }
            training_config.pop("arc", None)

    if "arc" in training_config:
        removed_sections["arc"] = {
            "reason": "ARC wrapper disabled for baseline/CARA-lite smoke; this validates ordinary diffusion training plumbing.",
            "discriminator_base_ckpt": str(training_config["arc"].get("discriminator_base_ckpt") or None)
            if isinstance(training_config["arc"], dict)
            else None,
        }
        training_config.pop("arc", None)

    training_config["use_ema"] = False
    report["disabled_training_config_sections"] = removed_sections
    report["training_wrapper_mode"] = "diffusion_cond_plain"


def _ensure_cara_conditioner(
    model_config: dict[str, Any],
    resolver: dict[str, Any],
    report: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        report["cara_native_conditioning"] = {"enabled": False}
        return

    model_section = model_config.setdefault("model", {})
    conditioning = model_section.setdefault("conditioning", {})
    configs = conditioning.setdefault("configs", [])
    cond_dim = conditioning.get("cond_dim")
    if not isinstance(cond_dim, int) or cond_dim <= 0:
        raise RuntimeError("Stable Audio model config does not expose a positive model.conditioning.cond_dim.")

    diffusion = model_section.setdefault("diffusion", {})
    diffusion_config = diffusion.setdefault("config", {})
    if int(diffusion_config.get("cond_token_dim") or 0) <= 0:
        raise RuntimeError("Stable Audio DiT config does not expose cond_token_dim for cross-attention CARA conditioning.")

    cross_attention_ids = diffusion.setdefault("cross_attention_cond_ids", [])
    if not isinstance(cross_attention_ids, list):
        raise RuntimeError("Stable Audio diffusion cross_attention_cond_ids must be a list for CARA conditioning.")

    desired = _cara_conditioner_specs(resolver)
    existing_ids = {str(item.get("id")) for item in configs if isinstance(item, dict)}
    added_ids: list[str] = []
    for item in desired:
        if item["id"] not in existing_ids:
            configs.append(item)
            added_ids.append(item["id"])
        if item["id"] not in cross_attention_ids:
            cross_attention_ids.append(item["id"])

    report["cara_native_conditioning"] = {
        "enabled": True,
        "method": "Stable Audio Tools int conditioners joined to DiT cross-attention conditioning",
        "conditioner_ids": [item["id"] for item in desired],
        "added_conditioner_ids": added_ids,
        "cross_attention_cond_ids": list(cross_attention_ids),
        "pool_count": int(resolver["pool_count"]),
        "family_count": int(resolver["family_count"]),
    }


def _cara_conditioner_specs(resolver: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "cara_pool_index",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": max(0, int(resolver["pool_count"]) - 1),
            },
        },
        {
            "id": "cara_pool_family_index",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": max(0, int(resolver["family_count"]) - 1),
            },
        },
    ]


def _context_conditioner_specs(resolver: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "cara_context_pool_index",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": max(0, int(resolver["pool_count"]) - 1),
            },
        },
        {
            "id": "cara_context_pool_family_index",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": max(0, int(resolver["family_count"]) - 1),
            },
        },
        {
            "id": "cara_context_policy_index",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": max(_CONTEXT_POLICY_INDEX.values()),
            },
        },
        {
            "id": "cara_context_count",
            "type": "int",
            "config": {
                "min_val": 0,
                "max_val": 8,
            },
        },
    ]


def _ensure_context_conditioner(
    model_config: dict[str, Any],
    resolver: dict[str, Any],
    report: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        report["context_diffusion_native_conditioning"] = {"enabled": False}
        return

    model_section = model_config.setdefault("model", {})
    conditioning = model_section.setdefault("conditioning", {})
    configs = conditioning.setdefault("configs", [])
    cond_dim = conditioning.get("cond_dim")
    if not isinstance(cond_dim, int) or cond_dim <= 0:
        raise RuntimeError("Stable Audio model config does not expose a positive model.conditioning.cond_dim.")

    diffusion = model_section.setdefault("diffusion", {})
    diffusion_config = diffusion.setdefault("config", {})
    if int(diffusion_config.get("cond_token_dim") or 0) <= 0:
        raise RuntimeError("Stable Audio DiT config does not expose cond_token_dim for context conditioning.")

    cross_attention_ids = diffusion.setdefault("cross_attention_cond_ids", [])
    if not isinstance(cross_attention_ids, list):
        raise RuntimeError("Stable Audio diffusion cross_attention_cond_ids must be a list for context conditioning.")

    desired = _context_conditioner_specs(resolver)
    existing_ids = {str(item.get("id")) for item in configs if isinstance(item, dict)}
    added_ids: list[str] = []
    for item in desired:
        if item["id"] not in existing_ids:
            configs.append(item)
            added_ids.append(item["id"])
        if item["id"] not in cross_attention_ids:
            cross_attention_ids.append(item["id"])

    report["context_diffusion_native_conditioning"] = {
        "enabled": True,
        "method": "Stable Audio Tools int conditioners joined to DiT cross-attention conditioning",
        "conditioner_ids": [item["id"] for item in desired],
        "added_conditioner_ids": added_ids,
        "cross_attention_cond_ids": list(cross_attention_ids),
        "note": (
            "These context conditioners are derived from locked source-disjoint context packs. "
            "They are a stronger context-aware branch than CARA-Strong alone, but still metadata context rather than raw audio-context embeddings."
        ),
    }


def _patch_loaded_model_cara_conditioners(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    resolver: dict[str, Any],
    report: dict[str, Any],
) -> None:
    from stable_audio_tools.models.conditioners import IntConditioner

    conditioning_config = model_config.get("model", {}).get("conditioning", {})
    cond_dim = conditioning_config.get("cond_dim")
    if not isinstance(cond_dim, int) or cond_dim <= 0:
        raise RuntimeError("Stable Audio model config does not expose a positive model.conditioning.cond_dim.")

    conditioner = getattr(model, "conditioner", None)
    if conditioner is None or not hasattr(conditioner, "conditioners"):
        raise RuntimeError("Loaded Stable Audio model does not expose a patchable conditioner ModuleDict.")

    conditioner_modules = conditioner.conditioners
    existing_ids = set(conditioner_modules.keys())
    added_ids: list[str] = []
    device = next(model.parameters()).device
    for spec in _cara_conditioner_specs(resolver):
        conditioner_id = str(spec["id"])
        config = dict(spec["config"])
        if conditioner_id not in conditioner_modules:
            conditioner_modules[conditioner_id] = IntConditioner(
                output_dim=cond_dim,
                min_val=int(config["min_val"]),
                max_val=int(config["max_val"]),
            ).to(device)
            added_ids.append(conditioner_id)

    patched_ids = sorted(conditioner_modules.keys())
    missing = [spec["id"] for spec in _cara_conditioner_specs(resolver) if spec["id"] not in conditioner_modules]
    if missing:
        raise RuntimeError(f"Loaded Stable Audio conditioner is missing CARA conditioner ids after patch: {missing}")

    report.setdefault("cara_native_conditioning", {})["loaded_model_conditioner_patched"] = True
    report["cara_native_conditioning"]["loaded_model_conditioner_added_ids"] = added_ids
    report["cara_native_conditioning"]["loaded_model_conditioner_preexisting_ids"] = sorted(existing_ids)
    report["cara_native_conditioning"]["loaded_model_conditioner_ids"] = patched_ids


def _patch_loaded_model_context_conditioners(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    resolver: dict[str, Any],
    report: dict[str, Any],
) -> None:
    from stable_audio_tools.models.conditioners import IntConditioner

    conditioning_config = model_config.get("model", {}).get("conditioning", {})
    cond_dim = conditioning_config.get("cond_dim")
    if not isinstance(cond_dim, int) or cond_dim <= 0:
        raise RuntimeError("Stable Audio model config does not expose a positive model.conditioning.cond_dim.")

    conditioner = getattr(model, "conditioner", None)
    if conditioner is None or not hasattr(conditioner, "conditioners"):
        raise RuntimeError("Loaded Stable Audio model does not expose a patchable conditioner ModuleDict.")

    conditioner_modules = conditioner.conditioners
    existing_ids = set(conditioner_modules.keys())
    added_ids: list[str] = []
    device = next(model.parameters()).device
    for spec in _context_conditioner_specs(resolver):
        conditioner_id = str(spec["id"])
        config = dict(spec["config"])
        if conditioner_id not in conditioner_modules:
            conditioner_modules[conditioner_id] = IntConditioner(
                output_dim=cond_dim,
                min_val=int(config["min_val"]),
                max_val=int(config["max_val"]),
            ).to(device)
            added_ids.append(conditioner_id)

    missing = [spec["id"] for spec in _context_conditioner_specs(resolver) if spec["id"] not in conditioner_modules]
    if missing:
        raise RuntimeError(f"Loaded Stable Audio conditioner is missing context conditioner ids after patch: {missing}")

    report.setdefault("context_diffusion_native_conditioning", {})["loaded_model_conditioner_patched"] = True
    report["context_diffusion_native_conditioning"]["loaded_model_conditioner_added_ids"] = added_ids
    report["context_diffusion_native_conditioning"]["loaded_model_conditioner_preexisting_ids"] = sorted(existing_ids)
    report["context_diffusion_native_conditioning"]["loaded_model_conditioner_ids"] = sorted(conditioner_modules.keys())


def _preview_cara_conditioner_outputs(
    model: torch.nn.Module,
    metadata: list[dict[str, Any]],
    resolver: dict[str, Any],
    report: dict[str, Any],
) -> None:
    conditioner = getattr(model, "conditioner", None)
    if conditioner is None or not hasattr(conditioner, "conditioners"):
        raise RuntimeError("Loaded Stable Audio model does not expose conditioner outputs for CARA preview.")

    device = next(model.parameters()).device
    preview: dict[str, dict[str, Any]] = {}
    missing_modules: list[str] = []
    for spec in _cara_conditioner_specs(resolver):
        conditioner_id = str(spec["id"])
        module = conditioner.conditioners[conditioner_id] if conditioner_id in conditioner.conditioners else None
        if module is None:
            missing_modules.append(conditioner_id)
            continue
        values = [int(row[conditioner_id]) for row in metadata]
        with torch.no_grad():
            tensor, mask = module(values, device)
        preview[conditioner_id] = {
            "values": values[:8],
            "tensor_shape": list(tensor.shape),
            "mask_shape": list(mask.shape) if mask is not None else None,
            "device": str(tensor.device),
        }

    if missing_modules:
        raise RuntimeError(f"Loaded Stable Audio conditioner is missing CARA modules: {missing_modules}")
    report["first_batch_cara_conditioner_preview"] = preview


def _preview_context_conditioner_outputs(
    model: torch.nn.Module,
    metadata: list[dict[str, Any]],
    resolver: dict[str, Any],
    report: dict[str, Any],
) -> None:
    conditioner = getattr(model, "conditioner", None)
    if conditioner is None or not hasattr(conditioner, "conditioners"):
        raise RuntimeError("Loaded Stable Audio model does not expose conditioner outputs for context preview.")

    device = next(model.parameters()).device
    preview: dict[str, dict[str, Any]] = {}
    missing_modules: list[str] = []
    for spec in _context_conditioner_specs(resolver):
        conditioner_id = str(spec["id"])
        module = conditioner.conditioners[conditioner_id] if conditioner_id in conditioner.conditioners else None
        if module is None:
            missing_modules.append(conditioner_id)
            continue
        values = [_safe_int(row.get(conditioner_id), fallback=0) for row in metadata]
        with torch.no_grad():
            tensor, mask = module(values, device)
        preview[conditioner_id] = {
            "values": values[:8],
            "tensor_shape": list(tensor.shape),
            "mask_shape": list(mask.shape) if mask is not None else None,
            "device": str(tensor.device),
        }

    if missing_modules:
        raise RuntimeError(f"Loaded Stable Audio conditioner is missing context modules: {missing_modules}")
    report["first_batch_context_conditioner_preview"] = preview


def _create_stable_audio_dataloader(create_dataloader_from_config: Any, *args: Any, num_workers: int, **kwargs: Any) -> Any:
    if num_workers != 0:
        return create_dataloader_from_config(*args, num_workers=num_workers, **kwargs)

    original_dataloader = torch.utils.data.DataLoader

    def single_process_dataloader(*loader_args: Any, **loader_kwargs: Any) -> Any:
        if int(loader_kwargs.get("num_workers") or 0) == 0:
            loader_kwargs["persistent_workers"] = False
        return original_dataloader(*loader_args, **loader_kwargs)

    torch.utils.data.DataLoader = single_process_dataloader
    try:
        return create_dataloader_from_config(*args, num_workers=0, **kwargs)
    finally:
        torch.utils.data.DataLoader = original_dataloader


def _research_alignment_for_variant(variant: str) -> dict[str, Any]:
    alignments = {
        "no_cara_baseline": {
            "stage": "same-data no-CARA baseline",
            "ordinary_prompt_unchanged": True,
            "cara_text_conditioning": False,
            "structured_pool_conditioner": False,
            "attribution_head": False,
            "auxiliary_pool_family_loss": False,
            "backbone_grads_from_attribution_loss": False,
            "satisfies_cara_strong_claim": False,
            "note": "Baseline trainer plumbing only; not CARA-Strong evidence.",
        },
        "cara_lite": {
            "stage": "CARA-lite prompt-control",
            "ordinary_prompt_unchanged": False,
            "cara_text_conditioning": True,
            "structured_pool_conditioner": False,
            "attribution_head": False,
            "auxiliary_pool_family_loss": False,
            "backbone_grads_from_attribution_loss": False,
            "satisfies_cara_strong_claim": False,
            "note": "Prompt-control smoke only; useful comparison, not CARA-Strong evidence.",
        },
        "cara_head": {
            "stage": "detached CARA attribution-head smoke",
            "ordinary_prompt_unchanged": True,
            "cara_text_conditioning": False,
            "structured_pool_conditioner": False,
            "attribution_head": True,
            "auxiliary_pool_family_loss": True,
            "backbone_grads_from_attribution_loss": False,
            "satisfies_cara_strong_claim": False,
            "note": "Detached audio-hidden-state head validates labels, taps, metrics, registry decoding, and leakage controls.",
        },
        "cara_strong": {
            "stage": "CARA-Strong non-detached attribution smoke",
            "ordinary_prompt_unchanged": True,
            "cara_text_conditioning": False,
            "structured_pool_conditioner": True,
            "attribution_head": True,
            "auxiliary_pool_family_loss": True,
            "backbone_grads_from_attribution_loss": True,
            "satisfies_cara_strong_claim": True,
            "note": (
                "Native Stable Audio CARA int conditioners plus non-detached hidden-state attribution loss are active. "
                "Report this as smoke evidence only; full research evidence still needs longer training and benchmark comparisons."
            ),
        },
    }
    return alignments[variant]


def _make_dataset_config(audio_dir: Path, metadata_module: Path, *, dataset_id: str) -> dict[str, Any]:
    return {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": dataset_id,
                "path": str(audio_dir),
                "custom_metadata_module": str(metadata_module),
            }
        ],
        "random_crop": False,
        "drop_last": True,
    }


def _mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in metric_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals) if counts.get(key)}


def _move_batch_audio_to_cuda(batch: Any) -> Any:
    if isinstance(batch, (list, tuple)) and batch and hasattr(batch[0], "to"):
        converted = list(batch)
        converted[0] = converted[0].to("cuda", non_blocking=True)
        return tuple(converted)
    return batch


def _evaluate_cara_split(
    *,
    training_wrapper: torch.nn.Module,
    dataloader: Any,
    split: str,
    max_batches: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    was_training = training_wrapper.training
    training_wrapper.eval()
    metric_rows: list[dict[str, float]] = []
    losses: list[float] = []
    samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = _move_batch_audio_to_cuda(batch)
            result = training_wrapper.training_step(batch, batch_idx)
            loss = result.get("loss") if isinstance(result, dict) else result
            if hasattr(loss, "detach"):
                losses.append(float(loss.detach().cpu()))
            latest_metrics = report.get("latest_cara_metrics")
            if isinstance(latest_metrics, dict):
                metric_rows.append({key: float(value) for key, value in latest_metrics.items() if isinstance(value, (int, float))})
            if isinstance(batch, (list, tuple)) and batch and hasattr(batch[0], "shape"):
                samples += int(batch[0].shape[0])
    if was_training:
        training_wrapper.train()
    return {
        "split": split,
        "batches": len(metric_rows),
        "samples": samples,
        "loss": sum(losses) / len(losses) if losses else None,
        "metrics": _mean_metrics(metric_rows),
    }


def _run_smoke_training(args: argparse.Namespace, report: dict[str, Any]) -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from stable_audio_tools.data.dataset import create_dataloader_from_config
    from stable_audio_tools.models.pretrained import get_pretrained_model
    from stable_audio_tools.training import create_training_wrapper_from_config

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for Stable Audio smoke training; CPU fallback is intentionally disabled.")
    input_root = Path(args.input_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["runtime_dirs"] = _configure_disk_safe_runtime_dirs(output_dir)
    manifest_path = input_root / args.manifest_relative_path
    audio_dir = input_root / args.audio_relative_path
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    report["stage"] = "validate_prepared_inputs"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared Stable Audio manifest not found: {manifest_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Prepared Stable Audio train audio folder not found: {audio_dir}")

    manifest_rows = _read_jsonl(manifest_path)
    context_mode = str(getattr(args, "context_mode", "none") or "none").strip()
    context_enabled = context_mode != "none"
    if context_enabled:
        report["stage"] = "enrich_manifest_rows_with_context"
        context_cache_dir = getattr(args, "context_cache_dir", None)
        context_smoke_dir = getattr(args, "context_smoke_dir", None)
        context_cache_path = Path(context_cache_dir) / str(getattr(args, "context_cache_relative_path", "context_cache_manifest.jsonl")) if context_cache_dir else None
        context_smoke_metrics_path = Path(context_smoke_dir) / "context_smoke_metrics.json" if context_smoke_dir else None
        if context_cache_path is None or not context_cache_path.exists():
            raise FileNotFoundError(f"Context cache manifest is required before full context training: {context_cache_path}")
        if context_smoke_metrics_path is None or not context_smoke_metrics_path.exists():
            raise FileNotFoundError(f"Context smoke metrics are required before full context training: {context_smoke_metrics_path}")
        report["context_diffusion_required_artifacts"] = {
            "context_cache_manifest": str(context_cache_path),
            "context_smoke_metrics": str(context_smoke_metrics_path),
        }
        _enrich_manifest_rows_with_context(
            manifest_rows,
            context_pack_dir=getattr(args, "context_pack_dir", None),
            context_pack_relative_path=str(getattr(args, "context_pack_relative_path", "context_pack_manifest.jsonl")),
            report=report,
        )
    train_rows = [row for row in manifest_rows if row.get("split") == "train"]
    validation_rows = [row for row in manifest_rows if row.get("split") == "validation"]
    test_rows = [row for row in manifest_rows if row.get("split") == "test"]
    label_summary = validate_cara_manifest_labels(manifest_rows)
    resolver = build_cara_registry_resolver(manifest_rows, split_manifest_path=input_root / "split_manifest.json")
    resolver_path = work_dir / "cara_registry_resolver.json"
    write_cara_registry_resolver(resolver_path, resolver)
    report["manifest_rows"] = len(manifest_rows)
    report["train_manifest_rows"] = len(train_rows)
    report["cara_label_summary"] = label_summary
    missing_codeword_rows = [
        row.get("chunk_id") or row.get("prepared_audio_path")
        for row in manifest_rows
        if row.get("cara_pool_codeword") in (None, "")
    ]
    report["cara_codeword_manifest_summary"] = {
        "source_rows_missing_cara_pool_codeword": len(missing_codeword_rows),
        "derived_from_cara_pool_id": len(missing_codeword_rows),
        "preview": missing_codeword_rows[:5],
    }
    report["cara_registry_resolver_path"] = str(resolver_path)
    report["cara_registry_hash"] = resolver["registry_hash"]
    report["cara_registry"] = {
        "pool_count": resolver["pool_count"],
        "family_count": resolver["family_count"],
        "decoded_cara_id_format": resolver["decoded_cara_id_format"],
        "manifest_lock_id": resolver.get("manifest_lock_id"),
    }
    if len(train_rows) < max(1, int(args.batch_size)):
        raise RuntimeError(f"Not enough train rows for batch_size={args.batch_size}: {len(train_rows)}")

    report["stage"] = "limit_smoke_audio_dir"
    smoke_audio_dir = _limit_audio_dir(audio_dir, work_dir, train_rows, int(args.max_train_files), split="train")
    report["limited_audio_file_count"] = sum(1 for path in smoke_audio_dir.rglob("*.wav") if path.is_file())

    report["stage"] = "write_metadata_module"
    metadata_module = _write_metadata_module(work_dir, manifest_rows, args.variant)
    report["smoke_audio_dir"] = str(smoke_audio_dir)
    report["metadata_module"] = str(metadata_module)
    report["variant"] = args.variant
    report["training_scope"] = args.training_scope
    report["context_mode"] = context_mode
    report["research_alignment"] = _research_alignment_for_variant(args.variant)
    if context_enabled:
        report["research_alignment"]["context_diffusion_branch"] = {
            "context_examples_as_conditioning": True,
            "source_disjoint_context_pack_required": True,
            "prompt_text_unchanged": True,
            "raw_audio_context_latents": False,
            "note": (
                "This branch applies the Context Diffusion lesson as an added source-disjoint context metadata lane "
                "inside Stable Audio cross-attention. It is expected to test persistence beyond prompt-only tags, "
                "but it is not yet the full image-paper equivalent of raw context tokens."
            ),
        }
    report["leakage_controls"] = {
        "prompt_only_probe": args.variant == "cara_lite",
        "audio_hidden_state_head_without_cara_text": args.variant == "cara_head",
        "shuffled_label_sanity_metric": args.variant in {"cara_head", "cara_strong"},
        "source_disjoint_splits": sorted(label_summary.get("split_counts", {}).keys()),
    }

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

    _prepare_baseline_training_config(model_config, report)
    _ensure_cara_conditioner(
        model_config,
        resolver,
        report,
        enabled=args.variant == "cara_strong",
    )
    _ensure_context_conditioner(
        model_config,
        resolver,
        report,
        enabled=context_enabled,
    )
    training_config = model_config.setdefault("training", {})
    training_config["learning_rate"] = float(args.learning_rate)
    _set_optimizer_lr(training_config.get("optimizer_configs"), float(args.learning_rate))
    report["learning_rate"] = float(args.learning_rate)
    report["ema_disabled_for_smoke"] = True
    num_workers = max(0, int(args.num_workers))
    report["num_workers"] = num_workers
    if num_workers == 0:
        report["dataloader_worker_mode"] = "single_process"
        report.setdefault("warnings", []).append(
            "Stable Audio DataLoader is running with num_workers=0 to avoid Azure worker-process RAM blow-up."
        )
    else:
        report["dataloader_worker_mode"] = "multiprocessing"

    dataset_config = _make_dataset_config(
        smoke_audio_dir,
        metadata_module,
        dataset_id=f"cara_stable_audio_{args.training_scope}_train",
    )
    dataset_config_path = work_dir / "dataset_config.json"
    model_config_path = work_dir / "model_config.smoke.json"
    dataset_config_path.write_text(json.dumps(dataset_config, indent=2, sort_keys=True), encoding="utf-8")
    model_config_path.write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    report["dataset_config_path"] = str(dataset_config_path)
    report["model_config_path"] = str(model_config_path)

    report["stage"] = "create_dataloader"
    train_dl = _create_stable_audio_dataloader(
        create_dataloader_from_config,
        dataset_config,
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        sample_rate=int(model_config["sample_rate"]),
        sample_size=int(model_config["sample_size"]),
        audio_channels=int(model_config.get("audio_channels", 2)),
        shuffle=True,
    )
    report["dataloader_batches_estimate"] = len(train_dl)
    requested_max_steps = int(args.max_steps)
    if requested_max_steps <= 0:
        if args.training_scope != "full":
            raise RuntimeError("max_steps=0 is only allowed for full training dataset-pass mode.")
        effective_max_steps = max(1, len(train_dl))
    else:
        effective_max_steps = requested_max_steps
    report["requested_max_steps"] = requested_max_steps
    report["effective_max_steps"] = effective_max_steps

    report["stage"] = "read_first_dataloader_batch"
    first_batch = next(iter(train_dl))
    audio, metadata = first_batch
    report["first_batch_audio_shape"] = list(audio.shape)
    report["first_batch_metadata_keys"] = sorted(metadata[0].keys()) if metadata else []

    if parse_bool(args.dry_run):
        report["status"] = "passed"
        report["smoke_mode"] = "dry_run_loader_only"
        return

    report["stage"] = "move_model_to_cuda"
    model = model.to("cuda")
    if args.variant == "cara_strong":
        report["stage"] = "patch_loaded_model_cara_conditioners"
        _patch_loaded_model_cara_conditioners(model, model_config, resolver, report)
        _preview_cara_conditioner_outputs(model, metadata, resolver, report)
    if context_enabled:
        report["stage"] = "patch_loaded_model_context_conditioners"
        _patch_loaded_model_context_conditioners(model, model_config, resolver, report)
        _preview_context_conditioner_outputs(model, metadata, resolver, report)
    report["stage"] = "create_training_wrapper"
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    if args.variant in {"cara_head", "cara_strong"}:
        report["stage"] = "attach_cara_attribution_head"
        training_wrapper = attach_cara_attribution_to_training_wrapper(
            training_wrapper,
            model,
            resolver=resolver,
            variant=args.variant,
            loss_weight=float(args.attribution_loss_weight),
            optimizer_lr=float(args.learning_rate),
            detach_features=args.variant == "cara_head",
            freeze_backbone=args.variant == "cara_head",
            report=report,
        )
    checkpoint_dir = output_dir / "checkpoints"
    report["stage"] = "create_trainer"
    logger = CSVLogger(save_dir=str(output_dir / "logs"), name="stable_audio_smoke")
    checkpoint_keep_last_n = max(0, int(args.checkpoint_keep_last_n))
    checkpoint_every = max(1, int(args.checkpoint_every))
    use_trainable_delta_checkpoint = args.training_scope == "full"
    callbacks: list[Callback] = []
    report["checkpoint_dir"] = str(checkpoint_dir)
    report["checkpoint_strategy"] = (
        "mounted_output_trainable_delta" if use_trainable_delta_checkpoint else "lightning_full_checkpoint"
    )
    if not use_trainable_delta_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="periodic",
            every_n_train_steps=checkpoint_every,
            save_top_k=-1,
            save_last=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        callbacks.append(checkpoint_callback)

    class CheckpointRetentionCallback(Callback):
        def _prune(self, trainer: pl.Trainer) -> None:
            summary = _prune_checkpoint_dir(checkpoint_dir, checkpoint_keep_last_n)
            report["checkpoint_retention"] = {
                "keep_last_n_periodic": checkpoint_keep_last_n,
                "last_prune_global_step": int(getattr(trainer, "global_step", 0)),
                **summary,
            }

        def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
            if int(getattr(trainer, "global_step", 0)) % checkpoint_every == 0:
                self._prune(trainer)

        def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            self._prune(trainer)

    class TrainableDeltaCheckpointCallback(Callback):
        def _save(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            summary = _save_trainable_delta_checkpoint(
                pl_module,
                checkpoint_dir / "trainable_delta.pt",
                global_step=int(getattr(trainer, "global_step", 0)),
                base_checkpoint=str(args.checkpoint),
                variant=str(args.variant),
                training_scope=str(args.training_scope),
            )
            report["trainable_delta_checkpoint"] = summary

        def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
            global_step = int(getattr(trainer, "global_step", 0))
            if global_step > 0 and global_step % checkpoint_every == 0:
                self._save(trainer, pl_module)

        def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            self._save(trainer, pl_module)

    if use_trainable_delta_checkpoint:
        callbacks.append(TrainableDeltaCheckpointCallback())
        report["checkpoint_retention"] = {
            "keep_last_n_periodic": 0,
            "last_checkpoint_present": False,
            "note": "Full runs write trainable_delta.pt to the mounted AzureML output path instead of Lightning .ckpt files.",
        }
    else:
        callbacks.append(CheckpointRetentionCallback())

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        max_steps=effective_max_steps,
        log_every_n_steps=10 if args.training_scope == "full" else 1,
        num_sanity_val_steps=0,
        enable_checkpointing=not use_trainable_delta_checkpoint,
        enable_progress_bar=args.training_scope != "full",
        default_root_dir=str(output_dir),
        limit_val_batches=0,
    )
    report["stage"] = "trainer_fit"
    train_started = time.time()
    trainer.fit(training_wrapper, train_dl)
    report["training_seconds"] = round(time.time() - train_started, 3)
    report["global_step"] = int(trainer.global_step)
    if use_trainable_delta_checkpoint:
        if "trainable_delta_checkpoint" not in report:
            report["trainable_delta_checkpoint"] = _save_trainable_delta_checkpoint(
                training_wrapper,
                checkpoint_dir / "trainable_delta.pt",
                global_step=int(trainer.global_step),
                base_checkpoint=str(args.checkpoint),
                variant=str(args.variant),
                training_scope=str(args.training_scope),
            )
    else:
        report["checkpoint_retention"] = {
            "keep_last_n_periodic": checkpoint_keep_last_n,
            **_prune_checkpoint_dir(checkpoint_dir, checkpoint_keep_last_n),
        }
    report["status"] = "passed" if trainer.global_step >= effective_max_steps else "failed"
    if trainer.logged_metrics:
        report["logged_metrics"] = {
            key: float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            for key, value in trainer.logged_metrics.items()
            if isinstance(value, (int, float)) or hasattr(value, "detach")
        }
    if args.variant in {"cara_head", "cara_strong"} and parse_bool(args.run_eval):
        report["stage"] = "evaluate_cara_splits"
        eval_results: dict[str, Any] = {}
        for split, rows in (("validation", validation_rows), ("test", test_rows)):
            if not rows:
                report["warnings"].append(f"No {split} rows found for CARA held-out evaluation.")
                continue
            split_audio_dir = input_root / f"stable_audio_open_small/audio/{split}"
            if not split_audio_dir.exists():
                report["warnings"].append(f"Prepared Stable Audio {split} audio folder not found: {split_audio_dir}")
                continue
            limited_split_audio_dir = _limit_audio_dir(
                split_audio_dir,
                work_dir,
                rows,
                int(args.max_eval_files),
                split=split,
            )
            split_dataset_config = _make_dataset_config(
                limited_split_audio_dir,
                metadata_module,
                dataset_id=f"cara_stable_audio_{args.training_scope}_{split}",
            )
            split_dl = _create_stable_audio_dataloader(
                create_dataloader_from_config,
                split_dataset_config,
                batch_size=int(args.batch_size),
                num_workers=num_workers,
                sample_rate=int(model_config["sample_rate"]),
                sample_size=int(model_config["sample_size"]),
                audio_channels=int(model_config.get("audio_channels", 2)),
                shuffle=False,
            )
            eval_results[split] = _evaluate_cara_split(
                training_wrapper=training_wrapper,
                dataloader=split_dl,
                split=split,
                max_batches=int(args.max_eval_batches),
                report=report,
            )
        report["heldout_evaluation"] = eval_results
    if args.variant in {"cara_head", "cara_strong"} and "latest_cara_decoded_preview" not in report:
        report["status"] = "failed"
        report["errors"].append("CARA attribution head did not produce registry-decoded predictions.")
    if args.training_scope == "full" and args.variant == "cara_strong":
        heldout = report.get("heldout_evaluation") if isinstance(report.get("heldout_evaluation"), dict) else {}
        validation_metrics = (heldout.get("validation") or {}).get("metrics") if isinstance(heldout, dict) else None
        test_metrics = (heldout.get("test") or {}).get("metrics") if isinstance(heldout, dict) else None
        if not validation_metrics or not test_metrics:
            report["status"] = "failed"
            report["errors"].append("Full CARA-Strong training requires validation and test held-out CARA metrics.")
    if trainer.global_step < effective_max_steps:
        report["errors"].append(f"Trainer stopped at global_step={trainer.global_step}, expected {effective_max_steps}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_relative_path", default="stable_audio_open_small/manifest.jsonl")
    parser.add_argument("--audio_relative_path", default="stable_audio_open_small/audio/train")
    parser.add_argument("--checkpoint", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--variant", choices=["no_cara_baseline", "cara_lite", "cara_head", "cara_strong"], default="no_cara_baseline")
    parser.add_argument("--run_name", default="cara-stable-audio-smoke")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--checkpoint_keep_last_n", type=int, default=1)
    parser.add_argument("--max_train_files", type=int, default=2048)
    parser.add_argument("--max_eval_files", type=int, default=512)
    parser.add_argument("--max_eval_batches", type=int, default=16)
    parser.add_argument("--attribution_loss_weight", type=float, default=0.05)
    parser.add_argument("--training_scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run_eval", default="true")
    parser.add_argument("--context_mode", choices=["none", "metadata_context_conditioning"], default="none")
    parser.add_argument("--context_pack_dir", default=None)
    parser.add_argument("--context_pack_relative_path", default="context_pack_manifest.jsonl")
    parser.add_argument("--context_cache_dir", default=None)
    parser.add_argument("--context_cache_relative_path", default="context_cache_manifest.jsonl")
    parser.add_argument("--context_smoke_dir", default=None)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "07_smoke_stable_audio_trainer",
        "status": "failed",
        "run_name": args.run_name,
        "variant": args.variant,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "attribution_loss_weight": float(args.attribution_loss_weight),
        "training_scope": args.training_scope,
        "max_train_files": args.max_train_files,
        "max_eval_files": args.max_eval_files,
        "max_eval_batches": args.max_eval_batches,
        "context_mode": args.context_mode,
        "context_pack_dir": args.context_pack_dir,
        "context_pack_relative_path": args.context_pack_relative_path,
        "context_cache_dir": args.context_cache_dir,
        "context_cache_relative_path": args.context_cache_relative_path,
        "context_smoke_dir": args.context_smoke_dir,
        "errors": [],
        "warnings": [],
    }

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _run_smoke_training(args, report)
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
    write_report(Path(args.output_dir), report, metadata, report_alias="stable_audio_smoke_trainer_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
