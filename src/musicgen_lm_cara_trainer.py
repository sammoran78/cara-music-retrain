from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from musicgen_cara_tokens import (
    build_cara_suffix_vocab,
    build_musicgen_registry_resolver,
    decode_cara_suffix,
    encode_cara_suffix,
    validate_musicgen_encodec_manifest,
)
from test_prep_common import base_metadata, parse_bool, write_report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_manifest_line"] = line_number
            rows.append(row)
    return rows


def _description_from_row(row: dict[str, Any], *, variant: str) -> str:
    description = (
        row.get("prompt")
        or row.get("description")
        or row.get("metadata_style_summary")
        or row.get("title")
        or row.get("chunk_id")
        or "music audio"
    )
    text = str(description).strip() or "music audio"
    if variant == "cara_lite":
        # Prompt-only control: visible CARA text, but no native suffix loss.
        text = f"{text} CARA pool {row.get('cara_pool_id')} family {row.get('cara_pool_family')}."
    return text


class MusicGenLMTokenDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        token_cache_dir: Path,
        suffix_vocab: dict[str, Any],
        *,
        max_token_frames: int,
        variant: str,
    ) -> None:
        self.rows = rows
        self.token_cache_dir = token_cache_dir
        self.suffix_vocab = suffix_vocab
        self.max_token_frames = max_token_frames
        self.variant = variant

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        token_path = self.token_cache_dir / str(row["encodec_token_path"])
        payload = torch.load(token_path, map_location="cpu", weights_only=False)
        codes = payload["codes"].long()
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2:
            raise RuntimeError(f"Expected MusicGen codes [K, T] or [1, K, T], got {tuple(codes.shape)} in {token_path}")
        codes = codes[:, : self.max_token_frames]
        if codes.shape[-1] < 2:
            raise RuntimeError(f"Too few EnCodec frames in {token_path}")
        return {
            "codes": codes,
            "suffix_tokens": torch.tensor(encode_cara_suffix(row, self.suffix_vocab), dtype=torch.long),
            "description": _description_from_row(row, variant=self.variant),
            "cara_pool_id": str(row["cara_pool_id"]),
            "cara_pool_index": int(row["cara_pool_index"]),
            "cara_pool_family_index": int(row["cara_pool_family_index"]),
            "chunk_id": str(row.get("chunk_id") or index),
            "split": str(row.get("split") or "unknown"),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    codebooks = int(batch[0]["codes"].shape[0])
    max_frames = max(int(item["codes"].shape[-1]) for item in batch)
    max_suffix_len = max(int(item["suffix_tokens"].numel()) for item in batch)
    codes = torch.zeros((len(batch), codebooks, max_frames), dtype=torch.long)
    audio_mask = torch.zeros((len(batch), codebooks, max_frames), dtype=torch.bool)
    suffix = torch.zeros((len(batch), max_suffix_len), dtype=torch.long)
    suffix_mask = torch.zeros((len(batch), max_suffix_len), dtype=torch.bool)
    descriptions: list[str] = []
    pool_ids: list[str] = []
    chunk_ids: list[str] = []
    for index, item in enumerate(batch):
        item_codes = item["codes"]
        item_suffix = item["suffix_tokens"]
        codes[index, :, : item_codes.shape[-1]] = item_codes
        audio_mask[index, :, : item_codes.shape[-1]] = True
        suffix[index, : item_suffix.numel()] = item_suffix
        suffix_mask[index, : item_suffix.numel()] = True
        descriptions.append(str(item["description"]))
        pool_ids.append(str(item["cara_pool_id"]))
        chunk_ids.append(str(item["chunk_id"]))
    return {
        "codes": codes,
        "audio_mask": audio_mask,
        "suffix_tokens": suffix,
        "suffix_mask": suffix_mask,
        "descriptions": descriptions,
        "cara_pool_id": pool_ids,
        "chunk_id": chunk_ids,
    }


def _first_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(output, dict):
        for item in output.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _pool_hidden(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.float()
    if tensor.ndim == 3:
        return tensor.float().mean(dim=1)
    dims = tuple(range(1, tensor.ndim - 1))
    return tensor.float().mean(dim=dims) if dims else tensor.float()


class CARASuffixHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, *, max_suffix_len: int = 96) -> None:
        super().__init__()
        self.position = nn.Embedding(max_suffix_len, hidden_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, pooled_hidden: torch.Tensor, suffix_len: int) -> torch.Tensor:
        positions = torch.arange(suffix_len, device=pooled_hidden.device).unsqueeze(0)
        hidden = pooled_hidden.unsqueeze(1) + self.position(positions)
        return self.proj(hidden)


class MusicGenLMWithCARA(nn.Module):
    def __init__(self, lm: nn.Module, *, suffix_vocab_size: int, detach_features: bool) -> None:
        super().__init__()
        self.lm = lm
        hidden_dim = int(getattr(lm, "dim", 0) or 0)
        if hidden_dim <= 0:
            raise RuntimeError("AudioCraft LM does not expose a usable hidden dimension at lm.dim.")
        self.cara_suffix_head = CARASuffixHead(hidden_dim, suffix_vocab_size)
        self.detach_features = detach_features
        self._features: list[torch.Tensor] = []
        self._hook_handle = self.lm.transformer.register_forward_hook(self._capture_transformer_output)

    def close(self) -> None:
        self._hook_handle.remove()

    def _capture_transformer_output(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is not None:
            self._features.append(tensor)

    def forward(self, codes: torch.Tensor, conditions: list[Any], suffix_len: int) -> dict[str, Any]:
        self._features.clear()
        lm_output = self.lm.compute_predictions(codes, conditions)
        if not self._features:
            raise RuntimeError("No MusicGen LM hidden state was captured from lm.transformer.")
        features = self._features[-1]
        if self.detach_features:
            features = features.detach()
        suffix_logits = self.cara_suffix_head(_pool_hidden(features), suffix_len)
        return {"lm_output": lm_output, "suffix_logits": suffix_logits}


def _conditions_from_descriptions(descriptions: list[str]) -> list[Any]:
    from audiocraft.modules.conditioners import ConditioningAttributes

    return [ConditioningAttributes(text={"description": description}) for description in descriptions]


def _set_trainable_mode(model: MusicGenLMWithCARA, variant: str, *, freeze_condition_provider: bool) -> str:
    for param in model.parameters():
        param.requires_grad_(True)
    if freeze_condition_provider and hasattr(model.lm, "condition_provider"):
        for param in model.lm.condition_provider.parameters():
            param.requires_grad_(False)
    if variant == "cara_probe":
        for param in model.lm.parameters():
            param.requires_grad_(False)
        for param in model.cara_suffix_head.parameters():
            param.requires_grad_(True)
        return "musicgen_lm_frozen_suffix_head_only"
    if variant in {"no_cara_baseline", "cara_lite"}:
        for param in model.cara_suffix_head.parameters():
            param.requires_grad_(False)
        return "musicgen_lm_audio_only"
    return "musicgen_lm_plus_cara_suffix_head"


def _configure_musicgen_transformer_runtime(lm: nn.Module, args: argparse.Namespace, report: dict[str, Any]) -> None:
    transformer = getattr(lm, "transformer", None)
    original_checkpointing = getattr(transformer, "checkpointing", None)
    requested_checkpointing = str(args.transformer_checkpointing or "torch")
    if transformer is not None and requested_checkpointing != "keep" and hasattr(transformer, "checkpointing"):
        transformer.checkpointing = requested_checkpointing

    attention_backend_status: dict[str, Any] = {"requested": args.efficient_attention_backend}
    try:
        from audiocraft.modules import transformer as transformer_module

        transformer_module.set_efficient_attention_backend(str(args.efficient_attention_backend))
        attention_backend_status["status"] = "set"
    except Exception as exc:
        attention_backend_status["status"] = "failed"
        attention_backend_status["error"] = str(exc)

    report["musicgen_transformer_runtime"] = {
        "original_checkpointing": original_checkpointing,
        "active_checkpointing": getattr(transformer, "checkpointing", None),
        "efficient_attention_backend": attention_backend_status,
        "note": "Real MusicGen LM smoke forces a public PyTorch checkpointing/attention path instead of AudioCraft fairinternal xformers checkpointing.",
    }


def _configure_model_dtype(model: nn.Module, args: argparse.Namespace, report: dict[str, Any]) -> torch.dtype:
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_by_name[str(args.model_dtype)]
    model.to(dtype=dtype)
    observed: dict[str, int] = {}
    for param in model.parameters():
        observed[str(param.dtype)] = observed.get(str(param.dtype), 0) + int(param.numel())
    report["model_dtype"] = {
        "requested": args.model_dtype,
        "active": str(dtype),
        "parameter_dtype_counts": observed,
        "note": "MusicGen is cast after checkpoint load so transformer LayerNorm/input dtypes match during training.",
    }
    return dtype


def _audio_token_loss(lm_output: Any, targets: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
    logits = lm_output.logits
    valid_mask = lm_output.mask.to(audio_mask.device) & audio_mask
    if valid_mask.sum().item() == 0:
        raise RuntimeError("MusicGen LM returned no valid token positions for loss computation.")
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    return F.cross_entropy(valid_logits, valid_targets)


def _suffix_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1).float()).sum() / mask.sum().clamp_min(1)


def _suffix_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, vocab: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    pred = logits.argmax(dim=-1)
    exact = ((pred == targets) | ~mask).all(dim=1).float().mean().item()
    decoded = [decode_cara_suffix(pred[index][mask[index]].tolist(), vocab, resolver) for index in range(pred.shape[0])]
    registry_valid = sum(1 for item in decoded if item["registry_valid"]) / max(1, len(decoded))
    hierarchical_valid = sum(1 for item in decoded if item["hierarchical_valid"]) / max(1, len(decoded))
    checksum_valid = sum(1 for item in decoded if item["checksum_valid"]) / max(1, len(decoded))
    return {
        "cara/suffix_exact": exact,
        "cara/registry_valid": registry_valid,
        "cara/hierarchical_valid": hierarchical_valid,
        "cara/checksum_valid": checksum_valid,
        "decoded_preview": decoded[:3],
    }


def _select_rows(rows: list[dict[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("split") or "").lower() == split]
    if not selected and split == "train":
        selected = rows
    if limit > 0:
        selected = selected[:limit]
    return selected


def _encodec_frame_count(row: dict[str, Any]) -> int:
    frame_count = row.get("encodec_frame_count")
    if frame_count not in (None, ""):
        return int(frame_count)
    shape = row.get("encodec_code_shape") or []
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return 0


def _filter_rows_with_enough_encodec_frames(
    rows: list[dict[str, Any]],
    *,
    min_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        frame_count = _encodec_frame_count(row)
        if frame_count >= min_frames:
            kept.append(row)
            continue
        rejected.append(
            {
                "chunk_id": row.get("chunk_id"),
                "split": row.get("split"),
                "prepared_audio_path": row.get("prepared_audio_path"),
                "encodec_token_path": row.get("encodec_token_path"),
                "encodec_frame_count": frame_count,
                "reject_reason": f"encodec_frame_count_lt_{min_frames}",
            }
        )
    return kept, rejected


def _save_delta(
    path: Path,
    model: MusicGenLMWithCARA,
    *,
    base_checkpoint: str,
    variant: str,
    global_step: int,
    resolver: dict[str, Any],
    suffix_vocab: dict[str, Any],
    trainable_only: bool = True,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    named_params = dict(model.named_parameters())
    state = {}
    for key, value in model.state_dict().items():
        param = named_params.get(key)
        if trainable_only and param is not None and not param.requires_grad:
            continue
        state[key] = value.detach().cpu()
    payload = {
        "format": "musicgen_lm_cara_delta_v1",
        "created_at": _utc_now(),
        "base_checkpoint": base_checkpoint,
        "variant": variant,
        "global_step": global_step,
        "registry_hash": resolver.get("registry_hash"),
        "suffix_vocab_hash": suffix_vocab.get("hash"),
        "state_dict": state,
        "state_dict_mode": "trainable_parameters_only" if trainable_only else "full_model",
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "format": payload["format"],
        "state_dict_tensors": len(state),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "global_step": global_step,
    }


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    token_cache_dir = Path(args.token_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = token_cache_dir / args.encodec_manifest_relative_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"MusicGen EnCodec manifest not found: {manifest_path}")

    raw_rows = _read_jsonl(manifest_path)
    validate_musicgen_encodec_manifest(raw_rows)
    all_rows, rejected_frame_rows = _filter_rows_with_enough_encodec_frames(
        raw_rows,
        min_frames=int(args.min_encodec_frames),
    )
    if rejected_frame_rows:
        (output_dir / "rejected_encodec_frame_rows.json").write_text(
            json.dumps(rejected_frame_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    train_rows = _select_rows(all_rows, split="train", limit=int(args.max_train_files))
    eval_rows = _select_rows(all_rows, split="validation", limit=int(args.max_eval_files))
    if not train_rows:
        raise RuntimeError("No MusicGen training rows are available after split/limit/frame-count selection.")
    resolver = build_musicgen_registry_resolver(all_rows)
    suffix_vocab = build_cara_suffix_vocab(all_rows)
    (output_dir / "cara_registry_resolver.json").write_text(json.dumps(resolver, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "cara_suffix_vocab.json").write_text(json.dumps(suffix_vocab, indent=2, sort_keys=True), encoding="utf-8")

    report.update(
        {
            "stage": "load_musicgen_lm",
            "variant": args.variant,
            "trainer_implementation": "real_audiocraft_musicgen_lm",
            "checkpoint": args.checkpoint,
            "encodec_manifest_rows": len(raw_rows),
            "encodec_rows_retained": len(all_rows),
            "rejected_encodec_frame_rows": {
                "count": len(rejected_frame_rows),
                "min_encodec_frames": int(args.min_encodec_frames),
                "preview": rejected_frame_rows[:5],
                "report_path": "rejected_encodec_frame_rows.json" if rejected_frame_rows else None,
            },
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "cara_registry": {
                "registry_hash": resolver["registry_hash"],
                "pool_count": resolver["pool_count"],
                "family_count": resolver["family_count"],
                "suffix_vocab_size": suffix_vocab["size"],
                "suffix_vocab_hash": suffix_vocab["hash"],
            },
            "prompt_policy": "ordinary_description_plus_visible_cara_text" if args.variant == "cara_lite" else "ordinary_description_unchanged",
        }
    )

    if not torch.cuda.is_available():
        raise RuntimeError("MusicGen LM trainer requires CUDA; CPU fallback is disabled for trainer jobs.")
    from audiocraft.models import MusicGen

    started = time.time()
    musicgen = MusicGen.get_pretrained(args.checkpoint, device="cuda")
    lm = musicgen.lm
    report["musicgen_checkpoint_load_seconds"] = round(time.time() - started, 3)
    _configure_model_dtype(lm, args, report)
    report["lm"] = {
        "class": type(lm).__name__,
        "dim": int(getattr(lm, "dim", 0) or 0),
        "cardinality": int(getattr(lm, "card", 0) or 0),
        "num_codebooks": int(getattr(lm, "num_codebooks", 0) or 0),
        "special_token_id": int(getattr(lm, "special_token_id", -1)),
    }
    _configure_musicgen_transformer_runtime(lm, args, report)
    compression_model = getattr(musicgen, "compression_model", None)
    report["compression_model"] = {
        "sample_rate": getattr(compression_model, "sample_rate", None),
        "channels": getattr(compression_model, "channels", None),
        "frame_rate": getattr(compression_model, "frame_rate", None),
        "num_codebooks": getattr(compression_model, "num_codebooks", None),
        "cardinality": getattr(compression_model, "cardinality", None),
    }

    wrapped = MusicGenLMWithCARA(lm, suffix_vocab_size=int(suffix_vocab["size"]), detach_features=args.variant == "cara_probe").to("cuda")
    optimizer_mode = _set_trainable_mode(wrapped, args.variant, freeze_condition_provider=parse_bool(args.freeze_condition_provider))
    report["optimizer_mode"] = optimizer_mode
    trainable_params = [param for param in wrapped.parameters() if param.requires_grad]
    report["trainable_parameter_count"] = int(sum(param.numel() for param in trainable_params))
    if not trainable_params and not (parse_bool(args.preflight_only) or parse_bool(args.dry_run)):
        raise RuntimeError("No trainable MusicGen LM/CARA parameters are enabled.")

    train_loader = DataLoader(
        MusicGenLMTokenDataset(
            train_rows,
            token_cache_dir,
            suffix_vocab,
            max_token_frames=int(args.max_token_frames),
            variant=args.variant,
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    first_batch = next(iter(train_loader))
    report["first_batch"] = {
        "codes_shape": list(first_batch["codes"].shape),
        "suffix_tokens_shape": list(first_batch["suffix_tokens"].shape),
        "chunk_id_preview": first_batch["chunk_id"][:3],
        "cara_pool_id_preview": first_batch["cara_pool_id"][:3],
        "description_preview": first_batch["descriptions"][:2],
    }
    if parse_bool(args.preflight_only) and parse_bool(args.preflight_forward_check):
        wrapped.eval()
        with torch.no_grad():
            codes = first_batch["codes"].to("cuda")
            audio_mask = first_batch["audio_mask"].to("cuda")
            conditions = _conditions_from_descriptions(first_batch["descriptions"])
            with torch.autocast(device_type="cuda", enabled=False):
                outputs = wrapped(codes, conditions, first_batch["suffix_tokens"].shape[1])
            audio_loss = _audio_token_loss(outputs["lm_output"], codes, audio_mask)
        report["preflight_forward_check"] = {
            "status": "passed",
            "audio_token_loss": float(audio_loss.detach().cpu()),
            "lm_logits_shape": list(outputs["lm_output"].logits.shape),
            "lm_mask_shape": list(outputs["lm_output"].mask.shape),
        }
    if parse_bool(args.preflight_only) or parse_bool(args.dry_run):
        report["status"] = "passed"
        report["stage"] = "preflight_complete"
        wrapped.close()
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate))
    report["stage"] = "trainer_fit"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "musicgen_lm_cara_delta.pt"
    max_steps = int(args.max_steps)
    checkpoint_every = max(1, int(args.checkpoint_every))
    global_step = 0
    losses: list[float] = []
    audio_losses: list[float] = []
    suffix_losses: list[float] = []
    latest_cara_metrics: dict[str, Any] = {}
    train_started = time.time()

    wrapped.train()
    while global_step < max_steps:
        for batch in train_loader:
            codes = batch["codes"].to("cuda")
            audio_mask = batch["audio_mask"].to("cuda")
            suffix_tokens = batch["suffix_tokens"].to("cuda")
            suffix_mask = batch["suffix_mask"].to("cuda")
            conditions = _conditions_from_descriptions(batch["descriptions"])
            with torch.autocast(device_type="cuda", enabled=False):
                outputs = wrapped(codes, conditions, suffix_tokens.shape[1])
            audio_loss = _audio_token_loss(outputs["lm_output"], codes, audio_mask)
            suffix_loss = torch.tensor(0.0, device="cuda")
            if args.variant in {"cara_probe", "cara_strong"}:
                suffix_loss = _suffix_loss(outputs["suffix_logits"], suffix_tokens, suffix_mask)
            loss = audio_loss if args.variant in {"no_cara_baseline", "cara_lite"} else audio_loss + float(args.attribution_loss_weight) * suffix_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            audio_losses.append(float(audio_loss.detach().cpu()))
            suffix_losses.append(float(suffix_loss.detach().cpu()))
            if args.variant in {"cara_probe", "cara_strong"}:
                latest_cara_metrics = _suffix_metrics(outputs["suffix_logits"].detach().cpu(), suffix_tokens.cpu(), suffix_mask.cpu(), suffix_vocab, resolver)
            if global_step % checkpoint_every == 0:
                report["trainable_delta_checkpoint"] = _save_delta(
                    checkpoint_path,
                    wrapped,
                    base_checkpoint=str(args.checkpoint),
                    variant=str(args.variant),
                    global_step=global_step,
                    resolver=resolver,
                    suffix_vocab=suffix_vocab,
                )
            if global_step >= max_steps:
                break

    report["training_seconds"] = round(time.time() - train_started, 3)
    report["global_step"] = global_step
    report["train_loss"] = sum(losses) / len(losses) if losses else None
    report["audio_token_loss"] = sum(audio_losses) / len(audio_losses) if audio_losses else None
    report["cara_suffix_loss"] = sum(suffix_losses) / len(suffix_losses) if suffix_losses else None
    report["latest_cara_metrics"] = latest_cara_metrics
    report["trainable_delta_checkpoint"] = _save_delta(
        checkpoint_path,
        wrapped,
        base_checkpoint=str(args.checkpoint),
        variant=str(args.variant),
        global_step=global_step,
        resolver=resolver,
        suffix_vocab=suffix_vocab,
    )
    report["status"] = "passed" if global_step >= max_steps else "failed"
    if args.variant in {"cara_probe", "cara_strong"} and not latest_cara_metrics.get("cara/registry_valid"):
        report.setdefault("warnings", []).append("MusicGen LM CARA suffix path produced no registry-valid decoded suffixes in the latest training batch.")
    wrapped.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encodec_manifest_relative_path", default="manifest.encodec.jsonl")
    parser.add_argument("--checkpoint", default="facebook/musicgen-small")
    parser.add_argument("--variant", choices=["no_cara_baseline", "cara_lite", "cara_probe", "cara_strong"], default="no_cara_baseline")
    parser.add_argument("--run_name", default="cara-musicgen-lm")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--attribution_loss_weight", type=float, default=0.05)
    parser.add_argument("--max_train_files", type=int, default=2048)
    parser.add_argument("--max_eval_files", type=int, default=512)
    parser.add_argument("--max_token_frames", type=int, default=512)
    parser.add_argument("--min_encodec_frames", type=int, default=2)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--model_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--transformer_checkpointing", choices=["keep", "none", "torch"], default="torch")
    parser.add_argument("--efficient_attention_backend", choices=["torch", "xformers"], default="torch")
    parser.add_argument("--preflight_forward_check", default="true")
    parser.add_argument("--freeze_condition_provider", default="true")
    parser.add_argument("--preflight_only", default="false")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "08_musicgen_lm_cara_trainer",
        "status": "failed",
        "run_name": args.run_name,
        "variant": args.variant,
        "errors": [],
        "warnings": [],
        "torch_version": torch.__version__,
    }
    try:
        _run(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
    metadata = base_metadata(
        test_name="08_musicgen_lm_cara_trainer",
        compute="gpu-smoke-h100",
        environment="azureml:env-musicgen-audiocraft:3",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="musicgen",
        environment_name="env-musicgen-audiocraft",
        environment_version="3",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="musicgen_lm_cara_trainer_report.json")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
