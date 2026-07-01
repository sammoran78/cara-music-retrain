from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_CARA_LABEL_FIELDS = (
    "cara_pool_id",
    "cara_pool_index",
    "cara_pool_family",
    "cara_pool_family_index",
)


def _stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_cara_manifest_labels(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        split = str(row.get("split") or "unknown")
        split_counts[split] += 1
        absent = [field for field in REQUIRED_CARA_LABEL_FIELDS if row.get(field) in (None, "")]
        if absent:
            missing.append(
                {
                    "row_index": row_index,
                    "chunk_id": row.get("chunk_id"),
                    "prepared_audio_path": row.get("prepared_audio_path"),
                    "missing_fields": absent,
                }
            )
            continue
        pool_counts[str(row["cara_pool_id"])] += 1
        family_counts[str(row["cara_pool_family"])] += 1
    if missing:
        preview = missing[:5]
        raise ValueError(f"Prepared manifest has {len(missing)} rows missing CARA label fields: {preview}")
    return {
        "row_count": sum(split_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "pool_count": len(pool_counts),
        "family_count": len(family_counts),
        "pool_support_min": min(pool_counts.values()) if pool_counts else 0,
        "pool_support_max": max(pool_counts.values()) if pool_counts else 0,
        "family_support_min": min(family_counts.values()) if family_counts else 0,
        "family_support_max": max(family_counts.values()) if family_counts else 0,
    }


def build_cara_registry_resolver(rows: Iterable[dict[str, Any]], *, split_manifest_path: Path | None = None) -> dict[str, Any]:
    pool_by_index: dict[int, str] = {}
    family_by_index: dict[int, str] = {}
    pool_to_family_index: dict[int, int] = {}
    pool_support: Counter[int] = Counter()
    family_support: Counter[int] = Counter()
    for row in rows:
        pool_index = int(row["cara_pool_index"])
        family_index = int(row["cara_pool_family_index"])
        pool_id = str(row["cara_pool_id"])
        family = str(row["cara_pool_family"])
        pool_by_index[pool_index] = pool_id
        family_by_index[family_index] = family
        pool_to_family_index[pool_index] = family_index
        pool_support[pool_index] += 1
        family_support[family_index] += 1

    pool_indices_by_family: dict[int, list[int]] = {}
    for pool_index, family_index in sorted(pool_to_family_index.items()):
        pool_indices_by_family.setdefault(family_index, []).append(pool_index)

    split_manifest: dict[str, Any] = {}
    if split_manifest_path and split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    resolver = {
        "format": "cara_registry_resolver_v1",
        "decoded_cara_id_format": "cara_pool_id from locked prepared manifest",
        "manifest_lock_id": split_manifest.get("created_at") or split_manifest.get("tir_id"),
        "pool_count": len(pool_by_index),
        "family_count": len(family_by_index),
        "pool_by_index": {str(key): pool_by_index[key] for key in sorted(pool_by_index)},
        "family_by_index": {str(key): family_by_index[key] for key in sorted(family_by_index)},
        "pool_to_family_index": {str(key): pool_to_family_index[key] for key in sorted(pool_to_family_index)},
        "pool_indices_by_family": {
            str(key): value for key, value in sorted(pool_indices_by_family.items())
        },
        "pool_support": {str(key): pool_support[key] for key in sorted(pool_support)},
        "family_support": {str(key): family_support[key] for key in sorted(family_support)},
        "source_split_manifest_hash": _stable_json_hash(split_manifest) if split_manifest else None,
    }
    resolver["registry_hash"] = _stable_json_hash(
        {
            "pool_by_index": resolver["pool_by_index"],
            "family_by_index": resolver["family_by_index"],
            "pool_to_family_index": resolver["pool_to_family_index"],
        }
    )
    return resolver


def write_cara_registry_resolver(path: Path, resolver: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolver, indent=2, sort_keys=True), encoding="utf-8")


def family_pool_mask(
    family_targets: torch.Tensor,
    pool_to_family_index: dict[int, int],
    num_pools: int,
) -> torch.Tensor:
    mask = torch.zeros((family_targets.numel(), num_pools), dtype=torch.bool, device=family_targets.device)
    for pool_index in range(num_pools):
        family_index = int(pool_to_family_index.get(pool_index, -1))
        if family_index >= 0:
            mask[:, pool_index] = family_targets == family_index
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, :] = True
    return mask


def masked_pool_logits(
    pool_logits: torch.Tensor,
    family_targets: torch.Tensor,
    pool_to_family_index: dict[int, int],
) -> torch.Tensor:
    mask = family_pool_mask(family_targets, pool_to_family_index, pool_logits.shape[-1])
    return pool_logits.masked_fill(~mask, torch.finfo(pool_logits.dtype).min / 4)


def _metadata_value(metadata: Any, index: int, key: str) -> Any:
    if isinstance(metadata, list):
        return metadata[index].get(key) if index < len(metadata) and isinstance(metadata[index], dict) else None
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, torch.Tensor):
            return value[index].item() if value.ndim else value.item()
        if isinstance(value, (list, tuple)):
            return value[index] if index < len(value) else None
        return value
    return None


def labels_from_metadata(metadata: Any, *, device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    if isinstance(metadata, list):
        batch_size = len(metadata)
    elif isinstance(metadata, dict):
        first = next(iter(metadata.values()), [])
        batch_size = int(first.shape[0]) if isinstance(first, torch.Tensor) and first.ndim else len(first)
    else:
        raise ValueError(f"Unsupported metadata type for CARA labels: {type(metadata).__name__}")

    pool_indices: list[int] = []
    family_indices: list[int] = []
    pool_ids: list[str] = []
    for index in range(batch_size):
        pool_index = _metadata_value(metadata, index, "cara_pool_index")
        family_index = _metadata_value(metadata, index, "cara_pool_family_index")
        pool_id = _metadata_value(metadata, index, "cara_pool_id")
        if pool_index in (None, "") or family_index in (None, "") or pool_id in (None, ""):
            raise ValueError(
                "Batch metadata is missing CARA attribution labels. "
                "Expected cara_pool_id, cara_pool_index, cara_pool_family_index for every sample."
            )
        pool_indices.append(int(pool_index))
        family_indices.append(int(family_index))
        pool_ids.append(str(pool_id))
    return {
        "pool_index": torch.tensor(pool_indices, dtype=torch.long, device=device),
        "family_index": torch.tensor(family_indices, dtype=torch.long, device=device),
        "pool_id": pool_ids,
    }


def _first_hidden_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor) and output.ndim >= 2 and output.shape[0] > 0:
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            found = _first_hidden_tensor(item)
            if found is not None:
                return found
    if isinstance(output, dict):
        for item in output.values():
            found = _first_hidden_tensor(item)
            if found is not None:
                return found
    return None


def pool_hidden_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.float()
    if tensor.ndim == 3:
        return tensor.float().mean(dim=1)
    dims = tuple(range(1, tensor.ndim - 1))
    return tensor.float().mean(dim=dims) if dims else tensor.float()


class CaraHiddenStateTapper:
    def __init__(self, model: nn.Module, *, max_taps: int = 4) -> None:
        self.model = model
        self.max_taps = max_taps
        self.handles: list[Any] = []
        self.features: list[torch.Tensor] = []
        self.tap_names: list[str] = []

    def _candidate_modules(self) -> list[tuple[str, nn.Module]]:
        candidates: list[tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            children = list(module.children())
            if children:
                continue
            text = f"{name} {module.__class__.__name__}".lower()
            if any(marker in text for marker in ("transformer", "dit", "block", "layer", "residual")):
                candidates.append((name, module))
        if not candidates:
            for name, module in self.model.named_modules():
                if name and not list(module.children()):
                    candidates.append((name, module))
        if not candidates:
            candidates.append(("model", self.model))
        if len(candidates) <= self.max_taps:
            return candidates
        step = max(1, len(candidates) // self.max_taps)
        selected = candidates[-self.max_taps * step :: step]
        return selected[-self.max_taps :]

    def register(self) -> dict[str, Any]:
        self.close()
        for name, module in self._candidate_modules():
            self.tap_names.append(name)

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, tap_name: str = name) -> None:
                tensor = _first_hidden_tensor(output)
                if tensor is not None:
                    self.features.append(pool_hidden_tensor(tensor))

            self.handles.append(module.register_forward_hook(hook))
        return {"tap_count": len(self.tap_names), "tap_names": list(self.tap_names)}

    def clear(self) -> None:
        self.features.clear()

    def pooled_features(self, batch_size: int) -> torch.Tensor:
        usable = [feature for feature in self.features if feature.ndim == 2 and feature.shape[0] == batch_size]
        if not usable:
            raise RuntimeError("No CARA hidden-state tap produced a batch-aligned feature tensor.")
        return torch.cat(usable, dim=-1)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.tap_names = []
        self.clear()


class CaraAttributionHead(nn.Module):
    def __init__(self, *, num_pools: int, num_families: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pool_classifier = nn.Linear(hidden_dim, num_pools)
        self.family_classifier = nn.Linear(hidden_dim, num_families)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.backbone(features.float())
        return {
            "pool_logits": self.pool_classifier(encoded),
            "family_logits": self.family_classifier(encoded),
        }


@dataclass
class AttributionLossResult:
    loss: torch.Tensor
    metrics: dict[str, float]
    decoded: dict[str, Any]


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor, *, bins: int = 10) -> torch.Tensor:
    ece = torch.zeros((), dtype=confidence.dtype, device=confidence.device)
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        ece = ece + in_bin.float().mean() * (confidence[in_bin].mean() - correct[in_bin].float().mean()).abs()
    return ece


def attribution_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor | list[str]],
    resolver: dict[str, Any],
) -> AttributionLossResult:
    pool_targets = labels["pool_index"]
    family_targets = labels["family_index"]
    if not isinstance(pool_targets, torch.Tensor) or not isinstance(family_targets, torch.Tensor):
        raise TypeError("CARA labels must contain tensor pool_index and family_index targets.")
    pool_to_family_index = {int(key): int(value) for key, value in resolver["pool_to_family_index"].items()}
    pool_logits = outputs["pool_logits"]
    family_logits = outputs["family_logits"]
    masked_logits = masked_pool_logits(pool_logits, family_targets, pool_to_family_index)
    family_loss = F.cross_entropy(family_logits, family_targets)
    pool_loss = F.cross_entropy(masked_logits, pool_targets)
    loss = pool_loss + 0.5 * family_loss

    family_pred = family_logits.argmax(dim=-1)
    pool_probs = F.softmax(masked_logits, dim=-1)
    pool_confidence, pool_pred = pool_probs.max(dim=-1)
    top_k = min(5, pool_logits.shape[-1])
    pool_topk = torch.topk(pool_probs, k=top_k, dim=-1).indices
    pool_correct = pool_pred == pool_targets
    family_correct = family_pred == family_targets
    top5_correct = (pool_topk == pool_targets.unsqueeze(-1)).any(dim=-1)
    entropy = -(pool_probs.clamp_min(1e-12) * pool_probs.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = math.log(max(2, pool_logits.shape[-1]))
    hierarchical_valid = torch.tensor(
        [
            pool_to_family_index.get(int(pool.item()), -1) == int(family.item())
            for pool, family in zip(pool_pred.detach().cpu(), family_pred.detach().cpu())
        ],
        dtype=torch.float32,
        device=pool_logits.device,
    )
    ece = expected_calibration_error(pool_confidence, pool_correct)
    pool_by_index = {int(key): value for key, value in resolver["pool_by_index"].items()}
    decoded_top1 = [pool_by_index.get(int(index), "UNKNOWN") for index in pool_pred.detach().cpu().tolist()]
    decoded_topk = [
        [pool_by_index.get(int(index), "UNKNOWN") for index in row]
        for row in pool_topk.detach().cpu().tolist()
    ]

    metrics = {
        "cara/pool_loss": float(pool_loss.detach().cpu()),
        "cara/family_loss": float(family_loss.detach().cpu()),
        "cara/attribution_loss": float(loss.detach().cpu()),
        "cara/pool_top1": float(pool_correct.float().mean().detach().cpu()),
        "cara/pool_top5": float(top5_correct.float().mean().detach().cpu()),
        "cara/family_top1": float(family_correct.float().mean().detach().cpu()),
        "cara/hierarchical_valid": float(hierarchical_valid.mean().detach().cpu()),
        "cara/pool_confidence": float(pool_confidence.mean().detach().cpu()),
        "cara/pool_entropy_normalized": float((entropy.mean() / max_entropy).detach().cpu()),
        "cara/ece": float(ece.detach().cpu()),
    }
    return AttributionLossResult(
        loss=loss,
        metrics=metrics,
        decoded={
            "top1_cara_pool_id": decoded_top1,
            "topk_cara_pool_id": decoded_topk,
            "registry_valid": all(value != "UNKNOWN" for value in decoded_top1),
        },
    )


def _extract_training_loss(result: Any) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, dict) and isinstance(result.get("loss"), torch.Tensor):
        return result["loss"]
    raise RuntimeError(f"Stable Audio training_step returned unsupported result for CARA loss: {type(result).__name__}")


def _add_head_params_to_optimizer(optimizer: Any, head: nn.Module) -> None:
    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "add_param_group"):
        return
    existing = {id(param) for group in optimizer.param_groups for param in group.get("params", [])}
    missing = [param for param in head.parameters() if param.requires_grad and id(param) not in existing]
    if missing:
        optimizer.add_param_group({"params": missing})


def _ensure_head_params_in_optimizers(configured: Any, head: nn.Module) -> Any:
    if hasattr(configured, "param_groups"):
        _add_head_params_to_optimizer(configured, head)
        return configured
    if isinstance(configured, dict):
        optimizer = configured.get("optimizer")
        if optimizer is not None:
            _add_head_params_to_optimizer(optimizer, head)
        return configured
    if isinstance(configured, tuple):
        for item in configured:
            _ensure_head_params_in_optimizers(item, head)
        return configured
    if isinstance(configured, list):
        for item in configured:
            _ensure_head_params_in_optimizers(item, head)
        return configured
    return configured


def attach_cara_attribution_to_training_wrapper(
    training_wrapper: nn.Module,
    model: nn.Module,
    *,
    resolver: dict[str, Any],
    variant: str,
    loss_weight: float,
    optimizer_lr: float,
    detach_features: bool,
    freeze_backbone: bool,
    report: dict[str, Any],
) -> nn.Module:
    import types

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    tapper = CaraHiddenStateTapper(model)
    tap_report = tapper.register()
    head = CaraAttributionHead(
        num_pools=int(resolver["pool_count"]),
        num_families=int(resolver["family_count"]),
    )
    setattr(training_wrapper, "cara_attribution_head", head)
    setattr(training_wrapper, "cara_hidden_state_tapper", tapper)
    original_training_step = training_wrapper.training_step
    original_configure_optimizers = getattr(training_wrapper, "configure_optimizers", None)

    def training_step_with_cara(self: nn.Module, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        tapper.clear()
        result = original_training_step(batch, batch_idx, *args, **kwargs)
        base_loss = _extract_training_loss(result)
        metadata = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        batch_size = int(batch[0].shape[0]) if isinstance(batch, (list, tuple)) and hasattr(batch[0], "shape") else 0
        features = tapper.pooled_features(batch_size)
        if detach_features:
            features = features.detach()
        labels = labels_from_metadata(metadata, device=features.device)
        head_outputs = self.cara_attribution_head(features)
        loss_result = attribution_loss_and_metrics(head_outputs, labels, resolver)
        total_loss = base_loss + float(loss_weight) * loss_result.loss
        if hasattr(self, "log"):
            try:
                self.log("cara/base_loss", base_loss, prog_bar=False, on_step=True)
                self.log("cara/weighted_attribution_loss", float(loss_weight) * loss_result.loss, prog_bar=True, on_step=True)
                for key, value in loss_result.metrics.items():
                    self.log(key, value, prog_bar=key in {"cara/pool_top1", "cara/family_top1"}, on_step=True)
            except Exception:
                pass
        report["latest_cara_decoded_preview"] = {
            "batch_idx": int(batch_idx),
            "top1_cara_pool_id": loss_result.decoded["top1_cara_pool_id"][:5],
            "registry_valid": loss_result.decoded["registry_valid"],
        }
        report["latest_cara_metrics"] = dict(loss_result.metrics)
        report["latest_cara_total_loss"] = float(total_loss.detach().cpu())
        if isinstance(result, dict):
            result = dict(result)
            result["loss"] = total_loss
            return result
        return total_loss

    training_wrapper.training_step = types.MethodType(training_step_with_cara, training_wrapper)
    if freeze_backbone:

        def configure_head_only_optimizer(self: nn.Module) -> Any:
            return torch.optim.AdamW(self.cara_attribution_head.parameters(), lr=float(optimizer_lr))

        training_wrapper.configure_optimizers = types.MethodType(configure_head_only_optimizer, training_wrapper)
    elif original_configure_optimizers is not None:

        def configure_optimizers_with_cara(self: nn.Module) -> Any:
            configured = original_configure_optimizers()
            return _ensure_head_params_in_optimizers(configured, self.cara_attribution_head)

        training_wrapper.configure_optimizers = types.MethodType(configure_optimizers_with_cara, training_wrapper)
    report["cara_attribution"] = {
        "variant": variant,
        "loss_weight": float(loss_weight),
        "optimizer_lr": float(optimizer_lr),
        "optimizer_mode": "head_only" if freeze_backbone else "stable_audio_plus_head",
        "detach_features": bool(detach_features),
        "freeze_backbone": bool(freeze_backbone),
        "head": "CaraAttributionHead(pool+family hierarchical classifiers)",
        **tap_report,
    }
    return training_wrapper
