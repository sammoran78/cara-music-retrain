from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from benchmark_testing_ace_step_audio import _call_pipeline, _load_pipeline
except ModuleNotFoundError:
    from src.benchmark_testing_ace_step_audio import _call_pipeline, _load_pipeline
try:
    from cara_attribution_head import CaraAttributionHead, CaraHiddenStateTapper, masked_pool_logits
except ModuleNotFoundError:
    from src.cara_attribution_head import CaraAttributionHead, CaraHiddenStateTapper, masked_pool_logits
try:
    from test_prep_common import base_metadata, parse_bool, write_report
except ModuleNotFoundError:
    from src.test_prep_common import base_metadata, parse_bool, write_report
try:
    from evaluation.cara_repairability import aggregate_repairability, resolve_prediction
except ModuleNotFoundError:
    from cara_repairability import aggregate_repairability, resolve_prediction


CARA_MODEL_ID = "hybrid_ace_step_cara_strong_full"
PENDING_STATUS = "pending_ace_step_native_scorer"
BLOCKED_MISSING_HEAD_STATUS = "blocked_missing_ace_native_head"
BLOCKED_INCOMPATIBLE_HEAD_STATUS = "blocked_incompatible_ace_native_head"


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


def _parse_model_ids(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _expected(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    return {
        "cara_pool_id": expected.get("cara_pool_id") or row.get("expected_cara_pool_id"),
        "cara_pool_index": expected.get("cara_pool_index") or row.get("expected_cara_pool_index"),
        "cara_pool_family": expected.get("cara_pool_family") or row.get("expected_cara_pool_family"),
        "cara_pool_family_index": expected.get("cara_pool_family_index") or row.get("expected_cara_pool_family_index"),
        "cara_pool_codeword": expected.get("cara_pool_codeword") or row.get("expected_cara_pool_codeword"),
    }


def _audio_path_exists(row: dict[str, Any], generated_audio_dir: Path) -> bool:
    audio_path = row.get("audio_path")
    if not audio_path:
        return False
    path = Path(str(audio_path))
    if path.is_absolute():
        return path.exists()
    return (generated_audio_dir / path).exists()


def _resolver_from_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_by_index: dict[str, str] = {}
    family_by_index: dict[str, str] = {}
    pool_to_family_index: dict[str, str] = {}
    pool_to_family_name: dict[str, str] = {}
    for row in rows:
        expected = _expected(row)
        pool_id = expected.get("cara_pool_id")
        pool_index = expected.get("cara_pool_index")
        family = expected.get("cara_pool_family")
        family_index = expected.get("cara_pool_family_index")
        if pool_id not in (None, "") and pool_index not in (None, ""):
            pool_by_index[str(int(pool_index))] = str(pool_id)
        if family not in (None, "") and family_index not in (None, ""):
            family_by_index[str(int(family_index))] = str(family)
        if pool_index not in (None, "") and family_index not in (None, ""):
            pool_to_family_index[str(int(pool_index))] = str(int(family_index))
        if pool_id not in (None, "") and family not in (None, ""):
            pool_to_family_name[str(pool_id)] = str(family)
    return {
        "format": "cara_ace_scoring_resolver_from_generation_manifest_v1",
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family_index": pool_to_family_index,
        "pool_to_family_name": pool_to_family_name,
        "pool_count": len(pool_by_index),
        "family_count": len(family_by_index),
    }


def _read_torch_dict(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Native head artifact is not a dictionary payload: {path}")
    return payload


def _candidate_head_paths(trained_model_data: Path) -> list[Path]:
    preferred = [
        trained_model_data / "checkpoints" / "ace_attribution_head.pt",
        trained_model_data / "checkpoints" / "cara_attribution_head.pt",
        trained_model_data / "checkpoints" / "native_cara_head.pt",
        trained_model_data / "ace_attribution_head.pt",
        trained_model_data / "cara_attribution_head.pt",
    ]
    discovered: list[Path] = []
    try:
        discovered = [
            path
            for path in trained_model_data.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".pt", ".pth"}
            and "head" in path.name.lower()
            and ("cara" in path.name.lower() or "attribution" in path.name.lower())
        ]
    except OSError:
        discovered = []
    candidates = preferred + discovered
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _normalise_head_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    marker = "cara_attribution_head."
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        text = str(key)
        clean_key = text.split(marker, 1)[1] if marker in text else text
        if clean_key.startswith("module."):
            clean_key = clean_key.removeprefix("module.")
        if clean_key in {
            "pool_classifier.weight",
            "pool_classifier.bias",
            "family_classifier.weight",
            "family_classifier.bias",
        } or clean_key.startswith("backbone."):
            result[clean_key] = value.float() if value.is_floating_point() else value
    return result


def _extract_head_state_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in (
        "cara_attribution_head_state_dict",
        "ace_attribution_head_state_dict",
        "native_cara_head_state_dict",
        "head_state_dict",
        "state_dict",
    ):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            state = _normalise_head_state(candidate)
            if state:
                return state
    return _normalise_head_state(payload)


def _normalise_resolver(resolver: dict[str, Any]) -> dict[str, Any]:
    pool_by_index = {
        str(int(key)): str(value)
        for key, value in (resolver.get("pool_by_index") or {}).items()
        if value not in (None, "")
    }
    family_by_index = {
        str(int(key)): str(value)
        for key, value in (resolver.get("family_by_index") or {}).items()
        if value not in (None, "")
    }
    pool_to_family_index = {
        str(int(key)): str(int(value))
        for key, value in (resolver.get("pool_to_family_index") or {}).items()
        if value not in (None, "")
    }
    pool_to_family_name = {
        str(key): str(value)
        for key, value in (resolver.get("pool_to_family_name") or {}).items()
        if value not in (None, "")
    }
    payload = dict(resolver)
    payload.update(
        {
            "pool_by_index": pool_by_index,
            "family_by_index": family_by_index,
            "pool_to_family_index": pool_to_family_index,
            "pool_to_family_name": pool_to_family_name,
            "pool_count": len(pool_by_index),
            "family_count": len(family_by_index),
        }
    )
    return payload


def _resolver_is_compatible(resolver: dict[str, Any], dims: tuple[int, int]) -> bool:
    pool_count, family_count = dims
    pool_by_index = resolver.get("pool_by_index") if isinstance(resolver.get("pool_by_index"), dict) else {}
    family_by_index = resolver.get("family_by_index") if isinstance(resolver.get("family_by_index"), dict) else {}
    return len(pool_by_index) >= pool_count and len(family_by_index) >= family_count


def _checkpoint_head_dims(head_state: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    pool_weight = head_state.get("pool_classifier.weight")
    family_weight = head_state.get("family_classifier.weight")
    if isinstance(pool_weight, torch.Tensor) and isinstance(family_weight, torch.Tensor):
        return int(pool_weight.shape[0]), int(family_weight.shape[0])
    return None


def _checkpoint_feature_dim(head_state: dict[str, torch.Tensor]) -> int | None:
    first_weight = head_state.get("backbone.0.weight")
    if isinstance(first_weight, torch.Tensor) and first_weight.ndim == 2:
        return int(first_weight.shape[1])
    return None


def _native_head_artifact(trained_model_data: Path) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for path in _candidate_head_paths(trained_model_data):
        if not path.exists():
            checked.append({"path": str(path), "status": "missing"})
            continue
        try:
            payload = _read_torch_dict(path)
            state = _extract_head_state_from_payload(payload)
            dims = _checkpoint_head_dims(state)
            feature_dim = _checkpoint_feature_dim(state)
            if dims is None or feature_dim is None:
                checked.append(
                    {
                        "path": str(path),
                        "status": "incompatible",
                        "reason": "No generic CaraAttributionHead pool/family classifier plus backbone input weights were found.",
                        "payload_format": payload.get("format"),
                        "keys_preview": sorted(str(key) for key in payload.keys())[:20],
                    }
                )
                continue
            head_resolver = payload.get("resolver") if isinstance(payload.get("resolver"), dict) else None
            normalised_head_resolver = _normalise_resolver(head_resolver) if isinstance(head_resolver, dict) else None
            resolver_status = "missing"
            if isinstance(normalised_head_resolver, dict):
                resolver_status = "compatible" if _resolver_is_compatible(normalised_head_resolver, dims) else "incompatible"
            return state, {
                "status": "available",
                "path": str(path),
                "pool_count": dims[0],
                "family_count": dims[1],
                "feature_dim": feature_dim,
                "payload_format": payload.get("format"),
                "resolver_status": resolver_status,
                "resolver": normalised_head_resolver if resolver_status == "compatible" else None,
                "checked": checked,
            }
        except Exception as exc:
            checked.append({"path": str(path), "status": "failed_to_read", "error": repr(exc)})
    return None, {
        "status": BLOCKED_MISSING_HEAD_STATUS,
        "reason": (
            "No ACE native CARA attribution-head artifact was found. Expected one of "
            "checkpoints/ace_attribution_head.pt, checkpoints/cara_attribution_head.pt, or a compatible "
            "*attribution*head*.pt payload with CaraAttributionHead weights."
        ),
        "checked": checked,
    }


class AceNativeHiddenStateExtractor:
    def __init__(self, pipe: Any, resolver: dict[str, Any], head_state: dict[str, torch.Tensor]) -> None:
        target = getattr(pipe, "transformer", None) or getattr(pipe, "model", None) or getattr(pipe, "unet", None)
        if target is None:
            raise RuntimeError("ACE-Step pipeline does not expose a transformer/model/unet target for hidden-state taps.")
        dims = _checkpoint_head_dims(head_state)
        if dims is None:
            raise RuntimeError("ACE native attribution-head checkpoint is missing classifier weights.")
        self.pipe = pipe
        self.target = target
        self.resolver = resolver
        self.pool_by_index = {int(key): str(value) for key, value in (resolver.get("pool_by_index") or {}).items()}
        self.family_by_index = {int(key): str(value) for key, value in (resolver.get("family_by_index") or {}).items()}
        self.pool_to_family_index = {int(key): int(value) for key, value in (resolver.get("pool_to_family_index") or {}).items()}
        if not self.pool_by_index or not self.family_by_index:
            raise RuntimeError("ACE resolver is missing pool/family maps.")
        num_pools, num_families = dims
        if len(self.pool_by_index) < num_pools or len(self.family_by_index) < num_families:
            raise RuntimeError(
                "ACE resolver is smaller than the native attribution head. "
                f"resolver pools/families={len(self.pool_by_index)}/{len(self.family_by_index)}, "
                f"checkpoint pools/families={num_pools}/{num_families}."
            )
        self.expected_feature_dim = _checkpoint_feature_dim(head_state)
        self.head_state = head_state
        self.tapper = CaraHiddenStateTapper(target)
        self.tap_report = self.tapper.register()
        self.head = CaraAttributionHead(num_pools=num_pools, num_families=num_families)
        self.initialized = False
        self.last_feature_alignment: dict[str, Any] = {}

    def close(self) -> None:
        self.tapper.close()

    def clear(self) -> None:
        self.tapper.clear()

    def _ensure_loaded(self, features: torch.Tensor) -> None:
        if self.initialized:
            return
        self.head.to(features.device)
        with torch.no_grad():
            self.head(features[:1])
        missing, unexpected = self.head.load_state_dict(self.head_state, strict=False)
        critical_missing = [key for key in missing if key.endswith(".weight") or key.endswith(".bias")]
        if critical_missing or unexpected:
            raise RuntimeError(
                "ACE native attribution-head checkpoint did not match the extractor head. "
                f"missing={critical_missing[:8]}, unexpected={list(unexpected)[:8]}"
            )
        self.head.eval()
        self.initialized = True

    def _pooled_features(self, *, batch_size: int) -> torch.Tensor:
        usable: list[torch.Tensor] = []
        observed_shapes: list[list[int]] = []
        adjustments: list[dict[str, Any]] = []
        for index, feature in enumerate(self.tapper.features):
            if not isinstance(feature, torch.Tensor):
                continue
            if feature.ndim != 2:
                observed_shapes.append([int(dim) for dim in feature.shape])
                continue
            rows = int(feature.shape[0])
            width = int(feature.shape[-1])
            observed_shapes.append([rows, width])
            if rows == batch_size:
                usable.append(feature)
                adjustments.append({"tap_index": index, "mode": "exact", "shape": [rows, width]})
            elif rows > batch_size and rows % batch_size == 0:
                branches = rows // batch_size
                usable.append(feature.reshape(batch_size, branches, width).mean(dim=1))
                adjustments.append(
                    {
                        "tap_index": index,
                        "mode": "mean_over_generation_branches",
                        "branches": branches,
                        "shape": [rows, width],
                    }
                )
            elif rows > batch_size:
                usable.append(feature[:batch_size])
                adjustments.append({"tap_index": index, "mode": "trim_to_requested_batch", "shape": [rows, width]})
        self.last_feature_alignment = {
            "requested_batch_size": int(batch_size),
            "observed_feature_shapes": observed_shapes[:16],
            "usable_feature_count": len(usable),
            "adjustments": adjustments[:16],
            "policy": "ACE native extraction uses generation-time DiT hidden-state taps; CFG/branch rows are averaged when batch-aligned.",
        }
        if not usable:
            raise RuntimeError(
                "No ACE DiT hidden-state tap produced a generation-aligned feature tensor. "
                f"requested_batch_size={batch_size}; observed_feature_shapes={observed_shapes[:16]}"
            )
        features = torch.cat(usable, dim=-1)
        actual_dim = int(features.shape[-1])
        expected_dim = int(getattr(self, "expected_feature_dim", None) or actual_dim)
        mode = "exact"
        if actual_dim < expected_dim:
            features = F.pad(features, (0, expected_dim - actual_dim))
            mode = "right_zero_pad_to_checkpoint_width"
        elif actual_dim > expected_dim:
            features = features[:, :expected_dim]
            mode = "right_trim_to_checkpoint_width"
        self.last_feature_alignment = {
            **self.last_feature_alignment,
            "actual_feature_dim": actual_dim,
            "expected_feature_dim": expected_dim,
            "feature_dim_mode": mode,
        }
        return features

    def predict_latest(self, *, batch_size: int = 1) -> dict[str, Any]:
        features = self._pooled_features(batch_size=batch_size)
        self._ensure_loaded(features)
        with torch.no_grad():
            outputs = self.head(features)
            family_logits = outputs["family_logits"]
            family_probs = F.softmax(family_logits, dim=-1)
            family_confidence, family_pred = family_probs.max(dim=-1)
            masked_logits = masked_pool_logits(outputs["pool_logits"], family_pred, self.pool_to_family_index)
            pool_probs = F.softmax(masked_logits, dim=-1)
            pool_confidence, pool_pred = pool_probs.max(dim=-1)
            top_k = min(5, pool_probs.shape[-1])
            top_values, top_indices = torch.topk(pool_probs, k=top_k, dim=-1)
            entropy = -(pool_probs.clamp_min(1e-12) * pool_probs.clamp_min(1e-12).log()).sum(dim=-1)
        pool_index = int(pool_pred[0].detach().cpu())
        family_index = int(family_pred[0].detach().cpu())
        return {
            "status": "scored",
            "source": "ace_step_dit_generation_hidden_state_tap",
            "cara_pool_index": pool_index,
            "cara_pool_id": self.pool_by_index.get(pool_index),
            "cara_pool_family_index": family_index,
            "cara_pool_family": self.family_by_index.get(family_index),
            "confidence": float(pool_confidence[0].detach().cpu()),
            "family_confidence": float(family_confidence[0].detach().cpu()),
            "entropy": float(entropy[0].detach().cpu()),
            "top_k": [
                {
                    "cara_pool_index": int(index),
                    "cara_pool_id": self.pool_by_index.get(int(index)),
                    "confidence": float(value),
                    "registry_valid": int(index) in self.pool_by_index,
                }
                for index, value in zip(top_indices[0].detach().cpu().tolist(), top_values[0].detach().cpu().tolist())
            ],
            "registry_valid": pool_index in self.pool_by_index,
            "feature_alignment": self.last_feature_alignment,
        }


def _lane_pending(rows: list[dict[str, Any]], generated_audio_dir: Path) -> dict[str, Any]:
    labelled = [row for row in rows if _expected(row).get("cara_pool_id")]
    evidence_counts = _lane_evidence_counts(rows, generated_audio_dir)
    examples: list[dict[str, Any]] = []
    for row in rows[:5]:
        expected = _expected(row)
        examples.append(
            {
                "lane": "hybrid_native",
                "status": PENDING_STATUS,
                "prompt_id": row.get("prompt_id"),
                "suite_id": row.get("suite_id"),
                "seed": row.get("seed"),
                "audio_path": row.get("audio_path"),
                "expected_pool_id": expected.get("cara_pool_id"),
                "expected_family": expected.get("cara_pool_family"),
                "predicted_pool_id": None,
                "resolved_pool_id": None,
                "confidence": None,
                "tier": "pending_native_scorer",
                "prediction_error": "ACE-Step generated audio exists, but the native DiT-head CARA extractor is not implemented yet.",
                "latent_evidence": row.get("latent_evidence"),
            }
        )
    return {
        "model_id": CARA_MODEL_ID,
        "variant": "cara_strong",
        "evidence_lane": "native",
        "status": PENDING_STATUS,
        "reason": "ACE-Step full generated audio is available; native Hybrid CARA scoring requires the follow-on ACE DiT-head extractor.",
        "count": len(rows),
        "labelled_count": len(labelled),
        "audio_file_count": evidence_counts["audio_file_count"],
        "latent_evidence_count": evidence_counts["latent_evidence_count"],
        "pool_exact_accuracy": None,
        "exact_pool_top1": None,
        "exact_pool_top3": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "family_accuracy": None,
        "family_or_genre_accuracy": None,
        "registry_valid_rate": None,
        "unattributable_rate": None,
        "repairability": {
            "status": PENDING_STATUS,
            "correctness_semantics": "not_scored",
            "labelled_total": len(labelled),
            "reason": "Repairability cannot be calculated until real Hybrid native/probe predictions exist.",
        },
        "repair_method_counts": {},
        "prediction_examples": examples,
    }


def _lane_blocked_missing_head(
    rows: list[dict[str, Any]],
    generated_audio_dir: Path,
    head_report: dict[str, Any],
) -> dict[str, Any]:
    lane = _lane_pending(rows, generated_audio_dir)
    lane.update(
        {
            "status": BLOCKED_MISSING_HEAD_STATUS,
            "reason": head_report.get("reason")
            or "ACE-Step native scoring needs a trained CARA attribution-head artifact before it can decode pool IDs.",
            "native_head": head_report,
            "repairability": {
                "status": BLOCKED_MISSING_HEAD_STATUS,
                "correctness_semantics": "not_scored",
                "labelled_total": lane["labelled_count"],
                "reason": "No native ACE CARA head artifact was available, so no prediction rows were emitted.",
            },
        }
    )
    for example in lane.get("prediction_examples") or []:
        example["status"] = BLOCKED_MISSING_HEAD_STATUS
        example["tier"] = "blocked_native_head"
        example["prediction_error"] = lane["reason"]
    return lane


def _lane_evidence_counts(rows: list[dict[str, Any]], generated_audio_dir: Path) -> dict[str, int]:
    latent_rows = [
        row
        for row in rows
        if isinstance(row.get("latent_evidence"), dict)
        and str(row.get("latent_evidence", {}).get("status") or "").lower() in {"captured", "present", "available"}
    ]
    return {
        "audio_file_count": sum(1 for row in rows if _audio_path_exists(row, generated_audio_dir)),
        "latent_evidence_count": len(latent_rows),
    }


def _score_prediction(row: dict[str, Any], prediction: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    decision = resolve_prediction(prediction, _expected(row), resolver)
    confidence = prediction.get("confidence")
    try:
        confidence_value = float(confidence) if confidence not in (None, "") else None
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "model_id": row.get("model_id"),
        "suite_id": row.get("suite_id"),
        "prompt_id": row.get("prompt_id"),
        "seed": row.get("seed"),
        "audio_path": row.get("audio_path"),
        "expected_pool_id": decision.expected_pool_id,
        "expected_family": decision.expected_family,
        "predicted_pool_id": decision.predicted_pool_id,
        "predicted_family": decision.predicted_family,
        "resolved_pool_id": decision.resolved_pool_id,
        "resolved_family": decision.resolved_family,
        "registry_valid": decision.registry_valid,
        "exact": decision.pool_exact_correct,
        "repairable": decision.pool_repaired_correct and decision.tier == "repairable_pool",
        "family_match": decision.family_correct,
        "unattributable": decision.tier == "unattributable",
        "tier": decision.tier,
        "repair_method": decision.repair_method,
        "repair_distance": decision.repair_distance,
        "confidence": confidence_value,
        "top_k": prediction.get("top_k") or [],
        "prediction_status": prediction.get("status"),
        "prediction_source": prediction.get("source"),
        "prediction_error": prediction.get("error"),
        "feature_alignment": prediction.get("feature_alignment"),
    }


def _prediction_examples(predictions: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "model_id": row.get("model_id"),
            "suite_id": row.get("suite_id"),
            "prompt_id": row.get("prompt_id"),
            "audio_path": row.get("audio_path"),
            "tier": row.get("tier"),
            "expected_pool_id": row.get("expected_pool_id"),
            "predicted_pool_id": row.get("predicted_pool_id"),
            "resolved_pool_id": row.get("resolved_pool_id"),
            "expected_family": row.get("expected_family"),
            "predicted_family": row.get("predicted_family"),
            "resolved_family": row.get("resolved_family"),
            "confidence": row.get("confidence"),
            "registry_valid": row.get("registry_valid"),
            "repair_method": row.get("repair_method"),
            "repair_distance": row.get("repair_distance"),
            "exact": row.get("exact"),
            "repairable": row.get("repairable"),
            "family_match": row.get("family_match"),
            "top_k": row.get("top_k") or [],
            "prediction_status": row.get("prediction_status"),
            "prediction_source": row.get("prediction_source"),
            "prediction_error": row.get("prediction_error"),
            "feature_alignment": row.get("feature_alignment"),
        }
        for row in predictions[:limit]
    ]


def _ece(scored: list[dict[str, Any]]) -> float | None:
    with_conf = [row for row in scored if isinstance(row.get("confidence"), (int, float))]
    if not with_conf:
        return None
    total = len(with_conf)
    value = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        bucket = [row for row in with_conf if lower < float(row["confidence"]) <= upper]
        if not bucket:
            continue
        avg_conf = sum(float(row["confidence"]) for row in bucket) / len(bucket)
        avg_acc = sum(1.0 for row in bucket if row.get("exact")) / len(bucket)
        value += len(bucket) / total * abs(avg_conf - avg_acc)
    return value


def _prediction_error_count(predictions: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in predictions
        if str(row.get("prediction_status") or "").strip().lower() in {"exception", "failed", "error"}
    )


def _prediction_status_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in predictions:
        counts[str(row.get("prediction_status") or "predicted").strip().lower() or "predicted"] += 1
    return dict(sorted(counts.items()))


def _prediction_error_examples(predictions: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in predictions:
        if str(row.get("prediction_status") or "").strip().lower() not in {"exception", "failed", "error"}:
            continue
        examples.append(
            {
                "model_id": row.get("model_id"),
                "suite_id": row.get("suite_id"),
                "prompt_id": row.get("prompt_id"),
                "prediction_status": row.get("prediction_status"),
                "prediction_error": row.get("prediction_error"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _fallback_peer_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    expected = [str(row.get("expected_pool_id") or "") for row in predictions]
    predicted = [str(row.get("resolved_pool_id") or row.get("predicted_pool_id") or "") for row in predictions]
    labels = sorted({label for label in expected if label})
    balanced = None
    macro = None
    if labels:
        recalls: list[float] = []
        f1s: list[float] = []
        for label in labels:
            tp = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred == label)
            fp = sum(1 for exp, pred in zip(expected, predicted) if exp != label and pred == label)
            fn = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            recalls.append(recall)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        balanced = sum(recalls) / len(recalls)
        macro = sum(f1s) / len(f1s)
    return {
        "exact_pool_top1": sum(1 for row in predictions if row.get("exact")) / len(predictions),
        "exact_pool_top3": sum(1 for row in predictions if row.get("exact")) / len(predictions),
        "balanced_accuracy": balanced,
        "macro_f1": macro,
        "family_accuracy": sum(1 for row in predictions if row.get("family_match")) / len(predictions),
        "registry_valid_rate": sum(1 for row in predictions if row.get("registry_valid")) / len(predictions),
    }


def _lane_metrics_from_predictions(scored_rows: list[dict[str, Any]], labelled_count: int) -> dict[str, Any]:
    if not scored_rows:
        return {"status": "missing_predictions", "count": 0, "labelled_count": labelled_count}
    denominator = len(scored_rows)
    error_count = _prediction_error_count(scored_rows)
    if error_count >= denominator:
        return {
            "status": "extractor_failed",
            "count": denominator,
            "labelled_count": labelled_count,
            "reason": "All ACE native prediction attempts failed inside the extractor.",
            "prediction_status_counts": _prediction_status_counts(scored_rows),
            "exception_rate": 1.0,
            "error_examples": _prediction_error_examples(scored_rows),
            "prediction_examples": _prediction_examples(scored_rows),
            "prediction_rows": scored_rows,
        }
    peer_metrics = _fallback_peer_metrics(scored_rows)
    repairability = aggregate_repairability(scored_rows)
    status = "scored_with_extractor_errors" if error_count else "scored"
    return {
        "model_id": CARA_MODEL_ID,
        "variant": "cara_strong",
        "evidence_lane": "native",
        "status": status,
        "count": denominator,
        "labelled_count": labelled_count,
        "exact_pool_top1": peer_metrics.get("exact_pool_top1"),
        "exact_pool_top3": peer_metrics.get("exact_pool_top3"),
        "balanced_accuracy": peer_metrics.get("balanced_accuracy"),
        "macro_f1": peer_metrics.get("macro_f1"),
        "family_accuracy": peer_metrics.get("family_accuracy"),
        "family_or_genre_accuracy": repairability.get("family_or_genre_accuracy"),
        "pool_exact_accuracy": sum(1 for row in scored_rows if row.get("exact")) / denominator,
        "pool_repaired_accuracy": repairability.get("pool_repaired_accuracy"),
        "pool_recovered_accuracy": repairability.get("pool_recovered_accuracy"),
        "unattributable_rate": repairability.get("unattributable_rate"),
        "registry_valid_rate": peer_metrics.get("registry_valid_rate"),
        "ece": _ece(scored_rows),
        "exception_rate": error_count / denominator,
        "repairability": repairability,
        "repair_method_counts": repairability.get("repair_method_counts") or {},
        "prediction_status_counts": _prediction_status_counts(scored_rows),
        "error_examples": _prediction_error_examples(scored_rows),
        "prediction_examples": _prediction_examples(scored_rows),
        "prediction_rows": scored_rows,
    }


def _write_native_predictions(
    *,
    rows: list[dict[str, Any]],
    generated_audio_dir: Path,
    output_dir: Path,
    trained_model_data: Path,
    checkpoint_dir: Path | None,
    checkpoint: str,
    duration_seconds: float,
    num_inference_steps: int,
    guidance_scale: float,
    max_native_predictions: int,
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labelled = [row for row in rows if _expected(row).get("cara_pool_id")]
    head_state, head_report = _native_head_artifact(trained_model_data)
    if head_state is None:
        lane = _lane_blocked_missing_head(rows, generated_audio_dir, head_report)
        return rows, lane, {"status": BLOCKED_MISSING_HEAD_STATUS, "native_head": head_report, "count": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("ACE-Step native CARA scoring is GPU-only once a native head is available; CUDA is not available.")

    candidates = labelled[:]
    if max_native_predictions > 0:
        candidates = candidates[:max_native_predictions]
    generated_resolver = _resolver_from_generation_rows(rows)
    head_resolver = head_report.get("resolver") if isinstance(head_report.get("resolver"), dict) else None
    resolver_source = "native_head_checkpoint_resolver" if isinstance(head_resolver, dict) else "generation_manifest_subset_resolver"
    resolver = head_resolver if isinstance(head_resolver, dict) else generated_resolver
    head_dims = _checkpoint_head_dims(head_state)
    if head_dims is not None and not _resolver_is_compatible(resolver, head_dims):
        raise RuntimeError(
            "ACE native scoring resolver is incompatible with the trained native head. "
            f"resolver_source={resolver_source}; resolver pools/families="
            f"{len((resolver.get('pool_by_index') or {}))}/{len((resolver.get('family_by_index') or {}))}; "
            f"checkpoint pools/families={head_dims[0]}/{head_dims[1]}. "
            "Expected the Step 13 head artifact to include the full training registry resolver."
        )
    pipe, adapter_report = _load_pipeline(
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        trained_model_data=trained_model_data,
        work_dir=output_dir / "runtime_cache",
        report=report,
    )
    report["native_head_resolver"] = {
        "source": resolver_source,
        "pool_count": len((resolver.get("pool_by_index") or {})),
        "family_count": len((resolver.get("family_by_index") or {})),
        "generated_manifest_pool_count": len((generated_resolver.get("pool_by_index") or {})),
        "generated_manifest_family_count": len((generated_resolver.get("family_by_index") or {})),
    }
    extractor = AceNativeHiddenStateExtractor(pipe, resolver, head_state)
    updated_by_key: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    scored_rows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(candidates, start=1):
            extractor.clear()
            try:
                _call_pipeline(
                    pipe,
                    prompt=str(row.get("prompt") or "CARA benchmark audio"),
                    seed=int(row.get("seed") or 0),
                    duration_seconds=float(row.get("duration_seconds") or duration_seconds),
                    num_inference_steps=int(row.get("num_inference_steps") or num_inference_steps),
                    guidance_scale=float(row.get("guidance_scale") or guidance_scale),
                )
                prediction = extractor.predict_latest(batch_size=1)
            except Exception as exc:
                prediction = {
                    "status": "exception",
                    "source": "ace_step_dit_generation_hidden_state_tap",
                    "error": repr(exc),
                    "cara_pool_id": None,
                    "cara_pool_index": None,
                    "cara_pool_family": None,
                    "cara_pool_family_index": None,
                    "confidence": None,
                    "registry_valid": False,
                    "top_k": [],
                    "feature_alignment": getattr(extractor, "last_feature_alignment", {}),
                    "policy": {"copied_from_expected": False},
                }
            updated = dict(row)
            updated["native_cara_prediction"] = prediction
            updated["native_cara_prediction_generated_at"] = _utc_now()
            updated["native_cara_prediction_policy"] = {
                "no_expected_label_copying": True,
                "source_prompt_replay": True,
                "trained_model_data": str(trained_model_data),
                "source_audio_path": row.get("audio_path"),
                "feature_source": "ace_step_dit_generation_hidden_state_tap",
                "resolver_source": resolver_source,
            }
            key = (row.get("model_id"), row.get("suite_id"), row.get("prompt_id"), row.get("seed"))
            updated_by_key[key] = updated
            scored = _score_prediction(updated, prediction, resolver)
            scored_rows.append(scored)
            report["native_predictions_completed"] = index
            report["native_predictions_total"] = len(candidates)
    finally:
        extractor.close()
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("model_id"), row.get("suite_id"), row.get("prompt_id"), row.get("seed"))
        updated_rows.append(updated_by_key.get(key, row))
    _write_jsonl(output_dir / "native_predictions.jsonl", scored_rows)
    _write_jsonl(output_dir / "scored_generation_manifest.jsonl", updated_rows)
    lane = _lane_metrics_from_predictions(scored_rows, len(labelled))
    lane.update(_lane_evidence_counts(rows, generated_audio_dir))
    lane["native_head"] = head_report
    lane["resolver_source"] = resolver_source
    return updated_rows, lane, {
        "status": "scored",
        "count": len(scored_rows),
        "native_head": head_report,
        "adapter": adapter_report,
        "resolver_source": resolver_source,
        "resolver_pool_count": len((resolver.get("pool_by_index") or {})),
        "resolver_family_count": len((resolver.get("family_by_index") or {})),
        "generated_manifest_pool_count": len((generated_resolver.get("pool_by_index") or {})),
        "generated_manifest_family_count": len((generated_resolver.get("family_by_index") or {})),
        "tap_report": extractor.tap_report,
        "prediction_file": "native_predictions.jsonl",
        "scored_generation_manifest": "scored_generation_manifest.jsonl",
    }


def _repairability_matrix(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tiers = [
        ("exact_pool", "Exact pool"),
        ("repairable_pool", "Repairable pool"),
        ("family_or_genre", "Family / genre fallback"),
        ("unattributable", "Unattributable"),
    ]
    rows: list[dict[str, Any]] = []
    for tier_id, label in tiers:
        row: dict[str, Any] = {"tier": tier_id, "label": label}
        for lane_id, lane in lanes.items():
            repairability = lane.get("repairability") if isinstance(lane.get("repairability"), dict) else {}
            counts = repairability.get("correct_tier_counts") if isinstance(repairability.get("correct_tier_counts"), dict) else {}
            rates = repairability.get("correct_tier_rates") if isinstance(repairability.get("correct_tier_rates"), dict) else {}
            count = counts.get(tier_id)
            rate = rates.get(tier_id)
            row[lane_id] = {
                "status": lane.get("status"),
                "count": int(count) if isinstance(count, int) else None,
                "rate": float(rate) if isinstance(rate, (int, float)) else None,
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)
    return {"format": "cara_repairability_matrix_v1", "lanes": list(lanes), "rows": rows}


def _benchmark_rows(lane: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = [
        ("pool_exact_accuracy", "Exact pool accuracy", "Predicted CARA pool-id exactly matches the known held-out pool."),
        ("pool_repaired_accuracy", "Repairable pool accuracy", "Wrong or malformed CARA-id uniquely repairs to the correct known pool."),
        ("family_or_genre_accuracy", "Family / genre fallback accuracy", "Pool is not exact, but attribution resolves to the correct CARA family or genre level."),
        ("unattributable_rate", "Unattributable rate", "No exact, repairable, or family-level CARA attribution can be recovered."),
        ("registry_valid_rate", "Registry-valid rate", "Predicted CARA-id is syntactically valid and present in the locked registry."),
        ("ece", "Calibration / ECE", "Expected calibration error for pool/family confidence."),
    ]
    return [
        {
            "id": metric_id,
            "metric": label,
            "description": description,
            "higher_is_better": metric_id not in {"unattributable_rate", "ece"},
            "hybrid_native": lane.get(metric_id) if isinstance(lane.get(metric_id), (int, float)) else None,
            "hybrid_native_status": lane.get("status"),
            "status": lane.get("status"),
        }
        for metric_id, label, description in metrics
    ]


def _repair_method_matrix(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(
        {
            str(method)
            for lane in lanes.values()
            if isinstance(lane.get("repair_method_counts"), dict)
            for method in lane.get("repair_method_counts", {}).keys()
        }
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {"method": method, "label": method.replace("_", " ")}
        for lane_id, lane in lanes.items():
            counts = lane.get("repair_method_counts") if isinstance(lane.get("repair_method_counts"), dict) else {}
            count = int(counts.get(method) or 0)
            total = lane.get("count") if isinstance(lane.get("count"), int) else None
            row[lane_id] = {
                "status": lane.get("status"),
                "count": count,
                "rate": (count / total) if total else None,
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)
    return {"format": "cara_repair_method_matrix_v1", "lanes": list(lanes), "rows": rows}


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    generated_audio_dir = Path(args.generated_audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_audio_dir / "generation_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing ACE-Step generation manifest: {manifest_path}")

    requested_model_ids = _parse_model_ids(args.model_ids)
    if requested_model_ids and any(model_id != CARA_MODEL_ID for model_id in requested_model_ids):
        raise RuntimeError(f"Unsupported ACE-Step scoring model ids: {requested_model_ids}")

    rows = [row for row in _read_jsonl(manifest_path) if str(row.get("model_id") or "") == CARA_MODEL_ID]
    if requested_model_ids:
        rows = [row for row in rows if str(row.get("model_id") or "") in requested_model_ids]
    if not rows:
        raise RuntimeError(f"No ACE-Step Hybrid rows found in {manifest_path}")

    native_extractor = parse_bool(args.native_extractor)
    trained_model_data = Path(args.trained_model_data) if str(args.trained_model_data or "").strip() else None
    checkpoint_dir = Path(args.checkpoint_dir) if str(args.checkpoint_dir or "").strip() else None
    if native_extractor:
        if trained_model_data is None or not trained_model_data.exists():
            raise RuntimeError(f"ACE-Step native scoring requires mounted trained_model_data: {trained_model_data}")
        rows, lane, native_extraction = _write_native_predictions(
            rows=rows,
            generated_audio_dir=generated_audio_dir,
            output_dir=output_dir,
            trained_model_data=trained_model_data,
            checkpoint_dir=checkpoint_dir,
            checkpoint=str(args.checkpoint),
            duration_seconds=float(args.duration_seconds),
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            max_native_predictions=max(0, int(args.max_native_predictions)),
            report=report,
        )
    else:
        lane = _lane_pending(rows, generated_audio_dir)
        native_extraction = {"status": PENDING_STATUS, "count": 0}
    lanes = {
        "hybrid_native": lane,
        "hybrid_external_probe": {
            "model_id": CARA_MODEL_ID,
            "variant": "cara_strong",
            "evidence_lane": "external_probe",
            "status": "pending_external_probe",
            "reason": "No post-hoc external probe has been implemented for ACE-Step generated audio yet.",
            "count": len(rows),
            "labelled_count": lane["labelled_count"],
            "repairability": {
                "status": "pending_external_probe",
                "correctness_semantics": "not_scored",
                "labelled_total": lane["labelled_count"],
            },
            "repair_method_counts": {},
            "prediction_examples": [],
        },
    }
    metrics = {
        "format": "cara_ace_step_benchmark_matrix_metrics_v1",
        "created_at": _utc_now(),
        "source_generation_manifest": str(manifest_path),
        "source_audio_output_dir": str(generated_audio_dir),
        "selected_model_ids": requested_model_ids or [CARA_MODEL_ID],
        "source_generation_row_count": len(rows),
        "generated_audio_count": len(rows),
        "by_model": dict(sorted(Counter(str(row.get("model_id") or "unknown") for row in rows).items())),
        "by_suite": dict(sorted(Counter(str(row.get("suite_id") or "unknown") for row in rows).items())),
        "lanes": lanes,
        "repairability_matrix": _repairability_matrix(lanes),
        "repair_method_matrix": _repair_method_matrix(lanes),
        "prediction_examples": {lane_id: item.get("prediction_examples") or [] for lane_id, item in lanes.items()},
        "benchmark_rows": _benchmark_rows(lane),
        "native_extraction": native_extraction,
        "scoring_policy": {
            "no_expected_label_copying": True,
            "native_extractor": (
                "ACE-Step native DiT-head extractor is active when a compatible native CARA head artifact exists; "
                "otherwise the lane is blocked without numeric metrics."
            ),
            "missing_prediction_behavior": "Hybrid metric cells remain pending until real native/probe predictions are written.",
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    }
    _write_json(output_dir / "metrics_latest.json", metrics)
    _write_jsonl(output_dir / "hybrid_scoring_pending_examples.jsonl", lane["prediction_examples"])
    report.update(
        {
            "status": "passed",
            "stage": "completed",
            "generated_audio_count": len(rows),
            "labelled_count": lane["labelled_count"],
            "audio_file_count": lane["audio_file_count"],
            "latent_evidence_count": lane["latent_evidence_count"],
            "metrics_available": lane.get("status") in {"scored", "scored_with_extractor_errors"},
            "native_scoring_status": lane.get("status"),
            "native_extraction": native_extraction,
            "output_metrics": "metrics_latest.json",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_audio_dir", required=True)
    parser.add_argument("--trained_model_data", default="")
    parser.add_argument("--checkpoint", default="ACE-Step/Ace-Step1.5")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_ids", default="")
    parser.add_argument("--native_extractor", default="true")
    parser.add_argument("--max_native_predictions", type=int, default=0)
    parser.add_argument("--duration_seconds", type=float, default=12.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "format": "cara_benchmark_ace_step_attribution_scoring_report_v1",
        "test_name": "25_benchmark_testing_ace_step_score",
        "stage": "start",
        "status": "failed",
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
        "dry_run": parse_bool(args.dry_run),
    }
    try:
        if report["dry_run"]:
            report["status"] = "planned"
            report["stage"] = "dry_run"
        else:
            run(args, report)
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        metadata = base_metadata(
            test_name="25_benchmark_testing_ace_step_score",
            compute="gpu-smoke-h100",
            environment="env-ace-step:5",
            dashboard_triggered=report.get("dashboard_triggered", False),
            report=report,
            model_family="ace_step",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_ace_step_score_report.json")
    return 0 if report.get("status") in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
