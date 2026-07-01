from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmark_testing_stable_audio_audio import (
    CONTEXT_DIFFUSION_MODEL_ID,
    _generate_one,
    _load_stable_audio_model,
    _load_trainable_delta,
    _structured_conditioning_from_prompt_row,
)
from cara_attribution_head import CaraAttributionHead, CaraHiddenStateTapper, masked_pool_logits
try:
    from evaluation.cara_repairability import aggregate_repairability, resolve_prediction
except ModuleNotFoundError:
    from cara_repairability import aggregate_repairability, resolve_prediction
from smoke_stable_audio_trainer import _configure_disk_safe_runtime_dirs
from test_prep_common import base_metadata, parse_bool, write_report

try:
    from evaluation.metrics_v2 import summarize_prediction_rows
except Exception:
    def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        expected = [str(row.get("expected_pool_id") or "") for row in rows]
        predicted = [str(row.get("predicted_pool_id") or "") for row in rows]
        labels = sorted({label for label in expected if label})
        exact = [exp and exp == pred for exp, pred in zip(expected, predicted)]
        balanced = None
        macro = None
        if labels:
            recalls = []
            f1s = []
            for label in labels:
                tp = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred == label)
                fp = sum(1 for exp, pred in zip(expected, predicted) if exp != label and pred == label)
                fn = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred != label)
                recalls.append(tp / (tp + fn) if tp + fn else 0.0)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            balanced = sum(recalls) / len(recalls)
            macro = sum(f1s) / len(f1s)
        with_conf = [row for row in rows if isinstance(row.get("confidence"), (int, float))]
        brier = (
            sum((float(row["confidence"]) - (1.0 if row.get("exact") else 0.0)) ** 2 for row in with_conf) / len(with_conf)
            if with_conf
            else None
        )
        return {
            "exact_pool_top1": sum(1 for item in exact if item) / len(rows),
            "exact_pool_top3": sum(1 for item in exact if item) / len(rows),
            "balanced_accuracy": balanced,
            "macro_f1": macro,
            "family_accuracy": sum(1 for row in rows if row.get("family_match")) / len(rows),
            "brier": brier,
            "ece": None,
            "registry_valid_rate": sum(1 for row in rows if row.get("registry_valid")) / len(rows),
            "repaired_rate": sum(1 for row in rows if row.get("tier") == "repairable_pool") / len(rows),
            "degraded_rate": sum(1 for row in rows if row.get("tier") == "family_or_genre") / len(rows),
            "exception_rate": sum(1 for row in rows if row.get("tier") == "exception") / len(rows),
        }


BASE_MODEL_ID = "base_stable_audio_open_small"
DIFFUSION_MODEL_ID = "diffusion_cara_strong_full_modest_arch"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _parse_model_ids(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _normalise_prediction(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    status = str(value.get("status") or "").strip().lower()
    if status.startswith("pending") or status in {"not_applicable", "n/a", "na", "missing"}:
        return None
    if status in {"exception", "failed", "error"}:
        return value
    if value.get("cara_pool_id") not in (None, "") or value.get("cara_pool_index") not in (None, ""):
        return value
    return None


def _resolver_from_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pool_by_index: dict[str, str] = {}
    family_by_index: dict[str, str] = {}
    pool_to_family_index: dict[str, str] = {}
    pool_to_family: dict[str, str] = {}
    for row in rows:
        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
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
            pool_to_family[str(pool_id)] = str(family)
    return {
        "format": "cara_scoring_resolver_from_generation_manifest_v1",
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family_index": pool_to_family_index,
        "pool_to_family": pool_to_family,
        "pool_count": len(pool_by_index),
        "family_count": len(family_by_index),
    }


def _resolver_maps(resolver: dict[str, Any]) -> dict[str, Any]:
    pool_by_index = {str(key): str(value) for key, value in (resolver.get("pool_by_index") or {}).items()}
    family_by_index = {str(key): str(value) for key, value in (resolver.get("family_by_index") or {}).items()}
    pool_to_family_index = {str(key): str(value) for key, value in (resolver.get("pool_to_family_index") or {}).items()}
    pool_to_family = {str(key): str(value) for key, value in (resolver.get("pool_to_family") or {}).items()}
    if not pool_to_family:
        for pool_index, family_index in pool_to_family_index.items():
            pool_id = pool_by_index.get(str(pool_index))
            family = family_by_index.get(str(family_index))
            if pool_id and family:
                pool_to_family[pool_id] = family
    return {
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family": pool_to_family,
    }


def _load_registry_resolver(trained_model_data: Path, generated_audio_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        trained_model_data / "cara_registry_resolver.json",
        trained_model_data / "work" / "cara_registry_resolver.json",
        trained_model_data / "outputs" / "cara_registry_resolver.json",
        generated_audio_dir / "cara_registry_resolver.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate)
    try:
        for candidate in trained_model_data.rglob("cara_registry_resolver.json"):
            return _read_json(candidate)
    except OSError:
        pass
    return _resolver_from_generation_rows(rows)


def _load_training_heldout_evaluation(model_data: Path) -> dict[str, Any]:
    candidates = [
        model_data / "stable_audio_smoke_trainer_report.json",
        model_data / "report.json",
        model_data / "work" / "stable_audio_smoke_trainer_report.json",
        model_data / "outputs" / "stable_audio_smoke_trainer_report.json",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        seen.add(str(candidate))
        if not candidate.exists():
            continue
        try:
            report = _read_json(candidate)
        except Exception as exc:
            return {"status": "report_unreadable", "report_path": str(candidate), "error": repr(exc)}
        heldout = report.get("heldout_evaluation")
        if isinstance(heldout, dict) and heldout:
            return {
                "status": "available",
                "report_path": str(candidate),
                "heldout_evaluation": heldout,
                "global_step": report.get("global_step") or report.get("trainer_global_step"),
            }
    try:
        for candidate in model_data.rglob("stable_audio_smoke_trainer_report.json"):
            if str(candidate) in seen:
                continue
            report = _read_json(candidate)
            heldout = report.get("heldout_evaluation")
            if isinstance(heldout, dict) and heldout:
                return {
                    "status": "available",
                    "report_path": str(candidate),
                    "heldout_evaluation": heldout,
                    "global_step": report.get("global_step") or report.get("trainer_global_step"),
                }
    except OSError:
        pass
    return {
        "status": "missing",
        "reason": "No heldout_evaluation block found in Stable Audio trainer report artifacts.",
        "model_data": str(model_data),
    }


def _resolver_pool_to_family_index(resolver: dict[str, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in (resolver.get("pool_to_family_index") or {}).items()}


def _extract_head_state(delta_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = delta_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        return {}
    head_state: dict[str, torch.Tensor] = {}
    marker = "cara_attribution_head."
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        text = str(key)
        if marker not in text:
            continue
        clean_key = text.split(marker, 1)[1]
        head_state[clean_key] = value.float() if value.is_floating_point() else value
    return head_state


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


class NativeCaraHiddenStateExtractor:
    def __init__(self, model: torch.nn.Module, resolver: dict[str, Any], head_state: dict[str, torch.Tensor]) -> None:
        if not head_state:
            raise RuntimeError("Trainable delta did not contain cara_attribution_head parameters.")
        self.model = model
        self.resolver = resolver
        self.pool_by_index = {int(key): str(value) for key, value in (resolver.get("pool_by_index") or {}).items()}
        self.family_by_index = {int(key): str(value) for key, value in (resolver.get("family_by_index") or {}).items()}
        self.pool_to_family_index = _resolver_pool_to_family_index(resolver)
        if not self.pool_by_index or not self.family_by_index:
            raise RuntimeError("CARA registry resolver is missing pool/family index maps.")
        checkpoint_dims = _checkpoint_head_dims(head_state)
        if checkpoint_dims is None:
            raise RuntimeError("CARA attribution head checkpoint is missing classifier weights.")
        num_pools, num_families = checkpoint_dims
        self.expected_feature_dim = _checkpoint_feature_dim(head_state)
        if len(self.pool_by_index) < num_pools or len(self.family_by_index) < num_families:
            raise RuntimeError(
                "CARA registry resolver is smaller than the trained attribution head. "
                f"resolver pools/families={len(self.pool_by_index)}/{len(self.family_by_index)}, "
                f"checkpoint pools/families={num_pools}/{num_families}."
            )
        self.head_state = head_state
        self.tapper = CaraHiddenStateTapper(model)
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
        device = features.device
        self.head.to(device)
        with torch.no_grad():
            self.head(features[:1])
        missing, unexpected = self.head.load_state_dict(self.head_state, strict=False)
        critical_missing = [key for key in missing if key.endswith(".weight") or key.endswith(".bias")]
        if critical_missing or unexpected:
            raise RuntimeError(
                "CARA attribution head checkpoint did not match the extractor head. "
                f"missing={critical_missing[:8]}, unexpected={list(unexpected)[:8]}"
            )
        self.head.eval()
        self.initialized = True

    def _pooled_generation_features(self, *, batch_size: int) -> torch.Tensor:
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
                continue
            if rows > batch_size and rows % batch_size == 0:
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
                continue
            if rows > batch_size:
                usable.append(feature[:batch_size])
                adjustments.append(
                    {
                        "tap_index": index,
                        "mode": "trim_to_requested_batch",
                        "shape": [rows, width],
                    }
                )
        self.last_feature_alignment = {
            "requested_batch_size": int(batch_size),
            "observed_feature_shapes": observed_shapes[:16],
            "usable_feature_count": len(usable),
            "adjustments": adjustments[:16],
            "policy": "Exact batch-aligned taps are used directly; duplicated generation branches such as CFG are averaged per requested sample.",
        }
        if not usable:
            raise RuntimeError(
                "No CARA hidden-state tap produced a generation-aligned feature tensor. "
                f"requested_batch_size={batch_size}; observed_feature_shapes={observed_shapes[:16]}"
            )
        features = torch.cat(usable, dim=-1)
        actual_dim = int(features.shape[-1])
        expected_dim = int(getattr(self, "expected_feature_dim", None) or actual_dim)
        feature_dim_mode = "exact"
        if actual_dim < expected_dim:
            features = F.pad(features, (0, expected_dim - actual_dim))
            feature_dim_mode = "right_zero_pad_to_checkpoint_width"
        elif actual_dim > expected_dim:
            features = features[:, :expected_dim]
            feature_dim_mode = "right_trim_to_checkpoint_width"
        self.last_feature_alignment = {
            **self.last_feature_alignment,
            "actual_feature_dim": actual_dim,
            "expected_feature_dim": expected_dim,
            "feature_dim_mode": feature_dim_mode,
        }
        return features

    def predict_latest(self, *, batch_size: int = 1) -> dict[str, Any]:
        features = self._pooled_generation_features(batch_size=batch_size)
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
        topk = [
            {
                "cara_pool_index": int(index),
                "cara_pool_id": self.pool_by_index.get(int(index)),
                "confidence": float(value),
                "registry_valid": int(index) in self.pool_by_index,
            }
            for index, value in zip(top_indices[0].detach().cpu().tolist(), top_values[0].detach().cpu().tolist())
        ]
        return {
            "status": "scored",
            "source": "stable_audio_dit_generation_hidden_state_tap",
            "cara_pool_index": pool_index,
            "cara_pool_id": self.pool_by_index.get(pool_index),
            "cara_pool_family_index": family_index,
            "cara_pool_family": self.family_by_index.get(family_index),
            "confidence": float(pool_confidence[0].detach().cpu()),
            "family_confidence": float(family_confidence[0].detach().cpu()),
            "entropy": float(entropy[0].detach().cpu()),
            "top_k": topk,
            "registry_valid": pool_index in self.pool_by_index,
            "feature_alignment": self.last_feature_alignment,
        }


def _needs_native_prediction(row: dict[str, Any]) -> bool:
    if str(row.get("model_id") or "") not in {DIFFUSION_MODEL_ID, CONTEXT_DIFFUSION_MODEL_ID}:
        return False
    return _normalise_prediction(row.get("native_cara_prediction")) is None


def _write_native_predictions(
    *,
    rows: list[dict[str, Any]],
    generated_audio_dir: Path,
    output_dir: Path,
    trained_model_data: Path,
    context_trained_model_data: Path | None,
    base_checkpoint: str,
    generation_steps: int,
    cfg_scale: float,
    max_native_predictions: int,
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [row for row in rows if _needs_native_prediction(row)]
    if max_native_predictions > 0:
        candidates = candidates[:max_native_predictions]
    if not candidates:
        return rows, {"status": "not_needed", "count": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("Native Stable Audio CARA extraction is GPU-only; CUDA is not available.")

    updated_by_key: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    tap_reports: dict[str, Any] = {}
    registry_hashes: dict[str, Any] = {}
    completed = 0
    for model_id in [DIFFUSION_MODEL_ID, CONTEXT_DIFFUSION_MODEL_ID]:
        model_candidates = [row for row in candidates if str(row.get("model_id") or "") == model_id]
        if not model_candidates:
            continue
        selected_trained_model_data = (
            context_trained_model_data
            if model_id == CONTEXT_DIFFUSION_MODEL_ID and context_trained_model_data is not None
            else trained_model_data
        )
        resolver = _load_registry_resolver(selected_trained_model_data, generated_audio_dir, rows)
        delta_payload = _load_trainable_delta(selected_trained_model_data)
        head_state = _extract_head_state(delta_payload)
        model, model_config, model_report = _load_stable_audio_model(
            model_id=model_id,
            base_checkpoint=base_checkpoint,
            trained_model_data=trained_model_data,
            context_trained_model_data=context_trained_model_data,
            resolver=resolver,
            report=report,
        )
        extractor = NativeCaraHiddenStateExtractor(model, resolver, head_state)
        try:
            for row in model_candidates:
                completed += 1
                extractor.clear()
                seed = int(row.get("seed") or 0)
                prompt = str(row.get("prompt") or "")
                structured_conditioning = _structured_conditioning_from_prompt_row(row, model_id=model_id)
                try:
                    _generate_one(
                        model=model,
                        model_config=model_config,
                        prompt=prompt,
                        conditioning_metadata=structured_conditioning,
                        seed=seed,
                        steps=int(row.get("generation_steps") or generation_steps),
                        cfg_scale=float(row.get("cfg_scale") or cfg_scale),
                    )
                    prediction = extractor.predict_latest(batch_size=1)
                except Exception as exc:
                    prediction = {
                        "status": "exception",
                        "source": "stable_audio_dit_generation_hidden_state_tap",
                        "error": repr(exc),
                        "cara_pool_id": None,
                        "cara_pool_index": None,
                        "cara_pool_family": None,
                        "cara_pool_family_index": None,
                        "confidence": None,
                        "family_confidence": None,
                        "registry_valid": False,
                        "top_k": [],
                        "feature_alignment": getattr(extractor, "last_feature_alignment", {}),
                        "policy": {"copied_from_expected": False},
                    }
                    report.setdefault("native_prediction_errors", []).append(
                        {
                            "index": completed,
                            "model_id": row.get("model_id"),
                            "suite_id": row.get("suite_id"),
                            "prompt_id": row.get("prompt_id"),
                            "seed": row.get("seed"),
                            "error": repr(exc),
                            "feature_alignment": getattr(extractor, "last_feature_alignment", {}),
                        }
                    )
                updated = dict(row)
                updated["native_cara_prediction"] = prediction
                updated["native_cara_prediction_generated_at"] = _utc_now()
                updated["native_cara_prediction_policy"] = {
                    "no_expected_label_copying": True,
                    "source_audio_path": row.get("audio_path"),
                    "source_prompt_replay": True,
                    "trained_model_data": str(selected_trained_model_data),
                    "structured_conditioning": structured_conditioning,
                    "structured_conditioning_supplied": bool(structured_conditioning),
                }
                key = (row.get("model_id"), row.get("suite_id"), row.get("prompt_id"), row.get("seed"))
                updated_by_key[key] = updated
                prediction_rows.append(
                    {
                        "model_id": row.get("model_id"),
                        "suite_id": row.get("suite_id"),
                        "prompt_id": row.get("prompt_id"),
                        "seed": row.get("seed"),
                        "audio_path": row.get("audio_path"),
                        "expected": row.get("expected"),
                        "native_cara_prediction": prediction,
                    }
                )
                report["native_predictions_completed"] = completed
                report["native_predictions_total"] = len(candidates)
        finally:
            extractor.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model_reports[model_id] = model_report
        tap_reports[model_id] = extractor.tap_report
        registry_hashes[model_id] = resolver.get("registry_hash")

    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("model_id"), row.get("suite_id"), row.get("prompt_id"), row.get("seed"))
        updated_rows.append(updated_by_key.get(key, row))
    _write_jsonl(output_dir / "native_predictions.jsonl", prediction_rows)
    _write_jsonl(output_dir / "scored_generation_manifest.jsonl", updated_rows)
    return updated_rows, {
        "status": "scored",
        "count": len(prediction_rows),
        "model_loads": model_reports,
        "tap_reports": tap_reports,
        "registry_hashes": registry_hashes,
        "prediction_file": "native_predictions.jsonl",
        "scored_generation_manifest": "scored_generation_manifest.jsonl",
    }


def _pool_from_prediction(prediction: dict[str, Any], maps: dict[str, Any]) -> str | None:
    if prediction.get("cara_pool_id") not in (None, ""):
        return str(prediction.get("cara_pool_id"))
    if prediction.get("cara_pool_index") not in (None, ""):
        return maps["pool_by_index"].get(str(int(prediction.get("cara_pool_index"))))
    return None


def _family_from_prediction(prediction: dict[str, Any], maps: dict[str, Any], pool_id: str | None) -> str | None:
    if prediction.get("cara_pool_family") not in (None, ""):
        return str(prediction.get("cara_pool_family"))
    if prediction.get("cara_pool_family_index") not in (None, ""):
        return maps["family_by_index"].get(str(int(prediction.get("cara_pool_family_index"))))
    if pool_id:
        return maps["pool_to_family"].get(pool_id)
    return None


def _score_prediction(row: dict[str, Any], prediction: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    decision = resolve_prediction(prediction, expected, resolver)
    pool_id = decision.resolved_pool_id or decision.predicted_pool_id
    family = decision.resolved_family or decision.predicted_family
    exact = decision.tier == "exact_pool" and decision.pool_exact_correct
    repairable = decision.tier == "repairable_pool" and decision.pool_repaired_correct
    family_match = decision.family_correct
    unattributable = decision.tier == "unattributable"
    confidence = prediction.get("confidence")
    if confidence in (None, ""):
        confidence = prediction.get("pool_confidence")
    try:
        confidence_value = float(confidence) if confidence not in (None, "") else None
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "model_id": row.get("model_id"),
        "suite_id": row.get("suite_id"),
        "prompt_id": row.get("prompt_id"),
        "expected_pool_id": decision.expected_pool_id,
        "expected_family": decision.expected_family,
        "predicted_pool_id": pool_id,
        "predicted_family": family,
        "resolved_pool_id": decision.resolved_pool_id,
        "resolved_family": decision.resolved_family,
        "registry_valid": decision.registry_valid,
        "exact": exact,
        "repairable": repairable,
        "family_match": family_match,
        "unattributable": unattributable,
        "tier": decision.tier,
        "repair_method": decision.repair_method,
        "repair_distance": decision.repair_distance,
        "confidence": confidence_value,
        "top_k": prediction.get("top_k") or [],
        "prediction_status": prediction.get("status"),
        "prediction_source": prediction.get("source") or prediction.get("adapter"),
        "prediction_error": prediction.get("error"),
        "feature_alignment": prediction.get("feature_alignment"),
        "audio_path": row.get("audio_path"),
    }


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
        avg_acc = sum(1.0 for row in bucket if row["exact"]) / len(bucket)
        value += len(bucket) / total * abs(avg_conf - avg_acc)
    return value


def _tier_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    tiers = ["exact_pool", "repairable_pool", "family_or_genre", "unattributable"]
    return {tier: sum(1 for row in predictions if row.get("tier") == tier) for tier in tiers}


def _correct_tier_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"exact_pool": 0, "repairable_pool": 0, "family_or_genre": 0, "unattributable": 0}
    for row in predictions:
        if row.get("exact"):
            counts["exact_pool"] += 1
        elif row.get("repairable"):
            counts["repairable_pool"] += 1
        elif row.get("family_match"):
            counts["family_or_genre"] += 1
        else:
            counts["unattributable"] += 1
    return counts


def _repair_method_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in predictions:
        key = str(row.get("repair_method") or "none")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _prediction_status_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in predictions:
        status = str(row.get("prediction_status") or "predicted").strip().lower()
        counts[status or "predicted"] += 1
    return dict(sorted(counts.items()))


def _prediction_error_count(predictions: list[dict[str, Any]]) -> int:
    error_statuses = {"exception", "failed", "error"}
    return sum(
        1
        for row in predictions
        if str(row.get("prediction_status") or "").strip().lower() in error_statuses
    )


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


def _prediction_examples(predictions: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in predictions[:limit]:
        examples.append(
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
        )
    return examples


def _lane_metrics(rows: list[dict[str, Any]], lane: str, resolver: dict[str, Any]) -> dict[str, Any]:
    labelled = [row for row in rows if (row.get("expected") or {}).get("cara_pool_id")]
    predictions: list[dict[str, Any]] = []
    for row in labelled:
        field = "native_cara_prediction" if lane == "native" else "external_probe_prediction"
        prediction = _normalise_prediction(row.get(field))
        if prediction is None:
            continue
        predictions.append(_score_prediction(row, prediction, resolver))
    if not labelled:
        return {"status": "no_labelled_rows", "count": 0, "labelled_count": 0}
    if not predictions:
        return {
            "status": "missing_predictions",
            "count": 0,
            "labelled_count": len(labelled),
            "reason": f"No real {lane} prediction fields were present in generation_manifest.jsonl.",
        }
    denominator = len(predictions)
    status_counts = _prediction_status_counts(predictions)
    error_count = _prediction_error_count(predictions)
    if error_count >= denominator:
        return {
            "status": "extractor_failed",
            "count": denominator,
            "labelled_count": len(labelled),
            "reason": (
                f"All {lane} prediction attempts failed inside the extractor/probe. "
                "Do not interpret this lane as zero-percent attribution."
            ),
            "prediction_status_counts": status_counts,
            "exception_rate": 1.0,
            "error_examples": _prediction_error_examples(predictions),
            "prediction_examples": _prediction_examples(predictions),
            "prediction_rows": predictions,
        }
    peer_metrics = summarize_prediction_rows(predictions)
    repairability = aggregate_repairability(predictions)
    lane_status = "scored_with_extractor_errors" if error_count else "scored"
    reason = (
        f"{error_count} of {denominator} {lane} prediction attempts failed inside the extractor/probe."
        if error_count
        else None
    )
    return {
        "status": lane_status,
        "count": denominator,
        "labelled_count": len(labelled),
        "reason": reason,
        "exact_pool_top1": peer_metrics.get("exact_pool_top1"),
        "exact_pool_top3": peer_metrics.get("exact_pool_top3"),
        "balanced_accuracy": peer_metrics.get("balanced_accuracy"),
        "macro_f1": peer_metrics.get("macro_f1"),
        "family_accuracy": peer_metrics.get("family_accuracy"),
        "brier": peer_metrics.get("brier"),
        "repaired_rate": peer_metrics.get("repaired_rate"),
        "degraded_rate": peer_metrics.get("degraded_rate"),
        "exception_rate": error_count / denominator,
        "pool_exact_accuracy": sum(1 for row in predictions if row["exact"]) / denominator,
        "pool_repaired_accuracy": repairability.get("pool_repaired_accuracy"),
        "pool_recovered_accuracy": repairability.get("pool_recovered_accuracy"),
        "family_or_genre_accuracy": repairability.get("family_or_genre_accuracy"),
        "unattributable_rate": repairability.get("unattributable_rate"),
        "registry_valid_rate": sum(1 for row in predictions if row["registry_valid"]) / denominator,
        "ece": peer_metrics.get("ece") if peer_metrics.get("ece") is not None else _ece(predictions),
        "repairability": repairability,
        "tier_counts": repairability.get("correct_tier_counts") or _correct_tier_counts(predictions),
        "resolution_tier_counts": _tier_counts(predictions),
        "repair_method_counts": repairability.get("repair_method_counts") or _repair_method_counts(predictions),
        "prediction_status_counts": status_counts,
        "error_examples": _prediction_error_examples(predictions),
        "prediction_examples": _prediction_examples(predictions),
        "prediction_rows": predictions,
    }


def _value(lane: dict[str, Any], key: str) -> float | None:
    value = lane.get(key)
    return float(value) if isinstance(value, (int, float)) and not math.isnan(float(value)) else None


def _benchmark_rows(
    diffusion_native: dict[str, Any],
    diffusion_probe: dict[str, Any],
    context_native: dict[str, Any],
    context_probe: dict[str, Any],
) -> list[dict[str, Any]]:
    specs = [
        ("pool_exact_accuracy", "Exact pool accuracy", True),
        ("pool_repaired_accuracy", "Repairable pool accuracy", True),
        ("family_or_genre_accuracy", "Family / genre fallback accuracy", True),
        ("unattributable_rate", "Unattributable rate", False),
        ("registry_valid_rate", "Registry-valid rate", True),
        ("ece", "Calibration / ECE", False),
    ]
    rows = []
    for key, metric, higher in specs:
        rows.append(
            {
                "id": key,
                "metric": metric,
                "higher_is_better": higher,
                "diffusion_native": _value(diffusion_native, key),
                "diffusion_external_probe": _value(diffusion_probe, key),
                "context_diffusion_native": _value(context_native, key),
                "context_diffusion_external_probe": _value(context_probe, key),
                "ar_native": None,
                "hybrid_native": None,
                "diffusion_native_status": diffusion_native.get("status"),
                "diffusion_native_reason": diffusion_native.get("reason"),
                "diffusion_external_probe_status": diffusion_probe.get("status"),
                "diffusion_external_probe_reason": diffusion_probe.get("reason"),
                "context_diffusion_native_status": context_native.get("status"),
                "context_diffusion_native_reason": context_native.get("reason"),
                "context_diffusion_external_probe_status": context_probe.get("status"),
                "context_diffusion_external_probe_reason": context_probe.get("reason"),
                "status": "scored_if_values_present",
            }
        )
    return rows


def _repairability_matrix(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tiers = [
        ("exact_pool", "Exact pool"),
        ("repairable_pool", "Repairable pool"),
        ("family_or_genre", "Family / genre fallback"),
        ("unattributable", "Unattributable"),
    ]
    lane_order = [
        "diffusion_native",
        "diffusion_external_probe",
        "context_diffusion_native",
        "context_diffusion_external_probe",
    ]
    rows: list[dict[str, Any]] = []
    for tier_id, label in tiers:
        row: dict[str, Any] = {"tier": tier_id, "label": label}
        for lane_id in lane_order:
            lane = lanes.get(lane_id) or {}
            repairability = lane.get("repairability") if isinstance(lane.get("repairability"), dict) else {}
            tier_counts = lane.get("tier_counts") if isinstance(lane.get("tier_counts"), dict) else {}
            tier_rates = repairability.get("correct_tier_rates") if isinstance(repairability.get("correct_tier_rates"), dict) else {}
            if not tier_rates:
                tier_rates = repairability.get("tier_rates") if isinstance(repairability.get("tier_rates"), dict) else {}
            row[lane_id] = {
                "status": lane.get("status"),
                "count": tier_counts.get(tier_id),
                "rate": tier_rates.get(tier_id),
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)
    return {
        "format": "cara_repairability_matrix_v1",
        "tiers": [tier_id for tier_id, _label in tiers],
        "lanes": lane_order,
        "rows": rows,
    }


def _repair_method_matrix(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lane_order = [
        "diffusion_native",
        "diffusion_external_probe",
        "context_diffusion_native",
        "context_diffusion_external_probe",
    ]
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
        for lane_id in lane_order:
            lane = lanes.get(lane_id) or {}
            counts = lane.get("repair_method_counts") if isinstance(lane.get("repair_method_counts"), dict) else {}
            count = int(counts.get(method) or 0)
            total = int(lane.get("count") or 0)
            row[lane_id] = {
                "status": lane.get("status"),
                "count": count,
                "rate": count / total if total else None,
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)
    return {"format": "cara_repair_method_matrix_v1", "lanes": lane_order, "rows": rows}


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    generated_audio_dir = Path(args.generated_audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["runtime_dirs"] = _configure_disk_safe_runtime_dirs(output_dir)
    manifest_path = generated_audio_dir / "generation_manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing generation manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Generation manifest has no rows: {manifest_path}")
    source_row_count = len(rows)
    selected_model_ids = _parse_model_ids(args.model_ids)
    if selected_model_ids:
        selected_set = set(selected_model_ids)
        rows = [row for row in rows if str(row.get("model_id") or "") in selected_set]
        if not rows:
            raise RuntimeError(
                "No generation manifest rows matched selected model_ids="
                f"{','.join(selected_model_ids)} in {manifest_path}."
            )
    report["source_generation_row_count"] = source_row_count
    report["selected_model_ids"] = selected_model_ids or sorted({str(row.get("model_id") or "unknown") for row in rows})
    report["selected_generation_row_count"] = len(rows)
    native_extraction: dict[str, Any] = {"status": "disabled"}
    if parse_bool(args.native_extractor):
        rows, native_extraction = _write_native_predictions(
            rows=rows,
            generated_audio_dir=generated_audio_dir,
            output_dir=output_dir,
            trained_model_data=Path(args.trained_model_data),
            context_trained_model_data=Path(args.context_trained_model_data)
            if str(args.context_trained_model_data or "").strip()
            else None,
            base_checkpoint=args.base_checkpoint,
            generation_steps=int(args.generation_steps),
            cfg_scale=float(args.cfg_scale),
            max_native_predictions=int(args.max_native_predictions),
            report=report,
        )
    resolver = _load_registry_resolver(Path(args.trained_model_data), generated_audio_dir, rows)
    _write_json(output_dir / "cara_registry_resolver.json", resolver)

    by_model = {model_id: [row for row in rows if str(row.get("model_id") or "") == model_id] for model_id in sorted({str(row.get("model_id") or "unknown") for row in rows})}
    base_rows = by_model.get(BASE_MODEL_ID, [])
    diffusion_rows = by_model.get(DIFFUSION_MODEL_ID, [])
    context_rows = by_model.get(CONTEXT_DIFFUSION_MODEL_ID, [])
    heldout_training_evaluation = {
        DIFFUSION_MODEL_ID: _load_training_heldout_evaluation(Path(args.trained_model_data)),
    }
    if str(args.context_trained_model_data or "").strip():
        heldout_training_evaluation[CONTEXT_DIFFUSION_MODEL_ID] = _load_training_heldout_evaluation(
            Path(args.context_trained_model_data)
        )
    base_probe = _lane_metrics(base_rows, "external_probe", resolver)
    diffusion_native = _lane_metrics(diffusion_rows, "native", resolver)
    diffusion_probe = _lane_metrics(diffusion_rows, "external_probe", resolver)
    context_native = _lane_metrics(context_rows, "native", resolver)
    context_probe = _lane_metrics(context_rows, "external_probe", resolver)
    benchmark_rows = _benchmark_rows(diffusion_native, diffusion_probe, context_native, context_probe)

    lanes = {
        "diffusion_native": {
            "model_id": DIFFUSION_MODEL_ID,
            "variant": "cara_strong",
            "evidence_lane": "native",
            "heldout_training_evaluation": heldout_training_evaluation.get(DIFFUSION_MODEL_ID),
            **{key: value for key, value in diffusion_native.items() if key != "prediction_rows"},
        },
        "diffusion_external_probe": {
            "model_id": DIFFUSION_MODEL_ID,
            "variant": "cara_strong",
            "evidence_lane": "external_probe",
            **{key: value for key, value in diffusion_probe.items() if key != "prediction_rows"},
        },
        "context_diffusion_native": {
            "model_id": CONTEXT_DIFFUSION_MODEL_ID,
            "variant": "cara_strong_context_conditioned",
            "evidence_lane": "native",
            "heldout_training_evaluation": heldout_training_evaluation.get(CONTEXT_DIFFUSION_MODEL_ID),
            **{key: value for key, value in context_native.items() if key != "prediction_rows"},
        },
        "context_diffusion_external_probe": {
            "model_id": CONTEXT_DIFFUSION_MODEL_ID,
            "variant": "cara_strong_context_conditioned",
            "evidence_lane": "external_probe",
            **{key: value for key, value in context_probe.items() if key != "prediction_rows"},
        },
        "base_external_probe": {
            "model_id": BASE_MODEL_ID,
            "variant": "released_base",
            "evidence_lane": "external_probe",
            **{key: value for key, value in base_probe.items() if key != "prediction_rows"},
        },
        "base_native": {
            "model_id": BASE_MODEL_ID,
            "variant": "released_base",
            "evidence_lane": "native",
            "status": "not_applicable",
            "reason": "Base checkpoint has no native CARA-id output channel.",
        },
    }
    metrics = {
        "format": "cara_benchmark_matrix_metrics_v1",
        "created_at": _utc_now(),
        "source_generation_manifest": str(manifest_path),
        "scored_generation_manifest": "scored_generation_manifest.jsonl" if (output_dir / "scored_generation_manifest.jsonl").exists() else None,
        "source_audio_output_dir": str(generated_audio_dir),
        "selected_model_ids": selected_model_ids,
        "source_generation_row_count": source_row_count,
        "generated_audio_count": len(rows),
        "native_extraction": native_extraction,
        "heldout_training_evaluation": heldout_training_evaluation,
        "native_prediction_errors": (report.get("native_prediction_errors") or [])[:25],
        "by_model": dict(sorted(Counter(str(row.get("model_id") or "unknown") for row in rows).items())),
        "by_suite": dict(sorted(Counter(str(row.get("suite_id") or "unknown") for row in rows).items())),
        "lanes": lanes,
        "repairability_matrix": _repairability_matrix(lanes),
        "repair_method_matrix": _repair_method_matrix(lanes),
        "prediction_examples": {
            lane_id: lane.get("prediction_examples") or []
            for lane_id, lane in lanes.items()
            if isinstance(lane, dict)
        },
        "benchmark_rows": benchmark_rows,
        "scoring_policy": {
            "no_expected_label_copying": True,
            "native_extractor": "Diffusion and Context Diffusion native predictions are replayed from Stable Audio generation hidden-state taps when absent.",
            "missing_prediction_behavior": "Metric cells remain pending unless generated or scored manifests contain real predicted CARA fields.",
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    }
    _write_json(output_dir / "metrics_latest.json", metrics)
    _write_json(output_dir / "benchmark_attribution_scoring_report.json", {**report, "metrics": metrics, "status": "passed"})
    _write_jsonl(
        output_dir / "prediction_rows.jsonl",
        diffusion_native.get("prediction_rows", [])
        + diffusion_probe.get("prediction_rows", [])
        + context_native.get("prediction_rows", [])
        + context_probe.get("prediction_rows", [])
        + base_probe.get("prediction_rows", []),
    )
    report.update(
        {
            "status": "passed",
            "stage": "completed",
            "generated_audio_count": len(rows),
            "native_extraction": native_extraction,
            "metrics_available": any(
                row.get("diffusion_native") is not None
                or row.get("context_diffusion_native") is not None
                or row.get("diffusion_external_probe") is not None
                or row.get("context_diffusion_external_probe") is not None
                for row in benchmark_rows
            ),
            "lane_statuses": metrics["lanes"],
            "output_metrics": "metrics_latest.json",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_audio_dir", required=True)
    parser.add_argument("--trained_model_data", required=True)
    parser.add_argument("--context_trained_model_data", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_checkpoint", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--native_extractor", default="true")
    parser.add_argument("--max_native_predictions", type=int, default=0)
    parser.add_argument("--model_ids", default="")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "format": "cara_benchmark_attribution_scoring_report_v1",
        "test_name": "16_benchmark_testing_stable_audio_score",
        "stage": "start",
        "status": "failed",
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
        "dry_run": parse_bool(args.dry_run),
        "warnings": [],
    }
    try:
        if report["dry_run"]:
            report["status"] = "planned"
            report["stage"] = "dry_run"
        else:
            run(args, report)
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = __import__("traceback").format_exc()
        raise
    finally:
        metadata = base_metadata(
            test_name="16_benchmark_testing_stable_audio_score",
            compute="gpu-smoke-h100",
            environment="env-stable-audio-tools:8",
            dashboard_triggered=report.get("dashboard_triggered", False),
            report=report,
            model_family="stable_audio_open_small",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_stable_audio_score_report.json")
    return 0 if report.get("status") in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
