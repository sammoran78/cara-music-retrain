from __future__ import annotations

import argparse
import json
import math
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmark_testing_musicgen_audio import (
    BASE_MODEL_ID,
    CARA_MODEL_ID,
    _prepare_musicgen_hf_auth,
    _apply_delta_to_lm,
    _generate_one,
    _load_delta,
)
try:
    from evaluation.cara_repairability import aggregate_repairability, resolve_prediction
except ModuleNotFoundError:
    from cara_repairability import aggregate_repairability, resolve_prediction
from musicgen_cara_tokens import decode_cara_suffix
from musicgen_lm_cara_trainer import CARASuffixHead, _first_tensor, _pool_hidden
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
            "exception_rate": sum(1 for row in rows if row.get("tier") == "unattributable") / len(rows),
        }


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


def _resolver_maps(resolver: dict[str, Any]) -> dict[str, Any]:
    pool_by_index = {str(key): value for key, value in (resolver.get("pool_by_index") or {}).items()}
    family_by_index = {str(key): value for key, value in (resolver.get("family_by_index") or {}).items()}
    pool_to_family = resolver.get("pool_to_family_name") or {}
    if not pool_by_index and resolver.get("pool_index"):
        pool_by_index = {str(index): pool_id for pool_id, index in (resolver.get("pool_index") or {}).items()}
    if not family_by_index and resolver.get("family_index"):
        family_by_index = {str(index): family for family, index in (resolver.get("family_index") or {}).items()}
    if not pool_to_family:
        pool_to_family_index = {str(key): value for key, value in (resolver.get("pool_to_family_index") or {}).items()}
        pool_to_family = {
            pool_id: family_by_index.get(str(pool_to_family_index.get(str(pool_index))))
            for pool_index, pool_id in pool_by_index.items()
            if pool_to_family_index.get(str(pool_index)) is not None
        }
    return {
        "pool_by_index": pool_by_index,
        "family_by_index": family_by_index,
        "pool_to_family": {str(key): value for key, value in pool_to_family.items() if value},
    }


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
        "audio_path": row.get("audio_path"),
    }


def _find_file(root: Path, name: str) -> Path:
    candidates = [root / name]
    candidates.extend(root.rglob(name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Missing {name} under {root}")


def _musicgen_native_adapter(decoded: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    policy = decoded.get("policy") if isinstance(decoded.get("policy"), dict) else {}
    if policy.get("copied_from_expected") is True:
        raise ValueError("MusicGen native prediction illegally copied expected labels.")
    maps = _resolver_maps(resolver)
    pool_id = decoded.get("cara_pool_id")
    pool_index = decoded.get("cara_pool_index")
    if pool_id in (None, "") and pool_index not in (None, ""):
        pool_id = maps["pool_by_index"].get(str(int(pool_index)))
    family_index = decoded.get("cara_pool_family_index")
    family = decoded.get("cara_pool_family")
    if family in (None, "") and family_index not in (None, ""):
        family = maps["family_by_index"].get(str(int(family_index)))
    registry_valid = bool(decoded.get("registry_valid")) and bool(pool_id)
    has_attribution_signal = any(value not in (None, "") for value in [pool_id, pool_index, family, family_index])
    return {
        "adapter": "musicgen_lm_suffix",
        "model_family": "musicgen",
        "status": "scored" if has_attribution_signal else "exception",
        "cara_pool_id": pool_id,
        "cara_pool_index": int(pool_index) if pool_index not in (None, "") else None,
        "cara_pool_family": family,
        "cara_pool_family_index": int(family_index) if family_index not in (None, "") else None,
        "confidence": decoded.get("confidence"),
        "family_confidence": decoded.get("family_confidence"),
        "registry_valid": registry_valid,
        "checksum_valid": bool(decoded.get("checksum_valid")),
        "hierarchical_valid": bool(decoded.get("hierarchical_valid")),
        "symbols": decoded.get("symbols"),
        "source": "musicgen_lm_suffix",
    }


def _suffix_head_state(delta_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = delta_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        return {}
    head_state: dict[str, torch.Tensor] = {}
    prefix = "cara_suffix_head."
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and str(key).startswith(prefix):
            head_state[str(key)[len(prefix) :]] = value.detach().cpu()
    return head_state


class MusicGenSuffixExtractor:
    def __init__(
        self,
        *,
        base_checkpoint: str,
        trained_model_data: Path,
        resolver: dict[str, Any],
        suffix_vocab: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MusicGen native suffix scoring is GPU-only; CUDA is not available.")
        _prepare_musicgen_hf_auth(report, base_checkpoint=base_checkpoint)
        from audiocraft.models import MusicGen

        self.model = MusicGen.get_pretrained(base_checkpoint, device="cuda")
        self.delta_report = _apply_delta_to_lm(self.model.lm, trained_model_data)
        _delta_path, delta_payload = _load_delta(trained_model_data)
        head_state = _suffix_head_state(delta_payload)
        if not head_state:
            raise RuntimeError("MusicGen delta did not contain cara_suffix_head parameters.")
        hidden_dim = int(getattr(self.model.lm, "dim", 0) or 0)
        if hidden_dim <= 0:
            raise RuntimeError("AudioCraft LM does not expose lm.dim for CARA suffix scoring.")
        self.head = CARASuffixHead(hidden_dim, int(suffix_vocab["size"])).to("cuda")
        missing, unexpected = self.head.load_state_dict(head_state, strict=False)
        critical_missing = [key for key in missing if key.endswith(".weight") or key.endswith(".bias")]
        if critical_missing or unexpected:
            raise RuntimeError(
                "MusicGen CARA suffix-head checkpoint did not match the scorer head. "
                f"missing={critical_missing[:8]}, unexpected={list(unexpected)[:8]}"
            )
        self.head.eval()
        self.resolver = resolver
        self.suffix_vocab = suffix_vocab
        self.features: list[torch.Tensor] = []
        self.hook = self.model.lm.transformer.register_forward_hook(self._capture)

    def close(self) -> None:
        self.hook.remove()

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is not None:
            self.features.append(tensor.detach())

    def predict(
        self,
        *,
        prompt: str,
        seed: int,
        suffix_len: int,
        duration_seconds: float,
        top_k: int,
        cfg_coef: float,
    ) -> dict[str, Any]:
        self.features.clear()
        _generate_one(
            self.model,
            prompt=prompt,
            seed=seed,
            duration_seconds=duration_seconds,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )
        if not self.features:
            raise RuntimeError("No MusicGen LM hidden state was captured during prompt replay.")
        with torch.no_grad():
            pooled = _pool_hidden(self.features[-1]).to("cuda")
            logits = self.head(pooled[:1], suffix_len)
            probs = F.softmax(logits, dim=-1)
            confidence_per_token, token_ids = probs.max(dim=-1)
            decoded = decode_cara_suffix(token_ids[0].detach().cpu().tolist(), self.suffix_vocab, self.resolver)
            confidence = float(confidence_per_token[0].mean().detach().cpu())
            decoded_prediction = {
                "source": "musicgen_lm_suffix_prompt_replay",
                "policy": {"copied_from_expected": False},
                "cara_pool_id": decoded.get("cara_pool_id"),
                "cara_pool_index": decoded.get("pool_index"),
                "cara_pool_family_index": decoded.get("family_index"),
                "confidence": confidence,
                "family_confidence": confidence,
                "registry_valid": bool(decoded.get("registry_valid")),
                "checksum_valid": bool(decoded.get("checksum_valid")),
                "hierarchical_valid": bool(decoded.get("hierarchical_valid")),
                "symbols": decoded.get("symbols"),
            }
            return _musicgen_native_adapter(decoded_prediction, self.resolver)


def _suffix_len(rows: list[dict[str, Any]], default: int) -> int:
    if default > 0:
        return default
    # CARA suffix format is fixed-width for current codewords; derive only the token count,
    # never a predicted label, from the manifest/resolver artifacts.
    lengths = []
    for row in rows:
        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        codeword = str(expected.get("cara_pool_codeword") or "")
        if codeword:
            lengths.append(1 + 1 + 1 + 1 + len(codeword) + 1 + 1 + 1)
    return max(lengths) if lengths else 21


def _lane_metrics(rows: list[dict[str, Any]], prediction_field: str, resolver: dict[str, Any]) -> dict[str, Any]:
    labelled = [row for row in rows if (row.get("expected") or {}).get("cara_pool_id")]
    scored_rows = []
    for row in labelled:
        prediction = _normalise_prediction(row.get(prediction_field))
        if prediction is None:
            continue
        scored_rows.append(_score_prediction(row, prediction, resolver))
    if not labelled:
        return {"status": "no_labelled_rows", "count": 0, "labelled_count": 0}
    if not scored_rows:
        return {
            "status": "missing_predictions",
            "count": 0,
            "labelled_count": len(labelled),
            "reason": f"No real {prediction_field} fields were present in generation_manifest.jsonl.",
        }
    summary = summarize_prediction_rows(scored_rows)
    repairability = aggregate_repairability(scored_rows)
    rates = {
        "pool_exact_accuracy": summary.get("exact_pool_top1"),
        "pool_repaired_accuracy": repairability.get("pool_repaired_accuracy"),
        "pool_recovered_accuracy": repairability.get("pool_recovered_accuracy"),
        "family_or_genre_accuracy": repairability.get("family_or_genre_accuracy"),
        "unattributable_rate": repairability.get("unattributable_rate"),
    }
    correct_tier_counts = repairability.get("correct_tier_counts")
    if not isinstance(correct_tier_counts, dict):
        correct_tier_counts = {
            "exact_pool": sum(1 for row in scored_rows if row.get("exact")),
            "repairable_pool": sum(1 for row in scored_rows if row.get("repairable")),
            "family_or_genre": sum(1 for row in scored_rows if row.get("family_match") and not row.get("exact") and not row.get("repairable")),
            "unattributable": sum(1 for row in scored_rows if not row.get("exact") and not row.get("repairable") and not row.get("family_match")),
        }
    return {
        "status": "scored",
        "count": len(scored_rows),
        "labelled_count": len(labelled),
        **summary,
        **rates,
        "repairability": repairability,
        "tier_counts": correct_tier_counts,
        "resolution_tier_counts": {
            tier: sum(1 for row in scored_rows if row.get("tier") == tier)
            for tier in ["exact_pool", "repairable_pool", "family_or_genre", "unattributable"]
        },
        "repair_method_counts": repairability.get("repair_method_counts"),
        "prediction_examples": [
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
            }
            for row in scored_rows[:5]
        ],
        "prediction_rows": scored_rows,
    }


def _value(lane: dict[str, Any], key: str) -> float | None:
    value = lane.get(key)
    return float(value) if isinstance(value, (int, float)) and not math.isnan(float(value)) else None


def _benchmark_rows(musicgen_native: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("exact_pool_top1", "Exact pool top-1", True),
        ("exact_pool_top3", "Exact pool top-3", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("macro_f1", "Macro-F1", True),
        ("family_accuracy", "Family accuracy", True),
        ("registry_valid_rate", "Registry-valid rate", True),
        ("ece", "Calibration / ECE", False),
        ("brier", "Brier score", False),
    ]
    return [
        {
            "id": key,
            "metric": metric,
            "higher_is_better": higher,
            "musicgen_native": _value(musicgen_native, key),
            "musicgen_native_status": musicgen_native.get("status"),
            "musicgen_native_reason": musicgen_native.get("reason"),
            "status": "scored_if_values_present",
        }
        for key, metric, higher in specs
    ]


def _repairability_matrix(lanes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tiers = [
        ("exact_pool", "Exact pool"),
        ("repairable_pool", "Repairable pool"),
        ("family_or_genre", "Family / genre fallback"),
        ("unattributable", "Unattributable"),
    ]
    lane_order = ["base_musicgen_external_probe", "musicgen_native", "musicgen_external_probe"]
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
    lane_order = ["base_musicgen_external_probe", "musicgen_native", "musicgen_external_probe"]
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
    trained_model_data = Path(args.trained_model_data)
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

    resolver = _read_json(_find_file(generated_audio_dir, "cara_registry_resolver.json"))
    suffix_vocab = _read_json(_find_file(generated_audio_dir, "cara_suffix_vocab.json"))
    target_rows = [row for row in rows if str(row.get("model_id") or "") == CARA_MODEL_ID]
    max_native = int(args.max_native_predictions)
    candidates = [
        row
        for row in target_rows
        if _normalise_prediction(row.get("native_cara_prediction")) is None
        or str((row.get("native_cara_prediction") or {}).get("status") or "").startswith("pending")
    ]
    if max_native > 0:
        candidates = candidates[:max_native]

    native_extraction: dict[str, Any] = {"status": "disabled"}
    if parse_bool(args.native_extractor) and candidates:
        extractor = MusicGenSuffixExtractor(
            base_checkpoint=str(args.base_checkpoint),
            trained_model_data=trained_model_data,
            resolver=resolver,
            suffix_vocab=suffix_vocab,
            report=report,
        )
        updated_by_key: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
        prediction_rows: list[dict[str, Any]] = []
        suffix_len = _suffix_len(rows, int(args.suffix_len))
        try:
            for index, row in enumerate(candidates, start=1):
                prediction = extractor.predict(
                    prompt=str(row.get("prompt") or ""),
                    seed=int(row.get("seed") or 0),
                    suffix_len=suffix_len,
                    duration_seconds=float(row.get("duration_seconds") or args.duration_seconds),
                    top_k=int(row.get("top_k") or args.top_k),
                    cfg_coef=float(row.get("cfg_coef") or args.cfg_coef),
                )
                updated = dict(row)
                updated["native_cara_prediction"] = prediction
                updated["native_cara_prediction_generated_at"] = _utc_now()
                updated["native_cara_prediction_policy"] = {
                    "no_expected_label_copying": True,
                    "source_prompt_replay": True,
                    "suffix_len": suffix_len,
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
                report["native_predictions_completed"] = index
                report["native_predictions_total"] = len(candidates)
                _write_json(
                    output_dir / "score_progress.json",
                    {
                        "status": "running",
                        "stage": "native_musicgen_prompt_replay",
                        "updated_at": _utc_now(),
                        "native_predictions_completed": index,
                        "native_predictions_total": len(candidates),
                        "latest_model_id": row.get("model_id"),
                        "latest_suite_id": row.get("suite_id"),
                        "latest_prompt_id": row.get("prompt_id"),
                        "latest_seed": row.get("seed"),
                    },
                )
        finally:
            native_extraction = {
                "status": "scored",
                "count": len(prediction_rows),
                "model_load": extractor.delta_report,
                "suffix_len": suffix_len,
                "prediction_file": "native_predictions.jsonl",
                "scored_generation_manifest": "scored_generation_manifest.jsonl",
            }
            extractor.close()
            del extractor
            torch.cuda.empty_cache()
        rows = [
            updated_by_key.get((row.get("model_id"), row.get("suite_id"), row.get("prompt_id"), row.get("seed")), row)
            for row in rows
        ]
        _write_jsonl(output_dir / "native_predictions.jsonl", prediction_rows)
        _write_jsonl(output_dir / "scored_generation_manifest.jsonl", rows)
    elif parse_bool(args.native_extractor):
        native_extraction = {"status": "not_needed", "count": 0}

    by_model = {model_id: [row for row in rows if str(row.get("model_id") or "") == model_id] for model_id in sorted({str(row.get("model_id") or "unknown") for row in rows})}
    musicgen_native = _lane_metrics(by_model.get(CARA_MODEL_ID, []), "native_cara_prediction", resolver)
    base_external_probe = _lane_metrics(by_model.get(BASE_MODEL_ID, []), "external_probe_prediction", resolver)
    musicgen_external_probe = _lane_metrics(by_model.get(CARA_MODEL_ID, []), "external_probe_prediction", resolver)
    lanes = {
        "base_musicgen_native": {
            "model_id": BASE_MODEL_ID,
            "variant": "released_base",
            "evidence_lane": "native",
            "status": "not_applicable",
            "reason": "Base MusicGen checkpoint has no native CARA suffix output channel.",
        },
        "base_musicgen_external_probe": {
            "model_id": BASE_MODEL_ID,
            "variant": "released_base",
            "evidence_lane": "external_probe",
            **{key: value for key, value in base_external_probe.items() if key != "prediction_rows"},
        },
        "musicgen_native": {
            "model_id": CARA_MODEL_ID,
            "variant": "cara_strong",
            "evidence_lane": "native",
            **{key: value for key, value in musicgen_native.items() if key != "prediction_rows"},
        },
        "musicgen_external_probe": {
            "model_id": CARA_MODEL_ID,
            "variant": "cara_strong",
            "evidence_lane": "external_probe",
            **{key: value for key, value in musicgen_external_probe.items() if key != "prediction_rows"},
        },
    }
    metrics = {
        "format": "cara_musicgen_benchmark_matrix_metrics_v1",
        "created_at": _utc_now(),
        "source_generation_manifest": str(manifest_path),
        "scored_generation_manifest": "scored_generation_manifest.jsonl" if (output_dir / "scored_generation_manifest.jsonl").exists() else None,
        "source_audio_output_dir": str(generated_audio_dir),
        "selected_model_ids": selected_model_ids,
        "source_generation_row_count": source_row_count,
        "generated_audio_count": len(rows),
        "native_extraction": native_extraction,
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
        "benchmark_rows": _benchmark_rows(lanes["musicgen_native"]),
        "scoring_policy": {
            "no_expected_label_copying": True,
            "native_extractor": "MusicGen native predictions are decoded from CARA suffix-head logits captured during prompt replay.",
            "missing_prediction_behavior": "Metric cells remain pending unless generated/scored manifests contain real predicted CARA fields.",
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    }
    _write_json(output_dir / "metrics_latest.json", metrics)
    _write_json(output_dir / "cara_registry_resolver.json", resolver)
    _write_jsonl(
        output_dir / "prediction_rows.jsonl",
        musicgen_native.get("prediction_rows", []) + base_external_probe.get("prediction_rows", []) + musicgen_external_probe.get("prediction_rows", []),
    )
    report.update(
        {
            "status": "passed",
            "stage": "completed",
            "generated_audio_count": len(rows),
            "native_extraction": native_extraction,
            "metrics_available": lanes["musicgen_native"].get("status") == "scored",
            "lane_statuses": lanes,
            "output_metrics": "metrics_latest.json",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_audio_dir", required=True)
    parser.add_argument("--trained_model_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_checkpoint", default="facebook/musicgen-small")
    parser.add_argument("--native_extractor", default="true")
    parser.add_argument("--max_native_predictions", type=int, default=0)
    parser.add_argument("--duration_seconds", type=float, default=12.0)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--cfg_coef", type=float, default=3.0)
    parser.add_argument("--suffix_len", type=int, default=21)
    parser.add_argument("--model_ids", default="")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "format": "cara_benchmark_musicgen_attribution_scoring_report_v1",
        "test_name": "18_benchmark_testing_musicgen_score",
        "stage": "start",
        "status": "failed",
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
        "dry_run": parse_bool(args.dry_run),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
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
            test_name="18_benchmark_testing_musicgen_score",
            compute="gpu-smoke-h100",
            environment="env-musicgen-audiocraft:3",
            dashboard_triggered=report.get("dashboard_triggered", False),
            report=report,
            model_family="musicgen",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_musicgen_score_report.json")
    return 0 if report.get("status") in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
