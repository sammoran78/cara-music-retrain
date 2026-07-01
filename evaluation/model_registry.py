from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.benchmark_spec import (
    comparison_cards,
    default_metric_rows,
    benchmark_spec,
    metric_definitions,
    model_lanes,
)


ROOT = Path(__file__).resolve().parents[1]
JOBS_LOG = ROOT / "registry" / "cara_strong" / "azure_training_jobs.jsonl"
POOL_REGISTRY = ROOT / "registry" / "cara_strong" / "pool_registry.locked.json"
LATEST_METRICS = ROOT / "evaluation" / "metrics_latest.json"
COMPARISON_LOG = ROOT / "evaluation_log.csv"


STABLE_AUDIO_FULL_JOB_NAME = "modest_arch_clgnkqrz4z"
STABLE_AUDIO_FULL_OUTPUT = (
    "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/"
    "stable_audio_full/cara-finetune-001-cara-strong-full-20260607-011523/"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _job_events() -> list[dict[str, Any]]:
    if not JOBS_LOG.exists():
        return []
    events: list[dict[str, Any]] = []
    with JOBS_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _latest_event_for(job_name: str) -> dict[str, Any] | None:
    for event in reversed(_job_events()):
        if event.get("job_name") == job_name:
            return event
    return None


def _registry_summary() -> dict[str, Any]:
    payload = _read_json(POOL_REGISTRY)
    if not payload:
        return {"pool_count": 0, "family_count": 0, "registry_hash": None}
    pool_index = payload.get("pool_index") or {}
    family_index = payload.get("family_index") or {}
    return {
        "pool_count": payload.get("pool_count") or len(pool_index),
        "family_count": payload.get("family_count") or len(family_index),
        "registry_hash": payload.get("registry_hash") or payload.get("hash"),
        "created_at": payload.get("created_at"),
    }


def model_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for lane in model_lanes():
        candidates.append(
            {
                "id": lane["model_id"],
                "model_id": lane["model_id"],
                "label": lane["label"],
                "family": lane["family"],
                "architecture": lane["architecture"],
                "role": lane["variant"],
                "variant": lane["variant"],
                "status": lane["status"],
                "native_cara_output": bool(lane["native_cara_output"]),
                "external_probe_comparable": lane["variant"] != "released_base" or lane["baseline_role"] == "released_base",
                "checkpoint": lane.get("checkpoint_uri"),
                "checkpoint_uri": lane.get("checkpoint_uri"),
                "output_path": lane.get("output_uri"),
                "output_uri": lane.get("output_uri"),
                "native_prediction_adapter": lane.get("native_prediction_adapter"),
                "generation_adapter": lane.get("generation_adapter"),
                "artifact_checks": lane.get("artifact_checks", []),
                "baseline_policy": "Native CARA-id metrics are N/A for released/base/no-CARA lanes; compare them through external probe and retrieval controls.",
                "latest_job": lane.get("latest_job"),
            }
        )
    return candidates


def baseline_comparison_policy() -> dict[str, Any]:
    return {
        "native_cara_metrics": "Only reported for models with a native CARA output channel: Diffusion CARA-Strong and MusicGen CARA-Strong.",
        "base_model": "The original baseline has no CARA-id channel, so native pool-id accuracy is marked N/A rather than zero.",
        "same_data_no_cara": "Same-data no-CARA fine-tunes are the primary fairness baseline where a completed lane exists.",
        "retrieval_baseline": "Post-hoc embedding retrieval is a floor that every CARA lane must beat under codeword-withheld scoring.",
        "external_probe_metrics": "All generated-audio lanes can be compared through the same external probe/retrieval evidence lane.",
        "statistical_controls": "Prompt-only, no-tag, tag-present, shuffled-label, random/prior, and retrieval controls are reported beside model metrics.",
        "headline_rule": "Headline claims use codeword-withheld rows and separate native-output accuracy from external-probe attribution accuracy.",
    }


def benchmark_rows() -> list[dict[str, Any]]:
    descriptions = {
        "exact_pool_top1": "Predicted CARA pool-id top-1 exactly matches the known held-out pool.",
        "exact_pool_top3": "Expected CARA pool appears in the top-3 predicted pool distribution.",
        "balanced_accuracy": "Per-pool balanced accuracy, not raw clip-weighted accuracy.",
        "macro_f1": "Macro-F1 across pools so small pools are visible.",
        "family_accuracy": "Attribution resolves to the correct CARA family when exact pool is weak.",
        "ece": "Expected calibration error for pool/family confidence.",
        "brier": "Brier score for calibrated confidence.",
        "registry_valid_rate": "Predicted CARA id is present in the locked registry.",
        "repaired_rate": "Predictions that require an auditable registry repair.",
        "degraded_rate": "Predictions routed to family/degraded state.",
        "exception_rate": "Predictions routed to exception/manual review state.",
        "prompt_leakage_delta": "Tag-present minus tag-withheld control delta.",
        "shuffled_label_delta": "CARA lane minus shuffled-label control delta.",
        "retrieval_lift": "Lift over post-hoc embedding retrieval baseline.",
        "no_cara_lift": "Lift over same-data no-CARA baseline.",
        "generation_quality_guard": "Quality guard metric; attribution gains should not come from audio collapse.",
    }
    return [
        {
            "id": definition["metric_id"],
            "metric": definition["label"],
            "description": descriptions.get(definition["metric_id"], definition["label"]),
            "higher_is_better": definition["higher_is_better"],
            "base_external_probe": None,
            "diffusion_native": None,
            "diffusion_external_probe": None,
            "ar_native": None,
            "hybrid_native": None,
            "status": "pending_evaluation",
        }
        for definition in metric_definitions()
    ]


def latest_results_summary() -> dict[str, Any]:
    latest_metrics = _read_json(LATEST_METRICS)
    return {
        "metrics_available": bool(latest_metrics),
        "metrics_path": str(LATEST_METRICS),
        "comparison_log_path": str(COMPARISON_LOG),
        "latest_metrics": latest_metrics,
    }


def evaluation_readiness_payload(suites: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = model_lanes()
    metric_rows = default_metric_rows(lanes)
    return {
        "format": "cara_benchmark_testing_evaluation_readiness_v2",
        "registry": _registry_summary(),
        "models": model_candidates(),
        "model_lanes": lanes,
        "suites": suites,
        "benchmark_spec": benchmark_spec(suites),
        "benchmark_rows": benchmark_rows(),
        "metric_rows": metric_rows,
        "comparison_cards": comparison_cards(metric_rows),
        "latest_results": latest_results_summary(),
        "baseline_comparison_policy": baseline_comparison_policy(),
        "launch_guard": {
            "dry_run_default": True,
            "required_confirmation": "LAUNCH BENCHMARK TESTING EVALUATION",
            "cost_policy": "Use existing Azure ML workspace resources only; no Marketplace endpoints or deployments.",
        },
    }
