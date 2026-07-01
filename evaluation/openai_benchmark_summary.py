from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.env import get_env


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "evaluation" / "generated"
LATEST_SUMMARY = GENERATED_DIR / "benchmark_tldr_latest.json"
SUMMARY_LOG = GENERATED_DIR / "benchmark_tldr_log.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def latest_summary_payload() -> dict[str, Any]:
    api_key_configured = bool(get_env("OPENAI_API_KEY"))
    sdk_available = importlib.util.find_spec("openai") is not None
    configured = api_key_configured and sdk_available
    payload: dict[str, Any] = {
        "configured": configured,
        "api_key_configured": api_key_configured,
        "sdk_available": sdk_available,
        "configuration_message": (
            "OpenAI configured"
            if configured
            else "OPENAI_API_KEY is not configured in .env"
            if not api_key_configured
            else "OpenAI Python SDK is not installed in the dashboard environment"
        ),
        "latest_path": str(LATEST_SUMMARY),
        "log_path": str(SUMMARY_LOG),
        "latest": None,
    }
    if LATEST_SUMMARY.exists():
        try:
            payload["latest"] = json.loads(LATEST_SUMMARY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload["latest"] = {"status": "invalid_json", "path": str(LATEST_SUMMARY)}
    return payload


def _benchmark_prompt(benchmark_payload: dict[str, Any], readiness_payload: dict[str, Any]) -> list[dict[str, str]]:
    goal = (
        "CARA evaluates whether fine-tuned generative music/audio models can emit or support defensible "
        "registry-resolved CARA attribution IDs. The benchmark must compare native CARA output for trained "
        "models against external-probe baselines, while keeping exact pool accuracy, repairable pool accuracy, "
        "family/genre fallback, unattributable rate, calibration, leakage controls, and generation quality separate."
    )
    compact_payload = {
        "goal": goal,
        "benchmark_rows": benchmark_payload.get("rows", []),
        "metric_rows": benchmark_payload.get("metric_rows", []),
        "comparison_cards": benchmark_payload.get("comparison_cards", []),
        "repairability_matrix": benchmark_payload.get("repairability_matrix"),
        "repair_method_matrix": benchmark_payload.get("repair_method_matrix"),
        "prediction_examples": benchmark_payload.get("prediction_examples", {}),
        "model_lanes": benchmark_payload.get("model_lanes", []),
        "benchmark_spec": benchmark_payload.get("benchmark_spec", {}),
        "latest_results": benchmark_payload.get("latest_results", {}),
        "baseline_comparison_policy": benchmark_payload.get("baseline_comparison_policy", {}),
        "benchmark_notes": benchmark_payload.get("notes", []),
        "models": readiness_payload.get("models", []),
        "registry": readiness_payload.get("registry", {}),
        "repairability": readiness_payload.get("repairability", {}),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are writing a concise research abstract for benchmark testing. Be careful, factual, and do not "
                "invent benchmark numbers. If metrics are pending, say that the benchmark framework is ready but "
                "comparative results are pending. Distinguish correct repairability from raw repair-method diagnostics. "
                "Keep the answer to one abstract-style paragraph plus one short TLDR sentence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(compact_payload, sort_keys=True),
        },
    ]


def generate_benchmark_summary(
    benchmark_payload: dict[str, Any],
    readiness_payload: dict[str, Any],
    *,
    generated_prompts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    api_key = get_env("OPENAI_API_KEY")
    model = get_env("OPENAI_MODEL", "gpt-5.5") or "gpt-5.5"
    reasoning_effort = get_env("OPENAI_REASONING_EFFORT", "high") or "high"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")

    from openai import OpenAI

    messages = _benchmark_prompt(benchmark_payload, readiness_payload)
    client = OpenAI(api_key=api_key)
    request_args: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if not model.lower().startswith("gpt-5"):
        request_args["temperature"] = 0.2
    response = client.chat.completions.create(**request_args)
    content = response.choices[0].message.content if response.choices else ""
    record = {
        "status": "generated",
        "created_at": _utc_now(),
        "model": model,
        "reasoning_effort_requested": reasoning_effort,
        "summary": content,
        "benchmark_payload": benchmark_payload,
        "readiness_snapshot": readiness_payload,
        "generated_prompts": generated_prompts or [],
        "prompt_messages": messages,
        "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
    }
    _write_json(LATEST_SUMMARY, record)
    _append_jsonl(SUMMARY_LOG, record)
    return record
