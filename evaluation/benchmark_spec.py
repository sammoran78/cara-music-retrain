from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "registry" / "cara_strong"
POOL_REGISTRY = REGISTRY_DIR / "pool_registry.locked.json"
LOCKED_MANIFEST = REGISTRY_DIR / "manifest.locked.jsonl"
TRAINING_JOBS = REGISTRY_DIR / "azure_training_jobs.jsonl"
LOCK_SUMMARY = REGISTRY_DIR / "lock_summary.json"
TRAINING_INCLUSION_RECEIPT = REGISTRY_DIR / "training_inclusion_receipt.json"

PROMPT_SET_VERSION = "v2"
PROMPT_SET_FORMAT = "cara_benchmark_prompt_set_v2"
BENCHMARK_SPEC_FORMAT = "cara_peer_review_benchmark_spec_v2"

CONDITION_TAG_PRESENT = "tag_present"
CONDITION_TAG_WITHHELD = "tag_withheld"
CONDITION_NO_TAG = "no_tag"
CONDITION_SHUFFLED_LABEL = "shuffled_label"
CONDITION_HELDOUT_AUDIO = "heldout_audio"
CONDITION_OPEN_QUALITY = "open_quality"

PROMPT_CONDITIONS = [
    CONDITION_TAG_PRESENT,
    CONDITION_TAG_WITHHELD,
    CONDITION_NO_TAG,
    CONDITION_SHUFFLED_LABEL,
    CONDITION_HELDOUT_AUDIO,
    CONDITION_OPEN_QUALITY,
]

CONDITION_DESCRIPTIONS = {
    CONDITION_TAG_PRESENT: "Visible CARA identifier sanity/control condition.",
    CONDITION_TAG_WITHHELD: "Headline condition: prompt omits CARA identifiers while expected pool/family remain known.",
    CONDITION_NO_TAG: "No CARA identifier and no pool/family text in the prompt.",
    CONDITION_SHUFFLED_LABEL: "Expected labels are intentionally shuffled to expose label-prior shortcuts.",
    CONDITION_HELDOUT_AUDIO: "Known held-out source row for audio/probe attribution scoring.",
    CONDITION_OPEN_QUALITY: "Open generation quality guard with no pool-accuracy claim.",
}

HEADLINE_SUCCESS_CRITERIA = [
    "Report headline results on tag_withheld rows.",
    "CARA-Strong must beat released-base, same-data no-CARA where available, retrieval, prompt-only, and shuffled-label controls.",
    "Use balanced accuracy, macro-F1, calibration, per-pool audit, and bootstrap confidence intervals before making peer-review claims.",
    "State only pool-level recoverable association; do not claim individual-work causality.",
]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _stable_digest(*parts: Any, length: int = 16) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _pool_codeword(pool_id: Any, fallback: Any = None) -> str | None:
    if fallback not in (None, ""):
        return str(fallback)
    parts = str(pool_id or "").split(":")
    if len(parts) >= 5 and parts[0] == "CARA":
        return parts[3]
    return str(pool_id or "").strip() or None


def _strip_cara_tokens(prompt: str, row: dict[str, Any]) -> str:
    cleaned = str(prompt or "CARA benchmark audio")
    removals = {
        row.get("cara_pool_id"),
        row.get("cara_registered_codeword"),
        row.get("cara_pool_codeword"),
        _pool_codeword(row.get("cara_pool_id")),
    }
    for token in sorted({str(item) for item in removals if item not in (None, "")}, key=len, reverse=True):
        cleaned = cleaned.replace(token, "")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip(" ,|-") or "High-quality original music audio."


def _prompt_for_condition(row: dict[str, Any], condition: str) -> str:
    base_prompt = str(row.get("prompt") or row.get("title") or "High-quality original music audio.").strip()
    base_prompt = base_prompt or "High-quality original music audio."
    codeword = _pool_codeword(row.get("cara_pool_id"), row.get("cara_pool_codeword")) or "UNKNOWN"
    family = str(row.get("cara_pool_family") or row.get("primary_genre") or "audio")
    if condition == CONDITION_TAG_PRESENT:
        return f"{base_prompt} CARA pool {codeword}."
    if condition == CONDITION_TAG_WITHHELD:
        return _strip_cara_tokens(base_prompt, row)
    if condition == CONDITION_NO_TAG:
        text = _strip_cara_tokens(base_prompt, row)
        for token in {family, str(row.get("primary_genre") or ""), str(row.get("secondary_genre") or "")}:
            if token:
                text = text.replace(token, "").replace(token.lower(), "")
        return text.strip(" ,|-") or "High-quality original music audio."
    if condition == CONDITION_SHUFFLED_LABEL:
        return _strip_cara_tokens(base_prompt, row)
    if condition == CONDITION_HELDOUT_AUDIO:
        return _strip_cara_tokens(base_prompt, row)
    if condition == CONDITION_OPEN_QUALITY:
        return f"High-quality original {family} music loop with clear production detail."
    return base_prompt


def _source_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_example_id": row.get("example_id"),
        "source_id": row.get("source_id"),
        "source_split": row.get("split"),
        "source_group_key": row.get("split_group_key"),
        "prepared_audio_path": row.get("prepared_audio_path") or row.get("local_audio_path"),
    }


def _expected_fields(row: dict[str, Any], *, include_label: bool = True) -> dict[str, Any]:
    if not include_label:
        return {
            "cara_pool_id": None,
            "cara_pool_index": None,
            "cara_pool_family": None,
            "cara_pool_family_index": None,
            "cara_pool_codeword": None,
        }
    return {
        "cara_pool_id": row.get("cara_pool_id"),
        "cara_pool_index": row.get("cara_pool_index"),
        "cara_pool_family": row.get("cara_pool_family"),
        "cara_pool_family_index": row.get("cara_pool_family_index"),
        "cara_pool_codeword": _pool_codeword(row.get("cara_pool_id"), row.get("cara_pool_codeword")),
    }


def _eligible_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heldout = [
        row
        for row in rows
        if str(row.get("split") or "").lower() in {"validation", "test"} and row.get("cara_pool_id")
    ]
    return heldout or [row for row in rows if row.get("cara_pool_id")]


def _deterministic_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _stable_digest(row.get("cara_pool_id"), row.get("example_id"), row.get("source_id")))
    return ordered[: min(count, len(ordered))]


def build_prompt_manifest_v2(
    rows: list[dict[str, Any]],
    *,
    seeds: int = 1,
    condition_limit: int = 80,
) -> list[dict[str, Any]]:
    """Build a deterministic peer-review prompt manifest without copying labels into predictions."""
    candidates = _deterministic_sample(_eligible_prompt_rows(rows), condition_limit)
    shuffled_labels = [row.get("cara_pool_id") for row in candidates]
    shuffled_labels = sorted(shuffled_labels, key=lambda value: _stable_digest("shuffle", value))
    if len(shuffled_labels) > 1:
        shuffled_labels = shuffled_labels[1:] + shuffled_labels[:1]

    manifest: list[dict[str, Any]] = []
    for condition in PROMPT_CONDITIONS:
        for row_index, row in enumerate(candidates):
            label_row = dict(row)
            if condition == CONDITION_SHUFFLED_LABEL:
                label_row["cara_pool_id"] = shuffled_labels[row_index]
            include_label = condition != CONDITION_OPEN_QUALITY
            for seed in range(max(1, int(seeds))):
                row_key = _stable_digest(row.get("example_id"), row.get("source_id"), row_index, length=10)
                prompt_id = f"bpsv2-{condition}-{row_key}-s{seed:02d}"
                manifest.append(
                    {
                        "format": "cara_benchmark_prompt_row_v2",
                        "prompt_set_version": PROMPT_SET_VERSION,
                        "prompt_id": prompt_id,
                        "suite_id": "peer_review_core",
                        "condition": condition,
                        "condition_label": condition.replace("_", " "),
                        "seed": seed,
                        "prompt": _prompt_for_condition(row, condition),
                        "expected": _expected_fields(label_row, include_label=include_label),
                        "source": _source_fields(row),
                        "leakage_policy": {
                            "prompt_contains_visible_cara": condition == CONDITION_TAG_PRESENT,
                            "headline_eligible": condition == CONDITION_TAG_WITHHELD,
                            "expected_label_is_shuffled": condition == CONDITION_SHUFFLED_LABEL,
                            "pool_accuracy_applicable": include_label,
                        },
                    }
                )
    return manifest


def suite_definitions(existing_suites: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    by_id = {str(suite.get("id")): dict(suite) for suite in existing_suites or []}
    return [
        {
            "id": condition,
            "label": condition.replace("_", " ").title(),
            "description": CONDITION_DESCRIPTIONS[condition],
            "prompt_count": 0 if condition == CONDITION_HELDOUT_AUDIO else 80,
            "condition": condition,
            "headline_eligible": condition == CONDITION_TAG_WITHHELD,
            "baseline_supported": True,
            "legacy_suite": by_id.get("heldout_audio_attribution" if condition == CONDITION_HELDOUT_AUDIO else "known_pool_prompt_recall"),
        }
        for condition in PROMPT_CONDITIONS
    ]


def benchmark_spec(existing_suites: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    registry = _read_json(POOL_REGISTRY, {}) or {}
    lock_summary = _read_json(LOCK_SUMMARY, {}) or {}
    tir = _read_json(TRAINING_INCLUSION_RECEIPT, {}) or {}
    return {
        "format": BENCHMARK_SPEC_FORMAT,
        "prompt_set_version": PROMPT_SET_VERSION,
        "registry_path": str(POOL_REGISTRY.relative_to(ROOT)),
        "registry_hash": (
            registry.get("registry_hash")
            or registry.get("hash")
            or (f"sha256:{tir['source_registry_sha256']}" if tir.get("source_registry_sha256") else None)
            or _file_sha256(POOL_REGISTRY)
        ),
        "manifest_path": str(LOCKED_MANIFEST.relative_to(ROOT)),
        "manifest_hash": f"sha256:{tir['source_manifest_sha256']}" if tir.get("source_manifest_sha256") else None,
        "pool_count": registry.get("pool_count") or lock_summary.get("pool_count") or len(registry.get("pool_index") or {}),
        "family_count": registry.get("family_count") or lock_summary.get("family_count") or len(registry.get("family_index") or {}),
        "suite_definitions": suite_definitions(existing_suites),
        "prompt_conditions": [
            {"id": condition, "description": CONDITION_DESCRIPTIONS[condition]}
            for condition in PROMPT_CONDITIONS
        ],
        "success_criteria": HEADLINE_SUCCESS_CRITERIA,
        "claim_language": "Recoverable, confidence-scored pool-level attribution under codeword-withheld evaluation.",
        "cost_policy": "Use existing Azure ML workspace resources only; no Marketplace endpoints or deployments.",
    }


def _job_events() -> list[dict[str, Any]]:
    return _read_jsonl(TRAINING_JOBS)


def _latest_event(*, model_family: str | None = None, variant: str | None = None, training_scope: str | None = None) -> dict[str, Any] | None:
    for event in reversed(_job_events()):
        if model_family and str(event.get("model_family") or "") != model_family:
            continue
        if variant and str(event.get("variant") or "") != variant:
            continue
        if training_scope and str(event.get("training_scope") or "") != training_scope:
            continue
        return event
    return None


def _artifact_check(path: str | None, required: bool = True) -> dict[str, Any]:
    if not path:
        return {"required": required, "status": "missing", "path": None}
    if path.startswith("azureml://"):
        return {"required": required, "status": "remote_unverified", "path": path}
    local = Path(path)
    if not local.is_absolute():
        local = ROOT / local
    return {
        "required": required,
        "status": "present" if local.exists() else "missing",
        "path": str(local),
        "size_bytes": local.stat().st_size if local.exists() and local.is_file() else None,
    }


def _lane_status(artifact_checks: list[dict[str, Any]], event: dict[str, Any] | None, *, native_cara_output: bool) -> str:
    if any(check.get("required") and check.get("status") == "missing" for check in artifact_checks):
        return "Blocked: missing checkpoint"
    if event and str(event.get("status") or "").lower() in {"failed", "canceled", "cancelled"}:
        return "Failed guardrail"
    if event and str(event.get("status") or "").lower() == "completed":
        return "Ready"
    if not native_cara_output:
        return "Ready"
    if artifact_checks and all(check.get("status") in {"present", "remote_unverified"} for check in artifact_checks):
        return "Ready"
    return "Blocked: missing checkpoint"


def model_lanes(extra_lanes: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    stable_full = _latest_event(model_family="stable_audio_open_small", variant="cara_strong", training_scope="full")
    context_full = _latest_event(
        model_family="stable_audio_open_small_context_diffusion",
        variant="cara_strong_context_conditioned",
        training_scope="full",
    )
    stable_no_cara = _latest_event(model_family="stable_audio_open_small", variant="no_cara_baseline")
    musicgen_full = _latest_event(model_family="musicgen", variant="cara_strong", training_scope="full")
    musicgen_no_cara = _latest_event(model_family="musicgen", variant="no_cara_baseline")

    lanes = [
        {
            "model_id": "base_stable_audio_open_small",
            "label": "Released Base · Stable Audio Open Small",
            "family": "stable_audio_open_small",
            "architecture": "diffusion",
            "variant": "released_base",
            "checkpoint_uri": "stabilityai/stable-audio-open-small",
            "output_uri": None,
            "native_prediction_adapter": None,
            "generation_adapter": "stable_audio",
            "native_cara_output": False,
            "baseline_role": "released_base",
        },
        {
            "model_id": "stable_audio_no_cara_baseline",
            "label": "Same-data Baseline · Stable Audio",
            "family": "stable_audio_open_small",
            "architecture": "diffusion",
            "variant": "same_data_no_cara",
            "checkpoint_uri": "stabilityai/stable-audio-open-small",
            "output_uri": (stable_no_cara or {}).get("output_path"),
            "native_prediction_adapter": "stable_audio_external_probe",
            "generation_adapter": "stable_audio",
            "native_cara_output": False,
            "baseline_role": "same_data_no_cara",
        },
        {
            "model_id": "diffusion_cara_strong_full_modest_arch",
            "label": "CARA-Strong · Stable Audio",
            "family": "stable_audio_open_small",
            "architecture": "diffusion",
            "variant": "cara_strong",
            "checkpoint_uri": "stabilityai/stable-audio-open-small",
            "output_uri": (stable_full or {}).get("output_path"),
            "native_prediction_adapter": "stable_audio_dit_hidden_state",
            "generation_adapter": "stable_audio",
            "native_cara_output": True,
            "baseline_role": "candidate",
        },
        {
            "model_id": "context_diffusion_cara_strong_full",
            "label": "Context CARA-Strong · Stable Audio",
            "family": "stable_audio_open_small_context_diffusion",
            "architecture": "context_diffusion",
            "variant": "cara_strong_context_conditioned",
            "checkpoint_uri": "stabilityai/stable-audio-open-small",
            "output_uri": (context_full or {}).get("output_path"),
            "native_prediction_adapter": "stable_audio_dit_hidden_state",
            "generation_adapter": "stable_audio",
            "native_cara_output": True,
            "baseline_role": "candidate",
        },
        {
            "model_id": "base_musicgen_small",
            "label": "Released Base · MusicGen Small",
            "family": "musicgen",
            "architecture": "autoregressive",
            "variant": "released_base",
            "checkpoint_uri": "facebook/musicgen-small",
            "output_uri": None,
            "native_prediction_adapter": None,
            "generation_adapter": "musicgen",
            "native_cara_output": False,
            "baseline_role": "released_base",
        },
        {
            "model_id": "musicgen_no_cara_baseline",
            "label": "Same-data Baseline · MusicGen",
            "family": "musicgen",
            "architecture": "autoregressive",
            "variant": "same_data_no_cara",
            "checkpoint_uri": "facebook/musicgen-small",
            "output_uri": (musicgen_no_cara or {}).get("output_path"),
            "native_prediction_adapter": "musicgen_external_probe",
            "generation_adapter": "musicgen",
            "native_cara_output": False,
            "baseline_role": "same_data_no_cara",
        },
        {
            "model_id": "musicgen_cara_strong_full",
            "label": "CARA-Strong · MusicGen",
            "family": "musicgen",
            "architecture": "autoregressive",
            "variant": "cara_strong",
            "checkpoint_uri": "facebook/musicgen-small",
            "output_uri": (musicgen_full or {}).get("output_path"),
            "native_prediction_adapter": "musicgen_lm_suffix",
            "generation_adapter": "musicgen",
            "native_cara_output": True,
            "baseline_role": "candidate",
        },
        {
            "model_id": "retrieval_baseline",
            "label": "Retrieval Baseline · Embedding NN",
            "family": "post_hoc",
            "architecture": "retrieval",
            "variant": "retrieval_baseline",
            "checkpoint_uri": None,
            "output_uri": None,
            "native_prediction_adapter": "embedding_retrieval",
            "generation_adapter": "post_hoc",
            "native_cara_output": False,
            "baseline_role": "retrieval_baseline",
        },
    ]
    lanes.extend(extra_lanes or [])
    event_by_lane = {
        "diffusion_cara_strong_full_modest_arch": stable_full,
        "context_diffusion_cara_strong_full": context_full,
        "stable_audio_no_cara_baseline": stable_no_cara,
        "musicgen_cara_strong_full": musicgen_full,
        "musicgen_no_cara_baseline": musicgen_no_cara,
    }
    for lane in lanes:
        checks = [
            _artifact_check(
                lane.get("output_uri"),
                required=lane.get("variant") in {"cara_strong", "cara_strong_context_conditioned", "same_data_no_cara"},
            ),
        ]
        lane["artifact_checks"] = checks
        lane["status"] = _lane_status(checks, event_by_lane.get(lane["model_id"]), native_cara_output=bool(lane.get("native_cara_output")))
        lane["latest_job"] = event_by_lane.get(lane["model_id"])
    return lanes


METRIC_DEFINITIONS = [
    ("exact_pool_top1", "Exact pool top-1", True, "pool"),
    ("exact_pool_top3", "Exact pool top-3", True, "pool"),
    ("balanced_accuracy", "Balanced accuracy", True, "pool"),
    ("macro_f1", "Macro-F1", True, "pool"),
    ("family_accuracy", "Family accuracy", True, "family"),
    ("ece", "Calibration ECE", False, "calibration"),
    ("brier", "Brier score", False, "calibration"),
    ("registry_valid_rate", "Registry-valid rate", True, "registry"),
    ("repaired_rate", "Repaired rate", False, "repair"),
    ("degraded_rate", "Degraded rate", False, "repair"),
    ("exception_rate", "Exception rate", False, "repair"),
    ("prompt_leakage_delta", "Prompt-leakage delta", False, "control"),
    ("shuffled_label_delta", "Shuffled-label delta", False, "control"),
    ("retrieval_lift", "Lift over retrieval", True, "control"),
    ("no_cara_lift", "Lift over no-CARA", True, "control"),
    ("generation_quality_guard", "Generation quality guard", True, "quality"),
]


def metric_definitions() -> list[dict[str, Any]]:
    return [
        {"metric_id": metric_id, "label": label, "higher_is_better": higher, "family": family}
        for metric_id, label, higher, family in METRIC_DEFINITIONS
    ]


def _lane_metric_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lanes = metrics.get("lanes") if isinstance(metrics.get("lanes"), dict) else {}
    lane_model_fallbacks = {
        "base_external_probe": ("base_stable_audio_open_small", "released_base", "external_probe"),
        "base_native": ("base_stable_audio_open_small", "released_base", "native"),
        "diffusion_external_probe": ("diffusion_cara_strong_full_modest_arch", "cara_strong", "external_probe"),
        "diffusion_native": ("diffusion_cara_strong_full_modest_arch", "cara_strong", "native"),
        "context_diffusion_external_probe": ("context_diffusion_cara_strong_full", "cara_strong_context_conditioned", "external_probe"),
        "context_diffusion_native": ("context_diffusion_cara_strong_full", "cara_strong_context_conditioned", "native"),
        "base_musicgen_external_probe": ("base_musicgen_small", "released_base", "external_probe"),
        "base_musicgen_native": ("base_musicgen_small", "released_base", "native"),
        "musicgen_external_probe": ("musicgen_cara_strong_full", "cara_strong", "external_probe"),
        "musicgen_native": ("musicgen_cara_strong_full", "cara_strong", "native"),
    }
    for lane_id, lane_metrics in lanes.items():
        if not isinstance(lane_metrics, dict):
            continue
        fallback_model_id, fallback_variant, fallback_evidence = lane_model_fallbacks.get(
            str(lane_id),
            (str(lane_id), str(lane_id), "native" if "native" in str(lane_id) else "external_probe"),
        )
        model_id = str(lane_metrics.get("model_id") or fallback_model_id)
        variant = str(lane_metrics.get("variant") or fallback_variant)
        evidence_lane = str(lane_metrics.get("evidence_lane") or fallback_evidence)
        suite_id = str(lane_metrics.get("suite_id") or "all")
        condition = str(lane_metrics.get("condition") or "mixed")
        for definition in metric_definitions():
            metric_id = definition["metric_id"]
            value = lane_metrics.get(metric_id)
            if value is None and metric_id == "exact_pool_top1":
                value = lane_metrics.get("pool_exact_accuracy")
            if value is None and metric_id == "family_accuracy":
                value = lane_metrics.get("family_or_genre_accuracy")
            if value is None and metric_id == "registry_valid_rate":
                value = lane_metrics.get("registry_valid_rate")
            rows.append(
                {
                    "format": "cara_metric_row_v2",
                    "model_id": model_id,
                    "variant": variant,
                    "evidence_lane": evidence_lane,
                    "suite_id": suite_id,
                    "condition": condition,
                    "metric_id": metric_id,
                    "metric_label": definition["label"],
                    "value": value,
                    "ci_low": lane_metrics.get(f"{metric_id}_ci_low"),
                    "ci_high": lane_metrics.get(f"{metric_id}_ci_high"),
                    "status": lane_metrics.get("status") or ("scored" if value is not None else "missing_predictions"),
                    "higher_is_better": definition["higher_is_better"],
                }
            )
    return rows


def default_metric_rows(lanes: list[dict[str, Any]] | None = None, metrics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if isinstance(metrics, dict):
        rows = _lane_metric_records(metrics)
        if rows:
            return rows
    rows = []
    for lane in lanes or model_lanes():
        evidence_lane = "native" if lane.get("native_cara_output") else "external_probe"
        for definition in metric_definitions():
            rows.append(
                {
                    "format": "cara_metric_row_v2",
                    "model_id": lane["model_id"],
                    "variant": lane["variant"],
                    "evidence_lane": evidence_lane,
                    "suite_id": "peer_review_core",
                    "condition": CONDITION_TAG_WITHHELD,
                    "metric_id": definition["metric_id"],
                    "metric_label": definition["label"],
                    "value": None,
                    "ci_low": None,
                    "ci_high": None,
                    "status": "missing_predictions" if lane.get("status") == "Ready" else lane.get("status"),
                    "higher_is_better": definition["higher_is_better"],
                }
            )
    return rows


def comparison_cards(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metric_rows:
        by_key[(str(row.get("model_id")), str(row.get("condition")), str(row.get("metric_id")))] = row

    def lookup(model_id: str, metric_id: str) -> dict[str, Any]:
        return (
            by_key.get((model_id, CONDITION_TAG_WITHHELD, metric_id))
            or by_key.get((model_id, "mixed", metric_id))
            or {}
        )

    def card(card_id: str, title: str, candidate: str, baseline: str, metric_id: str = "balanced_accuracy") -> dict[str, Any]:
        condition = CONDITION_TAG_WITHHELD
        cand = lookup(candidate, metric_id)
        base = lookup(baseline, metric_id)
        cand_value = cand.get("value")
        base_value = base.get("value")
        delta = cand_value - base_value if isinstance(cand_value, (int, float)) and isinstance(base_value, (int, float)) else None
        if delta is not None:
            status = "Peer-review complete"
        elif isinstance(cand_value, (int, float)):
            status = "Scored: missing baseline"
        elif isinstance(base_value, (int, float)):
            status = "Scored: missing candidate"
        else:
            status = "Scored: missing external probe"
        return {
            "id": card_id,
            "title": title,
            "candidate_model_id": candidate,
            "baseline_model_id": baseline,
            "metric_id": metric_id,
            "condition": condition,
            "candidate_value": cand_value,
            "baseline_value": base_value,
            "delta": delta,
            "ci_low": cand.get("ci_low"),
            "ci_high": cand.get("ci_high"),
            "status": status,
        }

    return [
        card("context_vs_diffusion", "Context Diffusion vs Diffusion", "context_diffusion_cara_strong_full", "diffusion_cara_strong_full_modest_arch"),
        card("context_vs_musicgen", "Context Diffusion vs MusicGen", "context_diffusion_cara_strong_full", "musicgen_cara_strong_full"),
        card("diffusion_vs_musicgen", "Diffusion vs MusicGen", "diffusion_cara_strong_full_modest_arch", "musicgen_cara_strong_full"),
        card("context_vs_retrieval", "Context Diffusion vs retrieval", "context_diffusion_cara_strong_full", "retrieval_baseline"),
        card("musicgen_vs_retrieval", "MusicGen vs retrieval", "musicgen_cara_strong_full", "retrieval_baseline"),
    ]


def prompt_set_summary(lock: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format": PROMPT_SET_FORMAT,
        "prompt_set_version": PROMPT_SET_VERSION,
        "conditions": PROMPT_CONDITIONS,
        "legacy_v1": lock if lock and lock.get("format") == "cara_benchmark_prompt_set_v1" else None,
        "reuse_policy": "All Diffusion, MusicGen, retrieval, and future model lanes must reuse the same v2 prompt rows for like-for-like scoring.",
    }


def manifest_summary(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    tir = _read_json(TRAINING_INCLUSION_RECEIPT, {}) or {}
    lock_summary = _read_json(LOCK_SUMMARY, {}) or {}
    if rows is None and lock_summary:
        return {
            "format": "cara_manifest_summary_v2",
            "row_count": lock_summary.get("accepted_count"),
            "split_counts": lock_summary.get("split_counts") or {},
            "pool_count": lock_summary.get("pool_count"),
            "family_count": lock_summary.get("family_count"),
            "low_power_pool_count": lock_summary.get("low_power_pool_count"),
            "manifest_hash": f"sha256:{tir['source_manifest_sha256']}" if tir.get("source_manifest_sha256") else None,
        }
    sample = rows if rows is not None else _read_jsonl(LOCKED_MANIFEST)
    by_split = Counter(str(row.get("split") or "unknown") for row in sample)
    by_pool = Counter(str(row.get("cara_pool_id") or "unknown") for row in sample if row.get("cara_pool_id"))
    by_family = Counter(str(row.get("cara_pool_family") or "unknown") for row in sample if row.get("cara_pool_family"))
    low_power = [pool for pool, count in by_pool.items() if count < 5]
    return {
        "format": "cara_manifest_summary_v2",
        "row_count": len(sample),
        "split_counts": dict(sorted(by_split.items())),
        "pool_count": len(by_pool),
        "family_count": len(by_family),
        "low_power_pool_count": len(low_power),
        "manifest_hash": _file_sha256(LOCKED_MANIFEST),
    }
