from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common.config import load_pipeline_config, load_project_config
from common.env import get_env
from data_pipeline.pool_allocator import (
    AllocatorPaths,
    RunOptions,
    _candidate_manifest_rows,
    list_assignments,
    list_pools,
    list_review_queue,
    load_allocator_config,
    read_progress_state,
    reset_allocator_state,
    run_pool_allocation,
    summarize_registry,
)
from data_pipeline.pool_allocator_v2 import (
    AllocatorV2Paths,
    RunOptionsV2,
    list_assignments_v2,
    list_pools_v2,
    list_review_queue_v2,
    load_allocator_v2_config,
    read_progress_state_v2,
    reset_allocator_v2_state,
    run_pool_allocation_v2,
    summarize_registry_v2,
)
from data_pipeline.manifest_utils import load_manifest_rows


app = FastAPI(title="CARA Attribution PoC")
ROOT = Path(__file__).resolve().parents[2]

_DOCS_MARKDOWN_FILES: dict[str, dict[str, Any]] = {
    "runbook": {
        "title": "CARA Fine-Tuning Runbook",
        "description": "Living implementation runbook for dataset prep, fine-tuning gates, Azure job policy, and benchmark procedure.",
        "path": ROOT / "docs" / "cara_finetuning_runbook.md",
    },
    "experiment-log": {
        "title": "Experiment Log",
        "description": "Append-only record of methodology changes, failed runs, fixes, and evidence checkpoints.",
        "path": ROOT / "docs" / "EXPERIMENT_LOG.md",
    },
}


class ValidateRequest(BaseModel):
    attr_string: str


class GenerateRequest(BaseModel):
    prompt: str


class DownloadNextRequest(BaseModel):
    subset_mode: str = "subset_role"
    subset_role: Optional[str] = "music_train_candidate"
    skip_audio: bool = False
    count: int = 1
    target_items: int = 2000


class RetryUnavailableRequest(BaseModel):
    mode: str = "temporary"


class PoolAllocationRunRequest(BaseModel):
    engine_version: str = "v1"
    subset_role: Optional[str] = None
    only_downloaded: bool = True
    limit: Optional[int] = None
    allow_relaxed_metadata: bool = False
    start_fresh: bool = False


class AzureMLTestPrepRunRequest(BaseModel):
    confirm_gpu: bool = False
    allow_prerequisite_override: bool = False


class TrainingManifestLockRequest(BaseModel):
    manifest_path: Optional[str] = None
    pool_registry_path: Optional[str] = None
    output_dir: Optional[str] = None
    require_audio_exists: bool = False
    dry_run: bool = False


class TrainingPreprocessRunRequest(BaseModel):
    dry_run: bool = False
    models: str = "all"
    compute_strategy: str = "prefer_h100_else_cpu"


class TrainingMusicGenTokenCacheRunRequest(BaseModel):
    dry_run: bool = False
    compute_strategy: str = "prefer_h100_else_cpu"


class TrainingMusicGenPreflightRunRequest(BaseModel):
    checkpoint: str = "facebook/musicgen-small"


class TrainingStableAudioPreflightRunRequest(BaseModel):
    checkpoint: str = "stabilityai/stable-audio-open-small"
    wrapper_check: bool = True


class TrainingContextDiffusionPackRunRequest(BaseModel):
    dry_run: bool = False
    max_contexts: int = 3
    selection_seed: str = "cara-context-v1"


class TrainingContextDiffusionCacheRunRequest(BaseModel):
    dry_run: bool = False


class TrainingContextDiffusionPreflightRunRequest(BaseModel):
    dry_run: bool = False


class TrainingContextDiffusionSmokeRunRequest(BaseModel):
    dry_run: bool = False
    max_steps: int = 250
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_train_rows: int = 4096
    max_eval_rows: int = 1024


class TrainingContextDiffusionFullRunRequest(BaseModel):
    dry_run: bool = False
    confirmation_phrase: str = ""
    max_steps: int = 20000
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_workers: int = 0
    precision: str = "16-mixed"
    checkpoint_every: int = 1000
    checkpoint_keep_last_n: int = 1
    max_train_files: int = 0
    max_eval_files: int = 0
    max_eval_batches: int = 0
    attribution_loss_weight: float = 0.05


class TrainingAcePreflightRunRequest(BaseModel):
    checkpoint: str = "ACE-Step/Ace-Step1.5"
    load_checkpoint: bool = False


class TrainingAzureUploadConfirmRequest(BaseModel):
    confirmed: bool = True


class TrainingStartRequest(BaseModel):
    model_family: str = "stable_audio_open_small"
    variant: str = "no_cara_baseline"
    training_scope: str = "smoke"
    run_name: str = "cara-stable-audio-smoke"
    max_steps: int = 250
    full_training_run: bool = False
    batch_size: int = 8
    num_workers: int = 0
    learning_rate: float = 1e-5
    attribution_loss_weight: float = 0.05
    checkpoint_keep_last_n: int = 1
    max_train_files: int = 2048
    max_eval_files: int = 512
    max_eval_batches: int = 16
    run_eval: bool = True
    checkpoint: str = "stabilityai/stable-audio-open-small"
    trainer_compute_target: str = "gpu-smoke-h100"
    dry_run: bool = False
    launch_confirmation: str = ""


class EvaluationRunRequest(BaseModel):
    model_ids: list[str]
    suite_ids: list[str]
    custom_prompt: Optional[str] = None
    seeds: int = 3
    dry_run: bool = True
    launch_confirmation: str = ""


class EvaluationPromptSetLockRequest(BaseModel):
    job_name: Optional[str] = None
    confirmed: bool = True


class EvaluationAudioBenchmarkRunRequest(BaseModel):
    model_ids: list[str] = ["base_stable_audio_open_small", "diffusion_cara_strong_full_modest_arch"]
    suite_ids: list[str] = ["known_pool_prompt_recall", "control_token_confound"]
    scope: str = "smoke"
    seed_ids: list[int] = [0]
    max_prompts: int = 20
    dry_run: bool = True
    launch_confirmation: str = ""


class EvaluationAudioBenchmarkRetryMissingRequest(BaseModel):
    source_job_name: Optional[str] = None
    dry_run: bool = True
    launch_confirmation: str = ""


class EvaluationAttributionScoringRunRequest(BaseModel):
    audio_job_name: Optional[str] = None
    model_ids: list[str] = []
    dry_run: bool = True
    force_rescore: bool = False
    launch_confirmation: str = ""


_download_lock = threading.Lock()
_job_lock = threading.Lock()
_job_stop_event = threading.Event()
_job_thread: Optional[threading.Thread] = None
_job_state: dict[str, Any] = {
    "running": False,
    "requested_stop": False,
    "started_at": None,
    "finished_at": None,
    "completed_batches": 0,
    "target_batches": 0,
    "target_items": 0,
    "last_message": None,
}
_last_activity: dict[str, Any] = {
    "subset_mode": None,
    "subset_role": None,
    "skip_audio": False,
    "last_summary": None,
    "last_batch_started_at": None,
    "last_batch_at": None,
    "last_error": None,
}

_TRAINING_HF_TOKEN_SECRET_NAME = "hf-token"
_pool_job_lock = threading.Lock()
_pool_job_thread: Optional[threading.Thread] = None
_pool_job_stop_event = threading.Event()
_pool_job_state: dict[str, Any] = {
    "running": False,
    "requested_stop": False,
    "started_at": None,
    "finished_at": None,
    "processed_assets": 0,
    "total_assets": 0,
    "percent_complete": 0.0,
    "current_phase": None,
    "current_asset": None,
    "current_asset_title": None,
    "current_pool_id": None,
    "counts": {},
    "last_message": None,
    "last_error": None,
    "latest_run_id": None,
    "options": {"subset_role": None, "only_downloaded": True, "limit": None, "allow_relaxed_metadata": False, "start_fresh": False},
}
_azureml_test_prep_audit_lock = threading.Lock()
_azureml_test_prep_submission_lock = threading.Lock()
_azureml_test_prep_submission_state: dict[str, dict[str, Any]] = {}
_azureml_test_prep_cache_lock = threading.Lock()
_azureml_test_prep_cache: dict[str, Any] = {
    "jobs": [],
    "refreshed_at": None,
    "refreshing": False,
    "last_error": None,
}
_azureml_test_prep_run_all_lock = threading.Lock()
_azureml_test_prep_run_all_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "current_test_id": None,
    "last_message": None,
    "last_error": None,
    "submitted_jobs": [],
}

_TRAINING_LOCK_DIR = ROOT / "registry" / "cara_strong"
_TRAINING_PREPROCESS_JOB_FILE = ROOT / "azureml" / "jobs" / "05_prepare_model_datasets.yml"
_TRAINING_PREPROCESS_JOB_FILES = {
    "all": ROOT / "azureml" / "jobs" / "05_prepare_model_datasets.yml",
    "stable_audio_open_small": ROOT / "azureml" / "jobs" / "05a_prepare_stable_audio_dataset.yml",
    "musicgen": ROOT / "azureml" / "jobs" / "05b_prepare_musicgen_dataset.yml",
}
_TRAINING_MUSICGEN_TOKEN_CACHE_JOB_FILE = ROOT / "azureml" / "jobs" / "06_cache_musicgen_encodec_tokens.yml"
_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE = ROOT / "azureml" / "jobs" / "07b_musicgen_trainer_preflight.yml"
_TRAINING_MUSICGEN_SMOKE_JOB_FILE = ROOT / "azureml" / "jobs" / "08_smoke_musicgen_ar_trainer.yml"
_TRAINING_MUSICGEN_FULL_JOB_FILE = ROOT / "azureml" / "jobs" / "12_full_musicgen_cara_strong_trainer.yml"
_TRAINING_STABLE_AUDIO_PREFLIGHT_JOB_FILE = ROOT / "azureml" / "jobs" / "07a_stable_audio_trainer_preflight.yml"
_TRAINING_STABLE_AUDIO_SMOKE_JOB_FILE = ROOT / "azureml" / "jobs" / "07_smoke_stable_audio_trainer.yml"
_TRAINING_STABLE_AUDIO_FULL_JOB_FILE = ROOT / "azureml" / "jobs" / "09_full_stable_audio_cara_strong_trainer.yml"
_TRAINING_CONTEXT_PACK_JOB_FILE = ROOT / "azureml" / "jobs" / "10_prepare_stable_audio_context_packs.yml"
_TRAINING_CONTEXT_CACHE_JOB_FILE = ROOT / "azureml" / "jobs" / "11_cache_stable_audio_context_metadata.yml"
_TRAINING_CONTEXT_PREFLIGHT_JOB_FILE = ROOT / "azureml" / "jobs" / "12_stable_audio_context_preflight.yml"
_TRAINING_CONTEXT_SMOKE_JOB_FILE = ROOT / "azureml" / "jobs" / "13_stable_audio_context_smoke.yml"
_TRAINING_CONTEXT_FULL_JOB_FILE = ROOT / "azureml" / "jobs" / "14_full_stable_audio_context_trainer.yml"
_TRAINING_ACE_PREFLIGHT_JOB_FILE = ROOT / "azureml" / "jobs" / "13_ace_step_env_preflight.yml"
_EVALUATION_STABLE_AUDIO_JOB_FILE = ROOT / "azureml" / "jobs" / "14_benchmark_testing_stable_audio_eval.yml"
_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE = ROOT / "azureml" / "jobs" / "15_benchmark_testing_stable_audio_audio.yml"
_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE = ROOT / "azureml" / "jobs" / "16_benchmark_testing_stable_audio_score.yml"
_EVALUATION_MUSICGEN_AUDIO_JOB_FILE = ROOT / "azureml" / "jobs" / "17_benchmark_testing_musicgen_audio.yml"
_EVALUATION_MUSICGEN_SCORE_JOB_FILE = ROOT / "azureml" / "jobs" / "18_benchmark_testing_musicgen_score.yml"
_TRAINING_JOB_REGISTRY = _TRAINING_LOCK_DIR / "azure_training_jobs.jsonl"
_EVALUATION_JOB_REGISTRY = ROOT / "evaluation" / "generated" / "azure_evaluation_jobs.jsonl"
_EVALUATION_PROMPT_SET_LOCK = ROOT / "evaluation" / "generated" / "benchmark_prompt_set.lock.json"
_EVALUATION_SCORING_METRICS_CACHE_DIR = ROOT / "evaluation" / "generated" / "scoring_metrics_cache"
_TRAINING_AZURE_UPLOAD_CONFIRMATION = _TRAINING_LOCK_DIR / "azure_upload_confirmed.json"
_TRAINING_SOURCE_URI = "azureml://datastores/ds_cara_raw_audio/paths/finetune-subset/"
_TRAINING_PREP_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/prepared/cara-strong-v0.4/"
_TRAINING_MUSICGEN_TOKEN_CACHE_URI = f"{_TRAINING_PREP_OUTPUT_URI}musicgen_encodec_cache/"
_TRAINING_MUSICGEN_PREFLIGHT_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/musicgen_preflight/"
_TRAINING_MUSICGEN_SMOKE_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/musicgen_smoke/"
_TRAINING_MUSICGEN_FULL_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/musicgen_full/"
_TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION = "real_audiocraft_musicgen_lm"
_TRAINING_STABLE_AUDIO_PREFLIGHT_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_preflight/"
_TRAINING_STABLE_AUDIO_SMOKE_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_smoke/"
_TRAINING_STABLE_AUDIO_FULL_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_full/"
_TRAINING_CONTEXT_ROOT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_context/"
_TRAINING_CONTEXT_PACK_OUTPUT_URI = f"{_TRAINING_CONTEXT_ROOT_URI}context_packs/"
_TRAINING_CONTEXT_CACHE_OUTPUT_URI = f"{_TRAINING_CONTEXT_ROOT_URI}context_cache/"
_TRAINING_CONTEXT_PREFLIGHT_OUTPUT_URI = f"{_TRAINING_CONTEXT_ROOT_URI}context_preflight/"
_TRAINING_CONTEXT_SMOKE_OUTPUT_URI = f"{_TRAINING_CONTEXT_ROOT_URI}context_smoke/"
_TRAINING_CONTEXT_FULL_OUTPUT_URI = f"{_TRAINING_CONTEXT_ROOT_URI}context_full/"
_TRAINING_ACE_PREFLIGHT_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/ace_step_preflight/"
_EVALUATION_STABLE_AUDIO_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/evaluation-runs/cara-strong-v0.4/stable_audio_benchmark_testing/"
_EVALUATION_MUSICGEN_OUTPUT_URI = "azureml://datastores/ds_cara_raw_audio/paths/evaluation-runs/cara-strong-v0.4/musicgen_benchmark_testing/"
_EVALUATION_AUDIO_BENCHMARK_CONFIRMATION = "LAUNCH AUDIO BENCHMARK"
_EVALUATION_AUDIO_BENCHMARK_LEGACY_CONFIRMATION = "LAUNCH ALL MODELS AUDIO BENCHMARK"
_EVALUATION_AUDIO_BENCHMARK_CONFIRMATIONS = {
    _EVALUATION_AUDIO_BENCHMARK_CONFIRMATION,
    _EVALUATION_AUDIO_BENCHMARK_LEGACY_CONFIRMATION,
    "LAUNCH DIFFUSION AUDIO BENCHMARK",
}
_EVALUATION_STABLE_AUDIO_TRAINED_JOB_NAME = "modest_arch_clgnkqrz4z"
_EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI = "azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_full/cara-finetune-001-cara-strong-full-20260607-011523/"
_TRAINING_STABLE_AUDIO_ENVIRONMENT = "azureml:env-stable-audio-tools:8"
_TRAINING_MUSICGEN_ENVIRONMENT = "azureml:env-musicgen-audiocraft:3"
_TRAINING_ACE_ENVIRONMENT = "azureml:env-ace-step:1"
_TRAINING_H100_COMPUTE = "gpu-smoke-h100"
_TRAINING_FULL_H100_COMPUTE = "gpu-fulltraining-h100"
_TRAINING_H100_COMPUTES = (_TRAINING_H100_COMPUTE, _TRAINING_FULL_H100_COMPUTE)
_TRAINING_CPU_COMPUTE = "cpu-prep-cluster"
_AZUREML_ACTIVE_STATUSES = {"notstarted", "queued", "preparing", "starting", "provisioning", "running", "finalizing"}
_GENERATED_AUDIO_JOB_ACTIONS = {
    "benchmark_testing_stable_audio_audio_submitted",
    "benchmark_testing_musicgen_audio_submitted",
    "benchmark_testing_audio_group_submitted",
}
_GENERATED_AUDIO_LEAF_JOB_ACTIONS = {
    "benchmark_testing_stable_audio_audio_submitted",
    "benchmark_testing_musicgen_audio_submitted",
}
_ATTRIBUTION_SCORE_JOB_ACTIONS = {
    "benchmark_testing_stable_audio_score_submitted",
    "benchmark_testing_musicgen_score_submitted",
    "benchmark_testing_attribution_score_group_submitted",
}
_ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS = {
    "benchmark_testing_stable_audio_score_submitted",
    "benchmark_testing_musicgen_score_submitted",
}
_EVALUATION_SCORE_MODEL_FAMILY_BY_ID = {
    "base_stable_audio_open_small": "stable_audio",
    "diffusion_cara_strong_full_modest_arch": "stable_audio",
    "context_diffusion_cara_strong_full": "stable_audio",
    "base_musicgen_small": "musicgen",
    "musicgen_cara_strong_full": "musicgen",
}
_AZUREML_JOB_PROGRESS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TRAINING_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}
_TRAINING_MODEL_PREP_SPECS = {
    "stable_audio_open_small": {"sample_rate": 44100, "channels": 2, "chunk_seconds": 11.88, "min_tail_seconds": 2.0},
    "musicgen": {"sample_rate": 32000, "channels": 1, "chunk_seconds": 30.0, "min_tail_seconds": 5.0},
}


def _evaluation_audio_confirmation_matches(value: Any, *, allow_legacy: bool = False) -> bool:
    phrase = str(value or "").strip()
    if phrase == _EVALUATION_AUDIO_BENCHMARK_CONFIRMATION:
        return True
    return allow_legacy and phrase in _EVALUATION_AUDIO_BENCHMARK_CONFIRMATIONS


_TRAINING_RUN_PROGRESS_ACTIONS = {
    "stable_audio_smoke_trainer_submitted",
    "stable_audio_full_trainer_submitted",
    "stable_audio_context_smoke_submitted",
    "stable_audio_context_full_submitted",
    "musicgen_ar_smoke_trainer_submitted",
    "musicgen_full_trainer_submitted",
}
_TRAINING_RUN_PROGRESS_MODEL_LABELS = {
    "stable_audio_open_small": "Diffusion",
    "stable_audio_open_small_context_diffusion": "Context Diffusion",
    "musicgen": "Autoregressive",
    "ace_step": "Hybrid",
}


def _training_read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _training_read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def _training_row_duration(row: dict[str, Any]) -> float:
    for key in ("duration_sec", "duration_seconds", "api_current_duration_s"):
        value = row.get(key)
        try:
            if value is not None and value != "":
                return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return 0.0


def _training_pool_id(row: dict[str, Any]) -> str:
    return str(row.get("cara_v2_source_pool_id") or row.get("cara_source_pool_id") or row.get("cara_pool_id") or "")


def _training_chunk_ranges(duration: float, chunk_seconds: float, min_tail_seconds: float) -> list[tuple[float, float, int]]:
    if duration <= 0:
        return [(0.0, chunk_seconds, 0)]
    chunks: list[tuple[float, float, int]] = []
    start = 0.0
    index = 0
    while start + chunk_seconds <= duration:
        chunks.append((start, chunk_seconds, index))
        start += chunk_seconds
        index += 1
    remaining = duration - start
    if not chunks:
        chunks.append((0.0, min(duration, chunk_seconds), 0))
    elif remaining >= min_tail_seconds:
        chunks.append((start, remaining, index))
    return chunks


def _training_expected_preprocess_plan(model_key: str) -> dict[str, Any]:
    if model_key not in _TRAINING_MODEL_PREP_SPECS:
        raise HTTPException(status_code=400, detail=f"Unsupported preprocessing progress model: {model_key}")
    spec = _TRAINING_MODEL_PREP_SPECS[model_key]
    manifest_path = ROOT / "data" / "cara_pool_manifest_v2.jsonl"
    rows = _training_read_jsonl(manifest_path)
    expected_chunks = 0
    expected_duration_seconds = 0.0
    valid_source_rows = 0
    rejected_source_rows = 0
    reject_reasons: dict[str, int] = {}
    for row in rows:
        reason = None
        if row.get("download_status") not in {None, "", "downloaded"}:
            reason = f"download_status:{row.get('download_status')}"
        elif not _training_pool_id(row):
            reason = "missing_pool_id"
        else:
            raw = row.get("local_audio_path") or row.get("source_file_path")
            audio_path = ROOT / str(raw or "")
            if not raw or not audio_path.exists():
                reason = "audio_file_missing"
            elif audio_path.suffix.lower() not in _TRAINING_AUDIO_SUFFIXES:
                reason = "unsupported_audio_extension"
        if reason:
            rejected_source_rows += 1
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
        valid_source_rows += 1
        for _, chunk_duration, _ in _training_chunk_ranges(
            _training_row_duration(row),
            float(spec["chunk_seconds"]),
            float(spec["min_tail_seconds"]),
        ):
            expected_chunks += 1
            expected_duration_seconds += chunk_duration
    return {
        "model": model_key,
        "manifest_path": str(manifest_path),
        "source_rows": len(rows),
        "valid_source_rows": valid_source_rows,
        "rejected_source_rows": rejected_source_rows,
        "reject_reasons": reject_reasons,
        "expected_chunks": expected_chunks,
        "expected_duration_seconds": round(expected_duration_seconds, 3),
        "expected_duration_hours": round(expected_duration_seconds / 3600.0, 3),
        "chunk_seconds": spec["chunk_seconds"],
        "min_tail_seconds": spec["min_tail_seconds"],
        "sample_rate": spec["sample_rate"],
        "channels": spec["channels"],
        "bytes_per_second": int(spec["sample_rate"]) * int(spec["channels"]) * 2,
    }


def _training_prepared_audio_prefix(model_key: str) -> str:
    return f"prepared/cara-strong-v0.4/{model_key}/audio/"


def _training_parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _training_latest_preprocess_event(model_key: str) -> dict[str, Any] | None:
    for event in reversed(_training_job_registry_events(500)):
        if event.get("action") == "training_preprocess_submitted" and event.get("model_family") == model_key:
            return event
    return None


def _training_latest_preprocess_jobs(*, registry_limit: int = 200) -> dict[str, Any]:
    status_by_job: dict[str, dict[str, Any]] = {}
    for event in _training_job_registry_events(registry_limit):
        if event.get("action") != "training_preprocess_status_observed":
            continue
        job_name = str(event.get("job_name") or "").strip()
        if job_name:
            status_by_job[job_name] = event

    latest_by_model: dict[str, Any] = {}
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "training_preprocess_submitted":
            continue
        model_key = str(event.get("model_family") or "").strip()
        if model_key not in _TRAINING_MODEL_PREP_SPECS or model_key in latest_by_model:
            continue
        job_name = str(event.get("job_name") or "").strip()
        observed = status_by_job.get(job_name, {})
        status = str(observed.get("status") or "").lower()
        active = status in _AZUREML_ACTIVE_STATUSES
        passed = status == "completed"
        if passed:
            reason = f"{model_key} preprocessing job {job_name} completed in Azure ML."
        elif active:
            reason = f"{model_key} preprocessing job {job_name} is {observed.get('status')}; use Check Prep Progress for blob-count progress."
        elif observed.get("status"):
            reason = f"Latest {model_key} preprocessing job {job_name} ended with status {observed.get('status')}."
        elif job_name:
            reason = f"Latest {model_key} preprocessing job {job_name} has been submitted; use Check Prep Progress or Operations / Azure Runs to observe completion."
        else:
            reason = f"Latest {model_key} preprocessing event does not include an Azure ML job name."
        latest_by_model[model_key] = {
            "passed": passed,
            "active": active,
            "reason": reason,
            "latest_job": {
                **event,
                **observed,
                "output_path": event.get("output_path"),
                "model_family": model_key,
            },
        }
    return latest_by_model


def _azureml_datastore_blob_list(prefix: str) -> list[dict[str, Any]]:
    settings = _azureml_settings()
    datastore = _azureml_client().datastores.get(settings["datastore_name"])
    account_name = getattr(datastore, "account_name", None)
    container_name = getattr(datastore, "container_name", None)
    if not account_name or not container_name:
        raise HTTPException(status_code=503, detail="Azure ML datastore does not expose blob account/container metadata.")
    key_command = [
        "az",
        "storage",
        "account",
        "keys",
        "list",
        "--resource-group",
        settings["resource_group"],
        "--account-name",
        str(account_name),
        "--only-show-errors",
        "--query",
        "[0].value",
        "--output",
        "tsv",
    ]
    completed = subprocess.run(key_command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise HTTPException(status_code=503, detail=f"Unable to read Azure storage account key: {completed.stderr.strip() or completed.stdout.strip()}")
    account_key = completed.stdout.strip()
    if not account_key:
        raise HTTPException(status_code=503, detail="Azure storage account key lookup returned an empty key.")
    try:
        from azure.storage.blob import ContainerClient

        container = ContainerClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            container_name=str(container_name),
            credential=account_key,
        )
        return [
            {
                "name": blob.name,
                "size": getattr(blob, "size", None),
                "last_modified": _azureml_iso(getattr(blob, "last_modified", None)),
            }
            for blob in container.list_blobs(name_starts_with=prefix)
        ]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to list Azure prepared blobs: {exc}") from exc


def _azureml_datastore_blob_text(prefix: str) -> str:
    settings = _azureml_settings()
    datastore = _azureml_client().datastores.get(settings["datastore_name"])
    account_name = getattr(datastore, "account_name", None)
    container_name = getattr(datastore, "container_name", None)
    if not account_name or not container_name:
        raise HTTPException(status_code=503, detail="Azure ML datastore does not expose blob account/container metadata.")
    key_command = [
        "az",
        "storage",
        "account",
        "keys",
        "list",
        "--resource-group",
        settings["resource_group"],
        "--account-name",
        str(account_name),
        "--only-show-errors",
        "--query",
        "[0].value",
        "--output",
        "tsv",
    ]
    completed = subprocess.run(key_command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise HTTPException(status_code=503, detail=f"Unable to read Azure storage account key: {completed.stderr.strip() or completed.stdout.strip()}")
    account_key = completed.stdout.strip()
    if not account_key:
        raise HTTPException(status_code=503, detail="Azure storage account key lookup returned an empty key.")
    try:
        from azure.storage.blob import ContainerClient

        container = ContainerClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            container_name=str(container_name),
            credential=account_key,
        )
        return container.get_blob_client(prefix).download_blob().readall().decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to read Azure blob {prefix}: {exc}") from exc


def _training_preprocess_progress(model_key: str) -> dict[str, Any]:
    plan = _training_expected_preprocess_plan(model_key)
    prefix = _training_prepared_audio_prefix(model_key)
    checked_at_dt = datetime.now(timezone.utc)
    blobs = _azureml_datastore_blob_list(prefix)
    wav_blobs = [
        blob
        for blob in blobs
        if str(blob.get("name") or "").lower().endswith(".wav") and int(blob.get("size") or 0) > 0
    ]
    bytes_per_second = max(1, int(plan["bytes_per_second"]))
    completed_duration_seconds = sum(max(0, int(blob.get("size") or 0) - 44) / bytes_per_second for blob in wav_blobs)
    completed_chunks = len(wav_blobs)
    expected_chunks = max(1, int(plan["expected_chunks"]))
    expected_duration_seconds = max(0.001, float(plan["expected_duration_seconds"]))
    latest_blob = max((str(blob.get("last_modified") or "") for blob in wav_blobs), default=None)
    chunk_percent = min(100.0, completed_chunks / expected_chunks * 100.0)
    duration_percent = min(100.0, completed_duration_seconds / expected_duration_seconds * 100.0)
    latest_event = _training_latest_preprocess_event(model_key)
    started_at = _training_parse_datetime((latest_event or {}).get("created_at"))
    elapsed_seconds = max(0.0, (checked_at_dt - started_at).total_seconds()) if started_at else None
    progress_ratio = max(completed_chunks / expected_chunks, completed_duration_seconds / expected_duration_seconds)
    estimated_total_seconds = elapsed_seconds / progress_ratio if elapsed_seconds is not None and progress_ratio > 0 else None
    estimated_remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds) if estimated_total_seconds is not None else None
    return {
        "model": model_key,
        "checked_at": checked_at_dt.isoformat(),
        "method": "read_only_azure_blob_count",
        "datastore_prefix": prefix,
        "job": {
            "job_name": (latest_event or {}).get("job_name"),
            "studio_url": (latest_event or {}).get("studio_url"),
            "compute": (latest_event or {}).get("compute"),
            "submitted_at": (latest_event or {}).get("created_at"),
        },
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "estimated_total_seconds": round(estimated_total_seconds, 3) if estimated_total_seconds is not None else None,
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 3) if estimated_remaining_seconds is not None else None,
        "expected": plan,
        "completed_chunks": completed_chunks,
        "completed_duration_seconds": round(completed_duration_seconds, 3),
        "completed_duration_hours": round(completed_duration_seconds / 3600.0, 3),
        "chunk_percent": round(chunk_percent, 2),
        "duration_percent": round(duration_percent, 2),
        "remaining_chunks_estimate": max(0, int(plan["expected_chunks"]) - completed_chunks),
        "latest_blob_modified": latest_blob,
        "note": "Estimate is based on prepared WAV blobs already visible in the datastore and elapsed time since dashboard submission. It does not affect the running Azure ML job.",
    }


def _training_test_prep_phase(job_name: str | None) -> dict[str, Any]:
    if not job_name:
        return {"status": "missing", "job_name": None}
    payload = _training_read_json(ROOT / "registry" / "azureml_test_prep" / "reports" / f"{job_name}.json", {}) or {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    return {
        "status": metadata.get("status") or report.get("status") or "missing",
        "job_name": job_name,
        "test_name": metadata.get("test_name") or report.get("test_name"),
        "compute": metadata.get("compute"),
        "environment": metadata.get("environment"),
        "gpu_name": metadata.get("gpu_name") or report.get("gpu_name"),
        "cuda_available": metadata.get("cuda_available", report.get("cuda_available")),
        "import_status": metadata.get("import_status") or report.get("import_status"),
    }


def _training_lock_state() -> dict[str, Any]:
    summary = _training_read_json(_TRAINING_LOCK_DIR / "lock_summary.json", None)
    receipt = _training_read_json(_TRAINING_LOCK_DIR / "training_inclusion_receipt.json", None)
    split_manifest = _training_read_json(_TRAINING_LOCK_DIR / "split_manifest.json", None)
    locked_manifest = _TRAINING_LOCK_DIR / "manifest.locked.jsonl"
    locked_registry = _TRAINING_LOCK_DIR / "pool_registry.locked.json"
    return {
        "locked": bool(summary and locked_manifest.exists() and locked_registry.exists()),
        "output_dir": str(_TRAINING_LOCK_DIR),
        "summary": summary,
        "receipt": receipt,
        "split_manifest": split_manifest,
        "paths": {
            "locked_manifest": str(locked_manifest),
            "pool_registry": str(locked_registry),
            "split_manifest": str(_TRAINING_LOCK_DIR / "split_manifest.json"),
            "training_inclusion_receipt": str(_TRAINING_LOCK_DIR / "training_inclusion_receipt.json"),
            "rejected_samples": str(_TRAINING_LOCK_DIR / "rejected_samples.jsonl"),
        },
    }


def _training_azure_upload_state() -> dict[str, Any]:
    payload = _training_read_json(_TRAINING_AZURE_UPLOAD_CONFIRMATION, None)
    return {
        "confirmed": bool(payload and payload.get("confirmed")),
        "confirmed_at": payload.get("confirmed_at") if isinstance(payload, dict) else None,
        "confirmed_by": payload.get("confirmed_by") if isinstance(payload, dict) else None,
        "source_root": _TRAINING_SOURCE_URI,
        "expected_manifest": f"{_TRAINING_SOURCE_URI}data/cara_pool_manifest_v2.jsonl",
        "expected_audio_root": f"{_TRAINING_SOURCE_URI}data/freesound/",
        "confirmation_path": str(_TRAINING_AZURE_UPLOAD_CONFIRMATION),
    }


def _training_job_registry_events(limit: int = 50) -> list[dict[str, Any]]:
    if not _TRAINING_JOB_REGISTRY.exists():
        return []
    events = []
    for line in _TRAINING_JOB_REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-max(1, min(limit, 500)):]


def _training_latest_context_full_output_path() -> str:
    for event in reversed(_training_job_registry_events(500)):
        if (
            event.get("action") == "stable_audio_context_full_submitted"
            and event.get("model_family") == "stable_audio_open_small_context_diffusion"
            and event.get("output_path")
        ):
            return str(event["output_path"])
    return _TRAINING_CONTEXT_FULL_OUTPUT_URI


def _training_append_job_event(event: dict[str, Any]) -> None:
    _TRAINING_JOB_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with _TRAINING_JOB_REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _evaluation_append_job_event(event: dict[str, Any]) -> None:
    _EVALUATION_JOB_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with _EVALUATION_JOB_REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _evaluation_stage_info(action: str) -> dict[str, str]:
    if action in {
        "benchmark_testing_stable_audio_audio_submitted",
        "benchmark_testing_musicgen_audio_submitted",
        "benchmark_testing_audio_group_submitted",
    }:
        return {
            "stage_label": "Generated-audio benchmark",
            "submitted": "Latest generated-audio benchmark job has been submitted; Azure status has not been refreshed.",
            "running": "Generated-audio benchmark is running in Azure ML.",
            "completed": "Generated-audio benchmark completed; inspect generation_manifest.jsonl and benchmark_audio_metrics.json.",
            "failed": "Generated-audio benchmark did not complete; inspect Azure logs before continuing.",
        }
    if action in {
        "benchmark_testing_stable_audio_score_submitted",
        "benchmark_testing_musicgen_score_submitted",
        "benchmark_testing_attribution_score_group_submitted",
    }:
        return {
            "stage_label": "Attribution scoring",
            "submitted": "Latest attribution scoring job has been submitted; Azure status has not been refreshed.",
            "running": "Attribution scoring is running in Azure ML.",
            "completed": "Attribution scoring completed; inspect metrics_latest.json for matrix-ready values or missing-prediction diagnostics.",
            "failed": "Attribution scoring did not complete; inspect Azure logs before continuing.",
        }
    return {
        "stage_label": "Benchmark setup",
        "submitted": "Latest benchmark testing setup job has been submitted; Azure status has not been refreshed.",
        "running": "Benchmark setup is running in Azure ML.",
        "completed": "Benchmark setup completed. Generated-audio scoring is still pending.",
        "failed": "Benchmark setup did not complete; inspect Azure logs before continuing.",
    }


def _latest_evaluation_job_state() -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    if not events:
        return None
    event = events[-1]
    action = str(event.get("action") or "")
    stage_info = _evaluation_stage_info(action)
    result: dict[str, Any] = {
        "job_name": event.get("job_name"),
        "created_at": event.get("created_at"),
        "action": event.get("action"),
        "status": "unknown",
        "stage_label": stage_info["stage_label"],
        "message": stage_info["submitted"],
        "studio_url": event.get("studio_url"),
        "output_path": event.get("output_path"),
        "model_ids": event.get("model_ids") or [],
        "suite_ids": event.get("suite_ids") or [],
        "seeds": event.get("seeds") if event.get("seeds") is not None else event.get("seed_ids"),
    }
    job_name = str(event.get("job_name") or "")
    if not job_name:
        return result
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
        status = str(summary.get("status") or "unknown")
        result.update(
            {
                "status": status,
                "azure": summary,
                "studio_url": summary.get("studio_url") or event.get("studio_url"),
            }
        )
        status_lower = status.lower()
        if status_lower == "completed":
            result["message"] = stage_info["completed"]
        elif status_lower in _AZUREML_ACTIVE_STATUSES:
            result["message"] = stage_info["running"]
        elif status_lower in {"failed", "canceled", "cancelled"}:
            result["message"] = stage_info["failed"]
    except Exception as exc:
        result["heartbeat_error"] = str(exc)
    return result


def _evaluation_prompt_manifest_uri(output_path: Any) -> str | None:
    text = str(output_path or "").strip()
    if not text:
        return None
    return f"{text.rstrip('/')}/prompt_manifest.jsonl"


def _evaluation_prompt_set_state(latest_job: dict[str, Any] | None = None) -> dict[str, Any]:
    from evaluation.benchmark_spec import prompt_set_summary

    lock = _training_read_json(_EVALUATION_PROMPT_SET_LOCK, None)
    latest = latest_job if latest_job is not None else _latest_evaluation_job_state()
    candidate_uri = _evaluation_prompt_manifest_uri((latest or {}).get("output_path"))
    latest_status = str((latest or {}).get("status") or "").lower()
    can_lock = bool(latest and latest_status == "completed" and candidate_uri)
    if lock:
        v2_summary = prompt_set_summary(lock)
        return {
            **v2_summary,
            "locked": True,
            "lock_path": str(_EVALUATION_PROMPT_SET_LOCK),
            "prompt_manifest_uri": lock.get("prompt_manifest_uri"),
            "legacy_prompt_manifest_uri": lock.get("prompt_manifest_uri"),
            "source_job_name": lock.get("source_job_name"),
            "source_output_path": lock.get("source_output_path"),
            "suite_ids": lock.get("suite_ids") or [],
            "model_ids": lock.get("model_ids") or [],
            "seeds": lock.get("seeds"),
            "locked_at": lock.get("locked_at"),
            "created_by": lock.get("created_by", "dashboard"),
            "can_lock": False,
            "reason": (
                "Benchmark Prompt Set v2 framework is active. The existing v1 prompt manifest is preserved as "
                "historical evidence and remains the mounted prompt URI until a v2 Azure setup job materializes rows."
            ),
        }
    v2_summary = prompt_set_summary(None)
    return {
        **v2_summary,
        "locked": False,
        "lock_path": str(_EVALUATION_PROMPT_SET_LOCK),
        "prompt_manifest_uri": candidate_uri,
        "source_job_name": (latest or {}).get("job_name"),
        "source_output_path": (latest or {}).get("output_path"),
        "suite_ids": (latest or {}).get("suite_ids") or [],
        "model_ids": (latest or {}).get("model_ids") or [],
        "seeds": (latest or {}).get("seeds"),
        "can_lock": can_lock,
        "reason": (
            "Latest benchmark setup completed; lock its prompt_manifest.jsonl before any generated-audio scoring run."
            if can_lock
            else "Run and complete benchmark setup before locking a canonical prompt set."
        ),
    }


def _evaluation_job_event_by_name(job_name: str | None) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    if not events:
        return None
    if not job_name:
        return events[-1]
    for event in reversed(events):
        if str(event.get("job_name") or "") == str(job_name):
            return event
    return None


def _evaluation_job_state_from_event(event: dict[str, Any]) -> dict[str, Any]:
    action = str(event.get("action") or "")
    stage_info = _evaluation_stage_info(action)
    result: dict[str, Any] = {
        "job_name": event.get("job_name"),
        "created_at": event.get("created_at"),
        "action": event.get("action"),
        "status": "unknown",
        "active": False,
        "stage_label": stage_info["stage_label"],
        "message": stage_info["submitted"],
        "studio_url": event.get("studio_url"),
        "output_path": event.get("output_path"),
        "model_ids": event.get("model_ids") or [],
        "suite_ids": event.get("suite_ids") or [],
        "seeds": event.get("seeds") if event.get("seeds") is not None else event.get("seed_ids"),
        "scope": event.get("scope"),
        "max_prompts": event.get("max_prompts"),
        "prompt_manifest_uri": event.get("prompt_manifest_uri"),
    }
    job_name = str(event.get("job_name") or "")
    if not job_name:
        return result
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
        status = str(summary.get("status") or "unknown")
        status_lower = status.lower()
        result.update(
            {
                "status": status,
                "active": status_lower in _AZUREML_ACTIVE_STATUSES,
                "azure": summary,
                "studio_url": summary.get("studio_url") or event.get("studio_url"),
            }
        )
        if status_lower == "completed":
            result["message"] = stage_info["completed"]
        elif status_lower in _AZUREML_ACTIVE_STATUSES:
            result["message"] = stage_info["running"]
        elif status_lower in {"failed", "canceled", "cancelled"}:
            result["message"] = stage_info["failed"]
    except Exception as exc:
        result["heartbeat_error"] = str(exc)
    return result


def _evaluation_job_state_from_event_fast(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    action = str(event.get("action") or "")
    stage_info = _evaluation_stage_info(action)
    status = str(event.get("status") or "recorded")
    child_jobs = event.get("child_jobs") if isinstance(event.get("child_jobs"), list) else []
    child_output_paths = {
        str(child.get("family") or ""): child.get("output_path")
        for child in child_jobs
        if isinstance(child, dict) and child.get("family") and child.get("output_path")
    }
    if event.get("output_path") and not child_output_paths and action in _GENERATED_AUDIO_LEAF_JOB_ACTIONS:
        family = "musicgen" if action == "benchmark_testing_musicgen_audio_submitted" else "stable_audio"
        child_output_paths[family] = event.get("output_path")
    return {
        "job_name": event.get("job_name"),
        "created_at": event.get("created_at"),
        "action": event.get("action"),
        "status": status,
        "active": False,
        "stage_label": stage_info["stage_label"],
        "message": event.get("message") or stage_info.get(status.lower()) or stage_info["submitted"],
        "studio_url": event.get("studio_url"),
        "output_path": event.get("output_path"),
        "model_ids": event.get("model_ids") or [],
        "suite_ids": event.get("suite_ids") or [],
        "seeds": event.get("seeds") if event.get("seeds") is not None else event.get("seed_ids"),
        "scope": event.get("scope"),
        "max_prompts": event.get("max_prompts"),
        "prompt_manifest_uri": event.get("prompt_manifest_uri"),
        "source_audio_job_name": event.get("source_audio_job_name"),
        "generated_audio_output_path": event.get("generated_audio_output_path"),
        "generated_audio_output_paths": child_output_paths,
        "generation_manifest_uri": event.get("generation_manifest_uri"),
        "metrics_uri": event.get("metrics_uri") or (
            f"{str(event.get('output_path') or '').rstrip('/')}/metrics_latest.json"
            if action in _ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS and event.get("output_path")
            else None
        ),
        "child_jobs": child_jobs,
    }


def _latest_evaluation_job_state_fast(
    action: str | None = None,
    *,
    scope: str | None = None,
    source_audio_job_name: str | None = None,
) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    for event in reversed(events):
        if action and str(event.get("action") or "") != action:
            continue
        if scope is not None and str(event.get("scope") or "") != scope:
            continue
        if source_audio_job_name is not None and str(event.get("source_audio_job_name") or "") != source_audio_job_name:
            continue
        return _evaluation_job_state_from_event_fast(event)
    return None


def _latest_generated_audio_job_state_fast(*, scope: str | None = None) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    for event in reversed(events):
        if str(event.get("action") or "") not in _GENERATED_AUDIO_JOB_ACTIONS:
            continue
        if scope is not None and str(event.get("scope") or "") != scope:
            continue
        return _evaluation_job_state_from_event_fast(event)
    return None


def _latest_attribution_score_job_state_fast(*, source_audio_job_name: str | None = None) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    if source_audio_job_name is not None:
        child_jobs: list[dict[str, Any]] = []
        seen_families: set[str] = set()
        for event in reversed(events):
            action = str(event.get("action") or "")
            if action not in _ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS:
                continue
            if str(event.get("source_audio_job_name") or "") != source_audio_job_name:
                continue
            family = "musicgen" if action == "benchmark_testing_musicgen_score_submitted" else "stable_audio"
            if family in seen_families:
                continue
            seen_families.add(family)
            child_jobs.append(
                {
                    "family": family,
                    "job_name": event.get("job_name"),
                    "metrics_uri": event.get("metrics_uri")
                    or f"{str(event.get('output_path') or '').rstrip('/')}/metrics_latest.json",
                    "output_path": event.get("output_path"),
                    "studio_url": event.get("studio_url"),
                }
            )
            if seen_families == {"stable_audio", "musicgen"}:
                break
        if child_jobs:
            latest_child_event = next(
                (
                    event
                    for event in reversed(events)
                    if str(event.get("source_audio_job_name") or "") == source_audio_job_name
                    and str(event.get("action") or "") in _ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS
                ),
                {},
            )
            return {
                "job_name": f"score-all-{source_audio_job_name}",
                "created_at": latest_child_event.get("created_at"),
                "action": "benchmark_testing_attribution_score_group_submitted",
                "status": "recorded",
                "active": False,
                "stage_label": "Attribution scoring",
                "message": "Latest family-specific scorer artifacts are grouped for benchmark comparison.",
                "studio_url": (child_jobs[0] or {}).get("studio_url"),
                "output_path": None,
                "metrics_uri": None,
                "source_audio_job_name": source_audio_job_name,
                "child_jobs": child_jobs,
            }
    for event in reversed(events):
        if str(event.get("action") or "") not in _ATTRIBUTION_SCORE_JOB_ACTIONS:
            continue
        if source_audio_job_name is not None and str(event.get("source_audio_job_name") or "") != source_audio_job_name:
            continue
        return _evaluation_job_state_from_event_fast(event)
    return None


def _active_generated_audio_job_state(limit: int = 50) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    for event in reversed(events[-max(1, min(limit, 500)) :]):
        if str(event.get("action") or "") not in _GENERATED_AUDIO_LEAF_JOB_ACTIONS:
            continue
        state = _evaluation_job_state_from_event(event)
        if state.get("active"):
            return state
    return None


def _active_attribution_score_job_state(limit: int = 50) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    for event in reversed(events[-max(1, min(limit, 500)) :]):
        if str(event.get("action") or "") not in _ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS:
            continue
        state = _evaluation_job_state_from_event(event)
        if state.get("active"):
            return state
    return None


def _active_evaluation_job_state(action: str | None = None, limit: int = 50) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    if not events:
        return None
    for event in reversed(events[-max(1, min(limit, 500)) :]):
        if action and str(event.get("action") or "") != action:
            continue
        state = _evaluation_job_state_from_event(event)
        if state.get("active"):
            return state
    return None


def _latest_completed_evaluation_job_state(action: str | None = None, limit: int = 100) -> dict[str, Any] | None:
    events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
    if not events:
        return None
    for event in reversed(events[-max(1, min(limit, 500)) :]):
        if action and str(event.get("action") or "") != action:
            continue
        state = _evaluation_job_state_from_event(event)
        if str(state.get("status") or "").lower() == "completed":
            return state
    return None


def _azureml_datastore_prefix_from_uri(uri: Any) -> str | None:
    text = str(uri or "").strip()
    marker = "azureml://datastores/"
    if not text.startswith(marker):
        return None
    remainder = text[len(marker) :]
    if "/paths/" not in remainder:
        return None
    datastore_name, prefix = remainder.split("/paths/", 1)
    settings = _azureml_settings()
    if datastore_name and datastore_name != settings["datastore_name"]:
        return None
    return prefix.lstrip("/")


def _generated_audio_artifact_summary(output_path: Any) -> dict[str, Any]:
    prefix = _azureml_datastore_prefix_from_uri(output_path)
    if not prefix:
        return {
            "available": False,
            "reason": "Output path is not an Azure ML datastore path that can be inspected.",
        }
    try:
        blobs = _azureml_datastore_blob_list(prefix)
    except Exception as exc:
        return {
            "available": False,
            "prefix": prefix,
            "reason": str(exc),
        }
    names = [str(blob.get("name") or "") for blob in blobs]
    wavs = [name for name in names if name.lower().endswith(".wav")]
    manifests = {
        "generation_manifest": any(name.endswith("generation_manifest.jsonl") for name in names),
        "audio_metrics": any(name.endswith("benchmark_audio_metrics.json") for name in names),
        "audio_plan": any(name.endswith("benchmark_audio_plan.json") for name in names),
        "report": any(name.endswith("benchmark_testing_stable_audio_audio_report.json") or name.endswith("benchmark_testing_musicgen_audio_report.json") for name in names),
        "metadata": any(name.endswith("metadata.json") for name in names),
    }
    latest_blob = max((str(blob.get("last_modified") or "") for blob in blobs), default=None)
    return {
        "available": True,
        "prefix": prefix,
        "blob_count": len(blobs),
        "wav_count": len(wavs),
        "manifest_count": sum(1 for present in manifests.values() if present),
        "manifests": manifests,
        "latest_blob_at": latest_blob,
        "generation_manifest_uri": f"{str(output_path).rstrip('/')}/generation_manifest.jsonl",
        "metrics_uri": f"{str(output_path).rstrip('/')}/benchmark_audio_metrics.json",
    }


def _generated_audio_progress_state(job_name: str | None = None) -> dict[str, Any]:
    event = _evaluation_job_event_by_name(job_name) if job_name else None
    if event is None:
        events = _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
        for candidate in reversed(events):
            if str(candidate.get("action") or "") in _GENERATED_AUDIO_JOB_ACTIONS:
                event = candidate
                break
    if event and str(event.get("action") or "") == "benchmark_testing_audio_group_submitted":
        child_progress = []
        for child in event.get("child_jobs") or []:
            child_name = child.get("job_name") if isinstance(child, dict) else None
            if child_name:
                try:
                    child_progress.append(_generated_audio_progress_state(str(child_name)))
                except HTTPException:
                    continue
        model_ids = list(event.get("model_ids") or [])
        suite_ids = list(event.get("suite_ids") or [])
        seed_ids = event.get("seed_ids") if event.get("seed_ids") is not None else event.get("seeds")
        if not isinstance(seed_ids, list):
            seed_ids = [seed_ids] if seed_ids is not None else [0]
        planned_values = [item.get("planned_generations") for item in child_progress]
        planned_generations = sum(int(value) for value in planned_values if isinstance(value, int)) if child_progress and all(isinstance(value, int) for value in planned_values) else None
        completed_generations = sum(int(item.get("completed_generations") or 0) for item in child_progress)
        if planned_generations is not None:
            planned_generations = max(planned_generations, completed_generations)
        progress_percent = (
            min(100.0, completed_generations / planned_generations * 100.0)
            if planned_generations
            else max((float(item.get("progress_percent") or 0.0) for item in child_progress), default=0.0)
        )
        by_model: dict[str, dict[str, int]] = {model_id: {suite_id: 0 for suite_id in suite_ids} for model_id in model_ids}
        model_counts: dict[str, dict[str, Any]] = {model_id: {"model_id": model_id, "completed": 0, "planned": 0} for model_id in model_ids}
        suite_counts: dict[str, dict[str, Any]] = {suite_id: {"suite_id": suite_id, "completed": 0, "planned": 0} for suite_id in suite_ids}
        latest_completed_item = None
        for item in child_progress:
            for row in item.get("model_progress") or []:
                model_id = str(row.get("model_id") or "")
                model_counts.setdefault(model_id, {"model_id": model_id, "completed": 0, "planned": 0})
                model_counts[model_id]["completed"] += int(row.get("completed") or 0)
                if row.get("planned") is not None:
                    model_counts[model_id]["planned"] += int(row.get("planned") or 0)
            for row in item.get("suite_progress") or []:
                suite_id = str(row.get("suite_id") or "")
                suite_counts.setdefault(suite_id, {"suite_id": suite_id, "completed": 0, "planned": 0})
                suite_counts[suite_id]["completed"] += int(row.get("completed") or 0)
                if row.get("planned") is not None:
                    suite_counts[suite_id]["planned"] += int(row.get("planned") or 0)
            for model_id, suite_map in (item.get("by_model_suite") or {}).items():
                by_model.setdefault(model_id, {suite_id: 0 for suite_id in suite_ids})
                for suite_id, count in suite_map.items():
                    by_model[model_id][suite_id] = by_model[model_id].get(suite_id, 0) + int(count or 0)
            candidate = item.get("latest_completed_item")
            if candidate and (
                latest_completed_item is None
                or str(candidate.get("completed_at") or "") > str(latest_completed_item.get("completed_at") or "")
            ):
                latest_completed_item = candidate
        model_progress = []
        for row in model_counts.values():
            planned = row.get("planned")
            completed = int(row.get("completed") or 0)
            planned_value = max(int(planned), completed) if planned else None
            model_progress.append(
                {
                    "model_id": row.get("model_id"),
                    "completed": completed,
                    "planned": planned_value,
                    "percent": round(min(100.0, completed / planned_value * 100.0), 2) if planned_value else None,
                }
            )
        suite_progress = []
        for row in suite_counts.values():
            planned = row.get("planned")
            completed = int(row.get("completed") or 0)
            planned_value = max(int(planned), completed) if planned else None
            suite_progress.append(
                {
                    "suite_id": row.get("suite_id"),
                    "completed": completed,
                    "planned": planned_value,
                    "percent": round(min(100.0, completed / planned_value * 100.0), 2) if planned_value else None,
                }
            )
        return {
            "format": "cara_audio_benchmark_progress_v2",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "method": "read_only_azure_blob_count",
            "job": _evaluation_job_state_from_event_fast(event),
            "scope": event.get("scope"),
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seed_ids": seed_ids,
            "max_prompts": int(event.get("max_prompts") or 0),
            "planned_generations": planned_generations,
            "completed_generations": completed_generations,
            "progress_percent": round(progress_percent, 2),
            "model_progress": model_progress,
            "suite_progress": suite_progress,
            "by_model_suite": by_model,
            "manifest_available": all(bool(item.get("manifest_available")) for item in child_progress) if child_progress else False,
            "metrics_available": all(bool(item.get("metrics_available")) for item in child_progress) if child_progress else False,
            "report_available": all(bool(item.get("report_available")) for item in child_progress) if child_progress else False,
            "latest_completed_item": latest_completed_item,
            "latest_blob_at": max((str(item.get("latest_blob_at") or "") for item in child_progress), default=None),
            "blob_error": "; ".join(str(item.get("blob_error")) for item in child_progress if item.get("blob_error")) or None,
            "child_jobs": child_progress,
            "note": "This all-model benchmark run uses separate approved Azure ML command jobs per architecture and aggregates their WAV outputs here.",
        }
    if not event or str(event.get("action") or "") not in _GENERATED_AUDIO_LEAF_JOB_ACTIONS:
        raise HTTPException(status_code=404, detail=f"No generated-audio benchmark event found for {job_name or 'latest run'}.")

    state = _evaluation_job_state_from_event_fast(event)
    model_ids = list(event.get("model_ids") or [])
    suite_ids = list(event.get("suite_ids") or [])
    seed_ids = event.get("seed_ids") if event.get("seed_ids") is not None else event.get("seeds")
    if not isinstance(seed_ids, list):
        seed_ids = [seed_ids] if seed_ids is not None else [0]
    max_prompts = int(event.get("max_prompts") or 0)
    suite_prompt_counts = {
        str(suite.get("id") or ""): int(suite.get("prompt_count") or 0)
        for suite in _evaluation_suites()
        if int(suite.get("prompt_count") or 0) > 0
    }
    planned_prompt_rows = sum(suite_prompt_counts.get(str(suite_id), 0) for suite_id in suite_ids)
    planned_generations = (
        max_prompts * len(model_ids) * max(1, len(seed_ids))
        if max_prompts > 0
        else (planned_prompt_rows * len(model_ids) * max(1, len(seed_ids)) if planned_prompt_rows > 0 else None)
    )
    prefix = _azureml_datastore_prefix_from_uri(event.get("output_path"))
    blobs: list[dict[str, Any]] = []
    blob_error = None
    if prefix:
        try:
            blobs = _azureml_datastore_blob_list(prefix)
        except Exception as exc:
            blob_error = str(exc)
    else:
        blob_error = "Output path is not an Azure ML datastore URI."

    blob_names = [str(blob.get("name") or "") for blob in blobs]
    wav_names = [name for name in blob_names if name.lower().endswith(".wav") and "/audio/" in name]
    by_model = {model_id: 0 for model_id in model_ids}
    by_suite = {suite_id: 0 for suite_id in suite_ids}
    by_model_suite: dict[str, dict[str, int]] = {
        model_id: {suite_id: 0 for suite_id in suite_ids}
        for model_id in model_ids
    }
    for name in wav_names:
        rel = name[len(prefix):].lstrip("/") if prefix and name.startswith(prefix) else name
        parts = rel.split("/")
        try:
            audio_index = parts.index("audio")
        except ValueError:
            continue
        model_id = parts[audio_index + 1] if len(parts) > audio_index + 1 else None
        suite_id = parts[audio_index + 2] if len(parts) > audio_index + 2 else None
        if model_id:
            by_model[model_id] = by_model.get(model_id, 0) + 1
        if suite_id:
            by_suite[suite_id] = by_suite.get(suite_id, 0) + 1
        if model_id and suite_id:
            by_model_suite.setdefault(model_id, {}).setdefault(suite_id, 0)
            by_model_suite[model_id][suite_id] += 1
    latest_wav_blob = None
    if wav_names:
        wav_blobs = [blob for blob in blobs if str(blob.get("name") or "") in set(wav_names)]
        latest_wav_blob = max(wav_blobs, key=lambda blob: str(blob.get("last_modified") or ""), default=None)
    latest_completed_item = None
    if latest_wav_blob:
        latest_name = str(latest_wav_blob.get("name") or "")
        rel = latest_name[len(prefix):].lstrip("/") if prefix and latest_name.startswith(prefix) else latest_name
        parts = rel.split("/")
        try:
            audio_index = parts.index("audio")
            latest_completed_item = {
                "model_id": parts[audio_index + 1] if len(parts) > audio_index + 1 else None,
                "suite_id": parts[audio_index + 2] if len(parts) > audio_index + 2 else None,
                "file": parts[-1] if parts else latest_name,
                "blob": latest_name,
                "completed_at": latest_wav_blob.get("last_modified"),
            }
        except ValueError:
            latest_completed_item = {
                "blob": latest_name,
                "completed_at": latest_wav_blob.get("last_modified"),
            }

    manifest_name = next((name for name in blob_names if name.endswith("generation_manifest.jsonl")), None)
    manifest_rows: list[dict[str, Any]] = []
    if manifest_name:
        try:
            manifest_rows = [
                json.loads(line)
                for line in _azureml_datastore_blob_text(manifest_name).splitlines()
                if line.strip()
            ]
        except Exception:
            manifest_rows = []
    if manifest_rows:
        planned_generations = len(manifest_rows)

    completed_generations = len(wav_names)
    progress_percent = (
        min(100.0, completed_generations / planned_generations * 100.0)
        if planned_generations
        else (100.0 if manifest_rows else 0.0)
    )
    model_progress = []
    for model_id in model_ids:
        planned_for_model = (
            max_prompts * max(1, len(seed_ids))
            if max_prompts > 0
            else (planned_prompt_rows * max(1, len(seed_ids)) if planned_prompt_rows > 0 else None)
        )
        if manifest_rows:
            planned_for_model = sum(1 for row in manifest_rows if str(row.get("model_id") or "") == model_id)
        completed_for_model = by_model.get(model_id, 0)
        model_progress.append(
            {
                "model_id": model_id,
                "completed": completed_for_model,
                "planned": max(planned_for_model, completed_for_model) if planned_for_model else None,
                "percent": round(min(100.0, completed_for_model / max(planned_for_model, completed_for_model) * 100.0), 2) if planned_for_model else None,
            }
        )

    suite_progress = []
    for suite_id in suite_ids:
        planned_for_suite = (
            suite_prompt_counts.get(str(suite_id), 0) * len(model_ids) * max(1, len(seed_ids))
            if max_prompts <= 0 and suite_prompt_counts.get(str(suite_id), 0) > 0
            else None
        )
        if manifest_rows:
            planned_for_suite = sum(1 for row in manifest_rows if str(row.get("suite_id") or "") == suite_id)
        completed_for_suite = by_suite.get(suite_id, 0)
        suite_progress.append(
            {
                "suite_id": suite_id,
                "completed": completed_for_suite,
                "planned": max(planned_for_suite, completed_for_suite) if planned_for_suite else None,
                "percent": round(min(100.0, completed_for_suite / max(planned_for_suite, completed_for_suite) * 100.0), 2) if planned_for_suite else None,
            }
        )

    return {
        "format": "cara_audio_benchmark_progress_v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "method": "read_only_azure_blob_count",
        "job": state,
        "scope": event.get("scope"),
        "model_ids": model_ids,
        "suite_ids": suite_ids,
        "seed_ids": seed_ids,
        "max_prompts": max_prompts,
        "planned_generations": planned_generations,
        "completed_generations": completed_generations,
        "progress_percent": round(progress_percent, 2),
        "model_progress": model_progress,
        "suite_progress": suite_progress,
        "by_model_suite": by_model_suite,
        "manifest_available": bool(manifest_name),
        "metrics_available": any(name.endswith("benchmark_audio_metrics.json") for name in blob_names),
        "report_available": any(name.endswith("benchmark_testing_stable_audio_audio_report.json") or name.endswith("benchmark_testing_musicgen_audio_report.json") for name in blob_names),
        "latest_completed_item": latest_completed_item,
        "latest_blob_at": max((str(blob.get("last_modified") or "") for blob in blobs), default=None),
        "blob_error": blob_error,
        "note": "This panel counts generated WAV artifacts in the approved Azure ML datastore for this model-family job.",
    }


def _generated_audio_result_state() -> dict[str, Any] | None:
    state = _latest_generated_audio_job_state_fast(scope="full") or _latest_generated_audio_job_state_fast()
    if not state:
        return None
    seed_ids = state.get("seeds") or []
    if not isinstance(seed_ids, list):
        seed_ids = [seed_ids]
    model_ids = state.get("model_ids") or []
    suite_ids = state.get("suite_ids") or []
    max_prompts = int(state.get("max_prompts") or 0)
    planned_generations = None
    if max_prompts > 0 and isinstance(model_ids, list) and isinstance(seed_ids, list):
        planned_generations = max_prompts * len(model_ids) * max(1, len(seed_ids))
    child_jobs = state.get("child_jobs") if isinstance(state.get("child_jobs"), list) else []
    child_output_paths = {
        str(child.get("family") or ""): child.get("output_path")
        for child in child_jobs
        if isinstance(child, dict) and child.get("family") and child.get("output_path")
    }
    output_path = state.get("output_path")
    if output_path and not child_output_paths:
        family = "musicgen" if state.get("action") == "benchmark_testing_musicgen_audio_submitted" else "stable_audio"
        child_output_paths[family] = output_path
    return {
        **state,
        "audio_artifacts": _generated_audio_artifact_summary(state.get("output_path")),
        "generated_audio_output_paths": child_output_paths,
        "planned_generations": planned_generations,
        "models_tested": len(model_ids) if isinstance(model_ids, list) else None,
        "suites_tested": len(suite_ids) if isinstance(suite_ids, list) else None,
        "result_stage": "generated_audio_completed_attribution_pending",
        "attribution_status": "Generated audio and manifests are available; native CARA/probe attribution scoring is still pending.",
    }


def _latest_attribution_scoring_result_state() -> dict[str, Any] | None:
    state = _latest_completed_evaluation_job_state("benchmark_testing_stable_audio_score_submitted")
    if not state:
        return None
    metrics_uri = f"{str(state.get('output_path') or '').rstrip('/')}/metrics_latest.json"
    state = {
        **state,
        "result_stage": "attribution_scoring_completed",
        "metrics_uri": metrics_uri,
    }
    try:
        prefix = _azureml_datastore_prefix_from_uri(metrics_uri)
        if prefix:
            state["metrics"] = json.loads(_azureml_datastore_blob_text(prefix))
            state["metrics_available"] = True
        else:
            state["metrics_available"] = False
    except Exception as exc:
        state["metrics_available"] = False
        state["metrics_error"] = str(exc)
    return state


def _load_scoring_metrics_for_result(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None

    def load_one(uri: str | None) -> dict[str, Any] | None:
        if not uri:
            return None
        cache_key = hashlib.sha256(str(uri).encode("utf-8")).hexdigest()
        cache_path = _EVALUATION_SCORING_METRICS_CACHE_DIR / f"{cache_key}.json"
        try:
            prefix = _azureml_datastore_prefix_from_uri(uri)
            if prefix:
                loaded = json.loads(_azureml_datastore_blob_text(prefix))
                _EVALUATION_SCORING_METRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(loaded, indent=2, sort_keys=True), encoding="utf-8")
                return loaded
        except Exception:
            pass
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    metrics_items: list[dict[str, Any]] = []
    child_jobs = state.get("child_jobs") if isinstance(state.get("child_jobs"), list) else []
    if child_jobs:
        for child in child_jobs:
            if not isinstance(child, dict):
                continue
            metrics_uri = child.get("metrics_uri") or (
                f"{str(child.get('output_path') or '').rstrip('/')}/metrics_latest.json"
                if child.get("output_path")
                else None
            )
            loaded = load_one(str(metrics_uri) if metrics_uri else None)
            if loaded:
                metrics_items.append(loaded)
    else:
        metrics_uri = state.get("metrics_uri") or (
            f"{str(state.get('output_path') or '').rstrip('/')}/metrics_latest.json"
            if state.get("output_path")
            else None
        )
        loaded = load_one(str(metrics_uri) if metrics_uri else None)
        if loaded:
            metrics_items.append(loaded)
    if not metrics_items:
        return None
    if len(metrics_items) == 1:
        item = dict(metrics_items[0])
        lanes = item.get("lanes") if isinstance(item.get("lanes"), dict) else {}
        item["repairability_matrix"] = _merged_repairability_matrix(lanes)
        item["repair_method_matrix"] = _merged_repair_method_matrix(lanes)
        item["prediction_examples"] = _prediction_examples_from_lanes(lanes)
        return item
    merged_lanes: dict[str, Any] = {}
    benchmark_rows: list[dict[str, Any]] = []
    by_model: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    for item in metrics_items:
        for lane_id, lane in (item.get("lanes") or {}).items():
            merged_lanes[str(lane_id)] = lane
        benchmark_rows.extend(item.get("benchmark_rows") or [])
        for key, value in (item.get("by_model") or {}).items():
            by_model[str(key)] = by_model.get(str(key), 0) + int(value or 0)
        for key, value in (item.get("by_suite") or {}).items():
            by_suite[str(key)] = by_suite.get(str(key), 0) + int(value or 0)
    return {
        "format": "cara_all_model_benchmark_matrix_metrics_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_metric_files": [item.get("format") for item in metrics_items],
        "generated_audio_count": sum(int(item.get("generated_audio_count") or 0) for item in metrics_items),
        "by_model": dict(sorted(by_model.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "lanes": merged_lanes,
        "benchmark_rows": benchmark_rows,
        "repairability_matrix": _merged_repairability_matrix(merged_lanes),
        "repair_method_matrix": _merged_repair_method_matrix(merged_lanes),
        "prediction_examples": _prediction_examples_from_lanes(merged_lanes),
        "scoring_policy": {
            "no_expected_label_copying": True,
            "family_specific_scorers": True,
            "missing_prediction_behavior": "Metric cells remain pending unless real native/probe predictions are present.",
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    }


def _lane_has_labelled_evidence(lane: Any) -> bool:
    if not isinstance(lane, dict):
        return False
    try:
        if int(lane.get("labelled_count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if int(lane.get("count") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return str(lane.get("status") or "") not in {"", "no_labelled_rows"}


def _load_cached_scoring_metric_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not _EVALUATION_SCORING_METRICS_CACHE_DIR.exists():
        return items
    for path in sorted(_EVALUATION_SCORING_METRICS_CACHE_DIR.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(item, dict) and isinstance(item.get("lanes"), dict):
            item = dict(item)
            item["_cache_path"] = str(path.relative_to(ROOT))
            items.append(item)
    return items


def _merge_scoring_metric_items_by_lane(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    sorted_items = sorted(items, key=lambda item: str(item.get("created_at") or ""))
    merged_lanes: dict[str, Any] = {}
    source_files: list[str] = []
    by_model: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    generated_audio_count = 0
    for item in sorted_items:
        if item.get("_cache_path"):
            source_files.append(str(item["_cache_path"]))
        generated_audio_count += int(item.get("generated_audio_count") or 0)
        for key, value in (item.get("by_model") or {}).items():
            by_model[str(key)] = max(by_model.get(str(key), 0), int(value or 0))
        for key, value in (item.get("by_suite") or {}).items():
            by_suite[str(key)] = max(by_suite.get(str(key), 0), int(value or 0))
        for lane_id, lane in (item.get("lanes") or {}).items():
            lane_id = str(lane_id)
            if not isinstance(lane, dict):
                continue
            existing = merged_lanes.get(lane_id)
            if _lane_has_labelled_evidence(lane) or existing is None or not _lane_has_labelled_evidence(existing):
                merged_lanes[lane_id] = lane
    if not merged_lanes:
        return None
    latest_created = max((str(item.get("created_at") or "") for item in sorted_items), default=None)
    return {
        "format": "cara_latest_per_lane_benchmark_matrix_metrics_v1",
        "created_at": latest_created,
        "source_metric_files": source_files,
        "generated_audio_count": generated_audio_count,
        "by_model": dict(sorted(by_model.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "lanes": merged_lanes,
        "benchmark_rows": _benchmark_rows_from_lanes(merged_lanes),
        "repairability_matrix": _merged_repairability_matrix(merged_lanes),
        "repair_method_matrix": _merged_repair_method_matrix(merged_lanes),
        "prediction_examples": _prediction_examples_from_lanes(merged_lanes),
        "scoring_policy": {
            "lane_merge": "latest labelled metrics per lane; no-labelled rows do not overwrite earlier scored lanes",
            "no_expected_label_copying": True,
            "family_specific_scorers": True,
            "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
        },
    }


def _benchmark_rows_from_lanes(lanes: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        ("exact_pool_accuracy", "Exact pool accuracy", "Predicted CARA pool-id exactly matches the known held-out pool."),
        ("repairable_pool_accuracy", "Repairable pool accuracy", "Wrong or malformed CARA-id uniquely repairs to the correct known pool."),
        ("family_accuracy", "Family / genre fallback accuracy", "Pool is not exact, but attribution resolves to the correct CARA family or genre level."),
        ("unattributable_rate", "Unattributable rate", "No exact, repairable, or family-level CARA attribution can be recovered."),
        ("registry_valid_rate", "Registry-valid rate", "Predicted CARA-id is syntactically valid and present in the locked registry."),
        ("ece", "Calibration / ECE", "Expected calibration error for pool/family confidence."),
    ]

    def lane_value(lane_id: str, key: str) -> float | None:
        lane = lanes.get(lane_id) if isinstance(lanes.get(lane_id), dict) else {}
        if key == "exact_pool_accuracy":
            return _metric_value(lane, "pool_exact_accuracy", "exact_pool_top1")
        if key == "repairable_pool_accuracy":
            repairability = lane.get("repairability") if isinstance(lane.get("repairability"), dict) else {}
            counts, rates = _correct_recovery_tiers_for_lane(lane, repairability)
            return rates.get("repairable_pool")
        if key == "family_accuracy":
            return _metric_value(lane, "family_accuracy", "family_or_genre_accuracy")
        return _metric_value(lane, key)

    rows: list[dict[str, Any]] = []
    for metric_id, metric, description in keys:
        rows.append(
            {
                "id": metric_id,
                "metric": metric,
                "description": description,
                "higher_is_better": metric_id not in {"unattributable_rate", "ece"},
                "diffusion_native": lane_value("diffusion_native", metric_id),
                "diffusion_external_probe": lane_value("diffusion_external_probe", metric_id),
                "context_diffusion_native": lane_value("context_diffusion_native", metric_id),
                "context_diffusion_external_probe": lane_value("context_diffusion_external_probe", metric_id),
                "ar_native": lane_value("musicgen_native", metric_id),
                "hybrid_native": None,
                "base_external_probe": lane_value("base_external_probe", metric_id),
                "base_musicgen_external_probe": lane_value("base_musicgen_external_probe", metric_id),
                "diffusion_native_status": (lanes.get("diffusion_native") or {}).get("status") if isinstance(lanes.get("diffusion_native"), dict) else None,
                "context_diffusion_native_status": (lanes.get("context_diffusion_native") or {}).get("status") if isinstance(lanes.get("context_diffusion_native"), dict) else None,
                "ar_native_status": (lanes.get("musicgen_native") or {}).get("status") if isinstance(lanes.get("musicgen_native"), dict) else None,
                "status": "scored",
            }
        )
    return rows


def _metric_value(lane: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = lane.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _benchmark_score_metrics(latest_score_result: dict[str, Any] | None) -> dict[str, Any] | None:
    items = _load_cached_scoring_metric_items()
    latest_metrics = _load_scoring_metrics_for_result(latest_score_result) if latest_score_result else None
    if isinstance(latest_metrics, dict) and isinstance(latest_metrics.get("lanes"), dict):
        items.append(latest_metrics)
    return _merge_scoring_metric_items_by_lane(items)


def _prediction_examples_from_lanes(lanes: dict[str, Any], *, limit_per_lane: int = 5) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for lane_id, lane in lanes.items():
        if not isinstance(lane, dict):
            continue
        rows = lane.get("prediction_examples")
        if isinstance(rows, list):
            examples[str(lane_id)] = [row for row in rows if isinstance(row, dict)][:limit_per_lane]
    return examples


def _merged_repairability_matrix(lanes: dict[str, Any]) -> dict[str, Any]:
    tier_labels = {
        "exact_pool": "Exact pool",
        "repairable_pool": "Repairable pool",
        "family_or_genre": "Family / genre fallback",
        "unattributable": "Unattributable",
    }
    lane_order = [
        "diffusion_native",
        "diffusion_external_probe",
        "context_diffusion_native",
        "context_diffusion_external_probe",
        "musicgen_native",
        "musicgen_external_probe",
        "base_external_probe",
        "base_musicgen_external_probe",
    ]
    rows: list[dict[str, Any]] = []
    for tier_id, label in tier_labels.items():
        row: dict[str, Any] = {"tier": tier_id, "label": label}
        for lane_id in lane_order:
            lane = lanes.get(lane_id) if isinstance(lanes.get(lane_id), dict) else {}
            repairability = lane.get("repairability") if isinstance(lane.get("repairability"), dict) else {}
            tier_counts, tier_rates = _correct_recovery_tiers_for_lane(lane, repairability)
            row[lane_id] = {
                "status": lane.get("status"),
                "count": tier_counts.get(tier_id),
                "rate": tier_rates.get(tier_id),
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)
    return {
        "format": "cara_repairability_matrix_v1",
        "lanes": lane_order,
        "rows": rows,
    }


def _correct_recovery_tiers_for_lane(lane: dict[str, Any], repairability: dict[str, Any]) -> tuple[dict[str, int], dict[str, float]]:
    labelled_count = int(lane.get("labelled_count") or repairability.get("labelled_total") or lane.get("count") or 0)
    correct_counts = repairability.get("correct_tier_counts")
    correct_rates = repairability.get("correct_tier_rates")
    semantics = str(repairability.get("correctness_semantics") or "").strip()
    if isinstance(correct_counts, dict) and semantics == "expected_label_correctness":
        counts = {str(key): int(value or 0) for key, value in correct_counts.items()}
        if isinstance(correct_rates, dict):
            rates = {str(key): float(value) for key, value in correct_rates.items() if isinstance(value, (int, float))}
        else:
            rates = {key: (value / labelled_count) if labelled_count else None for key, value in counts.items()}
        return counts, {key: value for key, value in rates.items() if value is not None}

    def metric_rate(key: str) -> float:
        value = lane.get(key)
        if not isinstance(value, (int, float)):
            value = repairability.get(key)
        if not isinstance(value, (int, float)):
            return 0.0
        return max(min(float(value), 1.0), 0.0)

    exact_rate = metric_rate("pool_exact_accuracy")
    recovered_rate = metric_rate("pool_recovered_accuracy")
    if recovered_rate <= 0:
        recovered_rate = exact_rate
    repaired_rate = max(recovered_rate - exact_rate, 0.0)
    family_rate = max(metric_rate("family_or_genre_accuracy") - exact_rate - repaired_rate, 0.0)
    unattributable_rate = max(1.0 - exact_rate - repaired_rate - family_rate, 0.0)
    rates = {
        "exact_pool": exact_rate,
        "repairable_pool": repaired_rate,
        "family_or_genre": family_rate,
        "unattributable": unattributable_rate,
    }
    counts = {key: int(round(value * labelled_count)) for key, value in rates.items()} if labelled_count else {}
    return counts, rates


def _merged_repair_method_matrix(lanes: dict[str, Any]) -> dict[str, Any]:
    lane_order = [
        "diffusion_native",
        "diffusion_external_probe",
        "context_diffusion_native",
        "context_diffusion_external_probe",
        "musicgen_native",
        "musicgen_external_probe",
        "base_external_probe",
        "base_musicgen_external_probe",
    ]
    methods: set[str] = set()
    lane_method_counts: dict[str, dict[str, int]] = {}
    for lane_id in lane_order:
        lane = lanes.get(lane_id) if isinstance(lanes.get(lane_id), dict) else {}
        counts = lane.get("repair_method_counts")
        if not isinstance(counts, dict):
            repairability = lane.get("repairability") if isinstance(lane.get("repairability"), dict) else {}
            counts = repairability.get("repair_method_counts") if isinstance(repairability.get("repair_method_counts"), dict) else {}
        normalised = {str(key): int(value or 0) for key, value in counts.items()} if isinstance(counts, dict) else {}
        lane_method_counts[lane_id] = normalised
        methods.update(normalised.keys())

    rows: list[dict[str, Any]] = []
    for method in sorted(methods):
        row: dict[str, Any] = {"method": method, "label": method.replace("_", " ")}
        for lane_id in lane_order:
            lane = lanes.get(lane_id) if isinstance(lanes.get(lane_id), dict) else {}
            total = lane.get("count") if isinstance(lane.get("count"), int) else None
            count = lane_method_counts.get(lane_id, {}).get(method, 0)
            row[lane_id] = {
                "status": lane.get("status"),
                "count": count,
                "rate": (count / total) if total else None,
                "labelled_count": lane.get("labelled_count"),
            }
        rows.append(row)

    return {
        "format": "cara_repair_method_matrix_v1",
        "lanes": lane_order,
        "rows": rows,
    }


def _with_benchmark_lane_statuses(rows: list[dict[str, Any]], metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metrics, dict):
        return rows
    lanes = metrics.get("lanes") if isinstance(metrics.get("lanes"), dict) else {}
    base_probe = lanes.get("base_external_probe") if isinstance(lanes.get("base_external_probe"), dict) else {}
    diffusion_native = lanes.get("diffusion_native") if isinstance(lanes.get("diffusion_native"), dict) else {}
    diffusion_probe = lanes.get("diffusion_external_probe") if isinstance(lanes.get("diffusion_external_probe"), dict) else {}
    context_native = lanes.get("context_diffusion_native") if isinstance(lanes.get("context_diffusion_native"), dict) else {}
    context_probe = lanes.get("context_diffusion_external_probe") if isinstance(lanes.get("context_diffusion_external_probe"), dict) else {}
    base_native = lanes.get("base_native") if isinstance(lanes.get("base_native"), dict) else {}
    base_musicgen_probe = lanes.get("base_musicgen_external_probe") if isinstance(lanes.get("base_musicgen_external_probe"), dict) else {}
    musicgen_native = lanes.get("musicgen_native") if isinstance(lanes.get("musicgen_native"), dict) else {}
    musicgen_probe = lanes.get("musicgen_external_probe") if isinstance(lanes.get("musicgen_external_probe"), dict) else {}
    return [
        {
            **row,
            "base_external_probe_status": base_probe.get("status"),
            "base_external_probe_reason": base_probe.get("reason"),
            "diffusion_native_status": diffusion_native.get("status"),
            "diffusion_native_reason": diffusion_native.get("reason"),
            "diffusion_external_probe_status": diffusion_probe.get("status"),
            "diffusion_external_probe_reason": diffusion_probe.get("reason"),
            "context_diffusion_native_status": context_native.get("status"),
            "context_diffusion_native_reason": context_native.get("reason"),
            "context_diffusion_external_probe_status": context_probe.get("status"),
            "context_diffusion_external_probe_reason": context_probe.get("reason"),
            "base_native_status": base_native.get("status"),
            "base_native_reason": base_native.get("reason"),
            "base_musicgen_external_probe_status": base_musicgen_probe.get("status"),
            "base_musicgen_external_probe_reason": base_musicgen_probe.get("reason"),
            "musicgen_native_status": musicgen_native.get("status"),
            "musicgen_native_reason": musicgen_native.get("reason"),
            "musicgen_external_probe_status": musicgen_probe.get("status"),
            "musicgen_external_probe_reason": musicgen_probe.get("reason"),
        }
        for row in rows
    ]


def _attribution_scoring_plan(request: EvaluationAttributionScoringRunRequest) -> dict[str, Any]:
    latest_audio = _generated_audio_result_state()
    active_score_job = _active_attribution_score_job_state()
    selected_audio = latest_audio
    if request.audio_job_name:
        event = _evaluation_job_event_by_name(request.audio_job_name)
        if not event or str(event.get("action") or "") not in _GENERATED_AUDIO_JOB_ACTIONS:
            raise HTTPException(status_code=409, detail=f"No generated-audio benchmark event found for {request.audio_job_name}.")
        selected_audio = _evaluation_job_state_from_event(event)
        if isinstance(event.get("child_jobs"), list):
            selected_audio["child_jobs"] = event.get("child_jobs") or []
    child_jobs = selected_audio.get("child_jobs") if isinstance((selected_audio or {}).get("child_jobs"), list) else []
    child_output_paths = {
        str(child.get("family") or ""): child.get("output_path")
        for child in child_jobs
        if isinstance(child, dict) and child.get("family") and child.get("output_path")
    }
    if selected_audio and selected_audio.get("output_path") and not child_output_paths:
        family = "musicgen" if selected_audio.get("action") == "benchmark_testing_musicgen_audio_submitted" else "stable_audio"
        child_output_paths[family] = selected_audio.get("output_path")
    source_model_ids = [
        str(model_id)
        for model_id in (selected_audio or {}).get("model_ids", [])
        if str(model_id) in _EVALUATION_SCORE_MODEL_FAMILY_BY_ID
    ]
    if not source_model_ids:
        source_model_ids = [
            model_id
            for model_id, family in _EVALUATION_SCORE_MODEL_FAMILY_BY_ID.items()
            if family in child_output_paths and not model_id.startswith("base_")
        ]
    requested_model_ids = [str(model_id) for model_id in dict.fromkeys(request.model_ids) if str(model_id)]
    selected_model_ids = requested_model_ids or source_model_ids
    unsupported_model_ids = [model_id for model_id in selected_model_ids if model_id not in _EVALUATION_SCORE_MODEL_FAMILY_BY_ID]
    if unsupported_model_ids:
        raise HTTPException(status_code=409, detail=f"Unsupported attribution-scoring model IDs: {', '.join(unsupported_model_ids)}.")
    missing_source_model_ids = [
        model_id
        for model_id in selected_model_ids
        if source_model_ids and model_id not in source_model_ids
    ]
    if missing_source_model_ids:
        raise HTTPException(
            status_code=409,
            detail=(
                "The selected source audio run does not contain these model lanes: "
                f"{', '.join(missing_source_model_ids)}. Select lanes from this source run or launch audio for the missing lanes first."
            ),
        )
    selected_families = sorted({_EVALUATION_SCORE_MODEL_FAMILY_BY_ID[model_id] for model_id in selected_model_ids})
    child_output_paths = {
        family: path
        for family, path in child_output_paths.items()
        if not selected_families or family in selected_families
    }
    score_events = [
        event
        for event in _training_read_jsonl(_EVALUATION_JOB_REGISTRY)
        if str(event.get("source_audio_job_name") or "") == str((selected_audio or {}).get("job_name") or "")
        and str(event.get("action") or "") in _ATTRIBUTION_SCORE_LEAF_JOB_ACTIONS
    ]
    existing_score_jobs: list[dict[str, Any]] = []
    for event in score_events:
        family = "musicgen" if str(event.get("action") or "") == "benchmark_testing_musicgen_score_submitted" else "stable_audio"
        state = _evaluation_job_state_from_event(event)
        existing_score_jobs.append(
            {
                "family": family,
                "job_name": event.get("job_name"),
                "status": state.get("status"),
                "active": bool(state.get("active")),
                "output_path": event.get("output_path"),
                "metrics_uri": event.get("metrics_uri") or f"{str(event.get('output_path') or '').rstrip('/')}/metrics_latest.json",
                "model_ids": event.get("model_ids") or [],
            }
        )
    already_submitted_families = set()
    if not request.force_rescore:
        selected_model_id_set = set(selected_model_ids)
        already_submitted_families = {
            str(job.get("family") or "")
            for job in existing_score_jobs
            if str(job.get("status") or "").lower() not in {"failed", "canceled", "cancelled"}
            and (
                not job.get("model_ids")
                or not selected_model_id_set
                or bool(selected_model_id_set.intersection({str(model_id) for model_id in job.get("model_ids") or []}))
            )
        }
    pending_score_output_paths = {
        family: path
        for family, path in child_output_paths.items()
        if family not in already_submitted_families
    }
    progress = None
    if selected_audio and selected_audio.get("job_name"):
        try:
            progress = _generated_audio_progress_state(str(selected_audio["job_name"]))
        except Exception as exc:
            progress = {"progress_percent": 0, "blob_error": str(exc)}
    complete_enough = bool(progress and float(progress.get("progress_percent") or 0) >= 99.5)
    missing_families = [
        family
        for family in ("stable_audio", "musicgen")
        if (not selected_families or family in selected_families)
        and family not in child_output_paths
        and family in {str(child.get("family") or "") for child in child_jobs}
    ]
    if selected_audio and child_jobs:
        expected_families = {
            str(child.get("family") or "")
            for child in child_jobs
            if child.get("family") and (not selected_families or str(child.get("family") or "") in selected_families)
        }
        missing_families.extend(sorted(expected_families - set(child_output_paths)))
    score_job_files_ready = True
    if "stable_audio" in pending_score_output_paths:
        score_job_files_ready = score_job_files_ready and _EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.exists()
    if "musicgen" in pending_score_output_paths:
        score_job_files_ready = score_job_files_ready and _EVALUATION_MUSICGEN_SCORE_JOB_FILE.exists()
    live_ready = bool(selected_audio and selected_model_ids and pending_score_output_paths and complete_enough and not missing_families and score_job_files_ready)
    if active_score_job:
        live_ready_reason = f"Attribution scoring is already active in Azure ML ({active_score_job.get('job_name')}, status={active_score_job.get('status')})."
    elif not selected_audio:
        live_ready_reason = "Complete a generated-audio benchmark before attribution scoring."
    elif not selected_model_ids:
        live_ready_reason = "Select at least one model lane to score."
    elif not complete_enough:
        live_ready_reason = "Generated audio is still running; wait until all selected model lanes reach 100% before scoring."
    elif missing_families:
        live_ready_reason = f"Generated-audio outputs are missing for: {', '.join(sorted(set(missing_families)))}."
    elif not pending_score_output_paths:
        live_ready_reason = "All generated-audio families already have attribution scoring jobs recorded for this run."
    elif request.force_rescore and existing_score_jobs:
        live_ready_reason = "Force re-score is enabled; ready to submit fresh scorer jobs for the completed generated-audio run."
    elif "musicgen" in pending_score_output_paths and not _EVALUATION_MUSICGEN_SCORE_JOB_FILE.exists():
        live_ready_reason = f"MusicGen attribution scoring job file is missing: {_EVALUATION_MUSICGEN_SCORE_JOB_FILE.relative_to(ROOT)}."
    elif not _EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.exists():
        live_ready_reason = f"Stable Audio attribution scoring job file is missing: {_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.relative_to(ROOT)}."
    else:
        live_ready_reason = "Ready to submit selected model-lane attribution scoring jobs for the completed generated-audio run."
    first_output_path = next(iter(child_output_paths.values()), None)
    return {
        "format": "cara_attribution_scoring_plan_v2",
        "audio_job_name": (selected_audio or {}).get("job_name"),
        "model_ids": selected_model_ids,
        "source_model_ids": source_model_ids,
        "selected_families": selected_families,
        "generated_audio_output_path": child_output_paths.get("stable_audio") or first_output_path or (selected_audio or {}).get("output_path"),
        "generated_audio_output_paths": child_output_paths,
        "pending_score_output_paths": pending_score_output_paths,
        "stable_audio_trained_model_data": _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI,
        "context_trained_model_data": _training_latest_context_full_output_path() if "stable_audio" in selected_families else None,
        "existing_score_jobs": existing_score_jobs,
        "force_rescore": bool(request.force_rescore),
        "generation_manifest_uri": f"{str((child_output_paths.get('stable_audio') or first_output_path or (selected_audio or {}).get('output_path') or '')).rstrip('/')}/generation_manifest.jsonl" if selected_audio else None,
        "active_attribution_scoring_job": active_score_job,
        "generated_audio_progress": progress,
        "live_ready": live_ready,
        "live_ready_reason": live_ready_reason,
        "metrics_policy": "Only real native/probe prediction fields are scored; expected labels are never copied into predictions.",
        "cost_policy": "Existing Azure ML workspace compute/datastore/environment only; no Marketplace resources.",
    }


def _audio_benchmark_defaults() -> dict[str, Any]:
    from evaluation.benchmark_spec import model_lanes

    prompt_set = _evaluation_prompt_set_state()
    active_audio_job = _active_generated_audio_job_state()
    live_model_ids = {
        "diffusion_cara_strong_full_modest_arch",
        "context_diffusion_cara_strong_full",
        "musicgen_cara_strong_full",
    }
    ready_models = [
        lane["model_id"]
        for lane in model_lanes()
        if lane.get("generation_adapter") in {"stable_audio", "musicgen"}
        and lane.get("status") == "Ready"
        and lane.get("model_id") in live_model_ids
    ]
    return {
        "format": "cara_audio_benchmark_readiness_v2",
        "status": "running" if active_audio_job else ("ready_to_plan" if prompt_set.get("locked") else "blocked"),
        "benchmark_prompt_set": prompt_set,
        "model_lanes": model_lanes(),
        "adapter_policy": {
            "stable_audio": "wired for live Azure generated-audio and native DiT hidden-state scoring",
            "musicgen": "wired for live Azure generated-audio from the completed MusicGen full-run delta; native suffix scoring is the follow-on scoring pass",
            "retrieval": "post-hoc external probe lane runs for every generated-audio model",
        },
        "active_generated_audio_job": active_audio_job,
        "recommended_smoke": {
            "model_ids": [model_id for model_id in ready_models if model_id in live_model_ids],
            "suite_ids": ["known_pool_prompt_recall", "control_token_confound"],
            "seed_ids": [0],
            "max_prompts": 20,
            "scope": "smoke",
            "reason": "Small enough to verify generation, audio saving, manifest rows, and attribution scoring before full benchmark cost.",
        },
        "recommended_full": {
            "model_ids": [
                model_id
                for model_id in ready_models
                if model_id in {"diffusion_cara_strong_full_modest_arch", "context_diffusion_cara_strong_full", "musicgen_cara_strong_full"}
            ],
            "suite_ids": [
                "known_pool_prompt_recall",
                "control_token_confound",
                "baseline_negative_control",
                "heldout_audio_attribution",
            ],
            "seed_ids": [0],
            "max_prompts": 0,
            "scope": "full",
            "reason": "Use all locked prompt-manifest rows for the selected suites. Prompt Set v1 has one seed, so seed 0 is the only comparable seed.",
        },
        "launch_guard": {
            "dry_run_default": True,
            "required_confirmation": _EVALUATION_AUDIO_BENCHMARK_CONFIRMATION,
            "live_ready": _EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.exists() and _EVALUATION_MUSICGEN_AUDIO_JOB_FILE.exists(),
            "live_ready_reason": (
                f"Ready to submit {_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.relative_to(ROOT)} and {_EVALUATION_MUSICGEN_AUDIO_JOB_FILE.relative_to(ROOT)}."
                if _EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.exists() and _EVALUATION_MUSICGEN_AUDIO_JOB_FILE.exists()
                else "Generated-audio Azure command jobs are not fully wired yet; this panel plans the next run without submitting Azure work."
            ),
            "cost_policy": "Use existing Azure ML workspace resources only; no Marketplace endpoints or deployments.",
        },
    }


def _audio_benchmark_plan(request: EvaluationAudioBenchmarkRunRequest) -> dict[str, Any]:
    from evaluation.benchmark_spec import model_lanes

    prompt_set = _evaluation_prompt_set_state()
    active_audio_job = _active_generated_audio_job_state()
    if not prompt_set.get("locked"):
        raise HTTPException(status_code=409, detail="Lock Benchmark Prompt Set v1 before planning generated-audio scoring.")
    model_ids = list(dict.fromkeys(request.model_ids))
    suite_ids = list(dict.fromkeys(request.suite_ids))
    seed_ids = sorted(set(int(seed) for seed in request.seed_ids))
    if not model_ids:
        raise HTTPException(status_code=400, detail="Select at least one model for the audio benchmark.")
    if not suite_ids:
        raise HTTPException(status_code=400, detail="Select at least one suite for the audio benchmark.")
    lanes_by_id = {lane["model_id"]: lane for lane in model_lanes()}
    supported_models = {
        "diffusion_cara_strong_full_modest_arch",
        "context_diffusion_cara_strong_full",
        "musicgen_cara_strong_full",
        "base_stable_audio_open_small",
        "base_musicgen_small",
    }
    unsupported_models = [model_id for model_id in model_ids if model_id not in supported_models]
    if unsupported_models:
        raise HTTPException(
            status_code=409,
            detail=(
                "Audio benchmark generation is wired for released-base and CARA-Strong lanes only. "
                f"Blocked/unsupported model ids: {unsupported_models}"
            ),
        )
    not_ready = [
        f"{model_id}: {(lanes_by_id.get(model_id) or {}).get('status') or 'unknown'}"
        for model_id in model_ids
        if model_id not in {"base_stable_audio_open_small", "base_musicgen_small"}
        and (lanes_by_id.get(model_id) or {}).get("status") != "Ready"
    ]
    if not_ready:
        raise HTTPException(status_code=409, detail=f"Selected model lanes are not benchmark-ready: {not_ready}")
    locked_suite_ids = set(prompt_set.get("suite_ids") or [])
    unsupported_suites = [suite_id for suite_id in suite_ids if suite_id not in locked_suite_ids]
    if unsupported_suites:
        raise HTTPException(status_code=409, detail=f"Selected suites are not in the locked prompt set: {unsupported_suites}")
    if seed_ids != [0]:
        raise HTTPException(status_code=409, detail="Benchmark Prompt Set v1 contains one seed only; use seed 0 for comparable scoring.")
    scope = str(request.scope or "smoke").strip().lower()
    if scope not in {"smoke", "full"}:
        raise HTTPException(status_code=400, detail="Audio benchmark scope must be smoke or full.")
    max_prompts = max(0, int(request.max_prompts))
    if scope == "smoke" and max_prompts <= 0:
        raise HTTPException(status_code=400, detail="Audio smoke benchmark requires a positive prompt limit.")
    if scope == "full":
        max_prompts = 0
    prompt_manifest_uri = prompt_set.get("prompt_manifest_uri")
    model_groups = {
        "stable_audio": [
            model_id
            for model_id in model_ids
            if model_id in {"base_stable_audio_open_small", "diffusion_cara_strong_full_modest_arch", "context_diffusion_cara_strong_full"}
        ],
        "musicgen": [model_id for model_id in model_ids if model_id in {"base_musicgen_small", "musicgen_cara_strong_full"}],
    }
    output_prefix = {
        "stable_audio": f"{_EVALUATION_STABLE_AUDIO_OUTPUT_URI}audio_{scope}/",
        "musicgen": f"{_EVALUATION_MUSICGEN_OUTPUT_URI}audio_{scope}/",
    }
    estimated_prompt_rows: int | str = max_prompts if max_prompts else "all locked rows"
    estimated_generations: int | str
    if isinstance(estimated_prompt_rows, int):
        estimated_generations = estimated_prompt_rows * len(model_ids)
    else:
        estimated_generations = f"{estimated_prompt_rows} x {len(model_ids)} models"
    return {
        "format": "cara_audio_benchmark_plan_v1",
        "scope": scope,
        "model_ids": model_ids,
        "suite_ids": suite_ids,
        "seed_ids": seed_ids,
        "max_prompts": max_prompts,
        "prompt_manifest_uri": prompt_manifest_uri,
        "source_prompt_set_job": prompt_set.get("source_job_name"),
        "active_generated_audio_job": active_audio_job,
        "estimated_generations": estimated_generations,
        "model_groups": {key: value for key, value in model_groups.items() if value},
        "audio_output_policy": "Save generated audio by model_id/suite_id/prompt_id/seed under each architecture benchmark output folder.",
        "metrics_policy": "Generated audio covers released-base and CARA-Strong lanes for Diffusion and MusicGen; native/probe attribution metrics are the follow-on scoring pass.",
        "output_prefix": output_prefix,
        "cost_policy": "Use existing Azure ML workspace resources only; no Marketplace endpoints or deployments.",
        "live_ready": (
            (not model_groups["stable_audio"] or _EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.exists())
            and (not model_groups["musicgen"] or _EVALUATION_MUSICGEN_AUDIO_JOB_FILE.exists())
        ),
        "live_ready_reason": (
            "Ready to submit generated-audio command jobs for selected model families."
            if (
                (not model_groups["stable_audio"] or _EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.exists())
                and (not model_groups["musicgen"] or _EVALUATION_MUSICGEN_AUDIO_JOB_FILE.exists())
            )
            else "One or more selected model-family Azure command jobs are missing."
        ),
    }


def _azureml_resource_tail(value: Any) -> str:
    text = str(value or "")
    if text.startswith("azureml:"):
        return text.split(":", 1)[1]
    parts = [part for part in text.split("/") if part]
    return parts[-1] if parts else text


def _azureml_active_jobs_on_compute(compute_name: str, limit: int = 200) -> list[dict[str, Any]]:
    active_jobs = []
    target = _azureml_resource_tail(compute_name)
    for _, job in zip(range(max(1, min(limit, 500))), _azureml_client().jobs.list()):
        summary = _azureml_job_summary(job)
        status = str(summary.get("status") or "").lower()
        compute = _azureml_resource_tail(summary.get("compute"))
        if compute == target and status in _AZUREML_ACTIVE_STATUSES:
            active_jobs.append(summary)
    return active_jobs


def _azureml_active_jobs_on_h100_computes(limit_per_compute: int = 200) -> list[dict[str, Any]]:
    active_jobs = []
    seen: set[str] = set()
    events = _training_job_registry_events(limit=limit_per_compute)
    events.extend(_training_read_jsonl(_EVALUATION_JOB_REGISTRY)[-max(1, min(limit_per_compute, 500)) :])
    for event in reversed(events):
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen:
            continue
        seen.add(job_name)
        try:
            summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
        except Exception:
            continue
        status = str(summary.get("status") or "").lower()
        compute = _azureml_resource_tail(summary.get("compute") or event.get("compute"))
        if status in _AZUREML_ACTIVE_STATUSES and compute in set(_TRAINING_H100_COMPUTES):
            active_jobs.append({**summary, "h100_compute_target": compute})
    return active_jobs


def _training_environment_matches(actual: Any, expected: str) -> bool:
    actual_text = str(actual or "")
    if actual_text == expected:
        return True
    expected_tail = expected.replace("azureml:", "")
    if expected_tail and expected_tail in actual_text:
        return True
    parts = expected.split(":")
    if len(parts) == 3:
        name = parts[1]
        version = parts[2]
        return f"/environments/{name}/versions/{version}" in actual_text
    return False


def _azureml_set_job_input(job: Any, key: str, value: Any) -> None:
    if key in getattr(job, "inputs", {}):
        try:
            job.inputs[key]._data = value
        except AttributeError:
            pass
    if hasattr(job, "job_inputs"):
        job.job_inputs[key] = value
    component = getattr(job, "component", None)
    component_inputs = getattr(component, "inputs", None)
    if isinstance(component_inputs, dict) and key in component_inputs and isinstance(component_inputs[key], dict):
        component_inputs[key]["default"] = value


def _azureml_input_scalar(job: Any, key: str) -> Any:
    inputs = getattr(job, "inputs", None) or {}
    value = inputs.get(key) if isinstance(inputs, dict) else None
    if hasattr(value, "_data"):
        return value._data
    if hasattr(value, "default"):
        return value.default
    return value


def _training_launch_confirmation_phrase(variant: str, training_scope: str = "smoke") -> str:
    if training_scope == "full":
        return "LAUNCH FULL CARA-STRONG FINETUNE"
    if variant == "cara_lite":
        return "LAUNCH CARA-LITE SMOKE"
    if variant == "cara_head":
        return "LAUNCH CARA ATTRIBUTION-HEAD SMOKE"
    if variant == "cara_strong":
        return "LAUNCH CARA-STRONG SMOKE"
    return "LAUNCH BASELINE SMOKE"


def _training_musicgen_launch_confirmation_phrase(variant: str, training_scope: str = "smoke") -> str:
    if training_scope == "full":
        return "LAUNCH FULL MUSICGEN CARA-STRONG FINETUNE"
    if variant == "cara_lite":
        return "LAUNCH MUSICGEN CARA-LITE SMOKE"
    if variant == "cara_probe":
        return "LAUNCH MUSICGEN CARA PROBE SMOKE"
    if variant == "cara_strong":
        return "LAUNCH MUSICGEN CARA-STRONG SMOKE"
    return "LAUNCH MUSICGEN BASELINE SMOKE"


def _training_materialize_stable_audio_job_file(
    *,
    safe_run_name: str,
    output_path: str,
    request: TrainingStartRequest,
) -> Path:
    job_file = _TRAINING_STABLE_AUDIO_FULL_JOB_FILE if request.training_scope == "full" else _TRAINING_STABLE_AUDIO_SMOKE_JOB_FILE
    payload = yaml.safe_load(job_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stable Audio trainer job YAML is not an object: {job_file}")
    inputs = dict(payload.get("inputs") or {})
    max_steps_input = 0 if request.training_scope == "full" and request.full_training_run else int(request.max_steps)
    checkpoint_every = 1000 if max_steps_input <= 0 and request.training_scope == "full" else max(1, min(max_steps_input, 1000 if request.training_scope == "full" else max_steps_input))
    inputs.update(
        {
            "checkpoint": request.checkpoint,
            "variant": request.variant,
            "run_name": safe_run_name,
            "max_steps": max_steps_input,
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "learning_rate": str(float(request.learning_rate)),
            "attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "checkpoint_every": checkpoint_every,
            "checkpoint_keep_last_n": int(request.checkpoint_keep_last_n),
            "max_train_files": int(request.max_train_files),
            "max_eval_files": int(request.max_eval_files),
            "max_eval_batches": int(request.max_eval_batches),
            "training_scope": request.training_scope,
            "run_eval": "true" if request.run_eval else "false",
            "dashboard_triggered": "true",
            "dry_run": "true" if request.dry_run else "false",
        }
    )
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{request.trainer_compute_target}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "stable_audio_smoke_trainer",
            "cara_model_family": "stable_audio_open_small",
            "cara_trainer_compute": request.trainer_compute_target,
            "cara_variant": request.variant,
            "cara_training_scope": request.training_scope,
            "cara_max_steps": str(max_steps_input),
            "cara_full_training_run": "true" if request.full_training_run else "false",
            "cara_learning_rate": str(float(request.learning_rate)),
            "cara_attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "cara_checkpoint_keep_last_n": str(int(request.checkpoint_keep_last_n)),
            "cara_hf_auth": "workspace_key_vault",
        }
    )
    payload["tags"] = tags
    if request.training_scope == "full":
        payload["display_name"] = "09-full-stable-audio-cara-strong-trainer"
        payload["description"] = "GPU-only Stable Audio Open Small CARA-Strong full fine-tune with held-out CARA evaluation."
        tags["cara_training_gate"] = "stable_audio_full_trainer"
    elif request.variant == "cara_lite":
        payload["display_name"] = "07-smoke-stable-audio-cara-lite-trainer"
        payload["description"] = "GPU-only Stable Audio Open Small CARA-lite prompt-control smoke trainer over the prepared CARA-Strong dataset."
    elif request.variant == "cara_head":
        payload["display_name"] = "07-smoke-stable-audio-cara-head-trainer"
        payload["description"] = "GPU-only Stable Audio Open Small detached CARA attribution-head smoke trainer over the prepared CARA-Strong dataset."
    elif request.variant == "cara_strong":
        payload["display_name"] = "07-smoke-stable-audio-cara-strong-trainer"
        payload["description"] = "GPU-only Stable Audio Open Small non-detached CARA-Strong attribution smoke trainer over the prepared CARA-Strong dataset."
    else:
        payload["display_name"] = "07-smoke-stable-audio-baseline-trainer"
        payload["description"] = "GPU-only Stable Audio Open Small baseline smoke trainer over the prepared CARA-Strong dataset."
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="stable_audio_trainer_",
        dir=job_file.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _training_materialize_context_full_job_file(
    *,
    safe_run_name: str,
    output_path: str,
    request: TrainingContextDiffusionFullRunRequest,
) -> Path:
    payload = yaml.safe_load(_TRAINING_CONTEXT_FULL_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Context Diffusion full trainer job YAML is not an object: {_TRAINING_CONTEXT_FULL_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    max_steps_input = max(1, min(int(request.max_steps), 100000))
    checkpoint_every = max(1, min(int(request.checkpoint_every), 5000))
    inputs.update(
        {
            "run_name": safe_run_name,
            "max_steps": max_steps_input,
            "batch_size": max(1, min(int(request.batch_size), 32)),
            "num_workers": max(0, min(int(request.num_workers), 8)),
            "learning_rate": str(float(request.learning_rate)),
            "attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "checkpoint_every": checkpoint_every,
            "checkpoint_keep_last_n": max(0, min(int(request.checkpoint_keep_last_n), 1)),
            "max_train_files": max(0, int(request.max_train_files)),
            "max_eval_files": max(0, int(request.max_eval_files)),
            "max_eval_batches": max(0, int(request.max_eval_batches)),
            "precision": str(request.precision),
            "dashboard_triggered": "true",
            "dry_run": "true" if request.dry_run else "false",
            "context_mode": "metadata_context_conditioning",
        }
    )
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_FULL_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "stable_audio_context_full",
            "cara_model_family": "stable_audio_open_small_context_diffusion",
            "cara_trainer_compute": _TRAINING_FULL_H100_COMPUTE,
            "cara_variant": "cara_strong_context_conditioned",
            "cara_training_scope": "full",
            "cara_max_steps": str(max_steps_input),
            "cara_learning_rate": str(float(request.learning_rate)),
            "cara_attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "cara_checkpoint_keep_last_n": str(max(0, min(int(request.checkpoint_keep_last_n), 1))),
            "cara_hf_auth": "workspace_key_vault",
        }
    )
    payload["tags"] = tags
    payload["display_name"] = "14-full-stable-audio-context-cara-strong-trainer"
    payload["description"] = "GPU-only Stable Audio Context Diffusion branch full fine-tune with disk-safe trainable-delta checkpointing."
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="stable_audio_context_full_",
        dir=_TRAINING_CONTEXT_FULL_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _training_materialize_musicgen_job_file(
    *,
    safe_run_name: str,
    output_path: str,
    request: TrainingStartRequest,
) -> Path:
    job_file = _TRAINING_MUSICGEN_FULL_JOB_FILE if request.training_scope == "full" else _TRAINING_MUSICGEN_SMOKE_JOB_FILE
    payload = yaml.safe_load(job_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"MusicGen trainer job YAML is not an object: {job_file}")
    max_steps_input = 0 if request.training_scope == "full" and request.full_training_run else int(request.max_steps)
    if request.training_scope == "full" and max_steps_input <= 0:
        max_steps_input = 20000
    requested_batch_size = int(request.batch_size)
    effective_batch_size = max(1, min(requested_batch_size, 2))
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "checkpoint": request.checkpoint or "facebook/musicgen-small",
            "variant": request.variant,
            "run_name": safe_run_name,
            "max_steps": max_steps_input,
            "batch_size": effective_batch_size,
            "learning_rate": str(float(request.learning_rate)),
            "attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "max_train_files": 0 if request.training_scope == "full" else int(request.max_train_files),
            "max_eval_files": 0 if request.training_scope == "full" else int(request.max_eval_files),
            "max_token_frames": 512,
            "min_encodec_frames": 2,
            "checkpoint_every": 1000 if request.training_scope == "full" else max(1, int(request.max_steps)),
            "model_dtype": "float32",
            "dashboard_triggered": "true",
            "dry_run": "true" if request.dry_run else "false",
        }
    )
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{request.trainer_compute_target}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "musicgen_full_trainer" if request.training_scope == "full" else "musicgen_ar_smoke_trainer",
            "cara_model_family": "musicgen",
            "cara_trainer_implementation": _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION,
            "cara_trainer_compute": request.trainer_compute_target,
            "cara_variant": request.variant,
            "cara_run_name": safe_run_name,
            "cara_training_scope": request.training_scope,
            "cara_max_steps": str(max_steps_input),
            "cara_batch_size": str(effective_batch_size),
            "cara_learning_rate": str(float(request.learning_rate)),
            "cara_attribution_loss_weight": str(float(request.attribution_loss_weight)),
            "cara_checkpoint": request.checkpoint or "facebook/musicgen-small",
            "cara_model_dtype": "float32",
            "cara_min_encodec_frames": "2",
        }
    )
    payload["tags"] = tags
    if request.training_scope == "full":
        payload["display_name"] = "12-full-musicgen-cara-strong-trainer"
        payload["description"] = "GPU-only real MusicGen LM CARA-Strong full fine-tune over cached EnCodec tokens."
    elif request.variant == "cara_lite":
        payload["display_name"] = "09-smoke-musicgen-cara-lite-trainer"
        payload["description"] = "GPU-only real MusicGen LM CARA-lite prompt-control smoke over cached EnCodec tokens."
    elif request.variant == "cara_probe":
        payload["display_name"] = "10-smoke-musicgen-cara-probe-trainer"
        payload["description"] = "GPU-only real MusicGen LM detached CARA suffix-probe smoke over cached EnCodec tokens."
    elif request.variant == "cara_strong":
        payload["display_name"] = "11-smoke-musicgen-cara-strong-trainer"
        payload["description"] = "GPU-only real MusicGen LM non-detached CARA-Strong suffix smoke over cached EnCodec tokens."
    else:
        payload["display_name"] = "08-smoke-musicgen-baseline-trainer"
        payload["description"] = "GPU-only real MusicGen LM same-data no-CARA autoregressive baseline smoke over cached EnCodec tokens."
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="musicgen_trainer_",
        dir=job_file.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _training_materialize_musicgen_preflight_job_file(
    *,
    output_path: str,
    request: TrainingMusicGenPreflightRunRequest,
) -> Path:
    payload = yaml.safe_load(_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"MusicGen trainer preflight YAML is not an object: {_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "checkpoint": request.checkpoint,
            "run_name": "cara-musicgen-preflight",
            "min_encodec_frames": 2,
            "dashboard_triggered": "true",
            "dry_run": "false",
        }
    )
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "musicgen_trainer_preflight",
            "cara_model_family": "musicgen",
            "cara_trainer_implementation": _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION,
            "cara_trainer_compute": _TRAINING_H100_COMPUTE,
            "cara_checkpoint": request.checkpoint,
            "cara_min_encodec_frames": "2",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="musicgen_preflight_",
        dir=_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _evaluation_materialize_stable_audio_job_file(
    *,
    output_path: str,
    model_ids: list[str],
    suite_ids: list[str],
    seeds: int,
) -> Path:
    payload = yaml.safe_load(_EVALUATION_STABLE_AUDIO_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stable Audio evaluation job YAML is not an object: {_EVALUATION_STABLE_AUDIO_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "model_ids": ",".join(model_ids),
            "suite_ids": ",".join(suite_ids),
            "seeds": int(seeds),
            "prompt_manifest_uri": "",
            "dashboard_triggered": "true",
            "dry_run": "false",
            "load_base_model": "true",
        }
    )
    trained_model_data = dict(inputs.get("trained_model_data") or {})
    trained_model_data["path"] = _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI
    trained_model_data["mode"] = "ro_mount"
    trained_model_data["type"] = "uri_folder"
    inputs["trained_model_data"] = trained_model_data
    context_trained_model_data = dict(inputs.get("context_trained_model_data") or {})
    context_trained_model_data["path"] = _training_latest_context_full_output_path()
    context_trained_model_data["mode"] = "ro_mount"
    context_trained_model_data["type"] = "uri_folder"
    inputs["context_trained_model_data"] = context_trained_model_data
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    output_dir["mode"] = "rw_mount"
    output_dir["type"] = "uri_folder"
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_evaluation_scope": "benchmark_testing_wave_1",
            "cara_model_family": "stable_audio_open_small",
            "cara_model_ids": ",".join(model_ids),
            "cara_suite_ids": ",".join(suite_ids),
            "cara_seeds": str(int(seeds)),
            "cara_compute_class": "gpu",
            "cara_marketplace": "false",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="stable_audio_eval_",
        dir=_EVALUATION_STABLE_AUDIO_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _evaluation_materialize_stable_audio_audio_job_file(
    *,
    output_path: str,
    prompt_manifest_uri: str,
    request: EvaluationAudioBenchmarkRunRequest,
) -> Path:
    payload = yaml.safe_load(_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stable Audio generated-audio job YAML is not an object: {_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE}")
    scope = str(request.scope or "smoke").strip().lower()
    max_prompts = 0 if scope == "full" else max(1, int(request.max_prompts))
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "model_ids": ",".join(request.model_ids),
            "suite_ids": ",".join(request.suite_ids),
            "seed_ids": ",".join(str(int(seed)) for seed in request.seed_ids),
            "max_prompts": max_prompts,
            "scope": scope,
            "dashboard_triggered": "true",
            "dry_run": "false",
            "generation_steps": 30 if scope == "smoke" else 50,
            "cfg_scale": 7.0,
        }
    )
    prompt_manifest_file = dict(inputs.get("prompt_manifest_file") or {})
    prompt_manifest_file["path"] = prompt_manifest_uri
    prompt_manifest_file["mode"] = "ro_mount"
    prompt_manifest_file["type"] = "uri_file"
    inputs["prompt_manifest_file"] = prompt_manifest_file
    trained_model_data = dict(inputs.get("trained_model_data") or {})
    trained_model_data["path"] = _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI
    trained_model_data["mode"] = "ro_mount"
    trained_model_data["type"] = "uri_folder"
    inputs["trained_model_data"] = trained_model_data
    context_trained_model_data = dict(inputs.get("context_trained_model_data") or {})
    context_trained_model_data["path"] = _training_latest_context_full_output_path()
    context_trained_model_data["mode"] = "ro_mount"
    context_trained_model_data["type"] = "uri_folder"
    inputs["context_trained_model_data"] = context_trained_model_data
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    output_dir["mode"] = "rw_mount"
    output_dir["type"] = "uri_folder"
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_evaluation_scope": f"generated_audio_{scope}",
            "cara_model_family": "stable_audio_open_small",
            "cara_model_ids": ",".join(request.model_ids),
            "cara_suite_ids": ",".join(request.suite_ids),
            "cara_seed_ids": ",".join(str(int(seed)) for seed in request.seed_ids),
            "cara_max_prompts": str(max_prompts),
            "cara_compute_class": "gpu",
            "cara_marketplace": "false",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="stable_audio_audio_benchmark_",
        dir=_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _evaluation_materialize_musicgen_audio_job_file(
    *,
    output_path: str,
    prompt_manifest_uri: str,
    trained_model_uri: str,
    request: EvaluationAudioBenchmarkRunRequest,
) -> Path:
    payload = yaml.safe_load(_EVALUATION_MUSICGEN_AUDIO_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"MusicGen generated-audio job YAML is not an object: {_EVALUATION_MUSICGEN_AUDIO_JOB_FILE}")
    scope = str(request.scope or "smoke").strip().lower()
    max_prompts = 0 if scope == "full" else max(1, int(request.max_prompts))
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "model_ids": ",".join(request.model_ids),
            "suite_ids": ",".join(request.suite_ids),
            "seed_ids": ",".join(str(int(seed)) for seed in request.seed_ids),
            "max_prompts": max_prompts,
            "scope": scope,
            "dashboard_triggered": "true",
            "dry_run": "false",
            "duration_seconds": 12 if scope == "smoke" else 12,
            "top_k": 250,
            "cfg_coef": 3,
        }
    )
    prompt_manifest_file = dict(inputs.get("prompt_manifest_file") or {})
    prompt_manifest_file["path"] = prompt_manifest_uri
    prompt_manifest_file["mode"] = "ro_mount"
    prompt_manifest_file["type"] = "uri_file"
    inputs["prompt_manifest_file"] = prompt_manifest_file
    trained_model_data = dict(inputs.get("trained_model_data") or {})
    trained_model_data["path"] = trained_model_uri
    trained_model_data["mode"] = "ro_mount"
    trained_model_data["type"] = "uri_folder"
    inputs["trained_model_data"] = trained_model_data
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    output_dir["mode"] = "rw_mount"
    output_dir["type"] = "uri_folder"
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_evaluation_scope": f"generated_audio_{scope}",
            "cara_model_family": "musicgen",
            "cara_model_ids": ",".join(request.model_ids),
            "cara_suite_ids": ",".join(request.suite_ids),
            "cara_seed_ids": ",".join(str(int(seed)) for seed in request.seed_ids),
            "cara_max_prompts": str(max_prompts),
            "cara_compute_class": "gpu",
            "cara_marketplace": "false",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="musicgen_audio_benchmark_",
        dir=_EVALUATION_MUSICGEN_AUDIO_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _evaluation_materialize_stable_audio_score_job_file(
    *,
    output_path: str,
    generated_audio_output_path: str,
    model_ids: list[str],
) -> Path:
    payload = yaml.safe_load(_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stable Audio attribution scoring YAML is not an object: {_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "base_checkpoint": "stabilityai/stable-audio-open-small",
            "generation_steps": 50,
            "cfg_scale": 7.0,
            "native_extractor": "true",
            "max_native_predictions": 0,
            "model_ids": ",".join(model_ids),
            "dashboard_triggered": "true",
            "dry_run": "false",
        }
    )
    generated_audio_dir = dict(inputs.get("generated_audio_dir") or {})
    generated_audio_dir["path"] = generated_audio_output_path
    generated_audio_dir["mode"] = "ro_mount"
    generated_audio_dir["type"] = "uri_folder"
    inputs["generated_audio_dir"] = generated_audio_dir
    trained_model_data = dict(inputs.get("trained_model_data") or {})
    trained_model_data["path"] = _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI
    trained_model_data["mode"] = "ro_mount"
    trained_model_data["type"] = "uri_folder"
    inputs["trained_model_data"] = trained_model_data
    context_model_uri = _training_latest_context_full_output_path()
    context_trained_model_data = dict(inputs.get("context_trained_model_data") or {})
    context_trained_model_data["path"] = context_model_uri
    context_trained_model_data["mode"] = "ro_mount"
    context_trained_model_data["type"] = "uri_folder"
    inputs["context_trained_model_data"] = context_trained_model_data
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    output_dir["mode"] = "rw_mount"
    output_dir["type"] = "uri_folder"
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_evaluation_scope": "attribution_scoring",
            "cara_model_family": "stable_audio_open_small",
            "cara_compute_class": "gpu",
            "cara_native_extractor": "true",
            "cara_model_ids": ",".join(model_ids),
            "cara_context_trained_model_data": context_model_uri,
            "cara_marketplace": "false",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="stable_audio_attribution_score_",
        dir=_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _evaluation_materialize_musicgen_score_job_file(
    *,
    output_path: str,
    generated_audio_output_path: str,
    trained_model_data: str,
    model_ids: list[str],
) -> Path:
    payload = yaml.safe_load(_EVALUATION_MUSICGEN_SCORE_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"MusicGen attribution scoring YAML is not an object: {_EVALUATION_MUSICGEN_SCORE_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "base_checkpoint": "facebook/musicgen-small",
            "native_extractor": "true",
            "max_native_predictions": 0,
            "duration_seconds": 12.0,
            "top_k": 250,
            "cfg_coef": 3.0,
            "suffix_len": 21,
            "model_ids": ",".join(model_ids),
            "dashboard_triggered": "true",
            "dry_run": "false",
        }
    )
    generated_audio_dir = dict(inputs.get("generated_audio_dir") or {})
    generated_audio_dir["path"] = generated_audio_output_path
    generated_audio_dir["mode"] = "ro_mount"
    generated_audio_dir["type"] = "uri_folder"
    inputs["generated_audio_dir"] = generated_audio_dir
    trained_model = dict(inputs.get("trained_model_data") or {})
    trained_model["path"] = trained_model_data
    trained_model["mode"] = "ro_mount"
    trained_model["type"] = "uri_folder"
    inputs["trained_model_data"] = trained_model
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    output_dir["mode"] = "rw_mount"
    output_dir["type"] = "uri_folder"
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_evaluation_scope": "attribution_scoring",
            "cara_model_family": "musicgen",
            "cara_compute_class": "gpu",
            "cara_native_extractor": "true",
            "cara_model_ids": ",".join(model_ids),
            "cara_marketplace": "false",
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="musicgen_attribution_score_",
        dir=_EVALUATION_MUSICGEN_SCORE_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _training_materialize_ace_preflight_job_file(
    *,
    output_path: str,
    request: TrainingAcePreflightRunRequest,
) -> Path:
    payload = yaml.safe_load(_TRAINING_ACE_PREFLIGHT_JOB_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"ACE-Step preflight YAML is not an object: {_TRAINING_ACE_PREFLIGHT_JOB_FILE}")
    inputs = dict(payload.get("inputs") or {})
    inputs.update(
        {
            "checkpoint": request.checkpoint,
            "load_checkpoint": "true" if request.load_checkpoint else "false",
            "dashboard_triggered": "true",
        }
    )
    payload["inputs"] = inputs
    payload["compute"] = f"azureml:{_TRAINING_H100_COMPUTE}"
    outputs = dict(payload.get("outputs") or {})
    output_dir = dict(outputs.get("output_dir") or {})
    output_dir["path"] = output_path
    outputs["output_dir"] = output_dir
    payload["outputs"] = outputs
    tags = dict(payload.get("tags") or {})
    tags.update(
        {
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "ace_step_env_preflight",
            "cara_model_family": "ace_step",
            "cara_expected_environment": _TRAINING_ACE_ENVIRONMENT,
        }
    )
    payload["tags"] = tags
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".materialized.yml",
        prefix="ace_step_preflight_",
        dir=_TRAINING_ACE_PREFLIGHT_JOB_FILE.parent,
        delete=False,
    )
    with temp:
        yaml.safe_dump(payload, temp, sort_keys=False)
    return Path(temp.name)


def _training_latest_stable_audio_preflight(*, raise_on_error: bool = False, registry_limit: int = 100) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "stable_audio_trainer_preflight_submitted":
            continue
        if event.get("model_family") != "stable_audio_open_small":
            continue
        latest = event
        break
    if latest is None:
        return {
            "passed": False,
            "active": False,
            "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
            "latest_job": None,
            "reason": "Run the Stable Audio trainer preflight before launching the baseline smoke trainer.",
        }
    job_name = str(latest.get("job_name") or "").strip()
    if not job_name:
        return {
            "passed": False,
            "active": False,
            "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
            "latest_job": latest,
            "reason": "The latest Stable Audio trainer preflight event does not include an Azure ML job name.",
        }
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "passed": False,
            "active": False,
            "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
            "latest_job": latest,
            "reason": f"Unable to read latest Stable Audio trainer preflight job {job_name}: {exc}",
        }
    status = str(summary.get("status") or "").lower()
    environment_matches = _training_environment_matches(summary.get("environment"), _TRAINING_STABLE_AUDIO_ENVIRONMENT)
    active = status in _AZUREML_ACTIVE_STATUSES
    passed = status == "completed" and environment_matches
    if passed:
        reason = f"Stable Audio trainer preflight passed on {_TRAINING_STABLE_AUDIO_ENVIRONMENT}."
    elif active:
        reason = f"Stable Audio trainer preflight job {job_name} is {summary.get('status')}; wait for it to complete."
    elif not environment_matches:
        reason = f"Latest Stable Audio trainer preflight used {summary.get('environment') or 'unknown environment'}; rerun preflight on {_TRAINING_STABLE_AUDIO_ENVIRONMENT}."
    else:
        reason = f"Latest Stable Audio trainer preflight job {job_name} ended with status {summary.get('status')}; inspect logs before launching smoke training."
    return {
        "passed": passed,
        "active": active,
        "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
        "latest_job": {
            **summary,
            "checkpoint": latest.get("checkpoint"),
            "wrapper_check": latest.get("wrapper_check"),
            "output_path": latest.get("output_path"),
        },
        "reason": reason,
    }


def _training_latest_context_diffusion_stage(
    *,
    action: str,
    stage: int,
    label: str,
    output_path: str,
    missing_reason: str,
    registry_limit: int = 100,
) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") == action and event.get("model_family") == "stable_audio_open_small_context_diffusion":
            latest = event
            break
    if latest is None:
        return {
            "stage": stage,
            "label": label,
            "passed": False,
            "active": False,
            "latest_job": None,
            "output_path": output_path,
            "reason": missing_reason,
        }
    job_name = str(latest.get("job_name") or "").strip()
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name)) if job_name else {}
    except Exception:
        summary = {
            "name": job_name,
            "status": latest.get("status") or "unknown",
            "studio_url": latest.get("studio_url"),
            "compute": latest.get("compute"),
            "environment": latest.get("environment"),
        }
    status = str(summary.get("status") or latest.get("status") or "").lower()
    active = status in _AZUREML_ACTIVE_STATUSES
    passed = status == "completed"
    if active:
        reason = f"{label} job {job_name} is {summary.get('status')}; wait for Azure ML completion."
    elif passed:
        reason = f"{label} completed; artifacts are expected under {output_path}."
    elif status:
        reason = f"{label} job {job_name} ended with status {summary.get('status')}; inspect logs before continuing."
    else:
        reason = f"{label} was submitted but Azure status has not been observed yet."
    return {
        "stage": stage,
        "label": label,
        "passed": passed,
        "active": active,
        "latest_job": {
            **latest,
            **summary,
            "output_path": latest.get("output_path") or output_path,
        },
        "output_path": latest.get("output_path") or output_path,
        "reason": reason,
    }


def _training_context_diffusion_ladder() -> dict[str, Any]:
    context_packs = _training_latest_context_diffusion_stage(
        action="stable_audio_context_packs_submitted",
        stage=10,
        label="Context pack lock",
        output_path=_TRAINING_CONTEXT_PACK_OUTPUT_URI,
        missing_reason="Submit step 10 to create source-disjoint context packs from the prepared Stable Audio manifest.",
    )
    context_cache = _training_latest_context_diffusion_stage(
        action="stable_audio_context_cache_submitted",
        stage=11,
        label="Context conditioning cache",
        output_path=_TRAINING_CONTEXT_CACHE_OUTPUT_URI,
        missing_reason="Submit step 11 after context packs complete to validate/cache context audio references.",
    )
    context_preflight = _training_latest_context_diffusion_stage(
        action="stable_audio_context_preflight_submitted",
        stage=12,
        label="Context conditioner preflight",
        output_path=_TRAINING_CONTEXT_PREFLIGHT_OUTPUT_URI,
        missing_reason="Submit step 12 after context cache completes to validate the context-conditioner contract.",
    )
    context_smoke = _training_latest_context_diffusion_stage(
        action="stable_audio_context_smoke_submitted",
        stage=13,
        label="Context conditioner smoke",
        output_path=_TRAINING_CONTEXT_SMOKE_OUTPUT_URI,
        missing_reason="Submit step 13 after context preflight completes to validate context/control lanes.",
    )
    context_full = _training_latest_context_diffusion_stage(
        action="stable_audio_context_full_submitted",
        stage=14,
        label="Full Context Diffusion fine-tune",
        output_path=_TRAINING_CONTEXT_FULL_OUTPUT_URI,
        missing_reason="Submit step 14 after context smoke completes to train the context-conditioned Stable Audio branch.",
    )
    next_stage = 10
    next_label = "Lock Context Packs"
    if context_packs.get("active"):
        next_stage, next_label = 10, "Context Packs Running"
    elif context_packs.get("passed") and not context_cache.get("passed"):
        next_stage, next_label = (11, "Cache Context Metadata") if not context_cache.get("active") else (11, "Context Cache Running")
    elif context_packs.get("passed") and context_cache.get("passed") and not context_preflight.get("passed"):
        next_stage, next_label = (12, "Run Context Preflight") if not context_preflight.get("active") else (12, "Context Preflight Running")
    elif context_preflight.get("passed") and not context_smoke.get("passed"):
        next_stage, next_label = (13, "Run Context Smoke") if not context_smoke.get("active") else (13, "Context Smoke Running")
    elif context_smoke.get("passed") and not context_full.get("passed"):
        next_stage, next_label = (14, "Launch Full Context Fine-Tune") if not context_full.get("active") else (14, "Full Context Fine-Tune Running")
    elif context_full.get("passed"):
        next_stage, next_label = 15, "Context Benchmark Pending"
    return {
        "context_packs": context_packs,
        "context_cache": context_cache,
        "context_preflight": context_preflight,
        "context_smoke": context_smoke,
        "context_full": context_full,
        "next_stage": next_stage,
        "next_label": next_label,
        "root_output_path": _TRAINING_CONTEXT_ROOT_URI,
        "trainer_status": (
            "full_context_trainer_completed"
            if context_full.get("passed")
            else "full_context_trainer_available"
            if context_smoke.get("passed")
            else "context_smoke_available_after_preflight"
            if context_preflight.get("passed")
            else "preflight_required_before_context_smoke"
        ),
    }


def _training_latest_musicgen_token_cache(*, registry_limit: int = 100) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") == "musicgen_encodec_cache_submitted" and event.get("model_family") == "musicgen":
            latest = event
            break
    if latest is None:
        return {
            "stage": 6,
            "label": "EnCodec token cache",
            "passed": False,
            "active": False,
            "latest_job": None,
            "reason": "Cache MusicGen EnCodec tokens after the prepared MusicGen dataset is complete.",
        }
    job_name = str(latest.get("job_name") or "").strip()
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except Exception:
        summary = {"name": job_name, "status": latest.get("status") or "unknown", "studio_url": latest.get("studio_url"), "compute": latest.get("compute"), "environment": latest.get("environment")}
    status = str(summary.get("status") or "").lower()
    artifact_summary = _training_musicgen_token_cache_artifact_summary()
    artifact_complete = bool(artifact_summary.get("complete"))
    passed = status == "completed" or artifact_complete
    active = status in _AZUREML_ACTIVE_STATUSES
    return {
        "stage": 6,
        "label": "EnCodec token cache",
        "passed": passed,
        "active": active,
        "latest_job": {**summary, "output_path": latest.get("output_path"), "artifact_summary": artifact_summary},
        "reason": (
            f"MusicGen EnCodec token-cache job {job_name} is {summary.get('status')}."
            if active
            else "MusicGen EnCodec token-cache artifacts are complete; proceeding from the output manifest/resolver/vocab."
            if artifact_complete and status != "completed"
            else "MusicGen EnCodec token cache completed."
            if status == "completed"
            else f"Latest MusicGen EnCodec token-cache job {job_name} ended with status {summary.get('status')}; {artifact_summary.get('reason') or 'final token-cache artifacts are not complete yet.'}"
        ),
    }


def _training_musicgen_token_cache_artifact_summary() -> dict[str, Any]:
    prefix = _azureml_datastore_prefix_from_uri(_TRAINING_MUSICGEN_TOKEN_CACHE_URI)
    if not prefix:
        return {"available": False, "complete": False, "reason": "MusicGen token-cache URI is not an inspectable datastore path."}
    summary_blob = f"{prefix.rstrip('/')}/encodec_cache_summary.json"
    try:
        summary = json.loads(_azureml_datastore_blob_text(summary_blob))
    except Exception as exc:
        return {
            "available": False,
            "complete": False,
            "prefix": prefix,
            "summary_blob": summary_blob,
            "reason": f"Missing final encodec_cache_summary.json ({exc}).",
        }
    cached = int(summary.get("cached_chunk_count") or 0)
    failed = int(summary.get("failed_chunk_count") or 0)
    source = int(summary.get("source_chunk_count") or 0)
    complete = cached > 0 and failed == 0 and (source <= 0 or cached >= source)
    return {
        "available": True,
        "complete": complete,
        "prefix": prefix,
        "summary_blob": summary_blob,
        "cached_chunk_count": cached,
        "failed_chunk_count": failed,
        "source_chunk_count": source,
        "resumed_token_count": int(summary.get("resumed_token_count") or 0),
        "registry_hash": summary.get("cara_registry_hash"),
        "suffix_vocab_hash": summary.get("cara_suffix_vocab_hash"),
        "reason": "Token-cache summary exists and all source chunks are cached." if complete else "Token-cache summary exists but row counts are incomplete or failed rows remain.",
    }


def _training_event_sort_key(event: dict[str, Any]) -> str:
    return str(event.get("created_at") or event.get("submitted_at") or event.get("timestamp") or "")


def _azureml_output_path(job: Any, key: str = "output_dir") -> str | None:
    outputs = getattr(job, "outputs", None) or {}
    value = outputs.get(key) if isinstance(outputs, dict) else None
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "_data"):
        return str(value._data)
    if isinstance(value, dict):
        path = value.get("path") or value.get("uri")
        return str(path) if path else None
    return str(value) if value else None


def _azureml_command_text(job: Any) -> str:
    command = getattr(job, "command", None)
    if command:
        return str(command)
    component = getattr(job, "component", None)
    component_command = getattr(component, "command", None)
    return str(component_command or "")


def _training_musicgen_real_lm_from_azure(job: Any, summary: dict[str, Any], tags: dict[str, Any]) -> bool:
    implementation = str(tags.get("cara_trainer_implementation") or "")
    if implementation == _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION:
        return True
    command = _azureml_command_text(job)
    description = str(summary.get("description") or "")
    return "musicgen_lm_cara_trainer.py" in command or "real MusicGen LM" in description


def _training_musicgen_event_from_azure_job(job: Any) -> dict[str, Any] | None:
    summary = _azureml_job_summary(job)
    tags = dict(summary.get("tags") or {})
    if tags.get("cara_model_family") != "musicgen":
        return None
    gate = str(tags.get("cara_training_gate") or "")
    action_by_gate = {
        "musicgen_trainer_preflight": "musicgen_trainer_preflight_submitted",
        "musicgen_ar_smoke_trainer": "musicgen_ar_smoke_trainer_submitted",
        "musicgen_full_trainer": "musicgen_full_trainer_submitted",
    }
    action = action_by_gate.get(gate)
    if action is None:
        return None
    variant = str(tags.get("cara_variant") or _azureml_input_scalar(job, "variant") or "no_cara_baseline")
    run_name = str(_azureml_input_scalar(job, "run_name") or tags.get("cara_run_name") or "")
    if action == "musicgen_trainer_preflight_submitted":
        variant = ""
        run_name = run_name or "cara-musicgen-preflight"
    implementation = (
        _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION
        if _training_musicgen_real_lm_from_azure(job, summary, tags)
        else str(tags.get("cara_trainer_implementation") or "")
    )
    max_steps_raw = tags.get("cara_max_steps") or _azureml_input_scalar(job, "max_steps")
    batch_size_raw = tags.get("cara_batch_size") or _azureml_input_scalar(job, "batch_size")
    learning_rate_raw = tags.get("cara_learning_rate") or _azureml_input_scalar(job, "learning_rate")
    attribution_loss_raw = tags.get("cara_attribution_loss_weight") or _azureml_input_scalar(job, "attribution_loss_weight")
    return {
        "action": action,
        "job_name": summary.get("name"),
        "studio_url": summary.get("studio_url"),
        "display_name": summary.get("display_name"),
        "created_at": summary.get("created_at"),
        "status": summary.get("status"),
        "compute": _azureml_resource_tail(summary.get("compute")),
        "environment": summary.get("environment"),
        "model_family": "musicgen",
        "trainer_implementation": implementation or None,
        "job_file": str(_TRAINING_MUSICGEN_FULL_JOB_FILE.relative_to(ROOT)) if action == "musicgen_full_trainer_submitted" else str(_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE.relative_to(ROOT)) if action == "musicgen_trainer_preflight_submitted" else str(_TRAINING_MUSICGEN_SMOKE_JOB_FILE.relative_to(ROOT)),
        "output_path": _azureml_output_path(job) or tags.get("cara_output_path"),
        "run_name": run_name,
        "variant": variant,
        "training_scope": tags.get("cara_training_scope") or ("full" if action == "musicgen_full_trainer_submitted" else "smoke"),
        "max_steps": int(max_steps_raw) if str(max_steps_raw or "").isdigit() else max_steps_raw,
        "batch_size": int(batch_size_raw) if str(batch_size_raw or "").isdigit() else batch_size_raw,
        "learning_rate": float(learning_rate_raw) if str(learning_rate_raw or "").strip() else None,
        "attribution_loss_weight": float(attribution_loss_raw) if str(attribution_loss_raw or "").strip() else None,
        "checkpoint": _azureml_input_scalar(job, "checkpoint") or tags.get("cara_checkpoint"),
        "model_dtype": _azureml_input_scalar(job, "model_dtype") or tags.get("cara_model_dtype"),
        "source": "azureml_job_tags",
    }


def _training_musicgen_events_with_azure_fallback(
    *,
    registry_limit: int = 100,
    azure_limit: int = 100,
    actions: set[str] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("model_family") != "musicgen":
            continue
        action = str(event.get("action") or "")
        if actions is not None and action not in actions:
            continue
        job_name = str(event.get("job_name") or "")
        if job_name:
            seen.add(job_name)
        events.append(event)
    try:
        for _, job in zip(range(max(1, min(azure_limit, 500))), _azureml_client().jobs.list()):
            event = _training_musicgen_event_from_azure_job(job)
            if event is None:
                continue
            action = str(event.get("action") or "")
            if actions is not None and action not in actions:
                continue
            job_name = str(event.get("job_name") or "")
            if job_name and job_name in seen:
                continue
            if job_name:
                seen.add(job_name)
            events.append(event)
    except Exception:
        pass
    return sorted(events, key=_training_event_sort_key, reverse=True)


def _training_latest_musicgen_preflight(*, registry_limit: int = 100) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for event in _training_musicgen_events_with_azure_fallback(
        registry_limit=registry_limit,
        azure_limit=100,
        actions={"musicgen_trainer_preflight_submitted"},
    ):
        if event.get("action") == "musicgen_trainer_preflight_submitted" and event.get("model_family") == "musicgen":
            latest = event
            break
    if latest is None:
        return {
            "stage": 7,
            "label": "MusicGen trainer preflight",
            "passed": False,
            "active": False,
            "required_environment": _TRAINING_MUSICGEN_ENVIRONMENT,
            "latest_job": None,
            "reason": "Run after the EnCodec token-cache manifest is available.",
        }
    job_name = str(latest.get("job_name") or "").strip()
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except Exception:
        summary = {"name": job_name, "status": latest.get("status") or "unknown", "studio_url": latest.get("studio_url"), "compute": latest.get("compute"), "environment": latest.get("environment")}
    status = str(summary.get("status") or "").lower()
    environment_matches = _training_environment_matches(summary.get("environment") or latest.get("environment"), _TRAINING_MUSICGEN_ENVIRONMENT)
    implementation_matches = latest.get("trainer_implementation") == _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION
    return {
        "stage": 7,
        "label": "MusicGen trainer preflight",
        "passed": status == "completed" and environment_matches and implementation_matches,
        "active": status in _AZUREML_ACTIVE_STATUSES,
        "required_environment": _TRAINING_MUSICGEN_ENVIRONMENT,
        "latest_job": {
            **summary,
            "checkpoint": latest.get("checkpoint"),
            "output_path": latest.get("output_path"),
            "trainer_implementation": latest.get("trainer_implementation"),
            "real_lm_trainer": implementation_matches,
        },
        "reason": (
            f"MusicGen trainer preflight job {job_name} is {summary.get('status')}."
            if status in _AZUREML_ACTIVE_STATUSES
            else "Latest MusicGen trainer preflight predates the real MusicGen LM trainer and must be rerun."
            if status == "completed" and environment_matches and not implementation_matches
            else f"MusicGen trainer preflight passed on {_TRAINING_MUSICGEN_ENVIRONMENT}."
            if status == "completed" and environment_matches and implementation_matches
            else f"Latest MusicGen trainer preflight job {job_name} ended with status {summary.get('status')}."
        ),
    }


def _training_latest_ace_preflight(*, registry_limit: int = 100) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") == "ace_step_env_preflight_submitted" and event.get("model_family") == "ace_step":
            latest = event
            break
    if latest is None:
        return {
            "stage": 2,
            "label": "ACE environment preflight",
            "passed": False,
            "active": False,
            "required_environment": _TRAINING_ACE_ENVIRONMENT,
            "latest_job": None,
            "reason": "Run the ACE-Step environment preflight before planner/tensor probes.",
        }
    job_name = str(latest.get("job_name") or "").strip()
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except Exception:
        summary = {
            "name": job_name,
            "status": latest.get("status") or "unknown",
            "studio_url": latest.get("studio_url"),
            "compute": latest.get("compute"),
            "environment": latest.get("environment"),
        }
    status = str(summary.get("status") or "").lower()
    environment_matches = _training_environment_matches(summary.get("environment") or latest.get("environment"), _TRAINING_ACE_ENVIRONMENT)
    active = status in _AZUREML_ACTIVE_STATUSES
    passed = status == "completed" and environment_matches
    if active:
        reason = f"ACE-Step environment preflight job {job_name} is {summary.get('status')}."
    elif passed:
        reason = f"ACE-Step environment preflight passed on {_TRAINING_ACE_ENVIRONMENT}."
    elif not environment_matches:
        reason = f"Latest ACE-Step preflight used {summary.get('environment') or 'unknown environment'}; rerun on {_TRAINING_ACE_ENVIRONMENT}."
    else:
        reason = f"Latest ACE-Step environment preflight job {job_name} ended with status {summary.get('status')}; inspect logs before continuing."
    return {
        "stage": 2,
        "label": "ACE environment preflight",
        "passed": passed,
        "active": active,
        "required_environment": _TRAINING_ACE_ENVIRONMENT,
        "latest_job": {
            **summary,
            "checkpoint": latest.get("checkpoint"),
            "load_checkpoint": latest.get("load_checkpoint"),
            "output_path": latest.get("output_path"),
        },
        "reason": reason,
    }


def _training_ace_ladder(preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    preflight = preflight or _training_latest_ace_preflight()
    steps = [
        {
            "stage": 1,
            "label": "ACE source + license review",
            "passed": True,
            "active": False,
            "reason": "Official ACE-Step v1.5 sources and the no-Marketplace Azure cost guardrail are documented.",
        },
        preflight,
        {
            "stage": 3,
            "label": "Prepare ACE tensors",
            "passed": False,
            "active": False,
            "locked": not bool(preflight.get("passed")),
            "reason": "Locked until ACE environment preflight passes. This stage will produce ACE-ready audio/tensor records with CARA registry labels.",
        },
        {
            "stage": 4,
            "label": "Planner survival probe",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until ACE tensors exist; measures exact, repairable, lost, and hallucinated CARA survival through the LM planner.",
        },
        {
            "stage": 5,
            "label": "DiT tap discovery",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until the ACE source path exposes stable mid/late DiT hidden-state hooks.",
        },
        {
            "stage": 6,
            "label": "Baseline LoRA smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until preflight, tensor preparation, and tap discovery are validated.",
        },
        {
            "stage": 7,
            "label": "CARA-lite planner smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until the same-data baseline LoRA smoke passes.",
        },
        {
            "stage": 8,
            "label": "Detached DiT head smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until CARA-lite planner survival controls are complete.",
        },
        {
            "stage": 9,
            "label": "Planner-preserved CARA smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until detached DiT head evidence exists.",
        },
        {
            "stage": 10,
            "label": "Planner-bypass CARA smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until planner-preserved and DiT-only attribution paths can be separated.",
        },
        {
            "stage": 11,
            "label": "Hybrid CARA-Strong smoke",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until planner and DiT attribution controls are complete.",
        },
        {
            "stage": 12,
            "label": "Full hybrid comparison",
            "passed": False,
            "active": False,
            "locked": True,
            "reason": "Locked until the Hybrid CARA-Strong smoke passes.",
        },
    ]
    active = next((step for step in steps if step.get("active")), None)
    next_step = active or next((step for step in steps if not step.get("passed")), steps[-1])
    return {
        "steps": steps,
        "next_stage": next_step["stage"],
        "next_label": next_step["label"],
        "reason": next_step.get("reason"),
    }


def _training_recent_h100_jobs_from_registry(*, registry_limit: int = 80) -> list[dict[str, Any]]:
    active_jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in reversed(_training_job_registry_events(registry_limit)):
        compute = _azureml_resource_tail(event.get("compute"))
        if compute not in _TRAINING_H100_COMPUTES:
            continue
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen:
            continue
        seen.add(job_name)
        status = str(event.get("status") or "").lower()
        if status and status not in _AZUREML_ACTIVE_STATUSES:
            continue
        if not status:
            continue
        active_jobs.append(
            {
                "name": job_name,
                "status": event.get("status"),
                "display_name": event.get("display_name"),
                "studio_url": event.get("studio_url"),
                "compute": event.get("compute"),
                "h100_compute_target": compute,
                "model_family": event.get("model_family"),
                "variant": event.get("variant"),
                "training_scope": event.get("training_scope"),
                "output_path": event.get("output_path"),
                "source": "local_registry_event",
            }
        )
    return active_jobs


def _training_active_musicgen_trainer_jobs(*, raise_on_error: bool = False, registry_limit: int = 50) -> list[dict[str, Any]]:
    active_jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in _training_musicgen_events_with_azure_fallback(
        registry_limit=registry_limit,
        azure_limit=100,
        actions={"musicgen_ar_smoke_trainer_submitted", "musicgen_full_trainer_submitted", "musicgen_trainer_preflight_submitted"},
    ):
        if event.get("action") not in {"musicgen_ar_smoke_trainer_submitted", "musicgen_full_trainer_submitted", "musicgen_trainer_preflight_submitted"}:
            continue
        if event.get("model_family") != "musicgen":
            continue
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen:
            continue
        seen.add(job_name)
        try:
            summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
        except Exception:
            if raise_on_error:
                raise
            continue
        if str(summary.get("status") or "").lower() in _AZUREML_ACTIVE_STATUSES:
            active_jobs.append({**summary, "run_name": event.get("run_name"), "variant": event.get("variant"), "training_scope": event.get("training_scope") or "smoke", "output_path": event.get("output_path")})
    return active_jobs


def _training_musicgen_smoke_sequence(*, registry_limit: int = 100) -> dict[str, Any]:
    variants = {
        "no_cara_baseline": {"stage": 8, "label": "MusicGen baseline smoke", "passed": False, "active": False, "latest_job": None, "latest_passed_job": None, "reason": "Run the no-CARA MusicGen baseline smoke after preflight."},
        "cara_lite": {"stage": 9, "label": "MusicGen CARA-lite smoke", "passed": False, "active": False, "latest_job": None, "latest_passed_job": None, "reason": "Run MusicGen CARA-lite after the baseline smoke has passed."},
        "cara_probe": {"stage": 10, "label": "MusicGen CARA suffix-probe smoke", "passed": False, "active": False, "latest_job": None, "latest_passed_job": None, "reason": "Run the detached suffix probe after CARA-lite has passed."},
        "cara_strong": {"stage": 11, "label": "MusicGen CARA-Strong smoke", "passed": False, "active": False, "latest_job": None, "latest_passed_job": None, "reason": "Run MusicGen CARA-Strong after the detached suffix probe has passed."},
    }
    seen_jobs: set[str] = set()
    for event in _training_musicgen_events_with_azure_fallback(
        registry_limit=registry_limit,
        azure_limit=200,
        actions={"musicgen_ar_smoke_trainer_submitted"},
    ):
        if event.get("action") != "musicgen_ar_smoke_trainer_submitted" or event.get("model_family") != "musicgen":
            continue
        variant = str(event.get("variant") or "no_cara_baseline")
        if variant not in variants:
            continue
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen_jobs:
            continue
        seen_jobs.add(job_name)
        azure_job = None
        try:
            azure_job = _azureml_client().jobs.get(job_name)
            summary = _azureml_job_summary(azure_job)
        except Exception:
            summary = {"name": job_name, "status": event.get("status") or "unknown", "studio_url": event.get("studio_url"), "compute": event.get("compute"), "environment": event.get("environment")}
        status = str(summary.get("status") or "").lower()
        environment_matches = _training_environment_matches(summary.get("environment") or event.get("environment"), _TRAINING_MUSICGEN_ENVIRONMENT)
        implementation_matches = event.get("trainer_implementation") == _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION
        serialized_variant = str(_azureml_input_scalar(azure_job, "variant") or "") if azure_job is not None else ""
        serialized_dashboard_triggered = str(_azureml_input_scalar(azure_job, "dashboard_triggered") or "").lower() if azure_job is not None else ""
        serialized_run_name = str(_azureml_input_scalar(azure_job, "run_name") or "") if azure_job is not None else ""
        requested_run_name = str(event.get("run_name") or "")
        command_inputs_match = serialized_variant == variant and serialized_dashboard_triggered == "true" and (not requested_run_name or serialized_run_name == requested_run_name)
        if azure_job is None:
            command_inputs_match = bool(event.get("dashboard_triggered")) and bool(requested_run_name) and str(event.get("variant") or "") == variant
        enriched = {
            **summary,
            "created_at": summary.get("created_at") or event.get("created_at"),
            "run_name": requested_run_name,
            "variant": variant,
            "output_path": event.get("output_path"),
            "max_steps": event.get("max_steps"),
            "batch_size": event.get("batch_size"),
            "learning_rate": event.get("learning_rate"),
            "trainer_implementation": event.get("trainer_implementation"),
            "real_lm_trainer": implementation_matches,
            "command_inputs": {
                "variant": serialized_variant or None,
                "dashboard_triggered": serialized_dashboard_triggered or None,
                "run_name": serialized_run_name or None,
                "match_registry_event": command_inputs_match,
            },
        }
        current = variants[variant]
        if current.get("latest_job") is None:
            current["latest_job"] = enriched
        if status in _AZUREML_ACTIVE_STATUSES:
            current["active"] = True
            current["latest_job"] = enriched
            current["reason"] = f"{current['label']} job {job_name} is {summary.get('status')}."
        if status == "completed" and environment_matches and command_inputs_match and implementation_matches:
            current["passed"] = True
            if current.get("latest_passed_job") is None:
                current["latest_passed_job"] = enriched
            if current.get("latest_job") is enriched:
                current["reason"] = f"{current['label']} passed on {_TRAINING_MUSICGEN_ENVIRONMENT}."
        elif current.get("latest_job") is enriched and status == "completed" and environment_matches and not implementation_matches:
            current["reason"] = f"Latest {current['label']} job {job_name} predates the real MusicGen LM trainer and does not count as passed."
        elif current.get("latest_job") is enriched and status in {"failed", "canceled", "cancelled"}:
            current["reason"] = f"Latest {current['label']} job {job_name} ended with status {summary.get('status')}; it does not count as passed."

    if not variants["no_cara_baseline"]["passed"]:
        next_variant = "no_cara_baseline"
    elif not variants["cara_lite"]["passed"]:
        next_variant = "cara_lite"
    elif not variants["cara_probe"]["passed"]:
        next_variant = "cara_probe"
    elif not variants["cara_strong"]["passed"]:
        next_variant = "cara_strong"
    else:
        return {
            "variants": variants,
            "next_variant": "full_cara_strong",
            "next_stage": 12,
            "next_label": "Full MusicGen LM CARA-Strong fine-tune",
            "reason": "MusicGen real-LM smoke ladder passed; Step 12 full CARA-Strong fine-tune is unlocked.",
        }
    return {
        "variants": variants,
        "next_variant": next_variant,
        "next_stage": variants[next_variant]["stage"],
        "next_label": variants[next_variant]["label"],
        "reason": variants[next_variant]["reason"],
    }


def _training_active_stable_audio_trainer_jobs(*, raise_on_error: bool = False, registry_limit: int = 50) -> list[dict[str, Any]]:
    active_jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    events = _training_job_registry_events(registry_limit)
    for event in reversed(events):
        if event.get("action") not in {"stable_audio_smoke_trainer_submitted", "stable_audio_full_trainer_submitted"}:
            continue
        if event.get("model_family") != "stable_audio_open_small":
            continue
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen:
            continue
        seen.add(job_name)
        try:
            summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
        except Exception:
            if raise_on_error:
                raise
            continue
        status = str(summary.get("status") or "").lower()
        if status in _AZUREML_ACTIVE_STATUSES:
            active_jobs.append(
                {
                    **summary,
                    "run_name": event.get("run_name"),
                    "variant": event.get("variant"),
                    "training_scope": event.get("training_scope") or "smoke",
                    "output_path": event.get("output_path"),
                }
            )
    return active_jobs


def _training_active_stable_audio_smoke_jobs(*, raise_on_error: bool = False, registry_limit: int = 50) -> list[dict[str, Any]]:
    return [
        job for job in _training_active_stable_audio_trainer_jobs(raise_on_error=raise_on_error, registry_limit=registry_limit)
        if str(job.get("training_scope") or "smoke") == "smoke"
    ]


def _training_stable_audio_smoke_sequence(*, registry_limit: int = 100) -> dict[str, Any]:
    variants = {
        "no_cara_baseline": {
            "stage": 5,
            "label": "Baseline smoke",
            "passed": False,
            "active": False,
            "latest_job": None,
            "latest_passed_job": None,
            "reason": "Run the no-CARA baseline smoke before CARA-lite.",
        },
        "cara_lite": {
            "stage": 6,
            "label": "CARA-lite smoke",
            "passed": False,
            "active": False,
            "latest_job": None,
            "latest_passed_job": None,
            "reason": "Run CARA-lite after the baseline smoke has passed.",
        },
        "cara_head": {
            "stage": 7,
            "label": "CARA attribution-head smoke",
            "passed": False,
            "active": False,
            "implemented": True,
            "latest_job": None,
            "latest_passed_job": None,
            "reason": "Run the detached CARA attribution-head smoke after CARA-lite has passed.",
        },
        "cara_strong": {
            "stage": 8,
            "label": "CARA-Strong smoke",
            "passed": False,
            "active": False,
            "implemented": True,
            "latest_job": None,
            "latest_passed_job": None,
            "reason": "Run CARA-Strong after the detached attribution-head smoke has passed.",
        },
    }
    seen_jobs: set[str] = set()
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "stable_audio_smoke_trainer_submitted":
            continue
        if event.get("model_family") != "stable_audio_open_small":
            continue
        variant = str(event.get("variant") or "no_cara_baseline")
        if variant not in variants:
            continue
        job_name = str(event.get("job_name") or "").strip()
        if not job_name or job_name in seen_jobs:
            continue
        seen_jobs.add(job_name)
        azure_job = None
        try:
            azure_job = _azureml_client().jobs.get(job_name)
            summary = _azureml_job_summary(azure_job)
        except Exception:
            summary = {
                "name": job_name,
                "status": event.get("status") or "unknown",
                "studio_url": event.get("studio_url"),
                "compute": event.get("compute"),
                "environment": event.get("environment"),
            }
        status = str(summary.get("status") or "").lower()
        environment_matches = _training_environment_matches(summary.get("environment") or event.get("environment"), _TRAINING_STABLE_AUDIO_ENVIRONMENT)
        enriched = {
            **summary,
            "created_at": summary.get("created_at") or event.get("created_at"),
            "run_name": event.get("run_name"),
            "variant": variant,
            "output_path": event.get("output_path"),
            "max_steps": event.get("max_steps"),
            "batch_size": event.get("batch_size"),
            "num_workers": event.get("num_workers"),
            "learning_rate": event.get("learning_rate"),
        }
        serialized_variant = str(_azureml_input_scalar(azure_job, "variant") or "") if azure_job is not None else ""
        serialized_dashboard_triggered = str(_azureml_input_scalar(azure_job, "dashboard_triggered") or "").lower() if azure_job is not None else ""
        serialized_run_name = str(_azureml_input_scalar(azure_job, "run_name") or "") if azure_job is not None else ""
        requested_run_name = str(event.get("run_name") or "")
        baseline_command_inputs_match = variant == "no_cara_baseline" and serialized_variant == "no_cara_baseline"
        strict_command_inputs_match = (
            serialized_variant == variant
            and serialized_dashboard_triggered == "true"
            and (not requested_run_name or serialized_run_name == requested_run_name)
        )
        command_inputs_match = baseline_command_inputs_match or strict_command_inputs_match
        enriched["command_inputs"] = {
            "variant": serialized_variant or None,
            "dashboard_triggered": serialized_dashboard_triggered or None,
            "run_name": serialized_run_name or None,
            "match_registry_event": command_inputs_match,
            "strict_match_registry_event": strict_command_inputs_match,
        }
        current = variants[variant]
        if current.get("latest_job") is None:
            current["latest_job"] = enriched
        if status in _AZUREML_ACTIVE_STATUSES:
            current["active"] = True
            current["latest_job"] = enriched
            current["reason"] = f"{current['label']} job {job_name} is {summary.get('status')}."
        if status == "completed" and environment_matches and command_inputs_match:
            current["passed"] = True
            if current.get("latest_passed_job") is None:
                current["latest_passed_job"] = enriched
            if current.get("latest_job") is enriched:
                current["reason"] = f"{current['label']} passed on {_TRAINING_STABLE_AUDIO_ENVIRONMENT}."
        elif status == "completed" and environment_matches and not command_inputs_match and current.get("latest_job") is enriched:
            current["latest_job"] = enriched
            current["reason"] = (
                f"Latest {current['label']} job {job_name} completed, but its Azure command inputs do not match the registry event; "
                f"got variant={serialized_variant or 'missing'}, dashboard_triggered={serialized_dashboard_triggered or 'missing'}, "
                f"run_name={serialized_run_name or 'missing'}. It does not count as passed."
            )
        elif current.get("latest_job") is enriched and status in {"failed", "canceled", "cancelled"}:
            current["reason"] = f"Latest {current['label']} job {job_name} ended with status {summary.get('status')}; it does not count as passed."

    baseline_passed = bool(variants["no_cara_baseline"]["passed"])
    cara_lite_passed = bool(variants["cara_lite"]["passed"])
    cara_head_passed = bool(variants["cara_head"]["passed"])
    if not baseline_passed:
        next_variant = "no_cara_baseline"
    elif not cara_lite_passed:
        next_variant = "cara_lite"
    elif not cara_head_passed:
        next_variant = "cara_head"
    else:
        next_variant = "cara_strong"
    return {
        "variants": variants,
        "next_variant": next_variant,
        "next_stage": variants[next_variant]["stage"],
        "next_label": variants[next_variant]["label"],
        "reason": variants[next_variant]["reason"],
    }


def _training_latest_stable_audio_full_training(*, registry_limit: int = 100) -> dict[str, Any]:
    latest_event: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "stable_audio_full_trainer_submitted":
            continue
        if event.get("model_family") != "stable_audio_open_small":
            continue
        latest_event = event
        break
    if latest_event is None:
        return {
            "stage": 9,
            "label": "Full CARA-Strong fine-tune",
            "passed": False,
            "active": False,
            "latest_job": None,
            "reason": "Run after steps 05-08 smoke tests have passed. The full run uses all train rows and held-out validation/test CARA metrics.",
        }
    job_name = str(latest_event.get("job_name") or "").strip()
    azure_job = None
    try:
        azure_job = _azureml_client().jobs.get(job_name)
        summary = _azureml_job_summary(azure_job)
    except Exception:
        summary = {
            "name": job_name,
            "status": latest_event.get("status") or "unknown",
            "studio_url": latest_event.get("studio_url"),
            "compute": latest_event.get("compute"),
            "environment": latest_event.get("environment"),
        }
    status = str(summary.get("status") or "").lower()
    serialized_variant = str(_azureml_input_scalar(azure_job, "variant") or "") if azure_job is not None else ""
    serialized_scope = str(_azureml_input_scalar(azure_job, "training_scope") or "") if azure_job is not None else ""
    serialized_dashboard_triggered = str(_azureml_input_scalar(azure_job, "dashboard_triggered") or "").lower() if azure_job is not None else ""
    serialized_run_name = str(_azureml_input_scalar(azure_job, "run_name") or "") if azure_job is not None else ""
    requested_run_name = str(latest_event.get("run_name") or "")
    environment_matches = _training_environment_matches(summary.get("environment") or latest_event.get("environment"), _TRAINING_STABLE_AUDIO_ENVIRONMENT)
    command_inputs_match = (
        serialized_variant == "cara_strong"
        and serialized_scope == "full"
        and serialized_dashboard_triggered == "true"
        and (not requested_run_name or serialized_run_name == requested_run_name)
    )
    enriched = {
        **summary,
        "run_name": latest_event.get("run_name"),
        "variant": latest_event.get("variant"),
        "training_scope": latest_event.get("training_scope") or "full",
        "output_path": latest_event.get("output_path"),
        "max_steps": latest_event.get("max_steps"),
        "batch_size": latest_event.get("batch_size"),
        "learning_rate": latest_event.get("learning_rate"),
        "command_inputs": {
            "variant": serialized_variant or None,
            "training_scope": serialized_scope or None,
            "dashboard_triggered": serialized_dashboard_triggered or None,
            "run_name": serialized_run_name or None,
            "match_registry_event": command_inputs_match,
        },
    }
    active = status in _AZUREML_ACTIVE_STATUSES
    passed = status == "completed" and environment_matches and command_inputs_match
    if active:
        reason = f"Full CARA-Strong fine-tune job {job_name} is {summary.get('status')}."
    elif passed:
        reason = f"Full CARA-Strong fine-tune completed on {_TRAINING_STABLE_AUDIO_ENVIRONMENT}."
    elif status == "completed" and not command_inputs_match:
        reason = "Latest full fine-tune completed but serialized Azure command inputs do not match the dashboard request, so it does not count as valid evidence."
    else:
        reason = f"Latest full fine-tune job {job_name} ended with status {summary.get('status')}; inspect outputs before using it as evidence."
    return {
        "stage": 9,
        "label": "Full CARA-Strong fine-tune",
        "passed": passed,
        "active": active,
        "latest_job": enriched,
        "reason": reason,
    }


def _training_latest_stable_audio_full_training_cached(*, registry_limit: int = 500) -> dict[str, Any]:
    """Fast local evidence for follow-on branches that only need the completed model URI.

    The full Diffusion gate can ask Azure ML for live status, but the Context
    Diffusion branch should not be locked merely because that live lookup is
    slow. Benchmarking already treats this trained model URI as the completed
    Stable Audio CARA-Strong candidate.
    """
    latest_event: dict[str, Any] | None = None
    trained_event: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "stable_audio_full_trainer_submitted":
            continue
        if event.get("model_family") != "stable_audio_open_small":
            continue
        if latest_event is None:
            latest_event = event
        output_path = str(event.get("output_path") or "")
        job_name = str(event.get("job_name") or "")
        if output_path == _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI or job_name == _EVALUATION_STABLE_AUDIO_TRAINED_JOB_NAME:
            trained_event = event
            break
    selected = trained_event or latest_event
    if selected is None:
        return {
            "stage": 9,
            "label": "Full CARA-Strong fine-tune",
            "passed": False,
            "active": False,
            "latest_job": None,
            "reason": "No Stable Audio full-run registry event is available yet.",
            "source": "local_registry_cache",
        }
    output_path = str(selected.get("output_path") or "")
    job_name = str(selected.get("job_name") or "")
    status = str(selected.get("status") or "").lower()
    matches_locked_candidate = (
        output_path == _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI
        or job_name == _EVALUATION_STABLE_AUDIO_TRAINED_JOB_NAME
    )
    passed = matches_locked_candidate or status == "completed"
    reason = (
        f"Full CARA-Strong fine-tune is available from cached trained-model evidence: {output_path or job_name}."
        if passed
        else f"Latest full fine-tune registry event is {job_name or 'unknown'}, but it is not the locked trained-model candidate."
    )
    return {
        "stage": 9,
        "label": "Full CARA-Strong fine-tune",
        "passed": passed,
        "active": False,
        "latest_job": {
            **selected,
            "name": job_name,
            "status": selected.get("status") or ("Completed" if passed else "unknown"),
            "output_path": output_path,
        },
        "reason": reason,
        "source": "local_registry_cache",
    }


def _training_latest_musicgen_full_training(*, registry_limit: int = 100) -> dict[str, Any]:
    latest_event: dict[str, Any] | None = None
    for event in reversed(_training_job_registry_events(registry_limit)):
        if event.get("action") != "musicgen_full_trainer_submitted":
            continue
        if event.get("model_family") != "musicgen":
            continue
        latest_event = event
        break
    if latest_event is None:
        return {
            "stage": 12,
            "label": "Full MusicGen LM CARA-Strong fine-tune",
            "passed": False,
            "active": False,
            "latest_job": None,
            "reason": "Run after the MusicGen real-LM smoke ladder has passed.",
        }
    job_name = str(latest_event.get("job_name") or "").strip()
    azure_job = None
    try:
        azure_job = _azureml_client().jobs.get(job_name)
        summary = _azureml_job_summary(azure_job)
    except Exception:
        summary = {
            "name": job_name,
            "status": latest_event.get("status") or "unknown",
            "studio_url": latest_event.get("studio_url"),
            "compute": latest_event.get("compute"),
            "environment": latest_event.get("environment"),
            "tags": latest_event.get("tags") or {},
        }
    status = str(summary.get("status") or "").lower()
    tags = dict(summary.get("tags") or {})
    serialized_variant = str(_azureml_input_scalar(azure_job, "variant") or "") if azure_job is not None else str(latest_event.get("variant") or "")
    serialized_scope = str(_azureml_input_scalar(azure_job, "training_scope") or "") if azure_job is not None else str(latest_event.get("training_scope") or "full")
    serialized_dashboard_triggered = str(_azureml_input_scalar(azure_job, "dashboard_triggered") or "").lower() if azure_job is not None else str(latest_event.get("dashboard_triggered") or "").lower()
    serialized_run_name = str(_azureml_input_scalar(azure_job, "run_name") or "") if azure_job is not None else str(latest_event.get("run_name") or "")
    requested_run_name = str(latest_event.get("run_name") or "")
    environment_matches = _training_environment_matches(summary.get("environment") or latest_event.get("environment"), _TRAINING_MUSICGEN_ENVIRONMENT)
    implementation_matches = (
        _training_musicgen_real_lm_from_azure(azure_job, summary, tags)
        if azure_job is not None
        else latest_event.get("trainer_implementation") == _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION
    )
    command_inputs_match = (
        serialized_variant == "cara_strong"
        and serialized_scope == "full"
        and serialized_dashboard_triggered == "true"
        and (not requested_run_name or serialized_run_name == requested_run_name)
    )
    enriched = {
        **summary,
        "run_name": latest_event.get("run_name"),
        "variant": latest_event.get("variant"),
        "training_scope": latest_event.get("training_scope") or "full",
        "output_path": latest_event.get("output_path"),
        "max_steps": latest_event.get("max_steps"),
        "batch_size": latest_event.get("batch_size"),
        "learning_rate": latest_event.get("learning_rate"),
        "trainer_implementation": _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION if implementation_matches else latest_event.get("trainer_implementation"),
        "command_inputs": {
            "variant": serialized_variant or None,
            "training_scope": serialized_scope or None,
            "dashboard_triggered": serialized_dashboard_triggered or None,
            "run_name": serialized_run_name or None,
            "match_registry_event": command_inputs_match,
        },
    }
    active = status in _AZUREML_ACTIVE_STATUSES
    passed = status == "completed" and environment_matches and command_inputs_match and implementation_matches
    if active:
        reason = f"Full MusicGen CARA-Strong fine-tune job {job_name} is {summary.get('status')}."
    elif passed:
        reason = f"Full MusicGen CARA-Strong fine-tune completed on {_TRAINING_MUSICGEN_ENVIRONMENT}."
    elif status == "completed" and not implementation_matches:
        reason = "Latest full MusicGen fine-tune completed but does not use the real MusicGen LM trainer, so it does not count as equivalent evidence."
    elif status == "completed" and not command_inputs_match:
        reason = "Latest full MusicGen fine-tune completed but serialized Azure command inputs do not match the dashboard request, so it does not count as valid evidence."
    else:
        reason = f"Latest full MusicGen fine-tune job {job_name} ended with status {summary.get('status')}; inspect outputs before using it as evidence."
    return {
        "stage": 12,
        "label": "Full MusicGen LM CARA-Strong fine-tune",
        "passed": passed,
        "active": active,
        "latest_job": enriched,
        "reason": reason,
    }


def _training_metric_observed_step(metrics: dict[str, Any]) -> int | None:
    histories = metrics.get("histories") if isinstance(metrics, dict) else None
    observed = 0
    if isinstance(histories, dict):
        for points in histories.values():
            if not isinstance(points, list):
                continue
            for point in points:
                try:
                    observed = max(observed, int(point.get("step") or 0))
                except (AttributeError, TypeError, ValueError):
                    continue
    latest = metrics.get("latest") if isinstance(metrics, dict) else None
    if isinstance(latest, dict):
        for key in ("global_step", "trainer/global_step", "step"):
            try:
                value = latest.get(key)
                if value is not None and math.isfinite(float(value)):
                    observed = max(observed, int(float(value)))
            except (TypeError, ValueError):
                continue
    return observed if observed > 0 else None


def _training_progress_model_key(models: str | None) -> str:
    normalized = ",".join(part.strip() for part in str(models or "latest").split(",") if part.strip()).lower()
    aliases = {
        "": "latest",
        "latest": "latest",
        "active": "latest",
        "all": "all",
        "stable": "stable_audio_open_small",
        "stable_audio": "stable_audio_open_small",
        "stable_audio_open_small": "stable_audio_open_small",
        "diffusion": "stable_audio_open_small",
        "context": "stable_audio_open_small_context_diffusion",
        "context_diffusion": "stable_audio_open_small_context_diffusion",
        "stable_audio_context": "stable_audio_open_small_context_diffusion",
        "stable_audio_open_small_context_diffusion": "stable_audio_open_small_context_diffusion",
        "musicgen": "musicgen",
        "autoregressive": "musicgen",
        "ar": "musicgen",
        "ace": "ace_step",
        "ace_step": "ace_step",
        "hybrid": "ace_step",
    }
    key = aliases.get(normalized)
    if key is None:
        raise HTTPException(status_code=400, detail=f"Unsupported training progress model selection: {models}")
    return key


def _training_progress_event_model_key(event: dict[str, Any]) -> str:
    family = str(event.get("model_family") or "").strip()
    if family:
        return family
    action = str(event.get("action") or "")
    if action.startswith("stable_audio_context_"):
        return "stable_audio_open_small_context_diffusion"
    if action.startswith("stable_audio_"):
        return "stable_audio_open_small"
    if action.startswith("musicgen_"):
        return "musicgen"
    if action.startswith("ace_step_"):
        return "ace_step"
    return "unknown"


def _training_progress_event_scope(event: dict[str, Any]) -> str:
    explicit = str(event.get("training_scope") or "").strip()
    if explicit:
        return explicit
    action = str(event.get("action") or "")
    if "full" in action:
        return "full"
    if "smoke" in action:
        return "smoke"
    return "training"


def _training_progress_event_is_candidate(event: dict[str, Any]) -> bool:
    action = str(event.get("action") or "")
    if action in _TRAINING_RUN_PROGRESS_ACTIONS:
        return True
    # Future branches should still be observable if they follow the registry
    # contract: a submitted trainer event with job_name and max_steps.
    return action.endswith("_trainer_submitted") and event.get("job_name") and event.get("max_steps")


def _training_count_prepared_train_chunks(max_train_files: int, *, model_key: str = "stable_audio_open_small") -> dict[str, Any]:
    if model_key == "musicgen":
        token_prefix = f"{(_azureml_uri_to_blob_prefix(_TRAINING_MUSICGEN_TOKEN_CACHE_URI) or 'prepared/cara-strong-v0.4/musicgen_encodec_cache').rstrip('/')}/tokens/train/"
        blobs = _azureml_datastore_blob_list(token_prefix)
        train_chunks = sum(
            1
            for blob in blobs
            if str(blob.get("name") or "").lower().endswith(".pt") and int(blob.get("size") or 0) > 0
        )
        effective = min(train_chunks, max_train_files) if max_train_files > 0 else train_chunks
        return {
            "method": "read_only_azure_token_blob_count",
            "datastore_prefix": token_prefix,
            "train_chunks": train_chunks,
            "effective_train_chunks": effective,
            "max_train_files": max_train_files,
        }

    # Context Diffusion reuses the Stable Audio prepared chunks; its extra
    # context artifacts are conditioning metadata, not the primary step unit.
    prefix = f"{_training_prepared_audio_prefix('stable_audio_open_small')}train/"
    blobs = _azureml_datastore_blob_list(prefix)
    train_chunks = sum(
        1
        for blob in blobs
        if str(blob.get("name") or "").lower().endswith(".wav") and int(blob.get("size") or 0) > 0
    )
    effective = train_chunks
    if max_train_files > 0:
        effective = min(train_chunks, max_train_files)
    return {
        "method": "read_only_azure_blob_count",
        "datastore_prefix": prefix,
        "train_chunks": train_chunks,
        "effective_train_chunks": effective,
        "max_train_files": max_train_files,
    }


def _training_datastore_metrics_csv_payload(output_path: Any) -> dict[str, Any] | None:
    output_prefix = _azureml_uri_to_blob_prefix(output_path)
    if not output_prefix:
        return None
    candidate_blobs = [
        f"{output_prefix.rstrip('/')}/logs/stable_audio_smoke/version_0/metrics.csv",
        f"{output_prefix.rstrip('/')}/logs/lightning_logs/version_0/metrics.csv",
        f"{output_prefix.rstrip('/')}/metrics.csv",
    ]
    errors: list[str] = []
    for blob_name in candidate_blobs:
        try:
            text = _azureml_datastore_blob_text(blob_name)
        except Exception as exc:
            errors.append(str(getattr(exc, "detail", None) or exc))
            continue
        latest: dict[str, float] = {}
        observed_step = 0
        row_count = 0
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            row_count += 1
            step_value = row.get("step") or row.get("global_step") or row.get("trainer/global_step")
            try:
                if step_value not in {None, ""}:
                    observed_step = max(observed_step, int(float(step_value)))
            except (TypeError, ValueError):
                pass
            for key, value in row.items():
                if key in {None, "", "step"} or value in {None, ""}:
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed):
                    latest[str(key)] = parsed
        return {
            "run_id": str(output_path),
            "latest": latest,
            "histories": {},
            "params": {},
            "tags": {},
            "observed_step": observed_step if observed_step > 0 else None,
            "source_blob": blob_name,
            "row_count": row_count,
            "fallback_errors": errors,
        }
    raise HTTPException(status_code=404, detail=f"No datastore metrics.csv found under {output_path}: {'; '.join(errors[-2:])}")


def _training_progress_job_summary(job_name: str, event: dict[str, Any]) -> dict[str, Any]:
    settings = _azureml_settings()
    command = [
        "az",
        "ml",
        "job",
        "show",
        "--subscription",
        settings["subscription_id"],
        "--resource-group",
        settings["resource_group"],
        "--workspace-name",
        settings["workspace_name"],
        "--name",
        job_name,
        "--only-show-errors",
        "--query",
        "{name:name,display_name:display_name,status:status,compute:compute,environment:environment,created_at:creation_context.created_at,start_time:start_time,end_time:end_time,studio_url:services.Studio.endpoint,tags:tags,properties:properties}",
        "--output",
        "json",
    ]
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=35, check=False)
        if completed.returncode == 0 and completed.stdout.strip():
            payload = json.loads(completed.stdout)
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {
        "name": job_name,
        "status": event.get("status") or "unknown",
        "studio_url": event.get("studio_url"),
        "compute": event.get("compute"),
        "environment": event.get("environment"),
        "created_at": event.get("created_at"),
        "tags": {},
        "properties": {},
    }


def _training_latest_training_progress(
    *,
    model: str = "latest",
    registry_limit: int = 200,
    live_status: bool = True,
) -> dict[str, Any] | dict[str, Any] | None:
    model_key = _training_progress_model_key(model)
    events = [
        event
        for event in reversed(_training_job_registry_events(registry_limit))
        if _training_progress_event_is_candidate(event)
        and (model_key in {"latest", "all"} or _training_progress_event_model_key(event) == model_key)
    ]
    if not events:
        return None
    if model_key == "all":
        progress_by_model: dict[str, Any] = {}
        seen_models: set[str] = set()
        for event in events:
            event_model = _training_progress_event_model_key(event)
            if event_model in seen_models:
                continue
            progress = _training_latest_training_progress(model=event_model, registry_limit=registry_limit, live_status=False)
            if progress is not None:
                progress_by_model[event_model] = progress
                seen_models.add(event_model)
        return progress_by_model

    selected_event: dict[str, Any] | None = None
    selected_summary: dict[str, Any] | None = None
    selected_status = ""
    for event in events:
        job_name = str(event.get("job_name") or "").strip()
        if not job_name:
            continue
        summary = (
            _training_progress_job_summary(job_name, event)
            if live_status
            else {
                "name": job_name,
                "status": event.get("status") or "unknown",
                "studio_url": event.get("studio_url"),
                "compute": event.get("compute"),
                "environment": event.get("environment"),
                "created_at": event.get("created_at"),
                "tags": {},
                "properties": {},
            }
        )
        status = str(summary.get("status") or "").lower()
        if selected_event is None:
            selected_event = event
            selected_summary = summary
            selected_status = status
        if status in _AZUREML_ACTIVE_STATUSES:
            selected_event = event
            selected_summary = summary
            selected_status = status
            break
    if selected_event is None or selected_summary is None:
        return None

    job_name = str(selected_event.get("job_name") or selected_summary.get("name") or "")
    max_steps = int(float(selected_event.get("max_steps") or 0))
    batch_size = int(float(selected_event.get("batch_size") or 0))
    event_model_key = _training_progress_event_model_key(selected_event)
    max_train_files = int(float(selected_event.get("max_train_files") or 0))
    checked_at = datetime.now(timezone.utc)
    start_time = _training_parse_datetime(selected_summary.get("start_time") or selected_event.get("created_at"))
    end_time = _training_parse_datetime(selected_summary.get("end_time"))
    elapsed_seconds = None
    if start_time:
        elapsed_seconds = max(0.0, ((end_time or checked_at) - start_time).total_seconds())

    metrics_payload: dict[str, Any] | None = None
    metrics_error = None
    metrics_source = None
    observed_step = None
    if event_model_key in {"stable_audio_open_small", "stable_audio_open_small_context_diffusion"}:
        try:
            datastore_metrics = _training_datastore_metrics_csv_payload(selected_event.get("output_path"))
            metrics_payload = datastore_metrics
            observed_step = datastore_metrics.get("observed_step") or _training_metric_observed_step(datastore_metrics)
            metrics_source = "azure_datastore_lightning_metrics_csv"
        except Exception as exc:
            metrics_error = str(getattr(exc, "detail", None) or exc)
    needs_metric_fallback = (
        observed_step is None
        or not isinstance((metrics_payload or {}).get("latest"), dict)
        or not (metrics_payload or {}).get("latest")
    )
    if needs_metric_fallback and event_model_key not in {"musicgen", "ace_step"}:
        try:
            mlflow_metrics = _azureml_job_metrics_payload(job_name, history_limit=2000)
            metrics_payload = mlflow_metrics
            observed_step = _training_metric_observed_step(mlflow_metrics)
            metrics_source = "azure_mlflow_step_metrics"
            metrics_error = None
        except Exception as exc:
            if metrics_error:
                metrics_error = f"{metrics_error}; MLflow metrics fallback failed: {str(getattr(exc, 'detail', None) or exc)}"
            else:
                metrics_error = str(getattr(exc, "detail", None) or exc)
    elif needs_metric_fallback and not metrics_error:
        metrics_error = "No fast branch metrics artifact is available for this trainer yet."

    chunk_counts: dict[str, Any] | None = None
    chunk_count_error = "Prepared train item denominator is not counted during routine run-progress refreshes; step percent uses observed_step / max_steps."
    if max_steps <= 0 and selected_event.get("full_training_run") and batch_size > 0 and chunk_counts:
        max_steps = max(1, math.ceil(int(chunk_counts.get("effective_train_chunks") or 0) / batch_size))
    if observed_step is None and selected_status == "completed" and max_steps > 0:
        observed_step = max_steps
    step_percent = min(100.0, observed_step / max(1, max_steps) * 100.0) if max_steps > 0 and observed_step is not None else None
    estimated_remaining_seconds = None
    if elapsed_seconds is not None and observed_step and max_steps > observed_step:
        estimated_remaining_seconds = elapsed_seconds * ((max_steps - observed_step) / observed_step)

    chunks_seen_estimate = (observed_step or 0) * batch_size if batch_size > 0 else None
    effective_train_chunks = int((chunk_counts or {}).get("effective_train_chunks") or 0)
    completed_epochs_estimate = None
    epoch_percent = None
    if chunks_seen_estimate is not None and effective_train_chunks > 0:
        completed_epochs_estimate = chunks_seen_estimate // effective_train_chunks
        epoch_percent = (chunks_seen_estimate % effective_train_chunks) / effective_train_chunks * 100.0

    latest_loss = None
    latest_metrics = (metrics_payload or {}).get("latest") if isinstance(metrics_payload, dict) else None
    if isinstance(latest_metrics, dict):
        for key in ("train/loss", "loss", "val/loss"):
            if key in latest_metrics:
                try:
                    latest_loss = float(latest_metrics[key])
                    break
                except (TypeError, ValueError):
                    pass

    return {
        "checked_at": checked_at.isoformat(),
        "model_key": event_model_key,
        "model_label": _TRAINING_RUN_PROGRESS_MODEL_LABELS.get(event_model_key, event_model_key),
        "job_name": job_name,
        "run_name": selected_event.get("run_name"),
        "studio_url": selected_summary.get("studio_url") or selected_event.get("studio_url"),
        "status": selected_summary.get("status"),
        "variant": selected_event.get("variant"),
        "training_scope": _training_progress_event_scope(selected_event),
        "action": selected_event.get("action"),
        "max_steps": max_steps or None,
        "observed_step": observed_step,
        "step_percent": round(step_percent, 2) if step_percent is not None else None,
        "batch_size": batch_size or None,
        "chunks_seen_estimate": chunks_seen_estimate,
        "train_chunks": (chunk_counts or {}).get("train_chunks"),
        "effective_train_chunks": (chunk_counts or {}).get("effective_train_chunks"),
        "completed_epochs_estimate": completed_epochs_estimate,
        "epoch_percent": round(epoch_percent, 2) if epoch_percent is not None else None,
        "latest_loss": latest_loss,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 3) if estimated_remaining_seconds is not None else None,
        "metrics_available": metrics_payload is not None,
        "metrics_source": metrics_source,
        "metrics_artifact": (metrics_payload or {}).get("source_blob") if isinstance(metrics_payload, dict) else None,
        "metrics_row_count": (metrics_payload or {}).get("row_count") if isinstance(metrics_payload, dict) else None,
        "metrics_error": metrics_error,
        "chunk_count": chunk_counts,
        "chunk_count_error": chunk_count_error,
        "note": (
            "Progress is read from Azure MLflow when available, otherwise from the run's datastore metrics.csv artifact. "
            "Chunks-seen and epoch progress are estimates from observed_step * batch_size against the prepared train/token count; shuffled training can revisit chunks across epochs."
        ),
    }


def _training_latest_stable_audio_training_progress(*, registry_limit: int = 100) -> dict[str, Any] | None:
    progress = _training_latest_training_progress(model="stable_audio_open_small", registry_limit=registry_limit)
    return progress if isinstance(progress, dict) and "job_name" in progress else None


def _training_preprocess_model_key(models: str) -> str:
    normalized = ",".join(part.strip() for part in str(models or "all").split(",") if part.strip()).lower()
    aliases = {
        "": "all",
        "all": "all",
        "both": "all",
        "stable": "stable_audio_open_small",
        "stable_audio": "stable_audio_open_small",
        "stable_audio_open_small": "stable_audio_open_small",
        "diffusion": "stable_audio_open_small",
        "musicgen": "musicgen",
        "autoregressive": "musicgen",
    }
    key = aliases.get(normalized)
    if key is None:
        raise HTTPException(status_code=400, detail=f"Unsupported preprocessing model selection: {models}")
    return key


def _training_select_preprocess_compute(strategy: str) -> dict[str, Any]:
    normalized = str(strategy or "prefer_h100_else_cpu").lower()
    if normalized in {"cpu", "cpu_only", "force_cpu"}:
        return {"compute": _TRAINING_CPU_COMPUTE, "strategy": normalized, "reason": "CPU compute was explicitly requested.", "blocking_h100_jobs": []}
    if normalized in {"h100", "gpu", "h100_only", "force_h100"}:
        return {"compute": _TRAINING_H100_COMPUTE, "strategy": normalized, "reason": "H100 compute was explicitly requested.", "blocking_h100_jobs": []}
    if normalized not in {"auto", "prefer_h100_else_cpu", "prefer_gpu_else_cpu"}:
        raise HTTPException(status_code=400, detail=f"Unsupported preprocessing compute strategy: {strategy}")
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        active_targets = sorted({str(job.get("h100_compute_target") or job.get("compute") or "unknown") for job in active_h100_jobs})
        return {
            "compute": _TRAINING_CPU_COMPUTE,
            "strategy": normalized,
            "reason": f"H100-backed compute is busy on {', '.join(active_targets)}; preprocessing was routed to CPU.",
            "blocking_h100_jobs": active_h100_jobs[:10],
        }
    return {
        "compute": _TRAINING_H100_COMPUTE,
        "strategy": normalized,
        "reason": f"No active Azure ML jobs were found on H100-backed compute targets {', '.join(_TRAINING_H100_COMPUTES)}; preprocessing can use {_TRAINING_H100_COMPUTE}.",
        "blocking_h100_jobs": [],
    }


def _training_readiness_payload(variant: str = "all") -> dict[str, Any]:
    model_scope = _training_preprocess_model_key(variant)
    include_stable_audio = model_scope in {"all", "stable_audio_open_small"}
    include_musicgen = model_scope in {"all", "musicgen"}
    phases = {
        "data_access": _training_test_prep_phase("strong_parrot_l70v7x8w6x"),
        "gpu_sanity": _training_test_prep_phase("upbeat_shirt_3xzkjyzdvd"),
        "musicgen_environment": _training_test_prep_phase("wheat_dog_0wh3fqkljk"),
        "stable_audio_environment": _training_test_prep_phase("tough_kite_mmmxk4yy9p"),
    }
    lock_state = _training_lock_state()
    azure_upload_state = _training_azure_upload_state()
    stable_audio_preflight = (
        _training_latest_stable_audio_preflight()
        if include_stable_audio
        else {"passed": False, "active": False, "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT, "latest_job": None, "reason": "Stable Audio readiness skipped for MusicGen-scoped gate refresh."}
    )
    stable_audio_smoke_sequence = (
        _training_stable_audio_smoke_sequence()
        if include_stable_audio
        else {"variants": {}, "next_variant": "no_cara_baseline", "next_stage": 5, "next_label": "Stable Audio smoke skipped", "reason": "Stable Audio smoke readiness skipped for MusicGen-scoped gate refresh."}
    )
    stable_audio_full_training = (
        _training_latest_stable_audio_full_training()
        if include_stable_audio
        else {"stage": 9, "label": "Full Stable Audio CARA-Strong fine-tune", "passed": False, "active": False, "reason": "Stable Audio full-run readiness skipped for MusicGen-scoped gate refresh.", "latest_job": None}
    )
    stable_audio_training_progress = None
    preprocess_jobs = _training_latest_preprocess_jobs()
    musicgen_token_cache = (
        _training_latest_musicgen_token_cache()
        if include_musicgen
        else {"stage": 6, "label": "MusicGen EnCodec token cache", "passed": False, "active": False, "reason": "MusicGen token-cache readiness skipped for Stable Audio-scoped gate refresh.", "latest_job": None}
    )
    musicgen_preflight = (
        _training_latest_musicgen_preflight()
        if include_musicgen
        else {"stage": 7, "label": "MusicGen trainer preflight", "passed": False, "active": False, "required_environment": _TRAINING_MUSICGEN_ENVIRONMENT, "reason": "MusicGen preflight readiness skipped for Stable Audio-scoped gate refresh.", "latest_job": None}
    )
    musicgen_smoke_sequence = (
        _training_musicgen_smoke_sequence()
        if include_musicgen
        else {"variants": {}, "next_variant": "no_cara_baseline", "next_stage": 8, "next_label": "MusicGen smoke skipped", "reason": "MusicGen smoke readiness skipped for Stable Audio-scoped gate refresh."}
    )
    smoke_phase_keys = ["data_access", "gpu_sanity"]
    if include_stable_audio:
        smoke_phase_keys.append("stable_audio_environment")
    if include_musicgen:
        smoke_phase_keys.append("musicgen_environment")
    smoke_ready = all(str(phases[key].get("status")).lower() == "passed" for key in smoke_phase_keys)
    lock_ready = bool(lock_state.get("locked"))
    active_stable_audio_trainer_jobs = _training_active_stable_audio_trainer_jobs() if include_stable_audio else []
    active_musicgen_trainer_jobs = _training_active_musicgen_trainer_jobs() if include_musicgen else []
    active_stable_audio_smoke_jobs = [job for job in active_stable_audio_trainer_jobs if str(job.get("training_scope") or "smoke") == "smoke"]
    active_stable_audio_smoke_job = active_stable_audio_trainer_jobs[0] if active_stable_audio_trainer_jobs else None
    active_musicgen_trainer_job = active_musicgen_trainer_jobs[0] if active_musicgen_trainer_jobs else None
    stable_audio_preflight_ready = bool(stable_audio_preflight.get("passed"))
    musicgen_preflight_ready = bool(musicgen_preflight.get("passed"))
    if include_musicgen and not include_stable_audio:
        base_launch_ready = smoke_ready and lock_ready and bool(azure_upload_state.get("confirmed")) and musicgen_preflight_ready
        launch_ready = base_launch_ready and active_musicgen_trainer_job is None
        training_launch_reason = (
            f"MusicGen trainer job {active_musicgen_trainer_job.get('name')} is already {active_musicgen_trainer_job.get('status')}; monitor it in Operations / Azure Runs or hard-stop it there before launching another."
            if active_musicgen_trainer_job
            else musicgen_preflight.get("reason")
            if not musicgen_preflight_ready
            else "MusicGen smoke trainer command job is available; complete the token-cache gate before launch."
            if launch_ready
            else "Pass MusicGen smoke tests, lock the CARA-Strong manifest, confirm the Azure upload, cache tokens, and pass the MusicGen preflight before smoke training can launch."
        )
    else:
        base_launch_ready = smoke_ready and lock_ready and bool(azure_upload_state.get("confirmed")) and stable_audio_preflight_ready
        launch_ready = base_launch_ready and active_stable_audio_smoke_job is None
        training_launch_reason = (
            f"Stable Audio trainer job {active_stable_audio_smoke_job.get('name')} is already {active_stable_audio_smoke_job.get('status')}; monitor it in Operations / Azure Runs or hard-stop it there before launching another."
            if active_stable_audio_smoke_job
            else stable_audio_preflight.get("reason")
            if not stable_audio_preflight_ready
            else "Stable Audio smoke trainer command job is available; complete the model-specific prepared dataset gate before launch."
            if launch_ready
            else "Pass smoke tests, lock the CARA-Strong manifest, confirm the Azure upload, and pass the trainer preflight before smoke training can be prepared."
        )
    gates = [
        {"id": "azure_data_access", "label": "Azure datastore access", "passed": str(phases["data_access"].get("status")).lower() == "passed"},
        {"id": "h100_cuda", "label": "H100 CUDA visibility", "passed": str(phases["gpu_sanity"].get("status")).lower() == "passed"},
        *([{"id": "stable_audio_tools", "label": "Stable Audio Tools environment", "passed": str(phases["stable_audio_environment"].get("status")).lower() == "passed"}] if include_stable_audio else []),
        *([{"id": "audiocraft", "label": "AudioCraft / MusicGen environment", "passed": str(phases["musicgen_environment"].get("status")).lower() == "passed"}] if include_musicgen else []),
        {"id": "manifest_lock", "label": "CARA-Strong manifest and registry lock", "passed": lock_ready},
        {"id": "azure_upload_complete", "label": "Full Azure dataset upload confirmed", "passed": bool(azure_upload_state.get("confirmed"))},
        *([{"id": "stable_audio_trainer_preflight", "label": "Stable Audio trainer preflight", "passed": stable_audio_preflight_ready}] if include_stable_audio else []),
        *([{"id": "musicgen_trainer_preflight", "label": "MusicGen trainer preflight", "passed": musicgen_preflight_ready}] if include_musicgen else []),
    ]
    return {
        "status": "ready_for_smoke_training" if base_launch_ready else "blocked",
        "smoke_tests_ready": smoke_ready,
        "training_launch_enabled": launch_ready,
        "training_launch_reason": training_launch_reason,
        "phases": phases,
        "gates": gates,
        "lock": lock_state,
        "azure_upload": azure_upload_state,
        "stable_audio_preflight": stable_audio_preflight,
        "stable_audio_smoke_sequence": stable_audio_smoke_sequence,
        "stable_audio_full_training": stable_audio_full_training,
        "stable_audio_training_progress": stable_audio_training_progress,
        "preprocess_jobs": preprocess_jobs,
        "musicgen_token_cache": musicgen_token_cache,
        "musicgen_preflight": musicgen_preflight,
        "musicgen_smoke_sequence": musicgen_smoke_sequence,
        "active_training_jobs": active_stable_audio_trainer_jobs + active_musicgen_trainer_jobs,
        "active_stable_audio_smoke_job": active_stable_audio_smoke_job,
        "active_musicgen_trainer_job": active_musicgen_trainer_job,
        "data_locations": {
            "local_audio_root": str(ROOT / "data" / "freesound"),
            "source_manifest": str(ROOT / "data" / "cara_pool_manifest_v2.jsonl"),
            "source_pool_registry": str(ROOT / "registry" / "pool_allocator_v2" / "pools.json"),
            "azure_source_root": _TRAINING_SOURCE_URI,
            "azure_datastore_audio": f"{_TRAINING_SOURCE_URI}data/freesound/",
            "azure_datastore_manifest": f"{_TRAINING_SOURCE_URI}data/cara_pool_manifest_v2.jsonl",
            "azure_prepared_root": _TRAINING_PREP_OUTPUT_URI,
            "azure_stable_audio_manifest": f"{_TRAINING_PREP_OUTPUT_URI}stable_audio_open_small/manifest.jsonl",
            "azure_musicgen_manifest": f"{_TRAINING_PREP_OUTPUT_URI}musicgen/manifest.jsonl",
            "azure_musicgen_encodec_cache": _TRAINING_MUSICGEN_TOKEN_CACHE_URI,
            "azure_musicgen_encodec_manifest": f"{_TRAINING_MUSICGEN_TOKEN_CACHE_URI}manifest.encodec.jsonl",
            "azure_musicgen_preflight_output_root": _TRAINING_MUSICGEN_PREFLIGHT_OUTPUT_URI,
            "azure_musicgen_smoke_output_root": _TRAINING_MUSICGEN_SMOKE_OUTPUT_URI,
            "azure_musicgen_full_output_root": _TRAINING_MUSICGEN_FULL_OUTPUT_URI,
            "azure_split_manifest": f"{_TRAINING_PREP_OUTPUT_URI}split_manifest.json",
            "azure_stable_audio_preflight_output_root": _TRAINING_STABLE_AUDIO_PREFLIGHT_OUTPUT_URI,
            "azure_stable_audio_smoke_output_root": _TRAINING_STABLE_AUDIO_SMOKE_OUTPUT_URI,
            "azure_stable_audio_full_output_root": _TRAINING_STABLE_AUDIO_FULL_OUTPUT_URI,
        },
        "cloud_job_policy": {
            "durable_submission": True,
            "browser_close_cancels_job": False,
            "stop_behavior": "Azure ML jobs keep running after the dashboard/browser closes. A job is stopped only by the explicit hard-stop action in Operations / Azure Runs.",
            "checkpoint_resume": "Trainer jobs should write checkpoints to Azure outputs and resume from the latest checkpoint after a failed attempt; preprocessing jobs are deterministic and can be resubmitted safely.",
            "preprocess_compute_strategy": "Prefer gpu-smoke-h100 only when no active Azure ML jobs exist on either H100-backed compute target; otherwise submit preprocessing to cpu-prep-cluster.",
            "musicgen_cpu_fallback_stages": "MusicGen dataset preparation and EnCodec token-cache jobs may use cpu-prep-cluster while either H100-backed compute target is busy. MusicGen preflight, smoke, and full trainer jobs are GPU-only.",
            "musicgen_token_cache": "MusicGen EnCodec caches bind each prepared audio chunk to its exact discrete audio-token target and CARA pool id before autoregressive training.",
            "run_progress_endpoint": "Use /api/training/run-progress for slower live MLflow training progress. Gate refresh intentionally avoids that expensive lookup.",
        },
        "submitted_training_jobs": _training_job_registry_events(20),
        "audio_window_policy": {
            "stable_audio_open_small": {
                "sample_rate_hz": 44100,
                "channels": "stereo",
                "max_window_seconds": 11.88,
                "pre_chunk_required": True,
                "note": "CARA-Strong training uses prepared 44.1 kHz stereo chunk files so pool-duration accounting, split lineage, and receipts are reproducible.",
            },
            "musicgen": {
                "sample_rate_hz": 32000,
                "channels": "mono for stock checkpoints; stereo only for stereo checkpoints",
                "max_window_seconds": 30.0,
                "pre_chunk_required": True,
                "note": "CARA-Strong training uses prepared 32 kHz mono chunk files for stock MusicGen checkpoints; stereo checkpoints need a separate stereo output profile.",
            },
        },
    }


def _training_context_diffusion_readiness_payload() -> dict[str, Any]:
    lock_state = _training_lock_state()
    azure_upload_state = _training_azure_upload_state()
    full_training = _training_latest_stable_audio_full_training_cached()
    context_ladder = _training_context_diffusion_ladder()
    full_training_passed = bool(full_training.get("passed"))
    inherited_ready = bool(lock_state.get("locked")) and bool(azure_upload_state.get("confirmed")) and full_training_passed
    stable_audio_preflight = {
        "stage": 4,
        "label": "Stable Audio trainer preflight",
        "passed": inherited_ready,
        "active": False,
        "latest_job": None,
        "reason": "Inherited from the completed Stable Audio CARA-Strong ladder; this Context Diffusion gate uses cached evidence rather than live Azure polling.",
    }
    stable_audio_smoke_sequence = {
        "variants": {
            "no_cara_baseline": {"stage": 5, "label": "Baseline smoke", "passed": inherited_ready, "active": False, "reason": "Inherited from the completed Stable Audio branch."},
            "cara_lite": {"stage": 6, "label": "CARA-lite smoke", "passed": inherited_ready, "active": False, "reason": "Inherited from the completed Stable Audio branch."},
            "cara_head": {"stage": 7, "label": "Attribution-head smoke", "passed": inherited_ready, "active": False, "reason": "Inherited from the completed Stable Audio branch."},
            "cara_strong": {"stage": 8, "label": "CARA-Strong smoke", "passed": inherited_ready, "active": False, "reason": "Inherited from the completed Stable Audio branch."},
        },
        "next_stage": 10 if inherited_ready else 5,
        "next_label": "Context packs" if inherited_ready else "Inherited Stable Audio evidence",
        "reason": "Context Diffusion reuses the completed Stable Audio smoke ladder as baseline evidence.",
    }
    preprocess_jobs = {
        "stable_audio_open_small": {
            "stage": 3,
            "label": "Prepare Stable Audio dataset",
            "passed": inherited_ready,
            "active": False,
            "latest_job": None,
            "reason": "Inherited from the prepared Stable Audio manifest used by the completed full run.",
        }
    }
    return {
        "status": "ready_for_context_packs" if full_training_passed else "blocked",
        "smoke_tests_ready": inherited_ready,
        "training_launch_enabled": False,
        "training_launch_reason": (
            "Context pack preparation can be submitted."
            if inherited_ready
            else "Lock the manifest, confirm Azure upload, and complete the Stable Audio full run before preparing Context Diffusion packs."
        ),
        "lock": lock_state,
        "azure_upload": azure_upload_state,
        "preprocess_jobs": preprocess_jobs,
        "stable_audio_preflight": stable_audio_preflight,
        "stable_audio_smoke_sequence": stable_audio_smoke_sequence,
        "stable_audio_full_training": full_training,
        "context_diffusion_ladder": context_ladder,
        "context_diffusion_launch": {
            "context_packs_enabled": inherited_ready and not bool(context_ladder["context_packs"].get("active")) and not bool(context_ladder["context_packs"].get("passed")),
            "context_cache_enabled": bool(context_ladder["context_packs"].get("passed")) and not bool(context_ladder["context_cache"].get("active")),
            "context_preflight_enabled": bool(context_ladder["context_cache"].get("passed")) and not bool(context_ladder["context_preflight"].get("active")),
            "context_smoke_enabled": bool(context_ladder["context_preflight"].get("passed")) and not bool(context_ladder["context_smoke"].get("active")),
            "context_full_enabled": (
                bool(context_ladder["context_smoke"].get("passed"))
                and not bool(context_ladder["context_full"].get("active"))
                and not bool(context_ladder["context_full"].get("passed"))
            ),
            "context_smoke_reason": (
                str(context_ladder["context_smoke"].get("reason") or "Context smoke can be launched.")
                if context_ladder["context_preflight"].get("passed")
                else "Run and pass context pack, cache, and preflight stages before implementing context smoke."
            ),
            "context_full_reason": (
                str(context_ladder["context_full"].get("reason") or "Full Context Diffusion fine-tune can be launched.")
                if context_ladder["context_smoke"].get("passed")
                else "Run and pass the context smoke before launching the full context fine-tune."
            ),
        },
        "data_locations": {
            "local_audio_root": str(ROOT / "data" / "freesound"),
            "source_manifest": str(ROOT / "data" / "cara_pool_manifest_v2.jsonl"),
            "source_pool_registry": str(ROOT / "registry" / "pool_allocator_v2" / "pools.json"),
            "azure_source_root": _TRAINING_SOURCE_URI,
            "azure_datastore_audio": f"{_TRAINING_SOURCE_URI}data/freesound/",
            "azure_datastore_manifest": f"{_TRAINING_SOURCE_URI}data/cara_pool_manifest_v2.jsonl",
            "azure_prepared_root": _TRAINING_PREP_OUTPUT_URI,
            "azure_stable_audio_manifest": f"{_TRAINING_PREP_OUTPUT_URI}stable_audio_open_small/manifest.jsonl",
            "azure_stable_audio_full_output_root": _TRAINING_STABLE_AUDIO_FULL_OUTPUT_URI,
            "azure_stable_audio_trained_model": _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI,
            "azure_stable_audio_context_root": _TRAINING_CONTEXT_ROOT_URI,
            "azure_stable_audio_context_packs": _TRAINING_CONTEXT_PACK_OUTPUT_URI,
            "azure_stable_audio_context_cache": _TRAINING_CONTEXT_CACHE_OUTPUT_URI,
            "azure_stable_audio_context_preflight": _TRAINING_CONTEXT_PREFLIGHT_OUTPUT_URI,
            "azure_stable_audio_context_smoke": _TRAINING_CONTEXT_SMOKE_OUTPUT_URI,
            "azure_stable_audio_context_full": _TRAINING_CONTEXT_FULL_OUTPUT_URI,
        },
        "cloud_job_policy": {
            "durable_submission": True,
            "browser_close_cancels_job": False,
            "cost_guardrail": "Use existing Azure ML workspace resources only; no Marketplace endpoints or new paid services.",
        },
        "submitted_training_jobs": _training_job_registry_events(20),
    }


def _training_hybrid_readiness_payload() -> dict[str, Any]:
    lock_state = _training_lock_state()
    azure_upload_state = _training_azure_upload_state()
    preflight = _training_latest_ace_preflight()
    ladder = _training_ace_ladder(preflight)
    active_h100_jobs = _training_recent_h100_jobs_from_registry()
    launch_enabled = (
        bool(lock_state.get("locked"))
        and bool(azure_upload_state.get("confirmed"))
        and not bool(preflight.get("active"))
        and not active_h100_jobs
    )
    if preflight.get("active"):
        launch_reason = str(preflight.get("reason") or "ACE-Step preflight is already active.")
    elif active_h100_jobs:
        targets = sorted({str(job.get("h100_compute_target") or job.get("compute") or "unknown") for job in active_h100_jobs})
        launch_reason = f"H100-backed compute is busy on {', '.join(targets)}; ACE-Step preflight will not fall back to CPU."
    elif not lock_state.get("locked"):
        launch_reason = "Lock the CARA-Strong manifest before ACE-Step preflight so the label registry is auditable."
    elif not azure_upload_state.get("confirmed"):
        launch_reason = "Confirm the Azure dataset upload before ACE-Step preflight so the source manifest/audio paths are checked."
    else:
        launch_reason = "ACE-Step environment preflight can be launched. Later tensor/planner/smoke stages remain locked until this passes."
    return {
        "status": "ready_for_ace_preflight" if launch_enabled else "blocked",
        "training_launch_enabled": launch_enabled,
        "training_launch_reason": launch_reason,
        "lock": lock_state,
        "azure_upload": azure_upload_state,
        "ace_preflight": preflight,
        "ace_ladder": ladder,
        "active_h100_jobs": active_h100_jobs,
        "data_locations": {
            "azure_source_root": _TRAINING_SOURCE_URI,
            "azure_datastore_audio": f"{_TRAINING_SOURCE_URI}data/freesound/",
            "azure_datastore_manifest": f"{_TRAINING_SOURCE_URI}data/cara_pool_manifest_v2.jsonl",
            "azure_ace_preflight_output_root": _TRAINING_ACE_PREFLIGHT_OUTPUT_URI,
            "future_ace_prepared_root": f"{_TRAINING_PREP_OUTPUT_URI}ace_step/",
        },
        "cloud_job_policy": {
            "durable_submission": True,
            "browser_close_cancels_job": False,
            "marketplace_resources_allowed": False,
            "preflight_compute_policy": "ACE-Step preflight is GPU-only because CUDA, model import, and DiT tap viability are the point of the gate. Readiness uses recent dashboard-submitted H100 jobs for responsiveness; launch submission performs a live two-target H100 check before queueing.",
            "trainer_compute_policy": "Future ACE smoke/train jobs must use existing Azure ML command jobs, approved computes, datastores, and environments only.",
        },
        "evidence_contract": {
            "model_family": "ace_step",
            "architecture": "LM planner plus DiT synthesizer",
            "sample_rate_hz": 48000,
            "latent_rate_hz": 25,
            "label_fields_required": [
                "cara_source_pool_id",
                "cara_pool_id",
                "cara_pool_index",
                "cara_pool_family",
                "cara_pool_family_index",
            ],
            "metrics": [
                "planner_survival_exact",
                "planner_survival_repairable",
                "planner_cara_lost",
                "planner_cara_hallucinated",
                "dit_family_top1",
                "dit_pool_top1",
                "dit_pool_top5",
                "registry_valid_rate",
                "shuffled_label_baseline",
                "source_disjoint_eval",
            ],
        },
        "submitted_training_jobs": [
            event
            for event in _training_job_registry_events(50)
            if event.get("model_family") == "ace_step"
        ],
    }


def _submit_training_preprocess_job(
    *,
    dry_run: bool = False,
    models: str = "all",
    compute_strategy: str = "prefer_h100_else_cpu",
) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        model_key = _training_preprocess_model_key(models)
        job_file = _TRAINING_PREPROCESS_JOB_FILES[model_key]
        if not job_file.exists():
            raise FileNotFoundError(f"Training preprocess job file not found: {job_file}")
        compute_selection = _training_select_preprocess_compute(compute_strategy)
        job = load_job(source=job_file)
        _azureml_set_job_input(job, "dashboard_triggered", "true")
        _azureml_set_job_input(job, "dry_run", "true" if dry_run else "false")
        job.compute = compute_selection["compute"]
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "model_dataset_preprocess",
            "cara_model_family": model_key,
            "cara_compute_strategy": compute_selection["strategy"],
            "cara_compute_selected": compute_selection["compute"],
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "training_preprocess_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": compute_selection["compute"],
            "compute_strategy": compute_selection["strategy"],
            "compute_reason": compute_selection["reason"],
            "blocking_h100_jobs": [
                {"name": item.get("name"), "status": item.get("status"), "display_name": item.get("display_name")}
                for item in compute_selection["blocking_h100_jobs"]
            ],
            "environment": summary.get("environment"),
            "model_family": model_key,
            "job_file": str(job_file.relative_to(ROOT)),
            "output_path": _TRAINING_PREP_OUTPUT_URI,
            "dry_run": dry_run,
        }
        _azureml_test_prep_audit(
            event
        )
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": _TRAINING_PREP_OUTPUT_URI,
            "dry_run": dry_run,
            "model_family": model_key,
            "compute_selected": compute_selection,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_musicgen_token_cache_job(
    *,
    dry_run: bool = False,
    compute_strategy: str = "prefer_h100_else_cpu",
) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not _TRAINING_MUSICGEN_TOKEN_CACHE_JOB_FILE.exists():
            raise FileNotFoundError(f"MusicGen token-cache job file not found: {_TRAINING_MUSICGEN_TOKEN_CACHE_JOB_FILE}")
        compute_selection = _training_select_preprocess_compute(compute_strategy)
        job = load_job(source=_TRAINING_MUSICGEN_TOKEN_CACHE_JOB_FILE)
        _azureml_set_job_input(job, "dashboard_triggered", "true")
        _azureml_set_job_input(job, "dry_run", "true" if dry_run else "false")
        job.compute = compute_selection["compute"]
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "musicgen_encodec_token_cache",
            "cara_model_family": "musicgen",
            "cara_compute_strategy": compute_selection["strategy"],
            "cara_compute_selected": compute_selection["compute"],
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "musicgen_encodec_cache_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": compute_selection["compute"],
            "compute_strategy": compute_selection["strategy"],
            "compute_reason": compute_selection["reason"],
            "blocking_h100_jobs": [
                {"name": item.get("name"), "status": item.get("status"), "display_name": item.get("display_name")}
                for item in compute_selection["blocking_h100_jobs"]
            ],
            "environment": summary.get("environment"),
            "model_family": "musicgen",
            "job_file": str(_TRAINING_MUSICGEN_TOKEN_CACHE_JOB_FILE.relative_to(ROOT)),
            "output_path": _TRAINING_MUSICGEN_TOKEN_CACHE_URI,
            "dry_run": dry_run,
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": _TRAINING_MUSICGEN_TOKEN_CACHE_URI,
            "dry_run": dry_run,
            "model_family": "musicgen",
            "compute_selected": compute_selection,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_musicgen_trainer_preflight_job(request: TrainingMusicGenPreflightRunRequest) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not _TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE.exists():
            raise FileNotFoundError(f"MusicGen trainer preflight job file not found: {_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE}")
        active_h100_jobs = _azureml_active_jobs_on_h100_computes()
        if active_h100_jobs:
            names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
            raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); MusicGen trainer preflight is GPU-only and will not fall back to CPU.")
        output_path = f"{_TRAINING_MUSICGEN_PREFLIGHT_OUTPUT_URI}musicgen-preflight-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/"
        materialized_job_file = _training_materialize_musicgen_preflight_job_file(output_path=output_path, request=request)
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "musicgen_trainer_preflight_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "model_family": "musicgen",
            "job_file": str(_TRAINING_MUSICGEN_PREFLIGHT_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "checkpoint": request.checkpoint,
            "trainer_implementation": _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION,
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {**summary, "output_path": output_path, "model_family": "musicgen"}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_musicgen_ar_trainer_job(request: TrainingStartRequest) -> dict[str, Any]:
    if request.variant not in {"no_cara_baseline", "cara_lite", "cara_probe", "cara_strong"}:
        raise HTTPException(status_code=400, detail="MusicGen variants are no_cara_baseline, cara_lite, cara_probe, and cara_strong.")
    if request.training_scope not in {"smoke", "full"}:
        raise HTTPException(status_code=400, detail="MusicGen training_scope must be smoke or full.")
    if request.training_scope == "full" and request.variant != "cara_strong":
        raise HTTPException(status_code=400, detail="Full MusicGen training is only implemented for cara_strong.")
    expected_confirmation = _training_musicgen_launch_confirmation_phrase(request.variant, request.training_scope)
    if str(request.launch_confirmation or "").strip() != expected_confirmation:
        raise HTTPException(status_code=409, detail=f"MusicGen {request.training_scope} launch requires typed confirmation: {expected_confirmation}")
    if request.trainer_compute_target not in {_TRAINING_H100_COMPUTE, _TRAINING_FULL_H100_COMPUTE}:
        raise HTTPException(status_code=400, detail="MusicGen trainer jobs are GPU-only; CPU fallback is disabled.")
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); MusicGen trainer jobs will not fall back to CPU.")
    max_allowed_steps = 100000 if request.training_scope == "full" else 2000
    if int(request.max_steps) < 1 or int(request.max_steps) > max_allowed_steps:
        raise HTTPException(status_code=400, detail=f"MusicGen {request.training_scope} max_steps must be between 1 and {max_allowed_steps}.")
    if int(request.batch_size) < 1 or int(request.batch_size) > 32:
        raise HTTPException(status_code=400, detail="MusicGen batch_size must be between 1 and 32.")
    if float(request.learning_rate) <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if float(request.attribution_loss_weight) < 0 or float(request.attribution_loss_weight) > 1:
        raise HTTPException(status_code=400, detail="attribution_loss_weight must be between 0 and 1.")
    token_cache = _training_latest_musicgen_token_cache()
    if not token_cache.get("passed"):
        raise HTTPException(status_code=409, detail=str(token_cache.get("reason") or "Cache MusicGen EnCodec tokens before launching trainer smoke."))
    preflight = _training_latest_musicgen_preflight()
    if not preflight.get("passed"):
        raise HTTPException(status_code=409, detail=str(preflight.get("reason") or "Run and pass MusicGen trainer preflight before launching trainer smoke."))
    sequence = _training_musicgen_smoke_sequence(registry_limit=200)
    if request.training_scope == "full" and not bool(sequence.get("variants", {}).get("cara_strong", {}).get("passed")):
        raise HTTPException(status_code=409, detail="Full MusicGen CARA-Strong fine-tuning unlocks only after step 11 CARA-Strong smoke has passed.")

    try:
        from azure.ai.ml import load_job

        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.run_name).strip("-") or "cara-musicgen-smoke"
        output_root = _TRAINING_MUSICGEN_FULL_OUTPUT_URI if request.training_scope == "full" else _TRAINING_MUSICGEN_SMOKE_OUTPUT_URI
        output_path = f"{output_root}{safe_run_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/"
        materialized_job_file = _training_materialize_musicgen_job_file(safe_run_name=safe_run_name, output_path=output_path, request=request)
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_variant = str(_azureml_input_scalar(submitted_job, "variant") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        serialized_run_name = str(_azureml_input_scalar(submitted_job, "run_name") or "")
        if serialized_variant != request.variant or serialized_dashboard_triggered != "true" or serialized_run_name != safe_run_name:
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized MusicGen trainer inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. expected variant={request.variant}, dashboard_triggered=true, run_name={safe_run_name}; "
                    f"got variant={serialized_variant or 'missing'}, dashboard_triggered={serialized_dashboard_triggered or 'missing'}, run_name={serialized_run_name or 'missing'}."
                ),
            )
        effective_batch_size = max(1, min(int(request.batch_size), 2))
        event = {
            "action": "musicgen_full_trainer_submitted" if request.training_scope == "full" else "musicgen_ar_smoke_trainer_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": request.trainer_compute_target,
            "environment": summary.get("environment"),
            "model_family": "musicgen",
            "trainer_implementation": _TRAINING_MUSICGEN_REAL_LM_IMPLEMENTATION,
            "job_file": str((_TRAINING_MUSICGEN_FULL_JOB_FILE if request.training_scope == "full" else _TRAINING_MUSICGEN_SMOKE_JOB_FILE).relative_to(ROOT)),
            "output_path": output_path,
            "run_name": safe_run_name,
            "variant": request.variant,
            "training_scope": request.training_scope,
            "max_steps": int(request.max_steps),
            "requested_batch_size": int(request.batch_size),
            "batch_size": effective_batch_size,
            "learning_rate": float(request.learning_rate),
            "attribution_loss_weight": float(request.attribution_loss_weight),
            "max_train_files": int(request.max_train_files),
            "min_encodec_frames": 2,
            "checkpoint": request.checkpoint or "facebook/musicgen-small",
            "model_dtype": "float32",
            "dry_run": request.dry_run,
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {**summary, "output_path": output_path, "model_family": "musicgen", "variant": request.variant, "training_scope": request.training_scope}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_ace_preflight_job(request: TrainingAcePreflightRunRequest) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not _TRAINING_ACE_PREFLIGHT_JOB_FILE.exists():
            raise FileNotFoundError(f"ACE-Step preflight job file not found: {_TRAINING_ACE_PREFLIGHT_JOB_FILE}")
        active_h100_jobs = _azureml_active_jobs_on_h100_computes()
        if active_h100_jobs:
            names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
            raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); ACE-Step preflight is GPU-only for CUDA/tap validation and will not fall back to CPU.")
        output_path = f"{_TRAINING_ACE_PREFLIGHT_OUTPUT_URI}ace-preflight-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/"
        materialized_job_file = _training_materialize_ace_preflight_job_file(output_path=output_path, request=request)
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "ace_step_env_preflight_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "required_environment": _TRAINING_ACE_ENVIRONMENT,
            "model_family": "ace_step",
            "job_file": str(_TRAINING_ACE_PREFLIGHT_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "checkpoint": request.checkpoint,
            "load_checkpoint": bool(request.load_checkpoint),
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {**summary, "output_path": output_path, "model_family": "ace_step"}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_stable_audio_trainer_preflight_job(request: TrainingStableAudioPreflightRunRequest) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not _TRAINING_STABLE_AUDIO_PREFLIGHT_JOB_FILE.exists():
            raise FileNotFoundError(f"Stable Audio trainer preflight job file not found: {_TRAINING_STABLE_AUDIO_PREFLIGHT_JOB_FILE}")
        hf_secret = _training_sync_hf_token_secret()
        job = load_job(source=_TRAINING_STABLE_AUDIO_PREFLIGHT_JOB_FILE)
        _azureml_set_job_input(job, "dashboard_triggered", "true")
        _azureml_set_job_input(job, "checkpoint", request.checkpoint)
        _azureml_set_job_input(job, "wrapper_check", "true" if request.wrapper_check else "false")
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
            "cara_training_gate": "stable_audio_trainer_preflight",
            "cara_model_family": "stable_audio_open_small",
            "cara_hf_auth": "workspace_key_vault",
            "cara_expected_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "stable_audio_trainer_preflight_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": summary.get("compute"),
            "environment": summary.get("environment"),
            "required_environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
            "model_family": "stable_audio_open_small",
            "job_file": str(_TRAINING_STABLE_AUDIO_PREFLIGHT_JOB_FILE.relative_to(ROOT)),
            "output_path": _TRAINING_STABLE_AUDIO_PREFLIGHT_OUTPUT_URI,
            "checkpoint": request.checkpoint,
            "wrapper_check": bool(request.wrapper_check),
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": _TRAINING_STABLE_AUDIO_PREFLIGHT_OUTPUT_URI,
            "model_family": "stable_audio_open_small",
            "checkpoint": request.checkpoint,
            "wrapper_check": bool(request.wrapper_check),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_context_diffusion_job(
    *,
    job_file: Path,
    action: str,
    output_path: str,
    dry_run: bool,
    compute_strategy: str = "prefer_h100_else_cpu",
    input_updates: dict[str, Any] | None = None,
    tag_updates: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not job_file.exists():
            raise FileNotFoundError(f"Context Diffusion job file not found: {job_file}")
        compute_selection = _training_select_preprocess_compute(compute_strategy)
        job = load_job(source=job_file)
        _azureml_set_job_input(job, "dashboard_triggered", "true")
        _azureml_set_job_input(job, "dry_run", "true" if dry_run else "false")
        for key, value in (input_updates or {}).items():
            _azureml_set_job_input(job, key, value)
        job.compute = compute_selection["compute"]
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
            "cara_model_family": "stable_audio_open_small_context_diffusion",
            "cara_cost_guardrail": "existing_azureml_workspace_only",
            "cara_compute_strategy": compute_selection["strategy"],
            "cara_compute_selected": compute_selection["compute"],
            **(tag_updates or {}),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": action,
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": compute_selection["compute"],
            "compute_strategy": compute_selection["strategy"],
            "compute_reason": compute_selection["reason"],
            "blocking_h100_jobs": [
                {"name": item.get("name"), "status": item.get("status"), "display_name": item.get("display_name")}
                for item in compute_selection["blocking_h100_jobs"]
            ],
            "environment": summary.get("environment"),
            "model_family": "stable_audio_open_small_context_diffusion",
            "job_file": str(job_file.relative_to(ROOT)),
            "output_path": output_path,
            "dry_run": dry_run,
        }
        if input_updates:
            event["inputs"] = input_updates
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "dry_run": dry_run,
            "model_family": "stable_audio_open_small_context_diffusion",
            "compute_selected": compute_selection,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_context_diffusion_packs_job(request: TrainingContextDiffusionPackRunRequest) -> dict[str, Any]:
    return _submit_context_diffusion_job(
        job_file=_TRAINING_CONTEXT_PACK_JOB_FILE,
        action="stable_audio_context_packs_submitted",
        output_path=_TRAINING_CONTEXT_PACK_OUTPUT_URI,
        dry_run=request.dry_run,
        input_updates={
            "max_contexts": max(1, min(8, int(request.max_contexts))),
            "selection_seed": str(request.selection_seed or "cara-context-v1"),
        },
        tag_updates={
            "cara_training_gate": "stable_audio_context_packs",
            "cara_step_id": "10",
        },
    )


def _submit_context_diffusion_cache_job(request: TrainingContextDiffusionCacheRunRequest) -> dict[str, Any]:
    return _submit_context_diffusion_job(
        job_file=_TRAINING_CONTEXT_CACHE_JOB_FILE,
        action="stable_audio_context_cache_submitted",
        output_path=_TRAINING_CONTEXT_CACHE_OUTPUT_URI,
        dry_run=request.dry_run,
        tag_updates={
            "cara_training_gate": "stable_audio_context_cache",
            "cara_step_id": "11",
        },
    )


def _submit_context_diffusion_preflight_job(request: TrainingContextDiffusionPreflightRunRequest) -> dict[str, Any]:
    return _submit_context_diffusion_job(
        job_file=_TRAINING_CONTEXT_PREFLIGHT_JOB_FILE,
        action="stable_audio_context_preflight_submitted",
        output_path=_TRAINING_CONTEXT_PREFLIGHT_OUTPUT_URI,
        dry_run=request.dry_run,
        tag_updates={
            "cara_training_gate": "stable_audio_context_preflight",
            "cara_step_id": "12",
        },
    )


def _submit_context_diffusion_smoke_job(request: TrainingContextDiffusionSmokeRunRequest) -> dict[str, Any]:
    try:
        from azure.ai.ml import load_job

        if not _TRAINING_CONTEXT_SMOKE_JOB_FILE.exists():
            raise FileNotFoundError(f"Context Diffusion smoke job file not found: {_TRAINING_CONTEXT_SMOKE_JOB_FILE}")
        job = load_job(source=_TRAINING_CONTEXT_SMOKE_JOB_FILE)
        input_updates = {
            "dashboard_triggered": "true",
            "dry_run": "true" if request.dry_run else "false",
            "max_steps": max(1, min(int(request.max_steps), 2000)),
            "batch_size": max(1, min(int(request.batch_size), 256)),
            "learning_rate": max(1e-6, min(float(request.learning_rate), 1e-2)),
            "max_train_rows": max(0, min(int(request.max_train_rows), 20000)),
            "max_eval_rows": max(0, min(int(request.max_eval_rows), 10000)),
        }
        for key, value in input_updates.items():
            _azureml_set_job_input(job, key, value)
        job.compute = _TRAINING_H100_COMPUTE
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
            "cara_model_family": "stable_audio_open_small_context_diffusion",
            "cara_cost_guardrail": "existing_azureml_workspace_only",
            "cara_compute_strategy": "gpu_only",
            "cara_compute_selected": _TRAINING_H100_COMPUTE,
            "cara_training_gate": "stable_audio_context_smoke",
            "cara_step_id": "13",
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        event = {
            "action": "stable_audio_context_smoke_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "compute_strategy": "gpu_only",
            "compute_reason": "Context smoke is GPU-only and uses the existing gpu-smoke-h100 Azure ML compute.",
            "environment": summary.get("environment"),
            "model_family": "stable_audio_open_small_context_diffusion",
            "job_file": str(_TRAINING_CONTEXT_SMOKE_JOB_FILE.relative_to(ROOT)),
            "output_path": _TRAINING_CONTEXT_SMOKE_OUTPUT_URI,
            "dry_run": request.dry_run,
            "inputs": input_updates,
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": _TRAINING_CONTEXT_SMOKE_OUTPUT_URI,
            "dry_run": request.dry_run,
            "model_family": "stable_audio_open_small_context_diffusion",
            "compute_selected": {"compute": _TRAINING_H100_COMPUTE, "strategy": "gpu_only"},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_context_diffusion_full_job(request: TrainingContextDiffusionFullRunRequest) -> dict[str, Any]:
    expected_confirmation = "LAUNCH CONTEXT FULL FINE-TUNE"
    if str(request.confirmation_phrase or "").strip() != expected_confirmation:
        raise HTTPException(status_code=409, detail=f"Context full launch requires typed confirmation: {expected_confirmation}")
    if int(request.max_steps) < 1 or int(request.max_steps) > 100000:
        raise HTTPException(status_code=400, detail="Context full max_steps must be between 1 and 100000.")
    if int(request.batch_size) < 1 or int(request.batch_size) > 32:
        raise HTTPException(status_code=400, detail="Context full batch_size must be between 1 and 32.")
    if int(request.num_workers) < 0 or int(request.num_workers) > 8:
        raise HTTPException(status_code=400, detail="Context full num_workers must be between 0 and 8. Use 0 to avoid Azure DataLoader worker OOM.")
    if int(request.checkpoint_keep_last_n) not in {0, 1}:
        raise HTTPException(status_code=400, detail="Context full checkpoint_keep_last_n must be 0 or 1; full runs use trainable_delta.pt.")
    if float(request.learning_rate) <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if float(request.attribution_loss_weight) < 0 or float(request.attribution_loss_weight) > 1:
        raise HTTPException(status_code=400, detail="attribution_loss_weight must be between 0 and 1.")

    ladder = _training_context_diffusion_ladder()
    if not ladder["context_smoke"].get("passed"):
        raise HTTPException(status_code=409, detail=str(ladder["context_smoke"].get("reason") or "Pass context smoke before launching full context fine-tune."))
    if ladder["context_full"].get("active"):
        latest = ladder["context_full"].get("latest_job") or {}
        raise HTTPException(status_code=409, detail=f"Context full job {latest.get('name') or latest.get('job_name') or 'latest'} is already active.")

    active_h100 = _training_recent_h100_jobs_from_registry(registry_limit=120)
    if active_h100:
        active = active_h100[0]
        raise HTTPException(
            status_code=409,
            detail=(
                f"H100-backed compute is already occupied by {active.get('name')} ({active.get('status')}) on {active.get('h100_compute_target')}. "
                "Wait for completion or hard-stop the active job before launching full context training."
            ),
        )

    try:
        from azure.ai.ml import load_job

        if not _TRAINING_CONTEXT_FULL_JOB_FILE.exists():
            raise FileNotFoundError(f"Context Diffusion full job file not found: {_TRAINING_CONTEXT_FULL_JOB_FILE}")
        safe_run_name = "cara-stable-audio-context-cara-strong-full"
        output_path = f"{_TRAINING_CONTEXT_FULL_OUTPUT_URI}{safe_run_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/"
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _training_materialize_context_full_job_file(
            safe_run_name=safe_run_name,
            output_path=output_path,
            request=request,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_variant = str(_azureml_input_scalar(submitted_job, "variant") or "")
        serialized_scope = str(_azureml_input_scalar(submitted_job, "training_scope") or "")
        serialized_context_mode = str(_azureml_input_scalar(submitted_job, "context_mode") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        serialized_run_name = str(_azureml_input_scalar(submitted_job, "run_name") or "")
        if (
            serialized_variant != "cara_strong"
            or serialized_scope != "full"
            or serialized_context_mode != "metadata_context_conditioning"
            or serialized_dashboard_triggered != "true"
            or serialized_run_name != safe_run_name
        ):
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized context trainer inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    "expected variant=cara_strong, training_scope=full, context_mode=metadata_context_conditioning, "
                    f"dashboard_triggered=true, run_name={safe_run_name}; got variant={serialized_variant or 'missing'}, "
                    f"training_scope={serialized_scope or 'missing'}, context_mode={serialized_context_mode or 'missing'}, "
                    f"dashboard_triggered={serialized_dashboard_triggered or 'missing'}, run_name={serialized_run_name or 'missing'}."
                ),
            )
        event = {
            "action": "stable_audio_context_full_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_FULL_H100_COMPUTE,
            "environment": summary.get("environment"),
            "model_family": "stable_audio_open_small_context_diffusion",
            "job_file": str(_TRAINING_CONTEXT_FULL_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "run_name": safe_run_name,
            "variant": "cara_strong_context_conditioned",
            "training_scope": "full",
            "context_mode": "metadata_context_conditioning",
            "max_steps": int(request.max_steps),
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "learning_rate": float(request.learning_rate),
            "attribution_loss_weight": float(request.attribution_loss_weight),
            "checkpoint_keep_last_n": int(request.checkpoint_keep_last_n),
            "checkpoint_strategy": "mounted_output_trainable_delta",
            "max_train_files": int(request.max_train_files),
            "max_eval_files": int(request.max_eval_files),
            "max_eval_batches": int(request.max_eval_batches),
            "dry_run": request.dry_run,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "dry_run": request.dry_run,
            "model_family": "stable_audio_open_small_context_diffusion",
            "variant": "cara_strong_context_conditioned",
            "training_scope": "full",
            "context_mode": "metadata_context_conditioning",
            "max_steps": int(request.max_steps),
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "learning_rate": float(request.learning_rate),
            "attribution_loss_weight": float(request.attribution_loss_weight),
            "checkpoint_keep_last_n": int(request.checkpoint_keep_last_n),
            "compute_selected": {"compute": _TRAINING_FULL_H100_COMPUTE, "strategy": "gpu_full_only"},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_stable_audio_smoke_trainer_job(request: TrainingStartRequest) -> dict[str, Any]:
    if request.model_family != "stable_audio_open_small":
        raise HTTPException(status_code=400, detail="Only the Stable Audio smoke trainer is implemented right now.")
    if request.training_scope not in {"smoke", "full"}:
        raise HTTPException(status_code=400, detail="Stable Audio training_scope must be smoke or full.")
    if request.variant not in {"no_cara_baseline", "cara_lite", "cara_head", "cara_strong"}:
        raise HTTPException(
            status_code=400,
            detail="Only no_cara_baseline, cara_lite, cara_head, and cara_strong Stable Audio smoke variants are implemented.",
        )
    if request.training_scope == "full" and request.variant != "cara_strong":
        raise HTTPException(status_code=400, detail="Full Stable Audio training is only implemented for the cara_strong variant.")
    expected_confirmation = _training_launch_confirmation_phrase(request.variant, request.training_scope)
    if str(request.launch_confirmation or "").strip() != expected_confirmation:
        raise HTTPException(
            status_code=409,
            detail=f"Stable Audio {request.training_scope} launch requires typed confirmation: {expected_confirmation}",
        )
    if request.trainer_compute_target not in {_TRAINING_H100_COMPUTE, _TRAINING_FULL_H100_COMPUTE}:
        raise HTTPException(status_code=400, detail="Stable Audio smoke training is GPU-only; CPU fallback is disabled.")
    max_allowed_steps = 100000 if request.training_scope == "full" else 2000
    full_training_sentinel = request.training_scope == "full" and bool(request.full_training_run) and int(request.max_steps) == 0
    if not full_training_sentinel and (int(request.max_steps) < 1 or int(request.max_steps) > max_allowed_steps):
        raise HTTPException(status_code=400, detail=f"{request.training_scope} max_steps must be between 1 and {max_allowed_steps}, or 0 for full-training-run dataset-pass mode.")
    if int(request.batch_size) < 1 or int(request.batch_size) > 32:
        raise HTTPException(status_code=400, detail="Smoke batch_size must be between 1 and 32.")
    if int(request.num_workers) < 0 or int(request.num_workers) > 8:
        raise HTTPException(status_code=400, detail="Stable Audio num_workers must be between 0 and 8. Use 0 to avoid Azure DataLoader worker OOM.")
    if int(request.checkpoint_keep_last_n) < 0 or int(request.checkpoint_keep_last_n) > 5:
        raise HTTPException(status_code=400, detail="checkpoint_keep_last_n must be between 0 and 5 to protect Azure node disk.")
    if float(request.learning_rate) <= 0:
        raise HTTPException(status_code=400, detail="learning_rate must be positive.")
    if float(request.attribution_loss_weight) < 0 or float(request.attribution_loss_weight) > 1:
        raise HTTPException(status_code=400, detail="attribution_loss_weight must be between 0 and 1.")
    active_trainer_jobs = _training_active_stable_audio_trainer_jobs(raise_on_error=True)
    if active_trainer_jobs:
        active = active_trainer_jobs[0]
        raise HTTPException(
            status_code=409,
            detail=(
                f"Stable Audio trainer job {active.get('name')} is already {active.get('status')}. "
                "Monitor it in Operations / Azure Runs, or use Hard stop there before launching another trainer run."
            ),
        )
    smoke_sequence = _training_stable_audio_smoke_sequence(registry_limit=200)
    if request.training_scope == "full" and not bool(smoke_sequence.get("variants", {}).get("cara_strong", {}).get("passed")):
        raise HTTPException(
            status_code=409,
            detail="Full CARA-Strong fine-tuning unlocks only after step 08 CARA-Strong smoke has passed.",
        )
    stable_audio_preflight = _training_latest_stable_audio_preflight(raise_on_error=True)
    if not stable_audio_preflight.get("passed"):
        raise HTTPException(
            status_code=409,
            detail=str(stable_audio_preflight.get("reason") or "Run and pass Stable Audio trainer preflight before launching smoke training."),
        )

    progress = _training_preprocess_progress("stable_audio_open_small")
    if float(progress.get("chunk_percent") or 0.0) < 99.5 and float(progress.get("duration_percent") or 0.0) < 99.5:
        raise HTTPException(
            status_code=409,
            detail="Stable Audio prepared dataset is not complete enough to launch smoke training.",
        )

    try:
        from azure.ai.ml import load_job

        job_file = _TRAINING_STABLE_AUDIO_FULL_JOB_FILE if request.training_scope == "full" else _TRAINING_STABLE_AUDIO_SMOKE_JOB_FILE
        if not job_file.exists():
            raise FileNotFoundError(f"Stable Audio trainer job file not found: {job_file}")
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.run_name).strip("-") or "cara-stable-audio-smoke"
        output_root = _TRAINING_STABLE_AUDIO_FULL_OUTPUT_URI if request.training_scope == "full" else _TRAINING_STABLE_AUDIO_SMOKE_OUTPUT_URI
        output_path = f"{output_root}{safe_run_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/"
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _training_materialize_stable_audio_job_file(
            safe_run_name=safe_run_name,
            output_path=output_path,
            request=request,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_variant = str(_azureml_input_scalar(submitted_job, "variant") or "")
        serialized_scope = str(_azureml_input_scalar(submitted_job, "training_scope") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        serialized_run_name = str(_azureml_input_scalar(submitted_job, "run_name") or "")
        if (
            serialized_variant != request.variant
            or serialized_scope != request.training_scope
            or serialized_dashboard_triggered != "true"
            or serialized_run_name != safe_run_name
        ):
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized trainer inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    f"expected variant={request.variant}, training_scope={request.training_scope}, dashboard_triggered=true, run_name={safe_run_name}; "
                    f"got variant={serialized_variant or 'missing'}, training_scope={serialized_scope or 'missing'}, dashboard_triggered={serialized_dashboard_triggered or 'missing'}, "
                    f"run_name={serialized_run_name or 'missing'}."
                ),
            )
        event = {
            "action": "stable_audio_full_trainer_submitted" if request.training_scope == "full" else "stable_audio_smoke_trainer_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": request.trainer_compute_target,
            "environment": summary.get("environment"),
            "model_family": "stable_audio_open_small",
            "job_file": str(job_file.relative_to(ROOT)),
            "output_path": output_path,
            "run_name": safe_run_name,
            "variant": request.variant,
            "training_scope": request.training_scope,
            "max_steps": 0 if request.training_scope == "full" and request.full_training_run else int(request.max_steps),
            "full_training_run": bool(request.full_training_run),
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "learning_rate": float(request.learning_rate),
            "attribution_loss_weight": float(request.attribution_loss_weight),
            "checkpoint_keep_last_n": int(request.checkpoint_keep_last_n),
            "max_train_files": int(request.max_train_files),
            "max_eval_files": int(request.max_eval_files),
            "max_eval_batches": int(request.max_eval_batches),
            "run_eval": bool(request.run_eval),
            "checkpoint": request.checkpoint,
            "dry_run": request.dry_run,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
        }
        _azureml_test_prep_audit(event)
        _training_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "dry_run": request.dry_run,
            "model_family": "stable_audio_open_small",
            "variant": request.variant,
            "training_scope": request.training_scope,
            "max_steps": 0 if request.training_scope == "full" and request.full_training_run else int(request.max_steps),
            "full_training_run": bool(request.full_training_run),
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "learning_rate": float(request.learning_rate),
            "attribution_loss_weight": float(request.attribution_loss_weight),
            "checkpoint_keep_last_n": int(request.checkpoint_keep_last_n),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _load_downloader_module():
    module_path = ROOT / "data_pipeline" / "02_freesound_downloader.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Freesound downloader module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("cara_freesound_downloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to build spec for Freesound downloader module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download_paths() -> tuple[dict[str, Any], Path, Path, Path, Path, Path]:
    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})
    return (
        freesound_cfg,
        ROOT / freesound_cfg.get("attribution_manifest_path", "data/attribution_manifest.jsonl"),
        ROOT / freesound_cfg.get("output_dir", "data/freesound"),
        ROOT / freesound_cfg.get("meta_dir", "data/freesound_meta"),
        ROOT / freesound_cfg.get("unavailable_log", "data/unavailable_freesound.csv"),
        ROOT / freesound_cfg.get("progress_path", "data/download_progress.json"),
    )


def _append_progress_activity(progress_path: Path, message: str, phase: str = "job", level: str = "info") -> None:
    progress = _load_json(progress_path, {"completed_ids": [], "metadata_only_ids": [], "unavailable_ids": [], "activity_log": []})
    activity_log = list(progress.get("activity_log", []))[-119:]
    activity_log.append({"ts": time.time(), "level": level, "phase": phase, "message": message})
    progress["activity_log"] = activity_log
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _is_retryable_unavailable_reason(reason: str) -> bool:
    lowered = (reason or "").lower()
    retryable_terms = (
        "temporary_",
        "rate limited",
        "429",
        "timeout",
        "timed out",
        "connection",
        "network",
        "access token expired",
        "token expired",
        "could not be refreshed",
        "refresh-token",
        "oauth",
    )
    permanent_terms = ("permanent_", "404", "403", "410", "not found", "forbidden")
    return any(term in lowered for term in retryable_terms) and not any(term in lowered for term in permanent_terms)


def _load_latest_unavailable_reasons(unavailable_log: Path) -> dict[str, str]:
    if not unavailable_log.exists():
        return {}
    reasons: dict[str, str] = {}
    with unavailable_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sound_id = str(row.get("sound_id") or "").strip()
            if sound_id:
                reasons[sound_id] = str(row.get("reason") or "")
    return reasons


def _run_download_job(request: DownloadNextRequest, count: int, target_batches: int) -> None:
    freesound_cfg, manifest_path, output_dir, meta_dir, unavailable_log, progress_path = _download_paths()
    downloader = _load_downloader_module()
    try:
        _append_progress_activity(progress_path, f"Background job started: {target_batches} batches × {count} items", phase="job")
        for batch_index in range(target_batches):
            if _job_stop_event.is_set():
                with _job_lock:
                    _job_state["requested_stop"] = True
                    _job_state["last_message"] = "Stop requested; ending between batches"
                _append_progress_activity(progress_path, "Stop requested; ending between batches", phase="job", level="warn")
                break
            acquired = _download_lock.acquire(blocking=False)
            if not acquired:
                _append_progress_activity(progress_path, "Download lock busy; retrying batch later", phase="job", level="warn")
                time.sleep(2)
                continue
            try:
                _last_activity.update({
                    "subset_mode": request.subset_mode,
                    "subset_role": request.subset_role,
                    "skip_audio": request.skip_audio,
                    "last_batch_started_at": time.time(),
                    "last_error": None,
                })
                _append_progress_activity(progress_path, f"Starting batch {batch_index + 1} / {target_batches}", phase="job")
                summary = downloader.download_freesound_subset(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    meta_dir=meta_dir,
                    unavailable_log=unavailable_log,
                    progress_path=progress_path,
                    limit=count,
                    skip_audio=request.skip_audio,
                    fetch_analysis=False,
                    require_confirmed=(request.subset_mode == "confirmed_only"),
                    subset_mode=request.subset_mode,
                    subset_role=request.subset_role,
                    bulk_metadata_batch_size=int(freesound_cfg.get("bulk_metadata_batch_size", 25)),
                    manifest_write_every=int(freesound_cfg.get("manifest_write_every", 25)),
                )
                _last_activity.update({
                    "subset_mode": request.subset_mode,
                    "subset_role": request.subset_role,
                    "skip_audio": request.skip_audio,
                    "last_summary": summary,
                    "last_batch_at": time.time(),
                    "last_error": None,
                })
                with _job_lock:
                    _job_state["completed_batches"] = int(_job_state.get("completed_batches", 0)) + 1
                    _job_state["last_message"] = f"Completed batch {_job_state['completed_batches']} / {target_batches}"
                if int(summary.get("requested", 0)) == 0:
                    with _job_lock:
                        _job_state["last_message"] = "No remaining items; background job complete"
                    _append_progress_activity(progress_path, "No remaining items; background job complete", phase="job")
                    break
            finally:
                _download_lock.release()
    except Exception as exc:
        _last_activity.update({
            "subset_mode": request.subset_mode,
            "subset_role": request.subset_role,
            "skip_audio": request.skip_audio,
            "last_summary": None,
            "last_batch_at": time.time(),
            "last_error": f"{exc}\n{traceback.format_exc()}",
        })
        with _job_lock:
            _job_state["last_message"] = str(exc)
        try:
            _append_progress_activity(progress_path, f"Background job failed: {exc}", phase="job", level="error")
        except Exception:
            pass
    finally:
        with _job_lock:
            _job_state["running"] = False
            _job_state["finished_at"] = time.time()
        _job_stop_event.clear()


@app.post("/api/data/download/start")
def data_download_start(request: DownloadNextRequest):
    valid_subset_modes = {"confirmed_only", "subset_role", "all_freesound"}
    if request.subset_mode not in valid_subset_modes:
        raise HTTPException(status_code=400, detail=f"Unsupported subset_mode: {request.subset_mode}")
    if request.subset_mode == "subset_role" and not (request.subset_role or "").strip():
        raise HTTPException(status_code=400, detail="subset_role is required when subset_mode is subset_role")
    count = max(1, min(int(request.count or 1), 25))
    target_items = max(1, min(int(request.target_items or count), 250000))
    target_batches = max(1, (target_items + count - 1) // count)
    global _job_thread
    with _job_lock:
        if _job_state.get("running"):
            raise HTTPException(status_code=409, detail="Download job is already running")
        _job_stop_event.clear()
        _job_state.update({
            "running": True,
            "requested_stop": False,
            "started_at": time.time(),
            "finished_at": None,
            "completed_batches": 0,
            "target_batches": target_batches,
            "target_items": target_items,
            "last_message": "Starting background download job",
        })
        _job_thread = threading.Thread(target=_run_download_job, args=(request, count, target_batches), daemon=True)
        _job_thread.start()
    return {"status": "started", "job": dict(_job_state)}


@app.post("/api/data/download/stop")
def data_download_stop():
    with _job_lock:
        if not _job_state.get("running"):
            return {"status": "not_running", "job": dict(_job_state)}
        _job_state["requested_stop"] = True
        _job_state["last_message"] = "Stop requested; current batch will finish first"
        _job_stop_event.set()
        return {"status": "stopping", "job": dict(_job_state)}


@app.post("/api/data/download/next")
def data_download_next(request: DownloadNextRequest):
    return data_download_start(request)


@app.post("/api/data/download/retry-unavailable")
def data_download_retry_unavailable(request: RetryUnavailableRequest):
    if request.mode not in {"temporary", "all"}:
        raise HTTPException(status_code=400, detail="mode must be temporary or all")
    if _job_state.get("running"):
        raise HTTPException(status_code=409, detail="Stop the download job before changing unavailable IDs")
    _, _, _, _, unavailable_log, progress_path = _download_paths()
    progress = _load_json(progress_path, {"completed_ids": [], "metadata_only_ids": [], "unavailable_ids": [], "unavailable_reasons": {}})
    unavailable_ids = {int(sound_id) for sound_id in progress.get("unavailable_ids", [])}
    unavailable_reasons = dict(progress.get("unavailable_reasons", {}) or {})
    latest_log_reasons = _load_latest_unavailable_reasons(unavailable_log)
    retry_ids: set[int] = set()
    for sound_id in unavailable_ids:
        key = str(sound_id)
        reason_info = unavailable_reasons.get(key)
        if request.mode == "all":
            retry_ids.add(sound_id)
            continue
        if isinstance(reason_info, dict):
            category = str(reason_info.get("category") or "")
            reason = str(reason_info.get("reason") or "")
            retryable = bool(reason_info.get("retryable", False)) or _is_retryable_unavailable_reason(f"{category}: {reason}")
        else:
            retryable = _is_retryable_unavailable_reason(str(reason_info or latest_log_reasons.get(key, "")))
        if retryable:
            retry_ids.add(sound_id)
    if not retry_ids:
        return {
            "status": "ok",
            "removed_count": 0,
            "remaining_unavailable_count": len(unavailable_ids),
            "unavailable_reason_count": len(unavailable_reasons),
        }
    progress["unavailable_ids"] = sorted(unavailable_ids - retry_ids)
    for sound_id in retry_ids:
        unavailable_reasons.pop(str(sound_id), None)
    progress["unavailable_reasons"] = unavailable_reasons
    progress["active_batch"] = None
    activity_log = list(progress.get("activity_log", []))[-119:]
    activity_log.append({
        "ts": time.time(),
        "level": "info",
        "phase": "retry",
        "message": f"Moved {len(retry_ids)} unavailable IDs back to retryable pool",
    })
    progress["activity_log"] = activity_log
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "removed_count": len(retry_ids),
        "remaining_unavailable_count": len(progress["unavailable_ids"]),
        "unavailable_reason_count": len(unavailable_reasons),
    }


def _load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_manifest_rows(path: Path, subset_mode: str = "confirmed_only", subset_role: str | None = None) -> int:
    return len(_manifest_candidate_ids(path, subset_mode=subset_mode, subset_role=subset_role))


def _manifest_candidate_ids(path: Path, subset_mode: str = "confirmed_only", subset_role: str | None = None) -> set[int]:
    if not path.exists():
        return set()
    candidate_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source") != "freesound":
                continue
            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                continue
            try:
                source_id_int = int(source_id)
            except ValueError:
                continue
            if subset_mode == "confirmed_only":
                if row.get("prefilter_status") == "confirmed":
                    candidate_ids.add(source_id_int)
                continue
            if subset_mode == "subset_role":
                if not row.get("include_in_subset"):
                    continue
                if subset_role and str(row.get("subset_role") or "") != subset_role:
                    continue
                candidate_ids.add(source_id_int)
                continue
            if subset_mode == "all_freesound":
                candidate_ids.add(source_id_int)
    return candidate_ids


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _pool_allocator_paths() -> AllocatorPaths:
    return AllocatorPaths.from_root(ROOT)


def _pool_allocator_v2_paths() -> AllocatorV2Paths:
    return AllocatorV2Paths.from_root(ROOT)


def _normalize_pool_engine(engine_version: str | None) -> str:
    engine = (engine_version or "v1").strip().lower()
    if engine in {"2", "v2", "pool_allocator_v2"}:
        return "v2"
    return "v1"


def _pool_allocator_candidate_count(options: RunOptions, engine_version: str = "v1") -> int:
    if _normalize_pool_engine(engine_version) == "v2":
        summary = summarize_registry_v2(_pool_allocator_v2_paths(), config=load_allocator_v2_config())
        return int(summary.get("candidate_asset_count", 0))
    rows = load_manifest_rows(_pool_allocator_paths().manifest_path)
    return len(_candidate_manifest_rows(rows, options))


def _pool_allocator_disk_download_count() -> tuple[int, int]:
    progress_path = ROOT / "data" / "download_progress.json"
    audio_dir = ROOT / "data" / "freesound"
    completed_ids = 0
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_ids = len(progress.get("completed_ids", []) or [])
        except json.JSONDecodeError:
            completed_ids = 0
    return completed_ids, _count_files(audio_dir)


def _pool_allocator_manifest_needs_reconcile(options: RunOptions, engine_version: str = "v1") -> tuple[bool, int, int, int]:
    if _normalize_pool_engine(engine_version) == "v2":
        candidate_count = _pool_allocator_candidate_count(options, engine_version="v2")
        completed_ids, audio_files_on_disk = _pool_allocator_disk_download_count()
        return False, candidate_count, completed_ids, audio_files_on_disk
    candidate_count = _pool_allocator_candidate_count(options)
    completed_ids, audio_files_on_disk = _pool_allocator_disk_download_count()
    expected_downloads = max(completed_ids, audio_files_on_disk)
    if not options.only_downloaded:
        return False, candidate_count, completed_ids, audio_files_on_disk
    needs_reconcile = expected_downloads > 0 and candidate_count < expected_downloads
    return needs_reconcile, candidate_count, completed_ids, audio_files_on_disk


def _reconcile_pool_allocator_manifest() -> dict[str, Any]:
    script_path = ROOT / "data_pipeline" / "12_reconcile_download_manifest.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parents[2] / "data_pipeline" / "12_reconcile_download_manifest.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Reconcile script not found at {script_path}")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = (completed.stdout or "").strip()
    return json.loads(stdout) if stdout else {}


def _run_pool_allocation_job(request: PoolAllocationRunRequest) -> None:
    engine_version = _normalize_pool_engine(request.engine_version)
    paths = _pool_allocator_paths()
    options = RunOptions(
        subset_role=(request.subset_role or "").strip() or None,
        only_downloaded=bool(request.only_downloaded),
        limit=max(1, int(request.limit)) if request.limit else None,
        allow_relaxed_metadata=bool(request.allow_relaxed_metadata),
        start_fresh=bool(request.start_fresh),
    )
    if engine_version == "v2":
        v2_paths = _pool_allocator_v2_paths()
        v2_options = RunOptionsV2(
            subset_role=(request.subset_role or "").strip() or None,
            only_downloaded=bool(request.only_downloaded),
            limit=max(1, int(request.limit)) if request.limit else None,
            allow_relaxed_metadata=bool(request.allow_relaxed_metadata),
            start_fresh=True if request.start_fresh is None else bool(request.start_fresh),
        )

        def _on_v2_progress(progress: dict[str, Any]) -> None:
            with _pool_job_lock:
                _pool_job_state["processed_assets"] = int(progress.get("processed_assets", 0))
                _pool_job_state["total_assets"] = int(progress.get("total_assets", 0))
                _pool_job_state["percent_complete"] = float(progress.get("percent_complete", 0.0))
                _pool_job_state["current_phase"] = progress.get("current_phase")
                _pool_job_state["current_asset"] = progress.get("current_asset")
                _pool_job_state["current_asset_title"] = progress.get("current_asset_title")
                _pool_job_state["current_pool_id"] = progress.get("current_pool_id")
                _pool_job_state["counts"] = dict(progress.get("counts", {}))
                activity_log = progress.get("activity_log", [])
                if activity_log:
                    _pool_job_state["last_message"] = activity_log[-1].get("message")

        try:
            result = run_pool_allocation_v2(
                v2_paths,
                options=v2_options,
                config=load_allocator_v2_config(),
                progress_callback=_on_v2_progress,
                stop_requested=_pool_job_stop_event.is_set,
            )
            with _pool_job_lock:
                _pool_job_state["running"] = False
                _pool_job_state["finished_at"] = time.time()
                _pool_job_state["processed_assets"] = int(result.get("processed_assets", 0))
                _pool_job_state["total_assets"] = int(result.get("total_assets", 0))
                _pool_job_state["percent_complete"] = 100.0 if result.get("status") == "completed" and int(result.get("total_assets", 0)) else float(_pool_job_state.get("percent_complete", 0.0))
                _pool_job_state["counts"] = dict(result.get("counts", {}))
                _pool_job_state["requested_stop"] = False
                _pool_job_state["last_message"] = "Pool allocation v2 run completed" if result.get("status") == "completed" else "Pool allocation v2 run paused"
                _pool_job_state["last_error"] = None
                _pool_job_state["latest_run_id"] = result.get("run_id")
                _pool_job_state["current_phase"] = result.get("status")
        except Exception as exc:
            with _pool_job_lock:
                _pool_job_state["running"] = False
                _pool_job_state["finished_at"] = time.time()
                _pool_job_state["last_message"] = "Pool allocation v2 run failed"
                _pool_job_state["last_error"] = f"{exc}\n{traceback.format_exc()}"
                _pool_job_state["requested_stop"] = False
        finally:
            _pool_job_stop_event.clear()
        return

    needs_reconcile, candidate_count, completed_ids, audio_files_on_disk = _pool_allocator_manifest_needs_reconcile(options)
    if needs_reconcile:
        with _pool_job_lock:
            _pool_job_state["last_message"] = (
                f"Manifest undercounts downloaded assets ({candidate_count} candidates vs {max(completed_ids, audio_files_on_disk)} downloads); reconciling manifest"
            )
        _reconcile_pool_allocator_manifest()

    def _on_progress(progress: dict[str, Any]) -> None:
        with _pool_job_lock:
            _pool_job_state["processed_assets"] = int(progress.get("processed_assets", 0))
            _pool_job_state["total_assets"] = int(progress.get("total_assets", 0))
            _pool_job_state["percent_complete"] = float(progress.get("percent_complete", 0.0))
            _pool_job_state["current_phase"] = progress.get("current_phase")
            _pool_job_state["current_asset"] = progress.get("current_asset")
            _pool_job_state["current_asset_title"] = progress.get("current_asset_title")
            _pool_job_state["current_pool_id"] = progress.get("current_pool_id")
            _pool_job_state["counts"] = dict(progress.get("counts", {}))
            activity_log = progress.get("activity_log", [])
            if activity_log:
                _pool_job_state["last_message"] = activity_log[-1].get("message")

    try:
        result = run_pool_allocation(
            paths,
            options=options,
            config=load_allocator_config(),
            progress_callback=_on_progress,
            stop_requested=_pool_job_stop_event.is_set,
        )
        with _pool_job_lock:
            _pool_job_state["running"] = False
            _pool_job_state["finished_at"] = time.time()
            _pool_job_state["processed_assets"] = int(result.get("processed_assets", 0))
            _pool_job_state["total_assets"] = int(result.get("total_assets", 0))
            _pool_job_state["percent_complete"] = 100.0 if result.get("status") == "completed" and int(result.get("total_assets", 0)) else float(_pool_job_state.get("percent_complete", 0.0))
            _pool_job_state["counts"] = dict(result.get("counts", {}))
            _pool_job_state["requested_stop"] = False
            _pool_job_state["last_message"] = "Pool allocation run completed" if result.get("status") == "completed" else "Pool allocation run paused"
            _pool_job_state["last_error"] = None
            _pool_job_state["latest_run_id"] = result.get("run_id")
            _pool_job_state["current_phase"] = result.get("status")
    except Exception as exc:
        with _pool_job_lock:
            _pool_job_state["running"] = False
            _pool_job_state["finished_at"] = time.time()
            _pool_job_state["last_message"] = "Pool allocation run failed"
            _pool_job_state["last_error"] = f"{exc}\n{traceback.format_exc()}"
            _pool_job_state["requested_stop"] = False
    finally:
        _pool_job_stop_event.clear()


@app.get("/api/data/status")
def data_status():
    data_dir = ROOT / "data"
    counts = {}
    if data_dir.exists():
        counts = {path.name: sum(1 for _ in path.rglob("*") if _.is_file()) for path in data_dir.iterdir() if path.is_dir()}
    coverage_path = ROOT / "data" / "enriched_metadata_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    return {"counts": counts, "coverage": coverage}


@app.get("/api/data/download-progress")
def data_download_progress(subset_mode: str = "subset_role", subset_role: Optional[str] = "music_train_candidate"):
    valid_subset_modes = {"confirmed_only", "subset_role", "all_freesound"}
    if subset_mode not in valid_subset_modes:
        raise HTTPException(status_code=400, detail=f"Unsupported subset_mode: {subset_mode}")

    config = load_pipeline_config()
    freesound_cfg = config.get("freesound", {})

    manifest_path = ROOT / freesound_cfg.get("attribution_manifest_path", "data/attribution_manifest.jsonl")
    progress_path = ROOT / freesound_cfg.get("progress_path", "data/download_progress.json")
    output_dir = ROOT / freesound_cfg.get("output_dir", "data/freesound")
    meta_dir = ROOT / freesound_cfg.get("meta_dir", "data/freesound_meta")
    unavailable_log = ROOT / freesound_cfg.get("unavailable_log", "data/unavailable_freesound.csv")

    progress = _load_json(progress_path, {
        "completed_ids": [],
        "metadata_only_ids": [],
        "unavailable_ids": [],
    })

    completed_ids = set(progress.get("completed_ids", []))
    metadata_only_ids = set(progress.get("metadata_only_ids", []))
    unavailable_ids = set(progress.get("unavailable_ids", []))
    active_subset_role: Optional[str] = None
    if subset_mode == "subset_role":
        active_subset_role = (subset_role or "").strip() or None
    candidate_ids = _manifest_candidate_ids(manifest_path, subset_mode=subset_mode, subset_role=active_subset_role)
    total_requested = len(candidate_ids)
    completed_candidate_ids = completed_ids & candidate_ids
    metadata_only_candidate_ids = metadata_only_ids & candidate_ids
    unavailable_candidate_ids = unavailable_ids & candidate_ids
    processed_ids = completed_candidate_ids | metadata_only_candidate_ids | unavailable_candidate_ids
    processed_count = len(processed_ids)
    remaining_count = max(total_requested - processed_count, 0)
    percent_complete = round((processed_count / total_requested) * 100, 2) if total_requested else 0.0
    last_summary = _last_activity.get("last_summary") if isinstance(_last_activity, dict) else None

    return {
        "status": "ready" if progress_path.exists() else "not_started",
        "subset_mode": subset_mode,
        "subset_role": active_subset_role,
        "last_batch_started_at": _last_activity.get("last_batch_started_at"),
        "last_batch_at": _last_activity.get("last_batch_at"),
        "last_error": _last_activity.get("last_error"),
        "manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "output_dir": str(output_dir),
        "meta_dir": str(meta_dir),
        "unavailable_log": str(unavailable_log),
        "total_requested": total_requested,
        "processed_count": processed_count,
        "remaining_count": remaining_count,
        "percent_complete": percent_complete,
        "downloaded_count": len(completed_candidate_ids),
        "metadata_only_count": len(metadata_only_candidate_ids),
        "unavailable_count": len(unavailable_candidate_ids),
        "global_downloaded_count": len(completed_ids),
        "global_metadata_only_count": len(metadata_only_ids),
        "global_unavailable_count": len(unavailable_ids),
        "audio_files_on_disk": _count_files(output_dir),
        "metadata_files_on_disk": _count_files(meta_dir),
        "last_updated": progress_path.stat().st_mtime if progress_path.exists() else None,
        "cached_metadata_hits": int((last_summary or {}).get("cached_metadata_hits", 0)),
        "bulk_metadata_calls": int((last_summary or {}).get("bulk_metadata_calls", 0)),
        "single_metadata_fallback_calls": int((last_summary or {}).get("single_metadata_fallback_calls", 0)),
        "downloads_skipped_existing": int((last_summary or {}).get("downloads_skipped_existing", 0)),
        "skipped_non_confirmed": int((last_summary or {}).get("skipped_non_confirmed", 0)),
        "api_requests_used_today": int((last_summary or {}).get("api_requests_used_today", 0)),
        "job": dict(_job_state),
        "progress": progress,
    }


@app.get("/api/data/pool-allocation/summary")
def pool_allocation_summary(engine: str = "v1"):
    engine_version = _normalize_pool_engine(engine)
    if engine_version == "v2":
        summary = summarize_registry_v2(_pool_allocator_v2_paths(), config=load_allocator_v2_config())
        completed_ids, audio_files_on_disk = _pool_allocator_disk_download_count()
        summary.update(
            {
                "completed_download_ids": completed_ids,
                "downloaded_audio_files_on_disk": audio_files_on_disk,
                "manifest_requires_reconcile": False,
            }
        )
        return summary
    summary = summarize_registry(_pool_allocator_paths(), config=load_allocator_config())
    options = RunOptions(subset_role="music_train_candidate", only_downloaded=True, allow_relaxed_metadata=True)
    needs_reconcile, candidate_count, completed_ids, audio_files_on_disk = _pool_allocator_manifest_needs_reconcile(options)
    summary.update(
        {
            "candidate_asset_count": candidate_count,
            "completed_download_ids": completed_ids,
            "downloaded_audio_files_on_disk": audio_files_on_disk,
            "manifest_requires_reconcile": needs_reconcile,
        }
    )
    return summary


@app.get("/api/sidebar/completion")
def sidebar_completion():
    items: dict[str, dict[str, Any]] = {}
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        progress = _load_json(ROOT / "data" / "download_progress.json", {})
        pool_progress = _load_json(ROOT / "registry" / "pool_allocator_v2" / "progress.json", {})
        downloaded_count = len(progress.get("completed_ids") or [])
        total_requested = int(pool_progress.get("total_assets") or downloaded_count or 0)
        remaining_count = max(total_requested - downloaded_count, 0)
        percent_complete = round((downloaded_count / total_requested) * 100, 2) if total_requested else 0.0
        dataset_complete = total_requested > 0 and downloaded_count >= total_requested
        items["dataset"] = {
            "complete": dataset_complete,
            "title": f"Dataset download complete · {downloaded_count:,}/{total_requested:,} files",
            "evidence": {
                "source": "data/download_progress.json",
                "percent_complete": percent_complete,
                "downloaded_count": downloaded_count,
                "total_requested": total_requested,
                "remaining_count": remaining_count,
            },
        }
    except Exception as exc:
        items["dataset"] = {"complete": False, "title": f"Dataset completion unavailable: {exc}"}

    try:
        pool_progress = _load_json(ROOT / "registry" / "pool_allocator_v2" / "progress.json", {})
        lock_summary = _load_json(_TRAINING_LOCK_DIR / "lock_summary.json", {})
        pool_count = int(lock_summary.get("pool_count") or (pool_progress.get("counts") or {}).get("new_pool_created") or 0)
        assigned_count = int(lock_summary.get("accepted_count") or pool_progress.get("processed_assets") or 0)
        total_assets = int(pool_progress.get("total_assets") or assigned_count or 0)
        pool_complete = (
            pool_count > 0
            and assigned_count > 0
            and str(pool_progress.get("status") or "").lower() == "completed"
            and (total_assets <= 0 or int(pool_progress.get("processed_assets") or 0) >= total_assets)
            and str(lock_summary.get("status") or "").lower() in {"", "ready", "completed"}
        )
        pool_item = {
            "complete": pool_complete,
            "title": f"Pool registry created · {pool_count:,} pools / {assigned_count:,} assigned files",
            "evidence": {
                "source": "registry/pool_allocator_v2/progress.json + registry/cara_strong/lock_summary.json",
                "pool_count": pool_count,
                "assigned_count": assigned_count,
                "total_assets": total_assets,
                "pool_progress_status": pool_progress.get("status"),
                "lock_status": lock_summary.get("status"),
            },
        }
        items["pool-creator"] = pool_item
        items["pool-viewer"] = {
            **pool_item,
            "title": f"Pool registry available · {pool_count:,} pools / {assigned_count:,} assigned files",
        }
    except Exception as exc:
        items["pool-creator"] = {"complete": False, "title": f"Pool completion unavailable: {exc}"}
        items["pool-viewer"] = {"complete": False, "title": f"Pool completion unavailable: {exc}"}

    try:
        from evaluation.benchmark_spec import model_lanes

        lanes_by_id = {str(lane.get("model_id") or ""): lane for lane in model_lanes()}
        diffusion = lanes_by_id.get("diffusion_cara_strong_full_modest_arch") or {}
        diffusion_complete = bool(diffusion.get("output_uri")) and str(diffusion.get("status") or "").lower() == "ready"
        items["finetune-diffusion"] = {
            "complete": diffusion_complete,
            "title": "Diffusion fine-tune complete · 9/9 steps" if diffusion_complete else "Diffusion fine-tune not complete",
            "evidence": {
                "source": "evaluation.benchmark_spec.model_lanes + registry/cara_strong/azure_training_jobs.jsonl",
                "status": diffusion.get("status"),
                "output_uri": diffusion.get("output_uri"),
                "latest_job": diffusion.get("latest_job"),
            },
        }
    except Exception as exc:
        items["finetune-diffusion"] = {"complete": False, "title": f"Diffusion completion unavailable: {exc}"}

    try:
        from evaluation.benchmark_spec import model_lanes

        lanes_by_id = {str(lane.get("model_id") or ""): lane for lane in model_lanes()}
        musicgen = lanes_by_id.get("musicgen_cara_strong_full") or {}
        musicgen_complete = bool(musicgen.get("output_uri")) and str(musicgen.get("status") or "").lower() == "ready"
        items["finetune-autoregressive"] = {
            "complete": musicgen_complete,
            "title": "Autoregressive fine-tune complete · 12/12 steps" if musicgen_complete else "Autoregressive fine-tune not complete",
            "evidence": {
                "source": "evaluation.benchmark_spec.model_lanes + registry/cara_strong/azure_training_jobs.jsonl",
                "status": musicgen.get("status"),
                "output_uri": musicgen.get("output_uri"),
                "latest_job": musicgen.get("latest_job"),
            },
        }
    except Exception as exc:
        items["finetune-autoregressive"] = {"complete": False, "title": f"Autoregressive completion unavailable: {exc}"}

    return {
        "generated_at": generated_at,
        "items": items,
    }


def _markdown_doc_payload(doc_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    path = Path(meta["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Documentation file not found: {doc_id}")
    stat = path.stat()
    return {
        "id": doc_id,
        "title": meta["title"],
        "description": meta["description"],
        "path": str(path.relative_to(ROOT)),
        "absolute_path": str(path),
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "content": path.read_text(encoding="utf-8", errors="replace"),
    }


@app.get("/api/docs/markdown")
def docs_markdown_index():
    docs = []
    for doc_id, meta in _DOCS_MARKDOWN_FILES.items():
        path = Path(meta["path"])
        stat = path.stat() if path.exists() else None
        docs.append(
            {
                "id": doc_id,
                "title": meta["title"],
                "description": meta["description"],
                "path": str(path.relative_to(ROOT)),
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
                "size_bytes": stat.st_size if stat else None,
                "available": path.exists(),
            }
        )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "docs": docs}


@app.get("/api/docs/markdown/{doc_id}")
def docs_markdown_file(doc_id: str):
    meta = _DOCS_MARKDOWN_FILES.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown documentation file: {doc_id}")
    return _markdown_doc_payload(doc_id, meta)


@app.get("/api/data/pool-allocation/run-status")
def pool_allocation_run_status(engine: str = "v1"):
    engine_version = _normalize_pool_engine(engine)
    summary = pool_allocation_summary(engine=engine_version)
    progress = read_progress_state_v2(_pool_allocator_v2_paths()) if engine_version == "v2" else read_progress_state(_pool_allocator_paths())
    return {
        "job": dict(_pool_job_state),
        "progress": progress,
        "summary": summary,
    }


@app.post("/api/data/pool-allocation/run")
def pool_allocation_run(request: PoolAllocationRunRequest):
    global _pool_job_thread
    with _pool_job_lock:
        if _pool_job_state.get("running"):
            raise HTTPException(status_code=409, detail="Pool allocation job is already running")
        _pool_job_stop_event.clear()
        _pool_job_state.update(
            {
                "running": True,
                "requested_stop": False,
                "started_at": time.time(),
                "finished_at": None,
                "processed_assets": 0,
                "total_assets": 0,
                "percent_complete": 0.0,
                "current_phase": "starting",
                "current_asset": None,
                "current_asset_title": None,
                "current_pool_id": None,
                "counts": {},
                "last_message": "Starting pool allocation job",
                "last_error": None,
                "latest_run_id": None,
                "options": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            }
        )
        _pool_job_thread = threading.Thread(target=_run_pool_allocation_job, args=(request,), daemon=True)
        _pool_job_thread.start()
    return {"status": "started", "job": dict(_pool_job_state)}


@app.post("/api/data/pool-allocation/reset")
def pool_allocation_reset(engine: str = "v1"):
    engine_version = _normalize_pool_engine(engine)
    with _pool_job_lock:
        if _pool_job_state.get("running"):
            raise HTTPException(status_code=409, detail="Cannot reset allocator while a job is running")
        result = reset_allocator_v2_state(_pool_allocator_v2_paths()) if engine_version == "v2" else reset_allocator_state(_pool_allocator_paths())
        _pool_job_stop_event.clear()
        _pool_job_state.update(
            {
                "running": False,
                "requested_stop": False,
                "started_at": None,
                "finished_at": None,
                "processed_assets": 0,
                "total_assets": 0,
                "percent_complete": 0.0,
                "current_phase": None,
                "current_asset": None,
                "current_asset_title": None,
                "current_pool_id": None,
                "counts": {},
                "last_message": "Allocator state cleared",
                "last_error": None,
                "latest_run_id": None,
                "options": {
                    "engine_version": engine_version,
                    "subset_role": None,
                    "only_downloaded": True,
                    "limit": None,
                    "allow_relaxed_metadata": False,
                    "start_fresh": False,
                },
            }
        )
    return {"status": "reset", "result": result, "job": dict(_pool_job_state)}


@app.post("/api/data/pool-allocation/stop")
def pool_allocation_stop():
    with _pool_job_lock:
        if not _pool_job_state.get("running"):
            return {"status": "not_running", "job": dict(_pool_job_state)}
        _pool_job_state["requested_stop"] = True
        _pool_job_state["last_message"] = "Stop requested; allocator will pause after the current asset"
        _pool_job_stop_event.set()
        return {"status": "stopping", "job": dict(_pool_job_state)}


@app.get("/api/data/pool-allocation/pools")
def pool_allocation_pools(engine: str = "v1"):
    return list_pools_v2(_pool_allocator_v2_paths()) if _normalize_pool_engine(engine) == "v2" else list_pools(_pool_allocator_paths(), config=load_allocator_config())


@app.get("/api/data/pool-allocation/assignments")
def pool_allocation_assignments(limit: int = 200, engine: str = "v1"):
    bounded_limit = max(1, min(int(limit), 1000))
    return list_assignments_v2(_pool_allocator_v2_paths(), limit=bounded_limit) if _normalize_pool_engine(engine) == "v2" else list_assignments(_pool_allocator_paths(), limit=bounded_limit)


@app.get("/api/data/pool-allocation/review-queue")
def pool_allocation_review_queue(limit: int = 200, engine: str = "v1"):
    bounded_limit = max(1, min(int(limit), 1000))
    return list_review_queue_v2(_pool_allocator_v2_paths(), limit=bounded_limit) if _normalize_pool_engine(engine) == "v2" else list_review_queue(_pool_allocator_paths(), limit=bounded_limit)


def _pool_viewer_manifest_path(engine: str = "v2") -> Path:
    return _pool_allocator_v2_paths().cara_manifest_path if _normalize_pool_engine(engine) == "v2" else _pool_allocator_paths().cara_manifest_path


def _pool_viewer_rows(engine: str = "v2") -> list[dict[str, Any]]:
    path = _pool_viewer_manifest_path(engine)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _duration_from_manifest_row(row: dict[str, Any]) -> float:
    for key in ("duration_seconds", "api_current_duration_s", "duration"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _pool_viewer_audio_path(row: dict[str, Any]) -> Path | None:
    for key in ("local_audio_path", "source_file_path"):
        raw_path = str(row.get(key) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    source_id = str(row.get("source_id") or row.get("raw_id") or "").strip()
    if source_id:
        audio_dir = ROOT / "data" / "freesound"
        for candidate in audio_dir.glob(f"{source_id}.*"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _pool_viewer_asset_summary(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = _pool_viewer_audio_path(row)
    return {
        "asset_id": row.get("cara_source_asset_id") or row.get("cara_v2_source_asset_id"),
        "pool_id": row.get("cara_source_pool_id") or row.get("cara_v2_source_pool_id"),
        "source": row.get("source"),
        "source_id": row.get("source_id") or row.get("raw_id"),
        "title": row.get("title") or row.get("api_current_name"),
        "artist": row.get("artist_primary") or row.get("author"),
        "licence_class": row.get("licence_class") or row.get("license_normalized"),
        "primary_genre": row.get("primary_genre"),
        "secondary_genre": row.get("secondary_genre"),
        "pool_family": row.get("cara_pool_family") or row.get("cara_v2_pool_family"),
        "duration_seconds": _duration_from_manifest_row(row),
        "style_tags": row.get("style_tags") or row.get("api_current_tags_json") or [],
        "audio_available": audio_path is not None,
        "audio_url": f"/api/data/pool-viewer/assets/{row.get('cara_source_asset_id') or row.get('cara_v2_source_asset_id')}/audio" if audio_path else None,
    }


@app.get("/api/data/pool-viewer/pools")
def pool_viewer_pools(engine: str = "v2"):
    rows = _pool_viewer_rows(engine)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        pool_id = str(row.get("cara_source_pool_id") or row.get("cara_v2_source_pool_id") or "").strip()
        if not pool_id:
            continue
        duration = _duration_from_manifest_row(row)
        entry = grouped.setdefault(
            pool_id,
            {
                "pool_id": pool_id,
                "pool_family": row.get("cara_pool_family") or row.get("cara_v2_pool_family"),
                "licence_class": row.get("licence_class") or row.get("license_normalized"),
                "territory": row.get("territory"),
                "primary_genres": set(),
                "asset_count": 0,
                "duration_seconds": 0.0,
                "audio_available_count": 0,
            },
        )
        entry["asset_count"] += 1
        entry["duration_seconds"] += duration
        if row.get("primary_genre"):
            entry["primary_genres"].add(row.get("primary_genre"))
        if _pool_viewer_audio_path(row):
            entry["audio_available_count"] += 1
    result = []
    for entry in grouped.values():
        entry = dict(entry)
        entry["primary_genres"] = sorted(entry["primary_genres"])
        result.append(entry)
    result.sort(key=lambda item: str(item.get("pool_id") or ""))
    return result


@app.get("/api/data/pool-viewer/pools/{pool_id}/assets")
def pool_viewer_pool_assets(pool_id: str, engine: str = "v2"):
    rows = [
        row
        for row in _pool_viewer_rows(engine)
        if str(row.get("cara_source_pool_id") or row.get("cara_v2_source_pool_id") or "") == pool_id
    ]
    return [_pool_viewer_asset_summary(row) for row in rows]


@app.get("/api/data/pool-viewer/assets/{asset_id}")
def pool_viewer_asset(asset_id: str, engine: str = "v2"):
    for row in _pool_viewer_rows(engine):
        current_asset_id = str(row.get("cara_source_asset_id") or row.get("cara_v2_source_asset_id") or "")
        if current_asset_id == asset_id:
            summary = _pool_viewer_asset_summary(row)
            return {
                **summary,
                "metadata": row,
            }
    raise HTTPException(status_code=404, detail="Pool asset not found")


@app.get("/api/data/pool-viewer/assets/{asset_id}/audio")
def pool_viewer_asset_audio(asset_id: str, engine: str = "v2"):
    for row in _pool_viewer_rows(engine):
        current_asset_id = str(row.get("cara_source_asset_id") or row.get("cara_v2_source_asset_id") or "")
        if current_asset_id != asset_id:
            continue
        audio_path = _pool_viewer_audio_path(row)
        if not audio_path:
            raise HTTPException(status_code=404, detail="Audio file not found for asset")
        return FileResponse(audio_path)
    raise HTTPException(status_code=404, detail="Pool asset not found")


@app.get("/api/registry/pools")
def registry_pools():
    pools_path = ROOT / "registry" / "pools.json"
    if not pools_path.exists():
        return []
    return json.loads(pools_path.read_text(encoding="utf-8"))


@app.get("/api/registry/hierarchy")
def registry_hierarchy():
    hierarchy_path = ROOT / "registry" / "hierarchy.json"
    if not hierarchy_path.exists():
        return {}
    return json.loads(hierarchy_path.read_text(encoding="utf-8"))


@app.get("/api/registry/codeword/{cw}")
def registry_codeword(cw: str):
    pools = registry_pools()
    for entry in pools:
        if entry.get("codeword") == cw:
            return entry
    raise HTTPException(status_code=404, detail="Codeword not found")


def _azureml_settings() -> dict[str, str]:
    config = load_project_config().get("azure_ml", {})
    return {
        "subscription_id": get_env("AZUREML_SUBSCRIPTION_ID", str(config.get("subscription_id") or "")) or "",
        "resource_group": get_env("AZUREML_RESOURCE_GROUP", str(config.get("resource_group") or "")) or "",
        "workspace_name": get_env("AZUREML_WORKSPACE_NAME", str(config.get("workspace_name") or "")) or "",
        "datastore_name": get_env("AZUREML_DATASTORE_NAME", str(config.get("datastore_name") or "ds_cara_raw_audio")) or "",
        "raw_audio_path": get_env("AZUREML_RAW_AUDIO_PATH", str(config.get("raw_audio_path") or "test-audio/")) or "",
    }


def _azureml_studio_url(job_name: str) -> str:
    settings = _azureml_settings()
    return (
        f"https://ml.azure.com/runs/{job_name}?wsid=/subscriptions/{settings['subscription_id']}"
        f"/resourcegroups/{settings['resource_group']}/workspaces/{settings['workspace_name']}"
    )


def _azureml_package_status() -> dict[str, bool]:
    status = {}
    for module in ("azure.ai.ml", "azure.identity", "azure.keyvault.secrets", "mlflow", "azureml.mlflow"):
        try:
            status[module] = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            status[module] = False
    return status


def _azureml_sdk_installed() -> bool:
    return all(_azureml_package_status().values())


def _azureml_missing_settings(settings: dict[str, str]) -> list[str]:
    return [key for key in ("subscription_id", "resource_group", "workspace_name") if not settings.get(key)]


@lru_cache(maxsize=1)
def _azureml_client():
    settings = _azureml_settings()
    missing = _azureml_missing_settings(settings)
    if missing:
        env_names = ", ".join(f"AZUREML_{key.upper()}" for key in missing)
        raise HTTPException(status_code=503, detail=f"Azure ML workspace is not configured. Set {env_names} in .env.")
    if not _azureml_sdk_installed():
        raise HTTPException(
            status_code=503,
            detail="Azure ML SDK is not installed. Run: .venv-dashboard/bin/python -m pip install -r gui/backend/requirements.txt",
        )
    try:
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential(process_timeout=20)
        return MLClient(
            credential,
            subscription_id=settings["subscription_id"],
            resource_group_name=settings["resource_group"],
            workspace_name=settings["workspace_name"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to create Azure ML client: {exc}") from exc


def _azureml_workspace_key_vault_url() -> str:
    settings = _azureml_settings()
    workspace = _azureml_client().workspaces.get(settings["workspace_name"])
    key_vault_arm_id = str(getattr(workspace, "key_vault", "") or "")
    vault_name = key_vault_arm_id.rstrip("/").split("/")[-1]
    if not vault_name:
        raise HTTPException(status_code=503, detail="Azure ML workspace did not return a Key Vault reference.")
    return f"https://{vault_name}.vault.azure.net/"


def _training_sync_hf_token_secret() -> dict[str, Any]:
    token = get_env("HF_TOKEN") or get_env("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise HTTPException(
            status_code=409,
            detail="HF_TOKEN is not set in .env. Add a Hugging Face token with gated-model read access before launching Stable Audio smoke training.",
        )
    if importlib.util.find_spec("azure.keyvault.secrets") is None:
        raise HTTPException(
            status_code=503,
            detail="Azure Key Vault SDK is not installed. Run: .venv-dashboard/bin/python -m pip install -r gui/backend/requirements.txt",
        )
    try:
        from azure.identity import AzureCliCredential
        from azure.keyvault.secrets import SecretClient

        vault_url = _azureml_workspace_key_vault_url()
        credential = AzureCliCredential(process_timeout=20)
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        try:
            secret_client.set_secret(_TRAINING_HF_TOKEN_SECRET_NAME, token)
        except Exception as secret_exc:
            message = str(secret_exc)
            if "does not have secrets set permission" not in message and "AccessDenied" not in message:
                raise
            existing = secret_client.get_secret(_TRAINING_HF_TOKEN_SECRET_NAME).value
            if not existing:
                raise
        return {
            "configured": True,
            "vault_url": vault_url,
            "secret_name": _TRAINING_HF_TOKEN_SECRET_NAME,
        }
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc)
        if "does not have secrets set permission" in message or "AccessDenied" in message:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Unable to sync HF_TOKEN to Azure Key Vault because the current Azure identity does not have "
                    "permission to set secrets on the workspace Key Vault. Grant this identity Key Vault secret "
                    "set/get access, or create secret 'hf-token' manually and grant this identity get access before launching."
                ),
            ) from exc
        raise HTTPException(status_code=503, detail=f"Unable to sync HF_TOKEN to Azure Key Vault: {exc}") from exc


def _azureml_iso(value: Any) -> Optional[str]:
    if value is None or value == {}:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    return None if text.strip() in {"", "{}"} else text


def _azureml_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _azureml_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_azureml_json_value(item) for item in value]
    return str(value)


def _azureml_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _azureml_json_value(value)
    if value is None:
        return {}
    if hasattr(value, "items"):
        return _azureml_json_value(dict(value.items()))
    return {}


def _azureml_job_summary(job: Any) -> dict[str, Any]:
    creation_context = getattr(job, "creation_context", None)
    compute = getattr(job, "compute", None)
    environment = getattr(job, "environment", None)
    return {
        "name": getattr(job, "name", None),
        "display_name": getattr(job, "display_name", None),
        "description": getattr(job, "description", None),
        "status": getattr(job, "status", None),
        "experiment_name": getattr(job, "experiment_name", None),
        "compute": str(compute) if compute else None,
        "environment": str(environment) if environment else None,
        "created_at": _azureml_iso(getattr(creation_context, "created_at", None)),
        "created_by": getattr(creation_context, "created_by", None),
        "start_time": _azureml_iso(getattr(job, "start_time", None)),
        "end_time": _azureml_iso(getattr(job, "end_time", None)),
        "studio_url": getattr(job, "studio_url", None),
        "tags": _azureml_mapping(getattr(job, "tags", None)),
        "properties": _azureml_mapping(getattr(job, "properties", None)),
        "services": _azureml_mapping(getattr(job, "services", None)),
    }


def _azureml_compute_summary(compute: Any) -> dict[str, Any]:
    return {
        "name": getattr(compute, "name", None),
        "type": getattr(compute, "type", None),
        "size": getattr(compute, "size", None),
        "location": getattr(compute, "location", None),
        "provisioning_state": getattr(compute, "provisioning_state", None),
        "min_instances": getattr(compute, "min_instances", None),
        "max_instances": getattr(compute, "max_instances", None),
        "idle_time_before_scale_down": getattr(compute, "idle_time_before_scale_down", None),
    }


def _azureml_environment_summary(environment: Any) -> dict[str, Any]:
    return {
        "name": getattr(environment, "name", None),
        "version": getattr(environment, "version", None),
        "description": getattr(environment, "description", None),
        "image": getattr(environment, "image", None),
        "tags": _azureml_mapping(getattr(environment, "tags", None)),
    }


def _azureml_operation_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Azure ML request failed: {exc}")


@app.get("/api/azureml/status")
def azureml_status():
    settings = _azureml_settings()
    packages = _azureml_package_status()
    sdk_installed = all(packages.values())
    missing = _azureml_missing_settings(settings)
    response: dict[str, Any] = {
        "configured": not missing,
        "sdk_installed": sdk_installed,
        "python_executable": sys.executable,
        "packages": packages,
        "missing_packages": [module for module, installed in packages.items() if not installed],
        "connected": False,
        "missing_settings": missing,
        "settings": {
            "workspace_name": settings["workspace_name"],
            "resource_group": settings["resource_group"],
            "datastore_name": settings["datastore_name"],
            "raw_audio_path": settings["raw_audio_path"],
        },
    }
    if missing or not sdk_installed:
        return response
    try:
        client = _azureml_client()
        next(iter(client.compute.list()), None)
        response["connected"] = True
        response["workspace"] = {
            "name": settings["workspace_name"],
        }
    except HTTPException as exc:
        response["error"] = exc.detail
    except Exception as exc:
        response["error"] = str(exc)
    return response


@app.get("/api/azureml/computes")
def azureml_computes():
    try:
        return [_azureml_compute_summary(compute) for compute in _azureml_client().compute.list()]
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.get("/api/azureml/environments")
def azureml_environments(limit: int = 200):
    try:
        environments = _azureml_client().environments.list()
        return [_azureml_environment_summary(environment) for _, environment in zip(range(max(1, min(limit, 1000))), environments)]
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.get("/api/azureml/jobs")
def azureml_jobs(limit: int = 100):
    try:
        jobs = _azureml_client().jobs.list()
        return [_azureml_job_summary(job) for _, job in zip(range(max(1, min(limit, 500))), jobs)]
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.get("/api/azureml/jobs/{job_name}")
def azureml_job(job_name: str):
    try:
        return _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _azureml_job_metrics_payload(job_name: str, history_limit: int = 500) -> dict[str, Any]:
    settings = _azureml_settings()
    if importlib.util.find_spec("mlflow") is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow is not installed. Run: python -m pip install mlflow-skinny azureml-mlflow",
        )
    try:
        workspace = _azureml_client().workspaces.get(settings["workspace_name"])
        tracking_uri = getattr(workspace, "mlflow_tracking_uri", None)
        if not tracking_uri:
            raise HTTPException(status_code=503, detail="Azure ML workspace did not return an MLflow tracking URI.")
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(job_name)
        latest = dict(run.data.metrics)
        histories = {}
        capped_limit = max(1, min(history_limit, 2000))
        for key in sorted(latest):
            points = client.get_metric_history(job_name, key)[-capped_limit:]
            histories[key] = [
                {"key": point.key, "value": point.value, "step": point.step, "timestamp": point.timestamp}
                for point in points
            ]
        return {
            "run_id": job_name,
            "latest": latest,
            "histories": histories,
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"MLflow metrics are not available for {job_name}: {exc}") from exc


def _azureml_uri_to_blob_prefix(uri: Any) -> str | None:
    text = str(uri or "").strip()
    marker = "/paths/"
    if not text or marker not in text:
        return None
    prefix = text.split(marker, 1)[1]
    return prefix.strip("/")


def _azureml_progress_number(mapping: dict[str, Any] | None, keys: list[str]) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
        except (TypeError, ValueError):
            continue
    return None


def _azureml_job_elapsed(summary: dict[str, Any], checked_at: datetime) -> float | None:
    properties = summary.get("properties") if isinstance(summary.get("properties"), dict) else {}
    start_time = None
    for candidate in (summary.get("start_time"), properties.get("StartTimeUtc"), summary.get("created_at")):
        start_time = _training_parse_datetime(candidate)
        if start_time:
            break
    end_time = _training_parse_datetime(summary.get("end_time"))
    if not start_time:
        return None
    return max(0.0, ((end_time or checked_at) - start_time).total_seconds())


def _azureml_progress_payload(
    *,
    job_name: str,
    checked_at: datetime,
    status: str | None,
    method: str,
    label: str,
    completed: int | float | None,
    total: int | float | None,
    unit: str,
    elapsed_seconds: float | None = None,
    latest_observed_at: str | None = None,
    note: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    percent = None
    estimated_remaining_seconds = None
    if completed is not None and total is not None and float(total) > 0:
        percent = min(100.0, max(0.0, float(completed) / float(total) * 100.0))
        if elapsed_seconds is not None and completed and float(completed) > 0 and float(completed) < float(total):
            estimated_remaining_seconds = elapsed_seconds * ((float(total) - float(completed)) / float(completed))
    return {
        "job_name": job_name,
        "checked_at": checked_at.isoformat(),
        "status": status,
        "method": method,
        "label": label,
        "completed": completed,
        "total": total,
        "unit": unit,
        "percent": round(percent, 2) if percent is not None else None,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 3) if estimated_remaining_seconds is not None else None,
        "latest_observed_at": latest_observed_at,
        "note": note,
        "error": error,
    }


def _azureml_musicgen_encodec_progress(summary: dict[str, Any], checked_at: datetime) -> dict[str, Any]:
    tags = summary.get("tags") if isinstance(summary.get("tags"), dict) else {}
    output_prefix = _azureml_uri_to_blob_prefix(tags.get("cara_output_path") or _TRAINING_MUSICGEN_TOKEN_CACHE_URI)
    token_prefix = f"{output_prefix.rstrip('/')}/tokens/" if output_prefix else "prepared/cara-strong-v0.4/musicgen_encodec_cache/tokens/"
    plan = _training_expected_preprocess_plan("musicgen")
    blobs = _azureml_datastore_blob_list(token_prefix)
    token_blobs = [
        blob
        for blob in blobs
        if str(blob.get("name") or "").lower().endswith(".pt") and int(blob.get("size") or 0) > 0
    ]
    completed = len(token_blobs)
    total = int(plan["expected_chunks"])
    latest_blob = max((str(blob.get("last_modified") or "") for blob in token_blobs), default=None)
    return _azureml_progress_payload(
        job_name=str(summary.get("name") or ""),
        checked_at=checked_at,
        status=summary.get("status"),
        method="read_only_encodec_token_blob_count",
        label="MusicGen EnCodec token cache",
        completed=completed,
        total=total,
        unit="token files",
        elapsed_seconds=_azureml_job_elapsed(summary, checked_at),
        latest_observed_at=latest_blob,
        note="Counts non-empty EnCodec .pt token files already visible in the Azure datastore. The final manifest is written only after the cache loop completes.",
    )


def _azureml_training_step_progress(summary: dict[str, Any], checked_at: datetime) -> dict[str, Any] | None:
    job_name = str(summary.get("name") or "").strip()
    if not job_name:
        return None
    metrics_payload = _azureml_job_metrics_payload(job_name, history_limit=2000)
    observed_step = _training_metric_observed_step(metrics_payload)
    tags = summary.get("tags") if isinstance(summary.get("tags"), dict) else {}
    params = metrics_payload.get("params") if isinstance(metrics_payload.get("params"), dict) else {}
    max_steps = _azureml_progress_number(params, ["max_steps", "max_train_steps", "trainer.max_steps", "steps"])
    if max_steps is None:
        max_steps = _azureml_progress_number(tags, ["cara_max_steps", "max_steps", "max_train_steps"])
    if (max_steps is None or max_steps <= 0) and str(tags.get("cara_full_training_run") or "").lower() == "true":
        batch_size = _azureml_progress_number(tags, ["cara_batch_size", "batch_size"]) or 0
        max_train_files = int(_azureml_progress_number(tags, ["cara_max_train_files", "max_train_files"]) or 0)
        if batch_size > 0:
            chunk_counts = _training_count_prepared_train_chunks(max_train_files)
            max_steps = max(1, math.ceil(int(chunk_counts.get("effective_train_chunks") or 0) / int(batch_size)))
    status = str(summary.get("status") or "").lower()
    if observed_step is None and status == "completed" and max_steps and max_steps > 0:
        observed_step = int(max_steps)
    if observed_step is None or not max_steps or max_steps <= 0:
        return None
    latest_observed_at = None
    histories = metrics_payload.get("histories") if isinstance(metrics_payload.get("histories"), dict) else {}
    for points in histories.values():
        if not isinstance(points, list):
            continue
        for point in points:
            try:
                timestamp = int(point.get("timestamp") or 0)
            except (AttributeError, TypeError, ValueError):
                continue
            if timestamp > 0:
                latest_observed_at = max(latest_observed_at or "", datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat())
    return _azureml_progress_payload(
        job_name=job_name,
        checked_at=checked_at,
        status=summary.get("status"),
        method="azure_mlflow_step_metrics",
        label="Training steps",
        completed=int(observed_step),
        total=int(max_steps),
        unit="steps",
        elapsed_seconds=_azureml_job_elapsed(summary, checked_at),
        latest_observed_at=latest_observed_at,
        note="Reads MLflow metric steps and compares them with configured max steps. Chunk/epoch progress remains an estimate for shuffled training.",
    )


def _azureml_job_progress_payload(job_name: str, *, force: bool = False) -> dict[str, Any]:
    now = time.time()
    cached = _AZUREML_JOB_PROGRESS_CACHE.get(job_name)
    if cached and not force and now - cached[0] < 60:
        return cached[1]
    checked_at = datetime.now(timezone.utc)
    summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    tags = summary.get("tags") if isinstance(summary.get("tags"), dict) else {}
    gate = str(tags.get("cara_training_gate") or "")
    model_family = str(tags.get("cara_model_family") or "")
    display_name = str(summary.get("display_name") or "")
    payload: dict[str, Any] | None = None
    try:
        if gate == "musicgen_encodec_token_cache" or "cache-musicgen-encodec" in display_name:
            payload = _azureml_musicgen_encodec_progress(summary, checked_at)
        elif model_family == "stable_audio_open_small" and ("trainer" in gate or "trainer" in display_name):
            payload = _azureml_training_step_progress(summary, checked_at)
        elif model_family == "musicgen" and ("trainer" in gate or "trainer" in display_name):
            payload = _azureml_training_step_progress(summary, checked_at)
    except Exception as exc:
        payload = _azureml_progress_payload(
            job_name=job_name,
            checked_at=checked_at,
            status=summary.get("status"),
            method="progress_unavailable",
            label="Progress",
            completed=None,
            total=None,
            unit="",
            elapsed_seconds=_azureml_job_elapsed(summary, checked_at),
            error=str(getattr(exc, "detail", None) or exc),
            note="Progress could not be estimated for this job without affecting the Azure ML run.",
        )
    if payload is None:
        payload = _azureml_progress_payload(
            job_name=job_name,
            checked_at=checked_at,
            status=summary.get("status"),
            method="progress_unavailable",
            label="Progress",
            completed=None,
            total=None,
            unit="",
            elapsed_seconds=_azureml_job_elapsed(summary, checked_at),
            note="No reliable progress denominator is available for this job type yet.",
        )
    _AZUREML_JOB_PROGRESS_CACHE[job_name] = (now, payload)
    return payload


@app.get("/api/azureml/job-progress")
def azureml_jobs_progress(limit: int = 200, active_only: bool = False):
    try:
        jobs = [_azureml_job_summary(job) for _, job in zip(range(max(1, min(limit, 500))), _azureml_client().jobs.list())]
        progress: dict[str, Any] = {}
        for job in jobs:
            job_name = str(job.get("name") or "").strip()
            if not job_name:
                continue
            if active_only and str(job.get("status") or "").lower() not in _AZUREML_ACTIVE_STATUSES:
                continue
            progress[job_name] = _azureml_job_progress_payload(job_name)
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "progress": progress,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.get("/api/azureml/job-progress/{job_name}")
def azureml_job_progress(job_name: str, force: bool = False):
    try:
        return _azureml_job_progress_payload(job_name, force=force)
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.get("/api/azureml/jobs/{job_name}/metrics")
def azureml_job_metrics(job_name: str, history_limit: int = 500):
    return _azureml_job_metrics_payload(job_name, history_limit=history_limit)


@app.get("/api/azureml/jobs/{job_name}/logs")
def azureml_job_logs(job_name: str, max_bytes: int = 200000):
    try:
        with tempfile.TemporaryDirectory(prefix="cara-azureml-logs-") as tmp_dir:
            _azureml_client().jobs.download(name=job_name, download_path=tmp_dir, all=False)
            remaining = max(1000, min(max_bytes, 1000000))
            files = []
            for path in sorted(Path(tmp_dir).rglob("*")):
                if not path.is_file() or remaining <= 0:
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                encoded = content.encode("utf-8")
                truncated = len(encoded) > remaining
                if truncated:
                    content = encoded[:remaining].decode("utf-8", errors="replace")
                remaining -= min(len(encoded), remaining)
                files.append({"path": str(path.relative_to(tmp_dir)), "content": content, "truncated": truncated})
            return {"job_name": job_name, "files": files}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Azure ML logs are not downloadable yet for {job_name}. Open the Studio link for live logs. Details: {exc}",
        ) from exc


@app.post("/api/azureml/jobs/{job_name}/cancel")
def azureml_job_cancel(job_name: str):
    try:
        _azureml_client().jobs.begin_cancel(job_name)
        return {"status": "cancelling", "job_name": job_name}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _azureml_test_prep_definitions() -> list[dict[str, Any]]:
    settings = _azureml_settings()
    input_path = f"azureml://datastores/{settings['datastore_name']}/paths/{settings['raw_audio_path']}"
    return [
        {
            "id": "01",
            "name": "01_data_access_test",
            "label": "Data Access",
            "description": "Verify Azure ML can read the private datastore audio folder and shared manifest.",
            "job_file": "azureml/jobs/01_data_access_test.yml",
            "compute": "cpu-prep-cluster",
            "environment": "azureml://registries/azureml/environments/sklearn-1.5/versions/1",
            "input_path": input_path,
            "gpu": False,
            "prerequisites": [],
        },
        {
            "id": "02",
            "name": "02_gpu_sanity_test",
            "label": "GPU Sanity",
            "description": "Start gpu-smoke-h100 and verify that PyTorch sees CUDA and the H100 device.",
            "job_file": "azureml/jobs/02_gpu_sanity_test.yml",
            "compute": "gpu-smoke-h100",
            "environment": "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/10",
            "input_path": None,
            "gpu": True,
            "prerequisites": ["01"],
        },
        {
            "id": "03",
            "name": "03_musicgen_env_test",
            "label": "MusicGen Environment",
            "description": "Verify AudioCraft import, CUDA, and shared dataset visibility in its dedicated environment.",
            "job_file": "azureml/jobs/03_musicgen_env_test.yml",
            "environment_file": "azureml/environments/env_musicgen_audiocraft.yml",
            "compute": "gpu-smoke-h100",
            "environment": "azureml:env-musicgen-audiocraft:3",
            "input_path": input_path,
            "gpu": True,
            "prerequisites": ["01", "02"],
        },
        {
            "id": "04",
            "name": "04_stableaudio_env_test",
            "label": "Stable Audio Environment",
            "description": "Verify Stable Audio Tools import, CUDA, and shared dataset visibility in its dedicated environment.",
            "job_file": "azureml/jobs/04_stableaudio_env_test.yml",
            "environment_file": "azureml/environments/env_stable_audio_tools.yml",
            "compute": "gpu-smoke-h100",
            "environment": _TRAINING_STABLE_AUDIO_ENVIRONMENT,
            "input_path": input_path,
            "gpu": True,
            "prerequisites": ["01", "02"],
        },
    ]


def _azureml_test_prep_definition(test_id: str) -> dict[str, Any]:
    definition = next((item for item in _azureml_test_prep_definitions() if item["id"] == test_id), None)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"Unknown Azure ML test-prep phase: {test_id}")
    return definition


def _azureml_test_prep_audit_path() -> Path:
    return ROOT / "registry" / "azureml_test_prep" / "audit.jsonl"


def _azureml_test_prep_report_cache_path(job_name: str) -> Path:
    return ROOT / "registry" / "azureml_test_prep" / "reports" / f"{job_name}.json"


def _azureml_test_prep_artifact_path(job_name: str) -> Path:
    return ROOT / "registry" / "azureml_test_prep" / "artifacts" / job_name


def _azureml_write_test_prep_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS.txt"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != checksum_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _azureml_test_prep_monitor_path(job_name: str) -> Path:
    return ROOT / "registry" / "azureml_test_prep" / "monitor" / f"{job_name}.jsonl"


def _azureml_test_prep_monitor_events(job_name: str, limit: int = 100) -> list[dict[str, Any]]:
    path = _azureml_test_prep_monitor_path(job_name)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-max(1, min(limit, 500)) :]


def _azureml_record_test_prep_monitor_observation(job_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    events = _azureml_test_prep_monitor_events(job_name)
    now = datetime.now(timezone.utc)
    observation = {
        "observed_at": now.isoformat(),
        "status": summary.get("status"),
        "compute": summary.get("compute"),
        "environment": summary.get("environment"),
    }
    should_write = not events or events[-1].get("status") != observation["status"]
    if events and not should_write:
        try:
            previous = datetime.fromisoformat(events[-1]["observed_at"])
            should_write = (now - previous).total_seconds() >= 60
        except (KeyError, TypeError, ValueError):
            should_write = True
    if should_write:
        path = _azureml_test_prep_monitor_path(job_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, sort_keys=True) + "\n")
        events.append(observation)
    return events[-100:]


def _azureml_test_prep_elapsed_seconds(created_at: Optional[str]) -> Optional[int]:
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at)
        return max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
    except ValueError:
        return None


def _azureml_test_prep_monitor_message(status: Optional[str]) -> str:
    normalized = str(status or "").lower()
    if normalized == "preparing":
        return (
            "Azure ML is preparing the run environment. During this stage Azure exposes a live "
            "control-plane heartbeat, but it does not expose downloadable image-build logs yet."
        )
    if normalized in {"queued", "notstarted", "starting", "provisioning"}:
        return "Azure ML has accepted the job and is allocating resources or preparing the run."
    if normalized == "running":
        return "The Azure ML command is running. Downloadable logs should now become available."
    if normalized in {"completed", "failed", "canceled", "cancelled", "notresponding", "paused"}:
        return "The Azure ML job is in a terminal or paused state. Downloadable logs are available."
    return f"Azure ML most recently reported state {status or 'unknown'}."


def _azureml_enrich_test_prep_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report", {})
    columns = report.get("manifest_columns", [])
    warnings = list(report.get("warnings", []))
    stable_names = {"title", "local_audio_path", "source_file_path", "freesound_id", "subset_role"}
    if columns and not stable_names.intersection(columns):
        warning = (
            "Manifest columns do not include expected stable field names. "
            "The CSV may be headerless or may have been exported without its schema row."
        )
        if warning not in warnings:
            warnings.append(warning)
    unnamed_count = sum(str(column).startswith("Unnamed:") for column in columns)
    if unnamed_count >= 5:
        warning = f"Manifest exposes {unnamed_count} unnamed columns; inspect the uploaded CSV export."
        if warning not in warnings:
            warnings.append(warning)
    if warnings:
        report["warnings"] = warnings
    return payload


def _azureml_test_prep_audit(event: dict[str, Any]) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CARA audio attribution survival",
        "phase": "test-prep",
        **event,
    }
    path = _azureml_test_prep_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _azureml_test_prep_audit_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _azureml_test_prep_audit_events(limit: int = 500) -> list[dict[str, Any]]:
    path = _azureml_test_prep_audit_path()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-max(1, min(limit, 2000)) :][::-1]


def _azureml_fetch_test_prep_jobs(limit: int = 50) -> list[dict[str, Any]]:
    settings = _azureml_settings()
    query = (
        "[?tags.cara_phase=='test-prep'].{"
        "name:name,display_name:display_name,status:status,experiment_name:experiment_name,"
        "compute:compute,environment:environment,created_at:creation_context.created_at,"
        "start_time:start_time,end_time:end_time,studio_url:services.Studio.endpoint,tags:tags,properties:properties}"
    )
    completed = subprocess.run(
        [
            "az",
            "ml",
            "job",
            "list",
            "--resource-group",
            settings["resource_group"],
            "--workspace-name",
            settings["workspace_name"],
            "--query",
            query,
            "--output",
            "json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "az ml job list failed")
    jobs = []
    for summary in json.loads(completed.stdout or "[]")[: max(1, min(limit, 100))]:
        tags = summary.get("tags", {})
        summary["test_id"] = str(tags.get("cara_test_id") or "")
        summary["test_name"] = tags.get("cara_test_name")
        summary["dashboard_triggered"] = str(tags.get("cara_dashboard_triggered") or "").lower() == "true"
        jobs.append(summary)
    return jobs


def _azureml_local_test_prep_jobs() -> list[dict[str, Any]]:
    jobs = []
    artifact_root = ROOT / "registry" / "azureml_test_prep" / "artifacts"
    for path in sorted(artifact_root.glob("*/azureml_job.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tags = payload.get("tags", {})
        if tags.get("cara_phase") != "test-prep":
            continue
        creation_context = payload.get("creation_context", {})
        services = payload.get("services", {})
        studio_service = services.get("Studio", {}) if isinstance(services, dict) else {}
        jobs.append(
            {
                "name": payload.get("name"),
                "display_name": payload.get("display_name"),
                "status": payload.get("status"),
                "experiment_name": payload.get("experiment_name"),
                "compute": payload.get("compute"),
                "environment": payload.get("environment"),
                "created_at": creation_context.get("created_at"),
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
                "studio_url": studio_service.get("endpoint"),
                "tags": tags,
                "properties": payload.get("properties", {}),
                "test_id": str(tags.get("cara_test_id") or ""),
                "test_name": tags.get("cara_test_name"),
                "dashboard_triggered": str(tags.get("cara_dashboard_triggered") or "").lower() == "true",
                "local_snapshot": True,
            }
        )
    return jobs


def _azureml_merge_test_prep_jobs(*job_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {}
    for jobs in job_lists:
        for job in jobs:
            name = str(job.get("name") or "")
            if name:
                merged[name] = {**merged.get(name, {}), **job}
    return list(merged.values())


def _azureml_cached_test_prep_job(job_name: str) -> Optional[dict[str, Any]]:
    with _azureml_test_prep_cache_lock:
        jobs = list(_azureml_test_prep_cache.get("jobs", []))
    return next((job for job in jobs if job.get("name") == job_name), None)


def _azureml_refresh_test_prep_jobs() -> None:
    try:
        jobs = _azureml_merge_test_prep_jobs(_azureml_local_test_prep_jobs(), _azureml_fetch_test_prep_jobs())
        with _azureml_test_prep_cache_lock:
            _azureml_test_prep_cache.update(
                {
                    "jobs": jobs,
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": None,
                }
            )
    except Exception as exc:
        with _azureml_test_prep_cache_lock:
            if not _azureml_test_prep_cache["jobs"]:
                _azureml_test_prep_cache["jobs"] = _azureml_local_test_prep_jobs()
            _azureml_test_prep_cache["last_error"] = str(exc)
    finally:
        with _azureml_test_prep_cache_lock:
            _azureml_test_prep_cache["refreshing"] = False


def _azureml_test_prep_cached_jobs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with _azureml_test_prep_cache_lock:
        if not _azureml_test_prep_cache["jobs"]:
            _azureml_test_prep_cache["jobs"] = _azureml_local_test_prep_jobs()
        cache = dict(_azureml_test_prep_cache)
        should_refresh = not cache["refreshing"]
        if should_refresh:
            _azureml_test_prep_cache["refreshing"] = True
    if should_refresh:
        threading.Thread(target=_azureml_refresh_test_prep_jobs, daemon=True).start()
    return list(cache["jobs"]), cache


def _azureml_test_prep_latest(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest = {}
    for job in sorted(jobs, key=lambda item: str(item.get("created_at") or ""), reverse=True):
        test_id = str(job.get("test_id") or "")
        if not test_id:
            continue
        status = str(job.get("status") or "").lower()
        priority = 0 if status not in {"completed", "failed", "canceled", "cancelled"} else 2 if status in {"canceled", "cancelled"} else 1
        existing_status = str(latest.get(test_id, {}).get("status") or "").lower()
        existing_priority = (
            0
            if existing_status and existing_status not in {"completed", "failed", "canceled", "cancelled"}
            else 2
            if existing_status in {"canceled", "cancelled"}
            else 1
        )
        if test_id not in latest or priority < existing_priority:
            latest[test_id] = job
    return latest


def _azureml_test_prep_passed(job: Optional[dict[str, Any]]) -> bool:
    return str((job or {}).get("status") or "").lower() in {"completed", "succeeded", "finished"}


def _azureml_test_prep_warnings(definition: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    latest = _azureml_test_prep_latest(jobs)
    warnings = []
    for test_id in definition.get("prerequisites", []):
        if not _azureml_test_prep_passed(latest.get(test_id)):
            warnings.append(f"Phase {test_id} has not passed yet.")
    return warnings


def _azureml_register_test_prep_environment(definition: dict[str, Any]) -> dict[str, Any]:
    environment_file = definition.get("environment_file")
    if not environment_file:
        raise HTTPException(status_code=400, detail=f"Phase {definition['id']} uses a curated environment and has nothing to register.")
    try:
        from azure.ai.ml import load_environment

        environment = load_environment(source=ROOT / environment_file)
        registered = _azureml_client().environments.create_or_update(environment)
        result = {
            "name": getattr(registered, "name", None),
            "version": getattr(registered, "version", None),
            "description": getattr(registered, "description", None),
        }
        _azureml_test_prep_audit(
            {
                "action": "environment_registered",
                "test_id": definition["id"],
                "environment": definition["environment"],
                "environment_file": environment_file,
                "result": result,
            }
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _azureml_submit_test_prep_job(
    definition: dict[str, Any],
    *,
    allow_prerequisite_override: bool = False,
) -> dict[str, Any]:
    jobs = _azureml_fetch_test_prep_jobs()
    warnings = _azureml_test_prep_warnings(definition, jobs)
    if warnings and not allow_prerequisite_override:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Test-prep prerequisites have not passed. Retry with an explicit prerequisite override for advanced use.",
                "warnings": warnings,
            },
        )
    try:
        from azure.ai.ml import load_job

        job = load_job(source=ROOT / definition["job_file"])
        job.inputs["dashboard_triggered"]._data = "true"
        job.tags = {
            **dict(job.tags or {}),
            "cara_dashboard_triggered": "true",
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        _azureml_test_prep_audit(
            {
                "action": "job_submitted",
                "test_id": definition["id"],
                "test_name": definition["name"],
                "job_name": summary.get("name"),
                "studio_url": summary.get("studio_url"),
                "compute": definition["compute"],
                "environment": definition["environment"],
                "input_path": definition.get("input_path"),
                "warnings": warnings,
            }
        )
        return {**summary, "test_id": definition["id"], "warnings": warnings}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _azureml_submit_test_prep_job_worker(
    definition: dict[str, Any],
    *,
    allow_prerequisite_override: bool,
) -> None:
    test_id = definition["id"]
    try:
        submitted = _azureml_submit_test_prep_job(
            definition,
            allow_prerequisite_override=allow_prerequisite_override,
        )
        with _azureml_test_prep_submission_lock:
            _azureml_test_prep_submission_state[test_id].update(
                {
                    "running": False,
                    "status": "submitted",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "job_name": submitted.get("name"),
                    "studio_url": submitted.get("studio_url"),
                }
            )
    except Exception as exc:
        with _azureml_test_prep_submission_lock:
            _azureml_test_prep_submission_state[test_id].update(
                {
                    "running": False,
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }
            )
        _azureml_test_prep_audit(
            {
                "action": "job_submission_failed",
                "test_id": test_id,
                "test_name": definition["name"],
                "error": str(exc),
            }
        )


def _azureml_queue_test_prep_job(
    definition: dict[str, Any],
    *,
    allow_prerequisite_override: bool,
) -> dict[str, Any]:
    test_id = definition["id"]
    with _azureml_test_prep_submission_lock:
        existing = _azureml_test_prep_submission_state.get(test_id, {})
        if existing.get("running"):
            raise HTTPException(status_code=409, detail=f"Phase {test_id} submission is already in progress.")
        existing_job_name = str(existing.get("job_name") or "")
        if existing_job_name:
            cached_job = _azureml_cached_test_prep_job(existing_job_name)
            cached_status = str((cached_job or {}).get("status") or "")
            if not cached_status or cached_status.lower() not in {"completed", "failed", "canceled", "cancelled"}:
                raise HTTPException(
                    status_code=409,
                    detail=f"Phase {test_id} Azure ML job {existing_job_name} is still {cached_status or 'active'}.",
                )
        state = {
            "running": True,
            "status": "submitting",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "job_name": None,
            "studio_url": None,
            "error": None,
        }
        _azureml_test_prep_submission_state[test_id] = state
    _azureml_test_prep_audit(
        {
            "action": "job_submission_requested",
            "test_id": test_id,
            "test_name": definition["name"],
            "compute": definition["compute"],
            "environment": definition["environment"],
            "input_path": definition.get("input_path"),
        }
    )
    threading.Thread(
        target=_azureml_submit_test_prep_job_worker,
        kwargs={
            "definition": definition,
            "allow_prerequisite_override": allow_prerequisite_override,
        },
        daemon=True,
    ).start()
    return dict(state)


def _azureml_wait_for_test_prep_job(job_name: str, poll_seconds: int = 20) -> dict[str, Any]:
    while True:
        job = _azureml_client().jobs.get(job_name)
        summary = _azureml_job_summary(job)
        if str(summary.get("status") or "").lower() in {"completed", "failed", "canceled", "cancelled"}:
            return summary
        time.sleep(poll_seconds)


def _azureml_test_prep_run_all_worker() -> None:
    try:
        for definition in _azureml_test_prep_definitions():
            with _azureml_test_prep_run_all_lock:
                _azureml_test_prep_run_all_state["current_test_id"] = definition["id"]
                _azureml_test_prep_run_all_state["last_message"] = f"Submitting phase {definition['id']}."
            if definition.get("environment_file"):
                _azureml_register_test_prep_environment(definition)
            submitted = _azureml_submit_test_prep_job(definition)
            job_name = str(submitted.get("name") or "")
            with _azureml_test_prep_run_all_lock:
                _azureml_test_prep_run_all_state["submitted_jobs"].append(job_name)
                _azureml_test_prep_run_all_state["last_message"] = f"Waiting for phase {definition['id']} job {job_name}."
            completed = _azureml_wait_for_test_prep_job(job_name)
            if not _azureml_test_prep_passed(completed):
                raise RuntimeError(f"Phase {definition['id']} ended with Azure ML status {completed.get('status')}.")
        with _azureml_test_prep_run_all_lock:
            _azureml_test_prep_run_all_state["last_message"] = "All four Azure ML test-prep phases passed."
    except Exception as exc:
        with _azureml_test_prep_run_all_lock:
            _azureml_test_prep_run_all_state["last_error"] = str(exc)
        _azureml_test_prep_audit({"action": "run_all_failed", "error": str(exc)})
    finally:
        with _azureml_test_prep_run_all_lock:
            _azureml_test_prep_run_all_state["running"] = False
            _azureml_test_prep_run_all_state["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/api/azureml/test-prep")
def azureml_test_prep():
    try:
        jobs, cache = _azureml_test_prep_cached_jobs()
        latest = _azureml_test_prep_latest(jobs)
        definitions = []
        for definition in _azureml_test_prep_definitions():
            definitions.append(
                {
                    **definition,
                    "latest_job": latest.get(definition["id"]),
                    "warnings": _azureml_test_prep_warnings(definition, jobs),
                }
            )
        return {
            "definitions": definitions,
            "history": jobs,
            "history_cache": {
                "refreshed_at": cache["refreshed_at"],
                "refreshing": cache["refreshing"],
                "last_error": cache["last_error"],
            },
            "audit": _azureml_test_prep_audit_events(),
            "submissions": dict(_azureml_test_prep_submission_state),
            "run_all": dict(_azureml_test_prep_run_all_state),
            "gpu_warning": (
                "This will start GPU compute on gpu-smoke-h100. "
                "Estimated Azure portal rate previously shown: approximately $10.12/hour. Confirm before running."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.post("/api/azureml/test-prep/{test_id}/run")
def azureml_test_prep_run(test_id: str, request: AzureMLTestPrepRunRequest):
    definition = _azureml_test_prep_definition(test_id)
    if definition["gpu"] and not request.confirm_gpu:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "This will start GPU compute on gpu-smoke-h100. "
                    "Estimated Azure portal rate previously shown: approximately $10.12/hour. Confirm before running."
                ),
                "requires_gpu_confirmation": True,
            },
        )
    return _azureml_queue_test_prep_job(
        definition,
        allow_prerequisite_override=request.allow_prerequisite_override,
    )


@app.post("/api/azureml/test-prep/{test_id}/jobs/{job_name}/cancel")
def azureml_test_prep_cancel(test_id: str, job_name: str):
    definition = _azureml_test_prep_definition(test_id)
    try:
        cached_job = _azureml_cached_test_prep_job(job_name)
        if cached_job and str(cached_job.get("test_id") or "") != test_id:
            raise HTTPException(status_code=409, detail=f"Azure ML job {job_name} does not belong to phase {test_id}.")
        status = str((cached_job or {}).get("status") or "")
        if status.lower() in {"completed", "failed", "canceled", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Azure ML job {job_name} is already {status}.")
        _azureml_client().jobs.begin_cancel(job_name)
        _azureml_test_prep_audit(
            {
                "action": "job_cancel_requested",
                "test_id": test_id,
                "test_name": definition["name"],
                "job_name": job_name,
                "cloud_status": status,
            }
        )
        return {"status": "cancelling", "job_name": job_name, "test_id": test_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.post("/api/azureml/test-prep/{test_id}/register-environment")
def azureml_test_prep_register_environment(test_id: str):
    return _azureml_register_test_prep_environment(_azureml_test_prep_definition(test_id))


@app.post("/api/azureml/test-prep/run-all")
def azureml_test_prep_run_all(request: AzureMLTestPrepRunRequest):
    if not request.confirm_gpu:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Run All submits short H100 phases after the CPU phase passes. "
                    "This will start GPU compute on gpu-smoke-h100. "
                    "Estimated Azure portal rate previously shown: approximately $10.12/hour. Confirm before running."
                ),
                "requires_gpu_confirmation": True,
            },
        )
    with _azureml_test_prep_run_all_lock:
        if _azureml_test_prep_run_all_state["running"]:
            raise HTTPException(status_code=409, detail="The Azure ML test-prep sequence is already running.")
        _azureml_test_prep_run_all_state.update(
            {
                "running": True,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "current_test_id": None,
                "last_message": "Starting sequential Azure ML test-prep run.",
                "last_error": None,
                "submitted_jobs": [],
            }
        )
    _azureml_test_prep_audit({"action": "run_all_started"})
    threading.Thread(target=_azureml_test_prep_run_all_worker, daemon=True).start()
    return dict(_azureml_test_prep_run_all_state)


@app.get("/api/azureml/test-prep/jobs/{job_name}/report")
def azureml_test_prep_report(job_name: str):
    try:
        cache_path = _azureml_test_prep_report_cache_path(job_name)
        if cache_path.exists():
            return _azureml_enrich_test_prep_report(json.loads(cache_path.read_text(encoding="utf-8")))
        job = _azureml_client().jobs.get(job_name)
        root = _azureml_test_prep_artifact_path(job_name)
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            _azureml_client().jobs.download(name=job_name, download_path=str(root), all=True)
        report_paths = sorted(root.rglob("report.json"))
        metadata_paths = sorted(root.rglob("metadata.json"))
        build_log_paths = sorted(root.rglob("20_image_build_log.txt"))
        if report_paths:
            report_path = report_paths[0]
            report = json.loads(report_path.read_text(encoding="utf-8"))
        elif build_log_paths:
            report_path = build_log_paths[0]
            build_log = report_path.read_text(encoding="utf-8", errors="replace")
            clean_build_log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", build_log).replace("\x08", "")
            diagnostic_lines = [
                line
                for line in clean_build_log.splitlines()
                if any(
                    marker in line.lower()
                    for marker in (
                        "error:",
                        "failed",
                        "exception",
                        "pkg-config",
                        "traceback",
                    )
                )
            ]
            report = {
                "test_name": getattr(job, "tags", {}).get("cara_test_name", job_name),
                "status": "failed",
                "failure_stage": "environment_image_build",
                "errors": diagnostic_lines[-30:] or clean_build_log.splitlines()[-30:],
                "message": "Azure ML failed while building the environment image, before the test script started.",
            }
        else:
            available_logs = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"No script report or image-build log is available yet for {job_name}.",
                    "available_logs": available_logs,
                },
            )
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8")) if metadata_paths else {}
        output = getattr(job, "outputs", {}).get("output_dir") if getattr(job, "outputs", None) else None
        payload = {
            "job_name": job_name,
            "report": report,
            "metadata": metadata,
            "output_artifact_location": getattr(output, "path", None),
            "report_path": str(report_path.relative_to(root)),
        }
        payload = _azureml_enrich_test_prep_report(payload)
        _azureml_write_test_prep_checksums(root)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Azure ML report is not downloadable yet for {job_name}: {exc}") from exc


@app.get("/api/azureml/test-prep/jobs/{job_name}/monitor")
def azureml_test_prep_monitor(job_name: str):
    try:
        _, cache = _azureml_test_prep_cached_jobs()
        cached_summary = _azureml_cached_test_prep_job(job_name) or {}
        with _azureml_test_prep_submission_lock:
            submission = next(
                (
                    dict(value)
                    for value in _azureml_test_prep_submission_state.values()
                    if value.get("job_name") == job_name
                ),
                {},
            )
        timeline = _azureml_test_prep_monitor_events(job_name)
        latest_observation = timeline[-1] if timeline else {}
        summary = {
            "name": job_name,
            **submission,
            **latest_observation,
            **cached_summary,
        }
        if not summary.get("status"):
            raise RuntimeError("No Azure ML state has been cached for this job yet.")
        timeline = _azureml_record_test_prep_monitor_observation(job_name, summary)
        status = str(summary.get("status") or "")
        cache_age_seconds = _azureml_test_prep_elapsed_seconds(cache.get("refreshed_at"))
        heartbeat_fresh = cache_age_seconds is not None and cache_age_seconds <= 90
        logs_downloadable = status.lower() in {
            "completed",
            "failed",
            "canceled",
            "cancelled",
            "notresponding",
            "paused",
        }
        return {
            "job_name": job_name,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "last_azure_heartbeat": timeline[-1].get("observed_at") if timeline else None,
            "azure_cache_refreshed_at": cache.get("refreshed_at"),
            "azure_cache_age_seconds": cache_age_seconds,
            "elapsed_seconds": _azureml_test_prep_elapsed_seconds(summary.get("created_at")),
            "status": status,
            "compute": summary.get("compute"),
            "environment": summary.get("environment"),
            "studio_url": summary.get("studio_url") or _azureml_studio_url(job_name),
            "logs_downloadable": logs_downloadable,
            "heartbeat_fresh": heartbeat_fresh,
            "heartbeat_error": cache.get("last_error"),
            "message": (
                _azureml_test_prep_monitor_message(status)
                if heartbeat_fresh
                else (
                    f"{_azureml_test_prep_monitor_message(status)} "
                    "The latest background Azure refresh is older than 90 seconds, so this response is using the most recent cached state."
                )
            ),
            "timeline": timeline,
        }
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Azure ML live monitor is unavailable for {job_name}: {exc}") from exc


@app.get("/api/training/status")
def training_status():
    checkpoint_path = ROOT / "checkpoints" / "attribution_head_v1.pt"
    readiness = _training_readiness_payload()
    return {
        "checkpoint_exists": checkpoint_path.exists(),
        "status": readiness["status"],
        "readiness": readiness,
    }


@app.get("/api/training/readiness")
def training_readiness(variant: str = "all"):
    return _training_readiness_payload(variant)


@app.get("/api/training/hybrid-readiness")
def training_hybrid_readiness():
    return _training_hybrid_readiness_payload()


@app.get("/api/training/context-diffusion-readiness")
def training_context_diffusion_readiness():
    return _training_context_diffusion_readiness_payload()


@app.get("/api/training/preprocess-progress")
def training_preprocess_progress(model: str = "stable_audio_open_small"):
    model_key = _training_preprocess_model_key(model)
    if model_key == "all":
        raise HTTPException(status_code=400, detail="Choose one model for preprocessing progress: stable_audio_open_small or musicgen.")
    return _training_preprocess_progress(model_key)


@app.get("/api/training/run-progress")
def training_run_progress(model: str = "latest"):
    progress = _training_latest_training_progress(model=model)
    if progress is None:
        return {"status": "not_started", "progress": None}
    if _training_progress_model_key(model) == "all":
        return {"status": "available", "progress": progress}
    return {"status": "available", "progress": progress}


@app.post("/api/training/lock-manifest")
def training_lock_manifest(request: TrainingManifestLockRequest):
    try:
        from data_pipeline.cara_strong_lock import (
            DEFAULT_MANIFEST,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_POOLS,
            lock_cara_strong_manifest,
        )

        manifest_path = Path(request.manifest_path).resolve() if request.manifest_path else DEFAULT_MANIFEST
        pool_registry_path = Path(request.pool_registry_path).resolve() if request.pool_registry_path else DEFAULT_POOLS
        output_dir = Path(request.output_dir).resolve() if request.output_dir else DEFAULT_OUTPUT_DIR
        summary = lock_cara_strong_manifest(
            manifest_path=manifest_path,
            pools_path=pool_registry_path,
            output_dir=output_dir,
            require_audio_exists=request.require_audio_exists,
            dry_run=request.dry_run,
        )
        return {"status": summary.get("status"), "summary": summary, "readiness": _training_readiness_payload()}
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"CARA-Strong manifest lock failed: {exc}") from exc


@app.post("/api/training/confirm-azure-upload")
def training_confirm_azure_upload(request: TrainingAzureUploadConfirmRequest):
    if request.confirmed and not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before confirming the Azure upload.")
    _TRAINING_AZURE_UPLOAD_CONFIRMATION.parent.mkdir(parents=True, exist_ok=True)
    if request.confirmed:
        payload = {
            "confirmed": True,
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
            "confirmed_by": "dashboard",
            "source_root": _TRAINING_SOURCE_URI,
            "expected_manifest": f"{_TRAINING_SOURCE_URI}data/cara_pool_manifest_v2.jsonl",
            "expected_audio_root": f"{_TRAINING_SOURCE_URI}data/freesound/",
        }
        _TRAINING_AZURE_UPLOAD_CONFIRMATION.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _training_append_job_event({"action": "azure_upload_confirmed", **payload})
    else:
        if _TRAINING_AZURE_UPLOAD_CONFIRMATION.exists():
            _TRAINING_AZURE_UPLOAD_CONFIRMATION.unlink()
        _training_append_job_event(
            {
                "action": "azure_upload_confirmation_cleared",
                "confirmed": False,
                "source_root": _TRAINING_SOURCE_URI,
            }
        )
    return {"status": "confirmed" if request.confirmed else "cleared", "readiness": _training_readiness_payload()}


@app.post("/api/training/preprocess-model-datasets")
def training_preprocess_model_datasets(request: TrainingPreprocessRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before preparing model datasets.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before preparing model datasets.")
    submitted = _submit_training_preprocess_job(
        dry_run=request.dry_run,
        models=request.models,
        compute_strategy=request.compute_strategy,
    )
    return {"status": "submitted", "job": submitted, "readiness": _training_readiness_payload()}


@app.post("/api/training/cache-musicgen-tokens")
def training_cache_musicgen_tokens(request: TrainingMusicGenTokenCacheRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before caching MusicGen EnCodec tokens.")
    submitted = _submit_musicgen_token_cache_job(
        dry_run=request.dry_run,
        compute_strategy=request.compute_strategy,
    )
    return {"status": "submitted", "job": submitted, "readiness": _training_readiness_payload()}


@app.post("/api/training/musicgen-preflight")
def training_musicgen_preflight(request: TrainingMusicGenPreflightRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before running MusicGen trainer preflight.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before running MusicGen trainer preflight.")
    token_cache = _training_latest_musicgen_token_cache()
    if not token_cache.get("passed"):
        raise HTTPException(status_code=409, detail=str(token_cache.get("reason") or "Cache MusicGen EnCodec tokens before running trainer preflight."))
    submitted = _submit_musicgen_trainer_preflight_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_readiness_payload()}


@app.post("/api/training/ace-preflight")
def training_ace_preflight(request: TrainingAcePreflightRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before running ACE-Step preflight.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before running ACE-Step preflight.")
    submitted = _submit_ace_preflight_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_hybrid_readiness_payload()}


@app.post("/api/training/stable-audio-preflight")
def training_stable_audio_preflight(request: TrainingStableAudioPreflightRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before running Stable Audio trainer preflight.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before running Stable Audio trainer preflight.")
    progress = _training_preprocess_progress("stable_audio_open_small")
    if float(progress.get("chunk_percent") or 0.0) < 99.5 and float(progress.get("duration_percent") or 0.0) < 99.5:
        raise HTTPException(status_code=409, detail="Stable Audio prepared dataset is not complete enough to run trainer preflight.")
    submitted = _submit_stable_audio_trainer_preflight_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_readiness_payload()}


@app.post("/api/training/context-diffusion/packs")
def training_context_diffusion_packs(request: TrainingContextDiffusionPackRunRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before preparing Context Diffusion packs.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before preparing Context Diffusion packs.")
    full_training = _training_latest_stable_audio_full_training_cached()
    if not full_training.get("passed"):
        raise HTTPException(
            status_code=409,
            detail=str(full_training.get("reason") or "Complete the original Stable Audio CARA-Strong full run before branching Context Diffusion."),
        )
    submitted = _submit_context_diffusion_packs_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_context_diffusion_readiness_payload()}


@app.post("/api/training/context-diffusion/cache")
def training_context_diffusion_cache(request: TrainingContextDiffusionCacheRunRequest):
    ladder = _training_context_diffusion_ladder()
    if not ladder["context_packs"].get("passed"):
        raise HTTPException(status_code=409, detail=str(ladder["context_packs"].get("reason") or "Complete context packs before caching context metadata."))
    submitted = _submit_context_diffusion_cache_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_context_diffusion_readiness_payload()}


@app.post("/api/training/context-diffusion/preflight")
def training_context_diffusion_preflight(request: TrainingContextDiffusionPreflightRunRequest):
    ladder = _training_context_diffusion_ladder()
    if not ladder["context_cache"].get("passed"):
        raise HTTPException(status_code=409, detail=str(ladder["context_cache"].get("reason") or "Complete context cache before context preflight."))
    submitted = _submit_context_diffusion_preflight_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_context_diffusion_readiness_payload()}


@app.post("/api/training/context-diffusion/smoke")
def training_context_diffusion_smoke(request: TrainingContextDiffusionSmokeRunRequest):
    ladder = _training_context_diffusion_ladder()
    if not ladder["context_preflight"].get("passed"):
        raise HTTPException(status_code=409, detail=str(ladder["context_preflight"].get("reason") or "Complete context preflight before context smoke."))
    if ladder["context_smoke"].get("active"):
        latest = ladder["context_smoke"].get("latest_job") or {}
        raise HTTPException(status_code=409, detail=f"Context smoke job {latest.get('name') or latest.get('job_name') or 'latest'} is already active.")
    submitted = _submit_context_diffusion_smoke_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_context_diffusion_readiness_payload()}


@app.post("/api/training/context-diffusion/full")
def training_context_diffusion_full(request: TrainingContextDiffusionFullRunRequest):
    submitted = _submit_context_diffusion_full_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_context_diffusion_readiness_payload()}


@app.post("/api/training/start")
def training_start(request: TrainingStartRequest):
    if not _training_lock_state().get("locked"):
        raise HTTPException(status_code=409, detail="Lock the CARA-Strong manifest before launching smoke training.")
    if not _training_azure_upload_state().get("confirmed"):
        raise HTTPException(status_code=409, detail="Confirm the full Azure dataset upload before launching smoke training.")
    if request.model_family == "musicgen":
        submitted = _submit_musicgen_ar_trainer_job(request)
    else:
        submitted = _submit_stable_audio_smoke_trainer_job(request)
    return {"status": "submitted", "job": submitted, "readiness": _training_readiness_payload()}


@app.post("/api/training/stop")
def training_stop():
    return {"status": "not_implemented"}


@app.post("/api/generate")
def generate(request: GenerateRequest):
    return {
        "prompt": request.prompt,
        "audio_url": None,
        "attr_string": "ATTR|PENDING@100|PENDING@00|PENDING@00|END",
        "validation_result": {"state": "exception"},
        "baseline_results": {},
    }


@app.post("/api/validate")
def validate(request: ValidateRequest):
    from registry.validate import CARACodebook
    from validation.validator import CARAValidator

    pools_path = ROOT / "registry" / "pools.json"
    hierarchy_path = ROOT / "registry" / "hierarchy.json"
    if not pools_path.exists() or not hierarchy_path.exists():
        raise HTTPException(status_code=400, detail="Registry not built yet")
    codebook = CARACodebook(pools_path, hierarchy_path)
    validator = CARAValidator(codebook)
    result = validator.validate(request.attr_string)
    return {
        "state": result.state.value,
        "original_string": result.original_string,
        "validated_string": result.validated_string,
        "errors": result.errors,
        "repairs": result.repairs,
    }


@app.get("/api/evaluation/metrics")
def evaluation_metrics():
    metrics_path = ROOT / "evaluation" / "metrics_latest.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _evaluation_suites() -> list[dict[str, Any]]:
    suites_path = ROOT / "evaluation" / "prompt_suites.json"
    if not suites_path.exists():
        return []
    return json.loads(suites_path.read_text(encoding="utf-8"))


@app.get("/api/evaluation/repairability-schema")
def evaluation_repairability_schema():
    from evaluation.cara_repairability import repairability_schema

    return repairability_schema()


@app.get("/api/evaluation/readiness")
def evaluation_readiness():
    from evaluation.cara_repairability import repairability_schema
    from evaluation.model_registry import evaluation_readiness_payload

    latest_job = _latest_evaluation_job_state_fast()
    payload = evaluation_readiness_payload(_evaluation_suites())
    latest_smoke_result = _latest_generated_audio_job_state_fast(scope="smoke")
    latest_full_result = _latest_generated_audio_job_state_fast(scope="full")
    latest_score_result = (
        _latest_attribution_score_job_state_fast(source_audio_job_name=str(latest_full_result.get("job_name")))
        if latest_full_result and latest_full_result.get("job_name")
        else _latest_attribution_score_job_state_fast()
    )
    payload["repairability"] = repairability_schema()
    payload["latest_evaluation_job"] = latest_job
    payload["active_generated_audio_job"] = _active_generated_audio_job_state()
    payload["latest_generated_audio_result"] = _latest_generated_audio_job_state_fast()
    payload["latest_generated_audio_smoke_result"] = latest_smoke_result
    payload["latest_generated_audio_full_result"] = latest_full_result
    payload["active_attribution_scoring_job"] = _active_attribution_score_job_state()
    payload["latest_attribution_scoring_result"] = latest_score_result
    payload["benchmark_prompt_set"] = _evaluation_prompt_set_state(latest_job)
    return payload


@app.get("/api/evaluation/prompt-set")
def evaluation_prompt_set():
    latest_job = _latest_evaluation_job_state()
    return {
        "format": "cara_benchmark_prompt_set_state_v2",
        "latest_evaluation_job": latest_job,
        "benchmark_prompt_set": _evaluation_prompt_set_state(latest_job),
    }


@app.get("/api/evaluation/audio-benchmark/readiness")
def evaluation_audio_benchmark_readiness():
    return _audio_benchmark_defaults()


@app.get("/api/evaluation/audio-benchmark/progress")
def evaluation_audio_benchmark_progress(job_name: Optional[str] = None):
    return _generated_audio_progress_state(job_name)


def _submit_stable_audio_audio_benchmark(
    request: EvaluationAudioBenchmarkRunRequest,
    plan: dict[str, Any],
    *,
    skip_duplicate_checks: bool = False,
) -> dict[str, Any]:
    if not _evaluation_audio_confirmation_matches(request.launch_confirmation, allow_legacy=True):
        raise HTTPException(
            status_code=409,
            detail=f"Type {_EVALUATION_AUDIO_BENCHMARK_CONFIRMATION} to submit a live generated-audio scoring job.",
        )
    if not _EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.exists():
        raise HTTPException(status_code=409, detail=f"Generated-audio job file is missing: {_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE}")
    active_audio_job = _active_generated_audio_job_state()
    if active_audio_job and not skip_duplicate_checks:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated-audio benchmark is already active in Azure ML "
                f"({active_audio_job.get('job_name')}, status={active_audio_job.get('status')}). "
                "Refresh the Testing page or open the running job instead of submitting a duplicate."
            ),
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs and not skip_duplicate_checks:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); generated-audio benchmark will not fall back to CPU.")

    try:
        from azure.ai.ml import load_job

        scope = str(plan["scope"])
        run_slug = datetime.now(timezone.utc).strftime(f"stable-audio-audio-{scope}-%Y%m%d-%H%M%S")
        output_path = f"{_EVALUATION_STABLE_AUDIO_OUTPUT_URI}audio_{scope}/{run_slug}/"
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _evaluation_materialize_stable_audio_audio_job_file(
            output_path=output_path,
            prompt_manifest_uri=str(plan["prompt_manifest_uri"]),
            request=request,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        expected_models = ",".join(request.model_ids)
        expected_suites = ",".join(request.suite_ids)
        expected_seed_ids = ",".join(str(int(seed)) for seed in request.seed_ids)
        expected_scope = str(plan["scope"])
        expected_max_prompts = str(int(plan["max_prompts"]))
        serialized_models = str(_azureml_input_scalar(submitted_job, "model_ids") or "")
        serialized_suites = str(_azureml_input_scalar(submitted_job, "suite_ids") or "")
        serialized_seed_ids = str(_azureml_input_scalar(submitted_job, "seed_ids") or "")
        serialized_scope = str(_azureml_input_scalar(submitted_job, "scope") or "")
        serialized_max_prompts = str(_azureml_input_scalar(submitted_job, "max_prompts") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        if (
            serialized_models != expected_models
            or serialized_suites != expected_suites
            or serialized_seed_ids != expected_seed_ids
            or serialized_scope != expected_scope
            or serialized_max_prompts != expected_max_prompts
            or serialized_dashboard_triggered != "true"
        ):
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized generated-audio inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    f"expected model_ids={expected_models}, suite_ids={expected_suites}, seed_ids={expected_seed_ids}, "
                    f"scope={expected_scope}, max_prompts={expected_max_prompts}, dashboard_triggered=true; "
                    f"got model_ids={serialized_models or 'missing'}, suite_ids={serialized_suites or 'missing'}, "
                    f"seed_ids={serialized_seed_ids or 'missing'}, scope={serialized_scope or 'missing'}, "
                    f"max_prompts={serialized_max_prompts or 'missing'}, dashboard_triggered={serialized_dashboard_triggered or 'missing'}."
                ),
            )
        event = {
            "action": "benchmark_testing_stable_audio_audio_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "job_file": str(_EVALUATION_STABLE_AUDIO_AUDIO_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "prompt_manifest_uri": plan["prompt_manifest_uri"],
            "model_ids": request.model_ids,
            "suite_ids": request.suite_ids,
            "seed_ids": request.seed_ids,
            "max_prompts": int(plan["max_prompts"]),
            "scope": plan["scope"],
            "trained_model_data": _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
            "marketplace_resources": False,
        }
        _azureml_test_prep_audit(event)
        _evaluation_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "model_ids": request.model_ids,
            "suite_ids": request.suite_ids,
            "seed_ids": request.seed_ids,
            "max_prompts": int(plan["max_prompts"]),
            "scope": plan["scope"],
            "prompt_manifest_uri": plan["prompt_manifest_uri"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_musicgen_audio_benchmark(
    request: EvaluationAudioBenchmarkRunRequest,
    plan: dict[str, Any],
    *,
    skip_duplicate_checks: bool = False,
) -> dict[str, Any]:
    from evaluation.benchmark_spec import model_lanes

    if not _evaluation_audio_confirmation_matches(request.launch_confirmation, allow_legacy=True):
        raise HTTPException(
            status_code=409,
            detail=f"Type {_EVALUATION_AUDIO_BENCHMARK_CONFIRMATION} to submit a live generated-audio scoring job.",
        )
    if not _EVALUATION_MUSICGEN_AUDIO_JOB_FILE.exists():
        raise HTTPException(status_code=409, detail=f"MusicGen generated-audio job file is missing: {_EVALUATION_MUSICGEN_AUDIO_JOB_FILE}")
    active_audio_job = _active_generated_audio_job_state()
    if active_audio_job and not skip_duplicate_checks:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated-audio benchmark is already active in Azure ML "
                f"({active_audio_job.get('job_name')}, status={active_audio_job.get('status')}). "
                "Refresh the Testing page or open the running job instead of submitting a duplicate."
            ),
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs and not skip_duplicate_checks:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); generated-audio benchmark will not fall back to CPU.")

    lanes_by_id = {lane["model_id"]: lane for lane in model_lanes()}
    trained_model_uri = (lanes_by_id.get("musicgen_cara_strong_full") or {}).get("output_uri")
    if "musicgen_cara_strong_full" in request.model_ids and not trained_model_uri:
        raise HTTPException(status_code=409, detail="MusicGen CARA-Strong full-run output is missing; cannot benchmark MusicGen CARA lane.")

    try:
        from azure.ai.ml import load_job

        scope = str(plan["scope"])
        run_slug = datetime.now(timezone.utc).strftime(f"musicgen-audio-{scope}-%Y%m%d-%H%M%S")
        output_path = f"{_EVALUATION_MUSICGEN_OUTPUT_URI}audio_{scope}/{run_slug}/"
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _evaluation_materialize_musicgen_audio_job_file(
            output_path=output_path,
            prompt_manifest_uri=str(plan["prompt_manifest_uri"]),
            trained_model_uri=str(trained_model_uri or _TRAINING_MUSICGEN_FULL_OUTPUT_URI),
            request=request,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        expected_models = ",".join(request.model_ids)
        expected_suites = ",".join(request.suite_ids)
        expected_seed_ids = ",".join(str(int(seed)) for seed in request.seed_ids)
        expected_scope = str(plan["scope"])
        expected_max_prompts = str(int(plan["max_prompts"]))
        serialized_models = str(_azureml_input_scalar(submitted_job, "model_ids") or "")
        serialized_suites = str(_azureml_input_scalar(submitted_job, "suite_ids") or "")
        serialized_seed_ids = str(_azureml_input_scalar(submitted_job, "seed_ids") or "")
        serialized_scope = str(_azureml_input_scalar(submitted_job, "scope") or "")
        serialized_max_prompts = str(_azureml_input_scalar(submitted_job, "max_prompts") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        if (
            serialized_models != expected_models
            or serialized_suites != expected_suites
            or serialized_seed_ids != expected_seed_ids
            or serialized_scope != expected_scope
            or serialized_max_prompts != expected_max_prompts
            or serialized_dashboard_triggered != "true"
        ):
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized MusicGen generated-audio inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    f"expected model_ids={expected_models}, suite_ids={expected_suites}, seed_ids={expected_seed_ids}, "
                    f"scope={expected_scope}, max_prompts={expected_max_prompts}, dashboard_triggered=true; "
                    f"got model_ids={serialized_models or 'missing'}, suite_ids={serialized_suites or 'missing'}, "
                    f"seed_ids={serialized_seed_ids or 'missing'}, scope={serialized_scope or 'missing'}, "
                    f"max_prompts={serialized_max_prompts or 'missing'}, dashboard_triggered={serialized_dashboard_triggered or 'missing'}."
                ),
            )
        event = {
            "action": "benchmark_testing_musicgen_audio_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "job_file": str(_EVALUATION_MUSICGEN_AUDIO_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "prompt_manifest_uri": plan["prompt_manifest_uri"],
            "model_ids": request.model_ids,
            "suite_ids": request.suite_ids,
            "seed_ids": request.seed_ids,
            "max_prompts": int(plan["max_prompts"]),
            "scope": plan["scope"],
            "trained_model_data": trained_model_uri,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
            "marketplace_resources": False,
        }
        _azureml_test_prep_audit(event)
        _evaluation_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "model_ids": request.model_ids,
            "suite_ids": request.suite_ids,
            "seed_ids": request.seed_ids,
            "max_prompts": int(plan["max_prompts"]),
            "scope": plan["scope"],
            "prompt_manifest_uri": plan["prompt_manifest_uri"],
            "trained_model_data": trained_model_uri,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.post("/api/evaluation/audio-benchmark/run")
def evaluation_audio_benchmark_run(request: EvaluationAudioBenchmarkRunRequest):
    plan = _audio_benchmark_plan(request)
    if request.dry_run:
        return {
            "status": "planned",
            "dry_run": True,
            "message": "Dry run only. No Azure ML audio benchmark job was submitted.",
            "plan": plan,
        }
    if not _evaluation_audio_confirmation_matches(request.launch_confirmation, allow_legacy=False):
        raise HTTPException(status_code=409, detail=f"Type {_EVALUATION_AUDIO_BENCHMARK_CONFIRMATION} to submit a live generated-audio benchmark.")
    active_audio_job = _active_generated_audio_job_state()
    if active_audio_job:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated-audio benchmark is already active in Azure ML "
                f"({active_audio_job.get('job_name')}, status={active_audio_job.get('status')}). "
                "Refresh the Testing page or open the running job instead of submitting a duplicate."
            ),
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); generated-audio benchmark will not fall back to CPU.")
    jobs: list[dict[str, Any]] = []
    model_groups = plan.get("model_groups") if isinstance(plan.get("model_groups"), dict) else {}
    stable_models = list(model_groups.get("stable_audio") or [])
    musicgen_models = list(model_groups.get("musicgen") or [])
    if stable_models:
        stable_request = request.copy(update={"model_ids": stable_models})
        jobs.append(
            {
                "family": "stable_audio",
                **_submit_stable_audio_audio_benchmark(stable_request, plan, skip_duplicate_checks=True),
            }
        )
    if musicgen_models:
        musicgen_request = request.copy(update={"model_ids": musicgen_models})
        jobs.append(
            {
                "family": "musicgen",
                **_submit_musicgen_audio_benchmark(musicgen_request, plan, skip_duplicate_checks=True),
            }
        )
    if not jobs:
        raise HTTPException(status_code=409, detail="No live generated-audio model families were selected.")
    group_name = datetime.now(timezone.utc).strftime(f"audio-all-{plan['scope']}-%Y%m%d-%H%M%S")
    group_event = {
        "action": "benchmark_testing_audio_group_submitted",
        "job_name": group_name,
        "studio_url": jobs[0].get("studio_url"),
        "compute": _TRAINING_H100_COMPUTE,
        "environment": "multi-environment",
        "job_file": "multi-family generated-audio run",
        "output_path": None,
        "prompt_manifest_uri": plan["prompt_manifest_uri"],
        "model_ids": request.model_ids,
        "suite_ids": request.suite_ids,
        "seed_ids": request.seed_ids,
        "max_prompts": int(plan["max_prompts"]),
        "scope": plan["scope"],
        "child_jobs": [
            {
                "family": job.get("family"),
                "job_name": job.get("name"),
                "studio_url": job.get("studio_url"),
                "output_path": job.get("output_path"),
            }
            for job in jobs
        ],
        "marketplace_resources": False,
    }
    _evaluation_append_job_event(group_event)
    return {
        "status": "submitted",
        "dry_run": False,
        "message": "Generated-audio benchmark Azure ML jobs submitted for the selected model families.",
        "plan": plan,
        "job": {
            "name": group_name,
            "status": "submitted",
            "studio_url": jobs[0].get("studio_url"),
            "output_path": None,
        },
        "jobs": jobs,
    }


@app.post("/api/evaluation/audio-benchmark/retry-missing")
def evaluation_audio_benchmark_retry_missing(request: EvaluationAudioBenchmarkRetryMissingRequest):
    if request.launch_confirmation != "RETRY MUSICGEN AUDIO ONLY":
        raise HTTPException(status_code=409, detail="Type RETRY MUSICGEN AUDIO ONLY to submit only the missing MusicGen generated-audio job.")
    source_job_name = str(request.source_job_name or "").strip()
    source_event = _evaluation_job_event_by_name(source_job_name) if source_job_name else None
    if not source_event:
        latest_full = _latest_generated_audio_job_state_fast(scope="full")
        source_event = _evaluation_job_event_by_name(str((latest_full or {}).get("job_name") or ""))
    if not source_event or str(source_event.get("action") or "") != "benchmark_testing_audio_group_submitted":
        raise HTTPException(status_code=409, detail="Retry requires an existing all-model generated-audio parent run.")
    progress = _generated_audio_progress_state(str(source_event.get("job_name") or ""))
    by_model = {str(row.get("model_id") or ""): row for row in progress.get("model_progress") or []}
    stable_complete = all(
        float((by_model.get(model_id) or {}).get("percent") or 0) >= 99.5
        for model_id in ("base_stable_audio_open_small", "diffusion_cara_strong_full_modest_arch")
    )
    musicgen_complete = all(
        float((by_model.get(model_id) or {}).get("percent") or 0) >= 99.5
        for model_id in ("base_musicgen_small", "musicgen_cara_strong_full")
    )
    if not stable_complete:
        raise HTTPException(status_code=409, detail="Stable Audio is not complete in the source all-model run; do not retry only MusicGen yet.")
    if musicgen_complete:
        raise HTTPException(status_code=409, detail="MusicGen is already complete in the source all-model run; retry is not needed.")
    child_jobs = [child for child in (source_event.get("child_jobs") or []) if isinstance(child, dict)]
    stable_child = next((child for child in child_jobs if child.get("family") == "stable_audio"), None)
    if not stable_child or not stable_child.get("output_path"):
        raise HTTPException(status_code=409, detail="Source all-model run does not contain a reusable Stable Audio child output path.")
    retry_request = EvaluationAudioBenchmarkRunRequest(
        model_ids=["base_musicgen_small", "musicgen_cara_strong_full"],
        suite_ids=list(source_event.get("suite_ids") or LIVE_WAVE_1_SUITE_IDS) if "LIVE_WAVE_1_SUITE_IDS" in globals() else list(source_event.get("suite_ids") or ["heldout_audio_attribution", "known_pool_prompt_recall", "control_token_confound", "baseline_negative_control"]),
        scope=str(source_event.get("scope") or "full"),
        seed_ids=[int(seed) for seed in (source_event.get("seed_ids") or source_event.get("seeds") or [0])],
        max_prompts=int(source_event.get("max_prompts") or 0),
        dry_run=request.dry_run,
        launch_confirmation="LAUNCH ALL MODELS AUDIO BENCHMARK",
    )
    plan = _audio_benchmark_plan(retry_request)
    if request.dry_run:
        return {
            "status": "planned",
            "dry_run": True,
            "message": "Dry run only. Stable Audio will be reused; only MusicGen would be submitted.",
            "source_job_name": source_event.get("job_name"),
            "reused_child_jobs": [stable_child],
            "plan": plan,
        }
    active_audio_job = _active_generated_audio_job_state()
    if active_audio_job:
        raise HTTPException(
            status_code=409,
            detail=f"Generated-audio benchmark is already active in Azure ML ({active_audio_job.get('job_name')}, status={active_audio_job.get('status')}).",
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); MusicGen retry will not fall back to CPU.")
    musicgen_job = {
        "family": "musicgen",
        **_submit_musicgen_audio_benchmark(retry_request, plan, skip_duplicate_checks=True),
    }
    group_name = datetime.now(timezone.utc).strftime("audio-all-full-retry-musicgen-%Y%m%d-%H%M%S")
    group_event = {
        "action": "benchmark_testing_audio_group_submitted",
        "job_name": group_name,
        "studio_url": musicgen_job.get("studio_url") or stable_child.get("studio_url"),
        "compute": _TRAINING_H100_COMPUTE,
        "environment": "multi-environment",
        "job_file": "multi-family generated-audio retry; Stable Audio reused, MusicGen resubmitted",
        "output_path": None,
        "prompt_manifest_uri": source_event.get("prompt_manifest_uri") or plan.get("prompt_manifest_uri"),
        "model_ids": list(source_event.get("model_ids") or ["base_stable_audio_open_small", "diffusion_cara_strong_full_modest_arch", "base_musicgen_small", "musicgen_cara_strong_full"]),
        "suite_ids": list(source_event.get("suite_ids") or retry_request.suite_ids),
        "seed_ids": list(source_event.get("seed_ids") or source_event.get("seeds") or retry_request.seed_ids),
        "max_prompts": int(source_event.get("max_prompts") or 0),
        "scope": str(source_event.get("scope") or "full"),
        "retry_of": source_event.get("job_name"),
        "retry_policy": "reuse_complete_stable_audio_child_and_resubmit_musicgen_only",
        "child_jobs": [
            stable_child,
            {
                "family": "musicgen",
                "job_name": musicgen_job.get("name"),
                "studio_url": musicgen_job.get("studio_url"),
                "output_path": musicgen_job.get("output_path"),
            },
        ],
        "marketplace_resources": False,
    }
    _evaluation_append_job_event(group_event)
    return {
        "status": "submitted",
        "dry_run": False,
        "message": "MusicGen-only generated-audio retry submitted. Stable Audio outputs are reused from the source run.",
        "source_job_name": source_event.get("job_name"),
        "reused_child_jobs": [stable_child],
        "plan": plan,
        "job": {"name": group_name, "status": "submitted", "studio_url": musicgen_job.get("studio_url"), "output_path": None},
        "jobs": [musicgen_job],
    }


def _submit_stable_audio_attribution_scoring(
    request: EvaluationAttributionScoringRunRequest,
    plan: dict[str, Any],
    *,
    skip_duplicate_checks: bool = False,
) -> dict[str, Any]:
    if request.launch_confirmation != "LAUNCH ATTRIBUTION SCORING":
        raise HTTPException(status_code=409, detail="Type LAUNCH ATTRIBUTION SCORING to submit a live attribution scoring job.")
    if not _EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.exists():
        raise HTTPException(status_code=409, detail=f"Attribution scoring job file is missing: {_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE}")
    if not plan.get("generated_audio_output_path"):
        raise HTTPException(status_code=409, detail="Complete a generated-audio benchmark before attribution scoring.")
    active_score_job = _active_attribution_score_job_state()
    if active_score_job and not skip_duplicate_checks:
        raise HTTPException(
            status_code=409,
            detail=f"Attribution scoring is already active in Azure ML ({active_score_job.get('job_name')}, status={active_score_job.get('status')}).",
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs and not skip_duplicate_checks:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); attribution scoring will not fall back to CPU.")
    try:
        from azure.ai.ml import load_job

        run_slug = datetime.now(timezone.utc).strftime("stable-audio-score-%Y%m%d-%H%M%S")
        output_path = f"{_EVALUATION_STABLE_AUDIO_OUTPUT_URI}scoring/{run_slug}/"
        generated_audio_output_path = str((plan.get("generated_audio_output_paths") or {}).get("stable_audio") or plan["generated_audio_output_path"])
        stable_model_ids = [
            str(model_id)
            for model_id in plan.get("model_ids", [])
            if _EVALUATION_SCORE_MODEL_FAMILY_BY_ID.get(str(model_id)) == "stable_audio"
        ]
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _evaluation_materialize_stable_audio_score_job_file(
            output_path=output_path,
            generated_audio_output_path=generated_audio_output_path,
            model_ids=stable_model_ids,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        serialized_dry_run = str(_azureml_input_scalar(submitted_job, "dry_run") or "").lower()
        serialized_model_ids = str(_azureml_input_scalar(submitted_job, "model_ids") or "")
        expected_model_ids = ",".join(stable_model_ids)
        if serialized_dashboard_triggered != "true" or serialized_dry_run != "false" or serialized_model_ids != expected_model_ids:
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized attribution-scoring inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    f"expected dashboard_triggered=true, dry_run=false; got dashboard_triggered={serialized_dashboard_triggered or 'missing'}, "
                    f"dry_run={serialized_dry_run or 'missing'}, model_ids={serialized_model_ids or 'missing'}."
                ),
            )
        event = {
            "action": "benchmark_testing_stable_audio_score_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "job_file": str(_EVALUATION_STABLE_AUDIO_SCORE_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "source_audio_job_name": plan.get("audio_job_name"),
            "generated_audio_output_path": generated_audio_output_path,
            "generation_manifest_uri": f"{generated_audio_output_path.rstrip('/')}/generation_manifest.jsonl",
            "model_ids": stable_model_ids,
            "trained_model_data": _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI,
            "context_trained_model_data": _training_latest_context_full_output_path(),
            "native_extractor": True,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
            "marketplace_resources": False,
        }
        _azureml_test_prep_audit(event)
        _evaluation_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "source_audio_job_name": plan.get("audio_job_name"),
            "generated_audio_output_path": generated_audio_output_path,
            "model_ids": stable_model_ids,
            "context_trained_model_data": _training_latest_context_full_output_path(),
            "metrics_uri": f"{output_path.rstrip('/')}/metrics_latest.json",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


def _submit_musicgen_attribution_scoring(
    request: EvaluationAttributionScoringRunRequest,
    plan: dict[str, Any],
    *,
    skip_duplicate_checks: bool = False,
) -> dict[str, Any]:
    if request.launch_confirmation != "LAUNCH ATTRIBUTION SCORING":
        raise HTTPException(status_code=409, detail="Type LAUNCH ATTRIBUTION SCORING to submit a live attribution scoring job.")
    if not _EVALUATION_MUSICGEN_SCORE_JOB_FILE.exists():
        raise HTTPException(status_code=409, detail=f"MusicGen attribution scoring job file is missing: {_EVALUATION_MUSICGEN_SCORE_JOB_FILE}")
    paths = plan.get("generated_audio_output_paths") if isinstance(plan.get("generated_audio_output_paths"), dict) else {}
    generated_audio_output_path = str(paths.get("musicgen") or "")
    if not generated_audio_output_path:
        raise HTTPException(status_code=409, detail="Complete a MusicGen generated-audio benchmark before MusicGen attribution scoring.")
    active_score_job = _active_attribution_score_job_state()
    if active_score_job and not skip_duplicate_checks:
        raise HTTPException(
            status_code=409,
            detail=f"Attribution scoring is already active in Azure ML ({active_score_job.get('job_name')}, status={active_score_job.get('status')}).",
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs and not skip_duplicate_checks:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); attribution scoring will not fall back to CPU.")
    from evaluation.benchmark_spec import model_lanes

    lanes_by_id = {lane["model_id"]: lane for lane in model_lanes()}
    trained_model_uri = (lanes_by_id.get("musicgen_cara_strong_full") or {}).get("output_uri")
    if not trained_model_uri:
        raise HTTPException(status_code=409, detail="MusicGen CARA-Strong full-run output is missing; cannot run MusicGen native suffix scoring.")
    try:
        from azure.ai.ml import load_job

        run_slug = datetime.now(timezone.utc).strftime("musicgen-score-%Y%m%d-%H%M%S")
        output_path = f"{_EVALUATION_MUSICGEN_OUTPUT_URI}scoring/{run_slug}/"
        hf_secret = _training_sync_hf_token_secret()
        musicgen_model_ids = [
            str(model_id)
            for model_id in plan.get("model_ids", [])
            if _EVALUATION_SCORE_MODEL_FAMILY_BY_ID.get(str(model_id)) == "musicgen"
        ]
        materialized_job_file = _evaluation_materialize_musicgen_score_job_file(
            output_path=output_path,
            generated_audio_output_path=generated_audio_output_path,
            trained_model_data=str(trained_model_uri),
            model_ids=musicgen_model_ids,
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        serialized_dry_run = str(_azureml_input_scalar(submitted_job, "dry_run") or "").lower()
        serialized_model_ids = str(_azureml_input_scalar(submitted_job, "model_ids") or "")
        expected_model_ids = ",".join(musicgen_model_ids)
        if serialized_dashboard_triggered != "true" or serialized_dry_run != "false" or serialized_model_ids != expected_model_ids:
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized MusicGen attribution-scoring inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. expected dashboard_triggered=true, dry_run=false; "
                    f"got dashboard_triggered={serialized_dashboard_triggered or 'missing'}, dry_run={serialized_dry_run or 'missing'}, "
                    f"model_ids={serialized_model_ids or 'missing'}."
                ),
            )
        event = {
            "action": "benchmark_testing_musicgen_score_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "job_file": str(_EVALUATION_MUSICGEN_SCORE_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "source_audio_job_name": plan.get("audio_job_name"),
            "generated_audio_output_path": generated_audio_output_path,
            "generation_manifest_uri": f"{generated_audio_output_path.rstrip('/')}/generation_manifest.jsonl",
            "model_ids": musicgen_model_ids,
            "trained_model_data": trained_model_uri,
            "native_extractor": True,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
            "marketplace_resources": False,
        }
        _azureml_test_prep_audit(event)
        _evaluation_append_job_event(event)
        return {
            **summary,
            "output_path": output_path,
            "source_audio_job_name": plan.get("audio_job_name"),
            "generated_audio_output_path": generated_audio_output_path,
            "model_ids": musicgen_model_ids,
            "metrics_uri": f"{output_path.rstrip('/')}/metrics_latest.json",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.post("/api/evaluation/attribution-scoring/run")
def evaluation_attribution_scoring_run(request: EvaluationAttributionScoringRunRequest):
    plan = _attribution_scoring_plan(request)
    if request.dry_run:
        return {
            "status": "planned",
            "dry_run": True,
            "message": "Dry run only. No Azure ML attribution scoring job was submitted.",
            "plan": plan,
        }
    if not plan.get("live_ready"):
        raise HTTPException(status_code=409, detail=str(plan.get("live_ready_reason") or "Attribution scoring is not ready."))
    paths = plan.get("pending_score_output_paths") if isinstance(plan.get("pending_score_output_paths"), dict) else {}
    if not paths:
        paths = plan.get("generated_audio_output_paths") if isinstance(plan.get("generated_audio_output_paths"), dict) else {}
    jobs: list[dict[str, Any]] = []
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); attribution scoring will not fall back to CPU.")
    if paths.get("stable_audio"):
        jobs.append({"family": "stable_audio", **_submit_stable_audio_attribution_scoring(request, plan, skip_duplicate_checks=True)})
    if paths.get("musicgen"):
        jobs.append({"family": "musicgen", **_submit_musicgen_attribution_scoring(request, plan, skip_duplicate_checks=True)})
    if not jobs and plan.get("generated_audio_output_path"):
        jobs.append({"family": "stable_audio", **_submit_stable_audio_attribution_scoring(request, plan, skip_duplicate_checks=True)})
    if not jobs:
        raise HTTPException(status_code=409, detail="No generated-audio output folders were available for attribution scoring.")
    group_name = datetime.now(timezone.utc).strftime("score-all-%Y%m%d-%H%M%S")
    group_event = {
        "action": "benchmark_testing_attribution_score_group_submitted",
        "job_name": group_name,
        "studio_url": jobs[0].get("studio_url"),
        "compute": _TRAINING_H100_COMPUTE,
        "environment": "multi-environment",
        "job_file": "multi-family attribution scoring run",
        "output_path": None,
        "source_audio_job_name": plan.get("audio_job_name"),
        "child_jobs": [
            {
                "family": job.get("family"),
                "job_name": job.get("name"),
                "studio_url": job.get("studio_url"),
                "output_path": job.get("output_path"),
                "metrics_uri": job.get("metrics_uri"),
                "model_ids": job.get("model_ids") or [],
            }
            for job in jobs
        ],
        "model_ids": plan.get("model_ids") or [],
        "marketplace_resources": False,
    }
    _evaluation_append_job_event(group_event)
    return {
        "status": "submitted",
        "dry_run": False,
        "message": "Attribution scoring Azure ML jobs submitted for selected model lanes.",
        "plan": plan,
        "job": {"name": group_name, "status": "submitted", "studio_url": jobs[0].get("studio_url")},
        "jobs": jobs,
    }


@app.post("/api/evaluation/prompt-set/lock")
def evaluation_prompt_set_lock(request: EvaluationPromptSetLockRequest):
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="Confirm the benchmark prompt set lock before continuing.")
    if _EVALUATION_PROMPT_SET_LOCK.exists():
        latest_job = _latest_evaluation_job_state()
        return {
            "status": "already_locked",
            "benchmark_prompt_set": _evaluation_prompt_set_state(latest_job),
            "readiness": evaluation_readiness(),
        }
    event = _evaluation_job_event_by_name(request.job_name)
    if not event:
        raise HTTPException(status_code=409, detail="No benchmark setup job event is available to lock.")
    job_name = str(event.get("job_name") or "")
    if not job_name:
        raise HTTPException(status_code=409, detail="Benchmark setup event is missing an Azure job name.")
    try:
        summary = _azureml_job_summary(_azureml_client().jobs.get(job_name))
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc
    if str(summary.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail=f"Benchmark setup job {job_name} is not completed yet; status={summary.get('status') or 'unknown'}.")
    prompt_manifest_uri = _evaluation_prompt_manifest_uri(event.get("output_path"))
    if not prompt_manifest_uri:
        raise HTTPException(status_code=409, detail="Benchmark setup event does not have an output path to lock.")
    lock_payload = {
        "format": "cara_benchmark_prompt_set_v1",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "dashboard",
        "source_job_name": job_name,
        "source_job_status": summary.get("status"),
        "source_job_display_name": summary.get("display_name"),
        "source_output_path": event.get("output_path"),
        "prompt_manifest_uri": prompt_manifest_uri,
        "control_metrics_uri": f"{str(event.get('output_path')).rstrip('/')}/control_metrics.json",
        "evaluation_plan_uri": f"{str(event.get('output_path')).rstrip('/')}/evaluation_plan.json",
        "model_ids": event.get("model_ids") or [],
        "suite_ids": event.get("suite_ids") or [],
        "seeds": event.get("seeds"),
        "prepared_data": event.get("prepared_data"),
        "trained_model_data": event.get("trained_model_data"),
        "cost_policy": "Use existing Azure ML workspace resources only; no Marketplace endpoints or deployments.",
        "reuse_policy": "All later Diffusion, AR, and Hybrid benchmark scoring jobs must use this prompt_manifest_uri for like-for-like comparison.",
    }
    _EVALUATION_PROMPT_SET_LOCK.parent.mkdir(parents=True, exist_ok=True)
    _EVALUATION_PROMPT_SET_LOCK.write_text(json.dumps(lock_payload, indent=2, sort_keys=True), encoding="utf-8")
    _azureml_test_prep_audit(
        {
            "action": "benchmark_prompt_set_locked",
            "job_name": job_name,
            "prompt_manifest_uri": prompt_manifest_uri,
            "lock_path": str(_EVALUATION_PROMPT_SET_LOCK.relative_to(ROOT)),
            "marketplace_resources": False,
        }
    )
    latest_job = _latest_evaluation_job_state()
    return {
        "status": "locked",
        "benchmark_prompt_set": _evaluation_prompt_set_state(latest_job),
        "readiness": evaluation_readiness(),
    }


@app.get("/api/evaluation/benchmarks")
def evaluation_benchmarks():
    from evaluation.model_registry import baseline_comparison_policy, benchmark_rows, latest_results_summary
    from evaluation.openai_benchmark_summary import latest_summary_payload
    from evaluation.benchmark_spec import benchmark_spec, comparison_cards, default_metric_rows, model_lanes

    latest_job = _latest_evaluation_job_state_fast()
    latest_audio_result = _latest_generated_audio_job_state_fast()
    latest_smoke_result = _latest_generated_audio_job_state_fast(scope="smoke")
    latest_full_result = _latest_generated_audio_job_state_fast(scope="full")
    latest_score_result = (
        _latest_attribution_score_job_state_fast(source_audio_job_name=str(latest_full_result.get("job_name")))
        if latest_full_result and latest_full_result.get("job_name")
        else _latest_attribution_score_job_state_fast()
    )
    latest_results = latest_results_summary()
    lanes = model_lanes()
    rows = benchmark_rows()
    metric_rows = default_metric_rows(lanes)
    audio_complete = bool(latest_full_result or latest_audio_result)
    if latest_audio_result:
        latest_results = {
            **latest_results,
            "audio_available": True,
            "audio_complete": audio_complete,
            "audio_progress_percent": None,
            "metric_stage": latest_audio_result.get("stage_label"),
            "audio_output_path": latest_audio_result.get("output_path"),
            "generation_manifest_uri": f"{str(latest_audio_result.get('output_path') or '').rstrip('/')}/generation_manifest.jsonl",
            "metrics_path": latest_results.get("metrics_path"),
            "native_attribution_status": "pending_attribution_extractor",
            "external_probe_status": "pending_external_probe",
        }
    score_metrics = _benchmark_score_metrics(latest_score_result)
    if isinstance(score_metrics, dict):
        rows = _with_benchmark_lane_statuses(score_metrics.get("benchmark_rows") or rows, score_metrics)
        metric_rows = default_metric_rows(lanes, score_metrics)
        latest_results = {
            **latest_results,
            "metrics_available": bool(score_metrics),
            "metric_stage": "attribution_scoring_completed",
            "metrics_path": (latest_score_result or {}).get("metrics_uri") or latest_results.get("metrics_path"),
            "latest_metrics": score_metrics,
            "native_attribution_status": (
                (score_metrics.get("lanes") or {}).get("diffusion_native", {}).get("status")
                or (score_metrics.get("lanes") or {}).get("musicgen_native", {}).get("status")
            ),
            "external_probe_status": (
                (score_metrics.get("lanes") or {}).get("base_external_probe", {}).get("status")
                or (score_metrics.get("lanes") or {}).get("base_musicgen_external_probe", {}).get("status")
            ),
        }
    return {
        "format": "cara_benchmark_testing_benchmarks_v2",
        "benchmark_spec": benchmark_spec(_evaluation_suites()),
        "model_lanes": lanes,
        "rows": rows,
        "metric_rows": metric_rows,
        "comparison_cards": comparison_cards(metric_rows),
        "repairability_matrix": (score_metrics or {}).get("repairability_matrix") if isinstance(score_metrics, dict) else None,
        "repair_method_matrix": (score_metrics or {}).get("repair_method_matrix") if isinstance(score_metrics, dict) else None,
        "prediction_examples": (score_metrics or {}).get("prediction_examples") if isinstance(score_metrics, dict) else {},
        "latest_results": latest_results,
        "baseline_comparison_policy": baseline_comparison_policy(),
        "openai_summary": latest_summary_payload(),
        "latest_evaluation_job": latest_job,
        "latest_generated_audio_result": latest_audio_result,
        "latest_generated_audio_smoke_result": latest_smoke_result,
        "latest_generated_audio_full_result": latest_full_result,
        "active_generated_audio_job": None,
        "latest_attribution_scoring_result": latest_score_result,
        "active_attribution_scoring_job": None,
        "benchmark_prompt_set": _evaluation_prompt_set_state(latest_job),
        "notes": [
            "Generated-audio completion means the selected model lanes produced saved WAVs and manifests; attribution accuracy remains pending until the native/probe scorer runs.",
            "Benchmark Prompt Set v2 defines tag-present, tag-withheld, no-tag, shuffled-label, held-out-audio, and open-quality conditions.",
            "All later model scoring runs must reuse the locked benchmark prompt set for like-for-like comparison.",
            "Base native CARA-id metrics are N/A because the original checkpoints have no native CARA output channel.",
            "Use external-probe and retrieval metrics to compare base, Diffusion, MusicGen, and future model lanes on the same generated-audio evidence lane.",
        ],
    }


@app.get("/api/evaluation/benchmark-summary")
def evaluation_benchmark_summary():
    from evaluation.openai_benchmark_summary import latest_summary_payload

    return latest_summary_payload()


@app.post("/api/evaluation/benchmark-summary")
def evaluation_benchmark_summary_refresh():
    from evaluation.cara_repairability import repairability_schema
    from evaluation.model_registry import (
        baseline_comparison_policy,
        benchmark_rows,
        evaluation_readiness_payload,
        latest_results_summary,
    )
    from evaluation.openai_benchmark_summary import generate_benchmark_summary
    from evaluation.benchmark_spec import benchmark_spec, comparison_cards, default_metric_rows, model_lanes

    latest_job = _latest_evaluation_job_state_fast()
    latest_audio_result = _latest_generated_audio_job_state_fast()
    latest_smoke_result = _latest_generated_audio_job_state_fast(scope="smoke")
    latest_full_result = _latest_generated_audio_job_state_fast(scope="full")
    latest_score_result = (
        _latest_attribution_score_job_state_fast(source_audio_job_name=str(latest_full_result.get("job_name")))
        if latest_full_result and latest_full_result.get("job_name")
        else _latest_attribution_score_job_state_fast()
    )
    latest_results = latest_results_summary()
    lanes = model_lanes()
    rows = benchmark_rows()
    metric_rows = default_metric_rows(lanes)
    audio_complete = bool(latest_full_result or latest_audio_result)
    if latest_audio_result:
        latest_results = {
            **latest_results,
            "audio_available": True,
            "audio_complete": audio_complete,
            "audio_progress_percent": None,
            "metric_stage": latest_audio_result.get("stage_label"),
            "audio_output_path": latest_audio_result.get("output_path"),
            "generation_manifest_uri": f"{str(latest_audio_result.get('output_path') or '').rstrip('/')}/generation_manifest.jsonl",
            "metrics_path": latest_results.get("metrics_path"),
            "native_attribution_status": "pending_attribution_extractor",
            "external_probe_status": "pending_external_probe",
        }
    score_metrics = _benchmark_score_metrics(latest_score_result)
    if isinstance(score_metrics, dict):
        rows = _with_benchmark_lane_statuses(score_metrics.get("benchmark_rows") or rows, score_metrics)
        metric_rows = default_metric_rows(lanes, score_metrics)
        latest_results = {
            **latest_results,
            "metrics_available": bool(score_metrics),
            "metric_stage": "attribution_scoring_completed",
            "metrics_path": (latest_score_result or {}).get("metrics_uri") or latest_results.get("metrics_path"),
            "latest_metrics": score_metrics,
            "native_attribution_status": (
                (score_metrics.get("lanes") or {}).get("diffusion_native", {}).get("status")
                or (score_metrics.get("lanes") or {}).get("musicgen_native", {}).get("status")
            ),
            "external_probe_status": (
                (score_metrics.get("lanes") or {}).get("base_external_probe", {}).get("status")
                or (score_metrics.get("lanes") or {}).get("base_musicgen_external_probe", {}).get("status")
            ),
        }
    benchmark_payload = {
        "format": "cara_benchmark_testing_benchmarks_v2",
        "benchmark_spec": benchmark_spec(_evaluation_suites()),
        "model_lanes": lanes,
        "rows": rows,
        "metric_rows": metric_rows,
        "comparison_cards": comparison_cards(metric_rows),
        "repairability_matrix": (score_metrics or {}).get("repairability_matrix") if isinstance(score_metrics, dict) else None,
        "repair_method_matrix": (score_metrics or {}).get("repair_method_matrix") if isinstance(score_metrics, dict) else None,
        "prediction_examples": (score_metrics or {}).get("prediction_examples") if isinstance(score_metrics, dict) else {},
        "latest_results": latest_results,
        "baseline_comparison_policy": baseline_comparison_policy(),
        "latest_evaluation_job": latest_job,
        "latest_generated_audio_result": latest_audio_result,
        "latest_generated_audio_smoke_result": latest_smoke_result,
        "latest_generated_audio_full_result": latest_full_result,
        "active_generated_audio_job": None,
        "latest_attribution_scoring_result": latest_score_result,
        "active_attribution_scoring_job": None,
        "benchmark_prompt_set": _evaluation_prompt_set_state(latest_job),
        "notes": [
            "Generated-audio completion means the selected model lanes produced saved WAVs and manifests; attribution accuracy remains pending until the native/probe scorer runs.",
            "A completed benchmark setup job only means manifests, controls, and artifact checks passed; generated-audio scoring is the next stage.",
            "All later model scoring runs must reuse the locked benchmark prompt set for like-for-like comparison.",
            "Base native CARA-id metrics are N/A because the original checkpoints have no native CARA output channel.",
            "Use external-probe metrics to compare base, Diffusion, AR, and Hybrid on the same generated-audio evidence lane.",
        ],
    }
    readiness_payload = evaluation_readiness_payload(_evaluation_suites())
    readiness_payload["repairability"] = repairability_schema()
    readiness_payload["benchmark_prompt_set"] = _evaluation_prompt_set_state()
    try:
        return generate_benchmark_summary(benchmark_payload, readiness_payload)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _submit_benchmark_testing_stable_audio_evaluation(request: EvaluationRunRequest, known_models: set[str], known_suites: set[str]) -> dict[str, Any]:
    model_ids = list(dict.fromkeys(request.model_ids))
    suite_ids = list(dict.fromkeys(request.suite_ids))
    if _EVALUATION_PROMPT_SET_LOCK.exists():
        lock = _training_read_json(_EVALUATION_PROMPT_SET_LOCK, {})
        raise HTTPException(
            status_code=409,
            detail=(
                "Benchmark Prompt Set v1 is already locked. Do not launch another setup prompt-sampling run; "
                f"use prompt_manifest_uri={lock.get('prompt_manifest_uri') or 'locked prompt manifest'} for model scoring."
            ),
        )
    supported_models = {"base_stable_audio_open_small", "diffusion_cara_strong_full_modest_arch"}
    unsupported_models = [model_id for model_id in model_ids if model_id not in supported_models]
    if unsupported_models:
        raise HTTPException(
            status_code=409,
            detail=(
                "Live wave-1 evaluation is wired for the base Stable Audio model and the completed Diffusion "
                f"CARA-Strong run only. Pending models cannot launch yet: {unsupported_models}"
            ),
        )
    if not supported_models.issubset(set(model_ids)):
        raise HTTPException(
            status_code=409,
            detail="Live wave-1 evaluation requires both base_stable_audio_open_small and diffusion_cara_strong_full_modest_arch.",
        )
    if not suite_ids:
        raise HTTPException(status_code=400, detail="Select at least one evaluation suite.")
    if request.custom_prompt and request.custom_prompt.strip():
        raise HTTPException(status_code=409, detail="Live wave-1 evaluation uses fixed benchmark testing suites only; clear the custom prompt before launch.")
    unknown_suites = [suite_id for suite_id in suite_ids if suite_id not in known_suites]
    if unknown_suites:
        raise HTTPException(status_code=400, detail=f"Unknown evaluation suites: {unknown_suites}")
    if int(request.seeds) < 1 or int(request.seeds) > 3:
        raise HTTPException(status_code=400, detail="Live wave-1 evaluation seeds must be between 1 and 3.")
    if str(request.launch_confirmation or "").strip() != "LAUNCH BENCHMARK TESTING EVALUATION":
        raise HTTPException(
            status_code=409,
            detail="Type LAUNCH BENCHMARK TESTING EVALUATION to submit the GPU benchmark testing job.",
        )
    active_h100_jobs = _azureml_active_jobs_on_h100_computes()
    if active_h100_jobs:
        names = ", ".join(str(job.get("name") or "unknown") for job in active_h100_jobs[:3])
        raise HTTPException(status_code=409, detail=f"H100-backed compute is busy ({names}); evaluation will not fall back to CPU.")

    try:
        from azure.ai.ml import load_job

        if not _EVALUATION_STABLE_AUDIO_JOB_FILE.exists():
            raise FileNotFoundError(f"Stable Audio evaluation job file not found: {_EVALUATION_STABLE_AUDIO_JOB_FILE}")
        run_slug = datetime.now(timezone.utc).strftime("stable-audio-wave1-%Y%m%d-%H%M%S")
        output_path = f"{_EVALUATION_STABLE_AUDIO_OUTPUT_URI}{run_slug}/"
        hf_secret = _training_sync_hf_token_secret()
        materialized_job_file = _evaluation_materialize_stable_audio_job_file(
            output_path=output_path,
            model_ids=model_ids,
            suite_ids=suite_ids,
            seeds=int(request.seeds),
        )
        try:
            job = load_job(source=materialized_job_file)
        finally:
            try:
                materialized_job_file.unlink()
            except OSError:
                pass
        job.environment_variables = {
            **dict(getattr(job, "environment_variables", None) or {}),
            "KEY_VAULT_URL": str(hf_secret["vault_url"]),
            "HF_TOKEN_SECRET_NAME": str(hf_secret["secret_name"]),
        }
        submitted = _azureml_client().jobs.create_or_update(job)
        summary = _azureml_job_summary(submitted)
        submitted_job = _azureml_client().jobs.get(str(summary.get("name")))
        serialized_models = str(_azureml_input_scalar(submitted_job, "model_ids") or "")
        serialized_suites = str(_azureml_input_scalar(submitted_job, "suite_ids") or "")
        serialized_seeds = str(_azureml_input_scalar(submitted_job, "seeds") or "")
        serialized_dashboard_triggered = str(_azureml_input_scalar(submitted_job, "dashboard_triggered") or "").lower()
        if (
            serialized_models != ",".join(model_ids)
            or serialized_suites != ",".join(suite_ids)
            or serialized_seeds != str(int(request.seeds))
            or serialized_dashboard_triggered != "true"
        ):
            try:
                _azureml_client().jobs.begin_cancel(str(summary.get("name")))
            except Exception:
                pass
            raise HTTPException(
                status_code=409,
                detail=(
                    "Azure ML serialized evaluation inputs do not match the dashboard request; "
                    f"cancel requested for {summary.get('name')}. "
                    f"expected model_ids={','.join(model_ids)}, suite_ids={','.join(suite_ids)}, seeds={int(request.seeds)}, dashboard_triggered=true; "
                    f"got model_ids={serialized_models or 'missing'}, suite_ids={serialized_suites or 'missing'}, seeds={serialized_seeds or 'missing'}, "
                    f"dashboard_triggered={serialized_dashboard_triggered or 'missing'}."
                ),
            )
        event = {
            "action": "benchmark_testing_stable_audio_evaluation_submitted",
            "job_name": summary.get("name"),
            "studio_url": summary.get("studio_url"),
            "compute": _TRAINING_H100_COMPUTE,
            "environment": summary.get("environment"),
            "job_file": str(_EVALUATION_STABLE_AUDIO_JOB_FILE.relative_to(ROOT)),
            "output_path": output_path,
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seeds": int(request.seeds),
            "trained_model_data": _EVALUATION_STABLE_AUDIO_TRAINED_MODEL_URI,
            "prepared_data": _TRAINING_PREP_OUTPUT_URI,
            "hf_auth": "workspace_key_vault",
            "hf_secret_name": str(hf_secret["secret_name"]),
            "marketplace_resources": False,
        }
        _azureml_test_prep_audit(event)
        _evaluation_append_job_event(event)
        return {**summary, "output_path": output_path, "model_ids": model_ids, "suite_ids": suite_ids, "seeds": int(request.seeds)}
    except HTTPException:
        raise
    except Exception as exc:
        raise _azureml_operation_error(exc) from exc


@app.post("/api/evaluation/run")
def evaluation_run(request: EvaluationRunRequest):
    from evaluation.model_registry import evaluation_readiness_payload

    suites = _evaluation_suites()
    readiness = evaluation_readiness_payload(suites)
    known_models = {model["id"] for model in readiness["models"]}
    known_suites = {suite["id"] for suite in suites}
    unknown_models = [model_id for model_id in request.model_ids if model_id not in known_models]
    unknown_suites = [suite_id for suite_id in request.suite_ids if suite_id not in known_suites]
    if unknown_models or unknown_suites:
        raise HTTPException(
            status_code=400,
            detail={"unknown_models": unknown_models, "unknown_suites": unknown_suites},
        )

    plan = {
        "models": [model for model in readiness["models"] if model["id"] in request.model_ids],
        "suites": [suite for suite in suites if suite["id"] in request.suite_ids],
        "custom_prompt": request.custom_prompt.strip() if request.custom_prompt else None,
        "seeds": request.seeds,
        "benchmark_prompt_set": _evaluation_prompt_set_state(),
        "native_output_lane": "Models with native_cara_output=true report native exact/repair/family/unattributable metrics.",
        "external_probe_lane": "All selected models, including base checkpoints, are eligible for external-probe attribution metrics.",
        "live_wave_1_scope": (
            "The first live Azure job is GPU-only and creates the benchmark testing prompt manifest, validates the completed "
            "Stable Audio CARA-Strong artifacts, loads the base Stable Audio checkpoint on CUDA, and writes baseline-control "
            "metrics. Generated-audio native/probe scoring is a follow-on stage using the saved prompt manifest."
        ),
        "audio_output_policy": (
            "Live evaluation jobs should save generated audio by model_id/suite_id/prompt_id/seed under the evaluation "
            "output folder, with manifest rows linking audio files to prompt text, seed, expected CARA labels, and "
            "native/probe attribution outputs."
        ),
        "baseline_policy": readiness["baseline_comparison_policy"],
        "cost_policy": readiness["launch_guard"]["cost_policy"],
    }
    if request.dry_run:
        return {
            "status": "planned",
            "dry_run": True,
            "message": "Dry run only. No Azure ML evaluation job was submitted.",
            "plan": plan,
        }
    if request.launch_confirmation != "LAUNCH BENCHMARK TESTING EVALUATION":
        raise HTTPException(
            status_code=409,
            detail="Type LAUNCH BENCHMARK TESTING EVALUATION to submit a live benchmark testing job.",
        )
    submitted = _submit_benchmark_testing_stable_audio_evaluation(request, known_models, known_suites)
    return {
        "status": "submitted",
        "dry_run": False,
        "message": "GPU benchmark testing job submitted to Azure ML.",
        "job": submitted,
        "plan": plan,
    }


@app.get("/api/evaluation/comparison")
def evaluation_comparison():
    log_path = ROOT / "evaluation_log.csv"
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))[:50]
