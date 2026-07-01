from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from smoke_stable_audio_trainer import _configure_disk_safe_runtime_dirs, _prepare_hf_auth
from test_prep_common import base_metadata, parse_bool, write_report


DEFAULT_DIFFUSION_MODEL_ID = "diffusion_cara_strong_full_modest_arch"
BASE_MODEL_ID = "base_stable_audio_open_small"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _pool_codeword(pool_id: Any) -> str | None:
    parts = str(pool_id or "").split(":")
    if len(parts) >= 5 and parts[0] == "CARA":
        return parts[3]
    return str(pool_id or "").strip() or None


def _suite_rows(rows: list[dict[str, Any]], suite_id: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    heldout = [row for row in rows if row.get("split") in {"validation", "test"} and row.get("cara_pool_id")]
    if not heldout:
        heldout = [row for row in rows if row.get("cara_pool_id")]
    if suite_id == "heldout_audio_attribution":
        count = min(120, len(heldout))
    candidates = heldout.copy()
    rng.shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def _make_prompt_manifest(rows: list[dict[str, Any]], suite_ids: list[str], seeds: int) -> list[dict[str, Any]]:
    suite_counts = {
        "heldout_audio_attribution": 120,
        "known_pool_prompt_recall": 80,
        "control_token_confound": 40,
        "baseline_negative_control": 80,
        "open_generation_quality": 32,
        "adversarial_persistence": 60,
        "shuffled_label_control": 80,
    }
    rng = random.Random(20260607)
    manifest: list[dict[str, Any]] = []
    for suite_id in suite_ids:
        sampled = _suite_rows(rows, suite_id, suite_counts.get(suite_id, 40), rng)
        if suite_id == "shuffled_label_control":
            shuffled_labels = [row.get("cara_pool_id") for row in sampled]
            rng.shuffle(shuffled_labels)
        else:
            shuffled_labels = []
        for index, row in enumerate(sampled):
            base_prompt = str(row.get("prompt") or row.get("title") or "CARA evaluation audio").strip()
            if suite_id == "control_token_confound":
                prompt = base_prompt.replace(str(row.get("cara_pool_id") or ""), "").replace(str(_pool_codeword(row.get("cara_pool_id")) or ""), "")
            elif suite_id == "adversarial_persistence":
                prompt = f"{base_prompt}. Transform the style, instrumentation, and genre while retaining the source attribution evidence."
            elif suite_id == "open_generation_quality":
                family = row.get("cara_pool_family") or row.get("primary_genre") or "audio"
                prompt = f"A high-quality original {family} music loop with clear production detail."
            else:
                prompt = base_prompt
            for seed in range(max(1, int(seeds))):
                expected_pool_id = shuffled_labels[index] if suite_id == "shuffled_label_control" and shuffled_labels else row.get("cara_pool_id")
                manifest.append(
                    {
                        "prompt_id": f"{suite_id}-{index:04d}-seed{seed:02d}",
                        "suite_id": suite_id,
                        "seed": seed,
                        "prompt": prompt,
                        "source_chunk_id": row.get("chunk_id"),
                        "source_id": row.get("source_id"),
                        "source_split": row.get("split"),
                        "prepared_audio_path": row.get("prepared_audio_path"),
                        "expected_cara_pool_id": expected_pool_id,
                        "expected_cara_pool_index": row.get("cara_pool_index"),
                        "expected_cara_pool_family": row.get("cara_pool_family"),
                        "expected_cara_pool_family_index": row.get("cara_pool_family_index"),
                        "expected_cara_pool_codeword": _pool_codeword(expected_pool_id),
                        "evidence_type": "source_ground_truth_audio" if suite_id == "heldout_audio_attribution" else "targeted_or_control_prompt",
                    }
                )
    return manifest


def _control_metrics(prompt_manifest: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [row for row in prompt_manifest if row.get("expected_cara_pool_id")]
    train_pool_counts = Counter(str(row.get("cara_pool_id")) for row in train_rows if row.get("cara_pool_id"))
    train_family_counts = Counter(str(row.get("cara_pool_family")) for row in train_rows if row.get("cara_pool_family"))
    most_common_pool = train_pool_counts.most_common(1)[0][0] if train_pool_counts else None
    most_common_family = train_family_counts.most_common(1)[0][0] if train_family_counts else None
    prior_pool_correct = sum(1 for row in labelled if row.get("expected_cara_pool_id") == most_common_pool)
    prior_family_correct = sum(1 for row in labelled if row.get("expected_cara_pool_family") == most_common_family)
    denominator = max(1, len(labelled))
    rng = random.Random(20260607)
    pool_ids = sorted(train_pool_counts)
    random_pool_correct = 0
    if pool_ids:
        for row in labelled:
            if rng.choice(pool_ids) == row.get("expected_cara_pool_id"):
                random_pool_correct += 1
    return {
        "labelled_prompt_count": len(labelled),
        "train_pool_count": len(train_pool_counts),
        "train_family_count": len(train_family_counts),
        "prior_pool_id": most_common_pool,
        "prior_pool_exact_accuracy": prior_pool_correct / denominator,
        "prior_family": most_common_family,
        "prior_family_accuracy": prior_family_correct / denominator,
        "deterministic_random_pool_exact_accuracy": random_pool_correct / denominator,
        "note": "These are statistical controls only. Native/probe attribution metrics are filled by generation and attribution scoring stages.",
    }


def _check_trained_artifacts(trained_model_data: Path) -> dict[str, Any]:
    candidates = {
        "trainable_delta": trained_model_data / "checkpoints" / "trainable_delta.pt",
        "report": trained_model_data / "stable_audio_smoke_trainer_report.json",
        "metadata": trained_model_data / "metadata.json",
        "registry_resolver": trained_model_data / "cara_registry_resolver.json",
    }
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
        }
        for key, path in candidates.items()
    }


def _load_base_model(report: dict[str, Any], checkpoint: str) -> None:
    report["stage"] = "load_base_model"
    _prepare_hf_auth(report)
    from stable_audio_tools.models.pretrained import get_pretrained_model

    model, model_config = get_pretrained_model(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    report["base_model_load"] = {
        "checkpoint": checkpoint,
        "device": device,
        "sample_rate": model_config.get("sample_rate"),
        "sample_size": model_config.get("sample_size"),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["runtime_dirs"] = _configure_disk_safe_runtime_dirs(output_dir)
    prepared_data = Path(args.prepared_data)
    trained_model_data = Path(args.trained_model_data)
    manifest_path = prepared_data / args.manifest_relative_path
    rows = _read_jsonl(manifest_path)
    model_ids = _split_csv(args.model_ids)
    suite_ids = _split_csv(args.suite_ids)
    if not rows:
        raise RuntimeError(f"Prepared manifest is empty or missing: {manifest_path}")
    if BASE_MODEL_ID not in model_ids or DEFAULT_DIFFUSION_MODEL_ID not in model_ids:
        raise RuntimeError("Wave-1 live evaluation currently requires base_stable_audio_open_small and diffusion_cara_strong_full_modest_arch.")
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark testing live evaluation is GPU-only; CUDA is not available.")

    train_rows = [row for row in rows if row.get("split") == "train"]
    prompt_manifest_uri = str(args.prompt_manifest_uri or "").strip()
    prompt_manifest_source = "generated_from_prepared_manifest"
    if prompt_manifest_uri:
        prompt_manifest_path = Path(prompt_manifest_uri)
        if not prompt_manifest_path.exists():
            raise RuntimeError(
                "A locked prompt manifest was requested but is not mounted as a local file. "
                f"Mount the locked prompt_manifest_uri as an Azure ML uri_file input before scoring: {prompt_manifest_uri}"
            )
        prompt_manifest = _read_jsonl(prompt_manifest_path)
        prompt_manifest_source = "locked_prompt_manifest"
    else:
        prompt_manifest = _make_prompt_manifest(rows, suite_ids, int(args.seeds))
    artifact_checks = _check_trained_artifacts(trained_model_data)
    if not artifact_checks["trainable_delta"]["exists"]:
        raise RuntimeError("Expected trainable delta checkpoint is missing from the trained model output folder.")

    report.update(
        {
            "stage": "build_evaluation_manifests",
            "prepared_manifest_path": str(manifest_path),
            "prepared_rows": len(rows),
            "train_rows": len(train_rows),
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seeds": int(args.seeds),
            "prompt_count": len(prompt_manifest),
            "prompt_manifest_source": prompt_manifest_source,
            "prompt_manifest_uri": prompt_manifest_uri or None,
            "artifact_checks": artifact_checks,
            "native_generation_status": "pending_generation_scoring",
            "external_probe_status": "pending_generation_scoring",
            "audio_output_policy": "Generated audio must be written by model_id/suite_id/prompt_id/seed in a later scoring stage.",
        }
    )
    _write_jsonl(output_dir / "prompt_manifest.jsonl", prompt_manifest)
    _write_json(output_dir / "control_metrics.json", _control_metrics(prompt_manifest, train_rows))
    _write_json(
        output_dir / "evaluation_plan.json",
        {
            "created_at": _utc_now(),
            "model_ids": model_ids,
            "suite_ids": suite_ids,
            "seeds": int(args.seeds),
            "prompt_manifest": "prompt_manifest.jsonl",
            "prompt_manifest_source": prompt_manifest_source,
            "input_prompt_manifest_uri": prompt_manifest_uri or None,
            "control_metrics": "control_metrics.json",
            "native_cara_metrics": "pending_generation_scoring",
            "external_probe_metrics": "pending_generation_scoring",
            "audio_outputs": "pending_generation_scoring",
        },
    )
    if parse_bool(args.load_base_model):
        _load_base_model(report, args.base_checkpoint)
    report["status"] = "passed"
    report["stage"] = "completed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_data", required=True)
    parser.add_argument("--trained_model_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_relative_path", default="stable_audio_open_small/manifest.jsonl")
    parser.add_argument("--model_ids", required=True)
    parser.add_argument("--suite_ids", required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--prompt_manifest_uri", default="")
    parser.add_argument("--base_checkpoint", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--load_base_model", default="true")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "14_benchmark_testing_stable_audio_eval",
        "status": "failed",
        "stage": "initializing",
        "created_at": _utc_now(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,
        "errors": [],
        "warnings": [],
    }
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        run(args, report)
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
    write_report(Path(args.output_dir), report, metadata, report_alias="benchmark_testing_stable_audio_eval_report.json")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
