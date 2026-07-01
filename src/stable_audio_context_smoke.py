from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from test_prep_common import base_metadata, json_safe, parse_bool, write_report


LANES = [
    "context_plus_prompt",
    "context_only",
    "prompt_only",
    "shuffled_context",
    "mismatched_family_context",
]


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
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False) + "\n")


def _hash_float(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _one_hot(index: int, size: int) -> list[float]:
    vector = [0.0] * size
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _chunk_features(row: dict[str, Any], *, pool_count: int, family_count: int, prefix: str) -> list[float]:
    prompt = str(row.get("prompt") or "")
    title = str(row.get("title") or "")
    pool_index = _safe_int(row.get("cara_pool_index"), -1)
    family_index = _safe_int(row.get("cara_pool_family_index"), -1)
    duration = min(1.0, max(0.0, _safe_float(row.get("duration_sec"), 0.0) / 12.0))
    return [
        duration,
        _hash_float(prompt, f"{prefix}:prompt"),
        _hash_float(title, f"{prefix}:title"),
        *(_one_hot(pool_index, pool_count) if pool_count <= 256 else []),
        *(_one_hot(family_index, family_count) if family_count <= 64 else []),
    ]


def _average_context_features(
    examples: list[dict[str, Any]],
    *,
    pool_count: int,
    family_count: int,
    prefix: str,
) -> list[float]:
    width = 3 + (pool_count if pool_count <= 256 else 0) + (family_count if family_count <= 64 else 0)
    if not examples:
        return [0.0] * width
    vectors = [_chunk_features(example, pool_count=pool_count, family_count=family_count, prefix=prefix) for example in examples]
    return [sum(vector[position] for vector in vectors) / len(vectors) for position in range(width)]


def _target_prompt_features(target: dict[str, Any]) -> list[float]:
    prompt = str(target.get("prompt") or "")
    return [
        _hash_float(prompt, "target_prompt:a"),
        _hash_float(prompt, "target_prompt:b"),
        min(1.0, max(0.0, _safe_float(target.get("duration_sec"), 0.0) / 12.0)),
    ]


def _control_lookup(control_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("context_pack_id") or ""): row for row in control_rows}


def _lane_examples(pack: dict[str, Any], control: dict[str, Any] | None, lane: str) -> list[dict[str, Any]]:
    if lane in {"context_plus_prompt", "context_only"}:
        return pack.get("context_examples") if isinstance(pack.get("context_examples"), list) else []
    if lane == "prompt_only":
        return []
    controls = control.get("controls") if isinstance(control, dict) else {}
    lane_row = controls.get(lane) if isinstance(controls, dict) else {}
    examples = lane_row.get("context_examples") if isinstance(lane_row, dict) else []
    return examples if isinstance(examples, list) else []


def _lane_feature_vector(
    pack: dict[str, Any],
    control: dict[str, Any] | None,
    lane: str,
    *,
    pool_count: int,
    family_count: int,
) -> list[float]:
    target = pack.get("target") if isinstance(pack.get("target"), dict) else {}
    prompt_features = _target_prompt_features(target) if lane != "context_only" else [0.0, 0.0, 0.0]
    examples = _lane_examples(pack, control, lane)
    context_features = (
        _average_context_features(examples, pool_count=pool_count, family_count=family_count, prefix=lane)
        if lane != "prompt_only"
        else _average_context_features([], pool_count=pool_count, family_count=family_count, prefix=lane)
    )
    lane_features = [1.0 if item == lane else 0.0 for item in LANES]
    return prompt_features + context_features + lane_features


class ContextLaneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        controls: dict[str, dict[str, Any]],
        *,
        lanes: list[str],
        pool_count: int,
        family_count: int,
    ) -> None:
        self.items: list[tuple[list[float], int, int, str, str]] = []
        for pack in rows:
            target = pack.get("target") if isinstance(pack.get("target"), dict) else {}
            pool_index = _safe_int(target.get("cara_pool_index"), -1)
            family_index = _safe_int(target.get("cara_pool_family_index"), -1)
            if pool_index < 0 or family_index < 0:
                continue
            control = controls.get(str(pack.get("context_pack_id") or ""))
            for lane in lanes:
                self.items.append(
                    (
                        _lane_feature_vector(
                            pack,
                            control,
                            lane,
                            pool_count=pool_count,
                            family_count=family_count,
                        ),
                        pool_index,
                        family_index,
                        str(pack.get("context_pack_id") or ""),
                        lane,
                    )
                )
        if not self.items:
            raise RuntimeError("No valid context-lane examples were available for smoke training.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        features, pool_index, family_index, pack_id, lane = self.items[index]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(pool_index, dtype=torch.long),
            torch.tensor(family_index, dtype=torch.long),
            pack_id,
            lane,
        )


class ContextSmokeHead(nn.Module):
    def __init__(self, input_dim: int, pool_count: int, family_count: int) -> None:
        super().__init__()
        hidden = max(64, min(512, input_dim * 2))
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pool_head = nn.Linear(hidden, pool_count)
        self.family_head = nn.Linear(hidden, family_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.pool_head(hidden), self.family_head(hidden)


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if str((row.get("target") or {}).get("split") or "") == "train"]
    eval_rows = [row for row in rows if str((row.get("target") or {}).get("split") or "") in {"validation", "test"}]
    if not train_rows:
        train_rows = rows[: max(1, int(len(rows) * 0.8))]
    if not eval_rows:
        eval_rows = rows[max(1, int(len(rows) * 0.8)) :] or rows[: min(len(rows), 256)]
    return train_rows, eval_rows


def _limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    return rows[:limit]


def _lane_metrics(
    model: ContextSmokeHead,
    dataset: ContextLaneDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for features, pool_target, family_target, pack_ids, lanes in loader:
            features = features.to(device)
            pool_target = pool_target.to(device)
            family_target = family_target.to(device)
            pool_logits, family_logits = model(features)
            pool_pred = pool_logits.argmax(dim=-1)
            family_pred = family_logits.argmax(dim=-1)
            pool_conf = torch.softmax(pool_logits, dim=-1).amax(dim=-1)
            family_conf = torch.softmax(family_logits, dim=-1).amax(dim=-1)
            for offset, lane in enumerate(lanes):
                lane_key = str(lane)
                counts[lane_key]["total"] += 1
                counts[lane_key]["pool_top1"] += int(pool_pred[offset].item() == pool_target[offset].item())
                counts[lane_key]["family_top1"] += int(family_pred[offset].item() == family_target[offset].item())
                if len(examples) < 40:
                    examples.append(
                        {
                            "context_pack_id": pack_ids[offset],
                            "lane": lane_key,
                            "expected_pool_index": int(pool_target[offset].item()),
                            "predicted_pool_index": int(pool_pred[offset].item()),
                            "pool_confidence": round(float(pool_conf[offset].item()), 6),
                            "expected_family_index": int(family_target[offset].item()),
                            "predicted_family_index": int(family_pred[offset].item()),
                            "family_confidence": round(float(family_conf[offset].item()), 6),
                            "pool_correct": bool(pool_pred[offset].item() == pool_target[offset].item()),
                            "family_correct": bool(family_pred[offset].item() == family_target[offset].item()),
                        }
                    )
    metrics: dict[str, Any] = {}
    for lane, counter in sorted(counts.items()):
        total = max(1, int(counter["total"]))
        metrics[lane] = {
            "total": int(counter["total"]),
            "pool_top1": counter["pool_top1"] / total,
            "family_top1": counter["family_top1"] / total,
        }
    return metrics, examples


def run_context_smoke(
    *,
    context_pack_dir: Path,
    context_cache_dir: Path,
    output_dir: Path,
    context_pack_relative_path: str,
    context_controls_relative_path: str,
    context_cache_relative_path: str,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    max_train_rows: int,
    max_eval_rows: int,
    dry_run: bool,
) -> dict[str, Any]:
    pack_path = context_pack_dir / context_pack_relative_path
    control_path = context_pack_dir / context_controls_relative_path
    cache_path = context_cache_dir / context_cache_relative_path
    print(f"[context-smoke] reading context packs from {pack_path}", flush=True)
    print(f"[context-smoke] reading context controls from {control_path}", flush=True)
    print(f"[context-smoke] reading context cache from {cache_path}", flush=True)
    if not pack_path.exists():
        raise FileNotFoundError(f"Context pack manifest not found: {pack_path}")
    if not control_path.exists():
        raise FileNotFoundError(f"Context controls manifest not found: {control_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Context cache manifest not found: {cache_path}")
    packs = _read_jsonl(pack_path)
    controls = _control_lookup(_read_jsonl(control_path))
    cache_rows = _read_jsonl(cache_path)
    if not packs or not controls or not cache_rows:
        raise RuntimeError("Context pack, control, and cache manifests must all be non-empty.")
    print(
        f"[context-smoke] loaded {len(packs):,} packs, {len(controls):,} controls, {len(cache_rows):,} cache rows",
        flush=True,
    )
    pool_count = max(_safe_int((row.get("target") or {}).get("cara_pool_index"), -1) for row in packs) + 1
    family_count = max(_safe_int((row.get("target") or {}).get("cara_pool_family_index"), -1) for row in packs) + 1
    if pool_count <= 1 or family_count <= 1:
        raise RuntimeError(f"Unexpected label cardinality: pool_count={pool_count}, family_count={family_count}")
    train_rows, eval_rows = _split_rows(packs)
    train_rows = _limit_rows(train_rows, max_train_rows)
    eval_rows = _limit_rows(eval_rows, max_eval_rows)
    train_dataset = ContextLaneDataset(
        train_rows,
        controls,
        lanes=["context_plus_prompt", "context_only", "prompt_only"],
        pool_count=pool_count,
        family_count=family_count,
    )
    eval_dataset = ContextLaneDataset(
        eval_rows,
        controls,
        lanes=LANES,
        pool_count=pool_count,
        family_count=family_count,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[context-smoke] using device={device}; train_examples={len(train_dataset):,}; eval_examples={len(eval_dataset):,}", flush=True)
    input_dim = int(train_dataset[0][0].numel())
    model = ContextSmokeHead(input_dim=input_dim, pool_count=pool_count, family_count=family_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    steps = max(1, int(max_steps))
    batch = max(1, int(batch_size))
    generator = torch.Generator().manual_seed(1337)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0, generator=generator)
    iterator = iter(train_loader)
    last_loss = math.nan
    if not dry_run:
        model.train()
        for step in range(1, steps + 1):
            try:
                features, pool_target, family_target, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                features, pool_target, family_target, _, _ = next(iterator)
            features = features.to(device)
            pool_target = pool_target.to(device)
            family_target = family_target.to(device)
            pool_logits, family_logits = model(features)
            loss = loss_fn(pool_logits, pool_target) + 0.5 * loss_fn(family_logits, family_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if step == 1 or step % 50 == 0 or step == steps:
                print(f"[context-smoke] trained step {step:,}/{steps:,}; loss={last_loss:.6f}", flush=True)
    lane_metrics, prediction_examples = _lane_metrics(model, eval_dataset, device=device, batch_size=batch)
    context_plus = lane_metrics.get("context_plus_prompt", {})
    shuffled = lane_metrics.get("shuffled_context", {})
    mismatched = lane_metrics.get("mismatched_family_context", {})
    report = {
        "test_name": "13_stable_audio_context_smoke",
        "status": "passed",
        "created_at": _utc_now(),
        "context_pack_rows": len(packs),
        "context_cache_rows": len(cache_rows),
        "train_pack_rows": len(train_rows),
        "eval_pack_rows": len(eval_rows),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "max_steps": steps,
        "batch_size": batch,
        "learning_rate": learning_rate,
        "last_loss": None if math.isnan(last_loss) else last_loss,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pool_count": pool_count,
        "family_count": family_count,
        "lane_metrics": lane_metrics,
        "context_signal_checks": {
            "context_plus_pool_top1": context_plus.get("pool_top1"),
            "shuffled_context_pool_top1": shuffled.get("pool_top1"),
            "mismatched_family_pool_top1": mismatched.get("pool_top1"),
            "context_shuffle_delta": (
                context_plus.get("pool_top1", 0.0) - shuffled.get("pool_top1", 0.0)
                if context_plus and shuffled
                else None
            ),
            "mismatched_context_delta": (
                context_plus.get("pool_top1", 0.0) - mismatched.get("pool_top1", 0.0)
                if context_plus and mismatched
                else None
            ),
        },
        "paper_alignment": {
            "context_examples_consumed_as_separate_features": True,
            "prompt_only_lane": True,
            "context_only_lane": True,
            "shuffled_context_lane": True,
            "mismatched_family_lane": True,
            "full_stable_audio_dit_training": False,
            "claim_scope": "context_conditioner_smoke_only",
        },
        "artifact_files": {
            "context_smoke_metrics": "context_smoke_metrics.json",
            "context_lane_predictions": "context_lane_predictions.jsonl",
            "context_conditioner_contract": "context_conditioner_contract.json",
        },
        "dry_run": dry_run,
    }
    if not dry_run:
        print(f"[context-smoke] writing artifacts to {output_dir}", flush=True)
        _write_json(output_dir / "context_smoke_metrics.json", report)
        _write_jsonl(output_dir / "context_lane_predictions.jsonl", prediction_examples)
        _write_json(
            output_dir / "context_conditioner_contract.json",
            {
                "format": "cara_context_conditioner_contract_v1",
                "input_dim": input_dim,
                "lanes": LANES,
                "pool_count": pool_count,
                "family_count": family_count,
                "next_step": "replace metadata features with frozen Stable Audio context latents before full context DiT fine-tuning",
            },
        )
    print("[context-smoke] completed with status=passed", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stable Audio Context Diffusion conditioner smoke.")
    parser.add_argument("--context_pack_dir", required=True)
    parser.add_argument("--context_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_pack_relative_path", default="context_pack_manifest.jsonl")
    parser.add_argument("--context_controls_relative_path", default="context_controls_manifest.jsonl")
    parser.add_argument("--context_cache_relative_path", default="context_cache_manifest.jsonl")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_train_rows", type=int, default=4096)
    parser.add_argument("--max_eval_rows", type=int, default=1024)
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "test_name": "13_stable_audio_context_smoke",
        "status": "running",
        "dashboard_triggered": parse_bool(args.dashboard_triggered),
    }
    try:
        report.update(
            run_context_smoke(
                context_pack_dir=Path(args.context_pack_dir),
                context_cache_dir=Path(args.context_cache_dir),
                output_dir=Path(args.output_dir),
                context_pack_relative_path=args.context_pack_relative_path,
                context_controls_relative_path=args.context_controls_relative_path,
                context_cache_relative_path=args.context_cache_relative_path,
                max_steps=max(1, int(args.max_steps)),
                batch_size=max(1, int(args.batch_size)),
                learning_rate=float(args.learning_rate),
                max_train_rows=max(0, int(args.max_train_rows)),
                max_eval_rows=max(0, int(args.max_eval_rows)),
                dry_run=parse_bool(args.dry_run),
            )
        )
        alias = "stable_audio_context_smoke_report.json"
    except Exception as exc:
        report.update({"status": "failed", "stage": "context_smoke", "error": str(exc)})
        alias = "stable_audio_context_smoke_report.json"
        raise
    finally:
        metadata = base_metadata(
            test_name="13_stable_audio_context_smoke",
            compute="gpu-smoke-h100",
            environment="azureml:env-stable-audio-tools:8",
            dashboard_triggered=parse_bool(args.dashboard_triggered),
            report=report,
            model_family="stable_audio_open_small_context_diffusion",
        )
        write_report(Path(args.output_dir), report, metadata, report_alias=alias)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
