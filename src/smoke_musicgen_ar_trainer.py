from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from musicgen_cara_tokens import (
    build_cara_suffix_vocab,
    build_musicgen_registry_resolver,
    decode_cara_suffix,
    encode_cara_suffix,
    validate_musicgen_encodec_manifest,
)
from test_prep_common import base_metadata, parse_bool, write_report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _variant_alignment(variant: str) -> dict[str, Any]:
    return {
        "no_cara_baseline": {
            "stage": "same-data no-CARA autoregressive baseline",
            "ordinary_prompt_unchanged": True,
            "cara_prompt_text": False,
            "detached_probe": False,
            "non_detached_cara_suffix_loss": False,
            "satisfies_cara_strong_claim": False,
        },
        "cara_lite": {
            "stage": "CARA-lite prompt-control autoregressive smoke",
            "ordinary_prompt_unchanged": False,
            "cara_prompt_text": True,
            "detached_probe": False,
            "non_detached_cara_suffix_loss": False,
            "satisfies_cara_strong_claim": False,
        },
        "cara_probe": {
            "stage": "detached CARA suffix probe smoke",
            "ordinary_prompt_unchanged": True,
            "cara_prompt_text": False,
            "detached_probe": True,
            "non_detached_cara_suffix_loss": False,
            "satisfies_cara_strong_claim": False,
        },
        "cara_strong": {
            "stage": "CARA-Strong autoregressive suffix smoke",
            "ordinary_prompt_unchanged": True,
            "cara_prompt_text": False,
            "detached_probe": False,
            "non_detached_cara_suffix_loss": True,
            "satisfies_cara_strong_claim": True,
        },
    }[variant]


class MusicGenTokenCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        token_cache_dir: Path,
        suffix_vocab: dict[str, Any],
        *,
        max_audio_tokens: int,
    ) -> None:
        self.rows = rows
        self.token_cache_dir = token_cache_dir
        self.suffix_vocab = suffix_vocab
        self.max_audio_tokens = max_audio_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        token_path = self.token_cache_dir / str(row["encodec_token_path"])
        payload = torch.load(token_path, map_location="cpu", weights_only=False)
        codes = payload["codes"].long().reshape(-1)
        if codes.numel() < 2:
            raise RuntimeError(f"Too few EnCodec tokens in {token_path}")
        codes = codes[: self.max_audio_tokens]
        return {
            "audio_tokens": codes,
            "suffix_tokens": torch.tensor(encode_cara_suffix(row, self.suffix_vocab), dtype=torch.long),
            "cara_pool_index": int(row["cara_pool_index"]),
            "cara_pool_family_index": int(row["cara_pool_family_index"]),
            "cara_pool_id": str(row["cara_pool_id"]),
            "chunk_id": str(row.get("chunk_id") or index),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    audio_len = max(item["audio_tokens"].numel() for item in batch)
    suffix_len = max(item["suffix_tokens"].numel() for item in batch)
    audio = torch.zeros((len(batch), audio_len), dtype=torch.long)
    audio_mask = torch.zeros((len(batch), audio_len), dtype=torch.bool)
    suffix = torch.zeros((len(batch), suffix_len), dtype=torch.long)
    suffix_mask = torch.zeros((len(batch), suffix_len), dtype=torch.bool)
    pool_ids: list[str] = []
    chunk_ids: list[str] = []
    pool_indices: list[int] = []
    family_indices: list[int] = []
    for index, item in enumerate(batch):
        tokens = item["audio_tokens"]
        suffix_tokens = item["suffix_tokens"]
        audio[index, : tokens.numel()] = tokens
        audio_mask[index, : tokens.numel()] = True
        suffix[index, : suffix_tokens.numel()] = suffix_tokens
        suffix_mask[index, : suffix_tokens.numel()] = True
        pool_ids.append(item["cara_pool_id"])
        chunk_ids.append(item["chunk_id"])
        pool_indices.append(item["cara_pool_index"])
        family_indices.append(item["cara_pool_family_index"])
    return {
        "audio_tokens": audio,
        "audio_mask": audio_mask,
        "suffix_tokens": suffix,
        "suffix_mask": suffix_mask,
        "cara_pool_id": pool_ids,
        "chunk_id": chunk_ids,
        "cara_pool_index": torch.tensor(pool_indices, dtype=torch.long),
        "cara_pool_family_index": torch.tensor(family_indices, dtype=torch.long),
    }


class TinyAutoregressiveCARAModel(nn.Module):
    def __init__(self, *, audio_vocab_size: int, suffix_vocab_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.audio_embed = nn.Embedding(audio_vocab_size, hidden_dim)
        self.audio_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.audio_head = nn.Linear(hidden_dim, audio_vocab_size)
        self.suffix_pos = nn.Embedding(64, hidden_dim)
        self.suffix_head = nn.Linear(hidden_dim, suffix_vocab_size)

    def forward(self, audio_tokens: torch.Tensor, suffix_len: int) -> dict[str, torch.Tensor]:
        hidden, _state = self.audio_rnn(self.audio_embed(audio_tokens))
        audio_logits = self.audio_head(hidden)
        context = hidden[:, -1, :]
        positions = torch.arange(suffix_len, device=audio_tokens.device).unsqueeze(0)
        suffix_hidden = context.unsqueeze(1) + self.suffix_pos(positions)
        return {
            "audio_logits": audio_logits,
            "audio_context": context,
            "suffix_logits": self.suffix_head(suffix_hidden),
        }


def _suffix_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, vocab: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    pred = logits.argmax(dim=-1)
    exact = ((pred == targets) | ~mask).all(dim=1).float().mean().item()
    decoded = [decode_cara_suffix(pred[index][mask[index]].tolist(), vocab, resolver) for index in range(pred.shape[0])]
    registry_valid = sum(1 for item in decoded if item["registry_valid"]) / max(1, len(decoded))
    hierarchical_valid = sum(1 for item in decoded if item["hierarchical_valid"]) / max(1, len(decoded))
    checksum_valid = sum(1 for item in decoded if item["checksum_valid"]) / max(1, len(decoded))
    return {
        "cara/suffix_exact": exact,
        "cara/registry_valid": registry_valid,
        "cara/hierarchical_valid": hierarchical_valid,
        "cara/checksum_valid": checksum_valid,
        "decoded_preview": decoded[:3],
    }


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    token_cache_dir = Path(args.token_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = token_cache_dir / args.encodec_manifest_relative_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"MusicGen EnCodec manifest not found: {manifest_path}")

    rows = _read_jsonl(manifest_path)
    if int(args.max_train_files) > 0:
        rows = rows[: int(args.max_train_files)]
    label_summary = validate_musicgen_encodec_manifest(rows)
    resolver = build_musicgen_registry_resolver(rows)
    suffix_vocab = build_cara_suffix_vocab(rows)
    (output_dir / "cara_registry_resolver.json").write_text(json.dumps(resolver, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "cara_suffix_vocab.json").write_text(json.dumps(suffix_vocab, indent=2, sort_keys=True), encoding="utf-8")

    report.update(
        {
            "stage": "validate_musicgen_token_cache",
            "variant": args.variant,
            "research_alignment": _variant_alignment(args.variant),
            "encodec_manifest_rows": len(rows),
            "cara_label_summary": label_summary,
            "cara_registry": {
                "registry_hash": resolver["registry_hash"],
                "pool_count": resolver["pool_count"],
                "family_count": resolver["family_count"],
                "suffix_vocab_size": suffix_vocab["size"],
                "suffix_vocab_hash": suffix_vocab["hash"],
            },
        }
    )

    if parse_bool(args.load_musicgen_checkpoint):
        report["stage"] = "load_musicgen_checkpoint"
        from audiocraft.models import MusicGen

        started = time.time()
        model = MusicGen.get_pretrained(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        compression_model = model.compression_model
        report["musicgen_checkpoint_load_seconds"] = round(time.time() - started, 3)
        report["musicgen_checkpoint"] = args.checkpoint
        report["compression_model"] = {
            "sample_rate": getattr(compression_model, "sample_rate", None),
            "channels": getattr(compression_model, "channels", None),
            "frame_rate": getattr(compression_model, "frame_rate", None),
            "num_codebooks": getattr(compression_model, "num_codebooks", None),
            "cardinality": getattr(compression_model, "cardinality", None),
        }

    dataset = MusicGenTokenCacheDataset(
        rows,
        token_cache_dir,
        suffix_vocab,
        max_audio_tokens=int(args.max_audio_tokens),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
    first_batch = next(iter(loader))
    report["first_batch"] = {
        "audio_tokens_shape": list(first_batch["audio_tokens"].shape),
        "suffix_tokens_shape": list(first_batch["suffix_tokens"].shape),
        "chunk_id_preview": first_batch["chunk_id"][:3],
        "cara_pool_id_preview": first_batch["cara_pool_id"][:3],
    }
    if parse_bool(args.preflight_only) or parse_bool(args.dry_run):
        report["status"] = "passed"
        report["stage"] = "preflight_complete"
        return

    report["stage"] = "trainer_fit"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_vocab_size = int(args.audio_vocab_size)
    model = TinyAutoregressiveCARAModel(
        audio_vocab_size=audio_vocab_size,
        suffix_vocab_size=int(suffix_vocab["size"]),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    if args.variant == "cara_probe":
        for name, param in model.named_parameters():
            if not name.startswith("suffix_"):
                param.requires_grad_(False)
        optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=float(args.learning_rate))
        report["optimizer_mode"] = "suffix_probe_only"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
        report["optimizer_mode"] = "audio_ar_plus_suffix" if args.variant == "cara_strong" else "audio_ar_only"

    global_step = 0
    losses: list[float] = []
    cara_metrics: dict[str, Any] = {}
    max_steps = int(args.max_steps)
    while global_step < max_steps:
        for batch in loader:
            audio_tokens = (batch["audio_tokens"].to(device) % audio_vocab_size).long()
            suffix_tokens = batch["suffix_tokens"].to(device)
            suffix_mask = batch["suffix_mask"].to(device)
            outputs = model(audio_tokens, suffix_tokens.shape[1])
            audio_loss = F.cross_entropy(
                outputs["audio_logits"][:, :-1, :].reshape(-1, audio_vocab_size),
                audio_tokens[:, 1:].reshape(-1),
            )
            suffix_loss = torch.tensor(0.0, device=device)
            if args.variant in {"cara_probe", "cara_strong"}:
                suffix_logits = outputs["suffix_logits"]
                if args.variant == "cara_probe":
                    suffix_logits = suffix_logits.detach() + (suffix_logits - suffix_logits.detach())
                suffix_loss = F.cross_entropy(
                    suffix_logits.reshape(-1, int(suffix_vocab["size"])),
                    suffix_tokens.reshape(-1),
                    reduction="none",
                )
                suffix_loss = (suffix_loss * suffix_mask.reshape(-1).float()).sum() / suffix_mask.sum().clamp_min(1)
            loss = audio_loss if args.variant in {"no_cara_baseline", "cara_lite"} else audio_loss + float(args.attribution_loss_weight) * suffix_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            if args.variant in {"cara_probe", "cara_strong"}:
                cara_metrics = _suffix_metrics(outputs["suffix_logits"].detach().cpu(), suffix_tokens.cpu(), suffix_mask.cpu(), suffix_vocab, resolver)
            if global_step >= max_steps:
                break

    report["global_step"] = global_step
    report["train_loss"] = sum(losses) / len(losses) if losses else None
    report["latest_cara_metrics"] = cara_metrics
    report["status"] = "passed" if global_step >= max_steps else "failed"
    if args.variant in {"cara_probe", "cara_strong"} and not cara_metrics.get("cara/registry_valid"):
        report["warnings"].append("CARA suffix smoke produced no registry-valid decoded suffixes; inspect logits before using as evidence.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encodec_manifest_relative_path", default="manifest.encodec.jsonl")
    parser.add_argument("--checkpoint", default="facebook/musicgen-small")
    parser.add_argument("--variant", choices=["no_cara_baseline", "cara_lite", "cara_probe", "cara_strong"], default="no_cara_baseline")
    parser.add_argument("--run_name", default="cara-musicgen-smoke")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--attribution_loss_weight", type=float, default=0.05)
    parser.add_argument("--max_train_files", type=int, default=2048)
    parser.add_argument("--max_audio_tokens", type=int, default=512)
    parser.add_argument("--audio_vocab_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--preflight_only", default="false")
    parser.add_argument("--load_musicgen_checkpoint", default="false")
    parser.add_argument("--dashboard_triggered", default="false")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "test_name": "08_smoke_musicgen_ar_trainer",
        "status": "failed",
        "run_name": args.run_name,
        "variant": args.variant,
        "errors": [],
        "warnings": [],
        "torch_version": torch.__version__,
    }
    try:
        _run(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))
        report["traceback"] = traceback.format_exc()
    metadata = base_metadata(
        test_name="08_smoke_musicgen_ar_trainer",
        compute="gpu-smoke-h100",
        environment="azureml:env-musicgen-audiocraft:3",
        dashboard_triggered=parse_bool(args.dashboard_triggered),
        report=report,
        model_family="musicgen",
        environment_name="env-musicgen-audiocraft",
        environment_version="3",
    )
    write_report(Path(args.output_dir), report, metadata, report_alias="musicgen_ar_smoke_trainer_report.json")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
