from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.baselines import build_keyword_pool_map, keyword_attribution, nn_attribution, prior_attribution, random_attribution
from evaluation.metrics import compute_all_metrics, parse_attr_string
from model.attribution_head import CARAAttributionHead
from model.constrained_decoder import ConstrainedCARADecoder
from model.stable_audio_integration import extract_conditioned_hidden_states, load_pretrained_model, generate_conditioned_audio
from validation.validator import CARAValidator


EVAL_PROMPTS = [
    {"prompt": "smooth jazz saxophone solo, warm tone, relaxed", "expected_primary_pool": "MULTI"},
    {"prompt": "hard techno kick drum loop, 140 BPM, distorted", "expected_primary_pool": "MULTI"},
    {"prompt": "orchestral string section, cinematic, dramatic", "expected_primary_pool": "MULTI"},
    {"prompt": "jazz-electronic fusion, saxophone over synthesizer pads", "expected_primary_pool": "MULTI"},
    {"prompt": "high quality audio, clear sound", "expected_primary_pool": "UNKNOWN"},
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pool_distribution(rows: list[dict[str, str]]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for row in rows:
        codeword = row.get("codeword") or row.get("primary_pool") or "UNKNOWN"
        counts[codeword] = counts.get(codeword, 0.0) + 1.0
    return counts


def _fake_generation(prompt: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    seed = abs(hash(prompt)) % (2**32)
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(1, 4, 64)).astype(np.float32)
    latent = rng.normal(size=(dim,)).astype(np.float32)
    return hidden, latent


def _real_generation(prompt: str) -> tuple[np.ndarray, np.ndarray]:
    model, _model_config, device = load_pretrained_model()
    latents = generate_conditioned_audio(model, prompt, device=device, return_latents=True)
    if hasattr(latents, "detach"):
        latent_np = latents.detach().cpu().numpy().astype(np.float32)
    else:
        latent_np = np.asarray(latents, dtype=np.float32)
    hidden_states = extract_conditioned_hidden_states(model, latents, prompt, device=device)
    if hidden_states:
        hidden = hidden_states[-1].detach().cpu().numpy().astype(np.float32)
    else:
        hidden = np.mean(latent_np, axis=-1, keepdims=False)
        if hidden.ndim == 2:
            hidden = hidden[:, :, None]
        elif hidden.ndim == 1:
            hidden = hidden[None, :, None]
        hidden = hidden.astype(np.float32)
    return hidden, latent_np.flatten().astype(np.float32)


def _load_head_if_available(codebook, checkpoint_path: Path):
    if not checkpoint_path.exists():
        return None
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    first_weight = state_dict.get("feature_net.0.weight")
    if first_weight is None:
        return None
    hidden_dim = int(first_weight.shape[1])
    head = CARAAttributionHead(hidden_dim, len(codebook.codeword_to_idx), num_slots=3)
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    return head


def _fake_head_outputs(codebook, hidden: np.ndarray):
    num_codewords = max(1, len(codebook.idx_to_codeword))
    cw_logits = []
    for slot in range(3):
        logits = np.zeros((1, num_codewords), dtype=np.float32)
        logits[0, slot % num_codewords] = 1.0
        cw_logits.append(logits)
    prob_bins = np.array([[50, 30, 20]], dtype=np.int64)
    return cw_logits, prob_bins


def run_full_evaluation(codebook, prompts: list[dict[str, str]], master_rows: list[dict[str, str]], index_vectors: np.ndarray, faiss_metadata: list[dict[str, Any]], head_checkpoint_path: Path | None = None):
    decoder = ConstrainedCARADecoder(codebook)
    validator = CARAValidator(codebook)
    keyword_pool_map = build_keyword_pool_map(master_rows)
    prior_distribution = _pool_distribution(master_rows)
    codewords = list(codebook.codeword_to_idx.keys())
    head = _load_head_if_available(codebook, head_checkpoint_path or Path("checkpoints/attribution_head_v1.pt"))
    results = []
    latent_dim = int(index_vectors.shape[1]) if index_vectors.size else 128
    for index, prompt_entry in enumerate(prompts):
        prompt = prompt_entry["prompt"]
        try:
            hidden, latent = _real_generation(prompt)
        except Exception:
            hidden, latent = _fake_generation(prompt, latent_dim)
        if head is not None:
            hidden_tensor = torch.tensor(hidden, dtype=torch.float32)
            with torch.no_grad():
                cw_logits, _prob_dist, prob_bins = head(hidden_tensor)
        else:
            cw_logits_np, prob_bins_np = _fake_head_outputs(codebook, hidden)
            cw_logits = [torch.tensor(item) for item in cw_logits_np]
            prob_bins = torch.tensor(prob_bins_np)
        attr_string_head = decoder.decode(cw_logits, prob_bins)[0]
        validation_result = validator.validate(attr_string_head)
        attr_string_nn = nn_attribution(latent, index_vectors, faiss_metadata)
        attr_string_kw = keyword_attribution(prompt, keyword_pool_map, prior_distribution)
        attr_string_prior = prior_attribution(prior_distribution)
        attr_string_random = random_attribution(codewords)
        head_slots = parse_attr_string(validation_result.validated_string)
        results.append(
            {
                "generation_id": f"gen_{index:04d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "expected_pool": prompt_entry["expected_primary_pool"],
                "model_version": "scaffold",
                "head_attribution": validation_result.validated_string,
                "head_state": validation_result.state.value,
                "head_pool1": head_slots[0][0] if head_slots else "",
                "head_conf1": head_slots[0][1] if head_slots else 0,
                "head_pool2": head_slots[1][0] if len(head_slots) > 1 else "",
                "head_conf2": head_slots[1][1] if len(head_slots) > 1 else 0,
                "head_pool3": head_slots[2][0] if len(head_slots) > 2 else "",
                "head_conf3": head_slots[2][1] if len(head_slots) > 2 else 0,
                "nn_attribution": attr_string_nn,
                "keyword_attribution": attr_string_kw,
                "prior_attribution": attr_string_prior,
                "random_attribution": attr_string_random,
                "head_errors": json.dumps(validation_result.errors),
                "head_repairs": json.dumps(validation_result.repairs),
            }
        )
    return results, compute_all_metrics(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-registry", default="master_registry.csv")
    parser.add_argument("--faiss-index", default="evaluation/faiss_index.npy")
    parser.add_argument("--faiss-metadata", default="evaluation/faiss_metadata.json")
    parser.add_argument("--codebook", default="registry/pools.json")
    parser.add_argument("--hierarchy", default="registry/hierarchy.json")
    parser.add_argument("--prompts", default="evaluation/prompts.json")
    parser.add_argument("--output-log", default="evaluation_log.csv")
    parser.add_argument("--metrics-output", default="evaluation/metrics_latest.json")
    parser.add_argument("--head-checkpoint", default="checkpoints/attribution_head_v1.pt")
    return parser.parse_args()


def main() -> None:
    from registry.validate import CARACodebook

    args = parse_args()
    codebook = CARACodebook(args.codebook, args.hierarchy)
    prompts_path = Path(args.prompts)
    prompts = _load_json(prompts_path) if prompts_path.exists() else EVAL_PROMPTS
    master_rows = _load_rows(Path(args.master_registry)) if Path(args.master_registry).exists() else []
    index_vectors = np.load(args.faiss_index) if Path(args.faiss_index).exists() else np.zeros((0, 0), dtype=np.float32)
    faiss_metadata = _load_json(Path(args.faiss_metadata)) if Path(args.faiss_metadata).exists() else []
    results, metrics = run_full_evaluation(codebook, prompts, master_rows, index_vectors, faiss_metadata, head_checkpoint_path=Path(args.head_checkpoint))
    output_path = Path(args.output_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row.keys()}) if results else ["generation_id"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(results), "metrics_output": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
