#!/usr/bin/env python3
"""
03_run_attribution_head.py

Run the trained CARA attribution head on the same probe prompts.
Produces Run B results: CARA attribution head accuracy.

This is separate from the linear probe (Tool 4) because the attribution head
is specifically trained for pool classification, while the linear probe is
a generic diagnostic.

Usage:
    python 03_run_attribution_head.py \
        --model-config /path/to/model_config.json \
        --dit-ckpt checkpoints/dit_frozen_v1.safetensors \
        --head-ckpt checkpoints/attribution_head_v1.pt \
        --prompts probe/prompts/probe_prompts.json \
        --output probe/results/reports/attribution_head.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from stable_audio_tools.inference.generation import generate_diffusion_cond
from probe.utils.hooks import DiTHiddenStateCollector
from probe.utils.metrics import (
    compute_pool_accuracy,
    compute_calibration_error,
    compute_degradation_states,
)
from probe.utils.pool_config import POOL_DEFINITIONS


def load_cara_model(model_config_path: str, dit_ckpt_path: str):
    """Load the CARA fine-tuned frozen DiT."""
    from stable_audio_tools.models.factory import create_model_from_config
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    state_dict = torch.load(dit_ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, model_config


def load_attribution_head(head_ckpt_path: str, device: torch.device):
    """Load the trained attribution head."""
    head = torch.load(head_ckpt_path, map_location=device)
    head.eval()
    return head


def decode_cara_output(logits: torch.Tensor, codeword_registry: dict) -> dict:
    """
    Decode attribution head logits into a structured CARA attribution string.
    Applies constrained decoding: only registered codewords are valid outputs.
    
    Returns dict with:
        - top3_pools: list of (pool_name, probability) for top 3 pools
        - cara_string: formatted ATTR|CW@PP|...|END string
        - degradation_state: A/B/C/D
    """
    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    # Get top-3 pool indices
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_pools = [
        (codeword_registry["id_to_pool"][i], float(probs[i]))
        for i in top3_idx
    ]
    
    # Format CARA string: probabilities as integer bins 00-99
    slots = []
    for pool_name, prob in top3_pools:
        codeword = codeword_registry["pool_to_codeword"][pool_name]
        pp = min(99, int(prob * 100))
        slots.append(f"{codeword}@{pp:02d}")
    
    cara_string = "ATTR|" + "|".join(slots) + "|END"
    
    return {
        "top3_pools": top3_pools,
        "cara_string": cara_string,
        "top1_pool": top3_pools[0][0],
        "top1_prob": top3_pools[0][1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dit-ckpt", required=True)
    parser.add_argument("--head-ckpt", required=True)
    parser.add_argument("--prompts",
                        default="probe/prompts/probe_prompts.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("CUDA not available, using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device(args.device)

    # Load model and head
    model, model_config = load_cara_model(args.model_config, args.dit_ckpt)
    model = model.to(device).eval()
    head = load_attribution_head(args.head_ckpt, device)
    sample_size = model_config["sample_size"]

    # Load prompts and codeword registry
    with open(args.prompts) as f:
        probe_data = json.load(f)
    prompts = probe_data["prompts"]
    pool_meta = probe_data["meta"]["pools"]  # {pool_name: label_id}

    # Build codeword registry for decoding
    codeword_registry = {
        "pool_to_codeword": {
            name: defn["codeword"]
            for name, defn in POOL_DEFINITIONS.items()
        },
        "id_to_pool": {
            defn["label_id"]: name
            for name, defn in POOL_DEFINITIONS.items()
        },
    }

    collector = DiTHiddenStateCollector(model, layer_indices=[-1])
    
    results_per_prompt = []
    correct_top1 = 0
    correct_top3 = 0

    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(prompts, desc="Running attribution head", unit="prompt")
        
        for i, prompt_entry in enumerate(pbar):
            pbar.set_postfix({"pool": prompt_entry['pool']})

            collector.reset()
            conditioning = [{
                "prompt": prompt_entry["text"],
                "seconds_total": prompt_entry["seconds_total"],
            }]

            with collector:
                _ = generate_diffusion_cond(
                    model,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=device,
                    seed=42 + i,
                )

            # Extract hidden states and pass through attribution head
            agg = collector.get_aggregated_states(strategy="mean_across_timesteps")
            hidden = torch.tensor(agg[-1]).float().to(device).unsqueeze(0)
            logits = head(hidden)
            
            decoded = decode_cara_output(logits, codeword_registry)
            gt_pool = prompt_entry["pool"]
            
            top1_correct = decoded["top1_pool"] == gt_pool
            top3_correct = any(p == gt_pool for p, _ in decoded["top3_pools"])
            
            if top1_correct:
                correct_top1 += 1
            if top3_correct:
                correct_top3 += 1

            # Update progress with running accuracy
            current_acc_top1 = correct_top1 / (i + 1)
            current_acc_top3 = correct_top3 / (i + 1)
            pbar.set_postfix({
                "pool": prompt_entry['pool'],
                "top1": f"{current_acc_top1:.3f}",
                "top3": f"{current_acc_top3:.3f}"
            })

            results_per_prompt.append({
                "prompt_id": prompt_entry["prompt_id"],
                "gt_pool": gt_pool,
                "gt_label_id": prompt_entry["label_id"],
                "predicted_pool": decoded["top1_pool"],
                "top1_prob": decoded["top1_prob"],
                "top3_pools": decoded["top3_pools"],
                "cara_string": decoded["cara_string"],
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
            })

    # Aggregate results
    n = len(prompts)
    acc_top1 = correct_top1 / n
    acc_top3 = correct_top3 / n

    # Per-pool breakdown
    per_pool = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results_per_prompt:
        pool = r["gt_pool"]
        per_pool[pool]["total"] += 1
        if r["top1_correct"]:
            per_pool[pool]["correct"] += 1
    per_pool_acc = {
        pool: d["correct"] / d["total"]
        for pool, d in per_pool.items()
    }

    print(f"\n{'='*50}")
    print(f"ATTRIBUTION HEAD RESULTS (Run B)")
    print(f"{'='*50}")
    print(f"Top-1 accuracy: {acc_top1:.3f} ({acc_top1*100:.1f}%)")
    print(f"Top-3 accuracy: {acc_top3:.3f} ({acc_top3*100:.1f}%)")
    print(f"\nPer-pool Top-1 accuracy:")
    for pool, acc in per_pool_acc.items():
        print(f"  {pool:25s}  {acc:.3f} ({acc*100:.1f}%)")

    output = {
        "run": "B_attribution_head",
        "model_ckpt": args.dit_ckpt,
        "head_ckpt": args.head_ckpt,
        "n_prompts": n,
        "results": {
            "top1_accuracy": float(acc_top1),
            "top3_accuracy": float(acc_top3),
            "per_pool_top1": per_pool_acc,
        },
        "per_prompt": results_per_prompt,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
