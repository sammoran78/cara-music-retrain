#!/usr/bin/env python3
"""
01_extract_hidden_states.py

Run all probe prompts through a model and save DiT hidden states to disk.
Run this on BOTH the base model and the CARA fine-tuned model.

Usage — Base model (Run A):
    python 01_extract_hidden_states.py \
        --model-source pretrained \
        --model-name stabilityai/stable-audio-open-small \
        --prompts probe/prompts/probe_prompts.json \
        --output-dir probe/results/base_hidden_states \
        --layer-indices -1 -2 -3 \
        --timestep-strategy mean_across_timesteps \
        --steps 100

Usage — CARA fine-tuned model (Run C):
    python 01_extract_hidden_states.py \
        --model-source checkpoint \
        --model-config /path/to/model_config.json \
        --ckpt-path checkpoints/dit_frozen_v1.safetensors \
        --prompts probe/prompts/probe_prompts.json \
        --output-dir probe/results/cara_hidden_states \
        --layer-indices -1 -2 -3 \
        --timestep-strategy mean_across_timesteps \
        --steps 100
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load HF_TOKEN from .env for gated model access
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val

# Enable HuggingFace download progress bars
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch

# Patch autocast before importing stable-audio-tools (it hardcodes "cuda")
if not torch.cuda.is_available():
    import torch.amp
    _orig_autocast = torch.amp.autocast
    class _CPUAutocast(_orig_autocast):
        def __init__(self, device_type="cuda", **kw):
            kw["enabled"] = False
            super().__init__("cpu", **kw)
    torch.amp.autocast = _CPUAutocast
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        torch.cuda.amp.autocast = _CPUAutocast

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from probe.utils.hooks import DiTHiddenStateCollector


def load_model(args):
    """Load model from pretrained HuggingFace or local checkpoint."""
    if args.model_source == "pretrained":
        print(f"Loading pretrained model: {args.model_name}")
        print("Downloading model (this may take a few minutes for first run)...")
        
        # Wrap with progress indication
        try:
            model, model_config = get_pretrained_model(args.model_name)
            print(" Model loaded successfully")
        except Exception as e:
            print(f" Failed to load model: {e}")
            raise
    elif args.model_source == "checkpoint":
        from stable_audio_tools.models.factory import create_model_from_config
        print(f"Loading model from checkpoint: {args.ckpt_path}")
        with open(args.model_config) as f:
            model_config = json.load(f)
        model = create_model_from_config(model_config)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(" Checkpoint loaded successfully")
    else:
        raise ValueError(f"Unknown model-source: {args.model_source}")

    return model, model_config


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_log_line(path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def save_partial_checkpoint(out_dir, all_label_ids, all_pool_names, all_prompt_ids,
                            all_states_by_layer, completed_count):
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "label_ids_partial.npy", np.array(all_label_ids))
    np.save(checkpoint_dir / "pool_names_partial.npy", np.array(all_pool_names, dtype=object))
    np.save(checkpoint_dir / "prompt_ids_partial.npy", np.array(all_prompt_ids, dtype=object))
    for layer_idx, states_list in all_states_by_layer.items():
        if not states_list:
            continue
        arr = np.stack(states_list, axis=0)
        np.save(checkpoint_dir / f"hidden_states_layer_{layer_idx}_partial.npy", arr)
    write_json(checkpoint_dir / "checkpoint_status.json", {
        "completed_prompts": completed_count,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def main():
    parser = argparse.ArgumentParser()
    # Model loading
    parser.add_argument("--model-source", choices=["pretrained", "checkpoint"],
                        required=True)
    parser.add_argument("--model-name", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--ckpt-path", default=None)
    # Probe config
    parser.add_argument("--prompts",
                        default="probe/prompts/probe_prompts.json")
    parser.add_argument("--output-dir",
                        default="probe/results/base_hidden_states")
    parser.add_argument("--layer-indices", nargs="+", type=int,
                        default=[-1, -2, -3],
                        help="Transformer block indices to hook. -1=last block.")
    parser.add_argument("--timestep-strategy",
                        choices=["mean_across_timesteps",
                                 "final_timestep",
                                 "midpoint_timestep"],
                        default="mean_across_timesteps")
    parser.add_argument("--steps", type=int, default=100,
                        help="Denoising steps (match your evaluation config)")
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, using CPU (stable-audio-tools hardcodes CUDA autocast)")
    else:
        device = torch.device(args.device)

    # Load model
    model, model_config = load_model(args)
    # Installed stable-audio-tools may not recognise "rf_denoiser"; alias it
    if getattr(model, "diffusion_objective", None) == "rf_denoiser":
        model.diffusion_objective = "rectified_flow"
    model = model.to(device).eval()
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    effective_cfg_scale = args.cfg_scale
    effective_sampler = "dpmpp-3m-sde" if model.diffusion_objective == "v" else "euler"
    if model.diffusion_objective != "v" and args.cfg_scale != 1.0:
        effective_cfg_scale = 1.0
        print("Using cfg_scale=1.0 for rectified-flow model to avoid installed stable-audio-tools CFG bug")

    # Load prompts
    with open(args.prompts) as f:
        probe_data = json.load(f)
    prompts = probe_data["prompts"]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts")
    print(f"Sampling with sampler={effective_sampler}, cfg_scale={effective_cfg_scale}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    heartbeat_path = out_dir / "heartbeat.json"
    prompt_log_path = out_dir / "prompt_progress.log"
    start_time = time.time()

    # Save run metadata
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "model_source": args.model_source,
            "model_name": getattr(args, "model_name", None),
            "ckpt_path": getattr(args, "ckpt_path", None),
            "layer_indices": args.layer_indices,
            "timestep_strategy": args.timestep_strategy,
            "steps": args.steps,
            "n_prompts": len(prompts),
            "checkpoint_every": args.checkpoint_every,
            "max_prompts": args.max_prompts,
        }, f, indent=2)
    write_json(progress_path, {
        "status": "running",
        "completed_prompts": 0,
        "total_prompts": len(prompts),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_prompt_id": None,
        "current_pool": None,
    })
    write_json(heartbeat_path, {
        "status": "running",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_prompts": 0,
        "total_prompts": len(prompts),
    })

    # Collect labels and states
    all_label_ids = []
    all_pool_names = []
    all_prompt_ids = []
    # One list per layer index
    all_states_by_layer = {idx: [] for idx in args.layer_indices}

    collector = DiTHiddenStateCollector(model, layer_indices=args.layer_indices)

    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(prompts, desc="Extracting hidden states", unit="prompt")
        
        for i, prompt_entry in enumerate(pbar):
            pbar.set_postfix({"pool": prompt_entry['pool']})
            write_json(heartbeat_path, {
                "status": "running",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_prompts": i,
                "total_prompts": len(prompts),
                "current_prompt_id": prompt_entry["prompt_id"],
                "current_pool": prompt_entry["pool"],
            })
            write_json(progress_path, {
                "status": "running",
                "completed_prompts": i,
                "total_prompts": len(prompts),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "elapsed_seconds": round(time.time() - start_time, 2),
                "current_prompt_id": prompt_entry["prompt_id"],
                "current_pool": prompt_entry["pool"],
                "last_completed_prompt_id": all_prompt_ids[-1] if all_prompt_ids else None,
                "last_completed_pool": all_pool_names[-1] if all_pool_names else None,
            })
            prompt_start = time.time()

            collector.reset()
            conditioning = [{
                "prompt": prompt_entry["text"],
                "seconds_total": prompt_entry["seconds_total"],
            }]

            # Run generation with hooks active
            with collector:
                _ = generate_diffusion_cond(
                    model,
                    steps=args.steps,
                    cfg_scale=effective_cfg_scale,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type=effective_sampler,
                    device=device,
                    seed=42 + i,  # fixed seed per prompt for reproducibility
                )

            # Collect aggregated states
            agg = collector.get_aggregated_states(
                strategy=args.timestep_strategy
            )
            for layer_idx in args.layer_indices:
                if layer_idx in agg:
                    # agg[layer_idx] shape: (1, hidden_dim) — squeeze batch dim
                    all_states_by_layer[layer_idx].append(
                        agg[layer_idx][0]  # shape: (hidden_dim,)
                    )

            all_label_ids.append(prompt_entry["label_id"])
            all_pool_names.append(prompt_entry["pool"])
            all_prompt_ids.append(prompt_entry["prompt_id"])

            prompt_duration = round(time.time() - prompt_start, 2)
            completed_count = i + 1
            append_log_line(
                prompt_log_path,
                f"completed {completed_count}/{len(prompts)} prompt_id={prompt_entry['prompt_id']} pool={prompt_entry['pool']} duration_s={prompt_duration}"
            )
            write_json(heartbeat_path, {
                "status": "running",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_prompts": completed_count,
                "total_prompts": len(prompts),
                "last_completed_prompt_id": prompt_entry["prompt_id"],
                "last_completed_pool": prompt_entry["pool"],
                "last_prompt_duration_s": prompt_duration,
            })
            write_json(progress_path, {
                "status": "running",
                "completed_prompts": completed_count,
                "total_prompts": len(prompts),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "elapsed_seconds": round(time.time() - start_time, 2),
                "last_completed_prompt_id": prompt_entry["prompt_id"],
                "last_completed_pool": prompt_entry["pool"],
                "last_prompt_duration_s": prompt_duration,
                "remaining_prompts": len(prompts) - completed_count,
            })
            print(
                f"[{completed_count}/{len(prompts)}] completed prompt_id={prompt_entry['prompt_id']} "
                f"pool={prompt_entry['pool']} duration_s={prompt_duration}",
                flush=True,
            )
            if args.checkpoint_every > 0 and completed_count % args.checkpoint_every == 0:
                save_partial_checkpoint(
                    out_dir,
                    all_label_ids,
                    all_pool_names,
                    all_prompt_ids,
                    all_states_by_layer,
                    completed_count,
                )
                append_log_line(prompt_log_path, f"checkpoint saved at {completed_count}/{len(prompts)}")

    # Save to disk as numpy arrays
    np.save(out_dir / "label_ids.npy", np.array(all_label_ids))
    np.save(out_dir / "pool_names.npy", np.array(all_pool_names))
    np.save(out_dir / "prompt_ids.npy", np.array(all_prompt_ids))

    for layer_idx, states_list in all_states_by_layer.items():
        arr = np.stack(states_list, axis=0)  # shape: (n_prompts, hidden_dim)
        layer_name = f"layer_{layer_idx}"
        np.save(out_dir / f"hidden_states_{layer_name}.npy", arr)
        print(f"  Saved {layer_name}: shape {arr.shape}")

    write_json(progress_path, {
        "status": "completed",
        "completed_prompts": len(prompts),
        "total_prompts": len(prompts),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "output_dir": str(out_dir),
    })
    write_json(heartbeat_path, {
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_prompts": len(prompts),
        "total_prompts": len(prompts),
    })
    append_log_line(prompt_log_path, f"run completed total_prompts={len(prompts)} elapsed_s={round(time.time() - start_time, 2)}")

    print(f"\nDone. States saved to {out_dir}")


if __name__ == "__main__":
    main()
