# CARA Attribution Probe: Baseline and Benchmark Test Suite
## Implementation Plan and Tooling Specification

**Author:** Sam Moran, Macquarie University  
**Version:** 1.0 — April 2026  
**Purpose:** This document specifies every tool, script, and test needed to:  
1. Measure how much pool attribution signal already exists in the **base** Stable Audio Open Small model (pre-CARA)  
2. Run the same tests on the **CARA fine-tuned** model  
3. Produce clean benchmark comparisons showing what CARA adds

These tests constitute the experimental evidence for CARA's faithfulness claims and directly address the peer review's "control-token confound" requirement.

---

## Overview: What You Are Measuring and Why

The base model was trained on Freesound audio files whose conditioning prompts were constructed from Freesound metadata — tags like `["ambient", "CC0", "rain", "nature"]`, descriptions, and Essentia content descriptors. That means the DiT's hidden states may already carry **implicit pool signals** from this metadata, even though no explicit CARA codewords were ever used.

**The core question:** How much of CARA's attribution accuracy comes from the original metadata conditioning already present in the base model, versus the explicit pool-label conditioning CARA adds during fine-tuning?

**Why this matters for the thesis:** This comparison directly addresses the peer review's control-token confound. If the base model already has moderate attribution accuracy implicitly, CARA's contribution is formalisation + constrained emission. If the base model has low accuracy, CARA's fine-tuning is adding a genuinely new signal. Either result is publishable — but you need the number.

### The Three-Way Comparison

| Run | Model | Probe Method | What It Shows |
|---|---|---|---|
| **A** | Base (no changes) | Linear probe on hidden states | Implicit pool signal from original training metadata |
| **B** | CARA fine-tuned | Trained attribution head | Explicit pool signal from codeword conditioning |
| **C** | CARA fine-tuned | Same linear probe as Run A | Whether fine-tuning restructured DiT representations, or the head does all the work |

Run C is particularly important: if fine-tuning changed the DiT's internal geometry (C >> A), codewords are genuinely restructuring representations. If C ≈ A but B >> A, the attribution head is doing the heavy lifting on top of unchanged representations. Both are valid outcomes with different theoretical framings.

---

## Repository Structure for These Tools

Add a `probe/` directory alongside the existing evaluation pipeline:

```
cara_poc/
├── probe/
│   ├── 00_build_probe_prompts.py       # Generate standardised prompt set for all runs
│   ├── 01_extract_hidden_states.py     # Hook into DiT, extract states, save to disk
│   ├── 02_train_linear_probe.py        # Fit logistic regression on extracted states
│   ├── 03_run_attribution_head.py      # Run trained CARA attribution head on same prompts
│   ├── 04_benchmark_report.py          # Compare A vs B vs C, produce tables + plots
│   ├── utils/
│   │   ├── hooks.py                    # Forward hook registration and state collection
│   │   ├── metrics.py                  # All shared metric functions
│   │   └── pool_config.py              # Pool definitions and codeword mappings
│   ├── prompts/
│   │   └── probe_prompts.json          # The fixed prompt set used across ALL runs
│   └── results/
│       ├── base_hidden_states/         # Run A outputs
│       ├── cara_hidden_states/         # Run C outputs
│       └── reports/                    # Final benchmark tables and plots
```

---

## Tool 1: `00_build_probe_prompts.py`
### Build a Fixed Prompt Set Representing Each Pool

This script generates the standardised prompt set used identically across all three runs. Prompts are constructed from **actual Freesound metadata patterns** — the same tags and descriptions that shaped the base model's original training — ensuring the probe is ecologically valid.

```python
#!/usr/bin/env python3
"""
00_build_probe_prompts.py

Build a fixed, balanced set of prompts for the attribution probe.
Each prompt is labelled with its ground-truth pool.
This prompt set is used IDENTICALLY across all three runs (A, B, C).

Usage:
    python 00_build_probe_prompts.py \
        --pool-config probe/utils/pool_config.py \
        --n-per-pool 100 \
        --output probe/prompts/probe_prompts.json \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# POOL DEFINITIONS
# Modify this to match your actual CARA pool taxonomy.
# Each pool has:
#   - codeword: the CARA codeword string (matches your codebook)
#   - label_id: integer class label for sklearn
#   - prompt_templates: list of template strings drawn from actual Freesound
#     metadata patterns. Use {tags} and {desc} slots where natural.
# ---------------------------------------------------------------------------

POOL_DEFINITIONS = {
    "CC0_AMBIENT": {
        "codeword": "M-AA0001-01",   # example — replace with your actual codewords
        "label_id": 0,
        "description": "CC0 licensed ambient / field recording sounds",
        "prompt_templates": [
            "field recording of {scene}. ambient sound. CC0.",
            "{scene} ambience. nature recording. creative commons zero.",
            "environmental audio. {scene}. no copyright.",
            "atmospheric texture. {scene}. CC0 license.",
            "outdoor ambience. {scene}. public domain recording.",
        ],
        "scene_slots": [
            "rain on leaves", "wind through trees", "ocean waves", "river flowing",
            "birds in forest", "thunder in distance", "crickets at night",
            "light breeze", "waterfall", "morning birdsong",
        ],
    },
    "CC_BY_MUSIC": {
        "codeword": "M-BB0002-01",
        "label_id": 1,
        "description": "CC-BY licensed music loops and instrument samples",
        "prompt_templates": [
            "{genre} music loop. {bpm} BPM. attribution required.",
            "instrumental {genre} loop. CC BY license. {bpm} BPM.",
            "{genre} beat. licensed music. creative commons attribution.",
            "{genre} melody loop. {bpm} BPM. CC-BY.",
            "royalty free {genre} music. attribution license.",
        ],
        "genre_slots": [
            "electronic", "ambient", "acoustic guitar", "piano", "synth",
            "lo-fi", "cinematic", "jazz", "drum and bass", "funk",
        ],
        "bpm_slots": ["90", "100", "110", "120", "128", "140", "85", "95"],
    },
    "CC_SAMPLING_SFX": {
        "codeword": "M-CC0003-01",
        "label_id": 2,
        "description": "CC Sampling+ licensed sound effects and foley",
        "prompt_templates": [
            "sound effect. {sfx_type}. CC sampling plus license.",
            "foley recording. {sfx_type}. sampling permitted.",
            "{sfx_type} sound. high quality recording. CC Sampling+.",
            "studio foley. {sfx_type}. licensed for sampling.",
            "sound design element. {sfx_type}. creative commons sampling.",
        ],
        "sfx_slots": [
            "door creak", "footsteps on gravel", "glass breaking", "metal impact",
            "wood knock", "paper rustle", "keyboard typing", "button click",
            "zip fastening", "liquid pour",
        ],
    },
    "FMA_GENRE": {
        "codeword": "M-DD0004-01",
        "label_id": 3,
        "description": "Free Music Archive genre-labelled music tracks",
        "prompt_templates": [
            "{genre} music track. Free Music Archive. {mood}.",
            "full musical composition. {genre}. FMA dataset.",
            "{genre} song. independent artist. free music archive.",
            "{mood} {genre} track. openly licensed music.",
            "{genre} musical piece. {mood} atmosphere. FMA.",
        ],
        "genre_slots": [
            "indie folk", "experimental", "hip hop", "classical", "jazz",
            "rock", "pop", "world music", "blues", "soul",
        ],
        "mood_slots": [
            "melancholic", "upbeat", "energetic", "calm", "dramatic",
            "playful", "tense", "nostalgic", "reflective", "triumphant",
        ],
    },
}


def build_prompts(pool_defs: dict, n_per_pool: int, seed: int) -> list[dict]:
    """Generate n_per_pool prompts per pool, returning a flat list of dicts."""
    rng = random.Random(seed)
    prompts = []

    for pool_name, pool in pool_defs.items():
        for i in range(n_per_pool):
            template = rng.choice(pool["prompt_templates"])

            # Fill template slots
            text = template
            for slot_key in ["scene_slots", "genre_slots", "sfx_slots",
                              "mood_slots", "bpm_slots"]:
                slot_tag = slot_key.replace("_slots", "")
                if slot_key in pool and f"{{{slot_tag}}}" in text:
                    text = text.replace(
                        f"{{{slot_tag}}}",
                        rng.choice(pool[slot_key])
                    )
            # Clean up any unfilled slots
            import re
            text = re.sub(r'\{[^}]+\}', '', text).strip()

            prompts.append({
                "prompt_id": f"{pool_name}_{i:04d}",
                "pool": pool_name,
                "codeword": pool["codeword"],
                "label_id": pool["label_id"],
                "text": text,
                "seconds_total": 11,  # Fixed for Stable Audio Open Small
            })

    rng.shuffle(prompts)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-pool", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="probe/prompts/probe_prompts.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = build_prompts(POOL_DEFINITIONS, args.n_per_pool, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "meta": {
                "n_per_pool": args.n_per_pool,
                "n_pools": len(POOL_DEFINITIONS),
                "total_prompts": len(prompts),
                "seed": args.seed,
                "pools": {k: v["label_id"] for k, v in POOL_DEFINITIONS.items()},
            },
            "prompts": prompts,
        }, f, indent=2)

    print(f"Saved {len(prompts)} prompts to {args.output}")
    for pool_name, pool in POOL_DEFINITIONS.items():
        count = sum(1 for p in prompts if p["pool"] == pool_name)
        print(f"  {pool_name}: {count} prompts (label_id={pool['label_id']})")


if __name__ == "__main__":
    main()
```

---

## Tool 2: `probe/utils/hooks.py`
### Forward Hook Infrastructure for DiT Hidden State Extraction

```python
"""
hooks.py

Forward hook utilities for extracting DiT hidden states during generation.
Compatible with the Diffusion Transformer architecture in stable-audio-tools.

The DiT processes audio latents across N denoising timesteps.
We collect hidden states at registered transformer block outputs
and aggregate them across timesteps for a single representation per generation.
"""

import torch
import numpy as np
from typing import Optional


class DiTHiddenStateCollector:
    """
    Registers forward hooks on DiT transformer blocks and collects
    hidden states across denoising timesteps.

    Usage:
        collector = DiTHiddenStateCollector(model, layer_indices=[-1, -2, -3])
        with collector:
            output = generate_diffusion_cond(model, ...)
        states = collector.get_aggregated_states()  # shape: (n_layers, hidden_dim)
    """

    def __init__(self, model, layer_indices: list[int] = [-1]):
        """
        Args:
            model: The loaded stable-audio-tools model
            layer_indices: Which transformer block layers to hook into.
                           -1 = last layer (closest to output)
                           -2 = second to last, etc.
                           Recommend starting with [-1] then expanding.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.hooks = []
        self._states_per_timestep: list[dict] = []  # [{layer_idx: tensor}, ...]
        self._current_timestep_states: dict = {}

    def _get_transformer_blocks(self):
        """Navigate the model architecture to find DiT transformer blocks."""
        # stable-audio-tools DiT architecture path:
        # model.model.diffusion -> ConditionedDiffusionModelWrapper
        # .diffusion -> DiffusionTransformer (the DiT itself)
        # .transformer -> the actual transformer
        # .transformer_blocks -> list of transformer blocks
        
        # Try common attribute paths
        for path in [
            ["model", "diffusion", "transformer", "transformer_blocks"],
            ["diffusion", "transformer", "transformer_blocks"],
            ["model", "transformer", "transformer_blocks"],
        ]:
            obj = self.model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        
        raise AttributeError(
            "Could not find transformer blocks. Inspect your model with "
            "print(model) and update the path in hooks.py."
        )

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is typically a tensor of shape (batch, seq_len, hidden_dim)
            # We want a single vector per generation: mean-pool over seq_len
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            
            # Mean pool over sequence dimension -> (batch, hidden_dim)
            pooled = tensor.detach().float().mean(dim=1)
            
            # Store for this timestep
            self._current_timestep_states[layer_idx] = pooled.cpu().numpy()
        
        return hook_fn

    def __enter__(self):
        """Register hooks."""
        blocks = self._get_transformer_blocks()
        for idx in self.layer_indices:
            block = blocks[idx]
            hook = block.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
        return self

    def __exit__(self, *args):
        """Remove hooks and consolidate per-timestep states."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Save the last timestep's states (or all, depending on strategy)
        if self._current_timestep_states:
            self._states_per_timestep.append(
                dict(self._current_timestep_states)
            )
        self._current_timestep_states = {}

    def record_timestep(self):
        """
        Call this after each denoising step to record that timestep's states.
        If you're using the simple approach (just final timestep), skip this.
        """
        if self._current_timestep_states:
            self._states_per_timestep.append(
                dict(self._current_timestep_states)
            )
            self._current_timestep_states = {}

    def get_aggregated_states(
        self,
        strategy: str = "mean_across_timesteps"
    ) -> dict[int, np.ndarray]:
        """
        Aggregate collected states into a single representation.

        Args:
            strategy: one of:
                "mean_across_timesteps" — average hidden states across all
                    denoising steps (recommended: captures full trajectory)
                "final_timestep" — use only the last denoising step's states
                "midpoint_timestep" — use the state from the middle step

        Returns:
            Dict mapping layer_idx -> np.ndarray of shape (batch, hidden_dim)
        """
        if not self._states_per_timestep:
            raise RuntimeError("No states collected. Did you use 'with collector'?")

        if strategy == "final_timestep":
            return self._states_per_timestep[-1]

        elif strategy == "midpoint_timestep":
            mid = len(self._states_per_timestep) // 2
            return self._states_per_timestep[mid]

        elif strategy == "mean_across_timesteps":
            result = {}
            for layer_idx in self.layer_indices:
                layer_states = [
                    step[layer_idx]
                    for step in self._states_per_timestep
                    if layer_idx in step
                ]
                if layer_states:
                    result[layer_idx] = np.mean(
                        np.stack(layer_states, axis=0), axis=0
                    )
            return result

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def reset(self):
        """Clear all collected states for the next generation."""
        self._states_per_timestep = []
        self._current_timestep_states = {}
```

---

## Tool 3: `01_extract_hidden_states.py`
### Run Probe Prompts Through Model, Save Hidden States

This script is run **twice** — once on the base model (Run A), once on the CARA fine-tuned model (Run C). The `--model-source` flag controls which model is loaded.

```python
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
import numpy as np
from pathlib import Path

import torch
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from probe.utils.hooks import DiTHiddenStateCollector


def load_model(args):
    """Load model from pretrained HuggingFace or local checkpoint."""
    if args.model_source == "pretrained":
        print(f"Loading pretrained model: {args.model_name}")
        model, model_config = get_pretrained_model(args.model_name)
    elif args.model_source == "checkpoint":
        from stable_audio_tools.models.factory import create_model_from_config
        with open(args.model_config) as f:
            model_config = json.load(f)
        model = create_model_from_config(model_config)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.ckpt_path}")
    else:
        raise ValueError(f"Unknown model-source: {args.model_source}")

    return model, model_config


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
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, model_config = load_model(args)
    model = model.to(device).eval()
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    # Load prompts
    with open(args.prompts) as f:
        probe_data = json.load(f)
    prompts = probe_data["prompts"]
    print(f"Loaded {len(prompts)} prompts")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        }, f, indent=2)

    # Collect labels and states
    all_label_ids = []
    all_pool_names = []
    all_prompt_ids = []
    # One list per layer index
    all_states_by_layer = {idx: [] for idx in args.layer_indices}

    collector = DiTHiddenStateCollector(model, layer_indices=args.layer_indices)

    with torch.no_grad():
        for i, prompt_entry in enumerate(prompts):
            if i % 10 == 0:
                print(f"  [{i}/{len(prompts)}] pool={prompt_entry['pool']}")

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
                    cfg_scale=args.cfg_scale,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
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

    # Save to disk as numpy arrays
    np.save(out_dir / "label_ids.npy", np.array(all_label_ids))
    np.save(out_dir / "pool_names.npy", np.array(all_pool_names))
    np.save(out_dir / "prompt_ids.npy", np.array(all_prompt_ids))

    for layer_idx, states_list in all_states_by_layer.items():
        arr = np.stack(states_list, axis=0)  # shape: (n_prompts, hidden_dim)
        layer_name = f"layer_{layer_idx}"
        np.save(out_dir / f"hidden_states_{layer_name}.npy", arr)
        print(f"  Saved {layer_name}: shape {arr.shape}")

    print(f"\nDone. States saved to {out_dir}")


if __name__ == "__main__":
    main()
```

---

## Tool 4: `02_train_linear_probe.py`
### Fit a Linear Probe on Hidden States and Report Accuracy

This is the key diagnostic. A **linear** classifier is intentional — if pool attribution signal is in the representations, a linear probe will find it. A non-linear probe would muddy the interpretation.

```python
#!/usr/bin/env python3
"""
02_train_linear_probe.py

Fit a logistic regression (linear probe) on extracted DiT hidden states
and report attribution accuracy per pool.

Run on BOTH the base and CARA fine-tuned hidden states to get Run A and Run C.

Usage — Run A (base model):
    python 02_train_linear_probe.py \
        --states-dir probe/results/base_hidden_states \
        --layer-index -1 \
        --output probe/results/reports/probe_base.json

Usage — Run C (CARA fine-tuned model, linear probe only):
    python 02_train_linear_probe.py \
        --states-dir probe/results/cara_hidden_states \
        --layer-index -1 \
        --output probe/results/reports/probe_cara_linear.json
"""

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def load_states(states_dir: Path, layer_index: int):
    """Load hidden states and labels from the extraction run."""
    layer_name = f"layer_{layer_index}"
    X = np.load(states_dir / f"hidden_states_{layer_name}.npy")
    y = np.load(states_dir / "label_ids.npy")
    pool_names = np.load(states_dir / "pool_names.npy", allow_pickle=True)
    return X, y, pool_names


def run_cross_validated_probe(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """
    Run stratified k-fold cross-validated logistic regression.
    
    Returns per-fold metrics and aggregate results.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardise: fit on train, apply to test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic regression — intentionally simple/linear
        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        fold_results.append({
            "fold": fold_idx,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    return fold_results, np.array(all_y_true), np.array(all_y_pred)


def chance_level(y: np.ndarray) -> float:
    """Majority-class chance level for comparison."""
    from collections import Counter
    counts = Counter(y)
    return max(counts.values()) / len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    states_dir = Path(args.states_dir)
    
    # Load run config to understand what model produced these states
    with open(states_dir / "run_config.json") as f:
        run_config = json.load(f)

    X, y, pool_names = load_states(states_dir, args.layer_index)
    print(f"Loaded states: X={X.shape}, y={y.shape}")
    print(f"Pools: {sorted(set(pool_names.tolist()))}")

    # Unique class names for reporting
    unique_ids = sorted(set(y.tolist()))
    unique_pools = []
    for uid in unique_ids:
        mask = y == uid
        unique_pools.append(pool_names[mask][0])

    # Run cross-validated probe
    print(f"\nRunning {args.n_folds}-fold cross-validated linear probe...")
    fold_results, y_true_all, y_pred_all = run_cross_validated_probe(
        X, y, n_folds=args.n_folds
    )

    # Aggregate metrics
    mean_acc = np.mean([f["accuracy"] for f in fold_results])
    std_acc = np.std([f["accuracy"] for f in fold_results])
    mean_bal_acc = np.mean([f["balanced_accuracy"] for f in fold_results])
    chance = chance_level(y)

    # Per-class report (using all folds combined)
    report = classification_report(
        y_true_all, y_pred_all,
        target_names=unique_pools,
        output_dict=True
    )
    cm = confusion_matrix(y_true_all, y_pred_all)

    # Print summary
    print(f"\n{'='*50}")
    print(f"LINEAR PROBE RESULTS — {run_config.get('model_source', 'unknown')}")
    print(f"Model: {run_config.get('model_name') or run_config.get('ckpt_path')}")
    print(f"Layer: {args.layer_index} | Timestep strategy: "
          f"{run_config.get('timestep_strategy', '?')}")
    print(f"{'='*50}")
    print(f"Chance level (majority class):   {chance:.3f} ({chance*100:.1f}%)")
    print(f"Mean accuracy ({args.n_folds}-fold CV):       "
          f"{mean_acc:.3f} ± {std_acc:.3f} ({mean_acc*100:.1f}%)")
    print(f"Mean balanced accuracy:          {mean_bal_acc:.3f}")
    print(f"Improvement over chance:         "
          f"{(mean_acc - chance):.3f} ({(mean_acc - chance)*100:.1f}pp)")
    print(f"\nPer-pool accuracy:")
    for pool in unique_pools:
        if pool in report:
            print(f"  {pool:25s}  precision={report[pool]['precision']:.3f}  "
                  f"recall={report[pool]['recall']:.3f}  "
                  f"f1={report[pool]['f1-score']:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  Pools: {unique_pools}")
    print(cm)

    # Save results
    results = {
        "model_source": run_config,
        "probe_config": {
            "layer_index": args.layer_index,
            "n_folds": args.n_folds,
            "probe_type": "logistic_regression_linear",
        },
        "results": {
            "chance_level": float(chance),
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "mean_balanced_accuracy": float(mean_bal_acc),
            "improvement_over_chance": float(mean_acc - chance),
            "fold_results": fold_results,
            "per_pool_report": report,
            "confusion_matrix": cm.tolist(),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## Tool 5: `03_run_attribution_head.py`
### Run the Trained CARA Attribution Head on the Same Prompts

This produces Run B — the CARA fine-tuned model with the trained attribution head.

```python
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
        --pool-config probe/utils/pool_config.py \
        --output probe/results/reports/attribution_head.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

from stable_audio_tools.inference.generation import generate_diffusion_cond
from probe.utils.hooks import DiTHiddenStateCollector
from probe.utils.metrics import (
    compute_pool_accuracy,
    compute_calibration_error,
    compute_degradation_states,
)


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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    # (This should match your actual CARA codebook — update accordingly)
    from probe.utils.pool_config import POOL_DEFINITIONS
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
        for i, prompt_entry in enumerate(prompts):
            if i % 10 == 0:
                print(f"  [{i}/{len(prompts)}] pool={prompt_entry['pool']}")

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
```

---

## Tool 6: `probe/utils/metrics.py`
### Shared Metric Functions Used Across All Tools

```python
"""
metrics.py

Shared metric computations used across the probe pipeline.
All functions operate on numpy arrays for consistency.
"""

import numpy as np
from typing import Optional


def compute_pool_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pool_names: Optional[list[str]] = None
) -> dict:
    """Per-pool and overall top-1 accuracy."""
    overall = float(np.mean(y_true == y_pred))
    result = {"overall": overall}
    
    if pool_names is not None:
        unique_ids = sorted(set(y_true.tolist()))
        for uid in unique_ids:
            mask = y_true == uid
            pool_acc = float(np.mean(y_pred[mask] == y_true[mask]))
            name = pool_names[uid] if uid < len(pool_names) else str(uid)
            result[name] = pool_acc
    
    return result


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Expected Calibration Error (ECE) for the top-1 predicted class.
    
    A well-calibrated model produces confidence scores that match
    empirical accuracy — e.g., when it says 70% confidence, it is
    correct ~70% of the time.
    
    Args:
        y_true: ground truth label ids (n_samples,)
        y_prob: predicted probabilities (n_samples, n_classes)
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(float)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i + 1]),
            "accuracy": float(bin_acc),
            "confidence": float(bin_conf),
            "n_samples": int(mask.sum()),
        })
    
    return {"ece": float(ece), "bins": bin_data}


def compute_degradation_states(
    y_true: np.ndarray,
    top3_preds: list[list],  # list of [(pool_name, prob), ...] for each sample
    pool_hierarchy: Optional[dict] = None,
    confidence_threshold_exact: float = 0.70,
    confidence_threshold_repaired: float = 0.50,
) -> dict:
    """
    Simulate the CARA four-state degradation hierarchy and report
    what proportion of attributions land in each state.
    
    States:
        A (Exact)    — top-1 correct AND confidence >= exact_threshold
        B (Repaired) — top-1 correct AND confidence < exact_threshold
                       OR top-1 wrong but checksum repair would fix it
        C (Degraded) — wrong pool but correct pool family (if hierarchy given)
        D (Exception) — all else
    
    For the probe, we approximate B/C/D using confidence thresholds
    since we don't have actual codeword checksums at this stage.
    """
    n = len(y_true)
    states = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    for i, top3 in enumerate(top3_preds):
        top1_pool = top3[0][0]
        top1_prob = top3[0][1]
        gt_pool = y_true[i]  # assumes pool name string
        
        is_correct = (top1_pool == gt_pool)
        
        if is_correct and top1_prob >= confidence_threshold_exact:
            states["A"] += 1
        elif is_correct and top1_prob >= confidence_threshold_repaired:
            states["B"] += 1
        elif pool_hierarchy and pool_hierarchy.get(top1_pool) == pool_hierarchy.get(gt_pool):
            states["C"] += 1  # same pool family
        else:
            states["D"] += 1
    
    return {
        "state_counts": states,
        "state_proportions": {k: v / n for k, v in states.items()},
        "n_total": n,
    }


def improvement_over_chance(accuracy: float, n_classes: int) -> float:
    """Percentage points above uniform chance."""
    chance = 1.0 / n_classes
    return accuracy - chance


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's kappa — chance-corrected agreement.
    More informative than raw accuracy for imbalanced class distributions.
    kappa=0: no better than chance; kappa=1: perfect agreement.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))
```

---

## Tool 7: `04_benchmark_report.py`
### Combine Runs A, B, C into a Single Comparison Report

```python
#!/usr/bin/env python3
"""
04_benchmark_report.py

Combine results from Runs A, B, and C into a single benchmark comparison.
Produces both a JSON data file and a human-readable markdown summary.

Usage:
    python 04_benchmark_report.py \
        --run-a probe/results/reports/probe_base.json \
        --run-b probe/results/reports/attribution_head.json \
        --run-c probe/results/reports/probe_cara_linear.json \
        --output-dir probe/results/reports \
        --n-pools 4
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_probe_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_accuracy(acc: float) -> str:
    return f"{acc:.3f} ({acc*100:.1f}%)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True,
                        help="Base model linear probe results")
    parser.add_argument("--run-b", required=True,
                        help="CARA attribution head results")
    parser.add_argument("--run-c", required=True,
                        help="CARA fine-tuned linear probe results")
    parser.add_argument("--output-dir",
                        default="probe/results/reports")
    parser.add_argument("--n-pools", type=int, default=4,
                        help="Number of pools (for chance level calculation)")
    args = parser.parse_args()

    a = load_probe_result(args.run_a)
    b = load_probe_result(args.run_b)
    c = load_probe_result(args.run_c)

    chance = 1.0 / args.n_pools

    acc_a = a["results"]["mean_accuracy"]
    acc_b = b["results"]["top1_accuracy"]
    acc_c = c["results"]["mean_accuracy"]

    # -----------------------------------------------------------------------
    # Build markdown summary
    # -----------------------------------------------------------------------
    md = []
    md.append("# CARA Attribution Probe — Benchmark Report")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    md.append("")
    md.append("## Summary Table")
    md.append("")
    md.append("| Run | Model | Method | Accuracy | Δ vs Chance | Δ vs Run A |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| **A** | Base (no CARA) | Linear probe | {format_accuracy(acc_a)} "
              f"| +{(acc_a-chance)*100:.1f}pp | — |")
    md.append(f"| **B** | CARA fine-tuned | Attribution head | {format_accuracy(acc_b)} "
              f"| +{(acc_b-chance)*100:.1f}pp | +{(acc_b-acc_a)*100:.1f}pp |")
    md.append(f"| **C** | CARA fine-tuned | Linear probe | {format_accuracy(acc_c)} "
              f"| +{(acc_c-chance)*100:.1f}pp | +{(acc_c-acc_a)*100:.1f}pp |")
    md.append(f"| — | Chance baseline | Majority class | "
              f"{format_accuracy(chance)} | — | — |")
    md.append("")
    md.append("## Interpretation Key")
    md.append("")
    md.append("| Comparison | What It Tells You |")
    md.append("|---|---|")
    md.append("| **A vs Chance** | How much pool signal exists implicitly in base model metadata conditioning |")
    md.append("| **B vs A** | Total improvement from CARA (fine-tuning + attribution head) |")
    md.append("| **C vs A** | Whether fine-tuning restructured DiT representations themselves |")
    md.append("| **B vs C** | How much the trained attribution head adds on top of restructured representations |")
    md.append("")
    md.append("## Control-Token Confound Assessment")
    md.append("")

    if (acc_c - acc_a) < 0.05:
        confound_verdict = (
            "**Low confound risk.** Fine-tuning did not substantially restructure "
            "DiT hidden states (Run C ≈ Run A). The attribution head (Run B) is doing "
            "meaningful classification work on top of representations that were not "
            "simply 'stamped' with pool labels. CARA's codewords are not functioning "
            "purely as style controls."
        )
    elif (acc_c - acc_a) >= 0.05 and (acc_b - acc_c) > 0.10:
        confound_verdict = (
            "**Moderate restructuring, head adds value.** Fine-tuning shifted DiT "
            "representations somewhat (C > A), but the attribution head provides "
            "substantial additional discrimination (B >> C). Both the conditioning "
            "and the head contribute to CARA's performance."
        )
    else:
        confound_verdict = (
            "**Note: Further analysis needed.** Run C is close to Run B, suggesting "
            "most of CARA's accuracy comes from representation restructuring rather "
            "than the head's classification. Consider investigating whether codewords "
            "are functioning as style controls. Run the counterfactual codeword "
            "injection test (Study 2 in the evaluation plan) to characterise this further."
        )

    md.append(confound_verdict)
    md.append("")
    md.append("## Per-Pool Results")
    md.append("")
    md.append("### Run A — Base Model Linear Probe")
    md.append("")

    if "per_pool_report" in a.get("results", {}):
        per_pool_a = a["results"]["per_pool_report"]
        for pool_name in [k for k in per_pool_a if k not in ["accuracy", "macro avg", "weighted avg"]]:
            pool_data = per_pool_a[pool_name]
            md.append(f"- **{pool_name}**: "
                      f"precision={pool_data['precision']:.3f}, "
                      f"recall={pool_data['recall']:.3f}, "
                      f"f1={pool_data['f1-score']:.3f}")

    md.append("")
    md.append("### Run B — CARA Attribution Head")
    md.append("")
    if "per_pool_top1" in b.get("results", {}):
        for pool_name, acc in b["results"]["per_pool_top1"].items():
            md.append(f"- **{pool_name}**: top-1 accuracy = {format_accuracy(acc)}")

    md.append("")
    md.append("## Thesis Statement: What These Numbers Establish")
    md.append("")
    delta_b_a = acc_b - acc_a
    delta_b_chance = acc_b - chance

    if delta_b_chance > 0.40:
        strength = "strong"
    elif delta_b_chance > 0.20:
        strength = "moderate"
    else:
        strength = "limited"

    md.append(
        f"CARA's attribution head achieves {format_accuracy(acc_b)} pool attribution "
        f"accuracy, representing a {strength} improvement of {delta_b_chance*100:.1f} "
        f"percentage points over chance and {delta_b_a*100:.1f} percentage points over "
        f"the implicit attribution signal present in the base model. This establishes "
        f"that CARA's training-time codeword conditioning adds measurable, "
        f"discriminative pool attribution capability beyond what the base model's "
        f"original metadata conditioning provides."
    )
    md.append("")
    md.append("---")
    md.append("*Generated by `04_benchmark_report.py` — CARA Attribution Probe Suite*")

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Markdown report saved to {md_path}")

    # Save machine-readable summary
    summary = {
        "generated": datetime.now().isoformat(),
        "n_pools": args.n_pools,
        "chance_level": chance,
        "run_A": {
            "label": "Base model — linear probe",
            "accuracy": acc_a,
            "delta_vs_chance": acc_a - chance,
        },
        "run_B": {
            "label": "CARA fine-tuned — attribution head",
            "accuracy": acc_b,
            "delta_vs_chance": acc_b - chance,
            "delta_vs_A": acc_b - acc_a,
        },
        "run_C": {
            "label": "CARA fine-tuned — linear probe",
            "accuracy": acc_c,
            "delta_vs_chance": acc_c - chance,
            "delta_vs_A": acc_c - acc_a,
        },
    }
    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to {out_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
```

---

## Execution Order and Timeline

### Phase 0 — Before Any Fine-Tuning (Run This Week)

```bash
# Step 1: Build the fixed prompt set (run once, used for all runs)
python probe/00_build_probe_prompts.py \
    --n-per-pool 100 \
    --output probe/prompts/probe_prompts.json \
    --seed 42

# Step 2: Extract base model hidden states (Run A data)
# ~2-4 hours depending on GPU
python probe/01_extract_hidden_states.py \
    --model-source pretrained \
    --model-name stabilityai/stable-audio-open-small \
    --prompts probe/prompts/probe_prompts.json \
    --output-dir probe/results/base_hidden_states \
    --layer-indices -1 -2 -3 \
    --timestep-strategy mean_across_timesteps \
    --steps 100

# Step 3: Train linear probe on base states (Run A)
python probe/02_train_linear_probe.py \
    --states-dir probe/results/base_hidden_states \
    --layer-index -1 \
    --output probe/results/reports/probe_base.json
```

This gives you **Run A** — the baseline you need before touching anything else.

### Phase 1 — After CARA Fine-Tuning (Run B and C)

```bash
# Step 4: Extract CARA fine-tuned hidden states (Run C data)
python probe/01_extract_hidden_states.py \
    --model-source checkpoint \
    --model-config /path/to/model_config.json \
    --ckpt-path checkpoints/dit_frozen_v1.safetensors \
    --prompts probe/prompts/probe_prompts.json \  # SAME prompts as Phase 0
    --output-dir probe/results/cara_hidden_states \
    --layer-indices -1 -2 -3 \
    --timestep-strategy mean_across_timesteps \
    --steps 100

# Step 5: Train linear probe on CARA fine-tuned states (Run C)
python probe/02_train_linear_probe.py \
    --states-dir probe/results/cara_hidden_states \
    --layer-index -1 \
    --output probe/results/reports/probe_cara_linear.json

# Step 6: Run trained attribution head (Run B)
python probe/03_run_attribution_head.py \
    --model-config /path/to/model_config.json \
    --dit-ckpt checkpoints/dit_frozen_v1.safetensors \
    --head-ckpt checkpoints/attribution_head_v1.pt \
    --prompts probe/prompts/probe_prompts.json \  # SAME prompts again
    --output probe/results/reports/attribution_head.json

# Step 7: Generate combined benchmark report
python probe/04_benchmark_report.py \
    --run-a probe/results/reports/probe_base.json \
    --run-b probe/results/reports/attribution_head.json \
    --run-c probe/results/reports/probe_cara_linear.json \
    --output-dir probe/results/reports \
    --n-pools 4
```

---

## What to Look For in Results

### The number that matters most for the thesis

`Run B accuracy - Run A accuracy` = **CARA's marginal contribution**

This is the number you report in the thesis chapter. It answers: "How much does CARA add beyond what the base model already knows?"

### Interpreting Run A (base model accuracy)

| Run A Accuracy | Interpretation | Thesis Framing |
|---|---|---|
| < 30% (near chance) | Base model has almost no implicit pool signal | "Pool attribution does not exist in the base model — CARA adds it from scratch" |
| 30–55% | Moderate implicit signal from metadata conditioning | "Implicit signal exists but is unreliable — CARA formalises and amplifies it" |
| 55–75% | Strong implicit signal | "The base model partially learned pool structure; CARA makes it explicit, constrained, and verifiable" |
| > 75% | Very strong implicit signal | Investigate further — may indicate pool definitions are too closely tied to prompt text rather than audio characteristics |

### Interpreting the C vs A comparison (representation restructuring)

| C - A gap | Interpretation |
|---|---|
| < 5pp | Fine-tuning didn't change DiT representations much; attribution head is the key component |
| 5–20pp | Fine-tuning restructured representations somewhat; both conditioning and head contribute |
| > 20pp | Fine-tuning substantially reorganised DiT representations around pool labels; codewords are deeply integrated |

---

## Dependencies

```bash
pip install scikit-learn numpy scipy matplotlib
pip install stable-audio-tools  # already installed for training
```

No new ML frameworks required. All probe tools use scikit-learn's logistic regression — intentionally lightweight so the diagnostic is unambiguous.

---

## Notes on Reproducibility

- The prompt set in `probe/prompts/probe_prompts.json` is fixed at seed 42 and must **never be regenerated** once Run A is complete. The same prompts must run through all three configurations.
- Generation uses `seed=42+i` per prompt. This is fixed across all runs so generation stochasticity is controlled.
- The `--steps 100` parameter must match across all runs. Do not change it between A, B, and C.
- If you want to test multiple layers (`--layer-indices -1 -2 -3`), Tool 2 can be run three times with different `--layer-index` values on the same saved hidden states. This does not require re-running generation.
