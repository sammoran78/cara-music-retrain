#!/usr/bin/env python3
"""
00_build_probe_prompts.py

Build a fixed, balanced set of prompts for the attribution probe.
Each prompt is labelled with its ground-truth pool.
This prompt set is used IDENTICALLY across all three runs (A, B, C).

Usage:
    python 00_build_probe_prompts.py \
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
# These match the Source-License-Genre structure from our pool schema
# ---------------------------------------------------------------------------

POOL_DEFINITIONS = {
    "Freesound-CC0-Electronic": {
        "codeword": "M-AA0001-01",
        "label_id": 0,
        "description": "CC0 licensed electronic music from Freesound",
        "prompt_templates": [
            "{genre} music loop. {bpm} BPM. CC0 license.",
            "electronic {subgenre} beat. public domain. {bpm} BPM.",
            "{genre} synthesizer loop. no copyright. {bpm} BPM.",
            "digital {subgenre} music. CC0. tempo {bpm}.",
            "{genre} production. royalty free. {bpm} beats per minute.",
        ],
        "genre_slots": ["electronic", "techno", "house", "synth", "EDM"],
        "subgenre_slots": ["ambient", "dance", "experimental", "minimal", "acid"],
        "bpm_slots": ["120", "128", "130", "140", "125", "135"],
    },
    "Freesound-CC-BY-Ambient": {
        "codeword": "M-BB0002-01",
        "label_id": 1,
        "description": "CC-BY licensed ambient sounds from Freesound",
        "prompt_templates": [
            "ambient {scene} soundscape. attribution required.",
            "{scene} field recording. CC BY license. atmospheric.",
            "environmental audio. {scene}. creative commons attribution.",
            "nature ambience. {scene}. CC-BY licensed.",
            "atmospheric {scene} recording. attribution license.",
        ],
        "scene_slots": [
            "forest", "ocean", "rain", "wind", "night",
            "morning", "underwater", "cave", "mountain", "desert"
        ],
    },
    "FMA-CC0-Jazz": {
        "codeword": "M-CC0003-01",
        "label_id": 2,
        "description": "CC0 licensed jazz from Free Music Archive",
        "prompt_templates": [
            "{style} jazz music. Free Music Archive. public domain.",
            "jazz {instrument} improvisation. FMA dataset. CC0.",
            "{style} jazz composition. no copyright. FMA.",
            "jazz ensemble. {style} style. public domain music.",
            "{instrument} jazz solo. Free Music Archive. CC0 license.",
        ],
        "style_slots": ["bebop", "smooth", "fusion", "traditional", "modern"],
        "instrument_slots": ["piano", "saxophone", "trumpet", "bass", "drums"],
    },
    "FMA-CC-BY-Classical": {
        "codeword": "M-DD0004-01",
        "label_id": 3,
        "description": "CC-BY licensed classical music from Free Music Archive",
        "prompt_templates": [
            "classical {ensemble} music. attribution required. FMA.",
            "{period} classical composition. CC BY license.",
            "{ensemble} performance. classical music. attribution.",
            "orchestral {mood} piece. Free Music Archive. CC-BY.",
            "classical {instrument} sonata. attribution license. FMA.",
        ],
        "ensemble_slots": ["orchestra", "quartet", "chamber", "symphony", "ensemble"],
        "period_slots": ["baroque", "romantic", "modern", "contemporary", "classical"],
        "mood_slots": ["dramatic", "peaceful", "energetic", "melancholic", "triumphant"],
        "instrument_slots": ["piano", "violin", "cello", "flute", "harpsichord"],
    },
}


def build_prompts(pool_defs: dict, n_per_pool: int, seed: int) -> list:
    """Generate n_per_pool prompts per pool, returning a flat list of dicts."""
    rng = random.Random(seed)
    prompts = []

    for pool_name, pool in pool_defs.items():
        for i in range(n_per_pool):
            template = rng.choice(pool["prompt_templates"])

            # Fill template slots
            text = template
            for slot_key in ["genre_slots", "subgenre_slots", "style_slots", 
                           "scene_slots", "instrument_slots", "ensemble_slots",
                           "period_slots", "mood_slots", "bpm_slots"]:
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
