"""
Synthetic ground-truth dataset generator for CARA evaluation.

Creates a controlled multi-pool dataset where each pool has distinct, known
generative signatures to validate attribution faithfulness.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class SyntheticPoolGenerator:
    """Generate synthetic audio pools with known characteristics."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define synthetic pool characteristics
        self.pool_definitions = {
            "Freesound-CC0-Electronic": {
                "tempo_range": (120, 140),
                "key_signature": ["C", "G", "Am"],
                "instrumentation": ["synth", "drum_machine", "bass_synth"],
                "timbre": "bright",
                "energy": "high"
            },
            "Freesound-CC-BY-Ambient": {
                "tempo_range": (60, 80),
                "key_signature": ["Dm", "Em", "F"],
                "instrumentation": ["pad", "field_recording", "drone"],
                "timbre": "dark",
                "energy": "low"
            },
            "FMA-CC0-Jazz": {
                "tempo_range": (90, 120),
                "key_signature": ["Bb", "Eb", "F"],
                "instrumentation": ["piano", "bass", "drums", "sax"],
                "timbre": "warm",
                "energy": "medium"
            },
            "FMA-CC-BY-Classical": {
                "tempo_range": (60, 100),
                "key_signature": ["D", "A", "G"],
                "instrumentation": ["strings", "piano", "woodwinds"],
                "timbre": "rich",
                "energy": "variable"
            }
        }
    
    def generate_metadata(self, pool_name: str, num_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic metadata for a pool."""
        if pool_name not in self.pool_definitions:
            raise ValueError(f"Unknown pool: {pool_name}")
        
        pool_def = self.pool_definitions[pool_name]
        metadata = []
        
        for i in range(num_samples):
            tempo = random.uniform(*pool_def["tempo_range"])
            key = random.choice(pool_def["key_signature"])
            instruments = random.sample(
                pool_def["instrumentation"], 
                k=random.randint(1, len(pool_def["instrumentation"]))
            )
            
            sample_metadata = {
                "file_id": f"{pool_name}_{i:04d}",
                "pool": pool_name,
                "tempo": tempo,
                "key": key,
                "instruments": instruments,
                "timbre": pool_def["timbre"],
                "energy": pool_def["energy"],
                "duration": random.uniform(5.0, 30.0),
                "ground_truth_pool": pool_name,  # Known ground truth
                "synthetic": True
            }
            metadata.append(sample_metadata)
        
        return metadata
    
    def generate_mixed_samples(self, pools: list[tuple[str, float]], num_samples: int) -> list[dict[str, Any]]:
        """Generate samples that mix characteristics from multiple pools."""
        mixed_metadata = []
        
        for i in range(num_samples):
            # Weighted selection of pool characteristics
            pool_weights = {pool: weight for pool, weight in pools}
            primary_pool = random.choices(
                list(pool_weights.keys()), 
                weights=list(pool_weights.values())
            )[0]
            
            # Mix characteristics from all pools based on weights
            mixed_chars = {
                "file_id": f"mixed_{i:04d}",
                "ground_truth_pools": pool_weights,
                "primary_pool": primary_pool,
                "synthetic": True,
                "mixed": True
            }
            
            # Blend tempo based on weights
            tempo = 0
            for pool, weight in pools:
                pool_def = self.pool_definitions[pool]
                tempo += weight * random.uniform(*pool_def["tempo_range"])
            mixed_chars["tempo"] = tempo
            
            mixed_metadata.append(mixed_chars)
        
        return mixed_metadata


def create_synthetic_evaluation_dataset(
    output_dir: Path,
    samples_per_pool: int = 1000,
    mixed_samples: int = 500
) -> dict[str, Any]:
    """Create a complete synthetic evaluation dataset."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticPoolGenerator()
    
    # Generate pure pool samples
    all_metadata = []
    pool_stats = {}
    
    for pool_name in generator.pool_definitions:
        pool_metadata = generator.generate_metadata(pool_name, samples_per_pool)
        all_metadata.extend(pool_metadata)
        pool_stats[pool_name] = len(pool_metadata)
    
    # Generate mixed samples for multi-pool attribution testing
    mixed_pools = [
        [("Freesound-CC0-Electronic", 0.7), ("Freesound-CC-BY-Ambient", 0.3)],
        [("FMA-CC0-Jazz", 0.6), ("FMA-CC-BY-Classical", 0.4)],
        [("Freesound-CC0-Electronic", 0.5), ("FMA-CC0-Jazz", 0.5)],
    ]
    
    for pool_mix in mixed_pools:
        mixed_metadata = generator.generate_mixed_samples(
            pool_mix, 
            mixed_samples // len(mixed_pools)
        )
        all_metadata.extend(mixed_metadata)
    
    # Write dataset
    dataset_path = output_dir / "synthetic_dataset.json"
    with dataset_path.open("w") as f:
        json.dump({
            "metadata": all_metadata,
            "pool_definitions": generator.pool_definitions,
            "statistics": {
                "total_samples": len(all_metadata),
                "pure_samples": sum(pool_stats.values()),
                "mixed_samples": len([m for m in all_metadata if m.get("mixed", False)]),
                "pools": pool_stats
            }
        }, f, indent=2)
    
    # Write evaluation config
    eval_config = {
        "dataset": "synthetic_cara_evaluation",
        "version": "1.0",
        "evaluation_metrics": [
            "pool_attribution_accuracy",
            "multi_pool_weight_correlation",
            "control_token_confound_score",
            "attribution_stability"
        ],
        "ground_truth_available": True
    }
    
    config_path = output_dir / "evaluation_config.json"
    with config_path.open("w") as f:
        json.dump(eval_config, f, indent=2)
    
    return {
        "dataset_path": str(dataset_path),
        "config_path": str(config_path),
        "statistics": pool_stats
    }


if __name__ == "__main__":
    # Generate synthetic dataset for testing
    output_dir = Path("data/synthetic_evaluation")
    result = create_synthetic_evaluation_dataset(output_dir)
    print(f"Created synthetic dataset at: {result['dataset_path']}")
    print(f"Pool statistics: {result['statistics']}")
