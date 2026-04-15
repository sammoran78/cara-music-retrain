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
from typing import Dict, List, Optional


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

    def __init__(self, model, layer_indices: List[int] = [-1]):
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
        self._states_per_timestep: List[dict] = []  # [{layer_idx: tensor}, ...]
        self._current_timestep_states: dict = {}

    def _get_transformer_blocks(self):
        """Navigate the model architecture to find DiT transformer blocks."""
        # stable-audio-tools DiT architecture path:
        # model.model -> ConditionedDiffusionModelWrapper
        # .model -> DiffusionTransformer (the DiT itself)
        # .transformer -> the actual transformer
        # .layers -> list of transformer blocks
        
        # Try the correct path based on model structure
        try:
            transformer = self.model.model.model.transformer
            return transformer.layers
        except AttributeError:
            # Fallback: print structure for debugging
            print("Model structure:")
            print(self.model)
            raise AttributeError(
                "Could not find transformer blocks. Inspect your model with print(model) and update the path in hooks.py."
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
    ) -> Dict[int, np.ndarray]:
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
