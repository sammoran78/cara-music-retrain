"""
Probe utilities for CARA attribution benchmarking.

Note: hooks.py requires torch — import DiTHiddenStateCollector directly
from probe.utils.hooks when needed.
"""

from .metrics import (
    compute_pool_accuracy,
    compute_calibration_error,
    compute_degradation_states,
    improvement_over_chance,
    cohens_kappa,
)
from .pool_config import (
    POOL_DEFINITIONS,
    POOL_HIERARCHY,
    CODEWORD_TO_POOL,
    LABEL_ID_TO_POOL,
    POOL_TO_CODEWORD,
    POOL_TO_LABEL_ID,
)

__all__ = [
    "compute_pool_accuracy",
    "compute_calibration_error",
    "compute_degradation_states",
    "improvement_over_chance",
    "cohens_kappa",
    "POOL_DEFINITIONS",
    "POOL_HIERARCHY",
    "CODEWORD_TO_POOL",
    "LABEL_ID_TO_POOL",
    "POOL_TO_CODEWORD",
    "POOL_TO_LABEL_ID",
]
