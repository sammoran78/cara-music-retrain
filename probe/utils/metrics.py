"""
metrics.py

Shared metric computations used across the probe pipeline.
All functions operate on numpy arrays for consistency.
"""

import numpy as np
from typing import Dict, List, Optional


def compute_pool_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pool_names: Optional[List[str]] = None
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
    top3_preds: List[list],  # list of [(pool_name, prob), ...] for each sample
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
