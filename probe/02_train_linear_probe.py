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
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

    # Progress bar for cross-validation
    pbar = tqdm(skf.split(X, y), total=n_folds, desc="Cross-validating", unit="fold")
    
    for fold_idx, (train_idx, test_idx) in enumerate(pbar):
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
        
        # Update progress bar with current fold accuracy
        pbar.set_postfix({"acc": f"{acc:.3f}"})

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
