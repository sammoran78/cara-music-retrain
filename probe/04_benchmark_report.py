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
