# CARA Attribution Probe Suite

This directory contains benchmarking tools to measure how much pool attribution signal exists in:
1. The base Stable Audio model (implicit from metadata)
2. The CARA fine-tuned model with attribution head
3. The CARA fine-tuned model's hidden representations

## Overview

The probe suite implements the three-way comparison from the CARA Attribution Probe Plan:

| Run | Model | Method | What It Measures |
|-----|-------|--------|------------------|
| **A** | Base model | Linear probe on hidden states | Implicit pool signal from original training |
| **B** | CARA fine-tuned | Trained attribution head | Explicit pool signal from CARA training |
| **C** | CARA fine-tuned | Linear probe on hidden states | Whether fine-tuning restructured representations |

## Installation

### Prerequisites

```bash
# Install required packages
pip install scikit-learn numpy scipy matplotlib
pip install stable-audio-tools torch torchaudio

# For Stable Audio Open 1.0 (base model)
pip install huggingface_hub
```

### Model Requirements

1. **Base Model**: Downloaded automatically from HuggingFace
   - Model: `stabilityai/stable-audio-open-small`
   - Size: ~1.2GB

2. **CARA Fine-tuned Model**: You need to provide:
   - Model config JSON
   - DiT checkpoint (.safetensors)
   - Attribution head checkpoint (.pt)

## Quick Start

### Phase 0: Before Fine-Tuning (Run Now)

```bash
# 1. Generate probe prompts (only once)
python probe/00_build_probe_prompts.py \
    --n-per-pool 100 \
    --output probe/prompts/probe_prompts.json

# 2. Extract base model hidden states
python probe/01_extract_hidden_states.py \
    --model-source pretrained \
    --model-name stabilityai/stable-audio-open-small \
    --prompts probe/prompts/probe_prompts.json \
    --output-dir probe/results/base_hidden_states \
    --device cuda

# 3. Train linear probe on base states (Run A)
python probe/02_train_linear_probe.py \
    --states-dir probe/results/base_hidden_states \
    --output probe/results/reports/probe_base.json
```

### Phase 1: After CARA Fine-Tuning

```bash
# 4. Extract CARA model hidden states
python probe/01_extract_hidden_states.py \
    --model-source checkpoint \
    --model-config /path/to/cara_model_config.json \
    --ckpt-path /path/to/cara_dit.safetensors \
    --prompts probe/prompts/probe_prompts.json \
    --output-dir probe/results/cara_hidden_states

# 5. Train linear probe on CARA states (Run C)
python probe/02_train_linear_probe.py \
    --states-dir probe/results/cara_hidden_states \
    --output probe/results/reports/probe_cara_linear.json

# 6. Run attribution head (Run B)
python probe/03_run_attribution_head.py \
    --model-config /path/to/cara_model_config.json \
    --dit-ckpt /path/to/cara_dit.safetensors \
    --head-ckpt /path/to/attribution_head.pt \
    --prompts probe/prompts/probe_prompts.json \
    --output probe/results/reports/attribution_head.json

# 7. Generate comparison report
python probe/04_benchmark_report.py \
    --run-a probe/results/reports/probe_base.json \
    --run-b probe/results/reports/attribution_head.json \
    --run-c probe/results/reports/probe_cara_linear.json \
    --output-dir probe/results/reports
```

## Understanding Results

### Key Metrics

1. **Run A Accuracy**: How much the base model already "knows" about pools
   - < 30%: Minimal implicit signal
   - 30-55%: Moderate implicit signal  
   - > 55%: Strong implicit signal

2. **B - A Delta**: CARA's marginal contribution
   - This is the main number for your thesis
   - Shows what CARA adds beyond base model

3. **C - A Delta**: Representation restructuring
   - < 5pp: Minimal restructuring (head does the work)
   - > 20pp: Significant restructuring (codewords deeply integrated)

### Control-Token Confound

The probe addresses whether CARA codewords act as:
- **Attribution signals** (desired) ✓
- **Style controls** (confound) ✗

Low C-A delta with high B-A delta indicates proper attribution behavior.

## Output Files

```
probe/
├── prompts/
│   └── probe_prompts.json          # Fixed prompt set (never regenerate!)
├── results/
│   ├── base_hidden_states/         # Run A extracted features
│   ├── cara_hidden_states/         # Run C extracted features
│   └── reports/
│       ├── probe_base.json         # Run A results
│       ├── probe_cara_linear.json  # Run C results
│       ├── attribution_head.json   # Run B results
│       ├── benchmark_summary.json  # Machine-readable comparison
│       └── benchmark_summary.md    # Human-readable report
```

## GUI Integration

The benchmark results are displayed in the frontend at:
- Component: `gui/frontend/src/components/BenchmarkResults.tsx`
- Endpoint: `/api/benchmark/results`

## Device Support

The scripts auto-detect the best available device:
- **CUDA** (NVIDIA GPU): Fastest, used by default
- **MPS** (Apple Silicon): Auto-detected if CUDA unavailable
- **CPU**: Fallback, slowest

Override with `--device mps` or `--device cpu` if needed.

## Troubleshooting

### Out of Memory
- Reduce batch size in generation
- Use CPU with `--device cpu` (slower)
- Extract states for fewer prompts with `--n-per-pool 50`

### Model Architecture Mismatch
- Check `probe/utils/hooks.py` transformer block paths
- Print model structure to debug: `print(model)`

### Different Results Between Runs
- Ensure same `--steps` parameter across all runs
- Use same prompt file (don't regenerate)
- Check random seeds are fixed

## Citation

If using these probe tools in research:

```bibtex
@software{cara_probe_2026,
  title = {CARA Attribution Probe Suite},
  author = {Moran, Sam},
  year = {2026},
  institution = {Macquarie University}
}
```
