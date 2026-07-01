# CARA Azure ML Test Prep

This folder defines the pre-training reproducibility layer for CARA audio
fine-tuning. It validates one shared source dataset before any model-specific
adapter or fine-tuning work begins.

The current fine-tuning preparation sequence, smoke defaults, prepared dataset
paths, and progress-monitoring policy are maintained in
`docs/cara_finetuning_runbook.md`.

The Stable Audio smoke trainer is defined in
`azureml/jobs/07_smoke_stable_audio_trainer.yml`. Run
`azureml/jobs/07a_stable_audio_trainer_preflight.yml` first; it uses CPU compute
to validate the deep Stable Audio training imports and wrapper construction
before spending H100 time. The smoke trainer itself is GPU-only and reads the
prepared dataset at
`azureml://datastores/ds_cara_raw_audio/paths/prepared/cara-strong-v0.4/`.
The trainer uses `env-stable-audio-tools:8` and expects Hugging Face gated-model
access to be supplied through the Azure ML workspace Key Vault secret
`hf-token`; the dashboard syncs this from local `.env` `HF_TOKEN` before
submission.

Shared Azure ML datastore input:

```text
azureml://datastores/ds_cara_raw_audio/paths/test-audio/
```

Expected mounted paths:

```text
data/freesound/
data/freesound_meta/test-manifest/tracks.csv
```

## CLI

Set the intended subscription before submitting jobs:

```bash
az account set --subscription 2aa6790f-891f-4a9b-8b7c-476ac25b0f82
```

Submit the CPU data-access phase:

```bash
az ml job create \
  --file azureml/jobs/01_data_access_test.yml \
  --resource-group rg-cara-audio-training-aue \
  --workspace-name rg-cara-audio-training-aue
```

GPU jobs use `gpu-smoke-h100` and must be explicitly confirmed before
submission. Do not use `gpu-fulltraining-h100` during test prep.

Register the dedicated model environments after the CPU and GPU sanity phases
pass:

```bash
az ml environment create \
  --file azureml/environments/env_musicgen_audiocraft.yml \
  --resource-group rg-cara-audio-training-aue \
  --workspace-name rg-cara-audio-training-aue

az ml environment create \
  --file azureml/environments/env_stable_audio_tools.yml \
  --resource-group rg-cara-audio-training-aue \
  --workspace-name rg-cara-audio-training-aue
```

Each job writes `report.json`, `report.txt`, and `metadata.json` to its Azure ML
output artifact folder. Failed pass conditions still write artifacts and then
exit non-zero so Azure job status remains meaningful.
