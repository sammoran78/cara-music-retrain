# Azure ML Test-Prep Evidence

This note records the reusable evidence for the CARA Azure ML pre-training validation layer.
Completed phases should not be rerun unless the underlying compute, environment, or shared
dataset changes.

## Phase Summary

| Phase | Result | Azure ML job | Compute | Reuse decision |
| --- | --- | --- | --- | --- |
| 01 Data Access | PASS | `strong_parrot_l70v7x8w6x` | `cpu-prep-cluster` | Reuse as the current repaired-manifest private-datastore access proof. |
| 02 GPU Sanity | PASS | `upbeat_shirt_3xzkjyzdvd` | `gpu-smoke-h100` | Reuse as the current H100 CUDA proof. Do not rerun for phases 03 or 04. |
| 03 MusicGen Environment | PASS | `wheat_dog_0wh3fqkljk` | `gpu-smoke-h100` | Reuse as the current AudioCraft import, H100 CUDA, and shared-dataset proof. |
| 04 Stable Audio Environment | PASS | `tough_kite_mmmxk4yy9p` | `gpu-smoke-h100` | Reuse as the current Stable Audio Tools import, H100 CUDA, and shared-dataset proof. |

## Phase 02 GPU Sanity Result

The dashboard-triggered Azure ML command job `upbeat_shirt_3xzkjyzdvd` completed successfully.

| Field | Value |
| --- | --- |
| Test | `02_gpu_sanity_test` |
| Status | `passed` |
| PyTorch | `2.2.2` |
| CUDA available | `true` |
| CUDA device count | `1` |
| GPU | `NVIDIA H100 NVL` |
| Curated environment | `azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/10` |
| Azure ML Studio | [Open run](https://ml.azure.com/runs/upbeat_shirt_3xzkjyzdvd?wsid=/subscriptions/2aa6790f-891f-4a9b-8b7c-476ac25b0f82/resourcegroups/rg-cara-audio-training-aue/workspaces/rg-cara-audio-training-aue&tid=776c987b-2fd1-4464-9928-8f076d411ebc) |

The complete downloaded Azure bundle is preserved at:

`registry/azureml_test_prep/artifacts/upbeat_shirt_3xzkjyzdvd/`

This includes Azure system logs, user stdout, named output reports, metadata, the Azure job
snapshot, and `SHA256SUMS.txt`. The dashboard-readable report cache is:

`registry/azureml_test_prep/reports/upbeat_shirt_3xzkjyzdvd.json`

## Phase 01 Repaired Manifest Result

The replacement Phase 01 job `strong_parrot_l70v7x8w6x` verified the repaired datastore copy.
It supersedes the earlier headerless-manifest Phase 01 result.

| Field | Value |
| --- | --- |
| Test | `01_data_access_test` |
| Status | `passed` |
| Audio files | `20` |
| Manifest rows | `20` |
| Manifest columns | `89` named columns |
| Schema warnings | none |
| Corrected audio path | `data/freesound/279681.aif` |
| Azure ML Studio | [Open run](https://ml.azure.com/runs/strong_parrot_l70v7x8w6x?wsid=/subscriptions/2aa6790f-891f-4a9b-8b7c-476ac25b0f82/resourcegroups/rg-cara-audio-training-aue/workspaces/rg-cara-audio-training-aue&tid=776c987b-2fd1-4464-9928-8f076d411ebc) |

The complete downloaded Azure bundle, job snapshot, and checksum manifest are preserved at:

`registry/azureml_test_prep/artifacts/strong_parrot_l70v7x8w6x/`

The dashboard-readable report cache is:

`registry/azureml_test_prep/reports/strong_parrot_l70v7x8w6x.json`

## Environment Registration Status

Both corrected custom environment definitions were registered successfully in the Azure ML
workspace on 2026-06-02:

| Environment | Version | Base image | Model package |
| --- | --- | --- | --- |
| `env-musicgen-audiocraft` | `2` | `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20260514.v1` | `git+https://github.com/facebookresearch/audiocraft.git` |
| `env-stable-audio-tools` | `2` | `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20260514.v1` | `git+https://github.com/Stability-AI/stable-audio-tools.git` |

The local definitions are:

- `azureml/environments/env_musicgen_audiocraft.yml`
- `azureml/environments/env_stable_audio_tools.yml`

The direct Azure ML registration snapshots and checksum manifest are preserved at:

- `registry/azureml_test_prep/snapshots/env-musicgen-audiocraft-v2_2026-06-02.json`
- `registry/azureml_test_prep/snapshots/env-stable-audio-tools-v2_2026-06-02.json`
- `registry/azureml_test_prep/snapshots/environment_registration_SHA256SUMS.txt`

Registration records the definitions. Azure ML built and validated both images successfully
during phases 03 and 04.

## Phase 03 Failed Image-Build Attempt

The first MusicGen validation job, `shy_garage_n7pzp789k0`, failed before the test script
started. Azure ML preserved only its image-build log, so no script-generated `report.json`
could exist.

The image-build log shows that AudioCraft requested `av==11.0.0`, and the PyAV build stopped
with:

`pkg-config is required for building PyAV`

The complete failed-attempt bundle and checksums are preserved at:

`registry/azureml_test_prep/artifacts/shy_garage_n7pzp789k0/`

The dashboard report endpoint now converts this kind of pre-script image-build failure into a
readable cached report. Environment version `2` adds Conda `av=11.0.0` and `pkg-config`, so pip
does not need to compile PyAV during the AudioCraft install.

## Phase 03 Version 2 Retry Monitor

The corrected MusicGen environment retry is Azure ML job `wheat_dog_0wh3fqkljk`. It was still
in Azure state `Preparing` at `2026-06-02T12:11:52.929358+00:00`.

Azure ML does not expose downloadable image-build logs while a command job is in `Preparing`.
The dashboard now records the Azure control-plane state approximately once per minute and shows
the cache timestamp, freshness, elapsed time, environment version, and direct Studio link.

The append-only heartbeat evidence for this retry is preserved at:

`registry/azureml_test_prep/monitor/wheat_dog_0wh3fqkljk.jsonl`

The retry completed successfully. The preserved heartbeat file records the image-build wait
without requiring a second submission.

## Phase 03 MusicGen Environment Result

| Field | Value |
| --- | --- |
| Test | `03_musicgen_env_test` |
| Status | `passed` |
| AudioCraft import | `ok` |
| PyTorch | `2.1.0+cu121` |
| CUDA available | `true` |
| CUDA device count | `1` |
| GPU | `NVIDIA H100 NVL` |
| Shared audio files | `20` |
| Shared manifest rows | `19` |
| Environment | `azureml:env-musicgen-audiocraft:2` |
| Azure ML Studio | [Open run](https://ml.azure.com/runs/wheat_dog_0wh3fqkljk?wsid=/subscriptions/2aa6790f-891f-4a9b-8b7c-476ac25b0f82/resourcegroups/rg-cara-audio-training-aue/workspaces/rg-cara-audio-training-aue&tid=776c987b-2fd1-4464-9928-8f076d411ebc) |

The complete downloaded Azure bundle, job snapshot, and checksum manifest are preserved at:

`registry/azureml_test_prep/artifacts/wheat_dog_0wh3fqkljk/`

The dashboard-readable report cache is:

`registry/azureml_test_prep/reports/wheat_dog_0wh3fqkljk.json`

## Phase 04 Stable Audio Environment Result

| Field | Value |
| --- | --- |
| Test | `04_stableaudio_env_test` |
| Status | `passed` |
| Stable Audio Tools import | `ok` |
| PyTorch | `2.7.1+cu126` |
| CUDA available | `true` |
| CUDA device count | `1` |
| GPU | `NVIDIA H100 NVL` |
| Shared audio files | `20` |
| Shared manifest rows | `19` |
| Environment | `azureml:env-stable-audio-tools:2` |
| Azure ML Studio | [Open run](https://ml.azure.com/runs/tough_kite_mmmxk4yy9p?wsid=/subscriptions/2aa6790f-891f-4a9b-8b7c-476ac25b0f82/resourcegroups/rg-cara-audio-training-aue/workspaces/rg-cara-audio-training-aue&tid=776c987b-2fd1-4464-9928-8f076d411ebc) |

The complete downloaded Azure bundle, job snapshot, and checksum manifest are preserved at:

`registry/azureml_test_prep/artifacts/tough_kite_mmmxk4yy9p/`

The dashboard-readable report cache is:

`registry/azureml_test_prep/reports/tough_kite_mmmxk4yy9p.json`

## Dataset Repair Status

The shared datastore manifest repair is complete. Phase 01 rerun `strong_parrot_l70v7x8w6x`
proved that Azure ML reads the uploaded header row, all 20 rows, and the corrected
`data/freesound/279681.aif` path. The Phase 02 CUDA result remains reusable.
