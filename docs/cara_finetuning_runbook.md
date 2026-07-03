# CARA Fine-Tuning Runbook

This runbook records the current CARA-Strong fine-tuning preparation flow. Keep
it updated whenever dashboard gates, Azure ML jobs, smoke defaults, or dataset
paths change.

## Source Dataset

The full fine-tuning upload uses the existing Azure ML datastore:

```text
azureml://datastores/ds_cara_raw_audio/paths/finetune-subset/
```

Expected layout:

```text
finetune-subset/data/cara_pool_manifest_v2.jsonl
finetune-subset/data/cara_pool_manifest_v2.csv
finetune-subset/data/freesound/
```

The smoke-test folder remains separate and should not be reused for full
fine-tuning prep:

```text
azureml://datastores/ds_cara_raw_audio/paths/test-audio/
```

## Dashboard Gate Order

### Diffusion / Stable Audio

1. Lock Manifest
2. Confirm Azure Upload
3. Prepare Stable Audio Dataset
4. Smoke Trainer Pending / Launch Smoke Trainer

### Autoregressive / MusicGen

1. Lock Manifest
2. Confirm Azure Upload
3. Prepare MusicGen Dataset
4. Cache EnCodec Tokens
5. Smoke Trainer Pending / Launch Smoke Trainer

`Confirm Azure Upload` is a human gate. It should only be set after the full
`finetune-subset` upload is complete in Azure ML storage.

The Stable Audio launch step submits a GPU-only Azure ML smoke trainer after the
prepared dataset reaches completion. MusicGen remains held until its
autoregressive trainer command job exists.

## Prepared Dataset Outputs

Prepared model-specific datasets are written under:

```text
azureml://datastores/ds_cara_raw_audio/paths/prepared/cara-strong-v0.4/
```

Stable Audio output:

```text
prepared/cara-strong-v0.4/stable_audio_open_small/
prepared/cara-strong-v0.4/stable_audio_open_small/manifest.jsonl
prepared/cara-strong-v0.4/stable_audio_open_small/audio/train/
prepared/cara-strong-v0.4/stable_audio_open_small/audio/validation/
prepared/cara-strong-v0.4/stable_audio_open_small/audio/test/
```

MusicGen output:

```text
prepared/cara-strong-v0.4/musicgen/
prepared/cara-strong-v0.4/musicgen/manifest.jsonl
prepared/cara-strong-v0.4/musicgen/audio/train/
prepared/cara-strong-v0.4/musicgen/audio/validation/
prepared/cara-strong-v0.4/musicgen/audio/test/
```

MusicGen EnCodec cache output:

```text
prepared/cara-strong-v0.4/musicgen_encodec_cache/
prepared/cara-strong-v0.4/musicgen_encodec_cache/manifest.encodec.jsonl
prepared/cara-strong-v0.4/musicgen_encodec_cache/tokens/
```

The `train`, `validation`, and `test` folders are disjoint split partitions.
They are not triplicate copies. Each prepared chunk is written to exactly one
split.

## Audio Windowing

Stable Audio Open Small prep:

- 44.1 kHz
- stereo
- WAV PCM
- target window: 11.88 seconds
- tail chunks kept when at least 2 seconds
- files shorter than 11.88 seconds are kept as one shorter prepared chunk

MusicGen prep:

- 32 kHz
- mono for stock MusicGen checkpoints
- WAV PCM
- target window: 30 seconds
- tail chunks kept when at least 5 seconds
- files shorter than 30 seconds are kept as one shorter prepared chunk

Raw audio may include `.ogg`, `.flac`, `.aif`, `.aiff`, `.mp3`, and `.wav`.
Preparation uses `ffmpeg` to decode, resample, channel-normalize, and write
model-specific WAV chunks.

## Training Manifests

Each model-specific preparation job creates a new training manifest. For every
prepared chunk, the manifest keeps the parent source lineage and CARA pool
metadata:

- `chunk_id`
- `source_example_id`
- `source`
- `source_id`
- `source_audio_path`
- `prepared_audio_path`
- `split`
- `chunk_index`
- `start_sec`
- `end_sec`
- `duration_sec`
- `sample_rate`
- `channels`
- `prompt`
- `title`
- `primary_genre`
- `secondary_genre`
- `style_tags`
- `artist_ids`
- `licence_class`
- `cara_pool_id`
- `cara_pool_index`
- `cara_pool_family`
- `cara_pool_family_index`
- `codeword_status`

If one parent file is split into several chunks, each child chunk receives the
same parent CARA pool id plus its own chunk timing and prepared audio path.

## Compute Policy

Preprocessing can use:

- `gpu-smoke-h100` only when no active jobs are found on either H100-backed
  compute target: `gpu-smoke-h100` or `gpu-fulltraining-h100`
- `cpu-prep-cluster` when either H100-backed compute target is queued, running,
  finalizing, or otherwise active

Training is GPU-only. Smoke or full fine-tuning should not silently fall back to
CPU. If the H100 is busy, training should queue, block, or warn.

Use the default `auto: H100 unless busy, then CPU` preprocessing route for
MusicGen while any Stable Audio smoke/full trainer is queued or running. Only
explicit H100 override should bypass this guard.

Autoregressive / MusicGen CPU fallback by stage:

- Step 05b MusicGen dataset preparation: CPU-capable. In `auto` mode the
  backend checks both H100-backed targets, `gpu-smoke-h100` and
  `gpu-fulltraining-h100`; if either is active, it submits to
  `cpu-prep-cluster`.
- Step 06 EnCodec token cache: CPU-capable, although slower. It uses the same
  two-target H100 availability check and falls back to `cpu-prep-cluster` when
  either H100-backed target is active.
- Step 07 MusicGen trainer preflight: GPU-only. It should block while H100 is
  occupied because it validates the trainer path, checkpoint load, and GPU-side
  training inputs.
- Steps 08-12 MusicGen smoke/full trainers: GPU-only. They must not silently
  fall back to CPU, because CPU trainer throughput would not be valid evidence
  for timing, feasibility, or full-run planning.

MusicGen preparation and EnCodec cache jobs use mounted AzureML folders
(`ro_mount` inputs and `rw_mount` outputs). Prepared WAV chunks and cached
token artifacts live in the Azure datastore, not the H100 node root disk. This
reduces the chance of repeating the Stable Audio full-run `AZ_BATCH_NODE_ROOT_DIR`
disk pressure failure while the AR ladder is prepared in parallel.

## Smoke Defaults

Current conservative smoke defaults:

- max steps: 250
- batch size: 8
- DataLoader workers: 0
- learning rate: 1e-5
- trainer compute: `gpu-smoke-h100`

Stable Audio smoke sequence:

1. Step 05: base trainer plumbing smoke over the prepared Stable Audio chunks
   (`no_cara_baseline`).
2. Step 06: CARA sidecar injection smoke with CARA sidecars enabled after the
   base smoke proves training plumbing (`cara_lite`).
3. Step 07: CARA attribution-head smoke (`cara_head`), using detached
   Stable Audio hidden-state taps to train only the CARA attribution head.
4. Step 08: CARA-Strong smoke (`cara_strong`), using the same attribution
   head with non-detached hidden-state auxiliary loss so the attribution
   objective can shape the trainable Stable Audio path.
5. Step 09: full CARA-Strong fine-tune (`cara_strong`, `training_scope=full`),
   using all prepared train rows plus held-out validation/test CARA metrics.

The first smoke run validates dataset loading, forward/backward pass,
checkpoint writes, logging, and resume mechanics. The second smoke isolates
CARA sidecar/injection behavior.

The dashboard must keep completed smoke steps visible and append new steps
rather than relabeling an old button. That gives benchmark testing a stable ladder:

## Benchmark Testing And Repairability

After a full model run completes, promotion is not based on a single generation
demo. The Testing and Benchmarks pages now use a shared benchmark testing evaluation
schema:

- Native CARA output metrics are reported only for models that have a native
  CARA path: Stable Audio CARA-Strong, MusicGen CARA-Strong, and Hybrid
  ACE-Step CARA-Strong.
- The original base models have no native CARA-id channel. Their native
  CARA-id accuracy is therefore `N/A`, not zero.
- Every model, including the base model, can still be compared in the
  external-probe lane by applying the same attribution probe to generated audio
  or hidden states.
- Random, prior, keyword, prompt-only, and shuffled-label controls must be
  reported beside the model metrics.
- Headline comparisons must separate native-output accuracy from external-probe
  accuracy.

Repairability metrics are reported as four mutually exclusive buckets:

1. Exact pool: the predicted CARA pool-id is registry-valid and exactly matches
   the expected held-out pool.
2. Repairable pool: the predicted id is malformed or wrong, but checksum or
   unique edit-distance repair maps it to the expected held-out registry pool.
3. Family / genre fallback: the pool cannot be recovered, but the result is
   attributable to the correct CARA family or genre level.
4. Unattributable: no exact, repairable, or family-level attribution can be
   recovered.

The dashboard may also show a repair-method matrix. That matrix is a resolution
audit only: `checksum_repair` or `unique_payload_edit_distance` means the raw
prediction could be mapped to a registry entry, but it is counted in the
repairability ladder only if the resolved pool matches the expected held-out
pool. Exact, repairable, family, and unattributable rows in the ladder are
therefore correctness buckets, not raw syntax buckets.

The Benchmarks page also shows a mechanical resolution matrix when lane-level
resolution counts are available. This matrix answers a different question: did
the emitted CARA-like string resolve to any valid registry pool, regardless of
whether that registry pool was the expected held-out target? It is useful as a
candidate influence signal for generated audio that may drift away from the
prompt target, but it is weaker evidence than the strict repairability ladder.
Current resolver limits are deliberately explicit: checksum repair is a
same-payload bad-check-digit repair, unique pool payload edit-distance repair is
capped at distance 2, and family fallback is capped at distance 4.

Interpret mechanical resolution by architecture. Stable Audio Diffusion and
Context Diffusion native attribution are closed-set classifier heads over the
locked registry pool indices; if extraction succeeds, the head will normally
emit a registry pool by construction. For those lanes, the mechanical matrix is
mostly an extractor/closed-set sanity check, and the strict ladder plus top-k,
confidence, calibration, and held-out metrics carry the attribution claim.
MusicGen's native lane emits a suffix/code sequence that must decode or repair
into the registry, so mechanical resolution is more informative for that
autoregressive architecture.

The `expected` CARA id in generated-audio benchmarks is copied from the locked
prompt manifest row, which is sampled from validation/test prepared-manifest
rows with known CARA labels. The text prompt is not just the codeword: it uses
the row's prompt/title metadata, and the suite condition determines whether a
visible CARA codeword is appended, stripped, no-tagged, or shuffled. For trained
candidate lanes, the generation job may also pass the target pool/family indices
as structured CARA conditioning for label-applicable conditions; open-quality and
shuffled-label controls suppress that structured target.

Future refinement: CARA calibration / reinforcement retuning
------------------------------------------------------------

The current benchmark reports the native attribution behavior after supervised
fine-tuning only. A later append-only branch may test whether an additional
CARA calibration or reinforcement-style pass can improve exact codeword recovery
without weakening the attribution claim. This should be treated as a follow-on
comparison, not as part of the current core result.

The proposed reward / preference target would keep the existing repairability
ladder:

- highest reward for exact expected pool recovery;
- smaller reward for repairable-to-expected-pool recovery;
- smaller reward again for correct family / genre fallback;
- no success credit for valid-but-wrong registry pools;
- penalty for invalid, unresolved, or hallucinated CARA ids;
- quality guardrails so attribution improvements do not come from degraded audio
  or trivial codeword copying.

For Diffusion and Context Diffusion, the safer implementation is likely not
classic RLHF. Candidate methods include auxiliary-loss retuning with a higher
CARA-head weight, reward-weighted continuation fine-tuning, DPO-style preference
pairs over generated outputs, or classifier-guided generation ablations. Any
claim must verify that the attribution head remains tied to noised-audio /
generation hidden states rather than learning a prompt-only shortcut.

For MusicGen, reinforcement-style suffix calibration is more natural because the
CARA output is token-like. The follow-on experiment can compare whether suffix
exactness improves when the reward is applied to decoded CARA suffixes while the
audio-token context remains fixed.

Required controls for this future branch:

- reuse the same locked prompt set, seeds, model lanes, and benchmark scoring
  contract;
- report pre-retune versus post-retune exact, repairable, family, invalid, and
  quality metrics side by side;
- include shuffled-label, prompt-only, codeword-withheld, and no-audio /
  no-hidden-state controls;
- preserve original checkpoints and write separate retuned deltas so the
  supervised baseline remains auditable;
- label any improvement as "calibrated CARA output" unless the audio-linked
  controls prove the reward did not merely train the model to print better ids.

The evaluation launch endpoint is dry-run by default. A live run must require
the typed phrase `LAUNCH BENCHMARK TESTING EVALUATION` and must use only existing
Azure ML workspace resources, command jobs, datastores, computes, and
environments.

The Benchmarks page can generate an OpenAI-assisted abstract/TLDR from the
current benchmark matrix, model state, repairability schema, and project goal.
This uses local `.env` values:

```text
OPENAI_API_KEY
OPENAI_MODEL
OPENAI_REASONING_EFFORT
```

Every generated abstract is saved for review:

```text
evaluation/generated/benchmark_tldr_latest.json
evaluation/generated/benchmark_tldr_log.jsonl
```

If OpenAI is later used to generate benchmark prompts, those prompt sets must
also be recorded in the generated artifact payload so the same prompts can be
reused across Diffusion, Autoregressive, Hybrid, and baseline evaluation runs.
Generated audio outputs from live benchmark runs should be saved under the
evaluation run output folder by model, suite, prompt id, and seed so qualitative
listening is possible without making it part of the primary attribution metric.

Before any generated-audio scoring run, lock one setup run's
`prompt_manifest.jsonl` as the canonical benchmark prompt set. The dashboard
stores this as:

```text
evaluation/generated/benchmark_prompt_set.lock.json
```

The lock records the setup Azure job name, output folder, suite ids, seed count,
and canonical `prompt_manifest_uri`. After this exists, later Diffusion,
Autoregressive, Hybrid, and baseline scoring jobs must consume that exact
manifest rather than resampling held-out chunks or regenerating prompts. This is
the prompt-comparability guardrail for like-for-like benchmark reporting.

Live evaluation wave 1 is wired as Azure ML job
`azureml/jobs/14_benchmark_testing_stable_audio_eval.yml` on `gpu-smoke-h100`. It is
GPU-only and does not fall back to CPU. It currently:

- validates the prepared Stable Audio manifest;
- validates the completed Stable Audio CARA-Strong trainable-delta output;
- loads the base Stable Audio checkpoint on CUDA;
- writes deterministic benchmark prompt/control manifests;
- writes statistical baseline-control metrics.

Wave 1 output is written under:

```text
azureml://datastores/ds_cara_raw_audio/paths/evaluation-runs/cara-strong-v0.4/stable_audio_benchmark_testing/<run>/
```

Expected files:

```text
evaluation_plan.json
prompt_manifest.jsonl
control_metrics.json
benchmark_testing_stable_audio_eval_report.json
metadata.json
```

Generated-audio native/probe scoring is the follow-on evaluation stage and must
reuse `prompt_manifest.jsonl` from wave 1 so prompts, seeds, expected CARA
labels, and baseline controls remain stable across model comparisons.

Generated-audio benchmark scoring is wired as the append-only follow-on Azure ML
job `azureml/jobs/15_benchmark_testing_stable_audio_audio.yml` on
`gpu-smoke-h100`. It is GPU-only, checks the locked prompt set through the
dashboard API, does not fall back to CPU, and requires the typed phrase
`LAUNCH DIFFUSION AUDIO BENCHMARK`.

Step 15 mounts the locked `prompt_manifest.jsonl` as a `uri_file`, loads the
base Stable Audio Open Small checkpoint, applies the completed CARA-Strong
`checkpoints/trainable_delta.pt` for the fine-tuned Diffusion model, and can
also apply the Context Diffusion branch trainable delta once step 14 completes.
Diffusion and Context Diffusion must therefore reuse the same prompt rows,
suite ids, seeds, audio-output contract, and scorer contract. The released
base diffusion lanes may be hidden from the dashboard headline matrix while the
candidate-vs-candidate comparison is being developed, but they remain optional
manual controls rather than primary claims. Step 15 writes generated WAVs plus
manifests under:

```text
azureml://datastores/ds_cara_raw_audio/paths/evaluation-runs/cara-strong-v0.4/stable_audio_benchmark_testing/audio_<scope>/<run>/
```

Expected Step 15 files:

```text
audio/<model_id>/<suite_id>/<prompt_id>.wav
generation_manifest.jsonl
benchmark_audio_plan.json
benchmark_audio_metrics.json
benchmark_testing_stable_audio_audio_report.json
metadata.json
```

The Step 15 metrics file must not infer CARA accuracy from expected labels. If a
native hidden-state attribution extractor or external probe has not run yet,
the metric status remains `pending_attribution_extractor` /
`pending_external_probe`. This preserves peer-reviewable separation between
generated audio evidence and later attribution/repairability scoring.

Step 16 attribution scoring is the first stage allowed to populate CARA
accuracy cells. It must consume generated-audio manifests, write real
prediction fields or explicit exception/unattributable rows, and then score
those predictions through the shared CARA repairability resolver. The resulting
`metrics_latest.json` must include:

```text
scored_generation_manifest.jsonl
native_predictions.jsonl
cara_registry_resolver.json
repairability_matrix
repair_method_matrix
prediction_examples
lanes.<lane_id>.repairability
lanes.<lane_id>.tier_counts
lanes.<lane_id>.repair_method_counts
```

The Benchmarks page displays `repairability_matrix` separately from the headline
matrix so exact pool matches, uniquely repaired pool ids, family/genre fallback,
and unattributable predictions are visible as distinct evidence tiers. A
`missing_predictions` lane means the native extractor or external probe did not
produce auditable CARA rows; it is not interchangeable with an unattributable
prediction, which is a scored failure state.

Repairability is intentionally strict. `repairable_pool` means the raw or
malformed CARA-id repairs uniquely to the expected held-out pool. A prediction
that repairs to some other valid registry pool is diagnostic evidence only; it
does not count as pool accuracy and should fall back to family/genre evidence
only if the resolved family matches.

The dashboard blocks accidental duplicate Step 16 submissions once scorer jobs
are recorded for a generated-audio run. If the scorer code changes and the same
audio outputs need to be re-evaluated, use the explicit `Force re-score current
full run` checkbox with the same typed confirmation. That creates new scorer
artifacts while preserving the older scoring attempts in the Azure job registry.

Current comparison suites:

- Held-out audio attribution: strongest source-disjoint evidence using prepared
  validation/test chunks and known CARA labels.
- Known-pool prompt recall: prompt-derived generation with expected pool/family
  targets from held-out metadata.
- Control-token confound: prompt-only, audio-hidden-only, CARA-text-removed, and
  shuffled-label probes.
- Open generation quality: registry validity, repairability, stability, and
  quality without known pool accuracy.
- Adversarial persistence: off-distribution prompts that test whether CARA
  attribution survives prompt drift.
- Baseline negative control: base checkpoints and statistical controls.
step 05 remains the baseline result, step 06 is the CARA-lite control, and later
steps can be added without rewriting the completed evidence trail.

The completed Stable Audio baseline smoke is the `no_cara_baseline` variant. It
passes CARA metadata through the dataset module for audit, but it does not add
structured `cara_pool_index` / `cara_pool_family` conditioners, does not train
an attribution head, and must not be reported as CARA-Strong evidence.

For `no_cara_baseline` and `cara_lite` smoke runs, the trainer removes the
Stable Audio Open Small `training.arc` block from the Hugging Face config at
runtime. That config contains a placeholder discriminator checkpoint path
(`/path/to/base/rf/model`), which would otherwise make Stable Audio Tools build
the ARC wrapper and fail before the baseline diffusion trainer can start. The
smoke path intentionally uses the plain conditional diffusion training wrapper;
ARC/adversarial discriminator training is not part of the baseline smoke.

Plan v0.4 compliance ladder:

- `no_cara_baseline`: ordinary prompt only; validates same-data trainer
  plumbing and style/pool recoverability floor.
- `cara_lite`: text prompt includes a CARA pool tag as a prompt-control /
  upper-bound check; no auxiliary attribution loss.
- `cara_head`: structured CARA metadata present; frozen/detached backbone
  features feed a hierarchical pool/family attribution head. This validates
  label mapping, DiT hidden-state taps, registry decoding, calibration metrics,
  and leakage controls, but does not claim CARA-Strong.
- `cara_strong`: ordinary prompt text stays comparable to the baseline, while
  `cara_pool_index` and `cara_pool_family_index` are added as native Stable
  Audio `int` conditioners joined to DiT cross-attention conditioning. A
  non-detached pool/family attribution loss is also added to the Stable Audio
  training step. This is the first variant that can support the CARA-Strong
  research claim, subject to longer benchmark comparisons.

Both attribution variants write `cara_registry_resolver.json` into the run
output. That resolver records the locked pool/family index maps, registry hash,
decoded CARA-id format, pool/family support, and manifest-lock reference. The
trainer fails fast if any prepared manifest row lacks `cara_pool_id`,
`cara_pool_index`, `cara_pool_family`, or `cara_pool_family_index`.
Prepared chunks should also carry `cara_pool_codeword` and
`cara_registered_codeword`. If an older prepared manifest lacks the short
`cara_pool_codeword`, the trainer derives it from the full
`cara_pool_id` and records the count in `cara_codeword_manifest_summary`.

Default attribution smoke setting:

- attribution loss weight: `0.05`

Completed Stable Audio attribution smoke evidence:

- Step 07 `cara_head`: job `eager_lemon_rx67hzy41k` passed at 250 steps on
  `gpu-smoke-h100` / `env-stable-audio-tools:8`. It used head-only AdamW,
  detached/frozen Stable Audio features, valid registry decoding, and last
  smoke metrics of `pool_top1=1.0`, `pool_top5=1.0`, `family_top1=1.0`,
  `hierarchical_valid=1.0`, `ece=0.15344902873039246`.
- Step 08 `cara_strong`: job `olive_pin_ms33zkw1l3` passed at 250 steps on
  `gpu-smoke-h100` / `env-stable-audio-tools:8`. It used the
  Stable-Audio-plus-head optimizer, non-detached hidden-state attribution loss,
  valid registry decoding, and last smoke metrics of `pool_top1=0.875`,
  `pool_top5=1.0`, `family_top1=1.0`, `hierarchical_valid=1.0`,
  `ece=0.15926945209503174`.

Those results validate trainer plumbing and the audio-hidden-state CARA path.
They do not replace full validation evidence because both jobs were 250-step
smokes capped at 2,048 train files, and their metrics are last training-batch
metrics rather than held-out split summaries.
They also predate the stricter full-run change that routes CARA pool/family
indices through native Stable Audio conditioner config rather than prompt text.
The follow-up step 08 rerun `neat_nose_24xwtv0kfw` failed with
`KeyError('cara_pool_index')`: the prepared batch metadata contained the CARA
index fields, and the model config listed them as DiT cross-attention
conditioners, but the already-loaded pretrained Stable Audio model still had
only its original conditioner modules. The trainer now patches the loaded
model's conditioner with the CARA `int` conditioners before creating the
Lightning wrapper, preserves the pretrained prompt/time conditioners, and
records a first-batch CARA conditioner preview before training starts. Run the
trainer preflight, then rerun step 08 once before a long full run to verify the
updated conditioner wiring on Azure.
The next step 08 rerun completed the conditioner preview and trained to about
step 170/250 before Azure OOM-killed a PyTorch `pt_data_worker` process
(`anon-rss` about 165 GB). The trainer and dashboard now default Stable Audio
training and evaluation to `num_workers=0`; this is slower but avoids
multiprocessing DataLoader worker RAM blow-up. A newer failed/canceled step 08
attempt keeps step 09 locked even when older step 08 evidence passed.
The first full step 09 attempt then trained to about `6000 / 7665` steps before
Azure ML reported `DiskFullError` under `AZ_BATCH_NODE_ROOT_DIR`. Treat this as
an operational storage failure, not a reason to change the CARA model design or
request a larger SKU first. The trainer now prunes checkpoint files during
training and keeps only `last.ckpt` plus the newest periodic checkpoint by
default (`checkpoint_keep_last_n=1`).

Full Stable Audio CARA-Strong handoff:

- Step 09 uses `azureml/jobs/09_full_stable_audio_cara_strong_trainer.yml`.
- The dashboard sends `training_scope=full`, `variant=cara_strong`,
  `max_train_files=0`, `max_eval_files=0`, `max_eval_batches=0`, and
  `run_eval=true`.
- The optional `Full training run` checkbox disables the manual max-steps box
  for step 09 and sends `max_steps=0` as a sentinel. The trainer resolves that
  to the full prepared train dataloader length, so the job runs a full dataset
  pass without using the smoke step cap. It is not an infinite Azure job.
- The full job writes to
  `azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_full/`.
- The full job uses mounted AzureML input/output folders and writes compact
  `checkpoints/trainable_delta.pt` snapshots during training. This artifact is
  a CARA trainable-parameter delta against the declared base checkpoint, not a
  full Lightning checkpoint with optimizer/backbone state. It is the preferred
  backup artifact on the current 128 GB node-root SKU.
- After a repeated `UserScriptFilledDisk` failure, the trainer also routes
  common Hugging Face, Torch, matplotlib, and temp runtime directories under
  the mounted output folder. Smoke runs still write stable Lightning checkpoint
  names, but full runs disable Lightning checkpointing and refresh
  `trainable_delta.pt` instead so a useful backup exists before final Azure job
  packaging. This should be tried before moving to a larger H100 SKU solely for
  disk space.
- The backend allows step 09 only after step 08 has passed, blocks CPU fallback,
  and requires the typed confirmation phrase
  `LAUNCH FULL CARA-STRONG FINETUNE`.
- The full report must include `heldout_evaluation.validation.metrics` and
  `heldout_evaluation.test.metrics`; otherwise the run is marked failed even if
  training reaches `max_steps`.
- Before launching step 09 after trainer-code changes, the dashboard keeps the
  append-only ladder intact but allows deliberate revalidation: re-run step 04
  Trainer Preflight, optionally re-run step 08 CARA-Strong Smoke, then launch
  step 09. Earlier passed jobs remain recorded as prior evidence.

## Matched Diffusion / Autoregressive Methodology

The fine-tuning comparison is designed as a like-for-like evidence ladder across
the two common architecture families. The implementation differs only where the
architecture forces it:

| Evidence layer | Stable Audio diffusion | MusicGen autoregressive |
| --- | --- | --- |
| Baseline | Same prepared chunks, no CARA signal | Same prepared chunks and EnCodec codes, no CARA signal |
| CARA-lite | CARA text/prompt control only | CARA text/prompt control only |
| Detached probe | Frozen/detached hidden states predict registry CARA labels | Frozen/detached real MusicGen LM hidden states predict registry CARA suffix |
| CARA-Strong | Native CARA conditioning plus non-detached attribution loss | Real MusicGen LM token loss plus non-detached CARA suffix loss from LM hidden states |
| Full run | Full train rows plus validation/test CARA metrics | Real MusicGen LM trainable-delta run over full cached token rows plus validation/test CARA metrics |

For benchmark testing, report these as matched methodology stages. Do not claim the
MusicGen suffix is an extra-superior method; it is the architecture-native
autoregressive expression of the same CARA evidence layer that Stable Audio
implements with hidden-state attribution taps.

MusicGen smoke sequence:

1. Step 05b: prepare the MusicGen 32 kHz mono chunk dataset.
2. Step 06: cache EnCodec token targets and write `manifest.encodec.jsonl`,
   `cara_registry_resolver.json`, and `cara_suffix_vocab.json`.
3. Step 07: real MusicGen LM trainer preflight over the cached token manifest.
4. Step 08: real MusicGen LM baseline smoke (`no_cara_baseline`).
5. Step 09: real MusicGen LM CARA-lite prompt-control smoke (`cara_lite`).
6. Step 10: real MusicGen LM detached CARA suffix-probe smoke (`cara_probe`).
7. Step 11: real MusicGen LM CARA-Strong suffix smoke (`cara_strong`).
8. Step 12: full real MusicGen LM CARA-Strong fine-tune, held until the real-LM
   smoke ladder and diffusion full-run evidence are reviewed.

The MusicGen CARA suffix target is built as:

```text
<CARA_BOS> FAMILY_<index> POOL_<index> <CARA_SEP> codeword characters <CARA_CHECK> CHECK_<2hex> <CARA_END>
```

Audio decoding ignores this suffix. The suffix is decoded separately through
the locked CARA registry, so reports can distinguish exact suffix match,
registry-valid repair, family/pool consistency, and checksum validity.

Step 06 deliberately loads the AudioCraft compression model only. It should not
load the full MusicGen LM/T5 planner just to cache audio tokens. Each cached
token payload and each `manifest.encodec.jsonl` row must preserve the same
chunk-level attribution binding: `chunk_id`, `source_example_id`,
`prepared_audio_path`, `encodec_token_path`, `cara_pool_id`,
`cara_pool_index`, `cara_pool_family`, and `cara_pool_family_index`.

Before any MusicGen preflight, smoke, or full trainer builds its DataLoader, it
filters cached manifest rows with fewer than `2` EnCodec frames. The rejected
rows are written to `rejected_encodec_frame_rows.json` and summarized in the
trainer report. This protects full runs from ultra-short token artifacts that a
small smoke subset may not sample, without recaching the valid token cache or
changing the CARA suffix mapping for retained chunks.

The AR trainer then builds `suffix_tokens` from that exact cached manifest row,
loads `facebook/musicgen-small` through AudioCraft, runs the real LM
`compute_predictions(codes, conditions)` path, and supervises the CARA suffix
from the LM hidden state belonging to that audio-token sample. The previous
`smoke_musicgen_ar_trainer.py` proxy is retained only as historical/debugging
evidence; it is not the matched full Autoregressive comparison.

## Hybrid ACE-Step v1.5 Planning

The dashboard includes a third fine-tuning page, `Finetune: Hybrid`, for
ACE-Step v1.5 with the 0.6B LM planner target selected as the first comparable
Hybrid CARA-Strong arm. Step 01 is documented research only. Steps 02-12 are
now wired as ordered Azure ML gates from the Hybrid dashboard. Each stage writes
its own report and metadata under the ACE-Step datastore branch so later review
can distinguish preflight, tensor preparation, planner probing, smoke evidence,
and full-training evidence.

Primary-source findings:

- ACE-Step v1.5 is a hybrid architecture: an LM planner transforms user queries
  into song blueprints, metadata, lyrics, and captions, then a DiT synthesizer
  renders the audio.
- The official v1.5 model zoo lists base/SFT/turbo DiT variants plus 0.6B,
  1.7B, and 4B LM variants. The base and SFT DiT variants are marked as easy to
  fine-tune; turbo variants are marked medium.
- The first CARA comparison target is the 0.6B planner configuration
  (`ACE-Step/acestep-5Hz-lm-0.6B`) with a base/SFT DiT path. Larger LM variants
  can be added later only as scale-ablation branches, because the first Hybrid
  result should remain comparable to the existing Stable Audio and MusicGen
  model sizes.
- The official repository points to Side-Step for advanced training. Side-Step
  training requires Python 3.11+ in its upstream guide, CUDA for training,
  ACE-Step checkpoints, and supports corrected-mode LoRA as the recommended
  stable path. The dashboard's Azure preflight environment currently uses
  Python 3.10 plus official CUDA PyTorch pip wheels because the prior conda
  PyTorch image failed at import time with a `libtorch_cpu.so` ITT symbol error
  before any ACE code could run.
- Side-Step preprocessing supports folder-only mode and JSON mode. CARA must
  use JSON mode, because folder-only mode derives captions from filenames and
  cannot preserve the locked registry fields. JSON mode lets Step 03 provide
  normal tag-withheld captions, per-sample `custom_tag` codewords for explicit
  CARA-lite/suffix lanes, and accepted metadata fields that preserve
  `cara_pool_id`, `cara_pool_index`, `cara_pool_family`,
  `cara_pool_family_index`, and `cara_source_pool_id`.
- Side-Step's documented input formats are `.wav`, `.mp3`, `.flac`, `.ogg`,
  `.opus`, and `.m4a`. The CARA source corpus can include additional decodable
  formats such as `.aif` / `.aiff`; Step 03 therefore uses `ffmpeg` to convert
  unsupported-but-present source files into 48 kHz stereo WAV files under the
  ACE output folder. The generated JSON points Side-Step at the converted WAV,
  while `original_audio_path` and `audio_conversion` preserve source lineage.
- Side-Step preprocessing converts raw audio into tensors in two passes:
  VAE/text encoder first, then DiT encoder condition encodings. This means ACE
  preprocessing should produce a Side-Step-compatible full ACE-Step
  `dataset.json`, CARA-labelled tensor manifest, and registry resolver, not
  only WAV chunks.

Current wired artifacts:

```text
azureml/environments/env_ace_step.yml
azureml/jobs/13_ace_step_env_preflight.yml
azureml/jobs/19_prepare_ace_step_tensors.yml
azureml/jobs/20_ace_step_planner_survival_probe.yml
azureml/jobs/21_ace_step_dit_tap_discovery.yml
azureml/jobs/22_ace_step_hybrid_smoke.yml
azureml/jobs/23_full_ace_step_hybrid_trainer.yml
src/ace_step_env_preflight.py
src/ace_step_hybrid_stages.py
```

The preflight writes reports under:

```text
azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/ace_step_preflight/
```

It checks CUDA, `diffusers.AceStepPipeline`, common LoRA/training imports,
`ffmpeg`, the source audio folder, and sampled manifest rows that can be
normalised into the CARA training label contract. The raw uploaded
`cara_pool_manifest_v2.jsonl` is allowed to provide `cara_source_pool_id` and
`cara_pool_family` only; Step 02/03 derive `cara_pool_id`, `cara_pool_index`,
and `cara_pool_family_index` deterministically before training artifacts are
written. It does not download the ACE checkpoint unless the dashboard checkbox is
enabled. The preflight report records the selected 0.6B planner checkpoint, DiT
variant, environment version, and comparison role.

The ACE-Step branch writes later artifacts under:

```text
azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/ace_step/
```

Compute policy:

- Step 02 preflight, Step 05 DiT tap discovery, Steps 06-11 smoke jobs, and
  Step 12 full Hybrid training are GPU-only. They check both H100-backed compute
  targets before submission and do not fall back to CPU, because CUDA and DiT
  hook viability are part of the evidence.
- Step 03 tensor preparation and Step 04 planner survival probing are
  H100-preferred but CPU-capable. If either H100-backed target is active, the
  dashboard may route these jobs to `cpu-prep-cluster` so preprocessing/probing
  can continue while a GPU fine-tune is running.
- All stages use existing Azure ML command jobs, datastores, environments, and
  approved compute targets only.

Benchmark testing ladder for ACE-Step should be append-only:

1. ACE source and license review: record the checkpoint, official training
   route, license/access terms, and existing-Azure-resource cost guardrail.
   This is a local dashboard action, not an Azure job. It writes
   `registry/cara_strong/ace_step_source_license_review.json` and a registry
   event, then unlocks the Azure preflight gate when the ordinary manifest/upload
   readiness checks also pass.
2. ACE environment preflight: validate CUDA, ACE-Step imports, Side-Step/LoRA
   dependencies, checkpoint access, and datastore access.
3. Prepare ACE dataset JSON + tensors from the same locked CARA manifest and
   source-disjoint splits. This is the Hybrid branch's equivalent of the
   Stable Audio and MusicGen dataset-preparation gates. It must use Side-Step
   JSON mode rather than folder-only mode. The JSON must be full ACE-Step format
   with `metadata` plus `samples`; every sample keeps a normal tag-withheld
   `caption`, a per-sample `custom_tag` containing the CARA codeword for
   explicit CARA-control lanes, and accepted metadata fields for
   `cara_source_pool_id`, `cara_pool_id`, `cara_pool_index`,
   `cara_pool_family`, `cara_pool_family_index`, split, source id, and registry
   linkage. If the raw manifest lacks `cara_pool_id` or index fields, Step 03
   derives them from the locked `cara_source_pool_id` and family values and
   writes that derivation into `dataset.json` and `ace_registry_resolver.json`.
   If a source file is not one of Side-Step's native formats, Step 03
   converts it to 48 kHz stereo WAV via `ffmpeg` and records both the converted
   `audio_path` and the `original_audio_path`. The tensor manifest must preserve
   the same fields plus the Side-Step audio-format / conversion audit. The
   current implementation rejects rows whose source audio cannot be found or
   whose `ffmpeg` conversion fails, records them in
   `rejected_audio_rows.jsonl`, and continues when the remaining dataset is
   valid. This avoids inserting placeholder paths into the ACE manifest while
   preserving a peer-reviewable audit trail for missing or unconvertible source
   files. The current implementation writes `ace_tensor_manifest.jsonl`,
   full-format `dataset.json`, `ace_registry_resolver.json`,
   `sidestep_commands.json`, and `rejected_audio_rows.jsonl`.
4. Planner survival probe: measure whether `cara_pool_index`,
   `cara_pool_family_index`, and CARA codeword survive LM planning, query
   rewriting, captions, structured metadata, and any CoT/planner outputs. This
   stage is a planner-only audit and must not be reported as audio training.
   Steps 04-12 now fail fast if `ace_tensor_manifest.jsonl` is missing required
   CARA fields, so a later smoke cannot silently train on a partial or
   unlabelled tensor contract.
5. DiT tap discovery: locate mid/late DiT hidden states and verify detached
   attribution-head compatibility.
6. Baseline LoRA smoke: same data, no CARA signal, loss/checkpoint/resume
   evidence.
7. CARA-lite planner smoke: prompt-only CARA control through the planner to
   expose leakage or text-normalisation effects.
8. Detached DiT attribution-head smoke: frozen/detached DiT hidden states
   predict CARA registry labels.
9. Planner-preserved CARA smoke: structured CARA is routed through the planner
   and remains recoverable at DiT taps.
10. Planner-bypass CARA smoke: direct or constrained CARA conditioning reaches
    DiT with planner rewriting reduced or bypassed, separating planner failure
    from DiT attribution failure.
11. Hybrid CARA-Strong smoke: non-detached CARA auxiliary loss plus DiT
    attribution metrics and planner-survival metrics.
12. Full hybrid comparison only after the smoke ladder proves the planner and
    DiT evidence path.

Side-Step handoff and claim scope:

- The full trainer job is prepared to call Side-Step when ACE checkpoints and
  Side-Step tensor directories are mounted and `run_sidestep=true`. The
  intended Side-Step path is corrected-mode LoRA over the selected base/SFT DiT
  target, using the CARA-labelled tensor manifest and resolver from Step 03.
- Step 12 full Hybrid training targets the existing
  `gpu-fulltraining-h100` Azure ML compute, matching the completed Stable Audio,
  Context Diffusion, and MusicGen full-run convention. Earlier ACE smoke,
  preflight, tap-discovery, and Side-Step input-preparation stages may still use
  `gpu-smoke-h100` as their GPU validation/preparation target, but the
  deployable full fine-tune should not launch on the smoke compute.
- Step 12 uses a separate full-training Azure ML environment,
  `azureml:env-ace-step-sidestep:3`, rather than the lighter
  `azureml:env-ace-step:5` preflight/smoke/benchmark environment. The Side-Step
  environment uses Python 3.11 and carries the Side-Step dependency set needed
  at runtime. The `:3` environment installs CUDA 12.8 Torch, torchaudio,
  torchvision, the upstream prebuilt `flash-attn` wheel, and the Side-Step
  dependency set in the Azure image. Step 12a / Step 12 then clone the
  Side-Step source tree at runtime and call
  `python train.py --plain --yes preprocess ...` or
  `python train.py --plain --yes train ...` directly from that checkout. The
  `--plain --yes` flags are required in Azure because Side-Step otherwise asks
  for interactive confirmation (`Start training? [Y/n]`) and can abort without
  producing adapter artifacts. This avoids Azure image-build failures where pip
  either tries to compile `flash-attn` before Torch is importable or rejects
  `--no-deps` inside the conda-file requirements section, and it avoids the
  installed Side-Step console wrapper failure where the generated `sidestep`
  executable imports a non-packaged top-level `train` module.
- Before Step 12 submits, the dashboard now validates the two required mounted
  inputs as AzureML datastore folders, not local container placeholders. It
  checks that `checkpoint_dir` and `sidestep_tensor_dir` are
  `azureml://datastores/.../paths/...` URI-folder inputs and probes the
  datastore prefixes for at least one blob. If either the ACE checkpoint bundle
  or Side-Step tensor folder is absent, the dashboard keeps Step 12 locked and
  the API rejects direct submission before an Azure job is prepared.
- The missing-input bridge is Step 12a, `Prepare Side-Step Inputs`. It runs as
  a separate GPU command job after the Hybrid CARA-Strong smoke has passed and
  before the full fine-tune. Step 12a materializes the public
  `ACE-Step/Ace-Step1.5` checkpoint bundle into
  `ace_step/checkpoints/`, regenerates the CARA JSON dataset with paths valid
  for the current Azure container, clones the Side-Step source runner, and runs
  `python train.py preprocess ...` into `ace_step/tensors/sidestep_tensors/`.
  The full Step 12 launch should only be attempted after the dashboard verifies
  both prefixes contain blobs.
- The current public `ACE-Step/Ace-Step1.5` checkpoint tree exposes the
  `acestep-v15-turbo` DiT bundle plus shared `vae/` and
  `Qwen3-Embedding-0.6B/` folders. The live deployable Side-Step path therefore
  uses `model_variant=turbo` / `dit_variant=turbo_dit` unless a separate
  base/SFT checkpoint bundle is deliberately mounted and documented.
- A failed Step 12 with `sidestep.available=false` is an environment failure,
  not model evidence. A failed Step 12 with missing `checkpoint_dir` or
  `sidestep_tensor_dir` means the ACE checkpoint bundle or Side-Step
  preprocessed tensors still need to be mounted as Azure inputs before full
  training can begin.
- The dashboard full-run button now requires the real Side-Step path
  (`run_sidestep=true`). A `run_sidestep=false` execution is still meaningful as
  a lightweight contract-adapter/plumbing check, but it is not accepted as Step
  12 completion, is not deployable, and must not be benchmarked as the ACE-Step
  equivalent of the completed Stable Audio or MusicGen full fine-tunes.
- If a historical Step 12 run completes in contract-only mode, the ladder should
  display it as a contract handoff result and keep the real Side-Step LoRA
  fine-tune gate open. The expected full run should take material GPU time once
  ACE checkpoints, Side-Step tensors, and the Side-Step training module are
  actually mounted.
- A run may claim `deployable_ace_adapter=true` only when the report shows
  Side-Step actually ran, the output contains the Side-Step adapter artifacts,
  and the run records the mounted checkpoint/tensor source paths.
- A `run_sidestep=true` run that exits successfully but produces no adapter
  artifact files is treated as failed. This prevents a command-line success from
  being mistaken for a deployable ACE-Step LoRA delta.
- A `run_sidestep=true` run whose logs contain Side-Step's interactive
  confirmation prompt or `Aborted.` is treated as a non-interactive launch
  failure. It should be rerun only after confirming the generated Side-Step
  command includes `--plain --yes`.
- Step 12 and Step 12a stream Side-Step subprocess output into the mounted
  Azure output folder while the job is still running. The machine-readable file
  is `training_progress.json`; the human-readable tail is
  `training_progress.log`. The dashboard reads `training_progress.json` through
  `/api/training/run-progress?model=ace_step` and the Azure Runs progress API,
  so long Hybrid runs can show status, observed step, percent, elapsed time,
  rough ETA, and the latest Side-Step output line without touching or
  interrupting the Azure job.
- Step parsing is deliberately conservative. Configuration lines such as
  `max_steps: 20000` or `--max-steps 20000` must not be treated as current
  progress. New progress artifacts include `step_source_line`; if an older
  running artifact reports `observed_step == max_steps` without that source
  line, the dashboard suppresses the false 100% display and shows a warning.
- Step 12 uses the same disk-safe storage rule as the completed Stable Audio
  full runs: the Azure output is mounted, full merged ACE checkpoints are not
  written by CARA, and the canonical saved artifact is
  `checkpoints/trainable_delta.pt`. In `run_sidestep=false` mode this is a
  compact CARA Hybrid contract delta. In `run_sidestep=true` mode it records the
  Side-Step LoRA/adapter delta artifacts and hashes under the mounted output
  folder rather than packaging a full base model copy.

Likelihood assessment:

- DiT hidden-state attribution is plausible/high because the synthesis stage is
  a DiT, conceptually close to the Stable Audio attribution-head branch.
- Structured CARA conditioning through the LM planner is uncertain/medium. The
  planner may preserve CARA structure into DiT conditioning, or it may rewrite
  the signal away as part of prompt normalisation.
- Either result is useful. Survival through the planner would be strong
  "attribution through a CoT bottleneck" evidence; failure would show a
  planner-mediated attribution loss mode with regulatory relevance.

ACE-Step smoke reports should log:

- `planner_survival_exact`
- `planner_survival_repairable`
- `planner_cara_lost`
- `planner_cara_hallucinated`
- `dit_family_top1`
- `dit_pool_top1`
- `dit_pool_top5`
- `registry_valid_rate`
- `shuffled_label_baseline`
- source-disjoint validation/test summary

Every report must state whether it is planner-only, prompt-control, detached
head-only, planner-preserved, planner-bypass, or Hybrid CARA-Strong evidence.
The full hybrid run stays locked until the baseline, CARA-lite, detached DiT
head, planner-preserved, planner-bypass, and Hybrid CARA-Strong smokes all have
clear pass/fail evidence.

After Step 12 completes, the ACE-Step branch is registered in the shared Testing
and Benchmarks pages as `hybrid_ace_step_cara_strong_full`. The shared Testing
page now treats model selection as the first explicit benchmark choice and can
submit an ACE-Step generated-audio child job against the same locked prompt
manifest used by Diffusion, Context Diffusion, and MusicGen. Native Hybrid
attribution scoring is still intentionally separate: until the ACE-Step native
DiT-head scorer is implemented, Hybrid generated-audio manifests report native
CARA prediction as pending rather than being routed through the Stable Audio or
MusicGen scorers. This keeps the workflow reproducible from the beginning:
completed Diffusion, Context Diffusion, MusicGen, and ACE-Step audio artifacts
remain append-only, and later scoring can reuse the saved generation manifests
without replacing or overwriting earlier evidence.

The ACE-Step generated-audio benchmark uses `azureml:env-ace-step:5`. This
environment adds the Azure Key Vault client packages used by the shared
Hugging Face token path. The ACE benchmark runner also treats public
`ACE-Step/...` checkpoints as public-checkpoint loads if Key Vault token
retrieval is unavailable, so missing Key Vault SDK imports are not allowed to
fail the job before the actual ACE pipeline or adapter load is tested. Step 24
also mounts the Step 12a checkpoint bundle from `ace_step/checkpoints/`. The
public `ACE-Step/Ace-Step1.5` layout is not assumed to be a ready generic
Diffusers root: the public bundle contains the ACE component layout
(`acestep-v15-turbo/`, `vae/`, and `Qwen3-Embedding-0.6B/`) rather than a root
`model_index.json`. Step 24 now converts that mounted bundle inside the job
output cache into the Diffusers `AceStepPipeline` layout before loading the
pipeline. This is required because the raw Transformers custom model exposes
ACE acoustic-latent generation, while the benchmark needs the full Diffusers
pipeline so the VAE produces comparable WAV audio. The conversion records its
input layout, selected DiT variant, output folder, elapsed time, and converted
file preview in the job report; it does not rewrite the mounted datastore
checkpoint. Adapter loading then tries the pipeline-level LoRA loader first and
falls back to component-level PEFT attachment on common diffusion targets such
as `transformer`. If the adapter is missing or cannot be loaded from the deployable
`checkpoints/trainable_delta.pt` / Side-Step LoRA artifacts, the benchmark must
fail explicitly rather than silently benchmarking the base ACE checkpoint as the
Hybrid lane.

When Step 24 falls back to PEFT component attachment, it must not leave the
generic `PeftModel` wrapper as `pipe.transformer`. That wrapper has a
text-model-style forward signature and can pass `input_ids` into ACE's DiT,
which fails because `AceStepTransformer1DModel.forward()` expects ACE diffusion
kwargs such as `hidden_states`, `timestep`, `encoder_hidden_states`, and
`context_latents`. The runner therefore loads the adapter through PEFT, activates
the `cara_hybrid` adapter, then unwraps back to the injected native base model
before generation.

Cost guardrail: ACE-Step/Side-Step work must use existing Azure ML workspace
resources, approved computes, datastores, environments, and command jobs only.
Do not add Marketplace resources or paid external training services without
explicit approval.

## Context Diffusion Branch

`Finetune: Context Diffusion` is a new append-only Stable Audio follow-on
branch. It does not replace the completed Diffusion CARA-Strong ladder. It
inherits steps 01-09 from the original Stable Audio branch, then adds a separate
context-conditioning experiment that can be compared side-by-side against:

- original Stable Audio CARA-Strong diffusion
- MusicGen autoregressive CARA-Strong
- ACE-Step hybrid, once available
- base checkpoints and retrieval/probe controls

The methodological change comes from Context Diffusion-style conditioning:
context examples should enter the model as first-class conditioning tokens
beside text, not as prompt text or sidecar labels. For Stable Audio, the
translation is audio-domain rather than image-domain: use frozen audio
autoencoder/latent summaries from selected source-disjoint audio context
examples, project them to the DiT cross-attention conditioning dimension, and
concatenate them with the ordinary text/time conditioning path. Stable Audio
still operates over audio latents, not literal spectrogram image pixels.

Senior-high-school TLDR: the first Stable Audio CARA-Strong run is like teaching
the model a catalog number for each training sound while it learns from that
sound. The Context Diffusion branch adds a second clue: before generating or
scoring a target sound, the model also receives a few short, related example
sounds from the same CARA pool or family. Those examples are not written into
the prompt as words. They are converted into model-readable audio context tokens
and placed beside the normal text conditioning. If this works, the model should
recover CARA pools more often, especially at the family/genre fallback level,
because it can compare the target against nearby examples rather than relying
only on a pool id and prompt text. If shuffled or wrong-family context also
improves scores, that would be a warning sign that the method is leaking labels
or learning shortcuts rather than real audio-linked attribution.

Expected benchmark difference: compared with the original Diffusion branch, the
Context Diffusion branch should ideally improve pool top-k, correctly repairable
pool rate, and family fallback accuracy while keeping the unattributable rate
lower. The strongest positive result would be high performance for
`context_plus_prompt`, useful performance for `context_only`, and a clear drop
for `shuffled_context` and `mismatched_family_context`. That pattern would show
that the extra context is helping because it is relevant audio evidence, not
because the dashboard or manifest accidentally gave the answer away.

Inherited ladder evidence:

1. Lock manifest.
2. Confirm Azure upload.
3. Prepare Stable Audio dataset.
4. Stable Audio trainer preflight.
5. Baseline smoke.
6. CARA-lite smoke.
7. Attribution-head smoke.
8. CARA-Strong smoke.
9. Full Stable Audio CARA-Strong fine-tune.

New Context Diffusion branch stages:

10. Design CARA context packs: select 1-3 context examples per target from the
    same CARA pool and/or family, with source-disjoint guarantees. Each context
    pack must record target chunk id, target source id, context chunk ids,
    context source ids, split, pool/family ids, registry hash, and the pack
    selection policy. This is now submitted through
    `azureml/jobs/10_prepare_stable_audio_context_packs.yml`, which reads the
    prepared Stable Audio manifest and writes `context_pack_manifest.jsonl` plus
    `context_controls_manifest.jsonl`.
11. Prepare context conditioning cache: validate context WAV references and
    write `context_cache_manifest.jsonl` beside the context packs. The current
    implementation is an audio metadata/probe cache submitted through
    `azureml/jobs/11_cache_stable_audio_context_metadata.yml`; it is sufficient
    for the first full context-conditioned branch, which uses locked
    source-disjoint context-pack metadata as additional Stable Audio
    cross-attention conditioners. A stricter future branch can replace these
    metadata conditioners with frozen Stable Audio latent/audio-token context
    embeddings.
12. Context conditioner preflight: validate context projection shape,
    cross-attention concatenation, prompt/context dropout, and first-batch
    metadata before launching training. This is now submitted through
    `azureml/jobs/12_stable_audio_context_preflight.yml` and records the planned
    C+P, context-only, prompt-only, shuffled-context, and mismatched-family
    lanes.
13. Context Diffusion smoke: run a short context-conditioner smoke through
    `azureml/jobs/13_stable_audio_context_smoke.yml`. This consumes the locked
    context packs and cache, trains a small context-lane head over
    context-plus-prompt, context-only, prompt-only, shuffled-context, and
    mismatched-family lanes, and writes `context_smoke_metrics.json`,
    `context_lane_predictions.jsonl`, and `context_conditioner_contract.json`.
    This is valid evidence that the context/control artifacts are consumable and
    that the controls are wired, but it is not yet full Stable Audio DiT
    context-fine-tuning evidence.
14. Full Context Diffusion fine-tune: train only after the context smoke proves
    the context artifacts and controls are sound. The first implemented full
    branch is submitted through
    `azureml/jobs/14_full_stable_audio_context_trainer.yml` and reuses the
    Stable Audio CARA-Strong trainer with extra context-pack conditioners:
    `cara_context_pool_index`, `cara_context_pool_family_index`,
    `cara_context_policy_index`, and `cara_context_count`. These fields are
    injected as Stable Audio Tools integer conditioners into DiT
    cross-attention beside the original CARA pool/family conditioners. Full
    runs are GPU-only on the existing `gpu-fulltraining-h100` compute and use
    the disk-safe mounted-output `checkpoints/trainable_delta.pt` strategy
    rather than accumulating Lightning `.ckpt` files on the Azure node disk.
    Claim scope: this is a context-aware metadata-conditioning branch; it is
    more direct than prompt tags, but still less strict than raw audio-context
    latent cross-attention.
15. Context benchmark scoring: compare the full context branch against the
    original diffusion branch and autoregressive branch using the locked
    benchmark prompt set, repairability ladder, and prediction-example audit.
    This stage is launched from the shared `Testing` page, not as a new
    fine-tuning job. The Context Diffusion page marks step 15 as the active
    handoff once step 14 is complete and links to Testing so the same Step 15
    generated-audio benchmark and Step 16 attribution scoring workflow is reused
    for every model lane.

Required controls:

- `context_plus_prompt`: context examples plus ordinary prompt.
- `context_only`: context examples with prompt text dropped.
- `prompt_only`: ordinary prompt with context dropped.
- `shuffled_context`: context examples from another target/pool.
- `mismatched_family_context`: deliberately wrong-family context.
- `no_context_baseline`: original Diffusion CARA-Strong inference path.

Required metrics:

- `context_pool_top1`
- `context_pool_top3`
- `context_family_top1`
- `context_registry_valid`
- `context_correctly_repairable`
- `context_unattributable`
- `context_dropout_sensitivity`
- `context_shuffle_delta`
- `mismatched_context_failure_rate`
- `source_disjoint_context_violation_count`

The Context Diffusion dashboard page now submits the first three new branch
stages through `/api/training/context-diffusion/packs`,
`/api/training/context-diffusion/cache`, and
`/api/training/context-diffusion/preflight`. These stages use the same
H100-preferred preprocessing policy as the model dataset prep and MusicGen token
cache: submit to `gpu-smoke-h100` when both H100-backed compute targets are
free, otherwise fall back to `cpu-prep-cluster`. Context smoke/full training
must remain locked until the conditioner and training wrapper actually consume
context tokens in Stable Audio's DiT cross-attention path. Do not use
Marketplace endpoints, Marketplace deployments, or new paid external services
for this branch without explicit approval.

The Context Diffusion page uses cached local evidence for the inherited Stable
Audio full-run completion (`modest_arch_clgnkqrz4z` and the locked trained-model
URI used by the benchmark suite). This keeps step 10 submittable when Azure
live status reads are slow, while the new context jobs still record their own
Azure job names, output paths, and completion states.

Step 10 context-pack generation must use indexed deterministic context/control
selection. Avoid per-target full-manifest scans for shuffled or mismatched
controls; on the 73k Stable Audio chunk manifest that becomes quadratic and can
make a GPU job appear idle with a blank `std_log.txt`. The current script logs
manifest load, bucket indexing, periodic row progress, artifact writes, and final
status to Azure `std_log.txt`.

Steps 11 and 12 must follow the same operational lessons from step 10 even
though they are lighter-weight jobs. They should print progress to Azure
`std_log.txt`, write JSON-safe artifacts through the shared report sanitizer,
and record compute as `h100_preferred_else_cpu` because the dashboard may route
the job to `gpu-smoke-h100` or `cpu-prep-cluster` depending on live H100 use.
Step 13 is GPU-only on `gpu-smoke-h100` and intentionally has a narrower claim:
it consumes context tokens/features and validates context controls, but it does
not claim that the full Stable Audio DiT has learned from context until the
future full context trainer is implemented.

## Progress Monitoring

Azure ML jobs are durable cloud jobs. Closing the browser does not stop them.
Only the explicit hard-stop action in `Operations / Azure Runs` requests
cancellation.

Dashboard locations:

- `Operations / Azure Runs`: live Azure job status, Studio links, logs, metrics.
- Fine-tuning page, `Check Prep Progress`: read-only progress estimate from
  prepared WAV blobs already visible in the datastore.
- Fine-tuning page, `Refresh Gates`: read-only refresh of
  `/api/training/readiness`. It updates local gate state, active Azure smoke
  job visibility, preflight status, and the smoke-sequence ladder; it does not
  submit Azure jobs.
- Fine-tuning page, `Dataset Prep Progress`: chunk count, estimated duration
  processed, remaining chunks, elapsed time, estimated time left, and progress
  bar.
- `Operations / Azure Runs`: selected jobs show elapsed time. Training ETA is
  shown when MLflow exposes both current step and configured max steps.
- Fine-tuning page, `Training Run Progress`: reads Azure MLflow metric history
  for the latest Stable Audio trainer job, reports observed step versus
  configured `max_steps`, and estimates chunks seen plus epoch progress as
  `observed_step * batch_size` over the prepared train chunk count. This is
  read-only and does not affect the cloud job.
- `Documentation / Runbook & Logs`: reads the live markdown files
  `docs/cara_finetuning_runbook.md` and `docs/EXPERIMENT_LOG.md` through the
  whitelisted `/api/docs/markdown` endpoints. Use this view to inspect the
  current runbook and append-only experiment log from inside the dashboard.
  Refreshing the page reloads the files from disk; it does not create or modify
  Azure jobs.

Smoke trainer launch safety:

- Launching a Stable Audio smoke run requires typing an exact confirmation
  phrase in the dashboard.
- The backend rejects `/api/training/start` unless that phrase is present.
- Immediately after Azure ML accepts a job, the backend re-reads the serialized
  Azure command inputs. If `variant`, `dashboard_triggered`, or `run_name` do
  not match the dashboard request, the backend requests cancellation and rejects
  the launch response.
- The dashboard submits smoke jobs from a per-run materialized Azure job YAML
  rather than relying on Azure ML SDK mutation of scalar inputs loaded from the
  static YAML. This prevents Azure from silently falling back to the static
  baseline defaults.
- A job with mismatched tags and command inputs must not be counted as a valid
  protocol step.
- The smoke-sequence readiness ladder uses the same rule: a completed Azure job
  passes a protocol step only when the serialized command inputs match the
  registry event. Tags alone are not sufficient evidence.
- Historical `no_cara_baseline` jobs may predate the typed launch-confirmation
  guard. They can still count as baseline evidence when Azure shows the job
  completed on the expected environment and the serialized command variant is
  `no_cara_baseline`. CARA-lite and later stages require the strict
  dashboard-triggered input match.

The prep-progress estimate does not affect the running Azure ML job. It lists
prepared `.wav` blobs and compares their count and inferred PCM duration to the
expected chunk plan. Its ETA is estimated from elapsed time since the dashboard
submission event and current prepared-audio progress.

Submitting a dataset-prep job is not the same as completing dataset prep. The
fine-tuning page should remain in a `Dataset prep running` state after stage 3
submission until the read-only progress estimate reaches completion. Only then
should the page present the Stable Audio branch as ready for smoke-trainer prep.

## Current Azure ML Jobs

Dataset prep jobs:

```text
azureml/jobs/05a_prepare_stable_audio_dataset.yml
azureml/jobs/05b_prepare_musicgen_dataset.yml
azureml/jobs/05_prepare_model_datasets.yml
```

MusicGen token-cache job:

```text
azureml/jobs/06_cache_musicgen_encodec_tokens.yml
```

MusicGen page state:

- The dashboard caches the last successful model-specific preprocessing
  progress check in browser `localStorage`.
- The readiness payload also reads the latest model-specific preprocessing job
  status from Azure ML, so a completed job can unlock the next stage even before
  the slower blob-count progress check returns.
- The MusicGen page calls readiness with `variant=autoregressive`, so gate
  refresh checks MusicGen-specific status without polling Stable Audio
  preflight/smoke/full-run state in the background.
- Switching away from Autoregressive / MusicGen and returning should preserve
  the last checked step-03 status.
- Submitting a new MusicGen preparation job clears that cached progress so stale
  evidence cannot mark the new job complete.

Azure Runs progress:

- Operations / Azure Runs polls workspace jobs every 15 seconds and polls
  percentage progress every 60 seconds.
- MusicGen EnCodec cache progress is estimated by counting non-empty
  `musicgen_encodec_cache/tokens/**/*.pt` blobs already visible in the Azure
  datastore against the expected MusicGen chunk count.
- Stable Audio and MusicGen trainer progress uses Azure MLflow step metrics
  where available. If a job type has no reliable denominator, the page shows no
  percentage rather than inventing one.
- These checks are read-only and do not affect the running Azure ML job.

Stable Audio smoke trainer job:

```text
azureml/jobs/07a_stable_audio_trainer_preflight.yml
azureml/jobs/07_smoke_stable_audio_trainer.yml
azureml/jobs/09_full_stable_audio_cara_strong_trainer.yml
```

Run `07a_stable_audio_trainer_preflight.yml` before the H100 smoke trainer. The
preflight runs on `cpu-prep-cluster`, uses the same `env-stable-audio-tools`
version as the smoke trainer, checks the deep Stable Audio training imports, loads
the gated checkpoint config through Key Vault HF auth, removes the placeholder
`training.arc` block for the baseline smoke path, and constructs the plain
diffusion training wrapper without calling `trainer.fit`.

The smoke trainer uses `azureml:env-stable-audio-tools:8`, which keeps the
working v2 preprocessing environment separate and adds explicit training
dependencies such as `pytorch-lightning`, `torchmetrics`, `prefigure`, `wandb`,
`webdataset`, `auraloss`, `matplotlib`, `descript-audiotools`, `azure-ai-ml`,
`descript-audio-codec`, `encodec`, `inf-cl`, `laion-clap`, `azure-identity`,
and `azure-keyvault-secrets`.

The Stable Audio Open Small checkpoint is a gated Hugging Face model. A smoke
trainer run can reach CUDA and dataset loading but still fail before training if
the Azure job cannot authenticate to a Hugging Face account with accepted access
to `stabilityai/stable-audio-open-small`. That failure is reported as a
`GatedRepoError` while downloading `model_config.json`.

Hugging Face access has two separate requirements:

1. The Hugging Face account must have accepted/granted access in the browser.
2. The Azure ML job must receive a token from that account.

The dashboard submit path reads `HF_TOKEN` from local `.env`, writes it to the
workspace Key Vault as secret `hf-token`, and passes only `KEY_VAULT_URL` plus
`HF_TOKEN_SECRET_NAME=hf-token` to the Azure ML job. The trainer retrieves the
secret at runtime with the job's user identity and sets `HF_TOKEN` in-process
before calling Stable Audio Tools `get_pretrained_model`. Do not put the token
in YAML, job names, tags, command arguments, or logs.

If the dashboard reports that the current Azure identity cannot set Key Vault
secrets, grant that identity secret `set`/`get` permission on the workspace Key
Vault or create the `hf-token` secret manually in that vault and grant the
identity secret `get` permission. The dashboard will use the existing secret if
it can read it. The registered smoke-trainer environment is
`env-stable-audio-tools:8`.

The smoke trainer writes reports and checkpoints under:

```text
azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_smoke/
```

The preflight writes reports under:

```text
azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_preflight/
```

The dashboard records durable submissions in:

```text
registry/cara_strong/azure_training_jobs.jsonl
```

### Training Run Progress Endpoint

The dashboard progress endpoint is branch-aware:

```text
/api/training/run-progress?model=latest
/api/training/run-progress?model=diffusion
/api/training/run-progress?model=context_diffusion
/api/training/run-progress?model=musicgen
/api/training/run-progress?model=all
```

It selects submitted trainer jobs from `registry/cara_strong/azure_training_jobs.jsonl`, prefers an active run when one exists, and reports observed step, configured max steps, percent complete, latest loss, elapsed time, ETA, and Studio URL. Stable Audio-derived trainers read the run output artifact in the datastore first, currently `logs/stable_audio_smoke/version_0/metrics.csv`, because this survives browser reloads and does not depend on live MLflow metric availability. Compatible branches can fall back to Azure MLflow if no metrics artifact is available. These checks are read-only and do not affect or restart the Azure ML job.

For Context Diffusion full training, the Context Diffusion page uses:

```text
/api/training/run-progress?model=context_diffusion
```

This keeps the progress display reusable for peer-review reruns and future operators, instead of relying on browser-local state or manual Azure Studio inspection.

### Benchmark Testing Model Selection

Context Diffusion Step 15 hands off to the shared Testing page rather than a separate branch-specific evaluator. The generated-audio benchmark launcher can select one candidate model lane, several lanes, or all lanes:

- Diffusion CARA-Strong
- Context Diffusion CARA-Strong
- MusicGen CARA-Strong
- ACE-Step Hybrid CARA-Strong

Single-lane reruns are valid when a new model has just completed fine-tuning and the existing locked prompt set should be reused. Multi-lane and all-lane runs are valid for direct side-by-side comparisons. In all cases, the selected lanes must use the same locked prompt manifest and seed policy; changing the selected model set must not regenerate or replace the prompt set.

The Testing page presents the stages in order:

1. Lock prompts once.
2. Run an audio smoke for newly wired lanes.
3. Run full generated audio for one, several, or all ready lanes.
4. Run attribution scoring only after full generated audio exists.
5. Read the Benchmarks page; it does not submit another cloud job.

These stages do not all need to be launched separately for every rerun. For a
newly completed model such as ACE-Step Hybrid, run a small smoke first, then the
full audio benchmark for only that model if the prompt set is already locked.
Attribution scoring should be launched only for lanes with implemented native or
probe scorers. As of this note, Stable Audio-derived and MusicGen scorers exist;
the ACE-Step native scorer remains pending.

For ACE-Step Hybrid, generated audio and CARA evidence are intentionally
separated. Step 24 converts the mounted ACE-Step custom-code bundle into a local
Diffusers `AceStepPipeline`, attaches the Side-Step adapter, and writes WAV
files so the same prompt manifest can be compared across model families. It also
captures continuous denoising-latent summaries from the Diffusers
`callback_on_step_end` hook. Those latents are useful audit evidence that the
adapter-conditioned DiT path ran, and they can support a later native
attribution-head extractor, but they are not readable CARA codewords by
themselves. Hybrid CARA pool IDs must come from a real ACE native scorer or
external probe that decodes the model states into the locked registry.

The current ACE Diffusers release can try to sort boolean attention masks on
CUDA inside its condition encoder, which fails on the H100 PyTorch stack with
`Sort currently does not support bool dtype on CUDA`. The benchmark runner
therefore applies a narrow runtime compatibility patch to the ACE
`_pack_sequences` helper: only the mask sort key is cast to integer, while the
packed mask returned to the pipeline stays boolean. This patch does not change
the CARA labels, adapter weights, prompts, or benchmark scoring contract; it
only lets the converted ACE pipeline proceed past condition packing.

Live launches use the neutral typed confirmation phrase:

```text
LAUNCH AUDIO BENCHMARK
```

The older `LAUNCH ALL MODELS AUDIO BENCHMARK` phrase is still accepted by the backend for compatibility with previous scripts and retry paths, but the dashboard should show the neutral phrase because subset launches are now expected.

The live gate must be typed after the final launch choices are made. The
dashboard clears the confirmation field whenever the operator changes dry-run
mode, smoke/full scope, selected model lanes, selected suites, or prompt limit,
and clears it again after a successful live submit. The top-level live endpoint
rejects empty, stale, or legacy confirmation text before any Azure ML submission
can occur. Internal retry and child-job paths may still accept legacy phrases
only for compatibility with previously generated retry code.

### Attribution Scoring Model Selection

Step 16 is a separate scoring stage after generated audio has completed. It does
not regenerate WAV files. The Testing page now exposes the same model-lane
selection pattern for attribution scoring:

- Score all lanes present in the selected source audio run.
- Score only Diffusion CARA-Strong.
- Score only Context Diffusion CARA-Strong.
- Score only MusicGen CARA-Strong.
- Score only ACE-Step Hybrid CARA-Strong.

The backend validates that every selected scoring lane is actually present in
the selected source audio run. The Azure scoring job receives the selected
`model_ids` as an explicit input, and the scorer filters
`generation_manifest.jsonl` to those rows before writing
`native_predictions.jsonl`, `scored_generation_manifest.jsonl`,
`metrics_latest.json`, and the repairability matrices.

For Context Diffusion scoring, `context_trained_model_data` must point to the
completed timestamped Step 14 output folder containing
`checkpoints/trainable_delta.pt`, not merely to the parent
`stable_audio_context/context_full/` folder. The backend resolves this from the
latest `stable_audio_context_full_submitted` registry event and shows the URI in
the Step 16 dry-run preflight panel.

For ACE-Step Hybrid scoring, Step 25 is GPU-backed and native-extractor aware,
but it still refuses to emit numeric metrics unless the trained Hybrid artifact
contains a compatible native CARA attribution head. If the Side-Step LoRA output
has no such head, Step 25 writes `blocked_missing_ace_native_head` and keeps
Hybrid native cells non-numeric. This preserves the no-label-leakage rule:
expected held-out labels are never copied into prediction fields, and continuous
latent summaries are not treated as discrete CARA codewords.

When a Step 13 native head is present, Step 25 must decode its logits with the
full resolver saved inside `checkpoints/ace_attribution_head.pt`. The selected
benchmark generation manifest usually contains only a subset of the 98 locked
pools and may omit one or more families, so rebuilding a resolver only from
generated rows can be smaller than the trained classifier head. The generated
manifest resolver is kept only as an audit/fallback path; the scoring report
should record `native_head_resolver.source=native_head_checkpoint_resolver` for
the normal post-Step-13 path.

Scoring jobs write new output folders under the evaluation datastore. They do
not overwrite generated audio or previous score artifacts. A later score may
become the dashboard's latest displayed metrics for that source/model lane, so
the selected lane and source job should be checked in the scoring dry-run before
typing:

```text
LAUNCH ATTRIBUTION SCORING
```

The repairability ladder is derived from the selected scoring output; it is not
a third Azure launch step.

The Benchmarks page should aggregate scores as latest labelled evidence per
lane. A Context-only score run is allowed to update the Context Diffusion lane,
but its `no_labelled_rows` Diffusion/MusicGen placeholders must not replace
previous valid Diffusion or MusicGen metrics. If a lane reports `0/0`, treat it
as `no labelled rows`, not as a real zero-percent result.

For Context Diffusion native scoring, a completed scorer that emits
`prediction_status=exception` and no `predicted_pool_id` is an extractor/runtime
failure state. It is not a defensible conclusion that the trained model has no
CARA signal until the exception has been surfaced in the prediction examples
and the evaluation-time model reconstruction is verified to include the same
context conditioner/head path used during training.

Current scoring implementation requirement: Step 15 and Step 16 must rebuild
the fine-tuned Stable Audio inference graph before loading a trainable delta.
For Diffusion CARA-Strong this means installing the CARA integer conditioner
from the locked registry resolver before `checkpoints/trainable_delta.pt` is
applied. For Context Diffusion this means installing both the CARA conditioner
and the context metadata conditioners before the context trainable delta is
applied. If any CARA/context conditioner tensor remains unmatched, the
generated-audio or scoring job should fail loudly instead of silently running as
the base model.

Step 15 and Step 16 replay structured CARA metadata only for label-applicable
benchmark rows. They suppress it for open-quality rows and shuffled-label
controls, because those lanes are designed to test prompt/generalisation or
class-prior leakage rather than supervised target conditioning. This keeps the
controlled generated-audio benchmark aligned with the training path while
preserving the leakage controls.

If the native extractor catches runtime exceptions while replaying generation,
the lane status is `extractor_failed` when every prediction failed, or
`scored_with_extractor_errors` when only some predictions failed. These statuses
must not be reported as 0% attribution. They mean the evaluation extractor needs
repair before a model-level conclusion can be made.

The Stable Audio native extractor is generation-aware. During generated-audio
replay, the DiT may expand each requested sample into multiple internal branches
for guidance or other generation bookkeeping, so hidden-state tap features can
arrive with more rows than the requested benchmark batch. Step 16 now records
the observed tap shapes and aligns those duplicated branches by averaging them
per requested sample before passing the features into the trained CARA
attribution head. This keeps training-time tap strictness intact while making
generated-audio scoring auditable instead of silently reporting extraction
failures as zero attribution.

Step 16 also checks the trained attribution head's expected input width from the
saved `cara_attribution_head.backbone.0.weight` tensor. If generated-audio
replay produces fewer or more concatenated tap features than the checkpointed
head expects, the extractor pads or trims the feature vector to the checkpoint
width and records `feature_dim_mode`, `actual_feature_dim`, and
`expected_feature_dim` in the prediction's `feature_alignment` field. Treat this
as an evaluation-time compatibility audit note: it allows the scorer to emit
predictions from the trained head, but it should be visible beside the metrics
whenever interpreting generated-audio attribution strength.

Step 16 also surfaces the trainer's prepared-audio held-out evaluation artifact
when it is present in the trained model folder. These validation/test metrics
come from real prepared chunks with known manifest labels and are displayed on
the Benchmarks page as "Held-out prepared-audio evidence". They are not the same
as generated-audio repairability, and they should be interpreted as evidence
that the trained attribution head can read labelled held-out audio features
before asking whether newly generated audio carries recoverable CARA tags.

### ACE-Step Hybrid Native Scoring Boundary

ACE-Step Hybrid Step 25 is now a native-scorer contract, not just a manifest
audit. It mounts the generated-audio folder, the trained Hybrid Side-Step output,
and the ACE checkpoint bundle on the H100 scoring path. The scorer first looks
for a compatible trained native CARA attribution head, such as
`checkpoints/ace_attribution_head.pt` or `checkpoints/cara_attribution_head.pt`,
with `CaraAttributionHead` pool/family classifier weights. Only then may it
replay the ACE DiT path, tap hidden states, and write `native_predictions.jsonl`
plus numeric Hybrid native metrics.

If the trained Hybrid output only contains the Side-Step LoRA adapter delta, the
expected Step 25 result is `blocked_missing_ace_native_head`. This is not a
model score and must remain non-numeric in the benchmark matrices. It means the
next methodological step is to train/export an ACE native attribution head from
ACE hidden-state evidence. Do not repair this state by copying expected prompt
labels into prediction fields or by treating continuous latent summaries as
discrete CARA codewords.

### ACE-Step Hybrid Step 13: Native DiT Attribution Head

The completed ACE-Step Step 12 Side-Step LoRA run is necessary but not
sufficient for native CARA metrics. It adapts the Hybrid generation path, but it
does not by itself emit a registry-resolved CARA prediction. To make the Hybrid
lane comparable to the Diffusion and Context Diffusion lanes, run Step 13 after
Step 12 completes.

Step 13 loads the completed Step 12 Side-Step LoRA artifact, replays ACE-Step
generation on CARA-labelled tensor-manifest rows, taps ACE DiT hidden states,
and trains a `CaraAttributionHead` over the same locked pool/family registry.
It writes the copied generation delta plus the native head into a new
head-augmented artifact folder:

- `checkpoints/trainable_delta.pt`
- `checkpoints/ace_attribution_head.pt`
- `cara_registry_resolver.json`
- `ace_native_head_metrics.json`
- `ace_native_head_examples.json`
- `training_progress.json`

The default Step 13 prompt policy keeps the CARA codeword out of the visible
prompt (`include_cara_tag_in_prompt=false`). This is intentional. The Hybrid
claim being tested is whether the completed LoRA-conditioned ACE DiT path carries
recoverable CARA evidence that a native head can read, not whether the model can
copy a visible codeword from text.

After Step 13 passes, rerun Hybrid attribution scoring. The benchmark model
registry should use the Step 13 output folder for Hybrid native scoring, while
the copied Step 12 `trainable_delta.pt` remains the generation adapter inside
that folder. If Step 25 still reports `blocked_missing_ace_native_head`, inspect
the Step 13 output folder first; it should contain
`checkpoints/ace_attribution_head.pt`.

Step 25 should also report that it decoded with the native-head checkpoint
resolver. A mismatch such as a 98-pool / 9-family checkpoint being decoded with a
45-pool / 8-family generated-manifest resolver is a scorer wiring failure, not a
model result. Fix the resolver source before interpreting Hybrid attribution
metrics.
