# CARA Freesound Sub-Dataset Construction Methodology

**Status:** working methodology document  
**Last updated:** 2026-05-29  
**Repository:** `cara-music-retrain`  
**Primary manifest:** `data/attribution_manifest.jsonl`  
**Human-readable export:** `data/attribution_manifest.csv`

## 1. Purpose

This document describes the current reproducible procedure used to construct a music-focused Freesound sub-dataset for CARA attribution experiments with Stable Audio Open Small. It is written as a methodology note for peer review: a reviewer with access to this repository, the public Stability attribution CSV, and appropriate Freesound credentials should be able to inspect, rerun, and audit the major data construction steps.

The purpose of the sub-dataset is not to recreate the entire Stable Audio Open Small training corpus. The purpose is to derive a controlled, inspectable, music-oriented subset from the original Freesound training attribution list, add CARA pool labels and subset inclusion decisions, and maintain those decisions in a central manifest that can later drive fine-tuning, sidecar generation, attribution-head training, and evaluation.

This document is intentionally written as a living methods chapter. It records the procedure that produced the current local research subset, and it leaves explicit placeholders for future sections on diffusion fine-tuning, autoregressive fine-tuning, checkpoint handling, attribution-persistence evaluation, and model-comparison results.

## 1.1 What Has Been Built So Far

The repository currently contains a reproducible data-construction workflow with the following completed components:

1. a direct parser for Stability AI's Freesound attribution CSV,
2. a canonical JSONL manifest that preserves original attribution provenance and later local annotations,
3. an offline CARA pool ontology and keyword labeler,
4. high-confidence CARA pool seed exports,
5. a deterministic music-focused subset selector,
6. a Freesound downloader and reconciliation path that writes local download state back to the manifest,
7. a replenishment script that replaces failed or unavailable candidates,
8. pool analytics for estimating clip-hours and pool viability,
9. a v2 pool allocator that converts the working subset into broad CARA source pools,
10. a read-only Pool Viewer interface for browsing the resulting v2 pool manifest with metadata and local audio playback.

The fine-tuning experiments themselves are not claimed as complete in this document. The completed contribution documented here is the construction, auditing, and allocation of the sub-dataset that will be used by those experiments.

## 2. Data Source

The source universe is the Freesound attribution CSV published by Stability AI for Stable Audio Open Small:

```text
https://info.stability.ai/hubfs/freesound_dataset_attribution2%20(1).csv?hsLang=en
```

The CSV columns observed during implementation were:

```text
id,title,author,license,url
```

The repository parses this CSV directly rather than scraping the public attribution web page. The direct CSV parse produced:

```text
total_rows: 472,618
unique_freesound_ids: 472,618
duplicate_rows: 0
rows_missing_sound_id: 0
rows_with_id_url_mismatch: 0
```

License counts from the source attribution CSV were:

```text
cc0: 266,324
cc-by: 194,840
sampling+: 11,454
```

These figures are recorded in `data/attribution_master_summary.json`.

## 3. Central Manifest Design

The current controlled data object is a line-delimited JSON manifest:

```text
data/attribution_manifest.jsonl
```

Each line is one JSON object representing one Freesound source item or working candidate. JSONL was adopted because the manifest stores nested state that is awkward and fragile in CSV, including candidate pool lists, matched keyword dictionaries, soft-target arrays, API metadata, download status, sidecar paths, and subset flags.

The CSV file:

```text
data/attribution_manifest.csv
```

is treated as a derived inspection export. It is useful for spreadsheet review, but the JSONL file is the intended source of truth for controlled row-level state.

The shared manifest utilities are implemented in:

```text
data_pipeline/manifest_utils.py
```

These helpers provide JSONL loading, JSONL saving, `source_id` indexing, update merging, and CSV export.

The methodological reason for treating the JSONL file as canonical is that the research object is no longer just an attribution table. It is a provenance-preserving state record. A row can simultaneously represent original model-training membership, local API enrichment state, download availability, CARA labels, subset inclusion, pool allocation, and future sidecar paths. Line-delimited JSON allows these row-level states to be updated incrementally while remaining auditable in version control and process logs.

## 4. Manifest Schema

The manifest contains four broad classes of fields.

### 4.1 Original Attribution Provenance

These fields preserve the source relationship to the original Stable Audio Open Small attribution list:

```text
source
source_id
raw_id
url_sound_id
id_matches_url
title
title_stem
author
license_raw
license_normalized
license_display
url
file_extension
original_training_dataset
original_training_manifest
originally_in_stable_audio_open_small
```

### 4.2 API and Metadata Enrichment

These fields are reserved for Freesound API state, local metadata, and analysis-derived descriptors:

```text
api_enrichment_status
api_last_checked_utc
api_current_name
api_current_license_raw
api_current_license_normalized
api_current_tags_json
api_current_description
api_current_duration_s
api_current_samplerate
api_current_channels
api_analysis_available
api_bpm
api_key
api_voice_instrumental
api_genre_inferred
api_error_message
prefilter_status
prefilter_reason
```

### 4.3 CARA Attribution Fields

These fields hold pool labels, confidence scores, and future codebook outputs:

```text
cara_label_status
cara_label_source
cara_label_updated_utc
cara_tier1
cara_tier2
cara_primary_pool
cara_candidate_pools_json
cara_soft_targets_json
cara_family_codeword
cara_codeword
cara_auto_label_score
cara_auto_label_confidence
cara_auto_label_bucket
cara_matched_keywords_json
```

### 4.4 Fine-Tuning Control Fields

These fields determine whether a row is active in the working fine-tuning subset and where local files are located:

```text
include_in_subset
subset_role
subset_note
download_status
local_audio_path
local_meta_path
local_sidecar_path
content_fingerprint
manifest_notes
```

## 5. Step-by-Step Reproduction Procedure

The current sub-dataset construction can be reproduced in the following stages.

### 5.1 Fetch and Normalize the Stability Attribution CSV

Script:

```text
data_pipeline/01_fetch_attribution_list.py
```

Function:

1. Downloads or reads the Stability Freesound attribution CSV.
2. Extracts canonical Freesound IDs.
3. Confirms agreement between the explicit `id` field and the Freesound URL.
4. Normalizes license strings.
5. Creates the initial manifest rows.
6. Writes the ID list and summary outputs.

Representative command:

```bash
python3 data_pipeline/01_fetch_attribution_list.py
```

Primary outputs:

```text
data/attribution_manifest.jsonl
data/attribution_manifest.csv
data/attribution_list.json
data/attribution_master_summary.json
```

The original implementation also used `data_pipeline/11_manifest_csv_to_jsonl.py` to migrate an earlier CSV manifest to JSONL. In the current workflow, JSONL creation is handled directly by `01_fetch_attribution_list.py`.

### 5.2 Define Experimental CARA Pool Ontology

Ontology file:

```text
registry/experimental_pool_ontology.json
```

This ontology defines broad source-license-genre CARA pool categories for offline labeling. It includes:

1. allowed licenses,
2. tier-1 genre categories,
3. positive keywords,
4. boosted keywords,
5. anti-keywords,
6. confidence thresholds.

The initial ontology includes categories such as:

```text
Ambient/Drone
Percussion/Drums
Voice/Vocal
Field Recording
Sound Effects
Electronic
Acoustic/Folk
Experimental/Noise
Jazz/Blues
Classical/Orchestral
```

The pool naming convention is:

```text
{Source}-{License}-{Genre}
```

Example:

```text
Freesound-CC0-Electronic
Freesound-CC-BY-Percussion/Drums
Freesound-CC-Sampling+-Ambient/Drone
```

### 5.3 Offline CARA Labeling

Script:

```text
data_pipeline/03_offline_pool_labeler.py
```

Function:

1. Reads rows from `data/attribution_manifest.jsonl`.
2. Builds a search text from `title_stem`, `title`, `author`, and `url`.
3. Scores each ontology category using keyword matches, boosted keyword matches, and anti-keyword penalties.
4. Calculates a confidence value from the top score, total positive score mass, and top-vs-second-place margin.
5. Assigns a confidence bucket: `none`, `low`, `medium`, or `high`.
6. Writes label CSVs for audit.
7. Updates the JSONL manifest with CARA label fields and subset flags.

Representative command:

```bash
python3 data_pipeline/03_offline_pool_labeler.py
```

Audit outputs:

```text
data/attribution_seed_labels.csv
data/attribution_seed_labels_high_conf.csv
data/attribution_seed_labels_summary.json
registry/experimental_pool_definitions.json
```

The initial full offline labeling pass over the 472,618-row attribution universe produced:

```text
high: 16,555
medium: 83,330
low: 57,308
none: 315,425
```

These counts are recorded in `data/attribution_seed_labels_summary.json`.

## 6. High-Confidence Label Definition

A row is considered high confidence when:

1. the ontology scorer assigns a non-empty `cara_primary_pool`,
2. the top genre score is sufficiently large,
3. the calculated confidence exceeds the ontology high-confidence threshold,
4. the winning category has enough separation from competing categories,
5. the row's license is in the allowed license set.

The confidence score combines:

```text
top genre score / total positive score mass
top-vs-second-place margin
```

This means a row with many ambiguous keyword hits can score lower than a row with fewer but cleaner, more diagnostic matches.

Examples of high-confidence assignments include:

```text
unusual synthetic drone.wav -> Freesound-CC-Sampling+-Ambient/Drone
90 bpm ATTACK LOOP 3 drums mixdown mastered 16 bit.wav -> Freesound-CC-BY-Percussion/Drums
light rain in forest.wav.WAV -> Freesound-CC-BY-Field Recording
bassline_oh.wav -> Freesound-CC-BY-Electronic
violin.open.strings.chords.wav -> Freesound-CC-BY-Classical/Orchestral
ind white noise flange.wav -> Freesound-CC-BY-Experimental/Noise
```

## 7. Music-Focused Subset Selection

The initial high-confidence set was useful but not sufficient as a fine-tuning subset, because it included many non-musical or weakly musical sources such as field recordings, sound effects, and unclassified material. A second selector therefore constructs a music-focused subset.

Script:

```text
data_pipeline/03b_select_music_subset.py
```

Detailed notes:

```text
docs/music_subset_selection.md
```

Function:

1. Reads the JSONL manifest.
2. Filters candidates by source, license, CARA tier, bucket threshold, pool availability, and rejection state.
3. Excludes categories that are not central to music-focused fine-tuning:
   `Sound Effects`, `Field Recording`, and `Unclassified`.
4. Scores eligible rows using label confidence, auto-label score, keyword evidence, duration, prefilter status, and tier bias.
5. Selects rows using tier targets, tier caps, pool caps, and license caps.
6. Writes the selected candidate CSV.
7. Optionally writes inclusion flags back to the JSONL manifest.

Representative command:

```bash
python3 data_pipeline/03b_select_music_subset.py --target-size 20000 --update-manifest
```

Primary outputs:

```text
data/music_subset_candidates.csv
data/music_subset_selection_summary.json
data/attribution_manifest.jsonl
```

The recorded 20,000-row selector run produced:

```text
candidate_count: 29,192
selected_count: 20,000
```

Selected license counts:

```text
cc-by: 10,126
cc0: 7,549
sampling+: 2,325
```

Selected tier counts from that run included:

```text
Percussion/Drums: 10,449
Acoustic/Folk: 2,725
Electronic: 2,179
Ambient/Drone: 1,400
Classical/Orchestral: 1,132
Jazz/Blues: 815
Experimental/Noise: 800
Voice/Vocal: 500
```

These figures are recorded in `data/music_subset_selection_summary.json`.

## 8. Downloading and Reconciling Local Assets

### 8.1 Downloader

Script:

```text
data_pipeline/02_freesound_downloader.py
```

Function:

1. Reads candidate rows from the JSONL manifest.
2. Fetches Freesound metadata.
3. Optionally downloads original-quality audio.
4. Writes metadata JSON files.
5. Updates manifest fields such as:
   `download_status`, `local_audio_path`, and `manifest_notes`.

The downloader requires Freesound credentials and must respect Freesound API limits.

### 8.2 Reconciliation

Script:

```text
data_pipeline/12_reconcile_download_manifest.py
```

Function:

1. Reads local downloaded audio.
2. Reads local metadata JSON files.
3. Reads download progress state.
4. Reconciles local files against manifest rows.
5. Infers additional genre/style summaries from local metadata.
6. Updates manifest fields including local paths, download state, style tags, and subset state.

This reconciliation step is important because API calls and downloads are fallible. The manifest therefore records what actually exists locally rather than what was merely requested.

Representative reconciliation command:

```bash
python3 data_pipeline/12_reconcile_download_manifest.py --auto-subset
```

The `--auto-subset` option allows the reconciliation step to preserve or apply the default subset role where local download state demonstrates that a row is usable for the working training subset.

## 9. Replenishment to 25,000 Working Candidates

After downloads and reconciliation, some selected rows were unavailable, failed, or otherwise unsuitable. A replenishment step adds replacements while preserving the target subset size.

Script:

```text
data_pipeline/03c_replenish_music_subset.py
```

Function:

1. Reads the manifest and download progress.
2. Identifies selected rows that failed or were unavailable.
3. Removes unsuccessful rows from the active subset.
4. Blocks authors with repeated unavailable items above a threshold.
5. Selects replacement rows using the same selection logic as `03b_select_music_subset.py`.
6. Optionally appends seed rows from `data/attribution_seed_labels.csv` if they are not already present in the working manifest.
7. Writes replacement rows and updates the manifest.

Representative command:

```bash
python3 data_pipeline/03c_replenish_music_subset.py --update-manifest
```

Recorded replenishment output:

```text
target_working_downloads: 25,000
current_candidates: 25,000
downloaded_candidates: 24,710
unsuccessful_candidates_removed: 290
replacements_needed: 290
replacements_selected: 290
shortfall_after_replenish: 0
```

These figures are recorded in `data/music_subset_replenish_report.json`.

## 10. Genre Label Normalization

After reconciliation and replenishment, the working subset is passed through an explicit genre-label normalization step. This step fixes historical slash-vs-space label drift before quantitative reporting, pool allocation, or fine-tuning export.

Script:

```text
data_pipeline/13_normalize_genre_labels.py
```

Shared normalization rules:

```text
data_pipeline/genre_normalization.py
```

Function:

1. Reads the canonical manifest at `data/attribution_manifest.jsonl`.
2. Targets rows in the active sub-dataset by default: `include_in_subset = true` or `subset_role = "music_train_candidate"`.
3. Normalizes CARA tier fields, inferred genre fields, pool-name suffixes, candidate-pool lists, style-summary prefixes, and genre style tokens.
4. Uses slash-form labels as the canonical reporting vocabulary.
5. Updates the JSONL manifest and derived CSV export.
6. Applies the same normalization to the current v2 pool manifest unless explicitly skipped.
7. Writes a machine-readable audit report.

Representative command:

```bash
python3 data_pipeline/13_normalize_genre_labels.py
```

Primary outputs:

```text
data/attribution_manifest.jsonl
data/attribution_manifest.csv
data/cara_pool_manifest_v2.jsonl
data/cara_pool_manifest_v2.csv
data/genre_normalization_report.json
```

The canonical label mappings include:

```text
Percussion Drums -> Percussion/Drums
Voice Vocal -> Voice/Vocal
Jazz Blues -> Jazz/Blues
Hip Hop Beats -> Hip-Hop/Beats
World Traditional -> World/Traditional
Rock Metal -> Rock/Metal
```

The current normalization run inspected `26,768` manifest rows, targeted `25,000` active sub-dataset rows, and changed `21,372` rows in at least one normalized field. It also normalized all `24,273` rows in `data/cara_pool_manifest_v2.jsonl`.

After normalization, the active sub-dataset uses one canonical spelling for each slash-form genre. For example, `Jazz Blues` and `Jazz/Blues` are no longer counted separately; they are reported together as `Jazz/Blues`.

## 11. Current Working Manifest State

As of the current repository state inspected on 2026-05-29:

```text
manifest_rows: 26,768
include_in_subset: 25,000
subset_role: music_train_candidate
downloaded rows: 25,002
unavailable rows: 660
not_downloaded rows: 1,106
```

The current manifest is therefore a working fine-tuning manifest rather than a complete 472,618-row copy of the original attribution universe. The full original-universe counts are preserved in summary files and can be regenerated from the public attribution CSV using `01_fetch_attribution_list.py`.

Current selected license counts:

```text
cc-by: 12,486
cc0: 10,203
sampling+: 2,311
```

Current selected tier counts include:

```text
Percussion/Drums: 7,730
Classical/Orchestral: 4,013
Electronic: 2,438
Acoustic/Folk: 2,247
Field Recording: 2,158
Ambient/Drone: 1,874
Experimental/Noise: 1,263
Jazz/Blues: 1,061
Voice/Vocal: 915
Hip-Hop/Beats: 455
Sound Effects: 350
Unclassified: 306
Rock/Metal: 147
World/Traditional: 43
```

The active sub-dataset tier names are now normalized before reporting. The selector still accepts historical aliases so older manifests remain readable, but the controlled manifest and v2 pool manifest are expected to use canonical slash-form labels after `13_normalize_genre_labels.py` is run.

## 12. Pool Analytics

Pool-level feasibility analysis is implemented in:

```text
data_pipeline/04_pool_analytics.py
```

The current analytics report is:

```text
data/pool_analytics_report.json
```

This report estimates available raw audio hours, usable clip counts, and pool coverage for model-specific clip durations. In the current report:

```text
downloaded_files: 19,617
raw_audio_hours: 164.818
model_a_11_88s usable_hours_total: 143.322
model_b_30s usable_hours_total: 126.45
```

This analysis is intended to support later decisions about model-specific pool construction and minimum viable pool duration.

## 13. CARA Source-Pool Allocation

The selected and downloaded sub-dataset is not only a list of audio files. For the CARA research question, each training item must also be associated with a source-pool identity that can later be encoded in sidecars, prompts, targets, or evaluation labels. The repository therefore includes a second-stage allocator that converts the working `music_train_candidate` subset into a training-oriented CARA pool manifest.

### 13.1 Pool Allocator v1

The first allocator is implemented in:

```text
data_pipeline/pool_allocator.py
```

It introduced the persistent allocation model: source assets are normalized, checked for duplicates, assigned to candidate pools, and written into a consolidated CARA training manifest. Its outputs are:

```text
data/cara_pool_manifest.jsonl
data/cara_pool_manifest.csv
registry/pool_allocator/
```

This path remains useful as a reference implementation, but the current preferred method for future training work is the v2 allocator.

### 13.2 Pool Allocator v2

The v2 allocator is implemented in:

```text
data_pipeline/pool_allocator_v2.py
```

The v2 allocator was added because the v1 path could fragment pools too finely around micro-style tags. For fine-tuning, that fragmentation risks creating many small, brittle labels rather than durable source-pool concepts. The v2 allocator therefore uses broad pool families first, then duration-packs each family into registered CARA source pools.

Current v2 pool families include:

```text
Atmosphere/Field
Percussion/Beats
Acoustic/Jazz/World
Tonal/Orchestral
Produced/Electronic
Experimental/Noise
Voice/Vocal
Sound Effects
Mixed/Unclassified
```

The current v2 registry namespace is:

```text
registry/pool_allocator_v2/
```

It contains persistent allocation state:

```text
assets.jsonl
pools.json
assignments.jsonl
duplicates.jsonl
runs.jsonl
progress.json
plan.json
```

The current v2 training manifest outputs are:

```text
data/cara_pool_manifest_v2.jsonl
data/cara_pool_manifest_v2.csv
```

As of the current inspected repository state, the v2 allocation run produced:

```text
candidate assets processed: 25,000
training rows written: 24,273
registered source pools: 98
fuzzy duplicate / review rows held out: 727
```

The current `data/cara_pool_manifest_v2.jsonl` row-level assignment statuses are:

```text
assigned: 24,175
new_pool_created: 98
```

The v2 allocator uses a 4-hour pool cap:

```text
max_pool_duration_seconds: 14,400
```

It also uses a 24-minute general artist cap:

```text
general_artist_cap_seconds: 1,440
```

When a broad family exceeds the 4-hour cap, the allocator creates spillover pools rather than creating a new pool for every micro-style signature. Exact identifier or content-fingerprint matches are treated as duplicates. Fuzzy title, artist, and duration matches are retained for review rather than silently entering the training manifest.

### 13.3 Pool Creator and Pool Viewer

The repository also includes GUI support for pool allocation and inspection.

The Pool Creator page is a Data-sidebar view that can launch and monitor pool allocation jobs. It uses backend endpoints in:

```text
gui/backend/main.py
```

The Pool Viewer page browses:

```text
data/cara_pool_manifest_v2.jsonl
```

as a read-only pool/file hierarchy. It supports pool drilldown, sortable columns, file metadata inspection, local audio playback, and browser-rendered waveform previews. This is methodologically useful because it allows qualitative inspection of pool coherence before training.

## 14. Stable Audio Tools Alignment

The sibling training toolkit is available locally at:

```text
../stable-audio-tools
```

The local revision previously recorded in `PLAN.md` was:

```text
50049e379e2fe8e35bffc99e57f23ade1c3471b7
```

The current project already contains compatibility scaffolding:

```text
model/stable_audio_integration.py
model/cara_metadata.py
model/prepare_dataset_config.py
evaluation/preencode_training_set.py
```

These modules prepare Stable Audio Tools dataset configs, custom metadata hooks, latent pre-encoding, and transformer hidden-state extraction. Future methodology sections should document the exact fine-tuning invocation, model configuration, checkpoint hashes, and evaluation schedule once those experiments are run.

## 15. Reproducibility Notes

A reviewer can reproduce the data construction logic from the repository by running the pipeline scripts in order. A minimal reproduction sequence is:

```bash
python3 data_pipeline/01_fetch_attribution_list.py
python3 data_pipeline/03_offline_pool_labeler.py
python3 data_pipeline/03b_select_music_subset.py --target-size 20000 --update-manifest
python3 data_pipeline/02_freesound_downloader.py --subset-mode subset_role --subset-role music_train_candidate --limit <N>
python3 data_pipeline/12_reconcile_download_manifest.py --auto-subset
python3 data_pipeline/03c_replenish_music_subset.py --update-manifest
python3 data_pipeline/13_normalize_genre_labels.py
python3 data_pipeline/04_pool_analytics.py
```

After the working subset has been downloaded and reconciled, the v2 pool manifest can be regenerated through the Pool Creator UI or by invoking the v2 allocator directly. The direct invocation path should be treated as implementation-specific and checked against `data_pipeline/pool_allocator_v2.py` and `gui/backend/main.py` at the commit under review, because pool-allocation arguments are still evolving more rapidly than the manifest and selector scripts.

The exact download outcome may differ over time because Freesound resources can be deleted, made unavailable, renamed, or rate-limited. The manifest records these outcomes through `download_status`, `api_enrichment_status`, `manifest_notes`, and local path fields. This is why the manifest is treated as a living audit object rather than merely a derived table.

For a strict peer-review reproduction, the following items should be archived or reported with the experimental result:

1. repository commit hash,
2. `../stable-audio-tools` commit hash or package version,
3. Freesound API access date range,
4. all generated summary JSON files,
5. manifest row count and subset row count,
6. v2 pool manifest row count and pool count,
7. exact fine-tuning dataset config,
8. model checkpoint identifiers and hashes.

## 16. Known Limitations

The current sub-dataset construction has several limitations that should be disclosed in peer-review materials.

1. The first-stage CARA labeling is text-based and offline. It does not initially listen to the audio.
2. The labeler relies heavily on filenames, author names, URLs, and metadata tags where available.
3. Freesound API limits restrict the rate at which full enrichment can be obtained.
4. Historical manifests may contain slash-vs-space label variants, but the current process includes `13_normalize_genre_labels.py` to normalize the active sub-dataset before final quantitative analysis.
5. The current working manifest is a curated fine-tuning manifest, not a complete mirror of the original Freesound attribution universe.
6. Some replacement rows were appended from seed labels during replenishment, which is recorded in `manifest_notes` and the replenishment report.
7. Download availability is historically unstable, so exact local audio availability is a time-dependent result.
8. The current v2 pool allocation is designed for broad source-pool stability, not for fine-grained musicological genre taxonomy.
9. The pool manifest is suitable for internal training/evaluation control, but it does not grant redistribution rights over downloaded Freesound audio.

## 17. Audit Trail

Key files for peer review are:

```text
data/attribution_master_summary.json
data/attribution_seed_labels_summary.json
data/music_subset_selection_summary.json
data/music_subset_replenish_report.json
data/genre_normalization_report.json
data/pool_analytics_report.json
data/attribution_manifest.jsonl
data/attribution_manifest.csv
data/cara_pool_manifest_v2.jsonl
data/cara_pool_manifest_v2.csv
registry/experimental_pool_ontology.json
registry/experimental_pool_definitions.json
registry/pool_allocator_v2/pools.json
registry/pool_allocator_v2/duplicates.jsonl
docs/music_subset_selection.md
```

The pipeline code most directly involved in sub-dataset construction is:

```text
data_pipeline/01_fetch_attribution_list.py
data_pipeline/03_offline_pool_labeler.py
data_pipeline/03b_select_music_subset.py
data_pipeline/03c_replenish_music_subset.py
data_pipeline/12_reconcile_download_manifest.py
data_pipeline/13_normalize_genre_labels.py
data_pipeline/genre_normalization.py
data_pipeline/04_pool_analytics.py
data_pipeline/pool_allocator_v2.py
data_pipeline/manifest_utils.py
```

The GUI code most directly involved in inspection and allocation review is:

```text
gui/backend/main.py
gui/frontend/src/pages/PoolCreatorPage.tsx
gui/frontend/src/pages/PoolViewerPage.tsx
gui/frontend/src/nav.ts
```

## 18. Planned Extensions

This document should be extended as the project moves into fine-tuning. The next methodology additions should cover:

1. final pool normalization and CARA codeword generation,
2. sidecar JSON generation for Stable Audio Tools training,
3. Stable Audio Open Small fine-tuning configuration,
4. any alternative model variants,
5. checkpoint pinning and hash reporting,
6. attribution persistence evaluation,
7. ablation comparisons between prompt-only, sidecar, codeword, and attribution-head approaches.
