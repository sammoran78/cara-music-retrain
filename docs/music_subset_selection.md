---
description: Official methodology for selecting the music-focused Freesound training subset
---

# Music-Focused Freesound Subset Selection

## Objective

This document describes the official offline procedure used to derive a music-focused subset from the larger Freesound attribution manifest for CARA attribution experiments. The intent is to select a subset that is materially more useful for music attribution and pool discrimination than a naive random sample or a confidence-only sample, while remaining fully reproducible from local manifest fields and avoiding additional Freesound API calls during subset construction.

The procedure is designed to prefer musically structured material and reduce the prevalence of entries dominated by ambient soundscapes, generic sound effects, environmental recordings, and other non-musical content that are less likely to support attribution learning.

## Canonical Inputs and Outputs

### Canonical input

The canonical input is:

- `data/attribution_manifest.jsonl`

This manifest is treated as the authoritative local state for Freesound rows and contains provenance fields, offline CARA labeling outputs, subset flags, enrichment placeholders, and local download state.

### Canonical implementation

The official subset selector is:

- `data_pipeline/03b_select_music_subset.py`

### Canonical outputs

The selector produces:

- `data/music_subset_candidates.csv`
- `data/music_subset_selection_summary.json`

When invoked with manifest write-back enabled, it also updates:

- `data/attribution_manifest.jsonl`

## Offline Signals Used

The subset is constructed using only fields already present in the manifest after prior offline processing. No Freesound API calls are required.

The current implementation uses the following manifest fields:

- `source`
- `source_id`
- `license_normalized`
- `prefilter_status`
- `cara_tier1`
- `cara_tier2`
- `cara_primary_pool`
- `cara_auto_label_bucket`
- `cara_auto_label_confidence`
- `cara_auto_label_score`
- `cara_matched_keywords_json`
- `api_current_duration_s`
- `include_in_subset`

These fields collectively provide enough information to exclude obviously non-target content, estimate musical relevance, and maintain diversity across the final subset.

## Candidate Gating

A row is eligible for selection only if all of the following are true:

- The row source is `freesound`.
- `license_normalized` is one of the currently allowed licenses:
  - `cc0`
  - `cc-by`
  - `sampling+`
- `cara_primary_pool` is non-empty.
- `cara_auto_label_bucket` meets or exceeds the current minimum threshold.
- `prefilter_status` is not explicitly `rejected`.
- `cara_tier1` is not in the excluded-tier list.

### Current minimum bucket threshold

- `medium`

### Current excluded tiers

- `Sound Effects`
- `Field Recording`
- `Unclassified`

These exclusions are intended to remove rows that are likely to be weak attribution signals for music-focused training, even if they are valid audio examples in a broader sense.

## Included Music-Oriented Tiers

The current included tiers are:

- `Electronic`
- `Percussion/Drums`
- `Acoustic/Folk`
- `Classical/Orchestral`
- `Jazz/Blues`
- `Ambient/Drone`
- `Experimental/Noise`
- `Voice/Vocal`

This list is intentionally broader than the target-heavy classes. Some weaker or noisier musical categories are still admitted so they can be down-weighted or capped rather than categorically removed, which preserves some stylistic breadth while keeping the subset musically oriented overall.

## Scoring Procedure

Each eligible row receives a composite score used for global ranking.

### Score components

The current score combines:

- `cara_auto_label_bucket`
  - converted to an ordinal rank (`none`, `low`, `medium`, `high`)
- `cara_auto_label_confidence`
- `cara_auto_label_score`
- `cara_matched_keywords_json`
  - used as evidence of stronger textual alignment with the assigned tier
- `api_current_duration_s`
  - medium and longer clips receive a small positive bonus
  - very short clips receive a small penalty
- `prefilter_status`
  - `confirmed` receives a positive bonus
  - `rejected` receives a strong penalty
- `include_in_subset`
  - existing inclusion receives a small continuity bonus
- explicit tier bias
  - positive bias for more structured musical tiers
  - negative bias for tiers that tend to be less useful for music attribution when overrepresented

### Current tier bias direction

The current implementation promotes:

- `Electronic`
- `Percussion/Drums`
- `Acoustic/Folk`
- `Classical/Orchestral`
- `Jazz/Blues`

The current implementation demotes:

- `Ambient/Drone`
- `Experimental/Noise`
- `Voice/Vocal`

The purpose of this bias is not to exclude those categories entirely, but to reduce the chance that they dominate the highest-ranked rows when the goal is attribution training on musically structured content.

## Balancing Procedure

After candidate gating and scoring, rows are sorted globally by descending composite score and selected in multiple passes.

### Pass 1: fill weighted tier targets

The first pass attempts to fill a target number of examples per tier. These targets are derived from explicit per-tier weights.

### Pass 2: fill remaining tier capacity from deferred rows

Rows deferred only because the first-pass tier target was reached are revisited. This allows the selector to keep filling strong examples while still encouraging a target composition.

### Pass 3: relax soft diversity constraints to fill the requested subset size

If the target size is still not reached, the selector performs a final fill pass. This pass relaxes soft diversity constraints while still respecting hard tier caps. The purpose is to avoid returning an undersized subset when there are enough good candidates overall.

## Diversity Controls

The selector uses diversity controls at several levels.

### Per-tier weighting

Current official weights for the tightened run are:

- `Electronic`: `0.31`
- `Percussion/Drums`: `0.25`
- `Acoustic/Folk`: `0.18`
- `Classical/Orchestral`: `0.11`
- `Jazz/Blues`: `0.08`
- `Ambient/Drone`: `0.04`
- `Experimental/Noise`: `0.02`
- `Voice/Vocal`: `0.01`

These weights reflect the current preference for material with stronger rhythmic, melodic, instrumental, or ensemble structure.

### Hard tier caps

Current hard cap fractions are applied to:

- `Ambient/Drone`: `0.07`
- `Experimental/Noise`: `0.04`
- `Voice/Vocal`: `0.025`

These caps prevent ambient-heavy, noise-heavy, or vocal-heavy material from dominating the final subset even when many such rows score well under other criteria.

### Soft pool and license caps

Before the final fill pass, the selector also applies soft caps to:

- per-pool representation
- per-license representation

This reduces over-concentration in a handful of dominant source-license-genre pools and keeps the selected subset more varied across the available material.

## Official Manifest Write-Back Policy

When the selector is run with manifest write-back enabled, the selected rows are marked in the canonical manifest with:

- `include_in_subset = true`
- `subset_role = "music_train_candidate"`
- `subset_note = "balanced_music_selector_v1"`

Rows previously marked with these exact flags but not present in the latest official selection have those same flags cleared. This keeps the manifest synchronized with the most recent approved subset definition and avoids stale subset annotations.

## Current Official 20k Run

The current official tightened run was executed with a target size of `20,000`.

### Candidate pool

- eligible candidates: `63,122`
- selected rows: `20,000`

### Selected tier counts

- `Percussion/Drums`: `5,597`
- `Electronic`: `4,309`
- `Acoustic/Folk`: `3,652`
- `Jazz/Blues`: `2,493`
- `Classical/Orchestral`: `2,215`
- `Ambient/Drone`: `859`
- `Experimental/Noise`: `622`
- `Voice/Vocal`: `253`

### Selected license counts

- `cc-by`: `9,886`
- `cc0`: `9,000`
- `sampling+`: `1,114`

### Manifest update counts

- rows selected and marked: `20,000`
- rows deselected from the prior subset state: `8,454`

## Reproducibility and Reviewability

This procedure is intended to be suitable for peer review because it is:

- deterministic for a fixed manifest state
- based on auditable manifest columns
- independent of online API calls during subset construction
- summarized in both CSV and JSON outputs
- written back into the canonical manifest with explicit subset metadata

Any reviewer can inspect the manifest, the selector script, the selection summary, and the exported candidate CSV to understand why rows were admitted or excluded.

## Recommended Review Knobs

If later review concludes that the subset remains too ambient-heavy, noise-heavy, vocal-heavy, or otherwise imbalanced, the intended adjustment knobs are:

- included-tier weights
- hard tier cap fractions
- explicit tier bias in row scoring
- minimum CARA bucket threshold
- excluded-tier list
- soft pool and license caps

These parameters can be changed without altering the manifest schema or re-running API enrichment.

## Example command

```bash
python3 data_pipeline/03b_select_music_subset.py \
  --target-size 20000 \
  --update-manifest \
  --subset-role music_train_candidate \
  --subset-note balanced_music_selector_v1 \
  --summary-output data/music_subset_selection_summary.json \
  --output-csv data/music_subset_candidates.csv
```
