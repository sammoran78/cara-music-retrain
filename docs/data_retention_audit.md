# Data Retention Audit

**Status:** deletion-candidate review  
**Inspected:** 2026-06-01  
**Scope:** top-level files and primary directories under `data/`  
**Action taken:** no files deleted

## Purpose

This audit flags stale or superseded files in `data/` without deleting them. The active workflow is manifest-backed:

```text
data/attribution_manifest.jsonl
```

The active training-oriented pool output is:

```text
data/cara_pool_manifest_v2.jsonl
```

Files are grouped below by retention policy so cleanup can be performed deliberately.

## Delete Now

These files are safe deletion candidates. They are stale, superseded, or operating-system debris and are not required by the active pipeline.

| File | Size | Reason |
| --- | ---: | --- |
| `data/.DS_Store` | 10 KB | macOS Finder metadata; not research data |
| `data/attribution_manifest.jsonl.backup_before_20k_restore_20260429_202418` | 64.3 MB | one-off pre-restore backup from 2026-04-29; superseded by the normalized canonical JSONL manifest |
| `data/freesound_attribution_enriched.csv` | 1.4 KB | failed early enrichment probe; superseded by the manifest-backed downloader and reconciliation flow |
| `data/freesound_attribution_enriched_progress.json` | 293 B | progress state for the failed early enrichment probe |
| `data/freesound_attribution_enriched_summary.json` | 422 B | stale summary showing `rows_with_valid_sound_id: 1` and `total_processed: 0` |
| `data/attribution_manifest_update_summary.json` | 101 B | stale Phase 3 manifest-update summary; superseded by later subset selection, replenishment, and genre normalization reports |

Expected deletion total:

```text
approximately 64.3 MB
```

## Archive Then Delete

These files are no longer authoritative, but retaining a compressed archive outside `data/` may be useful if the historical development path matters for the thesis audit trail.

| File | Size | Reason |
| --- | ---: | --- |
| `data/old_freesound_filtered_sources.csv` | 12 KB | legacy prefilter output; row state has already been backfilled into `data/attribution_manifest.jsonl` |
| `data/old_freesound_prefilter_progress.json` | 40 KB | legacy progress state; superseded by manifest flags and `data/download_progress.json` |
| `data/old_freesound_rejected_sources.csv` | 719 KB | legacy rejected-row output; retained only for the historical `scripts/backfill_prefilter_status.py` migration helper |
| `data/cara_pool_manifest.jsonl` | 60.6 MB | v1 allocator output; superseded for current training work by `data/cara_pool_manifest_v2.jsonl` |
| `data/cara_pool_manifest.csv` | 26.6 MB | derived CSV export of the superseded v1 allocator output |

Expected archive-and-delete total:

```text
approximately 88 MB
```

The v1 pool allocator implementation can remain in source control for comparison. Only its generated `data/cara_pool_manifest.*` outputs are superseded.

## Regenerate Before Publication

These files should not be deleted merely because they are older. They are useful audit artifacts, but they should be regenerated after the final subset and pool-allocation run so their statistics describe the publication dataset exactly.

| File | Reason |
| --- | --- |
| `data/music_subset_candidates.csv` | ranked output from the earlier 20,000-row selector run; retain as an intermediate audit artifact or regenerate for the final selection |
| `data/music_subset_selection_summary.json` | summary of the earlier 20,000-row selector run; retain for methods history, but do not present as the final 25,000-row subset composition |
| `data/music_subset_replenish_additions.csv` | records the 290 replacement rows used during replenishment; retain as an audit artifact |
| `data/music_subset_replenish_report.json` | records replenishment from the earlier selector state to the 25,000-row working set; retain as an audit artifact |
| `data/pool_analytics_report.json` | generated before the latest normalization and v2 allocation state; regenerate before quantitative reporting |
| `data/genre_normalization_report.json` | retain the current report as evidence of normalization; regenerate if the manifest changes again |

## Keep

These files and directories remain part of the active workflow or the reproducibility record.

| File or directory | Reason |
| --- | --- |
| `data/attribution_manifest.jsonl` | canonical row-level source of truth |
| `data/attribution_manifest.csv` | derived human-readable export of the canonical JSONL manifest |
| `data/freesound_dataset_attribution.csv` | local copy of the original Stability AI attribution source |
| `data/attribution_list.json` | parsed original Freesound ID list; reproducibility artifact |
| `data/attribution_master_summary.json` | original-source parse summary |
| `data/attribution_seed_labels.csv` | offline CARA seed labels; still used by replenishment |
| `data/attribution_seed_labels_high_conf.csv` | high-confidence seed export; useful for audit and later sidecar work |
| `data/attribution_seed_labels_summary.json` | summary of the full offline labeling pass |
| `data/download_progress.json` | active resumable download state |
| `data/freesound_api_usage.json` | persisted API throttling state |
| `data/unavailable_freesound.csv` | active unavailable-file log used by retry and reconciliation logic |
| `data/cara_pool_manifest_v2.jsonl` | current training-oriented v2 pool manifest |
| `data/cara_pool_manifest_v2.csv` | derived inspection export of the v2 pool manifest |
| `data/freesound/` | downloaded audio used for inspection and future training |
| `data/freesound_meta/` | local API metadata cache used by reconciliation and allocation |

## Cleanup Summary

The immediately deletable and archive-then-delete groups together account for approximately:

```text
152 MB
```

This is small compared with the approximately `92 GB` downloaded audio corpus. The main value of the cleanup is clarity rather than disk recovery: it reduces the chance that a historical v1 manifest or failed enrichment snapshot is mistaken for the current controlled dataset.

## Suggested Cleanup Sequence

1. Delete the `Delete Now` files.
2. Create a compressed archive outside `data/` for the `Archive Then Delete` files if historical reconstruction is desired.
3. Delete the archived originals from `data/`.
4. Regenerate `data/pool_analytics_report.json` after the final approved subset and v2 allocation run.
5. Keep using `data/attribution_manifest.jsonl` as the controlled source of truth and `data/cara_pool_manifest_v2.jsonl` as the current training-oriented allocation output.
