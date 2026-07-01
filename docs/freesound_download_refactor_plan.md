# Freesound Download Refactor Plan

**Goal:** Reduce Freesound API call volume and latency when downloading the flagged subset for retraining, by (1) targeting only prefilter-confirmed rows, (2) reusing already-fetched metadata/analysis on disk, (3) batching metadata where possible, and (4) applying adaptive rate limiting.

**Status:** Draft for review. No code changes yet.

---

## 1. Confirmed Context

### Manifest already flags the subset
`data_pipeline/02a_freesound_prefilter.py` writes the following fields back into `data/attribution_manifest.jsonl`:

- `prefilter_status`: `"confirmed"` or `"rejected"`
- `prefilter_reason`: why it was confirmed/rejected

So the **download target set** is exactly:
- `row["source"] == "freesound"`
- `row.get("prefilter_status") == "confirmed"`

All other rows should be **skipped entirely** — no metadata, no analysis, no download.

### Metadata/analysis already on disk for confirmed rows
During prefilter, each confirmed `sound_id` has had `fetch_sound()` and `fetch_analysis()` called. Their results can be persisted to `data/freesound_meta/<id>.json` once, and then **reused** by the downloader instead of re-calling the API.

### Existing downloader behaviour today
`data_pipeline/02_freesound_downloader.py`:
- iterates every `source == "freesound"` row, ignoring `prefilter_status`
- always calls `fetch_sound()` + `fetch_analysis()` per item
- does not check whether a local audio file already exists before calling `download/`
- writes progress JSON and rewrites the entire manifest after every item

---

## 2. Problems This Causes

- **Redundant metadata calls** for rows that were already fetched during prefilter
- **Wasted calls on rejected rows** (only confirmed should be downloaded)
- **Unnecessary re-downloads** if the audio file already exists locally
- **Fixed sleep throttle** is conservative but not adaptive to Freesound's actual rate limit headers
- **Per-item manifest rewrite** slows throughput for large manifests

---

## 3. Proposed Refactor

### 3.1 Target only confirmed subset
- New filter in `download_freesound_subset`:
  - keep row only if `row.get("source") == "freesound"` AND `row.get("prefilter_status") == "confirmed"`
- Add a CLI flag `--require-confirmed/--no-require-confirmed` (default: require confirmed)
- Surface same filter in the GUI backend so `/api/data/download-progress` and `/api/data/download/start` operate on the confirmed subset only

### 3.2 Cache-first metadata strategy
- Before any API call for a given `sound_id`:
  - If `meta_dir/<id>.json` exists AND contains the fields we need, **use it**
  - Only call `fetch_sound()` if cache is missing
- Persist full metadata JSON (including `analysis` subobject when available) to `meta_dir/<id>.json`

### 3.3 Opt-in analysis, not always-on
- Add parameter `fetch_analysis: bool = False`
- Download flow does **not** need Essentia analysis — it's only needed for prefiltering, which has already happened
- This alone roughly halves API calls for any subset that reaches the downloader

### 3.4 Batch metadata endpoint
- Where metadata is genuinely missing for multiple IDs, use:
  - `GET /apiv2/sounds/?ids=a,b,c,...&fields=id,name,original_filename,license,tags,previews,download,duration,filesize,type,samplerate,channels`
- Wrap in a new client method `FreesoundClient.fetch_sounds_bulk(ids, fields)`
- Batch size: configurable, default 25
- Fall back to single-ID `fetch_sound()` if the bulk call errors for an individual ID

### 3.5 Skip existing audio files
- Before calling `download/`:
  - If `output_dir/<id>.<ext>` exists AND `filesize > 0` AND (optionally) matches metadata `filesize` field, mark as `downloaded` without calling the API
- Still records the row in `progress.json`

### 3.6 Dedupe by `source_id`
- Build a dict keyed by `str(source_id)` so duplicate manifest rows never cause duplicate API calls

### 3.7 Adaptive rate limiting
- Replace the blanket `time.sleep(rate_limit_delay)` after every call with a **token bucket**:
  - Configurable: `requests_per_minute` (default e.g. 60), `requests_per_day` (default e.g. 2000 — below Freesound's 60/min and 2000/day guidance, adjust to confirmed limits)
  - Persist today's count in a small JSON file (`data/freesound_api_usage.json`) so a restart doesn't lose the daily count
- Always honour `Retry-After` on 429 (already done)
- Back off proportionally to remaining daily budget

### 3.8 Batched manifest writes
- Write `progress.json` every item (cheap, small)
- Rewrite full manifest every N items (configurable, default 25) and once at the end
- Keep `unavailable` CSV append-only (already done)

### 3.9 `fields=` filter on single-sound endpoint
- Even when batch endpoint is not used, explicitly request only needed fields on `GET /sounds/{id}/` to reduce payload and reduce retries on timeouts

---

## 4. Code Changes Summary

### `data_pipeline/freesound_api.py`
- Add `fetch_sounds_bulk(ids: list[int], fields: list[str]) -> dict[int, dict]`
- Add `fetch_sound(sound_id, fields: list[str] | None = None)`
- Add token-bucket throttle around `_request`, with optional persistence
- Keep existing auth/refresh logic untouched

### `data_pipeline/02_freesound_downloader.py`
- New signature:
  ```
  download_freesound_subset(
      manifest_path, output_dir, meta_dir, unavailable_log, progress_path,
      limit=None, skip_audio=False,
      fetch_analysis=False,
      require_confirmed=True,
      bulk_metadata_batch_size=25,
      manifest_write_every=25,
  )
  ```
- Filter confirmed rows first, dedupe by `source_id`
- Metadata resolution order:
  1. on-disk cache (`meta_dir/<id>.json`)
  2. bulk fetch for any missing batch
  3. single `fetch_sound()` fallback
- Skip download when local audio file already exists
- Throttle manifest rewrites to every N items

### `gui/backend/main.py`
- Pass `require_confirmed=True` and `fetch_analysis=False` when starting jobs
- `/api/data/download-progress` should report counts over the **confirmed** subset (total_requested = confirmed count, not all freesound rows)
- Optionally add counts: `skipped_rejected`, `skipped_cached_metadata`, `skipped_existing_audio`

### `gui/frontend/src/components/DatasetDownloadProgress.tsx`
- Add small info row: "Subset: confirmed-only" and show any new counters the backend exposes
- No behavioural change required beyond that

---

## 5. Configuration Additions

In pipeline config (`data_pipeline/config.yaml` under `freesound`):

- `requests_per_minute`: int
- `requests_per_day`: int
- `bulk_metadata_batch_size`: int (default 25)
- `manifest_write_every`: int (default 25)
- `api_usage_path`: default `data/freesound_api_usage.json`

All optional; sensible defaults coded in if missing.

---

## 6. Testing Plan

### Unit / local
- Small smoke test with `--limit 5 --skip-audio` on a known confirmed subset
- Verify:
  - zero metadata calls if all 5 are already cached
  - bulk endpoint is used when cache is cold
  - existing audio files are skipped
  - rejected rows are never touched

### Observability
- Log one summary at end:
  - `cached_metadata_hits`
  - `bulk_metadata_calls`
  - `single_metadata_fallback_calls`
  - `downloads_executed`
  - `downloads_skipped_existing`
  - `api_requests_used_today`

### Rollback
- All new flags default to current-ish behaviour where possible except:
  - `require_confirmed=True` (intentional behavior change, can be flipped off via CLI)

---

## 7. Expected Impact

| Change | Expected call reduction |
|---|---|
| Confirmed-only filter | Eliminates all calls for rejected rows |
| Metadata cache reuse | Up to ~50% on re-runs |
| Opt-in analysis | ~50% baseline reduction in `/analysis/` calls |
| Bulk metadata endpoint | ~10–25× fewer metadata calls for cold-cache batches |
| Skip existing audio | Cuts repeated downloads on resume |
| Adaptive throttle | Avoids 429s → fewer retries |

Net effect on a full flagged subset: roughly an order of magnitude fewer API calls in the common case.

---

## 8. Open Questions For User

- **Confirmed-only default**: OK to make `require_confirmed=True` the default and require an explicit flag to download rejected rows? (recommended: yes)
- **Daily budget**: what daily API ceiling should we target in the token bucket? (Freesound's limits for your account)
- **Analysis data retention**: do you want `fetch_analysis=False` by default for the downloader, given prefilter already captured analysis? (recommended: yes)
- **Bulk batch size**: 25 is a safe default; any reason to go higher/lower?

---

## 9. Implementation Order (once approved)

1. Add confirmed-only filter + dedupe + skip-existing (lowest risk, biggest relative cleanup)
2. Add cache-first metadata resolution + opt-in analysis
3. Add bulk metadata endpoint + client method
4. Add token-bucket throttle + usage persistence
5. Update GUI backend/frontend to reflect confirmed-only counts
6. Manifest write batching
7. Logging + end-of-run summary counters
