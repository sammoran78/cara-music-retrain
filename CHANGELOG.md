# Changelog

All notable changes to the CARA Attribution Proof-of-Concept implementation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pool definition schema v2.0 with source-license-genre structure
- CHANGELOG.md for tracking implementation progress
- New `Pool Creator` page in the CARA Attribution Console sidebar under `Data`, implemented as a dedicated pre-training CaRA pool-allocation view rather than a tab inside `Dataset`
- New pool-allocation backend service in `data_pipeline/pool_allocator.py` for manifest normalization, duplicate checks, candidate-pool filtering, style scoring, new-pool creation, and persistent assignment binding
- New pool-allocation API surface in `gui/backend/main.py`:
  - `GET /api/data/pool-allocation/summary`
  - `GET /api/data/pool-allocation/run-status`
  - `POST /api/data/pool-allocation/run`
  - `POST /api/data/pool-allocation/stop`
  - `GET /api/data/pool-allocation/pools`
  - `GET /api/data/pool-allocation/assignments`
  - `GET /api/data/pool-allocation/review-queue`
- Dedicated allocator registry namespace under `registry/pool_allocator/` with:
  - `assets.jsonl`
  - `pools.json`
  - `assignments.jsonl`
  - `duplicates.jsonl`
  - `runs.jsonl`
  - `progress.json`
- Dedicated consolidated CaRA training manifest outputs at:
  - `data/cara_pool_manifest.jsonl`
  - `data/cara_pool_manifest.csv`
- Live pool-allocation run progress tracking with persisted processed-count, total-count, percent-complete, current phase, current asset, current pool, and rolling activity log state
- Pool-allocation pause support that stops safely after the current asset and preserves resumable state on disk
- Relaxed metadata allocation mode (`allow_relaxed_metadata`) so Freesound-heavy rows without label/rightsholder data can still be allocated using licence, territory, genre, style, capacity, and artist-cap rules
- Automated tests for pool-allocation behavior in `tests/test_pool_allocator.py`, including duplicate handling, missing-rights review behavior, relaxed-mode allocation, rerun/resume behavior, and API roundtrips
- Pool Allocator v2 in `data_pipeline/pool_allocator_v2.py`, implemented as a separate broad-planning allocator that normalizes the downloaded manifest first, groups assets into broad pool families, then duration-packs each group into registered CaRA source pools
- Dedicated v2 registry namespace under `registry/pool_allocator_v2/` with `assets.jsonl`, `pools.json`, `assignments.jsonl`, `duplicates.jsonl`, `runs.jsonl`, `progress.json`, and `plan.json`
- Dedicated v2 training manifest outputs at `data/cara_pool_manifest_v2.jsonl` and `data/cara_pool_manifest_v2.csv`
- Pool Creator engine selector for switching between the legacy v1 asset-reactive allocator and the new v2 broad planned allocator
- Focused v2 behavior tests in `tests/test_pool_allocator_v2.py` for 4-hour spillover pools, artist-concentrated exceptions, micro-style consolidation, and v2 training manifest output
- New `Pool Viewer` page under the `Data` sidebar section for browsing the v2 CaRA pool manifest as pool folders and allocated source files
- Read-only Pool Viewer API endpoints for listing pool folders, listing assets inside a pool, fetching full asset metadata, and streaming local audio files for playback:
  - `GET /api/data/pool-viewer/pools`
  - `GET /api/data/pool-viewer/pools/{pool_id}/assets`
  - `GET /api/data/pool-viewer/assets/{asset_id}`
  - `GET /api/data/pool-viewer/assets/{asset_id}/audio`
- Literature review insights integrated into implementation plan
- Synthetic ground-truth evaluation dataset generator (`evaluation/synthetic_dataset.py`)
- Control-token confound test suite (`evaluation/control_token_tests.py`)
- Pool schema documentation (`docs/pool_schema.md`)
- Freesound enrichment crawler (`data_pipeline/02a_freesound_enrich_attribution.py`)
- Freesound enrichment documentation (`docs/freesound_enrichment.md`)
- CARA Attribution Probe Suite (`probe/`) for benchmarking base vs fine-tuned models
- Probe tools: prompt builder, hidden state extractor, linear probe, attribution head runner
- Benchmark report generator with control-token confound assessment
- Frontend component for displaying benchmark results (`BenchmarkResults.tsx`)
- Canonical local attribution manifest export at [`data/attribution_manifest.csv`](/Users/sammoran/Documents/GitHub/cara-music-retrain/data/attribution_manifest.csv)
- Shared attribution parsing helpers in `data_pipeline/attribution_utils.py`
- Offline subset labeler in `data_pipeline/03_offline_pool_labeler.py`
- Manifest updater in `data_pipeline/03a_update_manifest.py`
- Experimental pool ontology in `registry/experimental_pool_ontology.json`
- Experimental seed pool definitions in `registry/experimental_pool_definitions.json`
- Full local Freesound attribution ID export in `data/attribution_list.json`
- High-confidence subset export in `data/attribution_seed_labels_high_conf.csv`
- Shared JSONL manifest helpers in `data_pipeline/manifest_utils.py`
- Canonical JSONL manifest at `data/attribution_manifest.jsonl`
- Draft refactor plan for efficient confirmed-subset Freesound downloads in `docs/freesound_download_refactor_plan.md`
- Local music-focused subset selector in `data_pipeline/03b_select_music_subset.py` for building balanced Freesound training candidates from existing CARA labels
- Methods-style subset selection documentation in `docs/music_subset_selection.md` for internal review and peer-review-oriented auditability
- Peer-review methodology document in `docs/subdataset_methodology.md` covering attribution ingestion, manifest control, CARA labeling, subset selection, download reconciliation, replenishment, genre normalization, v2 pool allocation, and audit artifacts
- Shared genre-label normalization helpers in `data_pipeline/genre_normalization.py` plus `data_pipeline/13_normalize_genre_labels.py` for canonicalizing slash-vs-space labels across the active sub-dataset and v2 pool manifest
- GUI expansion plan in `PLAN.md` for turning the CARA Attribution Console into a responsive multi-view app with `Dataset`, `Finetune: Diffusion`, `Finetune: Autoregressive`, `Testing`, and `Benchmarks` views
- Responsive multi-view CARA Attribution Console shell: persistent desktop sidebar that collapses into a hamburger drawer on small screens (`gui/frontend/src/components/Sidebar.tsx`, `gui/frontend/src/nav.ts`)
- New page views under the sidebar: `gui/frontend/src/pages/DatasetPage.tsx` (wraps the existing console without redesign), `FinetuneDiffusionPage.tsx`, `FinetuneAutoregressivePage.tsx`, `TestingPage.tsx`, and `BenchmarksPage.tsx`
- Shared `FinetuneView` component (`gui/frontend/src/pages/FinetuneView.tsx`) providing run configuration, cloud/VM dispatch controls, live loss chart, and streaming log panel for both diffusion and autoregressive fine-tunes
- Shared page-header and placeholder-badge primitives (`gui/frontend/src/pages/PageHeader.tsx`) so non-Dataset views advertise that backend wiring is pending

### Changed
- Pool definitions now include source (Freesound/FMA) and license (CC0/CC-BY/etc) information
- Pool naming convention updated from `{Genre}` to `{Source}-{License}-{Genre}`
- Updated `data_pipeline/06_pool_assigner.py` to use new pool naming convention
- The CARA Attribution Console navigation now includes `Pool Creator` as a first-class page alongside `Dataset`, `Finetune`, `Testing`, and `Benchmarks`
- Pool allocation is no longer only an offline script concept; it is now a UI-triggered backend job with persistent progress, logs, stop/pause control, and resumable re-entry
- The manifest workflow now supports a second, training-oriented CaRA pool manifest that carries forward the original source rows plus allocator-enriched fields such as `cara_source_pool_id`, `cara_source_pool_assignment_status`, `cara_source_pool_reason_codes`, `isrc`, `iswc`, `record_label`, `rights_holder`, `primary_genre`, `secondary_genre`, `style_tags`, and `metadata_style_summary`
- Pool-allocation reruns now reuse finalized assignments (`assigned`, `new_pool_created`, `duplicate_found`) but allow prior `review_required` rows to be reprocessed when relaxed mode or better metadata is available
- Pool-allocation runs now checkpoint incrementally instead of only writing results at the very end, reducing the amount of work lost on interruption
- Pool Allocator v2 uses a 4-hour pool cap (`14,400` seconds), a 24-minute general artist cap (`1,440` seconds), and explicit `artist_concentrated_pool` records when the artist cap must be broken for catalogue-heavy material
- Pool Allocator v2 no longer creates pools from micro-style tag signatures; it maps manifest metadata into broad families such as `Atmosphere/Field`, `Percussion/Beats`, `Acoustic/Jazz/World`, `Tonal/Orchestral`, `Produced/Electronic`, `Experimental/Noise`, `Voice/Vocal`, and `Sound Effects`
- Pool Allocator v2 creates spillover pools when a broad group reaches the 4-hour cap rather than fragmenting every niche style into its own pool
- Pool Allocator v2 exposes the same Pool Creator progress/log UX through `engine=v2` query parameters on the existing pool-allocation endpoints
- Pool Allocator v2 treats exact identifier/fingerprint matches as duplicate records and keeps fuzzy title/artist/duration matches in read-only review
- Pool Viewer uses `data/cara_pool_manifest_v2.jsonl` as the browsable attribution manifest and supports sortable pool/file columns, pool-folder drilldown, file metadata modals, local audio playback, and browser-rendered waveform previews
- `data_pipeline/01_fetch_attribution_list.py` now parses the canonical Stability CSV directly instead of scraping the attribution webpage
- The attribution export now acts as a long-lived manifest with provenance, API-enrichment placeholders, CARA labeling columns, subset flags, and local asset tracking
- Offline CARA labels are now merged back into the manifest so it can serve as the central source of truth for fine-tuning decisions
- Core data pipeline scripts now write row-level updates back into the central JSONL manifest instead of treating intermediate CSVs as the main state store
- The CSV manifest is now a derived export generated from the canonical JSONL manifest
- Freesound downloads now target the `prefilter_status == "confirmed"` subset by default and reuse the manifest as the central source of truth for resumable download state
- `data_pipeline/freesound_api.py` now supports bulk metadata fetches (via the `search/text/` endpoint with `filter=id:(...)` since `GET /sounds/?ids=` is not a valid Freesound endpoint), optional field filtering, and persisted minute/day Freesound API throttling via `data/freesound_api_usage.json`
- `data_pipeline/02_freesound_downloader.py` now reuses cached metadata, skips existing local audio files, batches manifest writes, and reports download efficiency counters
- GUI dataset download progress now reports subset-aware totals plus metadata-cache, bulk-call, fallback-call, skipped-audio, and API-usage counters and can launch downloads from `confirmed_only`, `subset_role`, or `all_freesound` modes
- Music subset selection can now be automated locally from manifest metadata, favoring music-oriented CARA tiers while excluding sound effects and field recordings and balancing the final set across tier, pool, and license
- Reconciliation, subset selection, and pool allocation now share slash-form genre canonicalization so historical labels such as `Percussion Drums`, `Jazz Blues`, and `Voice Vocal` normalize to `Percussion/Drums`, `Jazz/Blues`, and `Voice/Vocal`
- Planned console expansion treats the existing `FREESOUND · MUSIC ATTRIBUTION POOL` page design and functionality as the completed `Dataset` page view

### Data
- Built a full local attribution manifest with `472,618` Freesound rows
- Generated offline CARA labels across the full manifest
- Marked `16,555` rows as high-confidence subset candidates for future fine-tuning experiments
- Wrote a tightened `20,000`-row music-focused Freesound subset back into `data/attribution_manifest.jsonl` with `subset_role = "music_train_candidate"` and `subset_note = "balanced_music_selector_v1"`
- Normalized the current `25,000`-row active sub-dataset to slash-form genre labels; `data/genre_normalization_report.json` records `21,372` active manifest rows with at least one normalized field

### Notes
- Local sibling `stable-audio-tools` checkout detected at `../stable-audio-tools`
- Current local `stable-audio-tools` pin observed at commit `50049e379e2fe8e35bffc99e57f23ade1c3471b7`
- Current pool-allocation execution model:
  - Source rows are still read from `data/attribution_manifest.jsonl`
  - The dedicated future-training manifest is `data/cara_pool_manifest.jsonl`
  - The v2 future-training manifest is `data/cara_pool_manifest_v2.jsonl`
  - Audio location is inferred from manifest state such as `local_audio_path` and `download_status`, which for current Freesound downloads normally points into `data/freesound`
  - The `Pool Creator` progress/log UX is now implemented specifically for long-running allocation passes and is separate from the existing dataset-download progress system
- Current Pool Allocator v2 run summary from the local 25,000-row `music_train_candidate` subset:
  - `25,000` candidate assets processed
  - `24,273` rows written to `data/cara_pool_manifest_v2.jsonl`
  - `98` registered v2 pools created under `registry/pool_allocator_v2/pools.json`
  - `727` fuzzy duplicate/review rows retained outside the training manifest

### Official subset selection process
- **Goal:** Produce a Freesound training subset that is materially more useful for music attribution than a naive random or confidence-only sample, while remaining reproducible from offline manifest fields alone and avoiding further Freesound API usage.
- **Implementation:** The official selector is `data_pipeline/03b_select_music_subset.py`. It reads the canonical manifest at `data/attribution_manifest.jsonl`, writes a ranked CSV at `data/music_subset_candidates.csv`, writes a machine-readable summary at `data/music_subset_selection_summary.json`, and can write the chosen rows back into the manifest via `include_in_subset`, `subset_role`, and `subset_note`.
- **Offline input signals:** Selection is driven only by fields already present in the manifest after offline labeling and prefiltering. The current implementation uses `cara_tier1`, `cara_tier2`, `cara_primary_pool`, `cara_auto_label_bucket`, `cara_auto_label_confidence`, `cara_auto_label_score`, `cara_matched_keywords_json`, `license_normalized`, `prefilter_status`, `api_current_duration_s`, `include_in_subset`, `source`, and `source_id`.
- **Candidate gating:** A row is eligible only if it is from Freesound, has an allowed license (`cc0`, `cc-by`, or `sampling+`), has a non-empty `cara_primary_pool`, meets the minimum CARA label bucket threshold (current default: `medium`), is not explicitly prefilter-rejected, and is not assigned to excluded tiers. The current excluded tiers are `Sound Effects`, `Field Recording`, and `Unclassified`.
- **Music-oriented inclusion policy:** The current included tiers are `Electronic`, `Percussion/Drums`, `Acoustic/Folk`, `Classical/Orchestral`, `Jazz/Blues`, `Ambient/Drone`, `Experimental/Noise`, and `Voice/Vocal`. This keeps musically relevant but potentially noisy material available while allowing the balancing stage to suppress over-representation from less attribution-useful classes.
- **Row scoring procedure:** Each candidate receives a composite score combining CARA label bucket rank, CARA confidence, CARA auto-label score, matched-keyword evidence, and a duration preference that rewards medium/longer clips and slightly penalizes very short clips. The score also adds a bonus for `prefilter_status == "confirmed"`, a strong penalty for `prefilter_status == "rejected"`, a small carry-forward bonus for rows already marked `include_in_subset`, and an explicit tier bias that promotes structured musical material (`Electronic`, `Percussion/Drums`, `Acoustic/Folk`, `Classical/Orchestral`, `Jazz/Blues`) while demoting `Ambient/Drone`, `Experimental/Noise`, and `Voice/Vocal`.
- **Balancing procedure:** Candidates are globally ranked by the composite score and then selected in multiple passes. The first pass fills per-tier targets derived from explicit tier weights, while also enforcing soft diversity caps by pool and by license. A second pass fills remaining tier capacity from deferred rows that were held back only because the initial tier target was reached. A final relaxation pass fills any remaining slots up to the target subset size while still respecting hard tier caps, which prevents the subset from underfilling when a soft diversity constraint would otherwise leave unused capacity.
- **Why the process is suitable for peer review:** The selection policy is deterministic for a fixed manifest, uses auditable manifest columns rather than opaque external judgments, records both CSV and JSON summaries, and writes its decisions back into the manifest in a way that preserves provenance. The methodology is therefore inspectable row-by-row, reproducible on another machine with the same manifest, and tunable without changing the underlying data source.
- **Current official defaults for the tightened music-focused run:** Included-tier weights are `Electronic 0.31`, `Percussion/Drums 0.25`, `Acoustic/Folk 0.18`, `Classical/Orchestral 0.11`, `Jazz/Blues 0.08`, `Ambient/Drone 0.04`, `Experimental/Noise 0.02`, and `Voice/Vocal 0.01`. Hard cap fractions are currently applied to `Ambient/Drone 0.07`, `Experimental/Noise 0.04`, and `Voice/Vocal 0.025`. Pool and license diversity are encouraged with soft caps before the final fill pass.
- **Current official 20k run summary:** The tightened run evaluated `63,122` eligible candidates and selected `20,000` rows. The resulting tier counts were `Percussion/Drums 5,597`, `Electronic 4,309`, `Acoustic/Folk 3,652`, `Jazz/Blues 2,493`, `Classical/Orchestral 2,215`, `Ambient/Drone 859`, `Experimental/Noise 622`, and `Voice/Vocal 253`. The selected license mix was `cc-by 9,886`, `cc0 9,000`, and `sampling+ 1,114`.
- **Manifest write-back policy:** The official write-back command marks the selected rows with `include_in_subset = true`, assigns `subset_role = "music_train_candidate"`, records `subset_note = "balanced_music_selector_v1"`, and clears those exact flags from previously selected rows that are no longer part of the current official subset. This keeps the canonical manifest synchronized with the latest approved selection rather than allowing stale subset annotations to accumulate.
- **Review guidance:** If peer review concludes that the current subset is still too ambient-heavy, noise-heavy, or stylistically imbalanced, the intended adjustment points are the included-tier weights, the tier bias in row scoring, the hard cap fractions, the minimum bucket threshold, and the excluded-tier list. Those knobs can be changed without requiring API re-enrichment or a different manifest schema.

### Planned
- Integrate C2PA manifest generation
- Add payout stability simulations
- Connect synthetic evaluation to actual model inference
- Implement adversarial prompt testing

## [0.2.0] - 2026-04-14

### Added
- Freesound OAuth helper script for authentication
- Freesound prefilter with rate limit handling and progress display
- Auto-refresh for expired Freesound tokens

### Fixed
- Unicode handling in Freesound ID extraction
- Rate limit poisoning in prefilter progress tracking
- 401 authentication errors with automatic token refresh

## [0.1.0] - 2026-04-09

### Added
- Initial repository structure and scaffolding
- CARA registry implementation (codebook generation, validation)
- Data pipeline foundation (attribution fetcher, metadata enricher)
- Genre mapping and pool assignment modules
- Soft target construction for multi-pool membership
- Four-state validation hierarchy (exact → repaired → degraded → exception)
- Model integration scaffolding
- Basic evaluation pipeline structure

### Technical Foundation
- Hamming distance ≥3 between codewords
- CRC-8 checksum validation
- Constrained formal language for attribution
- Integration with stable-audio-tools architecture
