# Execution Plan

This file tracks implementation progress for the CARA attribution proof-of-concept.

## Ground Rules

- `implement-plan.md` is the canonical specification and must not be edited.
- This `PLAN.md` is the working implementation tracker.
- Work should proceed in milestone order unless blocked by external dependencies.

## Current Status

- **Status:** In progress
- **Current milestone:** GUI expansion for dataset, fine-tuning, testing, and benchmarking workflows
- **Canonical spec:** `implement-plan.md`
- **Latest update:** 2026-04-25 - Planned multi-view CARA Attribution Console expansion

## Milestones

### Milestone 1 - Repository scaffolding and CARA registry foundation
- [x] Review canonical implementation plan
- [x] Extract target file structure and phase ordering
- [x] Create initial working repository structure
- [x] Implement `registry/generate_codebook.py`
- [x] Implement `registry/build_hierarchy.py`
- [x] Implement `registry/validate.py`

### Milestone 2 - Early data pipeline foundation
- [x] Implement `data_pipeline/01_fetch_attribution_list.py`
- [x] Implement `data_pipeline/02a_freesound_prefilter.py`
- [x] Implement `data_pipeline/04_metadata_enricher.py`
- [x] Create pipeline configuration templates

### Milestone 3 - Genre mapping and attribution preparation
- [x] Implement genre mapping and pool assignment
- [x] Implement soft target construction
- [x] Implement sidecar generation and master registry assembly

### Milestone 4 - Validation and inference utilities
- [x] Implement constrained decoding utilities
- [x] Implement four-state validation and repair pipeline

### Milestone 5 - Model and evaluation integration
- [x] Add model integration modules
- [x] Add evaluation pipeline and baseline scaffolding
- [x] Prepare GUI/backend scaffolding

### Remaining integration work
- [x] Confirm local sibling `stable-audio-tools` checkout exists at `../stable-audio-tools`
- [x] Record current local `stable-audio-tools` revision: `50049e379e2fe8e35bffc99e57f23ade1c3471b7`
- [ ] Decide whether to vendor, submodule, or otherwise explicitly pin `stable-audio-tools` for reproducible experiments
- [ ] Replace scaffolded/fake model execution paths with real Stable Audio Open Small integration
- [ ] Run the pipeline end-to-end on subset data and fix schema mismatches
- [ ] Add real frontend implementation beyond placeholder scaffold
- [ ] Add stronger automated tests for registry, pipeline, validation, and evaluation

### Literature Review Integration (2026-04-14)
- [x] Update pool definition schema to include source-license-genre structure
- [x] Create pool_schema.md documentation
- [x] Update pool assignment logic to use new naming convention
- [x] Create CHANGELOG.md for tracking progress
- [x] Implement synthetic ground-truth evaluation dataset
- [x] Add control-token confound tests
- [x] Create Freesound enrichment crawler (replaces aggressive prefilter)
- [x] Create test suite for enrichment crawler
- [ ] Integrate C2PA manifest generation
- [ ] Add payout stability simulations

### Current integration notes
- [x] Dataset config now matches `stable-audio-tools` `audio_dir` loader expectations
- [x] Real latent pre-encoding path added via `stable-audio-tools` dataloader + `model.pretransform.encode`
- [x] Real DiT transformer hidden states are now extracted via `return_info=True` on the Stable Audio transformer path during pre-encoding
- [x] Freesound source selection now has a mitigation-aware CSV prefilter stage before download, producing a smaller confirmed source CSV with metadata
- [x] Added Freesound enrichment crawler that treats attribution CSV as canonical and enriches with current metadata

### Subset-First Attribution Workflow (2026-04-18)
- [x] Phase 1 - Replace attribution page scraping with direct canonical CSV parsing and master-table export
- [x] Phase 1 - Export canonical `data/attribution_manifest.jsonl` plus `data/attribution_list.json`
- [x] Phase 2 - Define initial experimental CARA pool ontology for offline Freesound tagging
- [x] Phase 2 - Add machine-readable ontology config at `registry/experimental_pool_ontology.json`
- [x] Phase 3 - Implement ontology-driven offline labeler for attribution rows
- [x] Phase 3 - Export high-confidence subset and seed pool definitions for later sidecar generation
- [x] Phase 3 - Merge offline CARA labels and subset flags back into the manifest
- [x] Phase 3.5 - Add shared manifest JSONL utilities and CSV export bridge
- [x] Phase 3.5 - Update core data pipeline scripts to write their row-level state back into the central JSONL manifest
- [x] Phase 3.6 - Add canonical genre-label normalization for the active sub-dataset and v2 pool manifest
- [ ] Phase 4 - Add targeted API enrichment for uncertainty rows only
- [ ] Phase 5 - Convert high-confidence subset into sidecars and registry assets for fine-tuning
- [ ] Phase 6 - Run subset fine-tuning experiment and persistence evaluation

### CARA Attribution Console Expansion (2026-04-25)
- [x] Move the existing `FREESOUND · MUSIC ATTRIBUTION POOL` experience into the primary `Dataset` page view without redesigning its current layout or functionality
- [x] Replace the single-page console shell with a responsive app layout that uses a persistent sidebar on desktop and collapses into a hamburger navigation menu on smaller screens
- [x] Add top-level navigation views for `Dataset`, `Finetune: Diffusion`, `Finetune: Autoregressive`, `Testing`, and `Benchmarks`
- [x] Treat the existing dataset acquisition, subset selection, download progress, manifest state, tracked paths, Freesound API usage, and live download controls as the completed `Dataset` page view design
- [x] Design the `Finetune: Diffusion` view for configuring diffusion fine-tuning runs, selecting datasets/subsets, choosing run targets, showing cloud/VM execution state, and streaming live/results logs
- [x] Design the `Finetune: Autoregressive` view for configuring autoregressive fine-tuning runs with the same dataset selection, remote execution state, and log/result tracking patterns as diffusion fine-tuning
- [x] Design the `Testing` view for choosing base or fine-tuned model checkpoints, selecting test suites/prompts, running comparison jobs, and reviewing generated outputs plus attribution/probe diagnostics
- [x] Design the `Benchmarks` view for comparing base, diffusion-finetuned, and autoregressive-finetuned model results, including score deltas, attribution persistence, control-token behavior, prompt robustness, and improvement summaries
- [ ] Define shared frontend state/API contracts for long-running cloud jobs so dataset downloads, fine-tuning, testing, and benchmarks can all report status consistently
- [x] Keep implementation incremental: first ship static page views and navigation, then wire each view to real backend job endpoints as the training/testing runners mature

## Blockers / External Dependencies

- `stable-audio-tools` is available locally at `../stable-audio-tools`, but experiment reproducibility still needs an explicit pinning strategy inside this project.
- Freesound API credentials are required for full downloader implementation and live testing.
- Hugging Face and Weights & Biases credentials will be needed for later training phases.

## Notes

- Start with components that are deterministic and locally testable.
- Subset-first development is the default for Phases 1-5.
- Current repository state is a comprehensive scaffold with runnable local modules, not a fully production-hardened training system.
