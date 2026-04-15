# Changelog

All notable changes to the CARA Attribution Proof-of-Concept implementation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pool definition schema v2.0 with source-license-genre structure
- CHANGELOG.md for tracking implementation progress
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

### Changed
- Pool definitions now include source (Freesound/FMA) and license (CC0/CC-BY/etc) information
- Pool naming convention updated from `{Genre}` to `{Source}-{License}-{Genre}`
- Updated `data_pipeline/06_pool_assigner.py` to use new pool naming convention

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
