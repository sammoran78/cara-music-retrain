# CARA Attribution Proof-of-Concept: Implementation Plan
## Stable Audio Open Small Retraining with Constrained Attribution

**Author:** Sam Moran, Macquarie University  
**Version:** 2.0 — 9 April 2026  
**Purpose:** This document is a complete implementation specification. A programming agent should be able to work through it sequentially to build the entire system.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1: Data Acquisition Pipeline](#4-phase-1-data-acquisition-pipeline)
5. [Phase 2: CARA Registry and Codebook](#5-phase-2-cara-registry-and-codebook)
6. [Phase 3: Genre Mapping and Pool Assignment](#6-phase-3-genre-mapping-and-pool-assignment)
7. [Phase 4: Soft Target Construction and Sidecar Generation](#7-phase-4-soft-target-construction-and-sidecar-generation)
8. [Phase 5: Content Identity and Deduplication](#8-phase-5-content-identity-and-deduplication)
9. [Phase 6: Model Fine-Tuning (Single Frozen Checkpoint)](#9-phase-6-model-fine-tuning)
10. [Phase 7: Pre-Encoding and FAISS Index](#10-phase-7-pre-encoding-and-faiss-index)
11. [Phase 8: Attribution Head Training](#11-phase-8-attribution-head-training)
12. [Phase 9: Constrained Decoding and Output Assembly](#12-phase-9-constrained-decoding)
13. [Phase 10: Four-State Validation and Repair](#13-phase-10-validation-and-repair)
14. [Phase 11: Baselines and Evaluation](#14-phase-11-baselines-and-evaluation)
15. [Phase 12: GUI / Web Interface](#15-phase-12-gui)
16. [File Structure](#16-file-structure)
17. [Technical Risks and Mitigations](#17-risks)
18. [Appendix A: CARA Codeword Specification](#appendix-a)
19. [Appendix B: Four-State Attribution Hierarchy](#appendix-b)
20. [Appendix C: Thesis Relationship](#appendix-c)

---

## 1. Project Summary

### Research Question

Can pool-level creative attribution (via structured CARA codewords) survive the generative AI training-inference loop — i.e., can a model trained on CARA-tagged audio produce meaningful, verifiable attribution codewords at generation time?

### What This System Does

1. Re-downloads the ~486k source audio files used to train Stable Audio Open Small
2. Enriches each file with a CARA codeword representing its licensed pool membership
3. Fine-tunes the Stable Audio Open Small DiT with CARA codewords in the conditioning
4. Freezes that checkpoint permanently
5. Trains a separate attribution head on the frozen model's representations
6. At inference, generates audio AND a structured CARA attribution string
7. Validates, repairs, or degrades the attribution through a four-state hierarchy
8. Compares the learned attribution head against non-learned baselines
9. Reports accuracy, calibration, error rates, and repair statistics

### Key Design Principles

- **One model, one checkpoint.** All evaluations and baselines operate on the same frozen model. No second fine-tuning run. Apples-to-apples comparison.
- **No file duplication.** Each unique audio file appears exactly once in the training data regardless of how many pools it belongs to. Multi-pool membership is handled through soft attribution targets.
- **No free-form text in attribution.** The attribution channel is a constrained formal language of registered codewords, fixed separators, and integer probability bins. The model cannot hallucinate pool names.
- **Attribution failure ≠ attribution absence.** A four-state hierarchy (exact → repaired → degraded → exception) ensures that every output carries the best available attribution, never a silent null.
- **Content identity is separate from pool membership.** A perceptual fingerprint identifies the audio; pool assignments are registry relationships, not file properties.

### Model Being Used

- **Model:** Stable Audio Open Small (`stabilityai/stable-audio-open-small`)
- **Architecture:** Latent diffusion with DiT (transformer diffusion), T5-base text conditioning, VAE autoencoder
- **Output:** Up to 11 seconds stereo audio at 44.1kHz
- **Training toolkit:** `stable-audio-tools` (https://github.com/Stability-AI/stable-audio-tools)
- **Key detail:** The DiT was trained solely on the Freesound subset (~472k files). The autoencoder was trained on both Freesound and FMA. For this experiment we fine-tune only the DiT.

### Source Data

| Source | Total Files | Post-Filtering | License Mix | Metadata |
|---|---|---|---|---|
| Freesound | 472,618 | 266,324 CC0 + 194,840 CC-BY + 11,454 CC Sampling+ | Tags, description, user, duration, Essentia content-analysis descriptors (via API) |
| Free Music Archive | 13,874 | 8,967 CC-BY + 4,907 CC0 | Genre hierarchy (161 genres), artist, album, title (via tracks.csv / genres.csv) |

---

## 2. Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA PIPELINE                                                       │
│                                                                     │
│  Source Audio ──→ Perceptual Fingerprint ──→ Deduplication          │
│       │                                          │                  │
│       ▼                                          ▼                  │
│  Metadata Harvest ──→ Genre Mapping ──→ Pool Assignment             │
│       │                                      │                      │
│       ▼                                      ▼                      │
│  CARA Registry (codewords, hierarchy, checksums)                    │
│       │                                                             │
│       ▼                                                             │
│  Soft Target Construction (multi-pool probability targets)          │
│       │                                                             │
│       ▼                                                             │
│  Sidecar JSON per file + Master Registry CSV                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SEQUENTIAL TRAINING (all from one checkpoint)                       │
│                                                                     │
│  Step 1: Fine-tune DiT (diffusion loss only, CARA in prompt)       │
│       │                                                             │
│       ▼                                                             │
│  Step 2: FREEZE DiT checkpoint ──→ dit_finetuned_v1.safetensors    │
│       │                                                             │
│       ▼                                                             │
│  Step 3: Pre-encode full training set through frozen autoencoder   │
│          + extract DiT hidden states ──→ FAISS index               │
│       │                                                             │
│       ▼                                                             │
│  Step 4: Train attribution head on frozen DiT representations      │
│          (only head parameters update; DiT and AE frozen)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE                                                           │
│                                                                     │
│  Text Prompt ──→ T5 encoder ──→ DiT (frozen) ──→ AE decoder ──→ Audio
│                                     │                               │
│                                     └──→ Attribution Head           │
│                                              │                      │
│                                              ▼                      │
│                                     Constrained Decoding            │
│                                              │                      │
│                                              ▼                      │
│                                     Four-State Validation           │
│                                              │                      │
│                                              ▼                      │
│                            ATTR|CW1@PP|CW2@PP|CW3@PP|END           │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EVALUATION (all from same generation runs)                          │
│                                                                     │
│  Same 1000 generated files evaluated by ALL methods:               │
│    1. Attribution Head (learned)                                    │
│    2. Latent-space Nearest Neighbours (FAISS, non-learned)         │
│    3. Prompt Keyword Matching (text-only baseline)                  │
│    4. Prior Distribution (dataset statistics)                       │
│    5. Random Attribution (floor)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Ordering Matters

The sequential design ensures:
- Audio fidelity is validated before attribution work begins
- All baselines and the learned head operate on the exact same latent space
- The attribution head cannot degrade audio quality (DiT is frozen)
- Results distinguish between "the head learned something useful" and "attribution is trivially recoverable from audio features"

---

## 3. Environment Setup

### Prerequisites

```bash
# Python 3.10+
# CUDA-compatible GPU (minimum 24GB VRAM for training; 12GB for inference)
# ~2TB disk space for full dataset + latents + index

# Core dependencies
pip install stable-audio-tools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops
pip install wandb

# Data pipeline
pip install freesound-python   # Freesound API client
pip install pandas numpy scipy
pip install mutagen             # Audio metadata read/write
pip install acoustid chromaprint # Perceptual fingerprinting (requires fpcalc binary)
pip install crcmod              # CRC-8 checksums
pip install librosa             # Audio feature extraction

# Evaluation and indexing
pip install faiss-gpu           # or faiss-cpu
pip install scikit-learn
pip install matplotlib seaborn

# GUI
pip install fastapi uvicorn
pip install aiofiles python-multipart
# Frontend: React (Node.js 18+)
```

### API Keys Required

| Service | Purpose | How to Obtain |
|---|---|---|
| Freesound API (OAuth2) | Download audio + metadata | https://freesound.org/apiv2/apply |
| Weights & Biases | Training logging | https://wandb.ai (free academic tier) |
| HuggingFace | Model download | https://huggingface.co/settings/tokens |

### Hardware Estimates

| Task | Minimum GPU | Recommended | Time Estimate |
|---|---|---|---|
| Fine-tune DiT | 1× RTX 4090 (24GB) | 1× A6000 (48GB) | 1-2 weeks |
| Pre-encode training set | 1× RTX 3080 (12GB) | Same | 1-2 days |
| Train attribution head | 1× RTX 3080 (12GB) | Same | 2-3 days |
| Inference / evaluation | 1× RTX 3080 (12GB) | Same | Hours |

---

## 4. Phase 1: Data Acquisition Pipeline

### 4.1 Fetch Attribution List

Stability AI publishes the list of file IDs used to train Stable Audio Open Small on their attribution page.

**Script: `data_pipeline/01_fetch_attribution_list.py`**

```
Input:  Stability AI attribution page URL
Output: attribution_list.json — list of Freesound IDs and FMA track IDs
```

Tasks:
1. Fetch https://info.stability.ai/attributions
2. Parse to extract Freesound sound IDs and FMA track IDs
3. Save as structured JSON: `{"freesound_ids": [...], "fma_ids": [...]}`
4. Report total counts and compare against expected (472,618 Freesound + 13,874 FMA)

### 4.2 Freesound Downloader

**Script: `data_pipeline/02_freesound_downloader.py`**

```
Input:  attribution_list.json, Freesound API credentials
Output: audio files in data/freesound/, metadata JSONs in data/freesound_meta/
```

Tasks:
1. Authenticate via OAuth2
2. For each Freesound ID in the attribution list:
   a. Fetch metadata via `/apiv2/sounds/{id}/` — extract: tags, description, username, license, duration, name
   b. Fetch Essentia analysis via `/apiv2/sounds/{id}/analysis/` — extract: moods, genre (inferred), timbre, bpm, key, acoustic/electronic, voice/instrumental
   c. Download audio file via `/apiv2/sounds/{id}/download/` (OAuth2 required) OR download preview (MP3, no OAuth2 required — faster, lower quality)
   d. Save audio to `data/freesound/{id}.{ext}`
   e. Save metadata to `data/freesound_meta/{id}.json`
3. Handle rate limits: implement exponential backoff, respect `X-RateLimit-*` headers
4. Handle unavailable files: log to `data/unavailable_freesound.csv` with reason
5. Support resumption: track progress in `data/download_progress.json`
6. Report: total downloaded, total unavailable, total metadata-only

**Configuration (`config.yaml`):**
```yaml
freesound:
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"
  download_quality: "original"  # or "preview_hq" for faster PoC
  max_concurrent_downloads: 4
  rate_limit_delay_seconds: 0.5
  output_dir: "data/freesound"
  meta_dir: "data/freesound_meta"
```

**Important note on download time:** At ~472k files with rate limits, original-quality downloads could take 2-4 weeks. For initial development and testing, use a subset (1,000-10,000 files). For the PoC, preview quality (MP3, ~128kbps) may be acceptable and downloads without OAuth2 — discuss this tradeoff with supervisors.

### 4.3 FMA Downloader

**Script: `data_pipeline/03_fma_downloader.py`**

```
Input:  attribution_list.json (FMA IDs)
Output: audio files in data/fma/, metadata CSVs in data/fma_meta/
```

Tasks:
1. Download FMA metadata: `curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip`
2. Extract `tracks.csv`, `genres.csv`, `features.csv`
3. Filter `tracks.csv` to only the IDs in the attribution list
4. Download audio for matching tracks from `fma_large.zip` or `fma_full.zip`
   - Alternative: download from the GitHub mirror or Kaggle mirror
5. Save audio to `data/fma/{track_id}.mp3`
6. Save filtered metadata to `data/fma_meta/tracks_filtered.csv` and `data/fma_meta/genres.csv`
7. Report: total downloaded, total unavailable

**FMA metadata structure (tracks.csv):**
- Columns include: track_id, title, artist_name, album_title, genre_top (Tier 1 genre), genres (list of genre IDs), tags, license
- Genre IDs resolve via `genres.csv` which has: genre_id, title, parent (for hierarchy)

### 4.4 Metadata Enricher

**Script: `data_pipeline/04_metadata_enricher.py`**

```
Input:  data/freesound_meta/*.json, data/fma_meta/tracks_filtered.csv
Output: data/enriched_metadata.csv — unified metadata for all files
```

Tasks:
1. For each Freesound file: extract tags, description, Essentia descriptors, license, duration, BPM, key
2. For each FMA file: extract genres (from genre hierarchy), artist, album, license, duration
3. Merge into unified CSV with columns:
   ```
   source, source_id, filename, tags, description, license, duration_s, bpm, key,
   mood_tags, genre_inferred, acoustic_electronic, voice_instrumental,
   essentia_available, fma_genre_top, fma_genre_ids
   ```
4. Handle missing data: mark empty fields, don't invent values
5. Report: coverage statistics for each metadata field

---

## 5. Phase 2: CARA Registry and Codebook

### 5.1 Codeword Format Specification

```
[MODALITY]-[POOL_PAYLOAD]-[VERSION]-[CHECKSUM]

Example: M-K4T9X2-A1-E3
```

| Field | Length | Description | Values |
|---|---|---|---|
| MODALITY | 1 char | Content type | `M` = music (this PoC is music-only) |
| POOL_PAYLOAD | 6 chars | Opaque alphanumeric pool ID | Randomly generated, Hamming distance ≥ 3 between all valid payloads |
| VERSION | 2 chars | Registry/codebook version | `A1` for this PoC |
| CHECKSUM | 2 chars | CRC-8 over prefix `[MOD]-[PAYLOAD]-[VER]` | Hex-encoded |

The payload is deliberately opaque. `K4T9X2` conveys nothing without a registry lookup. This prevents informal interpretation and forces all consumers through the registry.

### 5.2 Inference Output Format

```
ATTR|CW1@PP|CW2@PP|CW3@PP|END
```

| Token | Meaning | Constraint |
|---|---|---|
| `ATTR` | Start delimiter | Fixed literal |
| `CW` | Registered codeword | Must exist in registry for declared modality |
| `@` | Codeword-probability separator | Fixed literal |
| `PP` | Probability bin | Two-digit integer 00-99 |
| `\|` | Slot separator | Fixed literal |
| `END` | End delimiter | Fixed literal |
| | | PP values must sum to 100 |

### 5.3 Codebook Generator

**Script: `registry/generate_codebook.py`**

```
Input:  Number of pools needed, modality, version string
Output: registry/codewords.csv, registry/pools.json
```

Implementation:

```python
import secrets
import crcmod
import csv
import json

crc8_func = crcmod.predefined.mkCrcFun('crc-8')

CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
PAYLOAD_LENGTH = 6
MIN_HAMMING_DISTANCE = 3

def hamming_distance(s1, s2):
    """Compute Hamming distance between two strings of equal length."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def generate_distant_payloads(count, min_hamming=MIN_HAMMING_DISTANCE):
    """
    Generate `count` random 6-char alphanumeric payloads such that
    every pair has Hamming distance >= min_hamming.

    With ~40 pools and 36^6 = 2.1 billion possible payloads, this
    is trivially achievable. Hamming distance >= 3 guarantees
    deterministic single-error correction.
    """
    payloads = []
    attempts = 0
    max_attempts = count * 1000

    while len(payloads) < count and attempts < max_attempts:
        candidate = ''.join(secrets.choice(CHARSET) for _ in range(PAYLOAD_LENGTH))
        if all(hamming_distance(candidate, existing) >= min_hamming
               for existing in payloads):
            payloads.append(candidate)
        attempts += 1

    if len(payloads) < count:
        raise RuntimeError(f"Could not generate {count} payloads with Hamming >= {min_hamming}")

    return payloads

def compute_checksum(modality, payload, version):
    """CRC-8 over the token prefix, rendered as 2-char uppercase hex."""
    prefix = f"{modality}-{payload}-{version}"
    crc = crc8_func(prefix.encode('ascii'))
    return f"{crc:02X}"

def build_codeword(modality, payload, version="A1"):
    """Build a complete CARA codeword with checksum."""
    checksum = compute_checksum(modality, payload, version)
    return f"{modality}-{payload}-{version}-{checksum}"

def generate_codebook(pool_definitions, modality="M", version="A1"):
    """
    Generate a full codebook from pool definitions.

    pool_definitions: list of dicts with at least 'pool_name', 'genre_tier1', 'genre_tier2'
    Returns: list of dicts with codeword fields added
    """
    count = len(pool_definitions)
    payloads = generate_distant_payloads(count)

    codebook = []
    for pool_def, payload in zip(pool_definitions, payloads):
        codeword = build_codeword(modality, payload, version)
        entry = {
            **pool_def,
            "modality": modality,
            "payload": payload,
            "version": version,
            "checksum": compute_checksum(modality, payload, version),
            "codeword": codeword,
        }
        codebook.append(entry)

    return codebook
```

### 5.4 Hierarchy Definition

**Script: `registry/build_hierarchy.py`**

```
Input:  Pool definitions with genre_tier1 groupings
Output: registry/hierarchy.json
```

The hierarchy has three levels:
1. **Root:** `M-ROOT-MUS-L0-XX` — all licensed music pools
2. **Family:** `M-FAM-{FAMILY}-L1-XX` — one per Tier 1 genre grouping (e.g., Jazz family, Electronic family)
3. **Pool:** `M-{PAYLOAD}-A1-XX` — the individual pools

Family codewords use the same format and checksum logic as pool codewords. They are registered in the same codebook but flagged as `level: "family"`.

```json
{
  "M-ROOT-MUS-L0-A1": {
    "level": "root",
    "name": "All licensed music pools",
    "children": ["M-FAM-JAZZ-L1-C7", "M-FAM-ELEC-L1-D9", "..."]
  },
  "M-FAM-JAZZ-L1-C7": {
    "level": "family",
    "name": "Jazz family",
    "parent": "M-ROOT-MUS-L0-A1",
    "children": ["M-K4T9X2-A1-E3", "M-R3P7N1-A1-F2", "..."]
  },
  "M-K4T9X2-A1-E3": {
    "level": "pool",
    "name": "Post-bop / acoustic jazz",
    "parent": "M-FAM-JAZZ-L1-C7",
    "children": []
  }
}
```

### 5.5 Checksum Validation Module

**Script: `registry/validate.py`**

```python
class CARACodebook:
    def __init__(self, codebook_path, hierarchy_path):
        """Load codebook and hierarchy from JSON files."""
        self.codewords = {}       # payload -> full entry
        self.hierarchy = {}       # codeword -> hierarchy entry
        self.idx_to_codeword = {} # integer index -> codeword string
        self.codeword_to_idx = {} # codeword string -> integer index
        # Load from files...

    def is_registered(self, codeword: str) -> bool:
        """Check if codeword exists in registry."""
        payload = self._extract_payload(codeword)
        return payload in self.codewords

    def checksum_valid(self, codeword: str) -> bool:
        """Verify CRC-8 checksum."""
        parts = codeword.split('-')
        if len(parts) != 4:
            return False
        modality, payload, version, claimed_checksum = parts
        expected = compute_checksum(modality, payload, version)
        return claimed_checksum == expected

    def nearest_codeword(self, malformed: str) -> tuple:
        """Find nearest valid codeword by Hamming distance on payload."""
        mal_payload = self._extract_payload(malformed)
        best_dist = float('inf')
        best_cw = None
        for payload, entry in self.codewords.items():
            dist = hamming_distance(mal_payload, payload)
            if dist < best_dist:
                best_dist = dist
                best_cw = entry['codeword']
        return best_cw, best_dist

    def all_within_distance(self, malformed: str, max_dist: int) -> list:
        """Find all valid codewords within given Hamming distance."""
        mal_payload = self._extract_payload(malformed)
        results = []
        for payload, entry in self.codewords.items():
            if hamming_distance(mal_payload, payload) <= max_dist:
                results.append(entry['codeword'])
        return results

    def get_parent_codeword(self, codeword: str) -> str:
        """Get parent/family codeword from hierarchy. Returns None if not found."""
        if codeword in self.hierarchy:
            parent_cw = self.hierarchy[codeword].get('parent')
            return parent_cw
        # Try to infer from payload proximity
        nearest, dist = self.nearest_codeword(codeword)
        if nearest and nearest in self.hierarchy:
            return self.hierarchy[nearest].get('parent')
        return None

    def get_root_codeword(self, modality: str) -> str:
        """Get root codeword for a modality."""
        for cw, entry in self.hierarchy.items():
            if entry.get('level') == 'root' and cw.startswith(modality):
                return cw
        return None

    def recompute_checksum(self, codeword: str) -> str:
        """Recompute and replace checksum on a codeword."""
        parts = codeword.split('-')
        modality, payload, version = parts[0], parts[1], parts[2]
        new_checksum = compute_checksum(modality, payload, version)
        return f"{modality}-{payload}-{version}-{new_checksum}"
```

---

## 6. Phase 3: Genre Mapping and Pool Assignment

### 6.1 Unified Genre Taxonomy

Freesound uses free-form tags; FMA has a formal 161-genre hierarchy. These must be unified into a single two-tier taxonomy that maps to CARA pools.

**Script: `data_pipeline/05_genre_mapper.py`**

```
Input:  data/enriched_metadata.csv, data/fma_meta/genres.csv
Output: data/genre_mapped.csv — adds genre_tier1, genre_tier2, primary_pool columns
```

**Approach:**

1. Define a target taxonomy of ~10-15 Tier 1 genres and ~30-40 Tier 2 genres
2. Build mapping tables:
   - Freesound tags → Tier 2 genre (keyword matching + Essentia descriptors)
   - FMA genre IDs → Tier 2 genre (direct mapping from FMA hierarchy)
3. Each Tier 2 genre = one CARA pool (subject to minimum membership threshold)
4. Pools with < 500 members merge into nearest Tier 2 neighbour

**Suggested Tier 1 genres (adjustable):**

```python
TIER1_GENRES = [
    "Electronic",
    "Acoustic/Folk",
    "Jazz",
    "Classical/Orchestral",
    "Rock/Metal",
    "Hip-Hop/Beats",
    "Ambient/Drone",
    "Percussion/Drums",
    "Sound Effects",
    "Field Recording",
    "Voice/Vocal",
    "World/Traditional",
    "Experimental/Noise",
]
```

**Tier 2 subdivides these** — e.g., Electronic → {Techno, House, Drum & Bass, Synthwave, IDM, ...}

**Freesound tag mapping strategy:**
- Primary: keyword matching against a curated tag→genre lookup table
- Secondary: Essentia descriptors (acoustic/electronic, mood, timbre) to disambiguate
- Fallback: if no genre can be inferred, assign to "Unclassified" pool (still gets a codeword)
- Multiple tags may suggest multiple genres — record all, assign primary pool by strongest match

### 6.2 Pool Assignment

**Script: `data_pipeline/06_pool_assigner.py`**

```
Input:  data/genre_mapped.csv, minimum_pool_size=500
Output: data/pool_assignments.csv — adds primary_pool, all_pools columns
```

Tasks:
1. Count files per Tier 2 genre
2. Merge any Tier 2 genre with < 500 files into its nearest neighbour (by tag co-occurrence)
3. Assign each file a primary pool (strongest genre affiliation)
4. Record all pool memberships (for soft target construction)
5. Report: pool sizes, distribution, merge decisions

---

## 7. Phase 4: Soft Target Construction and Sidecar Generation

### 7.1 Soft Target Construction

Each file has one primary pool, but may have affinity to multiple pools. Soft targets teach the attribution head that pool boundaries are fuzzy.

**Script: `data_pipeline/07_soft_target_builder.py`**

```
Input:  data/pool_assignments.csv, audio feature embeddings
Output: data/soft_targets.csv — per-file top-3 pool targets with probabilities
```

Implementation:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_pool_centroids(file_embeddings, pool_assignments):
    """
    Compute centroid embedding for each pool.
    file_embeddings: dict of source_id -> feature vector (from Essentia or librosa)
    pool_assignments: dict of source_id -> primary_pool_id
    """
    pool_vectors = {}
    for source_id, pool_id in pool_assignments.items():
        if pool_id not in pool_vectors:
            pool_vectors[pool_id] = []
        pool_vectors[pool_id].append(file_embeddings[source_id])

    centroids = {}
    for pool_id, vectors in pool_vectors.items():
        centroids[pool_id] = np.mean(vectors, axis=0)
    return centroids

def compute_soft_targets(file_embedding, pool_centroids, primary_pool_id,
                         num_slots=3, primary_boost=2.0, temperature=0.1):
    """
    Compute soft multi-pool attribution targets for a single file.

    Returns: list of (pool_id, probability_int) tuples, summing to 100
    """
    pool_ids = list(pool_centroids.keys())
    centroid_matrix = np.array([pool_centroids[pid] for pid in pool_ids])

    # Cosine similarity to all pool centroids
    sims = cosine_similarity(file_embedding.reshape(1, -1), centroid_matrix)[0]

    # Boost primary pool to ensure it remains dominant
    primary_idx = pool_ids.index(primary_pool_id)
    sims[primary_idx] += primary_boost

    # Softmax with temperature
    exp_sims = np.exp(sims / temperature)
    probs = exp_sims / exp_sims.sum()

    # Take top-K
    top_k_indices = np.argsort(probs)[-num_slots:][::-1]
    top_probs = probs[top_k_indices]

    # Normalise to sum to 100
    top_probs = np.round(top_probs / top_probs.sum() * 100).astype(int)
    # Fix rounding error
    top_probs[0] += 100 - top_probs.sum()

    return [(pool_ids[idx], int(prob)) for idx, prob in zip(top_k_indices, top_probs)]
```

**Feature embeddings for soft targets:**
- For Freesound files: use Essentia MFCCs + mood + timbre descriptors (available from API)
- For FMA files: use pre-computed features from `features.csv`
- Alternatively: use librosa to extract a standard feature set from all files
- These embeddings are used ONLY for soft target computation, not for model training

### 7.2 Sidecar JSON Generation

**Script: `data_pipeline/08_sidecar_generator.py`**

```
Input:  data/pool_assignments.csv, data/soft_targets.csv, registry/pools.json
Output: One .json sidecar file per audio file, co-located with the audio
```

Each audio file gets a sidecar JSON that `stable-audio-tools` will read during training:

```json
{
  "prompt": "[M-K4T9X2-A1-E3] ambient pad with soft reverb and slow attack, electronic, high quality",
  "cara_codeword": "M-K4T9X2-A1-E3",
  "cara_pool_name": "Post-bop / acoustic jazz",
  "cara_family_codeword": "M-FAM-JAZZ-L1-C7",
  "cara_soft_targets": [
    {"codeword": "M-K4T9X2-A1-E3", "probability": 72},
    {"codeword": "M-Q7L3H8-B1-D5", "probability": 18},
    {"codeword": "M-T8W2J6-A1-B4", "probability": 10}
  ],
  "source": "freesound",
  "source_id": "481723",
  "original_tags": ["ambient", "pad", "synth", "atmospheric"],
  "genre_tier1": "Electronic",
  "genre_tier2": "Ambient",
  "license": "CC0"
}
```

**Key design note:** The CARA codeword is prepended to the prompt in square brackets. This is the Option A conditioning approach — the T5 encoder tokenises it as part of the text. No model architecture changes are needed for this step.

### 7.3 Master Registry CSV

**Script: `data_pipeline/09_build_master_csv.py`**

```
Input:  All pipeline outputs
Output: master_registry.csv
```

One row per file:
```
source,source_id,filename,filepath,content_fingerprint,codeword,pool_name,family_codeword,
family_name,genre_tier1,genre_tier2,soft_target_1_cw,soft_target_1_prob,soft_target_2_cw,
soft_target_2_prob,soft_target_3_cw,soft_target_3_prob,license,duration_s,bpm,key,
original_tags,download_status
```

---

## 8. Phase 5: Content Identity and Deduplication

### 8.1 Why This Matters

In a future system, record labels might license pools by genre, artist, year, or other criteria. A single song could legitimately belong to many pools. The abuse vector is not multi-pool membership itself — it is when a rightsholder registers slightly different "versions" of the same audio to get separate content fingerprints and multiply training-data weight.

**Principle: Training-data weight is determined by content identity, not by pool membership count.** A file in 5 pools appears once in training with a soft target spanning those pools.

### 8.2 Perceptual Fingerprinting

**Script: `data_pipeline/10_fingerprint_dedup.py`**

```
Input:  All downloaded audio files
Output: data/fingerprints.csv, data/duplicates_report.csv
```

Tasks:
1. Compute a perceptual fingerprint for every audio file using Chromaprint (via `acoustid`/`fpcalc`)
2. Store fingerprints in `data/fingerprints.csv`: `source_id, filepath, fingerprint_hash`
3. Compare all fingerprint pairs to find near-duplicates
4. For any cluster of near-duplicates:
   a. Keep the highest-quality version
   b. Merge pool memberships from all versions into the kept file's record
   c. Remove duplicates from the training set
   d. Log all decisions to `data/duplicates_report.csv`
5. Report: total duplicates found, total removed, pool membership merges

```python
import subprocess
import hashlib

def compute_chromaprint(filepath):
    """Compute Chromaprint fingerprint using fpcalc."""
    result = subprocess.run(
        ['fpcalc', '-raw', filepath],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if line.startswith('FINGERPRINT='):
            return line.split('=', 1)[1]
    return None

def fingerprint_similarity(fp1, fp2):
    """
    Compare two Chromaprint fingerprints.
    Returns similarity score 0.0-1.0.
    Threshold of ~0.85 indicates near-duplicate.
    """
    # Chromaprint raw fingerprints are lists of integers
    # Compare using bit-level popcount similarity
    ints1 = [int(x) for x in fp1.split(',')]
    ints2 = [int(x) for x in fp2.split(',')]
    min_len = min(len(ints1), len(ints2))
    if min_len == 0:
        return 0.0
    matching_bits = sum(
        32 - bin(a ^ b).count('1')
        for a, b in zip(ints1[:min_len], ints2[:min_len])
    )
    total_bits = min_len * 32
    return matching_bits / total_bits
```

### 8.3 Anti-Abuse Logging

For the PoC, the deduplication report should track metrics that would matter in a production system:

```
duplicate_cluster_id, kept_file_id, removed_file_ids, similarity_score,
same_uploader, pool_memberships_before, pool_memberships_after, decision_reason
```

This data is useful for the thesis: it demonstrates awareness of the Sybil attack surface and shows how the architecture handles it.

---

## 9. Phase 6: Model Fine-Tuning (Single Frozen Checkpoint)

### 9.1 Overview

This is the ONE model training run. After this step, the DiT checkpoint is frozen and never modified again. All subsequent work (attribution head, baselines, evaluation) operates on this fixed checkpoint.

**Objective:** Fine-tune the DiT with CARA codewords prepended to text prompts. The training loss is the standard diffusion loss only — no attribution loss at this stage.

### 9.2 Setup

**Clone and install stable-audio-tools:**
```bash
git clone https://github.com/Stability-AI/stable-audio-tools.git
cd stable-audio-tools
pip install .
```

**Download pretrained model:**
```python
from stable_audio_tools import get_pretrained_model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
# Save unwrapped checkpoint for fine-tuning
torch.save(model.state_dict(), "checkpoints/pretrained_unwrapped.safetensors")
```

### 9.3 Custom Metadata Module

**File: `model/cara_metadata.py`**

This module is referenced in the dataset config and tells `stable-audio-tools` how to load metadata for each training file.

```python
import json
import os

def get_custom_metadata(info, audio):
    """
    Load CARA metadata from sidecar JSON alongside audio file.

    Called by stable-audio-tools during training data loading.
    The `info` dict contains the file path; `audio` contains the audio tensor.

    Returns a dict whose keys are added to the training metadata.
    The 'prompt' key is required by the diffusion conditioning system.
    """
    audio_path = info["path"]
    json_path = os.path.splitext(audio_path)[0] + ".json"

    if not os.path.exists(json_path):
        # Fallback: use filename as minimal prompt
        return {"prompt": os.path.basename(audio_path)}

    with open(json_path, 'r') as f:
        meta = json.load(f)

    # The prompt already has the CARA codeword prepended (done in sidecar generation)
    return {
        "prompt": meta["prompt"],
        # Store extra fields for later extraction (not used by diffusion loss)
        "cara_codeword": meta.get("cara_codeword", ""),
        "cara_soft_targets": json.dumps(meta.get("cara_soft_targets", [])),
    }
```

### 9.4 Dataset Config

**File: `model/dataset_config.json`**

```json
{
  "dataset_type": "audio_dir",
  "datasets": [
    {
      "id": "freesound_cara",
      "path": "/data/freesound/",
      "custom_metadata_module": "/code/model/cara_metadata.py"
    },
    {
      "id": "fma_cara",
      "path": "/data/fma/",
      "custom_metadata_module": "/code/model/cara_metadata.py"
    }
  ],
  "random_crop": true
}
```

### 9.5 Training Command

```bash
# Login to W&B
wandb login

# Run fine-tuning
python3 train.py \
  --dataset-config model/dataset_config.json \
  --model-config model/model_config.json \
  --pretrained-ckpt-path checkpoints/pretrained_unwrapped.safetensors \
  --name cara_finetune_v1 \
  --save-dir checkpoints/ \
  --num-gpus 1 \
  --batch-size 4 \
  --accum-batches 8
```

**Model config notes:**
- Use the `model_config.json` from the Stable Audio Open Small HuggingFace repo as the base
- The model_config defines the DiT architecture, autoencoder pretransform, T5 conditioning, and training hyperparameters
- Adjust learning rate to 1e-5 to 5e-5 (lower than original training for fine-tuning)
- The autoencoder is loaded as a pretransform and is NOT trained (frozen by default)

### 9.6 Validation and Freezing

After training completes:
1. Unwrap the checkpoint: `python3 unwrap_model.py --ckpt-path checkpoints/cara_finetune_v1/latest.ckpt`
2. Evaluate audio quality against the original pretrained model:
   - Generate test samples from standard prompts
   - Compare FDopenl3 scores (audio fidelity)
   - Compare FAD scores (Fréchet Audio Distance)
   - If quality has degraded unacceptably, adjust training hyperparameters and retrain
3. Once satisfied, designate this as the FROZEN checkpoint:
   ```bash
   cp checkpoints/cara_finetune_v1/unwrapped.safetensors checkpoints/dit_frozen_v1.safetensors
   ```
4. **This checkpoint is never modified again.** All subsequent work reads from it.

---

## 10. Phase 7: Pre-Encoding and FAISS Index

### 10.1 Pre-Encode Training Set

Run every training audio file through the frozen autoencoder encoder to get latent vectors. Also extract DiT hidden states at multiple layer depths.

**Script: `evaluation/preencode_training_set.py`**

```
Input:  All training audio files, dit_frozen_v1.safetensors
Output: data/latents/ directory with .npy files, data/dit_hidden_states/ directory
```

```python
import torch
import numpy as np
from stable_audio_tools import get_pretrained_model

def preencode_dataset(audio_dir, model, model_config, output_dir):
    """
    Encode all training audio through the frozen autoencoder.
    Also run a forward pass through the frozen DiT to extract hidden states.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    for audio_path in iterate_audio_files(audio_dir):
        source_id = extract_source_id(audio_path)

        # Load and preprocess audio
        audio = load_audio(audio_path, model_config["sample_rate"],
                          model_config["sample_size"])
        audio = audio.to(device)

        with torch.no_grad():
            # Encode through autoencoder
            latent = model.pretransform.encode(audio)
            np.save(f"{output_dir}/latents/{source_id}.npy",
                    latent.cpu().numpy())

            # Extract DiT hidden states at multiple depths
            # This requires hooking into the DiT forward pass
            hidden_states = extract_dit_hidden_states(
                model, latent,
                layer_indices=[0, 4, 8, -1]  # early, mid, late, final
            )
            for layer_idx, hs in hidden_states.items():
                np.save(
                    f"{output_dir}/dit_hidden/{source_id}_layer{layer_idx}.npy",
                    hs.cpu().numpy()
                )
```

### 10.2 Build FAISS Index

**Script: `evaluation/build_faiss_index.py`**

```
Input:  data/latents/*.npy, data/pool_assignments.csv
Output: evaluation/faiss_index.bin, evaluation/faiss_metadata.json
```

```python
import faiss
import numpy as np
import json

def build_index(latents_dir, pool_assignments, output_path):
    """
    Build a FAISS index over all training latent vectors.
    Store pool assignments alongside for NN-based attribution.
    """
    vectors = []
    metadata = []

    for source_id, pool_info in pool_assignments.items():
        latent_path = f"{latents_dir}/{source_id}.npy"
        latent = np.load(latent_path).flatten().astype('float32')
        vectors.append(latent)
        metadata.append({
            "source_id": source_id,
            "primary_pool": pool_info["primary_pool"],
            "codeword": pool_info["codeword"],
            "soft_targets": pool_info["soft_targets"],
        })

    vectors = np.array(vectors)
    dim = vectors.shape[1]

    # Normalise for cosine similarity
    faiss.normalize_L2(vectors)

    # Build index
    index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalised = cosine sim
    index.add(vectors)

    # Save
    faiss.write_index(index, f"{output_path}/faiss_index.bin")
    with open(f"{output_path}/faiss_metadata.json", 'w') as f:
        json.dump(metadata, f)

    print(f"Built FAISS index with {len(vectors)} vectors of dim {dim}")
```

---

## 11. Phase 8: Attribution Head Training

### 11.1 Architecture

The attribution head is a small network that reads the frozen DiT's hidden states and predicts CARA attribution in constrained format.

**File: `model/attribution_head.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CARAAttributionHead(nn.Module):
    """
    Predicts structured CARA attribution from frozen DiT hidden states.

    Output: per-slot codeword classification + probability distribution.
    The ATTR|...|END structure is assembled deterministically from these
    predictions — no free-text generation occurs.
    """

    def __init__(self, dit_hidden_dim, num_codewords, num_slots=3):
        super().__init__()
        self.num_slots = num_slots
        self.num_codewords = num_codewords

        # Shared feature extraction from DiT hidden states
        self.feature_net = nn.Sequential(
            nn.Linear(dit_hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Per-slot codeword classifiers
        # Each head outputs logits over all registered codewords
        self.cw_heads = nn.ModuleList([
            nn.Linear(256, num_codewords) for _ in range(num_slots)
        ])

        # Probability distribution head
        # Outputs num_slots logits -> softmax -> scale to 100
        self.prob_head = nn.Linear(256, num_slots)

    def forward(self, dit_hidden_states):
        """
        Args:
            dit_hidden_states: [batch, seq_len, hidden_dim] from frozen DiT

        Returns:
            cw_logits: list of num_slots tensors, each [batch, num_codewords]
            prob_dist: [batch, num_slots] probability distribution (sums to 1.0)
            prob_bins: [batch, num_slots] integer bins (sums to 100)
        """
        # Pool over sequence dimension
        features = dit_hidden_states.mean(dim=1)  # [batch, hidden_dim]
        features = self.feature_net(features)       # [batch, 256]

        # Predict codewords per slot
        cw_logits = [head(features) for head in self.cw_heads]

        # Predict probability distribution
        prob_logits = self.prob_head(features)
        prob_dist = F.softmax(prob_logits, dim=-1)

        # Convert to integer bins summing to 100
        prob_bins = self._to_bins(prob_dist)

        return cw_logits, prob_dist, prob_bins

    def _to_bins(self, prob_dist):
        """Convert probability distribution to integer bins summing to exactly 100."""
        bins = (prob_dist * 100).round().long()
        # Fix rounding errors
        remainder = 100 - bins.sum(dim=-1, keepdim=True)
        bins[:, 0] += remainder.squeeze(-1)
        return bins
```

### 11.2 Training Loop

**Script: `model/train_attribution_head.py`**

```python
def train_attribution_head(
    dit_hidden_states_dir,
    soft_targets_path,
    codebook,
    num_epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    device="cuda"
):
    """
    Train the attribution head on pre-extracted frozen DiT hidden states.

    The DiT is NOT loaded or modified. Only the head's parameters are trained.
    """
    # Load soft targets
    soft_targets = load_soft_targets(soft_targets_path)

    # Create dataset of (hidden_states, target_codeword_indices, target_probs)
    dataset = AttributionDataset(dit_hidden_states_dir, soft_targets, codebook)
    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialise head
    head = CARAAttributionHead(
        dit_hidden_dim=dataset.hidden_dim,
        num_codewords=codebook.num_codewords,
        num_slots=3
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        # Training
        head.train()
        train_losses = []
        for hidden_states, target_cw_indices, target_probs in train_loader:
            hidden_states = hidden_states.to(device)
            target_cw_indices = target_cw_indices.to(device)
            target_probs = target_probs.to(device)

            cw_logits, prob_dist, prob_bins = head(hidden_states)
            loss = cara_loss(cw_logits, prob_dist, target_cw_indices, target_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        head.eval()
        val_metrics = evaluate_head(head, val_loader, codebook, device)

        scheduler.step()

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            **val_metrics
        })

    # Save trained head
    torch.save(head.state_dict(), "checkpoints/attribution_head_v1.pt")
    return head


def cara_loss(cw_logits, prob_dist, target_cw_indices, target_prob_dist, num_slots=3):
    """
    Combined loss for codeword prediction and probability calibration.

    cw_logits: list of num_slots tensors, each [batch, num_codewords]
    prob_dist: [batch, num_slots] predicted probability distribution
    target_cw_indices: [batch, num_slots] ground truth codeword indices
    target_prob_dist: [batch, num_slots] ground truth probability distribution (sums to 1.0)
    """
    # Cross-entropy loss for codeword prediction per slot
    cw_loss = sum(
        F.cross_entropy(cw_logits[s], target_cw_indices[:, s])
        for s in range(num_slots)
    ) / num_slots

    # KL divergence for probability distribution
    # Ensure no zeros for log stability
    pred_log = torch.log(prob_dist + 1e-8)
    target_normed = target_prob_dist / (target_prob_dist.sum(dim=-1, keepdim=True) + 1e-8)
    prob_loss = F.kl_div(pred_log, target_normed, reduction='batchmean')

    return cw_loss + 0.5 * prob_loss  # weight the probability loss lower
```

---

## 12. Phase 9: Constrained Decoding and Output Assembly

### 12.1 Constrained Vocabulary

The attribution head CANNOT generate arbitrary text. Its output is assembled from discrete predictions into the fixed ATTR format.

**File: `model/constrained_decoder.py`**

```python
class ConstrainedCARADecoder:
    """
    Assembles a valid ATTR string from the attribution head's predictions.

    The vocabulary is restricted to:
    - ATTR (start delimiter)
    - END (end delimiter)
    - | (slot separator)
    - @ (codeword-probability separator)
    - Registered codewords only
    - 00-99 (probability bins)

    The model cannot generate anything outside this vocabulary because
    the output is assembled deterministically from classified slots,
    not generated token-by-token.
    """

    def __init__(self, codebook):
        self.codebook = codebook

    def decode(self, cw_logits, prob_bins):
        """
        Assemble ATTR string from head outputs.

        Args:
            cw_logits: list of num_slots tensors, each [batch, num_codewords]
            prob_bins: [batch, num_slots] integer bins summing to 100

        Returns:
            list of ATTR strings, one per batch element
        """
        batch_size = prob_bins.shape[0]
        results = []

        for b in range(batch_size):
            slots = []
            seen_codewords = set()

            for s in range(len(cw_logits)):
                # Get top codeword for this slot (skip duplicates)
                logits = cw_logits[s][b].clone()
                while True:
                    cw_idx = logits.argmax().item()
                    cw = self.codebook.idx_to_codeword[cw_idx]
                    if cw not in seen_codewords:
                        break
                    logits[cw_idx] = float('-inf')  # mask and try next

                seen_codewords.add(cw)
                pp = prob_bins[b, s].item()
                slots.append(f"{cw}@{pp:02d}")

            attr_string = "ATTR|" + "|".join(slots) + "|END"
            results.append(attr_string)

        return results

    def validate_format(self, attr_string):
        """Check structural validity of an assembled ATTR string."""
        if not attr_string.startswith("ATTR|") or not attr_string.endswith("|END"):
            return False, "Missing ATTR/END delimiters"

        body = attr_string[5:-4]  # strip ATTR| and |END
        slots = body.split("|")

        if len(slots) != 3:
            return False, f"Expected 3 slots, got {len(slots)}"

        total_prob = 0
        for slot in slots:
            if "@" not in slot:
                return False, f"Missing @ separator in slot: {slot}"
            cw, pp = slot.rsplit("@", 1)

            if not self.codebook.is_registered(cw):
                return False, f"Unregistered codeword: {cw}"
            if not self.codebook.checksum_valid(cw):
                return False, f"Invalid checksum: {cw}"
            if not pp.isdigit() or len(pp) != 2:
                return False, f"Invalid probability bin: {pp}"

            total_prob += int(pp)

        if total_prob != 100:
            return False, f"Probabilities sum to {total_prob}, expected 100"

        return True, "Valid"
```

---

## 13. Phase 10: Four-State Validation and Repair

### 13.1 Attribution State Hierarchy

Every output must carry one of four attribution states. Attribution NEVER silently disappears.

```
State A: EXACT VALID
  All codewords pass checksum + registry lookup. Probs sum to 100.
  → Record as-is.

State B: REPAIRABLE
  Token is within edit-distance 1 of exactly ONE valid codeword.
  → Correct deterministically. Record correction flag.

State C: DEGRADED BUT MANDATORY
  Exact codeword unrecoverable, but parent/family pool exists in hierarchy.
  → Fallback to family codeword. Record uncertainty flag.

State D: EXCEPTION
  Neither exact nor family attribution survives.
  → Record signed exception: what failed, why, coarsest provenance retained.
  → Model-level and workflow-level provenance always preserved.
```

### 13.2 Validator Implementation

**File: `validation/validator.py`**

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class AttrState(Enum):
    EXACT = "exact_valid"
    REPAIRED = "repaired"
    DEGRADED = "degraded_fallback"
    EXCEPTION = "exception"

@dataclass
class SlotResult:
    original_cw: str
    validated_cw: str
    probability: int
    state: AttrState
    repair_detail: str = ""

@dataclass
class ValidationResult:
    state: AttrState
    original_string: str
    validated_string: str
    slots: List[SlotResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)

class CARAValidator:
    """
    Four-state validation pipeline for CARA attribution strings.

    Principle: attribution failure ≠ attribution absence.
    Every output gets the best available attribution, never a silent null.
    """

    def __init__(self, codebook, max_edit_distance=1):
        self.codebook = codebook
        self.max_edit_distance = max_edit_distance

    def validate(self, attr_string: str) -> ValidationResult:
        """Full validation pipeline."""

        # Step 1: Structural check
        structural_ok, structural_msg = self._check_structure(attr_string)
        if not structural_ok:
            return self._handle_structural_failure(attr_string, structural_msg)

        # Step 2: Parse slots
        slots = self._parse_slots(attr_string)

        # Step 3: Validate each slot
        slot_results = []
        for cw, prob in slots:
            slot_results.append(self._validate_slot(cw, prob))

        # Step 4: Normalise probabilities if needed
        prob_sum = sum(s.probability for s in slot_results)
        if prob_sum != 100:
            slot_results = self._normalise_probabilities(slot_results)

        # Step 5: Determine overall state (worst slot wins)
        overall_state = self._determine_state(slot_results)

        # Step 6: Assemble validated string
        validated_string = self._assemble(slot_results)

        errors = [s.repair_detail for s in slot_results if s.state != AttrState.EXACT]
        repairs = [s.repair_detail for s in slot_results if s.state == AttrState.REPAIRED]

        return ValidationResult(
            state=overall_state,
            original_string=attr_string,
            validated_string=validated_string,
            slots=slot_results,
            errors=errors,
            repairs=repairs,
        )

    def _validate_slot(self, cw: str, prob: int) -> SlotResult:
        """Validate a single codeword through the four-state hierarchy."""

        # State A: Exact valid
        if self.codebook.is_registered(cw) and self.codebook.checksum_valid(cw):
            return SlotResult(cw, cw, prob, AttrState.EXACT)

        # State B attempt: Checksum recompute (payload exists but checksum wrong)
        if self.codebook.is_registered_payload(cw):
            corrected = self.codebook.recompute_checksum(cw)
            return SlotResult(cw, corrected, prob, AttrState.REPAIRED,
                              "checksum_recomputed")

        # State B attempt: Edit distance repair
        nearest, distance = self.codebook.nearest_codeword(cw)
        if distance <= self.max_edit_distance:
            alternatives = self.codebook.all_within_distance(cw, distance)
            if len(alternatives) == 1:
                return SlotResult(cw, nearest, prob, AttrState.REPAIRED,
                                  f"edit_distance_{distance}_unique")

        # State C: Parent pool fallback
        parent_cw = self.codebook.get_parent_codeword(cw)
        if parent_cw:
            return SlotResult(cw, parent_cw, prob, AttrState.DEGRADED,
                              f"parent_pool_fallback:{parent_cw}")

        # State C: Root fallback
        modality = cw[0] if len(cw) > 0 else "M"
        root_cw = self.codebook.get_root_codeword(modality)
        if root_cw:
            return SlotResult(cw, root_cw, prob, AttrState.DEGRADED,
                              f"root_pool_fallback:{root_cw}")

        # State D: Exception — nothing worked
        return SlotResult(cw, "UNRESOLVED", prob, AttrState.EXCEPTION,
                          f"all_validation_failed:original={cw}")

    def _determine_state(self, slot_results: List[SlotResult]) -> AttrState:
        """Overall state is the worst state across all slots."""
        states = [s.state for s in slot_results]
        for worst in [AttrState.EXCEPTION, AttrState.DEGRADED, AttrState.REPAIRED]:
            if worst in states:
                return worst
        return AttrState.EXACT

    def _check_structure(self, attr_string):
        if not attr_string.startswith("ATTR|"):
            return False, "missing_ATTR_prefix"
        if not attr_string.endswith("|END"):
            return False, "missing_END_suffix"
        body = attr_string[5:-4]
        slots = body.split("|")
        if len(slots) != 3:
            return False, f"expected_3_slots_got_{len(slots)}"
        for slot in slots:
            if "@" not in slot:
                return False, f"missing_@_separator:{slot}"
        return True, "ok"

    def _parse_slots(self, attr_string):
        body = attr_string[5:-4]
        slots = []
        for slot in body.split("|"):
            cw, pp = slot.rsplit("@", 1)
            slots.append((cw, int(pp)))
        return slots

    def _normalise_probabilities(self, slot_results):
        total = sum(s.probability for s in slot_results)
        if total == 0:
            for i, s in enumerate(slot_results):
                s.probability = 100 // len(slot_results)
            slot_results[0].probability += 100 - sum(s.probability for s in slot_results)
        else:
            for s in slot_results:
                s.probability = round(s.probability / total * 100)
            remainder = 100 - sum(s.probability for s in slot_results)
            slot_results[0].probability += remainder
        return slot_results

    def _assemble(self, slot_results):
        slots_str = "|".join(f"{s.validated_cw}@{s.probability:02d}" for s in slot_results)
        return f"ATTR|{slots_str}|END"

    def _handle_structural_failure(self, attr_string, reason):
        return ValidationResult(
            state=AttrState.EXCEPTION,
            original_string=attr_string,
            validated_string="ATTR|STRUCTURAL_FAILURE|END",
            slots=[],
            errors=[f"structural_failure:{reason}"],
            repairs=[],
        )
```

### 13.3 Unit Tests for All Four States

**File: `validation/test_validation.py`**

```python
def test_state_a_exact_valid():
    """All codewords valid, checksums pass, probs sum to 100."""
    result = validator.validate("ATTR|M-K4T9X2-A1-E3@45|M-Q7L3H8-B1-D5@30|M-B2N6R4-C1-A2@25|END")
    assert result.state == AttrState.EXACT
    assert result.validated_string == result.original_string

def test_state_b_checksum_repair():
    """Valid payload but wrong checksum → recompute."""
    result = validator.validate("ATTR|M-K4T9X2-A1-E8@45|M-Q7L3H8-B1-D5@30|M-B2N6R4-C1-A2@25|END")
    assert result.state == AttrState.REPAIRED
    assert "checksum_recomputed" in result.repairs[0]

def test_state_b_edit_distance_repair():
    """Payload one char off from a valid codeword → correct."""
    result = validator.validate("ATTR|M-K4T9X7-A1-E3@45|M-Q7L3H8-B1-D5@30|M-B2N6R4-C1-A2@25|END")
    assert result.state == AttrState.REPAIRED
    assert "edit_distance" in result.repairs[0]

def test_state_c_parent_fallback():
    """Unrecoverable codeword but parent pool exists → degrade."""
    result = validator.validate("ATTR|M-ZZZZZZ-A1-00@45|M-Q7L3H8-B1-D5@30|M-B2N6R4-C1-A2@25|END")
    assert result.state == AttrState.DEGRADED
    assert "fallback" in result.errors[0]

def test_state_d_exception():
    """Complete failure → exception record, not null."""
    result = validator.validate("GARBAGE_STRING")
    assert result.state == AttrState.EXCEPTION
    assert result.validated_string is not None  # never null

def test_probability_normalisation():
    """Probs don't sum to 100 → normalise."""
    result = validator.validate("ATTR|M-K4T9X2-A1-E3@40|M-Q7L3H8-B1-D5@30|M-B2N6R4-C1-A2@20|END")
    total = sum(s.probability for s in result.slots)
    assert total == 100
```

---

## 14. Phase 11: Baselines and Evaluation

### 14.1 Evaluation Protocol

Generate a fixed evaluation set of 1,000 audio files from standardised prompts. Each generation uses the SAME frozen model. Then compute all five attribution methods on the SAME generated audio.

**Script: `evaluation/run_evaluation.py`**

```python
def run_full_evaluation(
    model, model_config, head, codebook, faiss_index, faiss_metadata,
    prompts, device="cuda"
):
    """
    Generate evaluation audio and compute all attribution methods.
    All methods operate on the same model, same generations, same latent space.
    """
    results = []

    for prompt_entry in prompts:
        prompt = prompt_entry["prompt"]
        expected_pool = prompt_entry["expected_primary_pool"]

        # Generate audio from frozen model
        audio, dit_hidden_states, latent = generate_with_internals(
            model, model_config, prompt, device
        )

        # Method 1: Attribution Head (learned)
        cw_logits, prob_dist, prob_bins = head(dit_hidden_states)
        attr_string_head = constrained_decoder.decode(cw_logits, prob_bins)[0]
        validation_result = validator.validate(attr_string_head)

        # Method 2: Latent-space Nearest Neighbours (FAISS, non-learned)
        attr_string_nn = nn_attribution(latent, faiss_index, faiss_metadata, codebook)

        # Method 3: Prompt Keyword Matching
        attr_string_kw = keyword_attribution(prompt, keyword_pool_map)

        # Method 4: Prior Distribution (dataset statistics)
        attr_string_prior = prior_attribution(pool_size_distribution)

        # Method 5: Random
        attr_string_random = random_attribution(codebook)

        results.append({
            "prompt": prompt,
            "expected_pool": expected_pool,
            "head_attribution": attr_string_head,
            "head_state": validation_result.state.value,
            "nn_attribution": attr_string_nn,
            "keyword_attribution": attr_string_kw,
            "prior_attribution": attr_string_prior,
            "random_attribution": attr_string_random,
            # ... accuracy metrics computed below
        })

    return compute_all_metrics(results)
```

### 14.2 Baseline Implementations

**Baseline 2: Nearest-Neighbour Attribution**

```python
def nn_attribution(generated_latent, faiss_index, faiss_metadata, codebook, k=20):
    """
    Encode generated audio, find k nearest training examples,
    vote on pool attribution weighted by distance.
    """
    query = generated_latent.flatten().cpu().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query)

    distances, indices = faiss_index.search(query, k)

    # Weighted vote by similarity (distance = cosine similarity since normalised)
    pool_votes = {}
    for dist, idx in zip(distances[0], indices[0]):
        meta = faiss_metadata[idx]
        cw = meta["codeword"]
        pool_votes[cw] = pool_votes.get(cw, 0) + dist

    # Normalise to top-3 summing to 100
    sorted_pools = sorted(pool_votes.items(), key=lambda x: -x[1])[:3]
    total = sum(v for _, v in sorted_pools)
    slots = [(cw, round(v / total * 100)) for cw, v in sorted_pools]
    # Fix rounding
    slots[0] = (slots[0][0], slots[0][1] + (100 - sum(p for _, p in slots)))

    return "ATTR|" + "|".join(f"{cw}@{p:02d}" for cw, p in slots) + "|END"
```

**Baseline 3: Prompt Keyword Matching**

```python
def keyword_attribution(prompt, keyword_pool_map):
    """
    Simple keyword lookup from prompt text to pool distribution.
    keyword_pool_map: dict of keyword -> {pool_codeword: weight}
    Built from training data metadata (which keywords co-occur with which pools).
    """
    pool_scores = {}
    prompt_lower = prompt.lower()

    for keyword, pool_weights in keyword_pool_map.items():
        if keyword in prompt_lower:
            for cw, weight in pool_weights.items():
                pool_scores[cw] = pool_scores.get(cw, 0) + weight

    if not pool_scores:
        return prior_attribution(pool_size_distribution)

    sorted_pools = sorted(pool_scores.items(), key=lambda x: -x[1])[:3]
    total = sum(v for _, v in sorted_pools)
    slots = [(cw, round(v / total * 100)) for cw, v in sorted_pools]
    slots[0] = (slots[0][0], slots[0][1] + (100 - sum(p for _, p in slots)))

    return "ATTR|" + "|".join(f"{cw}@{p:02d}" for cw, p in slots) + "|END"
```

### 14.3 Metrics

**File: `evaluation/metrics.py`**

| Metric | What It Measures | Computation |
|---|---|---|
| Pool accuracy (top-1) | % where highest-confidence pool matches expected | `predicted_slot1_pool == expected_pool` |
| Pool accuracy (top-3) | % where expected pool appears in any slot | `expected_pool in [slot1, slot2, slot3]` |
| Sector accuracy | % correct modality classification | Should be ~100% in this PoC (all music) |
| Probability calibration | Correlation between predicted % and actual training composition | Pearson-r between predicted pool weights and pool sizes |
| Attribution state distribution | % of outputs in each of the four states | Count per state |
| Repair rate | % of outputs requiring repair (State B) | Count State B / total |
| Degradation rate | % falling to parent pool (State C) | Count State C / total |
| Exception rate | % reaching exception (State D) | Count State D / total |
| Repair precision | Of repaired codewords, % matching intended pool | Requires manual or cross-validation check |
| Checksum catch rate | % of corrupted tokens caught by checksum | Synthetic corruption test |
| Head vs NN delta | Accuracy difference between learned head and NN baseline | `acc_head - acc_nn` |
| Head vs Keyword delta | Accuracy difference between head and keyword baseline | `acc_head - acc_keyword` |

**Key interpretation guide for the thesis:**
- If Head > NN > Keyword > Prior > Random: the head learned something genuinely useful about the generative process
- If NN ≈ Head > Keyword: attribution is recoverable from latent space but the head doesn't add much over retrieval
- If Keyword ≈ Head: the attribution signal is mostly in the prompt text, not the audio
- If Head ≈ Prior: the head hasn't learned input-conditional attribution at all
- All outcomes are publishable — the question is which framing the results support

### 14.4 Evaluation Prompt Design

Create prompts that systematically test attribution:

```python
EVAL_PROMPTS = [
    # Pool-specific prompts (should strongly attribute to one pool)
    {"prompt": "smooth jazz saxophone solo, warm tone, relaxed", "expected_primary_pool": "M-K4T9X2-A1-E3"},
    {"prompt": "hard techno kick drum loop, 140 BPM, distorted", "expected_primary_pool": "M-Q7L3H8-B1-D5"},
    {"prompt": "orchestral string section, cinematic, dramatic", "expected_primary_pool": "M-B2N6R4-C1-A2"},

    # Cross-pool prompts (should show multi-pool attribution)
    {"prompt": "jazz-electronic fusion, saxophone over synthesizer pads", "expected_primary_pool": "MULTI"},
    {"prompt": "ambient orchestral drone, cinematic texture", "expected_primary_pool": "MULTI"},

    # Adversarial prompts (vague — tests whether model defaults to prior or infers)
    {"prompt": "high quality audio, clear sound", "expected_primary_pool": "UNKNOWN"},
    {"prompt": "music loop", "expected_primary_pool": "UNKNOWN"},
]
```

### 14.5 Evaluation Output

**File: `evaluation_log.csv`**

```
generation_id,timestamp,prompt,expected_pool,model_version,
head_attr_string,head_state,head_pool1,head_conf1,head_pool2,head_conf2,head_pool3,head_conf3,
nn_attr_string,nn_pool1,nn_conf1,
kw_attr_string,kw_pool1,kw_conf1,
prior_attr_string,random_attr_string,
head_correct_top1,head_correct_top3,nn_correct_top1,nn_correct_top3,
kw_correct_top1,prior_correct_top1,random_correct_top1,
head_errors,head_repairs,repair_success
```

---

## 15. Phase 12: GUI / Web Interface

### 15.1 Architecture

```
Browser (React)                    Backend (FastAPI)              Compute
├─ Dataset Dashboard          ←→   ├─ /api/data/status           ├─ Local GPU
├─ CARA Pool Manager          ←→   ├─ /api/registry/pools        │
├─ Training Control           ←→   ├─ /api/training/start|stop   │
├─ Generation Studio          ←→   ├─ /api/generate              │
├─ Attribution Inspector      ←→   ├─ /api/validate              │
├─ Metrics Dashboard          ←→   ├─ /api/evaluation/metrics    │
└─ Cloud Training (stretch)   ←→   └─ /api/cloud/submit          └─ Cloud GPU (optional)
```

### 15.2 Backend API Endpoints

**File: `gui/backend/main.py`**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CARA Attribution PoC")

# Data pipeline
@app.get("/api/data/status")        # Download progress, file counts, coverage
@app.post("/api/data/download")     # Start/resume download pipeline

# Registry
@app.get("/api/registry/pools")     # List all pools with stats
@app.get("/api/registry/hierarchy") # Pool hierarchy tree
@app.get("/api/registry/codeword/{cw}")  # Lookup a specific codeword

# Training
@app.get("/api/training/status")    # Current training state, loss curves (from W&B)
@app.post("/api/training/start")    # Launch training with config
@app.post("/api/training/stop")     # Stop training

# Generation
@app.post("/api/generate")          # Generate audio + CARA attribution
# Returns: audio file URL, ATTR string, validation result, all baseline results

# Validation
@app.post("/api/validate")          # Validate an arbitrary ATTR string
# Returns: ValidationResult with state, repairs, errors

# Evaluation
@app.get("/api/evaluation/metrics")       # All metrics from latest evaluation run
@app.post("/api/evaluation/run")          # Trigger full evaluation
@app.get("/api/evaluation/comparison")    # Head vs baselines comparison table
```

### 15.3 Key Frontend Screens

1. **Dataset Dashboard:** File counts per source, download progress bars, pool size distribution chart, missing file report
2. **CARA Pool Manager:** Table of all pools with codeword, member count, genre mapping. Hierarchy tree view. Click pool to see member samples.
3. **Training Control:** Start/stop buttons, config editor, embedded W&B charts (loss, learning rate, demo audio). Checkpoint list with audio quality scores.
4. **Generation Studio:** Text prompt input, generate button, audio player for output, CARA attribution string displayed with colour-coded validation state, side-by-side comparison of all 5 attribution methods
5. **Attribution Inspector:** For a selected generation, show pool distribution bar chart, nearest-neighbour list from FAISS with audio previews, repair log if applicable, hierarchy fallback path if degraded
6. **Metrics Dashboard:** Accuracy tables (head vs all baselines), state distribution pie chart, calibration plot, error rate trends, exportable for thesis

---

## 16. File Structure

```
cara-poc/
│
├── config.yaml                          # API keys, paths, global settings
├── master_registry.csv                  # Every file + codeword + metadata
├── evaluation_log.csv                   # Every generation + all method results
├── README.md
│
├── registry/
│   ├── generate_codebook.py             # Generate distant payloads + checksums
│   ├── build_hierarchy.py               # Build parent-child pool tree
│   ├── validate.py                      # CARACodebook class + CRC-8 logic
│   ├── pools.json                       # Pool definitions + metadata
│   ├── hierarchy.json                   # Parent-child fallback tree
│   └── codewords.csv                    # Flat lookup table
│
├── data_pipeline/
│   ├── 01_fetch_attribution_list.py     # Parse Stability attribution page
│   ├── 02_freesound_downloader.py       # Bulk download + metadata from Freesound API
│   ├── 03_fma_downloader.py             # FMA metadata + audio download
│   ├── 04_metadata_enricher.py          # Unify metadata into enriched CSV
│   ├── 05_genre_mapper.py               # Tags/genres → unified taxonomy
│   ├── 06_pool_assigner.py              # Assign files to pools (with merge logic)
│   ├── 07_soft_target_builder.py        # Compute multi-pool soft targets
│   ├── 08_sidecar_generator.py          # Write .json sidecar per audio file
│   ├── 09_build_master_csv.py           # Assemble master registry
│   ├── 10_fingerprint_dedup.py          # Perceptual fingerprinting + dedup
│   └── config.yaml                      # Pipeline-specific config
│
├── model/
│   ├── cara_metadata.py                 # Custom metadata module for stable-audio-tools
│   ├── attribution_head.py              # CARAAttributionHead nn.Module
│   ├── constrained_decoder.py           # Vocabulary masking + ATTR assembly
│   ├── train_attribution_head.py        # Training loop for head (frozen DiT)
│   ├── dataset_config.json              # stable-audio-tools dataset config
│   └── model_config.json                # stable-audio-tools model config
│
├── validation/
│   ├── validator.py                     # Four-state validation pipeline
│   ├── repair.py                        # Edit-distance repair + fallback
│   ├── codebook_distance.py             # Hamming distance utilities
│   └── test_validation.py              # Unit tests for all four states
│
├── evaluation/
│   ├── preencode_training_set.py        # Encode all training data → latents + hidden states
│   ├── build_faiss_index.py             # Build NN index over training latents
│   ├── baselines.py                     # All 5 baseline implementations
│   ├── metrics.py                       # All metric computations
│   ├── prompts.json                     # Standardised evaluation prompts
│   ├── run_evaluation.py                # Full evaluation protocol
│   └── visualise_results.py             # Charts and tables for thesis
│
├── gui/
│   ├── backend/
│   │   ├── main.py                      # FastAPI app
│   │   ├── routes/                      # API route modules
│   │   └── requirements.txt
│   └── frontend/
│       ├── package.json
│       ├── src/
│       │   ├── pages/                   # Dashboard, Generator, Inspector, Metrics
│       │   └── components/              # Reusable UI components
│       └── public/
│
├── checkpoints/
│   ├── pretrained_unwrapped.safetensors # Original Stable Audio Open Small
│   ├── dit_frozen_v1.safetensors        # THE frozen fine-tuned DiT (never modified)
│   └── attribution_head_v1.pt           # Trained attribution head
│
├── data/
│   ├── freesound/                       # Downloaded Freesound audio files
│   ├── freesound_meta/                  # Freesound metadata JSONs
│   ├── fma/                             # Downloaded FMA audio files
│   ├── fma_meta/                        # FMA metadata CSVs
│   ├── latents/                         # Pre-encoded latent vectors (.npy)
│   ├── dit_hidden/                      # Extracted DiT hidden states (.npy)
│   ├── enriched_metadata.csv
│   ├── genre_mapped.csv
│   ├── pool_assignments.csv
│   ├── soft_targets.csv
│   ├── fingerprints.csv
│   └── duplicates_report.csv
│
└── docs/
    ├── CARA_SPEC.md                     # Full CARA codeword specification
    └── EXPERIMENT_LOG.md                # Running notes on decisions and results
```

---

## 17. Technical Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Freesound files no longer available | Reduced training set | Accept partial dataset; report coverage %; use previews as fallback |
| Download takes too long (rate limits) | Delays Phase 1 by weeks | Start download early; use 10k subset for development; previews skip OAuth2 |
| Attribution head doesn't converge | Can't test learned attribution | NN baseline still demonstrates feasibility; try different DiT layer depths |
| Audio quality degrades during fine-tuning | Unusable model | Evaluate FDopenl3/FAD before freezing; reduce learning rate; fewer epochs |
| Genre→pool mapping too noisy | Weak attribution signal | Use Essentia descriptors; reduce to fewer cleaner pools; manual review |
| GPU memory insufficient | Training fails | DeepSpeed; gradient accumulation; reduce batch size; cloud fallback |
| `stable-audio-tools` API changes | Code breaks | Pin version; fork repo at start |
| Soft targets are too uniform | Head can't distinguish pools | Adjust temperature parameter; increase primary_boost; verify embedding quality |
| HREC scope questions | Ethics delay | Confirm CC-licensed data download is within approved protocol (not human subjects) |

---

## Appendix A: CARA Codeword Specification

### Format
```
[MODALITY]-[POOL_PAYLOAD]-[VERSION]-[CHECKSUM]
```

- MODALITY: 1 char (M/V/I/T)
- POOL_PAYLOAD: 6 chars, opaque alphanumeric, Hamming distance ≥ 3 between all valid payloads
- VERSION: 2 chars (registry version)
- CHECKSUM: 2 chars (CRC-8 hex over prefix)

### Inference Output
```
ATTR|CW1@PP|CW2@PP|CW3@PP|END
```
- 3 slots, probabilities sum to 100
- Only registered codewords and integer bins 00-99 permitted
- Structural tokens (ATTR, END, |, @) are deterministic, not generated

### Error Correction
- Sparse codebook: ~40 valid payloads out of 36^6 = 2.1 billion possible
- Hamming distance ≥ 3: guarantees single-error correction
- CRC-8 checksum: detects corruption before registry lookup
- Nearest-valid decoding: if exactly one valid codeword is within edit distance 1, repair deterministically

---

## Appendix B: Four-State Attribution Hierarchy

| State | Condition | Action | Manifest Record |
|---|---|---|---|
| A: Exact Valid | Checksum + registry pass | Record as-is | Exact pool-level attribution |
| B: Repairable | Within edit-distance 1 of unique valid CW | Correct deterministically | Corrected attribution + correction flag |
| C: Degraded | Exact fails but parent pool exists | Fallback to family codeword | Coarse attribution + uncertainty flag |
| D: Exception | All validation fails | Record failure details | Signed exception + coarsest provenance retained |

**Principle:** Attribution failure does not result in attribution absence. At worst, the system degrades to model-level and workflow-level provenance with an explicit record of what failed and why.

---

## Appendix C: Thesis Relationship

This experiment tests a corollary of Compression Bloom Theory: if AI generation creates homogenisation, can structured attribution make the provenance of that homogenisation traceable?

**Results contribute to:**
- **CARA feasibility:** Does pool-level attribution survive the training-inference loop?
- **Licensing model design:** Are pool confidence scores meaningful enough to inform royalty distribution?
- **Gap analysis:** Compare model attribution ("what the model thinks it's drawing from") with practitioner perception (Delphi panel / wide survey)
- **Policy argument:** Concrete evidence for/against AI training data attribution schemes proposed at WIPO, EU AI Act, and Australian Arts Law

**The key thesis claim is NOT:** "This proves exact causal authorship."

**The key thesis claim IS:** "This creates a robust, machine-readable, inference-time attribution record that is harder to hallucinate, easier to validate, and suitable for carriage within an interoperable provenance system."

**C2PA integration** is a post-validation packaging step outside the scope of this PoC. Once the four-state attribution is validated, wrapping it in a signed C2PA manifest with actions, ingredients, AI disclosure, and the namespaced CARA assertion is a transport/integrity problem, not a machine learning problem.

---

## Implementation Sequence Summary

```
Week 1-2:   Environment setup + start Freesound download + FMA download
Week 2-3:   Genre mapping + pool construction + codebook generation
Week 3-4:   Soft targets + sidecar generation + deduplication + master CSV
Week 4-6:   Fine-tune DiT (Option A, CARA in prompt) → FREEZE checkpoint
Week 6-7:   Pre-encode training set + build FAISS index
Week 7-9:   Train attribution head on frozen representations
Week 8-10:  Build constrained decoder + validator + repair pipeline
Week 10-12: Full evaluation protocol (all 5 methods, all metrics)
Week 6-14:  GUI development (parallel track)
Week 12-16: Write-up, visualisation, thesis chapter

Critical path: Freesound download (weeks 1-4) → Fine-tuning (weeks 4-6) → Everything else
Start the download on Day 1.
```
