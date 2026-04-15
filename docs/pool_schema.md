# CARA Pool Definition Schema v2.0

## Overview

CARA pools represent licensed catalogue segments that combine source, license, and genre information. This schema defines how pools are structured to support royalty attribution.

## Pool Naming Convention

```
{SOURCE}-{LICENSE}-{GENRE}
```

Examples:
- `Freesound-CC0-Electronic`
- `Freesound-CC-BY-Ambient`
- `FMA-CC-BY-Jazz`

## Pool Definition Structure

```json
{
  "pool_id": "string",           // Unique identifier
  "pool_name": "string",         // Human-readable name
  "source": "string",            // Data source (Freesound, FMA)
  "license": "string",           // License type (CC0, CC-BY, CC-Sampling+)
  "genre": "string",             // Primary genre classification
  "tier1_genre": "string",       // Coarse genre category
  "tier2_genre": "string",       // Fine genre category
  "description": "string",       // Pool description
  "metadata": {
    "isrc_prefix": "string",     // ISRC prefix if applicable
    "iswc_prefix": "string",     // ISWC prefix if applicable
    "cmo_territory": "string",   // CMO territory code
    "royalty_tier": "string"     // Royalty distribution tier
  },
  "statistics": {
    "file_count": "integer",     // Number of files in pool
    "total_duration": "float",   // Total duration in seconds
    "created_date": "string",    // ISO date
    "updated_date": "string"     // ISO date
  }
}
```

## License Categories

### Supported Licenses
- **CC0**: Public domain dedication
- **CC-BY**: Attribution required
- **CC-BY-SA**: Attribution + ShareAlike
- **CC-BY-NC**: Attribution + NonCommercial
- **CC-Sampling+**: Sampling Plus license

### License Hierarchy
1. **Open**: CC0
2. **Attribution**: CC-BY, CC-BY-SA
3. **Restricted**: CC-BY-NC, CC-Sampling+

## Genre Taxonomy

### Tier 1 Genres (Coarse)
- Electronic
- Acoustic/Folk
- Jazz
- Classical/Orchestral
- Rock/Metal
- Hip-Hop/Beats
- Ambient/Drone
- Percussion/Drums
- Sound Effects
- Field Recording
- Voice/Vocal
- World/Traditional
- Experimental/Noise
- Unclassified

### Tier 2 Genres (Fine)
Tier 2 genres are derived from:
1. Original metadata genre tags
2. FMA genre hierarchy
3. Inferred from audio features

## Pool Assignment Rules

1. **Primary Pool**: Assigned based on strongest signal from:
   - Source + License + Primary Genre
   
2. **Multi-Pool Membership**: Files can belong to multiple pools with weights:
   - Primary pool: 1.0 weight
   - Secondary pools: 0.5 weight
   - Tertiary pools: 0.25 weight

3. **Minimum Pool Size**: 500 files (configurable)
   - Smaller pools merge into nearest neighbor by genre similarity

## Integration Points

### ISRC/ISWC Mapping
- Pool metadata can include prefix patterns for identifier mapping
- Enables downstream royalty routing through existing infrastructure

### CMO Distribution
- Pool definitions include CMO territory codes
- Royalty tier indicates distribution priority

### C2PA Manifest
- Pool ID embedded in C2PA action assertions
- Pool metadata included in provenance claims

## Example Pool Definitions

```json
[
  {
    "pool_id": "FS-CC0-ELEC-001",
    "pool_name": "Freesound-CC0-Electronic",
    "source": "Freesound",
    "license": "CC0",
    "genre": "Electronic",
    "tier1_genre": "Electronic",
    "tier2_genre": "Techno",
    "description": "Public domain electronic music from Freesound",
    "metadata": {
      "cmo_territory": "GLOBAL",
      "royalty_tier": "open"
    }
  },
  {
    "pool_id": "FMA-CCBY-JAZZ-001",
    "pool_name": "FMA-CC-BY-Jazz",
    "source": "FMA",
    "license": "CC-BY",
    "genre": "Jazz",
    "tier1_genre": "Jazz",
    "tier2_genre": "Bebop",
    "description": "Attribution-required jazz from Free Music Archive",
    "metadata": {
      "cmo_territory": "US",
      "royalty_tier": "attribution"
    }
  }
]
```

## Migration from v1

Previous genre-only pools like "Electronic" should be migrated to:
- Determine source from file metadata
- Extract license from existing data
- Create new pool ID: "Freesound-CC-BY-Electronic"

## Validation Rules

1. Pool ID must be unique
2. Source must be in allowed list: ["Freesound", "FMA"]
3. License must be in supported licenses
4. Genre must be in Tier 1 taxonomy
5. Minimum pool size enforced during assignment
