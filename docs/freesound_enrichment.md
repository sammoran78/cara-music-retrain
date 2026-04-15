# Freesound Attribution Enrichment

## Overview

The Freesound enrichment crawler (`data_pipeline/02a_freesound_enrich_attribution.py`) treats the Stability AI attribution CSV as the canonical source of truth for which Freesound files were used in training. It enriches this historical membership list with current metadata from the Freesound API.

## Key Principles

### Historical Attribution vs Current State

The enrichment process maintains a clear distinction between:

1. **Historical Attribution**: What was included in the Stability training dataset (from the attribution CSV)
2. **Current Metadata**: What the Freesound API returns today

These are stored as separate column groups in the output CSV:
- `original_attribution_*` columns preserve the historical state
- `current_*` columns capture the current API response
- `changed_from_original_*` columns track differences

### No Filtering, Only Annotation

Unlike the prefilter script, the enrichment crawler:
- **Never drops rows** based on duration, license, or content type
- **Never uses music detection** as a gate
- **Preserves all attributed sounds** even if they're now deleted or inaccessible
- **Annotates characteristics** (e.g., `duration_ge_30s`, `current_license_family`) for research use

### Robust ID Extraction

Sound IDs are extracted using a conservative approach:
1. Check explicit ID columns (`id`, `sound_id`, `freesound_id`, etc.)
2. Extract from Freesound URLs if present
3. **No arbitrary digit extraction** - this prevents false matches

Rows without extractable IDs go to the errors CSV with the full row preserved.

## Output Files

### Main Enriched CSV
`data/freesound_attribution_enriched.csv`

Contains one row per attributed sound with:
- Provenance columns (attribution source, timestamp, etc.)
- Current API metadata (if available)
- Analysis data (if available)
- Research annotations (license changes, duration flags, etc.)

### Progress Tracking
`data/freesound_attribution_enriched_progress.json`

Enables resumable processing:
```json
{
  "processed_ids": [123, 456, ...],
  "successful_ids": [...],
  "failed_ids": [...],
  "not_found_ids": [...],
  "counters": {
    "ok": 1234,
    "not_found": 56,
    "forbidden": 0,
    "rate_limited": 0,
    "api_error": 2,
    "parse_error": 1
  }
}
```

### Error Log
`data/freesound_attribution_enriched_errors.csv`

Captures rows where no valid Freesound ID could be extracted:
```csv
row_index,error,raw_row_json
42,"No valid Freesound ID found","{...}"
```

### Run Summary
`data/freesound_attribution_enriched_summary.json`

Summary statistics for the enrichment run.

## Usage

### Basic Usage
```bash
python data_pipeline/02a_freesound_enrich_attribution.py
```

### Resume Interrupted Run
The script automatically resumes from where it left off:
```bash
python data_pipeline/02a_freesound_enrich_attribution.py
```

### Process Specific Range
```bash
# Skip first 1000 rows, process next 500
python data_pipeline/02a_freesound_enrich_attribution.py --offset 1000 --limit 500
```

### Refresh Already Processed
```bash
python data_pipeline/02a_freesound_enrich_attribution.py --refresh-existing
```

### Adjust Rate Limiting
```bash
# Slower crawl to avoid rate limits
python data_pipeline/02a_freesound_enrich_attribution.py --sleep-seconds 1.5
```

## Column Reference

### Provenance Columns
- `attribution_row_index`: Row number in source CSV
- `freesound_sound_id`: Extracted sound ID
- `original_attribution_*`: Data from attribution CSV

### Current State Columns
- `api_status`: ok, not_found, forbidden, rate_limited, api_error, parse_error
- `current_*`: Latest metadata from Freesound API
- `analysis_*`: Audio analysis data if available

### Research Annotations
- `current_license_normalized`: Standardized license name
- `current_license_family`: cc0, cc-by, cc-by-nc, sampling+, unknown
- `duration_ge_30s`: Whether duration ≥ 30 seconds
- `changed_from_original_*`: Change detection flags
- `deleted_or_unavailable_since_attribution`: True if sound returns 404

## License Normalization

The script normalizes licenses to canonical forms:
- `cc0`: CC0, Public Domain
- `cc-by`: Creative Commons Attribution
- `cc-by-nc`: Creative Commons Attribution-NonCommercial
- `sampling+`: Sampling Plus license
- `unknown`: Unrecognized license

## Handling API Responses

### Success (200 OK)
- All metadata fields populated
- Analysis data fetched if available
- Research flags computed

### Not Found (404)
- `api_status`: "not_found"
- `deleted_or_unavailable_since_attribution`: "true"
- Row still included in output

### Rate Limited (429)
- Script pauses and saves progress
- Resume with same command
- Consider increasing `--sleep-seconds`

### Other Errors
- Logged with error message
- Row still included with available data
- Check errors CSV for patterns

## Best Practices

1. **First Run**: Start with a small `--limit` to verify setup
2. **Rate Limits**: Use `--sleep-seconds 1.0` or higher for large runs
3. **Monitoring**: Check progress periodically in the JSON file
4. **Validation**: Spot-check enriched rows against Freesound website
5. **Backup**: Keep the original attribution CSV unchanged

## Downstream Usage

The enriched CSV can be filtered for specific research needs:

```python
import pandas as pd

# Load enriched data
df = pd.read_csv("data/freesound_attribution_enriched.csv")

# Filter for currently available, properly licensed files
candidates = df[
    (df["api_status"] == "ok") &
    (df["current_license_family"].isin(["cc0", "cc-by", "sampling+"])) &
    (df["duration_ge_30s"] == "true")
]

# Export filtered subset
candidates.to_csv("data/freesound_training_candidates_current.csv", index=False)
```

## Comparison with Prefilter

| Aspect | Prefilter | Enrichment |
|--------|-----------|------------|
| Purpose | Select subset for download | Enrich full attribution list |
| Input | Any Freesound CSV | Stability attribution CSV |
| Filtering | Yes (license, duration, music) | No (annotation only) |
| Output | Confirmed + Rejected CSVs | Single enriched CSV |
| Membership | Discovers new candidates | Preserves historical membership |

## Future Enhancements

- Add AI training preference extraction when Freesound API exposes it
- Compute additional audio features from analysis data
- Generate statistical reports on license/metadata changes
- Support incremental updates for only changed sounds
