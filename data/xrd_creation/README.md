# DARA XRD Analysis

Automated XRD phase identification and Rietveld refinement using [DARA](https://github.com/idocx/dara).

## Quick Start

```bash
# From project root:
./run_analysis.sh                      # Incremental (skip existing)
./run_analysis.sh --limit 10           # Test with 10 experiments
./run_analysis.sh --experiment NSC_249 # Single experiment
./run_analysis.sh --all                # Rerun all experiments
./run_analysis.sh --all --limit 20     # Rerun first 20 only
./run_analysis.sh --workers 4          # Parallel processing
./run_analysis.sh --export-only        # Consolidate JSON → Parquet only
```

## Output

**JSON results** (per experiment): `results/<experiment>_result.json` (includes success AND failures)

**Parquet files** (aggregated): `../parquet/`

- `xrd_refinements.parquet` - All experiments (success=True/False, error, error_type, Rwp, num_phases)
- `xrd_phases.parquet` - Individual phases from successful analyses only

**Note:** Failed analyses are included in `xrd_refinements.parquet` with `success=False`. Query failures with: `df[df['success'] == False]`

## Files

| File                | Purpose                              |
| ------------------- | ------------------------------------ |
| `analyze_single.py` | Single experiment analysis (CLI)     |
| `analyze_batch.py`  | Batch processing with Parquet export |
| `xrd_utils.py`      | Shared utilities and primitives      |

## Compute Requirements

- ~45 seconds per experiment (first run for a chemical system downloads CIFs)
- 576 experiments × 45 sec = ~7 hours sequential
- With `--workers 4`: ~1.8 hours
- **Tip:** Use `--limit 10` to test on subset first (~7.5 minutes)

**Incremental Processing:** By default, only experiments without existing results are analyzed. Use `--all` to force reanalysis of all experiments.

**Export-Only Mode:** `--export-only` skips all analysis and only aggregates existing JSON results into Parquet files. No new analyses are run.

## Result Schema

**Success:**

```json
{
  "experiment_name": "NSC_249",
  "success": true,
  "rwp": 12.04,
  "num_phases": 3,
  "phases": [
    {
      "phase_name": "Na3Zr2Si2PO12_...",
      "spacegroup": "R-3c",
      "weight_fraction": 0.45
    }
  ]
}
```

**Failure:**

```json
{
  "experiment_name": "NSC_145",
  "success": false,
  "error": "No peaks are detected in the pattern.",
  "error_type": "no_peaks_detected"
}
```

**Error types:** `no_peaks_detected`, `cif_download`, `timeout`, `no_chemical_system`, `dara_internal_error`, `analysis_error`

## Design Decisions

| Question                             | Decision                                                                                                    |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Filter low-quality patterns?         | No pre-filter. Flag post-analysis via `Rwp > 30%`                                                           |
| Only completed experiments?          | Yes, failed experiments have incomplete XRD                                                                 |
| Store failures in Parquet?           | Yes. `xrd_refinements` includes all (success + failures). Dashboard shows friendly failure messages.        |
| Handle partial CIF downloads?        | CIF cache tracks download attempts. Re-download if <3 attempts and incomplete. Metadata: `_cache_meta.json` |
| Retry failed analyses automatically? | No. Use `--all` flag or delete JSON result file to re-trigger                                               |
| Parallel experiment logs?            | Prefix all logs with `[experiment_name]` for clarity                                                        |
| Single-phase weight fraction?        | Dashboard shows 100% for single phase (by definition all crystalline content)                               |
| Compare target vs actual?            | Planned: add `target_achieved` field (TODO)                                                                 |
