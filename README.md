# A-Lab Experiments Dashboard

Complete pipeline for MongoDB → Parquet → Dashboard visualization.

```
┌─────────────────────────────────────────────────────────────────┐
│ Data Product Configuration │
│ (YAML/JSON defining: filters, schema, analyses, metadata) │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Project Selection │
│ • Read MongoDB │
│ • Filter by experiment_filter criteria │
│ • Store selected experiments list │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Data Transformation │
│ • mongodb_to_parquet.py (filtered subset) │
│ • Validate with Pydantic schema │
│ • Output: product_name/\*.parquet │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Analysis Pipeline (Configurable) │
│ • Load analyses from config │
│ • Run each analyzer: xrd_dara.py, powder_stats.py, etc. │
│ • Each writes results to parquet │
│ • Validate output schemas │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: S3 OpenData Upload │
│ • Upload parquet files directly to S3 │
│ • Embed metadata in Arrow schema │
│ • Available at s3://materialsproject-contribs │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ Dashboard (reads local parquet files) │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Generate Parquet files from MongoDB
./update_data.sh

# 2. (Optional) Run XRD phase analysis
cd data/xrd_creation
./run_batch_analysis.sh

# 3. Launch dashboard (auto-setup on first run)
./run_dashboard.sh
```

Dashboard opens at http://127.0.0.1:8050

**Note:** `run_dashboard.sh` automatically sets up the environment on first run (creates venv, installs dependencies)

---

## What Each Script Does

**`update_data.sh`** - Data pipeline (first time or when updating)

- Creates virtual environment (if needed)
- Installs data pipeline dependencies
- MongoDB → Parquet transformation
- Schema diagram generation
- Data validation

**`run_dashboard.sh`** - Dashboard launcher

- Creates dashboard virtual environment (first run only)
- Installs dashboard dependencies (first run only)
- Launches Plotly Dash server

**`data/xrd_creation/run_batch_analysis.sh`** - XRD batch phase analysis (optional, compute-intensive)

- Identifies crystalline phases using DARA for multiple experiments
- Generates refinement results
- Use `run_single_analysis.sh` for analyzing individual experiments

---

## Workflow

**Update Data:** MongoDB has new experiments

```bash
./update_data.sh              # Full transformation
./update_data.sh --fast       # Skip large arrays (faster)
./update_data.sh --test       # Test with 10 experiments
```

**View Dashboard:** Visualize the data

```bash
./run_dashboard.sh            # Launch with password
./run_dashboard.sh --no-pass  # Launch without password
```

To update data, run `./update_data.sh` separately before launching the dashboard.

**XRD Phase Analysis:** Identify crystalline phases (compute-intensive)

```bash
cd data/xrd_creation

# Batch analysis (multiple experiments)
./run_batch_analysis.sh                      # Incremental (skip existing)
./run_batch_analysis.sh --limit 10           # Test with 10 experiments
./run_batch_analysis.sh --all                # Rerun all experiments
./run_batch_analysis.sh --all --limit 20     # Rerun first 20 only
./run_batch_analysis.sh --experiment NSC_249 # Single experiment
./run_batch_analysis.sh --workers 4          # Parallel (4 workers)
./run_batch_analysis.sh --export-only        # Consolidate JSON → Parquet only

# Single experiment analysis
./run_single_analysis.sh NSC_249             # Analyze one experiment
./run_single_analysis.sh --list              # List available experiments
```

**Product Pipeline:** Create and manage data products for S3 OpenData

```bash
./run_product_pipeline.sh create                    # Create new product
./run_product_pipeline.sh list                      # List products
./run_product_pipeline.sh run --product <name>      # Run pipeline (dry run)
./run_product_pipeline.sh run --product <name> --upload  # Upload (MPContribs + S3)
./run_product_pipeline.sh status --product <name>   # Check status
```

---

## XRD Analysis Notes

**Compute Requirements:**

- ~45 seconds per experiment (first run downloads CIF references)
- 576 experiments × 45 sec = ~7 hours sequential
- With 4 workers: ~1.8 hours
- **Tip:** Use `--limit 10` to test on subset first

**Note on `--export-only`:** This flag skips all analysis and only aggregates existing JSON results into Parquet files. Useful for consolidating results after running individual analyses.

**Decisions Made (verify with lab team):**

| Question                            | Decision                                          |
| ----------------------------------- | ------------------------------------------------- |
| Skip low-quality patterns?          | No pre-filter. Flag post-analysis via `Rwp > 30%` |
| Only analyze completed experiments? | Yes, failed experiments have incomplete XRD       |
| Compare target vs actual phases?    | Planned: add `target_achieved` field (TODO)       |

---

## File Structure

```
A-Lab_Samples/
├── update_data.sh                # MongoDB → Parquet transformation
├── run_analysis.sh               # XRD phase analysis (DARA)
├── run_dashboard.sh              # Launch dashboard
├── run_product_pipeline.sh       # Data product pipeline
│
├── plotly_dashboard/
│   ├── app.py                    # Dash application (uses Parquet)
│   └── parquet_data_loader.py   # Parquet data loader
│
└── data/
    ├── requirements.txt          # Python dependencies
    ├── SCHEMA_DIAGRAM.md         # Auto-generated schema docs
    ├── mongodb_to_parquet.py     # MongoDB → Parquet transformation
    │
    ├── parquet/                  # Generated Parquet files
    │   ├── experiments.parquet
    │   ├── xrd_refinements.parquet  # DARA analysis results
    │   └── xrd_phases.parquet       # Identified phases
    │
    ├── pipeline/                 # Product pipeline system
    │   ├── product_pipeline.py   # Main pipeline orchestrator
    │   └── pipeline_runs.parquet # Pipeline execution history
    │
    ├── products/                 # Data product definitions
    │   └── schema/               # Pydantic schemas (auto-discovered)
    │
    ├── analyses/                 # Analysis plugins (auto-discovered)
    │   └── base_analyzer.py      # Base analyzer class
    │
    ├── config/                   # Configuration files
    │   ├── defaults.yaml         # Pipeline defaults
    │   └── filters.yaml          # Filter presets
    │
    ├── xrd_creation/             # XRD analysis pipeline
    │   ├── analyze_batch.py      # Batch processing
    │   ├── xrd_utils.py          # Shared utilities
    │   └── results/              # JSON results (per experiment)
    │
    └── tools/                    # Analysis utilities
        ├── analyze_mongodb.py    # Explore any MongoDB
        ├── compare_schemas.py    # MongoDB vs Parquet comparison
        └── generate_diagram.py   # Schema visualization
```

---

## Documentation

- **`data/SCHEMA_DIAGRAM.md`** - Auto-generated schema with relationships
- **`data/tools/README.md`** - Analysis tools (MongoDB explorer, schema comparison)

---

## Tools

```bash
cd data/tools

# Explore any MongoDB database
./analyze.sh temporary/release

# Compare MongoDB vs Parquet schemas
./compare.sh

# Generate schema diagram
./diagram.sh
```

All tools use the shared venv automatically.
