# Data Tools

Utilities for working with A-Lab experimental data.

## Quick Start

```bash
# Compare MongoDB collection against all Parquet files (PRIMARY WORKFLOW)
./analyze.sh temporary/release

# Generate schema diagrams
./diagram.sh

# Alternative: use dedicated comparison with defaults
./compare.sh
```

---

## analyze.sh - MongoDB Analysis & Comparison

**Intelligent wrapper that automatically chooses between database analysis and schema comparison.**

### Usage

```bash
# List all databases
./analyze.sh

# Analyze database structure (patterns, statistics)
./analyze.sh temporary

# Compare collection against all Parquet files (PRIMARY USE CASE)
./analyze.sh temporary/release

# Quick comparison with sampling
./analyze.sh temporary/release --sample 100

# Export comparison report
./analyze.sh temporary/release --output report.json
```

### How It Works

The script intelligently determines what to do based on your input:

- **No arguments** → Lists all MongoDB databases
- **Database only** (e.g., `temporary`) → Analyzes database structure using `analyze_mongodb.py`
- **Database/Collection** (e.g., `temporary/release`) → **Compares MongoDB collection against all `../parquet/*.parquet` files** using `compare_schemas.py`

### Output (Collection Comparison Mode)

When comparing a collection, you'll see:

- ✓ **Matched fields** between MongoDB and Parquet
- ⚠️ **Type mismatches** between sources
- ➕ **MongoDB only** fields (potentially missing in Parquet)
- ➕ **Parquet only** fields (computed or derived fields)
- **Coverage statistics** for all fields

---

## compare.sh - Direct Schema Comparison

**Dedicated comparison tool with pre-configured defaults.**

### Usage

```bash
# Default comparison (hardcoded MongoDB URI)
./compare.sh

# Quick analysis
./compare.sh --sample 100

# Export results
./compare.sh --output comparison_report.json
```

### Configuration

By default compares:

- MongoDB: `mongodb://localhost:27017/temporary/release`
- Parquet: `../parquet/`

Edit `compare.sh` to change these defaults.

### Manual Usage (Advanced)

For custom MongoDB URIs or different parquet directories:

```bash
source ../venv/bin/activate

python compare_schemas.py \
    --mongo mongodb://host:port/database/collection \
    --parquet /path/to/parquet/ \
    --sample 1000 \
    --output report.json
```

### Options

| Option               | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `--mongo, -m URI`    | MongoDB URI: `mongodb://host:port/database/collection` |
| `--parquet, -p PATH` | Parquet file, directory, or glob pattern               |
| `--sample, -s N`     | Number of docs to sample (default: 1000)               |
| `--output, -o FILE`  | Export results to JSON file                            |

### Understanding Comparison Results

| Result Type         | Meaning                                     | Action                                                       |
| ------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Matched Fields**  | Successfully mapped between sources         | Check for type mismatches                                    |
| **MongoDB Only**    | Field exists in MongoDB but not Parquet     | Evaluate: intentionally excluded? Low coverage? Need to add? |
| **Parquet Only**    | Derived/computed fields (IDs, calculations) | Usually OK, verify they're intentional                       |
| **Type Mismatches** | Different data types (e.g., int→int64)      | Usually safe, verify compatibility                           |
| **Coverage Issues** | Present in <90% of MongoDB documents        | Handle nulls appropriately                                   |

### Use Cases

1. **Migration Validation** - Verify MongoDB → Parquet transformation completeness
2. **Schema Monitoring** - Detect new MongoDB fields that need Parquet mapping
3. **Data Quality** - Identify missing or incomplete fields
4. **Documentation** - Generate schema comparison reports

---

## diagram.sh - Schema Diagram Generator

Generate visual database diagrams from Parquet files.

### Usage

```bash
# Display in terminal
./diagram.sh

# Generate Mermaid ERD (for GitHub/docs)
./diagram.sh --format mermaid

# Export documentation
./diagram.sh --output schema_docs.md
```

### Output Formats

- **Terminal** - Rich display with tables, columns, types, relationships, and row counts
- **Mermaid ERD** - GitHub/GitLab compatible entity-relationship diagrams
- **Markdown** - Complete schema documentation with tables

### Integration

Automatically runs in the main pipeline:

```bash
../../update_data.sh  # Generates data/SCHEMA_DIAGRAM.md
```

### Manual Usage

```bash
source ../venv/bin/activate
python generate_diagram.py /path/to/parquet/
```

---

## Tool Summary

| Tool           | Purpose                                    | Command                          |
| -------------- | ------------------------------------------ | -------------------------------- |
| **analyze.sh** | Compare MongoDB collection → Parquet files | `./analyze.sh temporary/release` |
| **compare.sh** | Schema comparison with defaults            | `./compare.sh`                   |
| **diagram.sh** | Generate visual schema diagrams            | `./diagram.sh`                   |

### Python Scripts

- **analyze_mongodb.py** - MongoDB database exploration and analysis
- **compare_schemas.py** - Detailed MongoDB vs Parquet comparison engine
- **generate_diagram.py** - Schema diagram generator (terminal, Mermaid, Markdown)

All tools are integrated into the main pipeline: `../../update_data.sh`
