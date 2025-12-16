#!/bin/bash
#
# Wrapper script to compare MongoDB against Parquet files
#
# Usage:
#   ./analyze.sh                                      # List databases
#   ./analyze.sh temporary/release                    # Compare collection to parquet
#   ./analyze.sh temporary/release --sample 100       # Quick comparison
#   ./analyze.sh temporary/release --output report.json  # Export results
#
# Compares MongoDB collection schemas against all parquet files in ../parquet/
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../venv/bin/activate

# Default parquet directory
PARQUET_DIR="../parquet"

# Build MongoDB URI
if [ -z "$1" ]; then
    # No arguments - list databases using the old analyzer
    URI="mongodb://localhost:27017"
    python analyze_mongodb.py "$URI"
elif [[ "$1" != *"/"* ]]; then
    # Database only (no collection) - analyze database
    URI="mongodb://localhost:27017/$1"
    python analyze_mongodb.py "$URI"
else
    # Has collection path - run comparison against parquet
    URI="mongodb://localhost:27017/$1"
    shift  # Remove first arg so we can pass remaining args
    
    echo "Comparing MongoDB collection to Parquet files..."
    echo "  MongoDB: $URI"
    echo "  Parquet: $PARQUET_DIR"
    echo ""
    
    # Run comparison
    python compare_schemas.py \
        --mongo "$URI" \
        --parquet "$PARQUET_DIR" \
        "$@"
fi

