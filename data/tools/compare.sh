#!/bin/bash
#
# Wrapper script to run schema comparison with venv
#
# Usage:
#   ./compare.sh
#   ./compare.sh --sample 100
#   ./compare.sh --output report.json
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../venv/bin/activate

# Run comparison with default or custom arguments
python compare_schemas.py \
    --mongo mongodb://localhost:27017/temporary/release \
    --parquet ../parquet/ \
    "$@"

