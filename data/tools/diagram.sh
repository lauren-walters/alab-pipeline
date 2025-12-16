#!/bin/bash
#
# Wrapper script to generate schema diagram with venv
#
# Usage:
#   ./diagram.sh                           # Terminal display
#   ./diagram.sh --format mermaid          # Mermaid ERD
#   ./diagram.sh --output schema.md        # Export to file
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../venv/bin/activate

# Run diagram generator
python generate_diagram.py ../parquet/ "$@"

