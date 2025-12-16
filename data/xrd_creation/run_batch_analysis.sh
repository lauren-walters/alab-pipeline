#!/bin/bash
# =============================================================================
# XRD Phase Analysis Runner
# =============================================================================
# Runs DARA XRD phase identification on A-Lab experiments
#
# Usage:
#   ./run_analysis.sh                      # Incremental (skip existing)
#   ./run_analysis.sh --limit 10           # First 10 experiments only
#   ./run_analysis.sh --all                # Rerun all experiments
#   ./run_analysis.sh --all --limit 20     # Rerun first 20 experiments
#   ./run_analysis.sh --experiment NSC_249 # Single experiment
#   ./run_analysis.sh --workers 4          # Use 4 parallel workers
#   ./run_analysis.sh --export-only        # Export results to Parquet only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

XRD_DIR="$SCRIPT_DIR/data/xrd_creation"
VENV_DIR="$XRD_DIR/venv"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  A-Lab XRD Phase Analysis (DARA)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Parquet data exists
PARQUET_DIR="data/parquet"
if [ ! -f "$PARQUET_DIR/experiments.parquet" ]; then
    echo -e "${RED}Error: No Parquet data found${NC}"
    echo "Run ./update_data.sh first to generate data from MongoDB"
    exit 1
fi

# Check/create venv
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}→ Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

echo -e "${YELLOW}→ Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}→ Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r "$XRD_DIR/requirements.txt"
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Run analysis
cd "$XRD_DIR"
python analyze_batch.py "$@"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ XRD Analysis complete!${NC}"
echo ""
echo "Results:"
echo "  JSON: data/xrd_creation/results/"
echo "  Parquet: data/parquet/xrd_refinements.parquet"
echo ""
echo "View in dashboard:"
echo "  ./run_dashboard.sh"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

