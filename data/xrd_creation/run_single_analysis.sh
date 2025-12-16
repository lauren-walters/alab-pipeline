#!/bin/bash
# =============================================================================
# DARA XRD Analysis Runner
# =============================================================================
# Runs XRD phase analysis on A-Lab experiments using DARA
#
# Usage:
#   ./run_analysis.sh NSC_249           # Analyze single experiment
#   ./run_analysis.sh --list            # List available experiments
#   ./run_analysis.sh --setup           # Setup virtual environment only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  DARA XRD Analysis${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}→ Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi
    
    echo -e "${YELLOW}→ Activating virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
    
    echo -e "${YELLOW}→ Installing dependencies...${NC}"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
}

show_help() {
    echo "Usage: ./run_analysis.sh [OPTIONS] [EXPERIMENT_NAME]"
    echo ""
    echo "Options:"
    echo "  --setup        Setup virtual environment only"
    echo "  --list, -l     List available experiments"
    echo "  --help, -h     Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run_analysis.sh NSC_249              # Run phase search"
    echo "  ./run_analysis.sh NSC_249 --mode refinement"
    echo "  ./run_analysis.sh --list"
    echo ""
    echo "Output:"
    echo "  results/<experiment>_result.json"
    echo "  results/<experiment>.xy"
}

# Parse arguments
EXPERIMENT=""
EXTRA_ARGS=""
SETUP_ONLY=false
LIST_ONLY=false

for arg in "$@"; do
    case $arg in
        --setup)
            SETUP_ONLY=true
            ;;
        --list|-l)
            LIST_ONLY=true
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --mode|--wmin|--wmax)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
        -*)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
        *)
            if [ -z "$EXPERIMENT" ]; then
                EXPERIMENT="$arg"
            else
                EXTRA_ARGS="$EXTRA_ARGS $arg"
            fi
            ;;
    esac
done

print_header

# Setup venv
setup_venv

if [ "$SETUP_ONLY" = true ]; then
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo ""
    echo "Run analysis with:"
    echo "  ./run_analysis.sh NSC_249"
    exit 0
fi

if [ "$LIST_ONLY" = true ]; then
    python analyze_single.py --list
    exit 0
fi

if [ -z "$EXPERIMENT" ]; then
    echo -e "${RED}Error: No experiment specified${NC}"
    echo ""
    show_help
    exit 1
fi

# Check if Parquet data exists
PARQUET_DIR="../parquet"
if [ ! -f "$PARQUET_DIR/experiments.parquet" ]; then
    echo -e "${RED}Error: No Parquet data found${NC}"
    echo "Run ./update_data.sh from project root first"
    exit 1
fi

# Run analysis
echo -e "${BLUE}Analyzing: $EXPERIMENT${NC}"
echo ""

python analyze_single.py "$EXPERIMENT" $EXTRA_ARGS

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Analysis complete!${NC}"
echo ""
echo "Results saved to:"
echo "  results/${EXPERIMENT}_result.json"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

