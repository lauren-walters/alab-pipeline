#!/bin/bash
# =============================================================================
# A-Lab Dashboard Launcher
# =============================================================================
# Launches Plotly dashboard with Parquet data
# Automatically sets up environment on first run
#
# Usage:
#   ./run_dashboard.sh              # Launch dashboard
#   ./run_dashboard.sh --no-pass    # Skip password authentication
#   ./run_dashboard.sh --help       # Show help
#
# Note: Run ./update_data.sh separately to refresh data from MongoDB
# =============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 A-Lab Dashboard Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
NO_AUTH=false

for arg in "$@"; do
    case $arg in
        --no-pass)
            NO_AUTH=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_dashboard.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-pass        Skip password authentication"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_dashboard.sh           # Launch with password"
            echo "  ./run_dashboard.sh --no-pass # Launch without password"
            echo ""
            echo "Data Source:"
            echo "  The dashboard loads data from Parquet files:"
            echo "  data/parquet/"
            echo ""
            echo "To update data:"
            echo "  Run ./update_data.sh separately to refresh from MongoDB"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check/create virtual environment for dashboard
VENV_DIR="plotly_dashboard/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 First time setup: Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    
    echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    
    echo -e "${BLUE}📚 Installing dashboard dependencies...${NC}"
    pip install -r plotly_dashboard/requirements.txt --quiet
    
    echo -e "${GREEN}✓ Environment setup complete!${NC}"
    echo ""
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source "$VENV_DIR/bin/activate"
    
    # Check if pyarrow is installed (needed for parquet files)
    if ! python -c "import pyarrow" 2>/dev/null; then
        echo -e "${YELLOW}📦 Installing missing dependencies...${NC}"
        pip install -r plotly_dashboard/requirements.txt --quiet
        echo -e "${GREEN}✓ Dependencies updated${NC}"
    fi
fi

# Check if Parquet data exists
PARQUET_DIR="data/parquet"
EXPERIMENTS_FILE="$PARQUET_DIR/experiments.parquet"

if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo -e "${YELLOW}⚠️  No Parquet data found${NC}"
    echo ""
    echo "Please run this first to generate data from MongoDB:"
    echo "  ./update_data.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Parquet data found${NC}"
echo -e "  Location: $PARQUET_DIR"
echo ""

# Start dashboard
echo -e "${GREEN}🌐 Starting Plotly Dashboard...${NC}"
echo ""
echo "Dashboard URL: http://127.0.0.1:8050"

if [ "$NO_AUTH" = true ]; then
    echo "Authentication: Disabled (--no-pass)"
    export ALAB_NO_AUTH=1
else
    echo "Authentication: Enabled (password: alab)"
fi

echo ""
echo "Press Ctrl+C to stop"
echo ""

cd plotly_dashboard

# Run the dashboard
python app.py

