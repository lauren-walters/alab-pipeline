#!/bin/bash
#
# Single command to update all parquet data files from MongoDB
#
# Usage:
#   ./update_data.sh              # Full update (includes all data)
#   ./update_data.sh --fast       # Skip large arrays (faster)
#   ./update_data.sh --test       # Process only 10 experiments
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  A-Lab Data Update: MongoDB → Parquet"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse arguments
EXTRA_ARGS=""
MODE="full"

for arg in "$@"; do
    case $arg in
        --fast)
            EXTRA_ARGS="--skip-temp-logs --skip-xrd-points"
            MODE="fast"
            shift
            ;;
        --test)
            EXTRA_ARGS="--limit 10"
            MODE="test"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            shift
            ;;
    esac
done

echo "Mode: $MODE"
echo ""

# Check if MongoDB is running
echo "→ Checking MongoDB connection..."
if ! mongosh --quiet --eval "db.version()" > /dev/null 2>&1; then
    echo "✗ MongoDB is not running!"
    echo ""
    echo "Start MongoDB with:"
    echo "  brew services start mongodb-community@7.0"
    exit 1
fi
echo "✓ MongoDB is running"
echo ""

# Check if virtual environment exists
if [ ! -d "data/venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv data/venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "→ Activating virtual environment..."
source data/venv/bin/activate

# Install/update dependencies
echo "→ Installing dependencies..."
pip install -q -r data/requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run transformation
echo "→ Transforming MongoDB data to Parquet..."
echo ""
python data/mongodb_to_parquet.py $EXTRA_ARGS

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Data transformation complete!"
echo ""

# Generate schema diagram
echo "→ Generating schema diagram..."
python data/tools/generate_diagram.py data/parquet/ \
    --format all \
    --output data/SCHEMA_DIAGRAM.md

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All tasks complete!"
echo ""
echo "📁 Parquet files:"
echo "   data/parquet/"
echo ""
echo "📋 Schema diagram:"
echo "   data/SCHEMA_DIAGRAM.md"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Next step: View your data in the dashboard!"
echo ""
echo "   ./run_dashboard.sh"
echo ""
echo "   The dashboard will automatically load data from the"
echo "   Parquet files and be available at http://127.0.0.1:8050"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

