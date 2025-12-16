#!/bin/bash
# =============================================================================
# A-Lab Pipeline Integration Test Runner
# =============================================================================
# Runs comprehensive integration tests for the A-Lab pipeline including:
# - Auto-discovery (schemas, analyses)
# - Filter configurations
# - Hook system
# - Edge cases and error handling
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --verbose    # Verbose output
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  A-Lab Pipeline Integration Tests${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if venv exists
if [ ! -d "data/venv" ]; then
    echo -e "${YELLOW}→ Creating virtual environment...${NC}"
    python3 -m venv data/venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

# Activate venv
echo -e "${YELLOW}→ Activating virtual environment...${NC}"
source data/venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}→ Checking dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r data/requirements.txt 2>&1 | grep -v "UserWarning" | grep -v "Valid config keys" || true
echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# Run tests
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Running Tests...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Run the test file
python3 data/pipeline/test_integrated_pipeline.py "$@"

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - Ensure MongoDB is running (for MongoDB tests)"
    echo "  - Check that all dependencies are installed"
    echo "  - Review test output above for details"
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exit $TEST_EXIT_CODE

