#!/bin/bash
# Master test runner for all RAGWall verification tests
# Usage: bash evaluations/run_all_tests.sh

set -e  # Exit on error

RAGWALL_ROOT="/Users/rjd/Documents/ragwall"
cd "$RAGWALL_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "                    RAGWALL COMPREHENSIVE TEST SUITE"
echo "================================================================================"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
if [ ! -d "competitor_test_env" ]; then
    echo -e "${RED}Error: competitor_test_env not found${NC}"
    echo "Run: python3 -m venv competitor_test_env"
    exit 1
fi

if ! competitor_test_env/bin/python3 -c "import llm_guard" 2>/dev/null; then
    echo -e "${RED}Error: llm-guard not installed${NC}"
    echo "Run: competitor_test_env/bin/pip install llm-guard"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Test 1: Basic 100-query comparison
echo "================================================================================"
echo -e "${YELLOW}TEST 1: Basic Detection Rate (100 queries)${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/benchmark/scripts/compare_real.py \
evaluations/benchmark/data/health_care_100_queries_matched.jsonl \
--systems llm_guard_real,rebuff_real,ragwall \
--summary evaluations/benchmark/results/summary_100_real.csv

echo -e "${GREEN}✓ Test 1 complete${NC}"
echo ""

# Test 2: Scale test (1000 queries)
echo "================================================================================"
echo -e "${YELLOW}TEST 2: Scale Test (1000 queries)${NC}"
echo "================================================================================"
if [ -f "evaluations/benchmark/data/health_care_1000_queries_converted.jsonl" ]; then
    PYTHONPATH=$RAGWALL_ROOT \
    competitor_test_env/bin/python3 \
    evaluations/benchmark/scripts/compare_real.py \
    evaluations/benchmark/data/health_care_1000_queries_converted.jsonl \
    --systems llm_guard_real,rebuff_real,ragwall \
    --summary evaluations/benchmark/results/summary_1000_real.csv

    echo -e "${GREEN}✓ Test 2 complete${NC}"
else
    echo -e "${YELLOW}⚠ Test 2 skipped (1000-query dataset not found)${NC}"
fi
echo ""

# Test 3: HRCR capability comparison
echo "================================================================================"
echo -e "${YELLOW}TEST 3: HRCR Reduction Capability${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/test_competitor_hrcr.py

echo -e "${GREEN}✓ Test 3 complete${NC}"
echo ""

# Test 4: Multi-language support
echo "================================================================================"
echo -e "${YELLOW}TEST 4: Multi-Language Support${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/test_multilanguage.py

echo -e "${GREEN}✓ Test 4 complete${NC}"
echo ""

# Test 5: Adversarial robustness
echo "================================================================================"
echo -e "${YELLOW}TEST 5: Adversarial Robustness${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/test_adversarial.py

echo -e "${GREEN}✓ Test 5 complete${NC}"
echo ""

# Test 6: Performance benchmark
echo "================================================================================"
echo -e "${YELLOW}TEST 6: Performance & Throughput${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/test_performance.py

echo -e "${GREEN}✓ Test 6 complete${NC}"
echo ""

# Test 7: Combined defense
echo "================================================================================"
echo -e "${YELLOW}TEST 7: Combined Defense (RAGWall + LLM-Guard)${NC}"
echo "================================================================================"
PYTHONPATH=$RAGWALL_ROOT \
competitor_test_env/bin/python3 \
evaluations/test_combined_defense.py

echo -e "${GREEN}✓ Test 7 complete${NC}"
echo ""

# Summary
echo "================================================================================"
echo -e "${GREEN}                    ALL TESTS COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - evaluations/benchmark/results/summary_100_real.csv"
echo "  - evaluations/benchmark/results/summary_1000_real.csv"
echo "  - evaluations/benchmark/results/per_query_real/"
echo ""
echo "View summary:"
echo "  cat evaluations/benchmark/results/summary_100_real.csv"
echo ""
echo "Next steps:"
echo "  - Review VERIFICATION_SUMMARY.md for analysis"
echo "  - Run HRCR reduction test: see REPRODUCE_48_PERCENT_HRCR_REDUCTION.md"
echo ""
