#!/bin/bash
#
# Quick verification script for all README claims
# Run this to independently verify all performance metrics
#

set -e

echo "=========================================================================="
echo "RAGWall Claims Verification Script"
echo "=========================================================================="
echo ""
echo "This script verifies all performance claims made in README.md"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results
PASS=0
FAIL=0

# Function to print test result
check_claim() {
    local test_name="$1"
    local expected="$2"
    local actual="$3"
    local tolerance="${4:-0.01}"  # Default 1% tolerance

    echo ""
    echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "${BLUE}TEST: $test_name${NC}"
    echo "Expected: $expected"
    echo "Actual:   $actual"

    # For numeric comparisons
    if [[ $expected =~ ^[0-9.]+$ ]] && [[ $actual =~ ^[0-9.]+$ ]]; then
        diff=$(echo "$actual - $expected" | bc -l | sed 's/^-//')
        if (( $(echo "$diff < $tolerance" | bc -l) )); then
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++))
        else
            echo -e "${RED}❌ FAIL (difference: $diff)${NC}"
            ((FAIL++))
        fi
    # For string/exact comparisons
    elif [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((FAIL++))
    fi
}

echo ""
echo "=========================================================================="
echo "1. HRCR@5 Reduction - Claim: 48% (from health_100_matched_eval)"
echo "=========================================================================="

if [ -f "reports/health_100_matched_eval/rag_ab_summary.json" ]; then
    HRCR5_DROP=$(python3 << 'EOF'
import json
with open('reports/health_100_matched_eval/rag_ab_summary.json', 'r') as f:
    data = json.load(f)
    print(f"{data['HRCR@5']['relative_drop'] * 100:.1f}")
EOF
)
    check_claim "HRCR@5 Reduction (%)" "48.2" "$HRCR5_DROP" "2.0"
else
    echo -e "${RED}❌ Test report not found: reports/health_100_matched_eval/rag_ab_summary.json${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "2. HRCR@10 Reduction - Claim: 48% (from health_100_matched_eval)"
echo "=========================================================================="

if [ -f "reports/health_100_matched_eval/rag_ab_summary.json" ]; then
    HRCR10_DROP=$(python3 << 'EOF'
import json
with open('reports/health_100_matched_eval/rag_ab_summary.json', 'r') as f:
    data = json.load(f)
    print(f"{data['HRCR@10']['relative_drop'] * 100:.1f}")
EOF
)
    check_claim "HRCR@10 Reduction (%)" "48.0" "$HRCR10_DROP" "2.0"
else
    echo -e "${RED}❌ Test report not found${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "3. Test Scale - Claim: 100 queries (50 attacked, 50 benign)"
echo "=========================================================================="

if [ -f "reports/health_100_matched_eval/rag_ab_summary.json" ]; then
    N_QUERIES=$(python3 << 'EOF'
import json
with open('reports/health_100_matched_eval/rag_ab_summary.json', 'r') as f:
    data = json.load(f)
    print(data['N_queries'])
EOF
)
    check_claim "Number of queries" "100" "$N_QUERIES"
else
    echo -e "${RED}❌ Test report not found${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "4. Benign Query Drift - Claim: 0% (perfect faithfulness)"
echo "=========================================================================="

if [ -f "reports/health_100_matched_eval/rag_ab_summary.json" ]; then
    DRIFT=$(python3 << 'EOF'
import json
with open('reports/health_100_matched_eval/rag_ab_summary.json', 'r') as f:
    data = json.load(f)
    print(f"{data['Benign_Jaccard@5']['drift'] * 100:.1f}")
EOF
)
    check_claim "Benign Query Drift (%)" "0.0" "$DRIFT"
else
    echo -e "${RED}❌ Test report not found${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "5. Bootstrap Validation - Claim: 50 samples with 95% CI"
echo "=========================================================================="

if [ -f "reports/health_100_matched_eval/rag_ab_summary.json" ]; then
    BOOTSTRAP=$(python3 << 'EOF'
import json
with open('reports/health_100_matched_eval/rag_ab_summary.json', 'r') as f:
    data = json.load(f)
    print(data['knobs']['bootstrap_samples'])
EOF
)
    check_claim "Bootstrap samples" "50" "$BOOTSTRAP"
else
    echo -e "${RED}❌ Test report not found${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "6. Spanish Support - Claim: 96% detection, 0% FPR"
echo "=========================================================================="

if [ -f "tests/test_spanish_production.py" ]; then
    echo "Running Spanish test suite..."
    python tests/test_spanish_production.py > /tmp/spanish_test_output.txt 2>&1

    SPANISH_DETECTION=$(grep "Attack Detection:" /tmp/spanish_test_output.txt | grep -oE "[0-9]+\.[0-9]+" | head -1)
    SPANISH_FPR=$(grep "False Positive Rate:" /tmp/spanish_test_output.txt | grep -oE "[0-9]+\.[0-9]+" | head -1)

    check_claim "Spanish Detection Rate (%)" "96.0" "$SPANISH_DETECTION" "2.0"
    check_claim "Spanish False Positive Rate (%)" "0.0" "$SPANISH_FPR"
else
    echo -e "${RED}❌ Spanish test not found${NC}"
    ((FAIL++))
fi

echo ""
echo "=========================================================================="
echo "7. Data Files Exist"
echo "=========================================================================="

echo ""
echo "${BLUE}Checking test data files...${NC}"

declare -a required_files=(
    "reports/health_100_matched_eval/rag_ab_summary.json"
    "data/health_care_100_queries_matched.jsonl"
    "docs/AB_EVAL_ENHANCED_RESULTS.md"
    "docs/PATTERN_ENHANCEMENT_SUMMARY.md"
    "docs/EVIDENCE_MAP_RAG_TESTS.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅${NC} $file"
        ((PASS++))
    else
        echo -e "  ${RED}❌${NC} $file (MISSING)"
        ((FAIL++))
    fi
done

echo ""
echo "=========================================================================="
echo "8. Credential Theft Disclosure Check"
echo "=========================================================================="

if [ -f "docs/AB_EVAL_ENHANCED_RESULTS.md" ]; then
    MENTIONS=$(grep -c "credential_theft" docs/AB_EVAL_ENHANCED_RESULTS.md || echo 0)
    echo ""
    echo "Credential theft weakness mentioned: $MENTIONS times in AB_EVAL_ENHANCED_RESULTS.md"

    if [ "$MENTIONS" -ge 4 ]; then
        echo -e "${GREEN}✅ PASS - Weakness is properly disclosed (>= 4 mentions)${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌ FAIL - Insufficient disclosure${NC}"
        ((FAIL++))
    fi
fi

echo ""
echo "=========================================================================="
echo "FINAL RESULTS"
echo "=========================================================================="
echo ""
echo -e "Tests Passed: ${GREEN}$PASS${NC}"
echo -e "Tests Failed: ${RED}$FAIL${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ ALL CLAIMS VERIFIED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "All README claims are accurate and supported by test evidence."
    echo ""
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ SOME CLAIMS NOT VERIFIED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Some tests failed. Check output above for details."
    echo ""
    exit 1
fi
