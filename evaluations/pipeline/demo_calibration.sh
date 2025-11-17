#!/bin/bash
# Demo script for threshold calibration workflow

set -e

echo "=== RAGWall Threshold Calibration Demo ==="
echo

# Step 1: Create calibration dataset
echo "Step 1: Creating calibration dataset..."
cat evaluations/pipeline/attacks/queries_attack.jsonl \
    evaluations/pipeline/attacks/queries_benign.jsonl \
    > evaluations/pipeline/calibration_full.jsonl

echo "✓ Created calibration_full.jsonl with $(wc -l < evaluations/pipeline/calibration_full.jsonl) queries"
echo

# Step 2: Run calibration
echo "Step 2: Running threshold calibration for healthcare domain..."
echo "  (This may take 2-5 minutes on first run due to model download)"
echo

PYTHONPATH=$PWD python3 evaluations/pipeline/scripts/calibrate_thresholds.py \
    evaluations/pipeline/calibration_full.jsonl \
    --domain healthcare \
    --target-fp 0.0 \
    --output evaluations/pipeline/runs/healthcare_threshold.json

echo
echo "✓ Calibration complete"
echo

# Step 3: Display results
echo "Step 3: Calibration results"
echo

if [ -f evaluations/pipeline/runs/healthcare_threshold.json ]; then
    python3 << 'EOF'
import json

with open('evaluations/pipeline/runs/healthcare_threshold.json') as f:
    cal = json.load(f)

print(f"Domain: {cal['domain']}")
print(f"Recommended threshold: {cal['threshold']:.4f}")
print(f"Detection rate: {cal['detection_rate']*100:.2f}%")
print(f"False-positive rate: {cal['false_positive_rate']*100:.2f}%")
print(f"Attack samples: {cal['attack_samples']}")
print(f"Benign samples: {cal['benign_samples']}")
print()
print("Score statistics:")
import statistics
print(f"  Benign mean: {statistics.mean(cal['benign_scores']):.4f}")
print(f"  Attack mean: {statistics.mean(cal['attack_scores']):.4f}")
print(f"  Separation: {statistics.mean(cal['attack_scores']) - statistics.mean(cal['benign_scores']):.4f}")
EOF

    echo
    echo "Full report saved to: evaluations/pipeline/runs/healthcare_threshold.json"
    echo

    # Step 4: Show usage
    echo "Step 4: How to use this threshold"
    echo

    THRESHOLD=$(python3 -c "import json; print(json.load(open('evaluations/pipeline/runs/healthcare_threshold.json'))['threshold'])")

    echo "Option A: Direct threshold override"
    echo "  RAGWALL_USE_TRANSFORMER_FALLBACK=1 \\"
    echo "  RAGWALL_TRANSFORMER_DOMAIN=healthcare \\"
    echo "  RAGWALL_TRANSFORMER_THRESHOLD=${THRESHOLD} \\"
    echo "    python3 your_app.py"
    echo

    echo "Option B: Domain-specific threshold"
    echo "  RAGWALL_TRANSFORMER_DOMAIN_THRESHOLDS=\"healthcare=${THRESHOLD}\" \\"
    echo "    python3 your_app.py"
    echo

    echo "Option C: Test with pipeline"
    echo "  RAGWALL_USE_TRANSFORMER_FALLBACK=1 \\"
    echo "  RAGWALL_TRANSFORMER_THRESHOLD=${THRESHOLD} \\"
    echo "    PYTHONPATH=\$PWD python3 evaluations/pipeline/scripts/run_pipeline.py ragwall attack"
    echo
fi

echo "=== Demo Complete ==="
echo
echo "For more information, see:"
echo "  - evaluations/pipeline/THRESHOLD_CALIBRATION_GUIDE.md"
echo "  - evaluations/pipeline/README.md"
