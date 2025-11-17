#!/bin/bash
# Fine-tune the transformer model with healthcare domain tokens

set -e

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  FINE-TUNING TRANSFORMER WITH HEALTHCARE DOMAIN TOKENS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
DATASET="evaluations/benchmark/data/health_care_1000_queries_converted.jsonl"
DOMAIN="healthcare"
DOMAIN_TOKEN="[DOMAIN_HEALTHCARE]"
BASE_MODEL="ProtectAI/deberta-v3-base-prompt-injection-v2"
OUTPUT_DIR="models/healthcare_domain_finetuned"
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Domain: $DOMAIN"
echo "  Domain Token: $DOMAIN_TOKEN"
echo "  Base Model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Check dataset exists
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  STARTING FINE-TUNING"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Run fine-tuning
PYTHONPATH=$PWD \
competitor_test_env/bin/python3 \
sanitizer/ml/fine_tune_domain_tokens.py \
"$DATASET" \
--domain "$DOMAIN" \
--domain-token "$DOMAIN_TOKEN" \
--base-model "$BASE_MODEL" \
--output-dir "$OUTPUT_DIR" \
--epochs $EPOCHS \
--batch-size $BATCH_SIZE \
--learning-rate $LEARNING_RATE

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  FINE-TUNING COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Model saved to: $OUTPUT_DIR/final"
echo ""
echo "To use the fine-tuned model:"
echo "  export RAGWALL_USE_TRANSFORMER_FALLBACK=1"
echo "  export RAGWALL_TRANSFORMER_MODEL=$OUTPUT_DIR/final"
echo "  export RAGWALL_TRANSFORMER_DOMAIN=$DOMAIN"
echo ""
echo "To test the fine-tuned model:"
echo "  PYTHONPATH=\$PWD \\"
echo "  RAGWALL_USE_TRANSFORMER_FALLBACK=1 \\"
echo "  RAGWALL_TRANSFORMER_MODEL=$OUTPUT_DIR/final \\"
echo "  RAGWALL_TRANSFORMER_DOMAIN=$DOMAIN \\"
echo "  competitor_test_env/bin/python3 \\"
echo "  evaluations/benchmark/scripts/compare_real.py \\"
echo "  evaluations/benchmark/data/health_care_1000_queries_converted.jsonl \\"
echo "  --systems ragwall"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
