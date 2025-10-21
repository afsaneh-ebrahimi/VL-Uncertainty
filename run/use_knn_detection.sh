#!/usr/bin/env bash
# Helper script to harvest a reference set and run VL_Uncertainty with k-NN geometric detection.
#
# Usage:
#   bash run/use_knn_detection.sh
#
# Environment variables can override defaults:
#   LVLM               LVLM backbone name (default: Qwen2-VL-2B-Instruct)
#   BENCHMARK          Benchmark dataset (default: MMVet)
#   LLM                Verification LLM (default: Qwen2.5-3B-Instruct)
#   REF_PATH           Output path for cached reference embeddings (default: reference/qwen2_mmvett.pt)
#   MAX_SAMPLES        Number of benchmark samples to scan while harvesting references (default: 200)
#   MAX_REFERENCE      Maximum number of accepted reference items (default: 128)
#   KNN_K              Nearest-neighbour count (default: 10)
#   GEOMETRIC_METRIC   Distance metric, "cosine" or "euclidean" (default: cosine)
#   GEOMETRIC_THRESHOLD Detection threshold used during inference (default: 0.5)
#   FORCE_REBUILD      Set to 1 to rebuild the reference cache even if it already exists
#
# The script first builds (or reuses) the reference cache and then runs VL_Uncertainty
# in geometric-only mode with k-NN scoring.

set -euo pipefail

LVLM_NAME=${LVLM:-"Qwen2-VL-2B-Instruct"}
BENCHMARK_NAME=${BENCHMARK:-"MMVet"}
LLM_NAME=${LLM:-"Qwen2.5-3B-Instruct"}
REF_PATH=${REF_PATH:-"reference/qwen2_mmvett.pt"}
MAX_SAMPLES=${MAX_SAMPLES:-200}
MAX_REFERENCE=${MAX_REFERENCE:-128}
KNN_K=${KNN_K:-10}
GEOMETRIC_METRIC=${GEOMETRIC_METRIC:-"cosine"}
GEOMETRIC_THRESHOLD=${GEOMETRIC_THRESHOLD:-0.5}
FORCE_REBUILD=${FORCE_REBUILD:-0}

mkdir -p "$(dirname "$REF_PATH")"

if [[ "$FORCE_REBUILD" == "1" || ! -f "$REF_PATH" ]]; then
  echo "[Step 1/2] Building reference embeddings at $REF_PATH"
  python run/build_reference.py \
    --lvlm "$LVLM_NAME" \
    --benchmark "$BENCHMARK_NAME" \
    --llm "$LLM_NAME" \
    --output "$REF_PATH" \
    --max_samples "$MAX_SAMPLES" \
    --max_reference "$MAX_REFERENCE" \
    --progress_bar
else
  echo "[Step 1/2] Reusing existing reference embeddings at $REF_PATH"
fi

echo "[Step 2/2] Running VL_Uncertainty with k-NN geometric detection"
python VL_Uncertainty.py \
  --lvlm "$LVLM_NAME" \
  --benchmark "$BENCHMARK_NAME" \
  --llm "$LLM_NAME" \
  --detection_strategy geometric_only \
  --geometric_score_types knn \
  --geometric_metric "$GEOMETRIC_METRIC" \
  --knn_k "$KNN_K" \
  --geometric_threshold "$GEOMETRIC_THRESHOLD" \
  --reference_path "$REF_PATH"
