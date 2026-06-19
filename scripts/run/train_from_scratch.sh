#!/bin/bash
# Train-from-scratch baselines (S_N): train the SAME target architecture with
# random initialization (no load_parameters_path).
#   S50  : TOKENS_B=50   (equal training-token budget)
#   S250 : TOKENS_B=250  (equal total-token budget)
#
# Usage: bash scripts/run/train_from_scratch.sh <MODEL_CONFIG> <TOKENS_B> [DATA_FILES]
source "$(dirname "$0")/env.sh"
MODEL_CONFIG="${1:?model config required}"
TOKENS_B="${2:?token budget (billions) required}"
DATA_FILES="${3:-$DATA_FILES}"
STEPS=$(steps_for_tokens "$TOKENS_B")

cd "$LIB_DIR/training/maxtext"
bash scripts/training.sh \
    --model="$MODEL_CONFIG" \
    --num_steps="$STEPS" \
    --lr="$PEAK_LR" \
    --global_batch_size="$GLOBAL_BATCH_SIZE" \
    --data_files="$DATA_FILES"
