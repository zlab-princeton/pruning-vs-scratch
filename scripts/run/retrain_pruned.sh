#!/bin/bash
# Retrain a pruned model (P200-R_N): continue training the pruned initialization
# on N billion DCLM tokens, taken from a file-index range disjoint from the
# pretraining slice (the library sets start_from_file_index; see its examples).
# N=50 is the main setting; {10,30,50,250,500} form the token-scaling study.
#
# Usage: bash scripts/run/retrain_pruned.sh <MODEL_CONFIG> <PRUNED_CKPT> [TOKENS_B]
#   MODEL_CONFIG : MaxText config of the pruned architecture
#                  (e.g. llama3.1-4b-width, llama3.1-4b-depth, llama3.1-1.5b-depth)
#   PRUNED_CKPT  : gs:// path to the pruned init checkpoint (from the library)
#   TOKENS_B     : retraining token budget in billions (default 50)
source "$(dirname "$0")/env.sh"
MODEL_CONFIG="${1:?model config required}"
PRUNED_CKPT="${2:?pruned checkpoint path required}"
TOKENS_B="${3:-50}"
STEPS=$(steps_for_tokens "$TOKENS_B")

cd "$LIB_DIR/training/maxtext"
bash scripts/finetuning.sh \
    --model="$MODEL_CONFIG" \
    --num_steps="$STEPS" \
    --lr="$PEAK_LR" \
    --global_batch_size="$GLOBAL_BATCH_SIZE" \
    --data_files="$DATA_FILES" \
    --load_parameters_path="$PRUNED_CKPT"
