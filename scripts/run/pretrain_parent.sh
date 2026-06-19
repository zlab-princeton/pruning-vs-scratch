#!/bin/bash
# Pretrain the parent (P200): train Llama-3.1-8B from scratch on 200B disjoint
# DCLM tokens. The resulting checkpoint is the model we later prune (pruning is
# done with the library; see the README "Pipeline" section).
#
# Usage: bash scripts/run/pretrain_parent.sh
source "$(dirname "$0")/env.sh"

STEPS=$(steps_for_tokens 200)   # 50000 steps

cd "$LIB_DIR/training/maxtext"
bash scripts/training.sh \
    --model=llama3.1-8b \
    --num_steps="$STEPS" \
    --lr="$PEAK_LR" \
    --global_batch_size="$GLOBAL_BATCH_SIZE" \
    --data_files="$DATA_FILES"

# Convert the parent Orbax checkpoint -> HuggingFace so the pruning methods can
# load it (run from $LIB_DIR/training/maxtext):
#   bash scripts/convert.sh orbax_to_hf --model=llama3.1-8b \
#       --orbax_ckpt_name=<parent_run> --step=49999 --hf_model_name=Llama-3.1-8B
