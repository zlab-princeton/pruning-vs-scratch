#!/bin/bash
# Evaluate a trained run on TPU through MaxText (where all our experiments ran).
# NOTE: the training scripts (pretrain/retrain/scratch) already run eval
# automatically after training. Use this only to (re)evaluate an existing run.
#
# Eval reads a parameter-only "direct" checkpoint, so this first generates one
# (gen_param_ckpt) from the trained Orbax run, then evaluates it. Reports zero-shot
# accuracy on the eight downstream tasks (WinoGrande, ARC-C, ARC-E, HellaSwag,
# PIQA, SciQ, BoolQ, OBQA) and perplexity on C4 / WikiText / CNN-DailyMail.
#
# Usage: bash scripts/run/eval.sh <MODEL_CONFIG> <ORBAX_RUN> <STEP>
#   MODEL_CONFIG : MaxText config of the model being evaluated
#   ORBAX_RUN    : the trained Orbax run name
#   STEP         : checkpoint step to evaluate (e.g. 12499 for a 50B run)
source "$(dirname "$0")/env.sh"
MODEL_CONFIG="${1:?model config required}"
ORBAX_RUN="${2:?orbax run name required}"
STEP="${3:?checkpoint step required}"
DIRECT_RUN="${ORBAX_RUN}_direct"

cd "$LIB_DIR/training/maxtext"
# 1) parameter-only (unrolled) checkpoint from the trained run
bash scripts/convert.sh gen_param_ckpt \
    --model="$MODEL_CONFIG" \
    --orbax_ckpt_name="$ORBAX_RUN" \
    --step="$STEP" \
    --direct_run_name="$DIRECT_RUN"
# 2) evaluate it
bash scripts/convert.sh eval \
    --model="$MODEL_CONFIG" \
    --direct_run_name="$DIRECT_RUN" \
    --hf_model_name=Llama-3.1-8B
