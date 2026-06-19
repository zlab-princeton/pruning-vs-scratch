#!/bin/bash
# Shared configuration for the reproduction scripts.
#
# These scripts are thin wrappers that drive the companion library
# (llm-pruning-collection) for the exact experiments in the paper. They run on
# Google Cloud TPU VM infrastructure (training is done with MaxText). Edit the
# values below, or export them in your shell before sourcing this file.
#
#   source scripts/run/env.sh
set -euo pipefail

# ---- Paths / infra (EDIT THESE) ----
# Local clone of https://github.com/zlab-princeton/llm-pruning-collection
export LIB_DIR="${LIB_DIR:-$HOME/llm-pruning-collection}"
# GCS bucket holding checkpoints
export BUCKET_NAME="${BUCKET_NAME:-your-gcs-bucket}"
# Pre-tokenized, shuffled DCLM shards (array_record, llama3 tokenizer), stored in
# your GCS bucket. We assume a TPU VM + GCS bucket; download the whole dataset
# into the bucket once:
#   huggingface-cli download Zephyr271828/dclm-llama3-64-shuffled \
#       --repo-type dataset --local-dir /tmp/dclm-llama3-64-shuffled
#   gsutil -m cp -r /tmp/dclm-llama3-64-shuffled gs://$BUCKET_NAME/datasets/
# One shuffled corpus; the disjoint pretrain (200B) / retrain (50B) splits are
# different file-index ranges (the library sets start_from_file_index, e.g. 0 for
# pretrain/scratch, 50 for retrain — see its scripts/examples/*.sh).
export DATA_FILES="${DATA_FILES:-gs://$BUCKET_NAME/datasets/dclm-llama3-64-shuffled/*.array_record}"

# ---- Fixed training recipe (paper; following Lingua) ----
export GLOBAL_BATCH_SIZE=512   # sequences
export SEQ_LEN=8192            # tokens
export PEAK_LR=3e-4            # swept over {1e-5,3e-5,1e-4,3e-4,1e-3}; 3e-4 default

# The library counts ~4M tokens/step (global_batch_size 512 × seq_len 8192).
# steps_for_tokens <tokens_in_billions> -> integer step count, matching the
# step budgets used in the paper's runs:
#   10B->2500  30B->7500  50B->12500  200B->50000  250B->62500  500B->125000
steps_for_tokens() {
  python3 -c "import sys; print(round(float(sys.argv[1])*1e9/4_000_000))" "$1"
}
