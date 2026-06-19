# Scripts

These scripts reproduce the paper's experiments by driving the companion library,
**[llm-pruning-collection](https://github.com/zlab-princeton/llm-pruning-collection)**.
All training and evaluation here target **Google Cloud TPU** (via MaxText), which
is where every experiment in the paper was run.

```
scripts/
└── run/                      # TPU training + evaluation (wrap the library)
    ├── env.sh                # paths, infra, training recipe, steps_for_tokens()
    ├── pretrain_parent.sh    # Stage 1 — pretrain Llama-3.1-8B on 200B DCLM (P200)
    ├── retrain_pruned.sh     # Stage 3 — retrain a pruned model (P200-R_N)
    ├── train_from_scratch.sh # Stage 3 — random-init baselines (S_N / S250)
    └── eval.sh               # Stage 4 — eval a run (accelerated lm-eval-harness)
```

## Prerequisites

Assumes a **Google Cloud TPU VM** (experiments used TPU v4-256) with a **GCS
bucket** for data and checkpoints (`BUCKET_NAME`).

- A local clone of the library, and your infra values set in [`run/env.sh`](run/env.sh)
  (`LIB_DIR`, `BUCKET_NAME`, `DATA_FILES`):
  ```bash
  git clone https://github.com/zlab-princeton/llm-pruning-collection ~/llm-pruning-collection
  export LIB_DIR=~/llm-pruning-collection
  # edit scripts/run/env.sh (or export LIB_DIR / BUCKET_NAME / DATA_FILES)
  ```
- The pre-tokenized, shuffled DCLM dataset
  ([`Zephyr271828/dclm-llama3-64-shuffled`](https://huggingface.co/datasets/Zephyr271828/dclm-llama3-64-shuffled)),
  downloaded **into your bucket** (once):
  ```bash
  huggingface-cli download Zephyr271828/dclm-llama3-64-shuffled \
      --repo-type dataset --local-dir /tmp/dclm-llama3-64-shuffled
  gsutil -m cp -r /tmp/dclm-llama3-64-shuffled gs://$BUCKET_NAME/datasets/
  # DATA_FILES = gs://$BUCKET_NAME/datasets/dclm-llama3-64-shuffled/*.array_record
  ```
  One shuffled corpus; the disjoint pretrain (200B) / retrain (50B) splits are
  different file-index ranges (the library sets `start_from_file_index`).

Examples below use **Minitron-width @ 50%** (`llama3.1-4b-width`).

## Stage 1 — Pretrain the parent (P200)

```bash
bash scripts/run/pretrain_parent.sh   # scripts/training.sh --model=llama3.1-8b --num_steps=50000

# Convert parent Orbax checkpoint -> HuggingFace (so the pruning code can load it):
cd $LIB_DIR/training/maxtext
# 1) parameter-only (unrolled) checkpoint from the full training checkpoint
bash scripts/convert.sh gen_param_ckpt --model=llama3.1-8b \
    --orbax_ckpt_name=<parent_run> --step=49999 --direct_run_name=<parent_direct>
# 2) convert it to HuggingFace
bash scripts/convert.sh orbax_to_hf --model=llama3.1-8b \
    --orbax_ckpt_name=<parent_run> --step=49999 --hf_model_name=Llama-3.1-8B
```

## Stage 2 — Prune (library)

Run pruning from the method's directory in the library (each has its own README).
Methods operate on the **HuggingFace** parent and emit a pruned HF checkpoint:

```bash
cd $LIB_DIR/pruning/minitron && bash scripts/install.sh && bash scripts/prune_llama3.1-8b.sh
```

Then convert the pruned HF checkpoint **back to MaxText (Orbax)** for TPU retraining
(target shapes per ratio follow the paper, Tables `tab:nas-*`):

```bash
cd $LIB_DIR/training/maxtext
bash scripts/convert.sh hf_to_orbax --model=llama3.1-4b-width \
    --orbax_ckpt_name=minitron_width_50pct --hf_model_name=<pruned_hf_ckpt>
```

## Stage 3 — Train for comparison (P200-R50 vs. S250)

```bash
# P200-R50 — retrain the pruned init on 50B tokens (loads the converted pruned ckpt)
bash scripts/run/retrain_pruned.sh llama3.1-4b-width \
    gs://$BUCKET_NAME/model_ckpts/maxtext/minitron_width_50pct/0/items 50

# S250 — train the same architecture from scratch on 250B tokens (random init)
bash scripts/run/train_from_scratch.sh llama3.1-4b-width 250
```

`retrain_pruned.sh` wraps `scripts/finetuning.sh` (with `--load_parameters_path`);
`train_from_scratch.sh` wraps `scripts/training.sh` (random init).

## Stage 4 — Evaluate

Both Stage-3 scripts run eval automatically after training. To (re-)evaluate a run
explicitly — reporting the eight zero-shot tasks plus C4 / WikiText / CNN-DailyMail
perplexity. Eval reads a parameter-only "direct" checkpoint, so generate one first
(`gen_param_ckpt`), then eval; the wrapper does both:

```bash
bash scripts/run/eval.sh llama3.1-4b-width <run_name> <step>   # gen_param_ckpt + eval
# == cd $LIB_DIR/training/maxtext
#    bash scripts/convert.sh gen_param_ckpt --model=llama3.1-4b-width \
#        --orbax_ckpt_name=<run_name> --step=<step> --direct_run_name=<run_name>_direct
#    bash scripts/convert.sh eval --model=llama3.1-4b-width \
#        --direct_run_name=<run_name>_direct --hf_model_name=Llama-3.1-8B
```
