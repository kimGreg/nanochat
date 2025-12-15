#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

WANDB_RUN="d8-1"

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)



python -m scripts.base_train --depth=8 --max_seq_len=1024 \
    --device_batch_size=1 --eval_tokens=1024 --core_metric_every=-1 \
    --total_batch_size=1024 --num_iterations=-1 --run=$WANDB_RUN

# # Number of processes/GPUs to use
# NPROC_PER_NODE=1
# 
# # pretrain the d8 model
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=8 --run=$WANDB_RUN
# # evaluate the model on a larger chunk of train/val data and draw some samples
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# # evaluate the model on CORE tasks
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval