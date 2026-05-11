#!/bin/bash

# ============================================================================
# Llama3.1-8B-Instruct SpecBlock multi-node training
# node0: GPUs 4,5,6,7 - master
# node1: GPUs 0,1,2,3,4,5
# 10 GPUs total, 2 draft layers
# ============================================================================

COMMON_ARGS="\
    scripts/train_specblock.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/specblock/llama3-8B-specblock.json \
    --train-data-path ./cache/offline-generated/llama-3.1-8b-instruct/train.jsonl \
    --eval-data-path ./cache/offline-generated/llama-3.1-8b-instruct/eval.jsonl \
    --build-dataset-num-proc 96 \
    --tp-size 1 \
    --sp-ulysses-size 1 \
    --output-dir ./model/Llama-3.1-8B-Instruct/specblock-layer2-allslot \
    --num-epochs 20 \
    --batch-size 3 \
    --draft-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --warmup-steps 500 \
    --scheduler-type linear \
    --attention-backend flex_attention \
    --max-length 1024 \
    --chat-template llama3 \
    --cache-dir ./cache \
    --dist-timeout 3600 \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.25 \
    --log-interval 50 \
    --save-interval 2500 \
    --eval-interval 2500 \
    --report-to wandb \
    --wandb-project specblock \
    --wandb-name llama3.1-8b-specblock-layer2-allslot-topk50 \
    --position-loss-weight 0.8 \
    --rank-start-step 0 \
    --num-ttt-blocks 3 \
    --draft-token-num 4 \
    --num-layers 2 \
    --draft-loss-topk 50 \
    --resume"

RDZV_ARGS="--rdzv-backend c10d --rdzv-endpoint node0:29500 --rdzv-conf timeout=1800"

# === node1 (worker, launch first) ===
# ssh node1
# cd /path/to/specblock && source .venv/bin/activate
# export HF_HOME=/path/to/hf_cache HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 uv run torchrun \
    --nproc_per_node 7 --nnodes 3 --node-rank 1 \
    ${RDZV_ARGS} \
    ${COMMON_ARGS}

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run torchrun \
    --nproc_per_node 6 --nnodes 3 --node-rank 2 \
    ${RDZV_ARGS} \
    ${COMMON_ARGS}

# === node0 (master, launch after) ===
# ssh node0
# cd /path/to/specblock && source .venv/bin/activate
# export HF_HOME=/path/to/hf_cache HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun \
    --nproc_per_node 4 --nnodes 3 --node-rank 0 \
    ${RDZV_ARGS} \
    ${COMMON_ARGS}
