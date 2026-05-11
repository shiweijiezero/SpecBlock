#!/bin/bash

# ============================================================================
# Qwen3-8B SpecBlock online training pipeline
# Same data prep flow; training runs through the SpecBlock training script
# ============================================================================

uv pip install -e .
uv pip install -e "./sglang-main/python[all]"

echo "================================================"
echo "Qwen3-8B SpecBlock online training pipeline"
echo "================================================"

# ---------- Stage 1: Download model + datasets ----------
echo ""
echo "Stage 1: Downloading model + datasets..."
hf download Qwen/Qwen3-8B
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# ---------- Stage 2: Prepare raw training data ----------
echo ""
echo "Stage 2: Preparing raw training data..."
mkdir -p ./cache/dataset
python scripts/prepare_data.py --dataset ultrachat --output-path ./cache/dataset
python scripts/prepare_data.py --dataset sharegpt --output-path ./cache/dataset
cat ./cache/dataset/sharegpt.jsonl ./cache/dataset/ultrachat.jsonl > ./cache/dataset/train_dataset.jsonl

# ---------- Stage 3: Re-generate answers with the target model ----------
echo ""
echo "Stage 3: Regenerating answers with the target model..."
bash scripts/run_offline_generate.sh \
    --model qwen3 \
    --mode offline \
    --gpus 0,1,2,3 \
    --input ./cache/dataset/train_dataset.jsonl \
    --output ./cache/offline-generated/qwen3-8b-instruct

# Merge shards and split into train/eval
OUTPUT_DIR=./cache/offline-generated/qwen3-8b-instruct
cat ${OUTPUT_DIR}/shard_*.jsonl > ${OUTPUT_DIR}/final_output.jsonl
echo "Merge done: $(wc -l < ${OUTPUT_DIR}/final_output.jsonl) rows"

shuf ${OUTPUT_DIR}/final_output.jsonl > ${OUTPUT_DIR}/shuffled.jsonl
tail -n 1500 ${OUTPUT_DIR}/shuffled.jsonl > ${OUTPUT_DIR}/eval.jsonl
head -n -1500 ${OUTPUT_DIR}/shuffled.jsonl > ${OUTPUT_DIR}/train.jsonl
rm ${OUTPUT_DIR}/shuffled.jsonl
echo "Split done: train=$(wc -l < ${OUTPUT_DIR}/train.jsonl) eval=$(wc -l < ${OUTPUT_DIR}/eval.jsonl)"

# ---------- Stage 4: SpecBlock training ----------
echo ""
echo "Stage 4: Starting SpecBlock training..."

# === Single-node training (6 GPUs) ===
CUDA_VISIBLE_DEVICES=0,2,3,4,5,7 uv run torchrun \
    --standalone \
    --nproc_per_node 6 \
    scripts/train_specblock.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config ./configs/specblock/qwen3-8B-specblock.json \
    --train-data-path ./cache/offline-generated/qwen3-8b-instruct/train.jsonl \
    --eval-data-path ./cache/offline-generated/qwen3-8b-instruct/eval.jsonl \
    --build-dataset-num-proc 96 \
    --tp-size 1 \
    --sp-ulysses-size 1 \
    --output-dir ./model/Qwen3-8B/specblock-layer2-seqpos \
    --num-epochs 20 \
    --batch-size 2 \
    --draft-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --warmup-steps 500 \
    --scheduler-type linear \
    --attention-backend flex_attention \
    --max-length 1024 \
    --chat-template qwen \
    --cache-dir ./cache \
    --dist-timeout 1000 \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.3 \
    --log-interval 50 \
    --save-interval 2500 \
    --eval-interval 2500 \
    --report-to wandb \
    --wandb-project specblock \
    --wandb-name qwen3-8b-specblock-layer2-seqpos \
    --position-loss-weight 0.8 \
    --rank-start-step 0 \
    --num-ttt-blocks 3 \
    --draft-token-num 4 \
    --num-layers 2

# ---------- Stage 5: SpecBlock inference evaluation ----------
echo ""
echo "Stage 5: Running SpecBlock evaluation..."
# config-list format: "batch,steps(max TTT block iterations),topk(max branch factor per rank),draft_tokens(tree budget)"

# Path to the trained SpecBlock draft. Either point at your own checkpoint folder
# (e.g. ./model/Qwen3-8B/specblock-layer2/<your_checkpoint>), or
# download our release weights first:
#   huggingface-cli download weijiezz/SpecBlock-Qwen3-8B --local-dir ./model/specblock-qwen3
DRAFT_CKPT=./model/specblock-qwen3

# Eval (greedy)
CUDA_VISIBLE_DEVICES=1 uv run benchmarks_hf/run_eval.py \
    --algorithm specblock \
    --model-path Qwen/Qwen3-8B \
    --draft-model-path ${DRAFT_CKPT} \
    --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
    --max-new-tokens 1024 \
    --config-list "1,2,10,90" \
    --temperature 0.0 \
    --output ./hf_results/specblock_qwen3_t0.jsonl

# Eval (sampling, temperature=1.0)
uv run benchmarks_hf/run_eval.py \
    --algorithm specblock \
    --model-path Qwen/Qwen3-8B \
    --draft-model-path ${DRAFT_CKPT} \
    --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
    --max-new-tokens 1024 \
    --config-list "1,2,10,90" \
    --temperature 1.0 \
    --output ./hf_results/specblock_qwen3_t1.jsonl

# ---------- Stage 6: Interactive Gradio UI ----------
echo ""
echo "Stage 6: Launching SpecBlock UI..."
CUDA_VISIBLE_DEVICES=1 uv run benchmarks_hf/ui/app.py
