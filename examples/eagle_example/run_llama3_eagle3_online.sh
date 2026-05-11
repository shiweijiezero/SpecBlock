#!/bin/bash

# ============================================================================
# Llama3.1-8B-Instruct EAGLE3 online training pipeline
# Custom offline data regeneration + the official v0.2.0 training script
# ============================================================================

uv pip install -e .
uv pip install -e "./sglang-main/python[all]"

echo "============================================"
echo "Llama3.1-8B-Instruct EAGLE3 online training pipeline"
echo "============================================"

# ---------- Stage 1: Download model + datasets ----------
echo ""
echo "Stage 1: Downloading model + datasets..."
hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# ---------- Stage 2: Prepare raw training data ----------
echo ""
echo "Stage 2: Preparing raw training data..."
mkdir -p ./cache/dataset
python scripts/prepare_data.py --dataset ultrachat --output-path ./cache/dataset
python scripts/prepare_data.py --dataset ultrachat_gen --output-path ./cache/dataset
python scripts/prepare_data.py --dataset sharegpt --output-path ./cache/dataset
cat ./cache/dataset/sharegpt_train.jsonl ./cache/dataset/ultrachat_train.jsonl ./cache/dataset/ultrachat_gen_train.jsonl > ./cache/dataset/train_dataset.jsonl
echo "Data preparation done: $(wc -l < ./cache/dataset/train_dataset.jsonl) rows (sharegpt ~68K + ultrachat ~207K + ultrachat_gen ~256K ≈ 531K)"

# ---------- Stage 3: Re-generate answers with the target model ----------
echo ""
echo "Stage 3: Regenerating answers with the target model..."
bash scripts/run_offline_generate.sh \
    --model llama \
    --mode offline \
    --gpus 0,1,2,3 \
    --input ./cache/dataset/train_dataset.jsonl \
    --output ./cache/offline-generated/llama-3.1-8b-instruct

# Merge shards and split into train/eval
OUTPUT_DIR=./cache/offline-generated/llama-3.1-8b-instruct
cat ${OUTPUT_DIR}/shard_*.jsonl > ${OUTPUT_DIR}/final_output.jsonl
echo "Merge done: $(wc -l < ${OUTPUT_DIR}/final_output.jsonl) rows"

# Split train/eval (eval fixed at 1500 rows, randomly shuffled)
shuf ./cache/offline-generated/llama-3.1-8b-instruct/final_output.jsonl > ./cache/offline-generated/llama-3.1-8b-instruct/shuffled.jsonl
tail -n 1500 ./cache/offline-generated/llama-3.1-8b-instruct/shuffled.jsonl > ./cache/offline-generated/llama-3.1-8b-instruct/eval.jsonl
head -n -1500 ./cache/offline-generated/llama-3.1-8b-instruct/shuffled.jsonl > ./cache/offline-generated/llama-3.1-8b-instruct/train.jsonl
rm ./cache/offline-generated/llama-3.1-8b-instruct/shuffled.jsonl
echo "Split done: train=$(wc -l < ./cache/offline-generated/llama-3.1-8b-instruct/train.jsonl) eval=$(wc -l < ./cache/offline-generated/llama-3.1-8b-instruct/eval.jsonl)"

# ---------- Stage 4: EAGLE3 online training ----------
echo ""
echo "Stage 4: Starting EAGLE3 training..."

mkdir -p ./outputs/llama3-8b-eagle3
cd /path/to/specblock && \
source .venv/bin/activate && \
mkdir -p logs && \
# Effective batch size = 4*2 = 8
# Effective warmup-steps = 1200*2 = 2400
HF_HOME=/path/to/hf_cache HUGGINGFACE_HUB_CACHE=/path/to/hf_cache CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config ./configs/eagle3/llama3-8B-eagle3.json \
  --train-data-path ./cache/offline-generated/llama-3.1-8b-instruct/train.jsonl \
  --eval-data-path ./cache/offline-generated/llama-3.1-8b-instruct/eval.jsonl \
  --build-dataset-num-proc 96 \
  --tp-size 1 \
  --sp-ulysses-size 1 \
  --output-dir ./model/Llama-3.1-8B-Instruct/eagle3_official \
  --num-epochs 20 \
  --batch-size 6 \
  --draft-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --warmup-steps 1200 \
  --scheduler-type linear \
  --attention-backend flex_attention \
  --max-length 1024 \
  --chat-template llama3 \
  --cache-dir ./cache \
  --dist-timeout 1000 \
  --target-model-backend sglang \
  --sglang-mem-fraction-static 0.4 \
  --ttt-length 7 \
  --log-interval 50 \
  --save-interval 2500 \
  --eval-interval 2500 \
  --report-to wandb \
  --wandb-project eagle3_official \
  --wandb-name llama3.1-8b-eagle3

echo ""
echo "============================================"
echo "Training done!"
echo "Model output dir: ./outputs/llama3-8b-eagle3"
echo "============================================"

# ---------- Stage 5: SGLang benchmark (optional) ----------
# config_list=(
#     "1,0,0,0"
#     "1,3,1,4"
#     "1,7,10,60"
# )
#
# SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_model_speedup.py \
#     --model-path meta-llama/Llama-3.1-8B-Instruct \
#     --speculative-draft-model-path ./outputs/llama3-8b-eagle3/epoch_0_step_12500 \
#     --port 30000 \
#     --enable-multi-turn-conversation \
#     --trust-remote-code \
#     --mem-fraction-static 0.7 \
#     --tp-size 1 \
#     --config-list "${config_list[@]}" \
#     --benchmark-list mtbench:80 gsm8k:200 humaneval:164 \
#     --temperature 0.0 1.0 \
#     --output ./results/eagle3_llama3_results.jsonl

# ---------- Stage 6: HuggingFace evaluation (optional) ----------
hf_config_list=(
 "1,7,10,60"
)
CUDA_VISIBLE_DEVICES=2 uv run benchmarks_hf/run_eval.py \
 --algorithm eagle3 \
 --model-path meta-llama/Llama-3.1-8B-Instruct \
 --draft-model-path model/Llama-3.1-8B-Instruct/eagle3_official/epoch_1_step_15000 \
 --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
 --max-new-tokens 1024 \
 --config-list "${hf_config_list[@]}" \
 --temperature 0.0 1.0 \
 --output ./hf_results/hf_eagle3_llama3_epoch_1_v2.jsonl

# Evaluate the official checkpoint
hf_config_list=(
 "1,7,10,60"
)
CUDA_VISIBLE_DEVICES=2 uv run benchmarks_hf/run_eval.py \
 --algorithm eagle3 \
 --model-path meta-llama/Llama-3.1-8B-Instruct \
 --draft-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
 --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
 --max-new-tokens 1024 \
 --config-list "${hf_config_list[@]}" \
 --temperature 0.0 1.0 \
 --output ./hf_results/hf_eagle3_llama3_official.jsonl

# Baseline evaluation (no speculative decoding)
CUDA_VISIBLE_DEVICES=5 uv run benchmarks_hf/run_eval.py \
 --algorithm baseline \
 --model-path meta-llama/Llama-3.1-8B-Instruct \
 --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
 --max-new-tokens 1024 \
 --temperature 0.0 1.0 \
 --output ./hf_results/hf_baseline_llama3.jsonl

# Evaluate via the upstream EAGLE repo scripts

# Clone https://github.com/SafeAILab/EAGLE separately, then:
# CUDA_VISIBLE_DEVICES=5 uv run /path/to/EAGLE/eagle/evaluation/gen_ea_answer_llama3chat.py \
#     --base-model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --ea-model-path   yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
#     --use_eagle3 --bench-name mt_bench \
#     --total-token 60 --depth 7 --top-k 10 --temperature 0.0 \
#     --model-id eagle3-llama31-8b \
#     --answer-file results/eagle3_official_mtbench.jsonl
