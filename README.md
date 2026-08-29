# SpecBlock

SpecBlock is a speculative decoding method that uses multi-block test-time training with cross-slot hidden injection between decoder layers. Compared to EAGLE3, SpecBlock achieves higher acceptance length on math, code, and reasoning benchmarks under the same draft-token budget.

This repository contains:
- The full training framework (`specforge/`) for both SpecBlock (our method) and EAGLE3 (baseline).
- A HuggingFace-style inference + evaluation harness (`benchmarks_hf/`) with a Gradio UI.
- A vendored SGLang fork (`sglang-main/`) that integrates the SpecBlock speculative worker for production-grade inference.

## Installation

```bash
pip install -e .
pip install -e sglang-main/python  # for SGLang-backed inference / training backend
```

CUDA 12.x, PyTorch ≥ 2.4, and an Ampere or newer GPU are recommended. See `requirements.txt` / `requirements-rocm.txt` for the full dependency list.

## Quick start

The full pipeline lives in the example scripts (`examples/specblock/run_llama3_specblock_online.sh` for Llama, `run_qwen3_specblock_online.sh` for Qwen3, `examples/eagle_example/run_llama3_eagle3_online.sh` for the EAGLE3 baseline). Each script is organised into 6 stages, marked by `# ---------- Stage N: ----------` headers. **Run the stages individually** — copy the block you need, edit paths/GPUs, and execute. The scripts are not designed to run end-to-end with a single `bash …`.

### Stage 1 — Install + download target model and raw data

```bash
uv pip install -e .
uv pip install -e "./sglang-main/python[all]"

hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset
```

### Stage 2 — Convert raw data → jsonl

```bash
mkdir -p ./cache/dataset
python scripts/prepare_data.py --dataset ultrachat --output-path ./cache/dataset
python scripts/prepare_data.py --dataset sharegpt  --output-path ./cache/dataset
cat ./cache/dataset/sharegpt.jsonl ./cache/dataset/ultrachat.jsonl \
    > ./cache/dataset/train_dataset.jsonl
```

### Stage 3 — Re-generate target answers (offline distillation), then split

```bash
bash scripts/run_offline_generate.sh \
    --model llama --mode offline --gpus 0,1,2,3 \
    --input  ./cache/dataset/train_dataset.jsonl \
    --output ./cache/offline-generated/llama-3.1-8b-instruct

OUTPUT_DIR=./cache/offline-generated/llama-3.1-8b-instruct
cat ${OUTPUT_DIR}/shard_*.jsonl > ${OUTPUT_DIR}/final_output.jsonl
shuf ${OUTPUT_DIR}/final_output.jsonl > ${OUTPUT_DIR}/shuffled.jsonl
tail -n 1500    ${OUTPUT_DIR}/shuffled.jsonl > ${OUTPUT_DIR}/eval.jsonl
head -n -1500   ${OUTPUT_DIR}/shuffled.jsonl > ${OUTPUT_DIR}/train.jsonl
rm ${OUTPUT_DIR}/shuffled.jsonl
```

Total cost ≈ 6h on 8×A100 for ShareGPT (~90k samples).

**Shortcut**: skip stages 1–3 by downloading our pre-processed jsonl. See [Datasets](#datasets) below.

### Stage 4 — Train the draft model

Copy the `# ---------- Stage 4: ----------` block from the example script, adjust `CUDA_VISIBLE_DEVICES` / `--nproc_per_node` / paths, then run. Example:

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,7 uv run torchrun --standalone --nproc_per_node 6 \
    scripts/train_specblock.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/specblock/llama3-8B-specblock.json \
    --train-data-path   ./cache/offline-generated/llama-3.1-8b-instruct/train.jsonl \
    --eval-data-path    ./cache/offline-generated/llama-3.1-8b-instruct/eval.jsonl \
    --output-dir ./model/Llama-3.1-8B-Instruct/specblock-layer2 \
    --num-epochs 20 --batch-size 2 --learning-rate 5e-5 \
    --attention-backend flex_attention --chat-template llama3 \
    --target-model-backend sglang --sglang-mem-fraction-static 0.3 \
    --num-ttt-blocks 3 --draft-token-num 4 --num-layers 2 \
    --rank-start-step 0 --position-loss-weight 0.8
```

The corresponding Qwen3 script uses `--chat-template qwen` and the Qwen-specific config. Multi-node training: see `examples/specblock/run_specblock_3node.sh`. Pre-trained checkpoints are listed under [Model checkpoints](#model-checkpoints) below.

### Stage 5 — Evaluate (HF backend)

```bash
python benchmarks_hf/run_eval.py \
    --algorithm specblock \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-path ./model/Llama-3.1-8B-Instruct/specblock-layer2 \
    --benchmark-list humaneval:164 math500:500 alpaca:200 nq_open:200 mtbench:80 wmt23:200 \
    --max-new-tokens 1024 --config-list "1,2,10,90" --temperature 0.0 \
    --output ./hf_results/specblock_llama.jsonl

python benchmarks_hf/analyze_results.py ./hf_results/specblock_llama.jsonl
```

Supported `--algorithm` values: `baseline`, `eagle3`, `specblock`. The benchmark list above matches the paper (HumanEval / MATH-500 / Alpaca / NQ / MT-Bench / WMT-23). The vanilla baseline defaults to Hugging Face eager attention to reproduce the paper protocol; set `BASELINE_ATTN_IMPL=sdpa` to use the faster modern SDPA baseline instead.

#### MetaX C500

The HF evaluator detects MetaX PyTorch builds automatically. Set the platform explicitly when the runtime does not identify itself in `torch.__version__`:

```bash
export SPECBLOCK_PLATFORM=metax
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
```

An ABI-compatible `sgl_kernel.so` built from MetaX `mcoplib` enables fused SiLU, residual RMSNorm, and RoPE:

```bash
export SPECBLOCK_SGL_KERNEL_LIBRARY=/path/to/sgl_kernel.so
```

The library is optional; SpecBlock falls back to native PyTorch operators when it is not configured. MetaX defaults to `DRAFT_COMPILE=2`; set `DRAFT_COMPILE=0 SPECBLOCK_INTERNAL_PREWARM=0` for the faster eager path on the current C500 stack. An experimental B=1 hybrid keeps batched target verification while using the compiled scalar draft path:

```bash
export SPECBLOCK_HYBRID_B1=1
```

The hybrid path currently targets greedy evaluation only and remains opt-in because compiled and eager draft numerics can produce different generation trajectories near target argmax boundaries.

### Stage 6 — Interactive UI

```bash
python benchmarks_hf/run_ui.py --host 127.0.0.1 --port 7860
```

Gradio side-by-side comparison of baseline / EAGLE3 / SpecBlock with live token streaming and accept-length visualization.

## Repository layout

```
specforge/                  # Training framework
├── core/                   # Trainers (eagle3, specblock)
├── data/                   # Data loading & preprocessing
├── modeling/draft/         # Draft model architectures
├── modeling/target/        # Target model backends (sglang, custom, hf)
└── layers/                 # Custom layers (embedding, linear, lm_head)

scripts/                    # Training & data-prep entry points
benchmarks_hf/              # HF-style inference & evaluation
├── algorithms/             # Algorithm wrappers (baseline, eagle3, specblock)
├── ui/                     # Gradio UI
└── run_eval.py             # Evaluation CLI

sglang-main/                # Vendored SGLang fork with specblock worker
examples/                   # End-to-end training launch scripts
```

## Model checkpoints

Pre-trained draft model weights compatible with the inference paths above.

| Target model | Draft method | Weights |
|---|---|---|
| Llama-3.1-8B-Instruct | SpecBlock (ours) | [weijiezz/SpecBlock-Llama-3.1-8B-Instruct](https://huggingface.co/weijiezz/SpecBlock-Llama-3.1-8B-Instruct) |
| Llama-3.1-8B-Instruct | EAGLE3 (baseline) | [weijiezz/EAGLE3-Llama-3.1-8B-Instruct](https://huggingface.co/weijiezz/EAGLE3-Llama-3.1-8B-Instruct) |
| Qwen3-8B | SpecBlock (ours) | [weijiezz/SpecBlock-Qwen3-8B](https://huggingface.co/weijiezz/SpecBlock-Qwen3-8B) |

Each draft model directory contains `config.json`, `model.safetensors`, and (for SpecBlock) `vocab_mapping.pt`. **SpecBlock drafts** are 2-layer transformers with cross-slot hidden injection (`shift_proj`) between decoder layers, trained with multi-block test-time training to produce dynamic-tree draft proposals. The **EAGLE3 baseline** is a single-decoder autoregressive drafter trained with the official EAGLE3 recipe and ships without `vocab_mapping.pt` (uses full vocab). All draft weights here are distilled against the listed target model with greedy-decoded ShareGPT + UltraChat answers. See `scripts/upload_drafts.py` for the upload pipeline.

## Datasets

Pre-processed training data (already passed through `prepare_data.py` and the offline target-model regeneration step).

| Dataset | Target model used for distillation | HuggingFace | ModelScope |
|---|---|---|---|
| SpecBlock-train-data-llama | Llama-3.1-8B-Instruct | [weijiezz/SpecBlock-train-data-llama](https://huggingface.co/datasets/weijiezz/SpecBlock-train-data-llama) | [JasonHaggard/SpecBlock-train-data-llama](https://modelscope.cn/datasets/JasonHaggard/SpecBlock-train-data-llama) |
| SpecBlock-train-data-qwen | Qwen3-8B | [weijiezz/SpecBlock-train-data-qwen](https://huggingface.co/datasets/weijiezz/SpecBlock-train-data-qwen) | [JasonHaggard/SpecBlock-train-data-qwen](https://modelscope.cn/datasets/JasonHaggard/SpecBlock-train-data-qwen) |

See `scripts/upload_data.py` for the upload pipeline.

## Acknowledgments

This codebase is built on top of [SpecForge](https://github.com/sgl-project/SpecForge), [EAGLE](https://github.com/SafeAILab/EAGLE), and [SGLang](https://github.com/sgl-project/sglang).
