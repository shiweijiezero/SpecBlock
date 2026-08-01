# benchmarks_hf

HuggingFace-based evaluation harness for speculative decoding. Lighter and more flexible than running through SGLang.

## Supported algorithms

- **baseline** — standard autoregressive decoding (no speculation)
- **eagle3** — autoregressive tree speculative decoding
- **specblock** — multi-block test-time training with cross-slot hidden injection (ours)

## Supported datasets

| Dataset | Default samples | Description |
|---|---|---|
| `mtbench` | 80 | Multi-turn dialogue quality |
| `gsm8k` | 200 | Math reasoning |
| `humaneval` | 164 | Code generation |
| `math500` | 500 | Math problems |
| `alpaca` | 200 | General instruction following |
| `wmt23` | 200 | Translation |
| `nq_qa` / `nq_rag` | 200 | Open-domain QA |
| `ceval` / `cmmlu` | varies | Chinese benchmarks |

## Quick start

```bash
# Baseline (no speculation)
python benchmarks_hf/run_eval.py \
    --algorithm baseline \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --benchmark-list mtbench:80 \
    --output ./hf_results/baseline_mtbench.jsonl

# EAGLE3
python benchmarks_hf/run_eval.py \
    --algorithm eagle3 \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-path ./model/Llama-3.1-8B-Instruct/eagle3 \
    --benchmark-list mtbench:80 humaneval:164 gsm8k:200 \
    --output ./hf_results/eagle3.jsonl

# SpecBlock (ours)
python benchmarks_hf/run_eval.py \
    --algorithm specblock \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-path ./model/Llama-3.1-8B-Instruct/specblock-layer2 \
    --benchmark-list mtbench:80 humaneval:164 gsm8k:200 \
    --output ./hf_results/specblock.jsonl
```

Algorithm-specific defaults are auto-applied. Override via `--config-list "<batch>,<steps>,<topk>,<draft_tokens>"`.

## Analyze results

```bash
python benchmarks_hf/analyze_results.py ./hf_results/specblock.jsonl
```

Reports per-benchmark acceptance length, throughput, and a target-vs-draft timing breakdown.

## Interactive UI

```bash
python benchmarks_hf/run_ui.py --host 127.0.0.1 --port 7860
```

Gradio UI for side-by-side comparison with live token streaming.

## Layout

```
benchmarks_hf/
├── run_eval.py             # evaluation CLI
├── analyze_results.py      # result aggregation
├── run_ui.py               # Gradio UI entry point
├── algorithms/             # algorithm wrappers (baseline / eagle3 / specblock)
├── ui/                     # Gradio UI internals
├── benchmark_datasets/     # dataset loaders
├── modeling/               # eagle3 helper modeling code
└── utils/                  # metrics, KV-cache helpers, prompt formatting
```

## True request batching

The HuggingFace evaluator supports true request-level batching for EAGLE-3 and SpecBlock. Set the first value in `--config-list` to the desired batch size. The target prefill and verification forwards are batched across active requests rather than emulated with serial B=1 calls.

```bash
python benchmarks_hf/run_eval.py \
    --algorithm specblock \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-path ./model/Llama-3.1-8B-Instruct/specblock-layer2 \
    --benchmark-list humaneval:164 \
    --config-list "8,2,10,90" \
    --output ./hf_results/specblock_b8.jsonl
```

The evaluator rejects `batch_size > 1` for algorithms without a true batching implementation. It also records active-batch shrinkage and separates prefill, drafting, target verification, and acceptance overhead.
