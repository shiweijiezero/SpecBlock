#!/usr/bin/env python3
"""Main evaluation script for benchmarking speculative decoding algorithms."""

import argparse
import json
from pathlib import Path
import sys
import time

from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_datasets import load_benchmark_dataset
from algorithms import get_algorithm


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding algorithms on various datasets"
    )

    # Model arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["baseline", "eagle3", "specblock"],
        help="Algorithm to use for evaluation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to target model (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default=None,
        help="Path to draft model (required for speculative decoding algorithms)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    # Dataset arguments
    parser.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        required=True,
        help='Benchmark list in format "dataset:num_samples" (e.g., "mtbench:80" "gsm8k:200"). Supported: mtbench, gsm8k, humaneval, math500, ceval, cmmlu, alpaca, wmt23, nq_qa, nq_rag',
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=None,
        help="Sampling temperature(s) for generation. Can specify multiple values. Example: --temperature 0.0 0.7 1.0",
    )

    # Algorithm-specific arguments
    parser.add_argument(
        "--config-list",
        type=str,
        nargs="+",
        default=None,
        help='Config list in format "batch_size,steps,topk,draft_tokens" (e.g., "1,7,10,60" "1,2,10,90"). Defaults: specblock → "1,2,10,90", eagle3 → "1,7,4,20".',
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (JSONL format)",
    )

    parser.add_argument(
        "--draft-quantize",
        type=str,
        default=None,
        choices=[None, "int8", "int8w", "int4"],
        help="Quantize draft model weights (default: None; int8=dynamic w8a8, int8w=weight-only w8a16, int4=weight-only w4a16)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.algorithm != "baseline":
        if args.draft_model_path is None:
            parser.error(f"--draft-model-path is required for {args.algorithm}")

    # Auto-default config_list per algorithm if omitted.
    if args.config_list is None:
        _ALGO_DEFAULT_CFG = {
            "specblock": "1,2,10,90",
            "eagle3":    "1,7,4,20",
        }
        if args.algorithm in _ALGO_DEFAULT_CFG:
            args.config_list = [_ALGO_DEFAULT_CFG[args.algorithm]]
            print(f"[run_eval] --config-list not specified, using {args.algorithm} default: {args.config_list[0]}")
        else:
            parser.error(f"--config-list is required for {args.algorithm} (no default defined)")

    # Parse benchmark-list format: "dataset:num_samples"
    datasets = []
    num_samples_per_dataset = []
    for bench in args.benchmark_list:
        if ":" in bench:
            dataset_name, num_str = bench.split(":", 1)
            datasets.append(dataset_name)
            num_samples_per_dataset.append(int(num_str))
        else:
            datasets.append(bench)
            num_samples_per_dataset.append(None)

    # Determine temperatures to use (aligned with bench_model_speedup.py)
    if args.temperature is not None:
        temperatures = args.temperature if isinstance(args.temperature, list) else [args.temperature]
    else:
        temperatures = [0.0]  # Default temperature

    # Parse config_list: "batch_size,steps,topk,draft_tokens[,tree_k[,tree_k_bfs[,diverse_beam]]]"
    configs = []
    for config_str in args.config_list:
        parts = config_str.split(',')
        if len(parts) < 4 or len(parts) > 7:
            parser.error(f"Invalid config format: {config_str}. Expected 'batch_size,steps,topk,draft_tokens[,tree_k[,tree_k_bfs[,diverse_beam]]]'")
        batch_size, steps, topk, draft_tokens = map(int, parts[:4])
        tree_k = int(parts[4]) if len(parts) >= 5 else 0
        tree_k_bfs = int(parts[5]) if len(parts) >= 6 else 0
        diverse_beam = int(parts[6]) if len(parts) >= 7 else 0
        configs.append({
            "batch_size": batch_size,
            "steps": steps,
            "topk": topk,
            "draft_tokens": draft_tokens,
            "tree_k": tree_k,
            "tree_k_bfs": tree_k_bfs,
            "diverse_beam": diverse_beam,
        })

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run evaluation for each config, dataset, and temperature combination
    for config_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {config_idx+1}/{len(configs)}: {config}")
        print(f"{'='*60}")

        # Initialize algorithm with this config
        print(f"Initializing algorithm: {args.algorithm}")
        algorithm_kwargs = {}
        if args.algorithm == "eagle3":
            algorithm_kwargs["depth"] = config["steps"]
            algorithm_kwargs["topk"] = config["topk"]
            algorithm_kwargs["draft_tokens"] = config["draft_tokens"]
        elif args.algorithm == "specblock":
            # steps=0 means use config.json default
            if config["steps"] > 0:
                algorithm_kwargs["max_blocks"] = config["steps"]
            algorithm_kwargs["total_tokens"] = config["draft_tokens"]
            if config["topk"] > 0:
                algorithm_kwargs["beam_width"] = config["topk"]
            if config.get("tree_k", 0) > 0:
                algorithm_kwargs["tree_K"] = config["tree_k"]
            if config.get("tree_k_bfs", 0) > 0:
                algorithm_kwargs["tree_K_bfs"] = config["tree_k_bfs"]
            if config.get("diverse_beam", 0) > 0:
                algorithm_kwargs["diverse_beam"] = True

        if args.draft_quantize:
            algorithm_kwargs["draft_quantize"] = args.draft_quantize

        algorithm = get_algorithm(
            args.algorithm,
            args.model_path,
            args.draft_model_path,
            args.device,
            **algorithm_kwargs
        )

        for dataset_idx, dataset_name in enumerate(datasets):
            num_samples = num_samples_per_dataset[dataset_idx]

            # Load dataset
            print(f"\n{'='*60}")
            print(f"Loading dataset: {dataset_name}")
            dataset_kwargs = {}
            if num_samples is not None:
                dataset_kwargs["num_samples"] = num_samples

            data = load_benchmark_dataset(dataset_name, **dataset_kwargs)
            print(f"Loaded {len(data)} samples")

            for temp in temperatures:
                print(f"\n{'='*60}")
                print(f"Running evaluation...")
                print(f"  Algorithm: {args.algorithm}")
                print(f"  Config: {config}")
                print(f"  Dataset: {dataset_name}")
                print(f"  Samples: {len(data)}")
                print(f"  Max new tokens: {args.max_new_tokens}")
                print(f"  Temperature: {temp}")
                print(f"{'='*60}")

                # Track metrics for this dataset
                start_time = time.time()
                total_tokens = 0
                accept_lengths = []
                accept_rates = []
                all_accept_lengths_raw = []  # For cumulative position accuracy
                # Time breakdown tracking
                total_prefill_time = 0.0
                total_draft_time = 0.0
                total_target_time = 0.0
                total_verify_time = 0.0
                total_wall_time = 0.0  # Sum of per-sample CUDA wall_time
                total_iterations = 0
                # Per-block draft forward times aggregation (specblock only)
                total_draft_forward_times = {}
                # Rank stats aggregation (specblock only)
                all_rank_stats = []
                all_block_pos_stats = []

                results = []
                warmup_done = False  # Skip first sample: triggers JIT compilation (torch.compile + Triton)
                pbar = tqdm(data, desc=f"{dataset_name} T={temp}", unit="sample")
                for sample in pbar:
                    # Generate response (pass full sample - base class handles turns/conversation)
                    response = algorithm.generate(
                        [sample],
                        max_new_tokens=args.max_new_tokens,
                        temperature=temp,
                    )[0]

                    if not warmup_done:
                        warmup_done = True
                        continue  # Discard first sample: JIT warmup skews timing

                    total_tokens += response["metrics"]["total_tokens"]
                    if "accept_length" in response["metrics"]:
                        accept_lengths.append(response["metrics"]["accept_length"])
                    if "accept_rate" in response["metrics"]:
                        accept_rates.append(response["metrics"]["accept_rate"])
                    if "accept_lengths_raw" in response["metrics"]:
                        all_accept_lengths_raw.extend(response["metrics"]["accept_lengths_raw"])
                    # Collect time breakdown
                    if "prefill_time" in response["metrics"]:
                        total_prefill_time += response["metrics"]["prefill_time"]
                    if "draft_time" in response["metrics"]:
                        total_draft_time += response["metrics"]["draft_time"]
                    if "target_time" in response["metrics"]:
                        total_target_time += response["metrics"]["target_time"]
                    if "verify_time" in response["metrics"]:
                        total_verify_time += response["metrics"]["verify_time"]
                    if "wall_time" in response["metrics"]:
                        total_wall_time += response["metrics"]["wall_time"]
                    if "iterations" in response["metrics"]:
                        total_iterations += response["metrics"]["iterations"]
                    if "draft_forward_times" in response["metrics"] and response["metrics"]["draft_forward_times"]:
                        for depth, t in response["metrics"]["draft_forward_times"].items():
                            total_draft_forward_times[depth] = total_draft_forward_times.get(depth, 0.0) + t
                    if "rank_stats" in response["metrics"] and response["metrics"]["rank_stats"]:
                        all_rank_stats.append(response["metrics"]["rank_stats"])
                    if "block_pos_stats" in response["metrics"] and response["metrics"]["block_pos_stats"]:
                        all_block_pos_stats.append(response["metrics"]["block_pos_stats"])

                    # Update progress bar with real-time metrics
                    elapsed = total_wall_time if total_wall_time > 0 else (time.time() - start_time)
                    cur_throughput = total_tokens / elapsed if elapsed > 0 else 0
                    cur_acc_len = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0
                    pbar.set_postfix({
                        "tok/s": f"{cur_throughput:.1f}",
                        "acc_len": f"{cur_acc_len:.2f}",
                        "tokens": total_tokens,
                    })

                    # Save individual result
                    result = {
                        "id": sample["id"],
                        "category": sample.get("category", "unknown"),
                        "algorithm": args.algorithm,
                        "config": config,
                        "dataset": dataset_name,
                        "temperature": temp,
                        "output": response["output"],
                        "metrics": response["metrics"],
                    }
                    results.append(result)

                duration = total_wall_time if total_wall_time > 0 else (time.time() - start_time)
                throughput = total_tokens / duration if duration > 0 else 0
                avg_acc_length = sum(accept_lengths) / len(accept_lengths) if accept_lengths else None
                avg_acc_rate = sum(accept_rates) / len(accept_rates) if accept_rates else None

                # Calculate time breakdown percentages
                total_stage_time = total_draft_time + total_target_time + total_verify_time
                draft_pct = (total_draft_time / total_stage_time * 100) if total_stage_time > 0 else None
                target_pct = (total_target_time / total_stage_time * 100) if total_stage_time > 0 else None
                verify_pct = (total_verify_time / total_stage_time * 100) if total_stage_time > 0 else None

                # Calculate average per-iteration times (in milliseconds)
                avg_draft_ms = (total_draft_time / total_iterations * 1000) if total_iterations > 0 else None
                avg_target_ms = (total_target_time / total_iterations * 1000) if total_iterations > 0 else None
                avg_verify_ms = (total_verify_time / total_iterations * 1000) if total_iterations > 0 else None

                # Calculate per-block average times (specblock only)
                avg_block_ms = {}
                if total_draft_forward_times and total_iterations > 0:
                    for depth in sorted(total_draft_forward_times.keys()):
                        avg_block_ms[depth] = total_draft_forward_times[depth] / total_iterations * 1000

                # Calculate cumulative position accuracy
                pos_acc_formatted = None
                pos_acc_list = None
                if all_accept_lengths_raw:
                    pos_acc = algorithm.compute_cumulative_position_accuracy(
                        all_accept_lengths_raw, max_depth=config.get("steps", 7)
                    )
                    pos_acc_formatted = pos_acc['formatted']
                    pos_acc_list = pos_acc['accuracies']

                # Aggregate rank stats across samples (specblock)
                agg_rank_stats = None
                if all_rank_stats:
                    # Average scalar metrics across samples
                    n = len(all_rank_stats)
                    agg_rank_stats = {
                        "mean_M": sum(s.get("mean_M", 0) for s in all_rank_stats) / n,
                        "mean_branch": sum(s.get("mean_branch", 0) for s in all_rank_stats) / n,
                        "rank0_precision": sum(s.get("rank0_precision", 0) for s in all_rank_stats) / n,
                        "rank0_recall": sum(s.get("rank0_recall", 0) for s in all_rank_stats) / n,
                        "rank0_f1": sum(s.get("rank0_f1", 0) for s in all_rank_stats) / n,
                    }

                # Aggregate block_pos_stats across samples
                agg_block_pos_stats = None
                if all_block_pos_stats:
                    agg_block_pos_stats = {}
                    for sample_stats in all_block_pos_stats:
                        for key, val in sample_stats.items():
                            if key not in agg_block_pos_stats:
                                agg_block_pos_stats[key] = {"correct": 0, "total": 0}
                            agg_block_pos_stats[key]["correct"] += val.get("correct", 0)
                            agg_block_pos_stats[key]["total"] += val.get("total", 0)
                    for key in agg_block_pos_stats:
                        t = agg_block_pos_stats[key]["total"]
                        c = agg_block_pos_stats[key]["correct"]
                        agg_block_pos_stats[key]["accuracy"] = c / t if t > 0 else 0.0

                # Create summary record (aligned with bench_model_speedup.py format)
                summary_record = {
                    "batch_size": config["batch_size"],
                    "steps": config["steps"],
                    "topk": config["topk"],
                    "num_draft_tokens": config["draft_tokens"],
                    "acc_length": float(f"{avg_acc_length:.2f}") if avg_acc_length is not None else None,
                    "acc_rate": float(f"{avg_acc_rate:.3f}") if avg_acc_rate is not None else None,
                    "duration": float(f"{duration:.2f}"),
                    "throughput": float(f"{throughput:.2f}"),
                    "completion_tokens": total_tokens,
                    "benchmark": dataset_name,
                    "temperature": temp,
                    "config": f"{config['batch_size']},{config['steps']},{config['topk']},{config['draft_tokens']}",
                    "algorithm": args.algorithm,
                    # Time breakdown (seconds and percentages)
                    "prefill_time": float(f"{total_prefill_time:.2f}") if total_prefill_time > 0 else None,
                    "draft_time": float(f"{total_draft_time:.2f}") if total_draft_time > 0 else None,
                    "target_time": float(f"{total_target_time:.2f}") if total_target_time > 0 else None,
                    "verify_time": float(f"{total_verify_time:.2f}") if total_verify_time > 0 else None,
                    "draft_pct": float(f"{draft_pct:.1f}") if draft_pct is not None else None,
                    "target_pct": float(f"{target_pct:.1f}") if target_pct is not None else None,
                    "verify_pct": float(f"{verify_pct:.1f}") if verify_pct is not None else None,
                    "total_iterations": total_iterations,
                    "avg_draft_ms": float(f"{avg_draft_ms:.2f}") if avg_draft_ms is not None else None,
                    "avg_target_ms": float(f"{avg_target_ms:.2f}") if avg_target_ms is not None else None,
                    "avg_verify_ms": float(f"{avg_verify_ms:.2f}") if avg_verify_ms is not None else None,
                    # Cumulative position accuracy
                    "position_accuracy": pos_acc_list,
                    "position_accuracy_formatted": pos_acc_formatted,
                    # Rank stats (specblock only)
                    "rank_stats": agg_rank_stats,
                    # Per-block per-position stats (specblock only)
                    "block_pos_stats": agg_block_pos_stats,
                    # Per-block draft forward times (specblock only)
                    "draft_forward_times": {str(k): float(f"{v:.4f}") for k, v in total_draft_forward_times.items()} if total_draft_forward_times else None,
                    "avg_block_ms": {str(k): float(f"{v:.2f}") for k, v in avg_block_ms.items()} if avg_block_ms else None,
                }

                # Print summary (aligned with bench_model_speedup.py format)
                acc_len_str = f"{avg_acc_length:.2f}" if avg_acc_length is not None else "N/A"
                acc_rate_str = f"{avg_acc_rate:.3f}" if avg_acc_rate is not None else "N/A"
                time_str = ""
                if draft_pct is not None:
                    avg_time_parts = []
                    if avg_draft_ms is not None:
                        avg_time_parts.append(f"draft={avg_draft_ms:.1f}ms")
                    if avg_target_ms is not None:
                        avg_time_parts.append(f"target={avg_target_ms:.1f}ms")
                    if avg_verify_ms is not None and avg_verify_ms > 0:
                        avg_time_parts.append(f"verify={avg_verify_ms:.1f}ms")
                    avg_time_str = " ".join(avg_time_parts)
                    time_str = f", draft={draft_pct:.1f}% target={target_pct:.1f}% verify={verify_pct:.1f}%, avg/iter: {avg_time_str}"
                print(f"    {dataset_name}: tokens={total_tokens}, "
                      f"acc_len={acc_len_str}, acc_rate={acc_rate_str}, "
                      f"duration={duration:.2f}s, throughput={throughput:.2f} tok/s{time_str}", flush=True)
                # Print per-block draft forward times if available
                if avg_block_ms:
                    block_parts = [f"B{d}={avg_block_ms[d]:.2f}ms" for d in sorted(avg_block_ms.keys())]
                    print(f"    Draft Block Avg/iter: {' | '.join(block_parts)}", flush=True)
                # Print cumulative position accuracy on separate line for clarity
                if pos_acc_formatted:
                    print(f"    Position Accuracy (cumulative): {pos_acc_formatted}", flush=True)
                # Print rank stats if available
                if agg_rank_stats:
                    print(f"    Rank Stats: M={agg_rank_stats['mean_M']:.2f}, "
                          f"Branch={agg_rank_stats['mean_branch']:.2f}, "
                          f"R0-Prec={agg_rank_stats['rank0_precision']:.3f}, "
                          f"R0-Recall={agg_rank_stats['rank0_recall']:.3f}, "
                          f"R0-F1={agg_rank_stats['rank0_f1']:.3f}", flush=True)
                if agg_block_pos_stats:
                    bp_parts = []
                    for key in sorted(agg_block_pos_stats.keys()):
                        s = agg_block_pos_stats[key]
                        bp_parts.append(f"{key}={s['accuracy']*100:.1f}%")
                    print(f"    Block-Pos Accuracy: {' '.join(bp_parts)}", flush=True)

                # Append summary record to output file immediately
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(summary_record, ensure_ascii=False) + '\n')

                all_results.extend(results)

        # Cleanup algorithm for this config
        algorithm.cleanup()

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Configs tested: {len(configs)}")
    print(f"Datasets tested: {len(datasets)}")
    print(f"Temperatures tested: {len(temperatures)}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
