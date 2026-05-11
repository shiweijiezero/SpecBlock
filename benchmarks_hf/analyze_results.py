#!/usr/bin/env python3
"""Analyze and compare benchmark results from multiple runs."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_stats(results: List[Dict]) -> Dict:
    """Calculate statistics from results."""
    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    total_time = sum(r["metrics"]["wall_time"] for r in results)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    stats = {
        "num_samples": len(results),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": tokens_per_sec,
    }

    # Calculate accept rate if available
    if "accept_rate" in results[0]["metrics"]:
        avg_accept_rate = sum(r["metrics"]["accept_rate"] for r in results) / len(results)
        stats["accept_rate"] = avg_accept_rate

    return stats


def generate_markdown_report(baseline_stats: Dict, results_stats: Dict[str, Dict], output_file: str):
    """Generate markdown comparison report."""
    lines = []
    lines.append("# Benchmark Comparison Report")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Algorithm | Samples | Tokens/sec | Speedup | Accept Rate |")
    lines.append("|-----------|---------|------------|---------|-------------|")

    # Baseline row
    baseline_tps = baseline_stats["tokens_per_second"]
    baseline_accept = baseline_stats.get("accept_rate", 0.0)
    lines.append(
        f"| Baseline | {baseline_stats['num_samples']} | "
        f"{baseline_tps:.2f} | 1.00x | {baseline_accept:.2%} |"
    )

    # Other algorithms
    for name, stats in results_stats.items():
        tps = stats["tokens_per_second"]
        speedup = tps / baseline_tps if baseline_tps > 0 else 0
        accept_rate = stats.get("accept_rate", 0.0)
        lines.append(
            f"| {name.title()} | {stats['num_samples']} | "
            f"{tps:.2f} | {speedup:.2f}x | {accept_rate:.2%} |"
        )

    lines.append("")

    # Detailed metrics
    lines.append("## Detailed Metrics")
    lines.append("")

    # Baseline details
    lines.append("### Baseline")
    lines.append("")
    lines.append(f"- **Samples**: {baseline_stats['num_samples']}")
    lines.append(f"- **Total Tokens**: {baseline_stats['total_tokens']}")
    lines.append(f"- **Total Time**: {baseline_stats['total_time']:.2f}s")
    lines.append(f"- **Tokens/second**: {baseline_stats['tokens_per_second']:.2f}")
    if "accept_rate" in baseline_stats:
        lines.append(f"- **Accept Rate**: {baseline_stats['accept_rate']:.2%}")
    lines.append("")

    # Other algorithms details
    for name, stats in results_stats.items():
        lines.append(f"### {name.title()}")
        lines.append("")
        lines.append(f"- **Samples**: {stats['num_samples']}")
        lines.append(f"- **Total Tokens**: {stats['total_tokens']}")
        lines.append(f"- **Total Time**: {stats['total_time']:.2f}s")
        lines.append(f"- **Tokens/second**: {stats['tokens_per_second']:.2f}")
        speedup = stats['tokens_per_second'] / baseline_tps if baseline_tps > 0 else 0
        lines.append(f"- **Speedup**: {speedup:.2f}x")
        if "accept_rate" in stats:
            lines.append(f"- **Accept Rate**: {stats['accept_rate']:.2%}")
        lines.append("")

    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare benchmark results"
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results JSONL file",
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs='+',
        required=True,
        help="Paths to result JSONL files to compare against baseline",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.md",
        help="Output markdown file (default: comparison.md)",
    )

    args = parser.parse_args()

    # Load baseline
    print(f"Loading baseline from {args.baseline}")
    baseline_results = load_jsonl(args.baseline)
    baseline_stats = calculate_stats(baseline_results)
    baseline_algo = baseline_results[0]["algorithm"]
    print(f"  Algorithm: {baseline_algo}")
    print(f"  Samples: {baseline_stats['num_samples']}")
    print(f"  Tokens/sec: {baseline_stats['tokens_per_second']:.2f}")
    print()

    # Load other results
    results_stats = {}
    for result_file in args.results:
        print(f"Loading results from {result_file}")
        results = load_jsonl(result_file)
        algo_name = results[0]["algorithm"]
        stats = calculate_stats(results)
        results_stats[algo_name] = stats

        speedup = stats["tokens_per_second"] / baseline_stats["tokens_per_second"]
        print(f"  Algorithm: {algo_name}")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Tokens/sec: {stats['tokens_per_second']:.2f}")
        print(f"  Speedup: {speedup:.2f}x")
        if "accept_rate" in stats:
            print(f"  Accept rate: {stats['accept_rate']:.2%}")
        print()

    # Generate report
    print(f"Generating comparison report: {args.output}")
    generate_markdown_report(baseline_stats, results_stats, args.output)
    print("Done!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBaseline ({baseline_algo}): {baseline_stats['tokens_per_second']:.2f} tokens/sec")
    for name, stats in results_stats.items():
        speedup = stats['tokens_per_second'] / baseline_stats['tokens_per_second']
        print(f"{name.title()}: {stats['tokens_per_second']:.2f} tokens/sec ({speedup:.2f}x speedup)")
    print()


if __name__ == "__main__":
    main()
