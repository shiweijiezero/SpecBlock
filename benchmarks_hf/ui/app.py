"""Gradio UI application for speculative decoding visualization."""

import html
import os
import gradio as gr
from typing import List, Dict, Any, Tuple
import time
import sys
from pathlib import Path

# Support both module import and gradio CLI hot reload
try:
    from .runner import get_runner
    from .components import (
        render_streaming_output,
        render_streaming_chunk,
        render_metrics_html,
        render_timing_breakdown,
        render_accept_lengths_chart,
        render_position_accuracy_chart,
        render_block_pos_accuracy,
        render_block_pos_accuracy_chart,
        get_legend_html,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ui.runner import get_runner
    from ui.components import (
        render_streaming_output,
        render_streaming_chunk,
        render_metrics_html,
        render_timing_breakdown,
        render_accept_lengths_chart,
        render_position_accuracy_chart,
        render_block_pos_accuracy,
        render_block_pos_accuracy_chart,
        get_legend_html,
    )


# Algorithm configurations
ALGORITHMS = {
    "baseline": {
        "name": "Baseline (No Speculation)",
        "requires_draft": False,
        "params": [],
        "defaults": {},
    },
    "eagle3": {
        "name": "EAGLE3",
        "requires_draft": True,
        "params": ["draft_tokens", "depth", "topk"],
        "defaults": {"draft_tokens": 60, "depth": 7, "topk": 10},
    },
    "specblock": {
        "name": "SpecBlock",
        "requires_draft": True,
        "params": ["total_tokens", "depth", "beam_width", "draft_quantize"],
        "defaults": {"total_tokens": 90, "depth": 2, "beam_width": 10, "draft_quantize": "None"},
        "param_labels": {"depth": "Max Blocks"},
    },
}


# Base directory for scanning draft model checkpoints
MODEL_BASE_DIR = "./model"


def scan_draft_models() -> List[str]:
    """Scan MODEL_BASE_DIR for checkpoint directories containing config.json."""
    if not os.path.isdir(MODEL_BASE_DIR):
        return []
    results = []
    for root, dirs, files in os.walk(MODEL_BASE_DIR):
        if "config.json" in files:
            results.append(root)
            dirs.clear()  # Don't descend into checkpoint dirs
    # Sort by modification time, newest first
    results.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return results


def refresh_draft_models():
    """Refresh the draft model dropdown choices."""
    choices = scan_draft_models()
    return gr.update(choices=choices)


def load_model(
    algorithm: str,
    model_path: str,
    draft_model_path: str,
    draft_tokens: int,
    depth: int,
    topk: int,
    diff_steps: int,
    ngram_size: int,
    total_tokens: int,
    beam_width: int,
    draft_quantize: str = "None",
) -> str:
    """Load model with specified configuration."""
    if not model_path:
        return "Error: Please specify target model path."

    algo_config = ALGORITHMS.get(algorithm)
    if algo_config is None:
        return f"Error: Unknown algorithm {algorithm}"

    if algo_config["requires_draft"] and not draft_model_path:
        return f"Error: {algo_config['name']} requires a draft model path."

    # Build kwargs based on algorithm
    kwargs = {}
    if algorithm == "specblock":
        # SpecBlock uses depth→max_blocks mapping
        kwargs["total_tokens"] = total_tokens
        kwargs["max_blocks"] = depth
        kwargs["beam_width"] = beam_width
        if draft_quantize and draft_quantize != "None":
            kwargs["draft_quantize"] = draft_quantize
    else:
        if "draft_tokens" in algo_config["params"]:
            kwargs["draft_tokens"] = draft_tokens
        if "depth" in algo_config["params"]:
            kwargs["depth"] = depth
        if "topk" in algo_config["params"]:
            kwargs["topk"] = topk
        if "diff_steps" in algo_config["params"]:
            kwargs["diff_steps"] = diff_steps
        if "ngram_size" in algo_config["params"]:
            kwargs["ngram_size"] = ngram_size
        if "total_tokens" in algo_config["params"]:
            kwargs["total_tokens"] = total_tokens

    try:
        runner = get_runner()
        msg = runner.load_algorithm(
            algorithm,
            model_path=model_path,
            draft_model_path=draft_model_path if algo_config["requires_draft"] else None,
            **kwargs
        )
        return msg
    except Exception as e:
        return f"Error: {e}"


def generate_response(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
):
    """Generate response with streaming visualization.

    EAGLE/DART-aligned design:
    - Output rendered into gr.Chatbot (Gradio's message-diff avoids O(N) DOM rebuild).
    - Per-iter we only build an O(1) HTML chunk (orange highlight on accepted drafts),
      appended to an accumulator. No re-render of all prior iterations.
    - Detailed multi-color iteration view + charts are built ONCE on final yield;
      they live in collapsible Accordions so they don't slow the streaming path.
    - 20ms throttle (50 fps cap) drops intermediate UI frames when generation is
      faster than the browser can paint.
    """
    runner = get_runner()

    if runner.algorithm is None:
        yield (
            [{"role": "assistant", "content": "Please load a model first."}],
            "", "", None, None, None, "", "",
        )
        return

    # Immediate placeholder so user gets visual feedback during prefill
    # (which for 8B target on cold cache can run several hundred ms before
    # the first iter yields).
    yield (
        [{"role": "assistant", "content": '<span style="color:#94a3b8;">Generating…</span>'}],
        "", "", None, None, None, "", "",
    )

    tokenizer = getattr(runner.algorithm, 'tokenizer', None)

    accumulated_html = ""
    iterations: List[Dict[str, Any]] = []
    final_metrics = None
    final_output = None
    is_first = True
    last_yield = 0.0
    YIELD_THROTTLE = 0.02  # 50 fps cap

    try:
        for result in runner.generate_streaming(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            is_final = bool(result.get("final"))

            if is_final:
                final_metrics = result.get("metrics", {})
                final_output = result.get("output", "")
                # Replace simple-streaming iterations with detail-rich versions
                # (with tree_info etc.) for post-gen accordion render.
                if result.get("iteration_details"):
                    iterations = result["iteration_details"]
            else:
                iterations.append(result)
                # Incremental chunk render: O(1) per iter, append-only.
                chunk = render_streaming_chunk(result, tokenizer, is_first)
                accumulated_html += chunk
                is_first = False

            # Throttle non-final yields to 50fps cap.
            now = time.perf_counter()
            if not is_final and (now - last_yield) < YIELD_THROTTLE:
                continue
            last_yield = now

            # Baseline mode falls back to escaped final text (no iter chunks).
            if not accumulated_html and final_output:
                accumulated_html = html.escape(final_output)

            chat_history = [
                {"role": "assistant", "content": f'<div style="line-height:1.8;white-space:pre-wrap;">{accumulated_html}</div>'},
            ]

            metrics = (
                final_metrics if final_metrics
                else (iterations[-1].get("current_metrics", {}) if iterations else {})
            )
            metrics_html_str = render_metrics_html(metrics) if metrics else ""
            timing_html_str = render_timing_breakdown(final_metrics) if final_metrics else ""

            # Charts + detailed iter view: only on final (one-time O(N) render).
            chart_fig = pos_acc_fig = bp_fig = None
            bp_html_str = ""
            detail_html_str = ""
            if final_metrics:
                if "accept_lengths_raw" in final_metrics:
                    chart_fig = render_accept_lengths_chart(final_metrics["accept_lengths_raw"])
                    pos_acc_fig = render_position_accuracy_chart(final_metrics["accept_lengths_raw"])
                if "block_pos_stats" in final_metrics:
                    bp_fig = render_block_pos_accuracy_chart(final_metrics["block_pos_stats"])
                    bp_html_str = render_block_pos_accuracy(final_metrics["block_pos_stats"])
                detail_html_str = render_streaming_output(iterations, tokenizer=tokenizer, show_details=False)

            yield (
                chat_history,
                metrics_html_str,
                timing_html_str,
                chart_fig,
                pos_acc_fig,
                bp_fig,
                bp_html_str,
                detail_html_str,
            )

    except Exception as e:
        yield (
            [{"role": "assistant", "content": f"Error during generation: {e}"}],
            "", "", None, None, None, "", "",
        )


def update_param_visibility(algorithm: str) -> Tuple:
    """Update parameter visibility and labels based on selected algorithm."""
    algo_config = ALGORITHMS.get(algorithm, {})
    params = algo_config.get("params", [])
    requires_draft = algo_config.get("requires_draft", False)
    param_labels = algo_config.get("param_labels", {})
    defaults = algo_config.get("defaults", {})

    return (
        gr.Dropdown(visible=requires_draft),  # draft_model_path
        gr.Button(visible=requires_draft),  # refresh_btn
        gr.Slider(minimum=1, maximum=128, step=1, visible="draft_tokens" in params, value=defaults.get("draft_tokens", 60), label="Draft Tokens"),
        gr.Slider(minimum=1, maximum=15, step=1, visible="depth" in params, label=param_labels.get("depth", "Depth"), value=defaults.get("depth", 7)),
        gr.Slider(minimum=1, maximum=20, step=1, visible="topk" in params, label=param_labels.get("topk", "Top-K"), value=defaults.get("topk", 10)),
        gr.Slider(minimum=1, maximum=10, step=1, visible="diff_steps" in params, value=defaults.get("diff_steps", 1), label="Diff Steps"),
        gr.Slider(minimum=2, maximum=10, step=1, visible="ngram_size" in params, value=defaults.get("ngram_size", 3), label="N-gram Size"),
        gr.Slider(minimum=10, maximum=300, step=1, visible="total_tokens" in params, value=defaults.get("total_tokens", 60), label="Total Tokens"),
        gr.Slider(minimum=1, maximum=50, step=1, visible="beam_width" in params, value=defaults.get("beam_width", 10), label="Beam Width"),
        gr.Dropdown(choices=["None", "int8", "int4"], visible="draft_quantize" in params, value=defaults.get("draft_quantize", "None"), label="Draft Quantize"),
    )


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(
        title="SpecForge - Speculative Decoding Visualization",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown("# SpecForge - Speculative Decoding Visualization")
        gr.Markdown("Visualize draft token acceptance/rejection in real-time.")

        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")

                algorithm = gr.Dropdown(
                    choices=list(ALGORITHMS.keys()),
                    value="specblock",
                    label="Algorithm",
                )

                model_path = gr.Textbox(
                    label="Target Model Path",
                    value="meta-llama/Llama-3.1-8B-Instruct",
                    placeholder="meta-llama/Llama-3.1-8B-Instruct",
                )

                with gr.Row():
                    draft_model_path = gr.Dropdown(
                        choices=scan_draft_models(),
                        label="Draft Model Path",
                        allow_custom_value=True,
                        visible=False,
                        scale=4,
                    )
                    refresh_btn = gr.Button("Refresh", visible=False, scale=1)

                gr.Markdown("### Algorithm Parameters")

                with gr.Row():
                    draft_tokens = gr.Slider(
                        minimum=1, maximum=128, value=60, step=1,
                        label="Draft Tokens",
                        visible=True,
                    )
                    depth = gr.Slider(
                        minimum=1, maximum=15, value=7, step=1,
                        label="Depth",
                        visible=False,
                    )

                with gr.Row():
                    topk = gr.Slider(
                        minimum=1, maximum=20, value=10, step=1,
                        label="Top-K",
                        visible=False,
                    )
                    diff_steps = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Diff Steps",
                        visible=False,
                    )

                with gr.Row():
                    ngram_size = gr.Slider(
                        minimum=2, maximum=10, value=3, step=1,
                        label="N-gram Size",
                        visible=True,
                    )
                    total_tokens = gr.Slider(
                        minimum=10, maximum=300, value=60, step=1,
                        label="Total Tokens",
                        visible=False,
                    )

                with gr.Row():
                    beam_width = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="Beam Width",
                        visible=False,
                    )
                    draft_quantize = gr.Dropdown(
                        choices=["None", "int8", "int4"],
                        value="None",
                        label="Draft Quantize",
                        visible=False,
                    )

                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Generation Parameters")

                max_new_tokens = gr.Slider(
                    minimum=16, maximum=2048, value=1024, step=16,
                    label="Max New Tokens",
                )

                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="Temperature (0 = greedy)",
                )

            # Right column: Input/Output
            with gr.Column(scale=2):
                gr.Markdown("### Input")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                )

                generate_btn = gr.Button("Generate", variant="primary")

                gr.Markdown("### Output")
                # gr.Chatbot does message-level diff updates internally, so the
                # browser only repaints the changed message — orders of magnitude
                # cheaper than gr.HTML innerHTML replace on every yield.
                # `sanitize_html=False` lets the orange-highlight `<span>` through.
                chatbot = gr.Chatbot(
                    label="Generated Text",
                    sanitize_html=False,
                    show_label=False,
                    height=400,
                )

                gr.Markdown("### Metrics")
                metrics_html = gr.HTML()

                with gr.Row():
                    timing_html = gr.HTML()

                # Heavy visualization moved into collapsible accordions —
                # rendered ONCE on final yield, not per-iter, so streaming
                # stays smooth.
                with gr.Accordion("📋 Detailed iterations", open=False):
                    legend = gr.HTML(get_legend_html())
                    detail_html = gr.HTML()

                with gr.Accordion("📊 Charts", open=False):
                    with gr.Row():
                        chart_plot = gr.Plot()
                        pos_acc_plot = gr.Plot()
                    with gr.Row():
                        block_pos_plot = gr.Plot()
                        block_pos_html = gr.HTML()

        # Event handlers
        algorithm.change(
            fn=update_param_visibility,
            inputs=[algorithm],
            outputs=[
                draft_model_path,
                refresh_btn,
                draft_tokens,
                depth,
                topk,
                diff_steps,
                ngram_size,
                total_tokens,
                beam_width,
                draft_quantize,
            ],
        )

        refresh_btn.click(
            fn=refresh_draft_models,
            outputs=[draft_model_path],
        )

        load_btn.click(
            fn=load_model,
            inputs=[
                algorithm,
                model_path,
                draft_model_path,
                draft_tokens,
                depth,
                topk,
                diff_steps,
                ngram_size,
                total_tokens,
                beam_width,
                draft_quantize,
            ],
            outputs=[load_status],
        )

        generate_btn.click(
            fn=generate_response,
            inputs=[prompt, max_new_tokens, temperature],
            outputs=[chatbot, metrics_html, timing_html, chart_plot, pos_acc_plot, block_pos_plot, block_pos_html, detail_html],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # 使用 gradio app.py 命令启动可自动热重载
    # 或者直接 python app.py 也可以运行（但无热重载）
    demo.launch()
