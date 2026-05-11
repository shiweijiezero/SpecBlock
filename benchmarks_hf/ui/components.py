"""UI visualization components for speculative decoding."""

from typing import List, Dict, Any, Optional
import html


def render_token_html(
    text: str,
    token_type: str = "normal",
    tooltip: Optional[str] = None
) -> str:
    """Render a single token with color coding.

    Args:
        text: Token text to display
        token_type: One of "accepted", "rejected", "bonus", "normal"
        tooltip: Optional tooltip text

    Returns:
        HTML string for the token
    """
    colors = {
        "accepted": "#86efac",      # Light green
        "rejected": "#fca5a5",      # Light red
        "bonus": "#93c5fd",         # Light blue
        "normal": "#d1d5db",        # Light gray
        "draft": "#fde047",         # Light yellow
    }

    bg_color = colors.get(token_type, colors["normal"])
    escaped_text = html.escape(text)

    if tooltip:
        escaped_tooltip = html.escape(tooltip)
        return f'<span style="background-color: {bg_color}; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; font-family: monospace;" title="{escaped_tooltip}">{escaped_text}</span>'
    else:
        return f'<span style="background-color: {bg_color}; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; font-family: monospace;">{escaped_text}</span>'


def render_iteration_tokens(
    iteration_data: Dict[str, Any],
    tokenizer = None,
    is_first_iteration: bool = False,
    is_last_iteration: bool = False
) -> str:
    """Render tokens from a single iteration for main output.

    Only shows accepted tokens + bonus. Rejected tokens are shown in iteration details.

    Args:
        iteration_data: Dict with new_tokens, draft_tokens, accepted_count, etc.
        tokenizer: Tokenizer for decoding individual tokens
        is_first_iteration: If True, render root token (first iteration has no previous bonus)
        is_last_iteration: If True, render bonus token (last iteration's bonus won't appear as next root)

    Returns:
        HTML string showing the iteration's tokens
    """
    parts = []

    new_tokens = iteration_data.get("new_tokens", [])
    accepted_count = iteration_data.get("accepted_count", 0)
    bonus_text = iteration_data.get("bonus_text")
    bonus_token = iteration_data.get("bonus_token")

    if not new_tokens and not bonus_text:
        return ""

    # new_tokens structure: [root, draft_1, draft_2, ..., draft_accepted_count]
    # - root (index 0): previous iteration's bonus (already rendered in prev iteration)
    # - draft tokens (index 1+): accepted draft predictions
    # - bonus is separate, will be rendered as next iteration's root
    #
    # To avoid double rendering:
    # - Only render root on first iteration (no previous bonus to duplicate)
    # - Only render bonus on last iteration (won't appear as next root)

    # Render new_tokens
    # new_tokens structure: [root, accepted_drafts..., bonus/correction]
    # - root (index 0): current_token (previous iteration's bonus)
    # - accepted drafts (index 1 to accepted_count): accepted draft predictions
    # - bonus/correction (last): next_token from verification
    if tokenizer is not None and new_tokens:
        # Check if last token in new_tokens is the bonus/correction
        last_is_bonus = bonus_token is not None and len(new_tokens) > 0 and new_tokens[-1] == bonus_token

        for i, token_id in enumerate(new_tokens):
            token_text = tokenizer.decode([token_id])
            if i == 0:
                # Root token - only render on first iteration
                if is_first_iteration:
                    parts.append(render_token_html(token_text, "accepted", "First token"))
                # Skip root on subsequent iterations (was previous bonus)
            elif last_is_bonus and i == len(new_tokens) - 1:
                # Last token is bonus/correction - render as bonus (blue)
                parts.append(render_token_html(token_text, "bonus", "Bonus"))
            else:
                # Accepted draft token
                parts.append(render_token_html(token_text, "accepted", f"Draft #{i}"))
    elif new_tokens:
        # Fallback without tokenizer
        new_text = iteration_data.get("new_text", "")
        parts.append(render_token_html(new_text, "accepted", f"Accepted {len(new_tokens)} tokens"))

    # Note: rejected tokens are NOT shown in main output, only in iteration details

    return "".join(parts)


def render_streaming_chunk(
    iteration_data: Dict[str, Any],
    tokenizer,
    is_first: bool,
) -> str:
    """Build HTML for a SINGLE iteration's incremental output (EAGLE/DART style).

    Returns just the new text contributed by this iteration, with green
    highlight on draft-accepted tokens. Plain text for root + bonus.

    Designed for O(1) per-iter cost: caller appends to an accumulator string
    so the full output never gets re-rendered. Pair with `gr.Chatbot` so
    Gradio's internal message-diff avoids O(N) DOM rebuild.
    """
    new_tokens = iteration_data.get("new_tokens", [])
    accepted_count = iteration_data.get("accepted_count", 0)
    bonus_token = iteration_data.get("bonus_token")

    if not new_tokens:
        return ""
    if tokenizer is None:
        return html.escape(iteration_data.get("new_text", ""))

    # new_tokens layout: [root, draft_1, ..., draft_accepted_count, bonus?]
    # - root (index 0): previous iter's bonus → skip on non-first iter (dedup)
    # - draft tokens (index 1 to accepted_count): orange highlight
    # - bonus (last, if matches bonus_token): plain
    last_is_bonus = (
        bonus_token is not None and len(new_tokens) > 0
        and new_tokens[-1] == bonus_token
    )

    parts = []
    for i, tid in enumerate(new_tokens):
        text = tokenizer.decode([tid])
        escaped = html.escape(text)
        if i == 0:
            if is_first:
                parts.append(escaped)
            # else: skip — already rendered as previous iter's bonus
        elif last_is_bonus and i == len(new_tokens) - 1:
            parts.append(escaped)  # bonus: plain text
        else:
            parts.append(f'<span style="color: #16a34a;">{escaped}</span>')

    return "".join(parts)


def render_tree_structure(
    iteration_data: Dict[str, Any],
    tokenizer = None,
) -> str:
    """Render the draft tree as a visual tree diagram with box-drawing characters.

    Reconstructs a trie from all_paths and renders each node with:
      - Color: Green (accepted), Red (rejected on selected path), Yellow (non-selected)
      - Rank badge: r0 (green), r1-r2 (yellow), r3+ (red) from per-node rank predictions

    Args:
        iteration_data: Dict with tree_info (all_paths with ranks), accepted_count, etc.
        tokenizer: Optional tokenizer (unused, texts come from all_paths).

    Returns:
        HTML string with collapsible tree diagram, or "" if no tree data.
    """
    tree_info = iteration_data.get("tree_info", {})
    all_paths = tree_info.get("all_paths", [])
    accepted_count = iteration_data.get("accepted_count", 0)

    if not all_paths:
        return ""

    # Build trie from paths.
    # Each trie node: {token_id, text, children: OrderedDict, is_on_selected, accepted, rank}
    # Children keyed by token_id — safe because same parent + same token = same trie node.
    root = {"token_id": None, "text": None, "children": {}, "rank": None}

    for path_info in all_paths:
        tokens = path_info.get("tokens", [])
        texts = path_info.get("texts", [])
        ranks = path_info.get("ranks", [])
        block_slots = path_info.get("block_slots", [])
        is_selected = path_info.get("is_selected", False)

        node = root
        for depth, (tok, txt) in enumerate(zip(tokens, texts)):
            rank_val = ranks[depth] if depth < len(ranks) else None
            bs_val = block_slots[depth] if depth < len(block_slots) else None
            if tok not in node["children"]:
                node["children"][tok] = {
                    "token_id": tok,
                    "text": txt,
                    "children": {},
                    "is_on_selected": False,
                    "accepted": False,
                    "rank": rank_val,
                    "block_slot": bs_val,
                }
            child = node["children"][tok]
            # Update rank if we have it (prefer non-None)
            if rank_val is not None and child["rank"] is None:
                child["rank"] = rank_val
            if bs_val is not None and child.get("block_slot") is None:
                child["block_slot"] = bs_val
            if is_selected:
                child["is_on_selected"] = True
                # Accepted = on selected path AND within accepted prefix (root + accepted drafts)
                if depth <= accepted_count:
                    child["accepted"] = True
            node = child

    # Render tree recursively with box-drawing connectors
    lines = []

    def _render_node(node, prefix="", is_last=True, is_root=True):
        children = list(node["children"].values())

        if not is_root and node["text"] is not None:
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            rank_val = node.get("rank")
            is_give_up = (rank_val is not None and rank_val >= 3)

            # Color by status: acceptance takes priority over give-up
            # rank badge already shows r3+ in red, so no need to override color
            if node["accepted"]:
                color = "#86efac"
                label = " \u2713"
            elif is_give_up:
                # r3+ and NOT accepted → gray with ✂
                color = "#d1d5db"
                label = " \u2702"
            elif node["is_on_selected"]:
                color = "#fca5a5"
                label = " \u2717"
            else:
                color = "#fde047"
                label = ""

            # Rank badge
            if rank_val is not None:
                if rank_val == 0:
                    rank_badge = ' <span style="font-size:0.65em; color:#10b981; font-weight:bold;">r0</span>'
                elif rank_val >= 3:
                    rank_badge = f' <span style="font-size:0.65em; color:#ef4444; font-weight:bold;">r{rank_val}</span>'
                else:
                    rank_badge = f' <span style="font-size:0.65em; color:#f59e0b;">r{rank_val}</span>'
            else:
                rank_badge = ""

            # Block-slot badge
            bs_val = node.get("block_slot")
            if bs_val is not None:
                block_idx, slot_idx = bs_val
                bs_badge = f' <span style="font-size:0.6em; color:#6366f1;">b{block_idx}p{slot_idx + 1}</span>'
            else:
                bs_badge = ""

            text_escaped = html.escape(node["text"])
            extra_style = "text-decoration:line-through; opacity:0.6;" if (is_give_up and not node["accepted"]) else ""
            token_html = (
                f'<span style="background:{color}; padding:1px 4px; border-radius:3px; '
                f'font-family:monospace; font-size:0.85em; {extra_style}">{text_escaped}</span>'
                f'<span style="font-size:0.7em; color:#94a3b8;">{label}</span>'
                f'{rank_badge}{bs_badge}'
            )
            line_prefix = f'<span style="color:#94a3b8; font-family:monospace; white-space:pre;">{html.escape(prefix + connector)}</span>'
            lines.append(f'{line_prefix}{token_html}')

        # Recurse into children
        child_prefix = prefix + ("    " if is_last else "\u2502   ") if not is_root else ""
        for i, child in enumerate(children):
            _render_node(child, child_prefix, is_last=(i == len(children) - 1), is_root=False)

    _render_node(root)

    if not lines:
        return ""

    tree_html = "<br>".join(lines)
    return f'''
    <details style="margin-top: 8px;" open>
        <summary style="cursor: pointer; color: #475569; font-size: 0.8em; font-weight: 500; padding: 4px; background: #e2e8f0; border-radius: 4px;">
            Draft Tree ({len(all_paths)} paths)
        </summary>
        <div style="margin-top: 4px; padding: 8px; background: #fafafa; border-radius: 4px; border: 1px solid #e2e8f0; line-height: 1.8; overflow-x: auto;">
            {tree_html}
        </div>
    </details>
    '''


def render_tree_paths(
    iteration_data: Dict[str, Any],
    tokenizer = None,
    max_paths: int = 10,
) -> str:
    """Render tree paths visualization for Eagle3.

    Shows all candidate paths with accepted prefix in green, rejected suffix in red.
    """
    tree_info = iteration_data.get("tree_info", {})
    all_paths = tree_info.get("all_paths", [])
    selected_idx = tree_info.get("selected_path_idx", 0)
    accepted_count = iteration_data.get("accepted_count", 0)

    if not all_paths:
        return ""

    # Get the accepted tokens from selected path for comparison
    # accepted_tokens includes root + accepted draft tokens (total: accepted_count + 1)
    selected_path_info = next((p for p in all_paths if p.get("is_selected")), all_paths[0] if all_paths else None)
    if not selected_path_info:
        return ""

    selected_tokens = selected_path_info.get("tokens", [])
    # The accepted prefix: tokens[0:accepted_count+1] (root + accepted drafts)
    accepted_prefix = selected_tokens[:accepted_count + 1]

    # Limit paths shown
    paths_to_show = all_paths[:max_paths]
    has_more = len(all_paths) > max_paths

    path_rows = []
    for path_info in paths_to_show:
        path_idx = path_info.get("path_idx", 0)
        tokens = path_info.get("tokens", [])
        texts = path_info.get("texts", [])
        is_selected = path_info.get("is_selected", False)

        # Use pre-decoded texts if available, otherwise decode
        if not texts and tokenizer is not None:
            texts = [tokenizer.decode([t]) for t in tokens]
        elif not texts:
            texts = [f"[{t}]" for t in tokens]

        # Build path visualization
        # Compare each token with accepted_prefix
        # Token 0 is root (previous bonus), tokens 1+ are draft predictions
        token_parts = []
        for i, (token_id, text) in enumerate(zip(tokens, texts)):
            if i == 0:
                # Root token (previous iteration's bonus) - gray/normal
                token_parts.append(render_token_html(text, "normal", "Root"))
            elif i < len(accepted_prefix) and token_id == accepted_prefix[i]:
                # Matches accepted prefix - green
                token_parts.append(render_token_html(text, "accepted", f"Draft #{i}"))
            else:
                # Doesn't match - red (rejected/wrong branch)
                token_parts.append(render_token_html(text, "rejected", f"Rejected #{i}"))

        path_html = " → ".join(token_parts)

        if is_selected:
            # Highlight selected path with stronger styling
            row_style = "background: linear-gradient(90deg, #dbeafe 0%, #e0f2fe 100%); border: 2px solid #3b82f6; box-shadow: 0 1px 3px rgba(59, 130, 246, 0.3);"
            marker = "★"
            marker_style = "color: #3b82f6; font-weight: bold;"
        else:
            row_style = "background: #f8fafc; border: 1px solid #e2e8f0;"
            marker = ""
            marker_style = "color: #94a3b8;"

        path_rows.append(f'''
            <div style="padding: 4px 8px; margin: 2px 0; border-radius: 4px; {row_style} font-size: 0.8em;">
                <span style="{marker_style} width: 35px; display: inline-block;">{marker} P{path_idx}</span>
                {path_html}
            </div>
        ''')

    # Build "more paths" section if needed
    more_html = ""
    if has_more:
        remaining_paths = all_paths[max_paths:]
        more_rows = []
        for path_info in remaining_paths:
            path_idx = path_info.get("path_idx", 0)
            tokens = path_info.get("tokens", [])
            texts = path_info.get("texts", [])
            is_selected = path_info.get("is_selected", False)

            if not texts and tokenizer is not None:
                texts = [tokenizer.decode([t]) for t in tokens]
            elif not texts:
                texts = [f"[{t}]" for t in tokens]

            token_parts = []
            for i, (token_id, text) in enumerate(zip(tokens, texts)):
                if i == 0:
                    token_parts.append(render_token_html(text, "normal", "Root"))
                elif i < len(accepted_prefix) and token_id == accepted_prefix[i]:
                    token_parts.append(render_token_html(text, "accepted", f"Draft #{i}"))
                else:
                    token_parts.append(render_token_html(text, "rejected", f"Rejected #{i}"))

            path_html = " → ".join(token_parts)

            if is_selected:
                row_style = "background: linear-gradient(90deg, #dbeafe 0%, #e0f2fe 100%); border: 2px solid #3b82f6; box-shadow: 0 1px 3px rgba(59, 130, 246, 0.3);"
                marker = "★"
                marker_style = "color: #3b82f6; font-weight: bold;"
            else:
                row_style = "background: #f8fafc; border: 1px solid #e2e8f0;"
                marker = ""
                marker_style = "color: #94a3b8;"

            more_rows.append(f'''
                <div style="padding: 4px 8px; margin: 2px 0; border-radius: 4px; {row_style} font-size: 0.8em;">
                    <span style="{marker_style} width: 35px; display: inline-block;">{marker} P{path_idx}</span>
                    {path_html}
                </div>
            ''')

        more_html = f'''
        <details style="margin-top: 4px;">
            <summary style="cursor: pointer; color: #64748b; font-size: 0.75em; padding: 4px; background: #f1f5f9; border-radius: 4px;">
                ... and {len(remaining_paths)} more paths (click to expand)
            </summary>
            <div style="margin-top: 4px;">
                {"".join(more_rows)}
            </div>
        </details>
        '''

    # Wrap in collapsible details element (default closed)
    return f'''
    <details style="margin-top: 8px;">
        <summary style="cursor: pointer; color: #475569; font-size: 0.8em; font-weight: 500; padding: 4px; background: #e2e8f0; border-radius: 4px;">
            🌳 Draft Tree Paths ({len(all_paths)} total) - click to expand
        </summary>
        <div style="margin-top: 4px;">
            {"".join(path_rows)}
            {more_html}
        </div>
    </details>
    '''




def render_iteration_detail(
    iteration_data: Dict[str, Any],
    iteration_idx: int,
    tokenizer = None,
) -> str:
    """Render detailed view of a single iteration (selected path).

    Shows: Draft path → Accepted → Rejected → Bonus, plus draft tree with rank annotations.
    """
    new_tokens = iteration_data.get("new_tokens", [])
    accepted_count = iteration_data.get("accepted_count", 0)
    rejected_token = iteration_data.get("rejected_token")
    rejected_text = iteration_data.get("rejected_text", "")
    bonus_token = iteration_data.get("bonus_token")
    bonus_text = iteration_data.get("bonus_text", "")
    tree_info = iteration_data.get("tree_info", {})
    rank_info = iteration_data.get("rank_info", {})

    if not new_tokens:
        return ""

    # Build the selected path visualization
    path_parts = []
    accepted_parts = []

    # Get selected path from tree_info (preferred) or fallback to other sources
    all_paths = tree_info.get("all_paths", [])
    selected_path_info = next((p for p in all_paths if p.get("is_selected")), None)
    accepted_texts = tree_info.get("accepted_texts", [])

    if selected_path_info:
        # Use the selected path from tree_info (this is the correct path!)
        path_tokens = selected_path_info.get("tokens", [])
        path_texts = selected_path_info.get("texts", [])

        # path_tokens[0] is root (previous bonus), path_tokens[1:] are draft predictions
        # accepted_count is the number of accepted draft tokens (not including root)
        for i, text in enumerate(path_texts):
            if i == 0:
                # Root token (skip for path display, but include in path_parts)
                path_parts.append(render_token_html(text, "normal", "Root"))
            elif i <= accepted_count:
                # Accepted draft token
                path_parts.append(render_token_html(text, "accepted", f"Draft #{i}"))
                accepted_parts.append(render_token_html(text, "accepted", f"Draft #{i}"))
            else:
                # Rejected draft token
                path_parts.append(render_token_html(text, "rejected", f"Rejected #{i}"))
    elif accepted_texts:
        # Fallback: Use pre-decoded texts from tree_info (incomplete path)
        for text in accepted_texts:
            path_parts.append(render_token_html(text, "accepted"))
            accepted_parts.append(render_token_html(text, "accepted"))

        # Rejected token
        if rejected_text:
            path_parts.append(render_token_html(rejected_text, "rejected"))
    elif tokenizer is not None:
        # Final fallback: use draft_tokens for full path visualization
        draft_tokens = iteration_data.get("draft_tokens", [])
        draft_text = iteration_data.get("draft_text", [])

        if draft_tokens:
            # Show all draft tokens with accept/reject coloring
            for i, token_id in enumerate(draft_tokens):
                if i < len(draft_text):
                    token_text = draft_text[i]
                else:
                    token_text = tokenizer.decode([token_id])

                if i < accepted_count:
                    # Accepted draft token
                    path_parts.append(render_token_html(token_text, "accepted", f"Draft #{i+1}"))
                    accepted_parts.append(render_token_html(token_text, "accepted", f"Draft #{i+1}"))
                else:
                    # Rejected draft token
                    path_parts.append(render_token_html(token_text, "rejected", f"Rejected #{i+1}"))
        else:
            # Fallback to new_tokens if no draft_tokens
            for i in range(min(accepted_count, len(new_tokens))):
                token_text = tokenizer.decode([new_tokens[i]])
                path_parts.append(render_token_html(token_text, "accepted"))
                accepted_parts.append(render_token_html(token_text, "accepted"))

            # Rejected token (if any)
            if rejected_token is not None and rejected_token > 0:
                rej_text = tokenizer.decode([rejected_token])
                path_parts.append(render_token_html(rej_text, "rejected"))
    else:
        # Fallback without tokenizer
        if accepted_count > 0:
            accepted_parts.append(render_token_html(f"[{accepted_count} tokens]", "accepted"))
            path_parts.append(render_token_html(f"[{accepted_count} tokens]", "accepted"))
        if rejected_text:
            path_parts.append(render_token_html(rejected_text, "rejected"))

    # Bonus token (common handling for all branches)
    if tokenizer is not None and bonus_token is not None:
        bonus_html = render_token_html(tokenizer.decode([bonus_token]), "bonus")
    elif tokenizer is not None and len(new_tokens) > accepted_count:
        bonus_t = tokenizer.decode([new_tokens[-1]])
        bonus_html = render_token_html(bonus_t, "bonus")
    else:
        bonus_html = render_token_html(bonus_text or "?", "bonus")

    # Build detail HTML
    path_str = " → ".join([p for p in path_parts]) if path_parts else "(empty)"
    accepted_str = " ".join(accepted_parts) if accepted_parts else "(none)"
    rejected_str = render_token_html(rejected_text, "rejected") if rejected_text else "(none)"

    # Render tree structure (branching diagram with rank annotations) and flat paths
    tree_structure_html = render_tree_structure(iteration_data, tokenizer)
    tree_html = render_tree_paths(iteration_data, tokenizer)

    # Compact blocks info in header
    num_blocks = rank_info.get("num_blocks", 0) if rank_info else 0
    blocks_label = f" | {num_blocks} blocks" if num_blocks > 1 else ""

    detail_html = f'''
    <div style="padding: 8px; margin: 4px 0; background: #f1f5f9; border-radius: 6px; font-size: 0.85em; border: 1px solid #e2e8f0;">
        <div style="color: #64748b; margin-bottom: 4px; font-weight: 500;">Iteration #{iteration_idx} | Accept: {accepted_count}{blocks_label}</div>
        <div style="margin: 2px 0;"><span style="color: #64748b; width: 60px; display: inline-block;">Path:</span> {path_str}</div>
        <div style="margin: 2px 0;"><span style="color: #64748b; width: 60px; display: inline-block;">Accept:</span> {accepted_str}</div>
        <div style="margin: 2px 0;"><span style="color: #64748b; width: 60px; display: inline-block;">Reject:</span> {rejected_str}</div>
        <div style="margin: 2px 0;"><span style="color: #64748b; width: 60px; display: inline-block;">Bonus:</span> {bonus_html}</div>
        {tree_structure_html}
        {tree_html}
    </div>
    '''
    return detail_html


def render_streaming_output(
    iterations: List[Dict[str, Any]],
    tokenizer = None,
    show_details: bool = False
) -> str:
    """Render full streaming output from all iterations.

    Args:
        iterations: List of iteration data dicts
        tokenizer: Optional tokenizer for per-token decoding
        show_details: Whether to show iteration boundaries

    Returns:
        HTML string with full colored output
    """
    output_parts = []
    detail_parts = []

    # Filter out final iterations for counting
    non_final_iterations = [it for it in iterations if not it.get("final")]
    total_iterations = len(non_final_iterations)

    for i, iteration in enumerate(iterations):
        if iteration.get("final"):
            continue

        # Determine position for avoiding duplicate token rendering
        is_first = (i == 0)
        is_last = (i == total_iterations - 1)

        # Main output (colored tokens)
        iteration_html = render_iteration_tokens(
            iteration,
            tokenizer=tokenizer,
            is_first_iteration=is_first,
            is_last_iteration=is_last
        )
        if show_details:
            output_parts.append(f'<span style="color: #666; font-size: 0.8em;">[{i}]</span>')
        output_parts.append(iteration_html)

        # Collapsible detail
        detail_parts.append(render_iteration_detail(iteration, i, tokenizer=tokenizer))

    # Main output section
    output_html = f'<div style="line-height: 1.8; padding: 10px; background: #f8fafc; border-radius: 8px; color: #1f2937; border: 1px solid #e2e8f0;">{" ".join(output_parts)}</div>'

    # Collapsible details section
    if detail_parts:
        details_html = f'''
        <details style="margin-top: 10px;">
            <summary style="cursor: pointer; padding: 8px; background: #e2e8f0; border-radius: 6px; color: #1f2937;">
                📋 Iteration Details ({len(detail_parts)} iterations)
            </summary>
            <div style="max-height: 400px; overflow-y: auto; margin-top: 5px; color: #1f2937;">
                {"".join(detail_parts)}
            </div>
        </details>
        '''
    else:
        details_html = ""

    return output_html + details_html


def render_metrics_html(metrics: Dict[str, Any], baseline_tps: float = 34.5) -> str:
    """Render metrics as an HTML dashboard.

    Args:
        metrics: Dict with accept_length, tokens_so_far, elapsed_time, etc.
        baseline_tps: Baseline tokens per second for speedup calculation (default: 34.5, Llama-3.1-8B AR)

    Returns:
        HTML string for metrics display
    """
    accept_length = metrics.get("accept_length", 0)
    tokens_so_far = metrics.get("tokens_so_far", metrics.get("total_tokens", 0))
    elapsed_time = metrics.get("elapsed_time", metrics.get("wall_time", 0))
    tokens_per_second = tokens_so_far / elapsed_time if elapsed_time > 0 else 0

    # Calculate estimated speedup vs baseline
    speedup = tokens_per_second / baseline_tps if baseline_tps > 0 else 0
    speedup_color = "#10b981" if speedup >= 1.0 else "#ef4444"  # Green if >= 1x, red otherwise

    html_parts = [
        '<div style="display: flex; gap: 20px; padding: 10px; background: #f1f5f9; border-radius: 8px; color: #1f2937; border: 1px solid #e2e8f0; flex-wrap: wrap;">',
        f'<div style="text-align: center;"><div style="font-size: 1.5em; font-weight: bold;">{accept_length:.2f}</div><div style="font-size: 0.8em; color: #64748b;">Accept Length</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 1.5em; font-weight: bold;">{tokens_so_far}</div><div style="font-size: 0.8em; color: #64748b;">Tokens</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 1.5em; font-weight: bold;">{tokens_per_second:.1f}</div><div style="font-size: 0.8em; color: #64748b;">Tokens/sec</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 1.5em; font-weight: bold; color: {speedup_color};">{speedup:.2f}x</div><div style="font-size: 0.8em; color: #64748b;">Speedup (vs {baseline_tps:.1f} t/s)</div></div>',
        f'<div style="text-align: center;"><div style="font-size: 1.5em; font-weight: bold;">{elapsed_time:.2f}s</div><div style="font-size: 0.8em; color: #64748b;">Time</div></div>',
        '</div>'
    ]

    return "".join(html_parts)


def render_timing_breakdown(metrics: Dict[str, Any]) -> str:
    """Render timing breakdown as HTML bars.

    Args:
        metrics: Dict with draft_time, target_time, verify_time, iterations, etc.

    Returns:
        HTML string for timing breakdown
    """
    draft_time = metrics.get("draft_time", 0)
    target_time = metrics.get("target_time", metrics.get("verify_time", 0))
    prefill_time = metrics.get("prefill_time", 0)
    other_time = metrics.get("other_time", 0)
    iterations = metrics.get("iterations", 1)
    # Use real wall_time as total to ensure percentages sum to 100%
    total_time = metrics.get("wall_time", draft_time + target_time + prefill_time + other_time)

    if total_time == 0:
        return ""

    # Calculate average time per iteration for fair comparison across different sequence lengths
    avg_draft_time = draft_time / iterations if iterations > 0 else 0
    avg_target_time = target_time / iterations if iterations > 0 else 0

    prefill_pct = prefill_time / total_time * 100
    draft_pct = draft_time / total_time * 100
    target_pct = target_time / total_time * 100
    other_pct = other_time / total_time * 100

    html_parts = [
        '<div style="padding: 10px; background: #f1f5f9; border-radius: 8px; color: #1f2937; margin-top: 10px; border: 1px solid #e2e8f0;">',
        '<div style="font-size: 0.9em; margin-bottom: 5px; font-weight: 500;">Timing Breakdown</div>',
        '<div style="display: flex; height: 24px; border-radius: 4px; overflow: hidden;">',
    ]

    # Prefill segment
    if prefill_pct > 0.5:
        html_parts.append(
            f'<div style="width: {prefill_pct}%; background: #a78bfa; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: white;">Prefill {prefill_pct:.1f}%</div>'
        )

    # Per-block draft time breakdown (for SpecBlock with multiple TTT blocks)
    draft_forward_times = metrics.get("draft_forward_times")
    if draft_forward_times and len(draft_forward_times) > 1:
        # Normalize keys to int (may be str if loaded from JSON)
        draft_forward_times = {int(k): v for k, v in draft_forward_times.items()}
        # Show per-block segments within the draft bar
        block_colors = ['#fbbf24', '#f59e0b', '#d97706', '#b45309', '#92400e']
        for depth in sorted(draft_forward_times.keys()):
            blk_time = draft_forward_times[depth]
            blk_pct = blk_time / total_time * 100
            color = block_colors[depth % len(block_colors)]
            html_parts.append(
                f'<div style="width: {blk_pct}%; background: {color}; display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: #1f2937;">B{depth} {blk_pct:.1f}%</div>'
            )
    else:
        html_parts.append(
            f'<div style="width: {draft_pct}%; background: #fbbf24; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #1f2937;">Draft {draft_pct:.1f}%</div>'
        )

    # Show tree-build overhead if block-level times are available
    if draft_forward_times and len(draft_forward_times) > 1:
        block_gpu_total = sum(draft_forward_times.values())
        tree_build_time = max(0, draft_time - block_gpu_total)
        tree_pct = tree_build_time / total_time * 100
        if tree_pct > 0.5:
            html_parts.append(
                f'<div style="width: {tree_pct}%; background: #fb923c; display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: #1f2937;">Tree {tree_pct:.1f}%</div>'
            )

    html_parts.append(
        f'<div style="width: {target_pct}%; background: #60a5fa; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: white;">Target {target_pct:.1f}%</div>'
    )
    if other_pct > 0.5:
        html_parts.append(
            f'<div style="width: {other_pct}%; background: #94a3b8; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: white;">Other {other_pct:.1f}%</div>'
        )
    html_parts.append('</div>')

    # Per-block average time details
    if draft_forward_times and len(draft_forward_times) > 1:
        block_details = []
        for depth in sorted(draft_forward_times.keys()):
            blk_time = draft_forward_times[depth]
            avg_blk = blk_time / iterations if iterations > 0 else 0
            block_details.append(f"B{depth}: {avg_blk*1000:.2f}ms")
        html_parts.append(
            f'<div style="font-size: 0.8em; color: #64748b; margin-top: 5px;">Avg Draft per block: {" | ".join(block_details)} | Avg Target: {avg_target_time*1000:.2f}ms ({iterations} iters)</div>'
        )
    else:
        html_parts.append(
            f'<div style="font-size: 0.8em; color: #64748b; margin-top: 5px;">Avg Draft: {avg_draft_time*1000:.2f}ms | Avg Target: {avg_target_time*1000:.2f}ms ({iterations} iters)</div>'
        )

    # Tree profile details (if available)
    tree_profile = metrics.get("tree_profile", {})
    if tree_profile:
        tp_parts = []
        for k in sorted(tree_profile.keys()):
            if k != 'total':
                tp_parts.append(f"{k}: {tree_profile[k]:.1f}ms")
        if tp_parts:
            html_parts.append(
                f'<div style="font-size: 0.75em; color: #94a3b8; margin-top: 3px;">Tree profile (last iter): {" | ".join(tp_parts)}</div>'
            )

    html_parts.append('</div>')

    return "".join(html_parts)


def render_accept_lengths_chart(accept_lengths: List[int]):
    """Render accept lengths as a Plotly bar chart.

    Args:
        accept_lengths: List of accept lengths per iteration

    Returns:
        Plotly figure object (or None if empty)
    """
    import plotly.graph_objects as go

    if not accept_lengths:
        return None

    avg_accept = sum(accept_lengths) / len(accept_lengths)
    avg_tokens_per_iter = avg_accept + 1
    max_len = max(accept_lengths)

    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(accept_lengths))),
            y=accept_lengths,
            marker_color='#4ade80',
            hovertemplate='Iteration %{x}<br>Accept Length: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text=f'Accept Lengths per Iteration<br><sup>Avg Accept: {avg_accept:.2f} | Avg Tokens/Iter: {avg_tokens_per_iter:.2f} | Max: {max_len} | Iterations: {len(accept_lengths)}</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Iteration',
        yaxis_title='Accept Length',
        height=250,
        margin=dict(l=50, r=20, t=60, b=40),
        paper_bgcolor='#f1f5f9',
        plot_bgcolor='#e2e8f0',
        font=dict(color='#1f2937'),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#cbd5e1')

    return fig


def render_position_accuracy_chart(accept_lengths: List[int], max_depth: int = None):
    """Render cumulative position accuracy as a Plotly bar chart.

    Shows the probability that at least N consecutive draft tokens are accepted.
    - pos1: P(accept_length >= 1) = probability first draft token is accepted
    - pos2: P(accept_length >= 2) = probability first two draft tokens are both accepted
    - etc.

    Args:
        accept_lengths: List of accept lengths per iteration
        max_depth: Maximum depth to show (default: max observed + 1)

    Returns:
        Plotly figure object (or None if empty)
    """
    import plotly.graph_objects as go

    if not accept_lengths:
        return None

    total_iters = len(accept_lengths)
    max_observed = max(accept_lengths)

    # Determine max depth to display
    if max_depth is None:
        max_depth = max_observed + 1
    else:
        max_depth = max(max_depth, max_observed + 1)

    # Calculate cumulative accuracy for each position
    # pos_i accuracy = count(accept_length >= i) / total_iters
    positions = list(range(1, max_depth + 1))
    accuracies = []

    for pos in positions:
        count = sum(1 for al in accept_lengths if al >= pos)
        acc = count / total_iters * 100
        accuracies.append(acc)

    # Create bar chart with color gradient (green for high, red for low)
    colors = [f'rgba({int(255 * (1 - acc/100))}, {int(200 * acc/100)}, 100, 0.8)' for acc in accuracies]

    fig = go.Figure(data=[
        go.Bar(
            x=[f'≥{p}' for p in positions],
            y=accuracies,
            marker_color=colors,
            text=[f'{acc:.1f}%' for acc in accuracies],
            textposition='outside',
            hovertemplate='Position %{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text=f'Cumulative Position Accuracy<br><sup>P(accept ≥ N) for consecutive draft tokens | {total_iters} iterations</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Minimum Accept Length',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 105]),  # Leave room for text labels
        height=250,
        margin=dict(l=50, r=20, t=60, b=40),
        paper_bgcolor='#f1f5f9',
        plot_bgcolor='#e2e8f0',
        font=dict(color='#1f2937'),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#cbd5e1')

    return fig


def render_block_pos_accuracy(block_pos_stats: Dict[str, Any]) -> str:
    """Render per-block per-position acceptance accuracy as an HTML table.

    Args:
        block_pos_stats: Dict from _aggregate_block_pos_stats, mapping
            "b{block}_p{pos}" -> {correct, total, accuracy}

    Returns:
        HTML string with accuracy table
    """
    if not block_pos_stats:
        return ""

    # Parse keys and organize by block
    blocks = {}  # block_idx -> [(pos, stats)]
    for key, stats in sorted(block_pos_stats.items()):
        parts = key.split("_")
        block_idx = int(parts[0][1:])  # "b0" -> 0
        pos_idx = int(parts[1][1:])    # "p1" -> 1
        if block_idx not in blocks:
            blocks[block_idx] = []
        blocks[block_idx].append((pos_idx, stats))

    rows = []
    for block_idx in sorted(blocks.keys()):
        positions = sorted(blocks[block_idx], key=lambda x: x[0])
        for pos_idx, stats in positions:
            acc = stats["accuracy"] * 100
            correct = stats.get("correct", stats.get("accepted", 0))
            total = stats.get("total", stats.get("iter_proposed", 0))
            # Color based on accuracy
            if acc >= 70:
                color = "#10b981"
            elif acc >= 40:
                color = "#f59e0b"
            else:
                color = "#ef4444"
            rows.append(
                f'<tr>'
                f'<td style="padding:2px 8px; font-family:monospace;">b{block_idx}_p{pos_idx}</td>'
                f'<td style="padding:2px 8px; color:{color}; font-weight:bold;">{acc:.1f}%</td>'
                f'<td style="padding:2px 8px; color:#64748b;">{correct}/{total}</td>'
                f'</tr>'
            )

    return f'''
    <details style="margin-top: 8px;" open>
        <summary style="cursor: pointer; color: #475569; font-size: 0.8em; font-weight: 500; padding: 4px; background: #e2e8f0; border-radius: 4px;">
            Per-Block Per-Position Coverage (all tree nodes)
        </summary>
        <table style="margin-top: 4px; font-size: 0.8em; border-collapse: collapse;">
            <tr style="background:#e2e8f0;">
                <th style="padding:2px 8px; text-align:left;">Position</th>
                <th style="padding:2px 8px; text-align:left;">Coverage</th>
                <th style="padding:2px 8px; text-align:left;">Hit/Total Nodes</th>
            </tr>
            {"".join(rows)}
        </table>
    </details>
    '''


def render_block_pos_accuracy_chart(block_pos_stats: Dict[str, Any]):
    """Render per-block per-position accuracy as a Plotly grouped bar chart.

    Args:
        block_pos_stats: Dict from _aggregate_block_pos_stats

    Returns:
        Plotly figure object (or None if empty)
    """
    import plotly.graph_objects as go

    if not block_pos_stats:
        return None

    # Parse and organize by block
    blocks = {}
    for key, stats in sorted(block_pos_stats.items()):
        parts = key.split("_")
        block_idx = int(parts[0][1:])
        pos_idx = int(parts[1][1:])
        if block_idx not in blocks:
            blocks[block_idx] = []
        blocks[block_idx].append((pos_idx, stats["accuracy"] * 100))

    # Build traces per block
    traces = []
    colors = ['#4ade80', '#60a5fa', '#fbbf24', '#f87171', '#a78bfa']
    for block_idx in sorted(blocks.keys()):
        positions = sorted(blocks[block_idx], key=lambda x: x[0])
        x_labels = [f'p{p}' for p, _ in positions]
        y_values = [acc for _, acc in positions]
        color = colors[block_idx % len(colors)]
        traces.append(go.Bar(
            name=f'Block {block_idx}',
            x=x_labels,
            y=y_values,
            marker_color=color,
            text=[f'{v:.1f}%' for v in y_values],
            textposition='outside',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text='Per-Block Per-Position Acceptance Accuracy',
            font=dict(size=14)
        ),
        xaxis_title='Position (slot within block)',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[0, 105]),
        barmode='group',
        height=250,
        margin=dict(l=50, r=20, t=60, b=40),
        paper_bgcolor='#f1f5f9',
        plot_bgcolor='#e2e8f0',
        font=dict(color='#1f2937'),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#cbd5e1')

    return fig


def get_legend_html() -> str:
    """Get the color legend HTML."""
    return '''
    <div style="display: flex; gap: 15px; padding: 10px; background: #f1f5f9; border-radius: 8px; color: #1f2937; margin-bottom: 10px; border: 1px solid #e2e8f0; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <span style="background: #86efac; width: 16px; height: 16px; border-radius: 3px; display: inline-block;"></span>
            <span style="font-size: 0.9em;">Accepted</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <span style="background: #fca5a5; width: 16px; height: 16px; border-radius: 3px; display: inline-block;"></span>
            <span style="font-size: 0.9em;">Rejected</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <span style="background: #93c5fd; width: 16px; height: 16px; border-radius: 3px; display: inline-block;"></span>
            <span style="font-size: 0.9em;">Bonus</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <span style="background: #fde047; width: 16px; height: 16px; border-radius: 3px; display: inline-block;"></span>
            <span style="font-size: 0.9em;">Draft (unused)</span>
        </div>
    </div>
    '''
