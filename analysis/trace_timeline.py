"""Turn an ATIF trace from the swegemma harness into a readable Markdown timeline.

Usage:
    python3 analysis/trace_timeline.py runs/2026-09-27-baseline-demo/trace_fastapi_15588.json

Writes timeline_<task>.md next to the trace and prints a short summary.
"""

import json
import sys
from pathlib import Path

GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}


def load_steps(path):
    trace = json.loads(Path(path).read_text())
    problem = next((s["message"] for s in trace["steps"] if s["source"] == "user"), "")
    return trace, problem


def build_rows(trace):
    """One row per tool call: time, gap since previous call, tool, arguments, tokens."""
    rows, prev_t, prev_prompt = [], 0.0, 0
    for step in trace["steps"]:
        if step["source"] != "agent":
            continue
        metrics = step.get("metrics", {})
        for call in step.get("tool_calls") or []:
            t = call["extra"]["elapsed_s"]
            prompt = metrics.get("prompt_tokens", 0)
            rows.append({
                "step": step["step_id"],
                "t": t,
                "gap": t - prev_t,
                "tool": call["function_name"],
                "args": json.dumps(call["arguments"])[:80],
                "thinking_tokens": metrics.get("completion_tokens", 0),
                "context_tokens": prompt,
                # A big drop in context size means the harness summarized older turns.
                "compacted": prev_prompt and prompt < prev_prompt * 0.6,
                "reasoning": (step.get("message") or "").strip(),
            })
            prev_t, prev_prompt = t, prompt
    return rows


def to_markdown(task, problem, rows, metrics):
    graph_calls = sum(r["tool"] in GRAPH_TOOLS for r in rows)
    lines = [
        f"# Timeline: {task}",
        "",
        f"- Tool calls: {len(rows)} ({graph_calls} graph tools)",
        f"- Last call at: {rows[-1]['t']:.0f} s" if rows else "- No tool calls",
        f"- Tokens written by the model: {metrics.get('total_completion_tokens', 0):,}",
        "",
        "## Problem statement (first 600 characters)",
        "",
        "> " + problem.split("Problem Statement:")[-1].strip()[:600].replace("\n", "\n> "),
        "",
        "## Tool calls",
        "",
        "| Step | Time (s) | Gap (s) | Tool | Arguments | Tokens written | Context tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tool = f"**{r['tool']}**" if r["tool"] in GRAPH_TOOLS else r["tool"]
        ctx = f"{r['context_tokens']:,}" + (" ⚠ compacted" if r["compacted"] else "")
        args = r["args"].replace("|", "\\|")
        lines.append(f"| {r['step']} | {r['t']:.0f} | {r['gap']:.0f} | {tool} | `{args}` | {r['thinking_tokens']:,} | {ctx} |")
    lines += ["", "## What the agent said before each call", ""]
    for r in rows:
        if r["reasoning"]:
            lines.append(f"**Step {r['step']}:** {r['reasoning'][:500]}")
            lines.append("")
    return "\n".join(lines)


def main(path):
    trace, problem = load_steps(path)
    rows = build_rows(trace)
    task = Path(path).stem.removeprefix("trace_")
    out = Path(path).with_name(f"timeline_{task}.md")
    out.write_text(to_markdown(task, problem, rows, trace.get("final_metrics", {})))
    slow = sorted(rows, key=lambda r: r["gap"], reverse=True)[:3]
    print(f"{task}: {len(rows)} tool calls, wrote {out}")
    print("Slowest gaps:", ", ".join(f"step {r['step']} {r['gap']:.0f}s" for r in slow))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(arg)
