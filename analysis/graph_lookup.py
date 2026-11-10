"""Look up a task in its code graph: which code names the issue mentions,
which functions the reference fix changed, and how many graph steps apart they are.

Usage:
    .venv/bin/python analysis/graph_lookup.py <instance_id>

Needs data/tasks.jsonl and the task's graph in data/graphs/<repo>_<commit>.json.
"""
import json
import re
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks.jsonl"
GRAPHS = ROOT / "data" / "graphs"


def load_task(instance_id):
    for line in TASKS.open():
        task = json.loads(line)
        if task["instance_id"] == instance_id:
            return task
    sys.exit(f"Task {instance_id} not found in {TASKS}")


def load_graph(task):
    path = GRAPHS / f"{task['instance_id'].split('_')[0]}_{task['base_commit']}.json"
    if not path.exists():
        sys.exit(f"Graph not downloaded: {path.name}. Ask Claude to fetch it.")
    g = json.loads(path.read_text())
    nodes = {n["id"]: n.get("text", "") for n in g["nodes"]}
    neighbors = {n: set() for n in nodes}
    for e in g["edges"]:  # treat "calls" edges as two-way for distance
        if e["source"] in neighbors and e["target"] in neighbors:
            neighbors[e["source"]].add(e["target"])
            neighbors[e["target"]].add(e["source"])
    return nodes, neighbors


def issue_names(text, nodes):
    """Graph nodes whose short name (last part) appears as a word in the issue."""
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text))
    by_short = {}
    for node in nodes:
        by_short.setdefault(node.rsplit(".", 1)[-1], []).append(node)
    return {w: by_short[w] for w in sorted(words) if w in by_short}


def changed_nodes(patch, nodes):
    """Smallest graph node whose code contains each hunk's removed (or context) lines."""
    found = set()
    for hunk in re.split(r"^@@.*$", patch, flags=re.M)[1:]:
        lines = hunk.splitlines()
        old = [l[1:].strip() for l in lines if l.startswith("-") and not l.startswith("---")]
        ctx = [l[1:].strip() for l in lines if l.startswith(" ")]
        probe = [l for l in (old or ctx) if len(l) > 8]
        if not probe:
            continue
        hits = [n for n, code in nodes.items() if code and all(p in code for p in probe[:3])]
        if hits:
            found.add(min(hits, key=lambda n: len(nodes[n])))
    return sorted(found)


def shortest_path(neighbors, start, goal):
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for nxt in neighbors[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    return None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    task = load_task(sys.argv[1])
    nodes, neighbors = load_graph(task)

    print(f"# {task['instance_id']}  ({len(nodes)} nodes)\n")
    print("Issue (first 300 chars):", task["problem_statement"][:300].replace("\n", " "), "\n")

    named = issue_names(task["problem_statement"], nodes)
    print("Names in the issue that match graph nodes:")
    for word, matches in named.items():
        print(f"  {word:30} -> {', '.join(matches[:3])}{' …' if len(matches) > 3 else ''}")

    targets = changed_nodes(task["patch"], nodes)
    print("\nFunctions/classes the reference fix changed:")
    for t in targets:
        print(f"  {t}")

    print("\nSteps from each issue name to each changed node (calls edges, either direction):")
    for word, matches in named.items():
        for t in targets:
            best = min((p for m in matches if (p := shortest_path(neighbors, m, t))), key=len, default=None)
            steps = len(best) - 1 if best else "no path"
            print(f"  {word:30} -> {t.rsplit('.', 1)[-1]:25} {steps}")


if __name__ == "__main__":
    main()
