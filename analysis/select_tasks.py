"""Apply the exclusion rule to all tasks, compute issue features, and pick
equal multi-file and single-file groups with a fixed seed.

Usage:
    .venv/bin/python analysis/select_tasks.py

Needs data/tasks.jsonl and the validation of all tasks (runs/2026-10-21-validate-all/). Writes:
    analysis/output/task_features.csv   every task: features, group, excluded (and why), selected
    analysis/output/selected_tasks.txt  the 30 picked instance_ids, one per line

Rules (from docs/07-experiment-plan.md, decided before any runs):
- Exclude if the issue is empty or template-only after removing the PR template.
- Exclude if it names no code (file, folder, function, class, error) AND has < 10 words of real text.
- Exclude if the reference fix changes no library file (only docs examples or scripts).
- Multi-file = the fix changes 2+ library files; single-file = exactly 1.

Change Oct 21 (plan change log): only tasks that are solvable in the sandbox are
eligible (fail with no patch, pass with the reference fix). Only 13 multi-file tasks
are, so each group gets 13 (was 15): all 13 multi + 13 single drawn with the seed.
"""
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks.jsonl"
VALIDATION = ROOT / "runs" / "2026-10-21-validate-all" / "validation.jsonl"
OUT = ROOT / "analysis" / "output"

SEED = 20261016  # fixed so the pick can be repeated; recorded in the plan
PER_GROUP = 15  # upper limit; capped at the smaller group's solvable pool
MIN_WORDS = 10

# Library code lives here; tests, docs_src/ examples and scripts/ don't count
LIBRARY = re.compile(r"^(fastapi|rich|src)/")
# PR-template sections whose content is boilerplate, not a description of the change
TEMPLATE_SECTIONS = {"ai disclaimer", "checklist"}
TEMPLATE_LINES = re.compile(r"^\s*(discussion:|- \[[ x]\]|##+ |pull request$)", re.I)


def changed_files(patch):
    return re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)


def lines_changed(patch, files):
    """Added + removed lines inside the given files."""
    count, current = 0, None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")) and current in files:
            count += 1
    return count


def real_text(issue):
    """The issue with the PR template removed: comments, folded transcripts,
    boilerplate sections, checklists and repeated lines (the title is often repeated)."""
    text = re.sub(r"<!--.*?-->", "", issue, flags=re.S)
    text = re.sub(r"https?://\S+", "", text)  # a bare link isn't a description, and its path looks like a folder
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)
    kept, seen, skip = [], set(), False
    for line in text.splitlines():
        heading = re.match(r"^#+\s*(.+)", line)
        if heading:
            skip = heading.group(1).strip().lower() in TEMPLATE_SECTIONS
            continue
        if skip or TEMPLATE_LINES.match(line) or not line.strip():
            continue
        key = line.strip().lower()
        if key not in seen:
            seen.add(key)
            kept.append(line)
    return "\n".join(kept)


def names_code(text):
    """True if the text names a file, folder, function, class or error."""
    patterns = [
        r"`[^`]+`",                                # anything in backticks
        r"\b[\w/]+\.(py|toml|cfg|json|ya?ml)\b",   # file names
        r"\b\w+/\w+",                              # folder paths
        r"\b[a-z]+_[a-z_]+\b",                     # snake_case names
        r"\b[A-Z][a-z]+[A-Z]\w*",                  # CamelCase names
        r"\b\w+\.\w+\(",                           # a.b( calls
        r"\b\w*(Error|Exception|Warning)\b",       # error types
        r"Traceback",
    ]
    return any(re.search(p, text) for p in patterns)


def solvable_tasks():
    """Tasks whose tests fail with no patch and pass with the reference fix."""
    checks = {}
    for line in VALIDATION.open():
        r = json.loads(line)
        checks.setdefault(r["task"], {})[r["check"]] = r["resolved"]
    return {t for t, c in checks.items() if c.get("reference") and not c.get("baseline")}


def features(task, solvable):
    files = changed_files(task["patch"])
    lib = [f for f in files if LIBRARY.match(f)]
    text = real_text(task["problem_statement"])
    words = len(re.findall(r"[A-Za-z]{2,}", text))
    named = names_code(text)

    if words == 0:
        reason = "empty or template-only issue"
    elif not named and words < MIN_WORDS:
        reason = f"names no code and < {MIN_WORDS} words"
    elif not lib:
        reason = "fix changes no library file"
    elif task["instance_id"] not in solvable:
        reason = "not solvable in the sandbox"
    else:
        reason = ""

    return {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "library_files": len(lib),
        "lines_changed": lines_changed(task["patch"], set(lib)),
        "issue_words": words,
        "names_code": named,
        "group": "multi" if len(lib) >= 2 else "single" if lib else "",
        "excluded": reason,
    }


def proportional(counts, total):
    """Split `total` across repos in proportion to `counts` (largest remainder)."""
    pool = sum(counts.values())
    exact = {r: total * n / pool for r, n in counts.items()}
    share = {r: int(x) for r, x in exact.items()}
    for r in sorted(exact, key=lambda r: exact[r] - share[r], reverse=True)[: total - sum(share.values())]:
        share[r] += 1
    return share


def main():
    solvable = solvable_tasks()
    rows = [features(json.loads(line), solvable) for line in TASKS.open()]
    rng = random.Random(SEED)
    pools = {g: [r for r in rows if r["group"] == g and not r["excluded"]] for g in ("multi", "single")}
    per_group = min(PER_GROUP, *(len(p) for p in pools.values()))
    picked = []
    for group, eligible in pools.items():
        by_repo = {}
        for r in sorted(eligible, key=lambda r: r["instance_id"]):
            by_repo.setdefault(r["repo"], []).append(r)
        quota = proportional({k: len(v) for k, v in by_repo.items()}, per_group)
        for repo in sorted(by_repo):
            picked += rng.sample(by_repo[repo], quota[repo])
    chosen = {r["instance_id"] for r in picked}
    for r in rows:
        r["selected"] = r["instance_id"] in chosen

    OUT.mkdir(exist_ok=True)
    with (OUT / "task_features.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (OUT / "selected_tasks.txt").write_text("\n".join(sorted(chosen)) + "\n")

    print(f"{len(rows)} tasks, seed {SEED}, {per_group} per group\n\nExcluded:")
    for r in rows:
        if r["excluded"]:
            print(f"  {r['instance_id']:18} {r['excluded']}")
    print("\nEligible pool and pick, by group and repo:")
    for group in ("multi", "single"):
        for repo in sorted({r["repo"] for r in rows}):
            pool = sum(1 for r in rows if r["group"] == group and r["repo"] == repo and not r["excluded"])
            got = sum(1 for r in picked if r["group"] == group and r["repo"] == repo)
            if pool:
                print(f"  {group:6} {repo:18} pool {pool:3}  picked {got}")
    print(f"\nWrote {OUT / 'task_features.csv'} and {OUT / 'selected_tasks.txt'}")


if __name__ == "__main__":
    main()
