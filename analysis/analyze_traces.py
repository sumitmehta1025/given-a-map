"""Turn experiment results into one row per attempt, with the measures the plan promises.

Usage:
    # A run folder written by experiment/run_experiment.py (results/run<N>/)
    .venv/bin/python analysis/analyze_traces.py runs/2026-10-19-smoke/results/run0

    # Loose traces (e.g. the Sep 27 demo): give the setup and budget they ran with
    .venv/bin/python analysis/analyze_traces.py --traces runs/2026-09-27-baseline-demo/trace_*.json \
        --setup B --budget-minutes 5 --out analysis/output/sep27

Needs data/tasks.jsonl (reference fixes) and analysis/output/task_features.csv (task groups).
Writes to --out (default analysis/output/<run folder name>/):
    attempts.csv   every column below, one row per attempt
    sheet_h1_h2.csv, sheet_h3.csv, sheet_h4.csv   the measurement sheet's column order, ready to paste

Rules (docs/07-experiment-plan.md, locked Oct 17). H3: each failed attempt gets the first
category that matches, in this order:
  1 Crash or tool error       the attempt errored, the patch or tests couldn't be applied,
                              no tool calls, or at least half the tool calls returned errors
  2 Never found the code      never opened a file the reference fix changes
  3 Found it, never edited    opened a reference file; edited no existing file
  4 Edited the wrong place    edited existing files, none of them a reference file
  5 Right place, wrong fix    edited a reference file; a fail-to-pass test still fails
  6 Fixed it, broke something fail-to-pass tests pass; another test fails
  Other                       none of the above
"Hit the limit" is a flag on every failure: the agent ran out of time, tool calls or turns.
H4: a search_similar_code query is a "Sentence" if it contains a space, else a "Code name".
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "tasks.jsonl"
FEATURES = ROOT / "analysis" / "output" / "task_features.csv"

GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}
UNCOUNTED_TOOLS = {"submit_patch", "get_status"}  # the harness doesn't charge these to the call budget
SETUP_NAMES = {"A": "Grep only", "B": "Graph + grep", "C": "Graph + grep, prompted", "D": "Graph required"}
GROUP_NAMES = {"multi": "Multi-file", "single": "Single-file"}
LIBRARY = re.compile(r"^(fastapi|rich|src)/")  # same rule as select_tasks.py
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
VIEW_COMMAND = re.compile(r"\b(cat|sed|head|tail|nl|less|more|awk)\b")
# Harness errors that mean the attempt broke, not that the agent's fix was wrong
CRASH_ERRORS = ("Sandbox execution error", "Failed to apply", "Evaluation error",
                "Missing test specification", "Unexpected evaluation worker error")
LIMIT_ERRORS = ("exceeded session timeout", "exceeded tool call budget", "exceeded turns budget",
                "exceeded maximum allowed LLM turns")
NO_TIME_FLAG_SLACK_S = 60  # legacy rows without the harness's end reason: see hit_limit()

CATEGORIES = ["Crash or tool error", "Never found the code", "Found it, never edited",
              "Edited the wrong place", "Right place, wrong fix", "Fixed it, broke something", "Other"]


def norm(path):
    """Repo-relative path: drop ./, /workspace/ and a/ b/ diff prefixes."""
    path = path.strip().strip("'\"")
    path = re.sub(r"^(/workspace/|\./)+", "", path)
    return path


def patch_files(patch, existing_only=False):
    """Files a unified diff changes. existing_only skips files the diff creates."""
    files = []
    for m in re.finditer(r"^--- (\S+)\n\+\+\+ b/(\S+)", patch or "", re.M):
        if existing_only and m.group(1) == "/dev/null":
            continue
        files.append(m.group(2))
    return files


def reference_files(task):
    """Files the reference fix changes: library files, or all non-test files if it touches none.
    Files the fix creates can't be opened beforehand, so they count only if it creates nothing else."""
    files = patch_files(task["patch"], existing_only=True) or patch_files(task["patch"])
    lib = [f for f in files if LIBRARY.match(f)]
    return set(lib or [f for f in files if not TEST_PATH.search(f)])


def fail_to_pass_tests(test_patch):
    """Test functions the reference test patch adds or changes (the harness's own rule)."""
    added = re.findall(r"^\+\s*(?:async\s+)?def\s+(test_\w+)\s*\(", test_patch, re.M)
    if added:
        return set(added)
    hunks = re.findall(r"^@@[^@]+@@\s*(?:async\s+)?def\s+(test_\w+)", test_patch, re.M)
    context = re.findall(r"^ \s*(?:async\s+)?def\s+(test_\w+)\s*\(", test_patch, re.M)
    return set(hunks + context)


def failed_tests(test_output):
    """Base names of tests pytest reports as FAILED or ERROR."""
    names = re.findall(r"^(?:FAILED|ERROR) \S*::(\w+)", test_output or "", re.M)
    return {n for n in names}


def tool_calls(trace):
    """(name, arguments, elapsed_s, observation dict) for every tool call in the trace."""
    calls = []
    for step in trace["steps"]:
        if step["source"] != "agent":
            continue
        try:
            obs = json.loads((step.get("observation") or {}).get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            obs = {}
        for c in step.get("tool_calls") or []:
            calls.append((c["function_name"], c.get("arguments") or {}, c["extra"]["elapsed_s"],
                          obs if isinstance(obs, dict) else {}))
    return calls


def last_event_s(trace):
    """Seconds from task start to the last thing recorded: a thought, call or tool result."""
    times = [0.0]
    for step in trace["steps"]:
        times.append((step.get("extra") or {}).get("elapsed_s") or 0)
        times.append(((step.get("observation") or {}).get("extra") or {}).get("elapsed_s") or 0)
        times += [c["extra"].get("elapsed_s") or 0 for c in step.get("tool_calls") or []]
    return max(times)


def is_error(obs):
    """A tool error, or a malformed call the framework rejected before the tool ran
    (e.g. edit_file missing a required argument: {"error": "Invoking `edit_file()` failed ..."})."""
    return obs.get("status") == "error" or ("error" in obs and "status" not in obs)


def is_malformed(obs):
    return "error" in obs and "status" not in obs


def counted_calls(calls):
    """Tool calls as the harness counts them against the 100-call budget: excludes
    submit_patch / get_status and malformed calls that never reached a tool."""
    return sum(name not in UNCOUNTED_TOOLS and not is_malformed(obs) for name, _, _, obs in calls)


def opened_files(calls):
    opened = set()
    for name, args, _, _ in calls:
        if name in ("read_file", "edit_file") and args.get("filepath"):
            opened.add(norm(args["filepath"]))
        elif name == "run_command" and VIEW_COMMAND.search(args.get("command", "")):
            opened.update(norm(p) for p in re.findall(r"[\w./-]+\.py\b", args["command"]))
    return opened


def edited_files_from_trace(calls):
    """Fallback when no patch was saved: files changed by successful edit tools,
    minus files the agent created itself (scratch scripts like repro.py)."""
    edited, created = set(), set()
    for name, args, _, obs in calls:
        if obs.get("status") != "ok":
            continue
        if name == "write_file" and args.get("filepath"):
            created.add(norm(args["filepath"]))
        elif name == "edit_file" and args.get("filepath"):
            edited.add(norm(args["filepath"]))
        elif name == "run_command" and "sed -i" in args.get("command", ""):
            edited.update(norm(p) for p in re.findall(r"[\w./-]+\.py\b", args["command"]))
    return edited - created


def hit_limit(row, calls, budget):
    """True if the agent was stopped by the time, tool-call or turn budget.
    Uses the harness's end reason when recorded (runs from Oct 19 on). Older rows: no
    submit_patch call, and either the call budget was used up or the last call came
    within 60 s of the time limit (the agent may think for a minute before the cutoff)."""
    error = (row.get("agent_error") or "").lower()
    if error:
        return any(e in error for e in LIMIT_ERRORS)
    if any(name == "submit_patch" for name, *_ in calls):
        return False
    last = calls[-1][2] if calls else 0
    return len(calls) >= budget["max_tool_calls"] or last >= budget["max_time_minutes"] * 60 - NO_TIME_FLAG_SLACK_S


def categorize(row, task, calls, patch, test_output):
    """First matching H3 category, and the evidence for it (one short line)."""
    refs = reference_files(task)
    errors = sum(is_error(obs) for *_, obs in calls)
    agent_error = row.get("agent_error") or ""
    if row.get("status") == "error" or any(e in agent_error for e in CRASH_ERRORS):
        return CATEGORIES[0], f"harness error: {(row.get('error') or agent_error)[:80]}"
    if not calls:
        return CATEGORIES[0], "no tool calls"
    if errors * 2 >= len(calls):
        return CATEGORIES[0], f"{errors} of {len(calls)} tool calls returned errors"

    opened = opened_files(calls)
    if not opened & refs:
        return CATEGORIES[1], f"never opened {', '.join(sorted(refs))}"

    if patch is not None:
        edited = {f for f in patch_files(patch, existing_only=True) if not TEST_PATH.search(f)}
    else:
        edited = {f for f in edited_files_from_trace(calls) if not TEST_PATH.search(f)}
    if not edited:
        return CATEGORIES[2], f"opened {', '.join(sorted(opened & refs))}; no edit"
    if not edited & refs:
        return CATEGORIES[3], f"edited {', '.join(sorted(edited))}; reference: {', '.join(sorted(refs))}"

    if test_output is None:
        return CATEGORIES[4], "edited a reference file; no test output saved, so 5 vs 6 unchecked"
    f2p = fail_to_pass_tests(task["test_patch"])
    failed = failed_tests(test_output)
    if failed & f2p or (not failed and row.get("test_exit_code") not in (0, None)):
        return CATEGORIES[4], f"fail-to-pass still failing: {', '.join(sorted(failed & f2p)) or 'tests did not run'}"
    if failed:
        return CATEGORIES[5], f"other tests failing: {', '.join(sorted(failed))[:80]}"
    return CATEGORIES[6], "edited a reference file and no test failed, but not resolved"


def search_queries(calls):
    """(query, results returned, kind) for every search_similar_code call."""
    out = []
    for name, args, _, obs in calls:
        if name != "search_similar_code":
            continue
        q = str(args.get("query", ""))
        count = obs.get("count", len(obs.get("results") or []))
        out.append((q, count, "Sentence" if " " in q.strip() else "Code name"))
    return out


def analyze(row, trace_path, patch, test_output, budget, tasks, groups):
    task = tasks[row["task"]]
    trace = json.loads(Path(trace_path).read_text())
    calls = tool_calls(trace)
    resolved = bool(row.get("resolved"))
    category, evidence = ("", "") if resolved else categorize(row, task, calls, patch, test_output)
    limited = not resolved and hit_limit(row, calls, budget)
    timed_out = "timeout" in (row.get("agent_error") or "").lower() or (
        limited and not row.get("agent_error") and len(calls) < budget["max_tool_calls"])
    # A timed-out agent used the whole budget; the trace stops at its last recorded event
    seconds = budget["max_time_minutes"] * 60 if timed_out else last_event_s(trace)
    return {
        "run": row["run"], "setup": SETUP_NAMES[row["setup"]], "task": row["task"],
        "group": GROUP_NAMES.get(groups.get(row["task"], ""), ""),
        "resolved": int(resolved),
        "agent_minutes": round(seconds / 60, 2),
        "tool_calls": counted_calls(calls),
        "all_calls": len(calls),
        "malformed_calls": sum(is_malformed(obs) for *_, obs in calls),
        "graph_tool_calls": sum(name in GRAPH_TOOLS for name, *_ in calls),
        # Did the agent follow setup D's required order? (first call search, second neighbors)
        "first_calls": " > ".join(name for name, *_ in calls[:2]),
        "category": category,
        "hit_limit": "" if resolved else ("Yes" if limited else "No"),
        "evidence": evidence,
        "submitted": int(any(name == "submit_patch" for name, *_ in calls)),
        "queries": search_queries(calls),
    }


def read_optional(path):
    return path.read_text() if path.exists() else None


def from_run_folder(folder):
    """Yield (row, trace, patch, test_output, budget) for each attempt in a run folder."""
    setups = {}  # each session's config lists the setups it ran; the latest wins
    for path in sorted(folder.glob("run_config.session*.json")) + [folder / "run_config.json"]:
        setups.update(json.loads(path.read_text())["setups"])
    for line in (folder / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        setup_dir = folder / f"setup_{row['setup']}"
        budget = setups[row["setup"]]["budget"]
        # Saved by the run script from Oct 19 on; older folders have neither
        saved = (setup_dir / "patches").exists() or (setup_dir / "test_outputs").exists()
        patch = read_optional(setup_dir / "patches" / f"{row['task']}.patch")
        output = read_optional(setup_dir / "test_outputs" / f"{row['task']}.log")
        if saved:  # no file means the harness had nothing to save (empty patch / no tests run)
            patch, output = patch or "", output or ""
        yield row, setup_dir / "traces" / f"trace_{row['task']}.json", patch, output, budget


def from_loose_traces(paths, setup, budget):
    for p in paths:
        task = Path(p).stem.removeprefix("trace_")
        row = {"run": 0, "setup": setup, "task": task, "resolved": False}
        yield row, Path(p), None, None, budget


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_folders", nargs="*", type=Path, help="results/run<N> folders")
    p.add_argument("--traces", nargs="*", default=[], help="loose trace files (no results folder)")
    p.add_argument("--setup", choices=["A", "B"], default="B", help="setup the loose traces ran in")
    p.add_argument("--budget-minutes", type=float, default=4)
    p.add_argument("--max-calls", type=int, default=100)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if not args.run_folders and not args.traces:
        p.error("give a run folder or --traces")

    tasks = {t["instance_id"]: t for t in map(json.loads, TASKS.open())}
    groups = {r["instance_id"]: r["group"] for r in csv.DictReader(FEATURES.open())}
    attempts = []
    for folder in args.run_folders:
        attempts += from_run_folder(folder)
    budget = {"max_time_minutes": args.budget_minutes, "max_tool_calls": args.max_calls}
    attempts += from_loose_traces(args.traces, args.setup, budget)
    results = [analyze(*a, tasks, groups) for a in attempts]

    out = args.out or ROOT / "analysis" / "output" / (args.run_folders[0].name if args.run_folders else "traces")
    out.mkdir(parents=True, exist_ok=True)
    cols = [k for k in results[0] if k != "queries"]
    write_csv(out / "attempts.csv", cols, [[r[c] for c in cols] for r in results])
    write_csv(out / "sheet_h1_h2.csv",
              ["Run", "Setup", "Task", "Group", "Resolved (1 = yes)", "Minutes", "Tool calls"],
              [[r["run"], r["setup"], r["task"], r["group"], r["resolved"], r["agent_minutes"], r["tool_calls"]]
               for r in results])
    write_csv(out / "sheet_h3.csv", ["Run", "Setup", "Task", "Failure category", "Hit the limit?"],
              [[r["run"], r["setup"], r["task"], r["category"], r["hit_limit"]]
               for r in results if not r["resolved"]])
    write_csv(out / "sheet_h4.csv",
              ["Run", "Setup", "Task", "Query", "Results returned", "Query type", "Zero results (1 = yes)"],
              [[r["run"], r["setup"], r["task"], q, n, kind, int(n == 0)]
               for r in results if r["setup"] != SETUP_NAMES["A"] for q, n, kind in r["queries"]])

    print(f"{len(results)} attempts -> {out}\n")
    for r in results:
        verdict = "RESOLVED" if r["resolved"] else f"{r['category']} | hit limit: {r['hit_limit']}"
        print(f"run {r['run']} {r['setup']:12} {r['task']:16} {r['agent_minutes']:5.2f} min "
              f"{r['tool_calls']:3} calls ({r['graph_tool_calls']} graph)  {verdict}")
        if r["evidence"]:
            print(f"{'':36}{r['evidence']}")
        for q, n, kind in r["queries"]:
            print(f"{'':36}search_similar_code({q!r}) -> {n} results [{kind}]")


if __name__ == "__main__":
    main()
