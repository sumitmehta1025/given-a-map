"""Check that each selected task is solvable in the harness's sandbox, without the model.

Usage (Kaggle CPU notebook, experiment/kaggle/validate.ipynb; no GPU needed):
    python validate_tasks.py

For each task, runs the harness's own test step (Phase 2) twice:
  baseline   no patch                 the fail-to-pass tests should FAIL
  reference  the task's reference fix  every test should PASS (resolved)
A task whose reference fix doesn't resolve it can't be solved by any agent here.

Writes results/validate/{baseline,reference}/ (test outputs, harness's writer) and
results/validate/validation.jsonl (one row per task and check).

Why (Oct 21): in run 1, rich_3296's agent made the same change as the reference fix, yet
4 other tests failed, which suggests those tests fail in this sandbox regardless.
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from run_experiment import HARNESS_VERSION, HERE, KAGGLE, install_harness, log


def failed_tests(output):
    return sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", output or "", re.M)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", help="Task IDs (default: all in selected_tasks.txt)")
    p.add_argument("--all", action="store_true", help="Every task in tasks.jsonl (for re-selection)")
    p.add_argument("--data-dir", default=KAGGLE["data_dir"])
    p.add_argument("--wheelhouse", default=KAGGLE["wheelhouse"])
    p.add_argument("--work-dir", default=KAGGLE["work_dir"])
    p.add_argument("--skip-install", action="store_true")
    args = p.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.work_dir) / "results" / "validate"

    if not args.skip_install:
        install_harness(args.wheelhouse)
    from importlib.metadata import version
    assert version("swegemma") == HARNESS_VERSION, f"Harness is {version('swegemma')}, plan uses {HARNESS_VERSION}"
    from adk_eval_core.tracing import SessionTrace
    from swegemma.config import EvalConfig
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks
    from swegemma.results import append_task_result

    by_id = {t.instance_id: t for t in load_tasks(data_dir / "tasks.jsonl")}
    task_ids = sorted(by_id) if args.all else args.tasks or (HERE / "selected_tasks.txt").read_text().split()

    def evaluator(check):
        cfg = EvalConfig(
            tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
            results_dir=out / check, submission_dir=data_dir / "sample_submission", models=None,
            sandbox="subprocess", wheels_dir=data_dir / "wheels", verbose=False,
            skip_agent_patch=(check == "baseline"),
        )
        ev = Evaluator(cfg)
        if check == "reference":
            async def reference_patch(task, *_, **__):  # stands in for the agent: submit the real fix
                return task.patch, None, SessionTrace()
            ev._run_agent_sandbox = reference_patch
        return ev

    evaluators = {c: evaluator(c) for c in ("baseline", "reference")}
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "validation.jsonl"
    for i, task_id in enumerate(task_ids, 1):
        for check, ev in evaluators.items():
            started = time.time()
            r = asyncio.run(ev.evaluate_task(task=by_id[task_id], task_index=i, total_tasks=len(task_ids)))
            append_task_result(r, out / check)
            row = {"task": task_id, "check": check, "resolved": bool(r.resolved),
                   "test_exit_code": r.test_exit_code, "failed": failed_tests(r.test_output),
                   "error": r.error_message, "seconds": round(time.time() - started, 1)}
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"[{i}/{len(task_ids)}] {task_id:16} {check:9} resolved={row['resolved']} "
                f"failed={len(row['failed'])} {row['seconds']}s")

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    bad = [r["task"] for r in rows if r["check"] == "reference" and not r["resolved"]]
    log(f"Done. Reference fix does NOT resolve {len(bad)} of {len(task_ids)} tasks: {bad}")


if __name__ == "__main__":
    main()
