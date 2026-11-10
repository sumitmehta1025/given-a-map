"""Run one experiment run: every selected task under each setup (A grep only, B grep + graph,
C grep + graph with the sample's prompt line that points to search_similar_code).

Usage (on Kaggle, from the notebook in experiment/kaggle/; or on the Google Cloud VM):
    python run_experiment.py --run 1                      # all 26 selected tasks x 3 setups
    python run_experiment.py --run 0 --tasks requests_7502   # smoke test

Built from the organizers' "Getting Started" notebook (harness swegemma 0.2.11), so the
model server, compaction and sandbox settings match what the leaderboard uses.

What it does:
1. Installs the harness from the wheelhouse and checks the harness version.
2. Starts one vLLM server (Gemma 4 31B, 4-bit) shared by all setups.
3. For each task, runs each setup. Order alternates task by task (A-B-C on even positions,
   C-B-A on odd), so no setup always gets a "fresh" server.
4. Setup A gets an empty graph folder, so no graph data exists for it at all.
5. Appends one line per attempt to results/run<N>/attempts.jsonl. Re-running the same
   command skips attempts already recorded, so a crashed run can resume.
6. Saves each attempt's patch and test output (patches/, test_outputs/ in each setup
   folder), which the H3 failure categories need.
7. Hang guard (added Oct 21, after run 1 froze for 11 hours inside one attempt): if an
   attempt runs past --hang-minutes, the script dumps every thread's stack to the log and
   exits. The notebook restarts it; the hung attempt is retried once, then recorded as an
   error ("hung"), which the analyzer labels "Crash or tool error".
8. Startup guard and deadline (added Oct 22, after run 1 session 2 froze for 7+ hours while
   restarting): the same hard exit covers loading the model (--startup-minutes). With
   --deadline (Unix time, set by the notebook), no attempt starts unless it can finish
   before then; the script exits with code 3 so the notebook stops and saves the results
   instead of using quota.

Plan: docs/07-experiment-plan.md (locked Oct 17).
"""
import argparse
import asyncio
import faulthandler
import glob
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETUPS = {"A": "A_grep_only", "B": "B_grep_graph", "C": "C_graph_prompted",  # C added Oct 24
          "D": "D_graph_required"}  # D added Oct 25 (exploratory; run with --setups D)
MODEL = "gemma-4-31b-it-qat-w4a16-ct"
HARNESS_VERSION = "0.2.11"  # swegemma version in the wheelhouse on Oct 19; all runs must match
ORDER_SEED = 20261016       # same seed as task selection; offset by run number below
DEADLINE_EXIT = 3           # exit code: stopped at the deadline (the notebook doesn't restart)

KAGGLE = {
    "data_dir": "/kaggle/input/competitions/gemma-4-developer-agent",
    "wheelhouse": "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse",
    "model_dir": f"/kaggle/input/models/google/gemma-4/other/{MODEL}/2",
    "work_dir": "/kaggle/working",
}

# Environment and logging settings copied from the starter notebook
ENV = {
    "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
    "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
    "LITELLM_LOG": "WARNING", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OTEL_SDK_DISABLED": "true", "OTEL_PYTHON_DISABLED": "true",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def install_harness(wheelhouse):
    """Install the wheels exactly as the starter notebook does."""
    for pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                    "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
        for pth in glob.glob(pattern):
            try:
                os.unlink(pth)
            except OSError:
                pass
    tmp = Path("/tmp/wheelhouse")
    tmp.mkdir(parents=True, exist_ok=True)
    for w in Path(wheelhouse).glob("*.whl"):
        if "cutlass" in w.name.lower():
            continue
        name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
        if not (tmp / name).exists():
            try:
                os.symlink(w, tmp / name)
            except OSError:
                shutil.copyfile(w, tmp / name)
    wheels = sorted(str(w) for w in tmp.glob("*.whl"))
    log(f"Installing {len(wheels)} wheels...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels],
                   check=True)
    importlib.invalidate_caches()


def prepare_agents(data_dir, work_dir):
    """Writable copies of the setups, with the sample's LoRA adapters added
    (they're binary and identical for A and B, so they aren't kept in git)."""
    agents = {}
    for key, folder in SETUPS.items():
        dst = work_dir / "agents" / folder
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(HERE / "setups" / folder, dst)
        shutil.copytree(data_dir / "sample_submission" / "adapters", dst / "adapters")
        agents[key] = dst
    return agents


def start_server(agent_dir, model_dir):
    """One vLLM server for all setups (same model and adapters). Settings from the starter notebook."""
    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    litellm.drop_params = True
    declared = discover_declared_models(agent_dir)
    assert declared == {MODEL}, f"Expected only {MODEL}, got {sorted(declared)}"
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}
    ranks = [r for a in adapter_map.values() if (r := getattr(a, "rank", None)) is not None and r > 0]
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tp = 4 if gpus >= 4 else (2 if gpus >= 2 else 1)

    cfg = VllmConfig(
        model=str(model_dir), port=8000, host="127.0.0.1",
        tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768,
        dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)),
        max_lora_rank=max(ranks) if ranks else 128, scheduler_reserve_full_isl=False,
        tensor_parallel_size=tp, startup_timeout=60 * 20,
    )
    server = VllmServer(cfg, adapter_manifest=adapters)
    server.start()
    log(f"vLLM server up on {server.base_url} (tensor parallel {tp}, {gpus} GPUs)")
    models = server.create_model_registry(aliases=[MODEL], model_prefix="openai/",
                                          api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"))
    return server, models, adapters, gpus


def read_budget(agent_dir):
    """Budget from the setup's own eval_config.yaml, read the same way as the starter notebook."""
    import yaml
    raw = yaml.safe_load((agent_dir / "eval_config.yaml").read_text()) or {}
    ev = raw.get("evaluation", raw)
    turns = ev.get("max_turns", ev.get("max_llm_calls"))
    return {
        "timeout_seconds": int(ev.get("timeout_seconds", 300)),
        "verification_timeout_seconds": int(ev.get("verification_timeout_seconds", 300)),
        "max_tool_calls": int(ev.get("max_tool_calls", 100)),
        "max_time_minutes": float(ev.get("max_time_minutes", 60.0)),
        "max_turns": int(turns) if turns is not None else None,
    }


def make_evaluator(key, agent_dir, data_dir, results_dir, models, adapters, empty_dir):
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    budget = read_budget(agent_dir)
    limits, gen_constraints = build_submission_limits()
    graphs = data_dir / "graphs" if key != "A" else empty_dir
    embeddings = data_dir / "embeddings" if key != "A" else empty_dir
    cfg = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
        results_dir=results_dir, submission_dir=agent_dir, models=models,
        sandbox="subprocess", **budget,
        limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                        token_threshold=14336, event_retention_size=5),
        graph_dir=str(graphs), embeddings_dir=str(embeddings),
        wheels_dir=data_dir / "wheels", verbose=False,
    )
    return Evaluator(cfg), budget, str(graphs)


def done_attempts(path):
    """Attempts already recorded: finished ones, and ones given up on after hanging twice."""
    if not path.exists():
        return set()
    return {(r["setup"], r["task"]) for r in map(json.loads, path.read_text().splitlines())
            if r.get("status") == "ok" or r.get("hung")}


def handle_previous_hang(out, attempts_path):
    """If the last session died inside an attempt (pending.json left behind), log the hang.
    First hang: retry. Second hang of the same attempt: record it as an error and move on."""
    pending = out / "pending.json"
    if not pending.exists():
        return
    hang = json.loads(pending.read_text())
    hangs_path = out / "hangs.jsonl"
    with hangs_path.open("a") as f:
        f.write(json.dumps({**hang, "detected": datetime.now(timezone.utc).isoformat()}) + "\n")
    count = sum(1 for r in map(json.loads, hangs_path.read_text().splitlines())
                if (r["setup"], r["task"]) == (hang["setup"], hang["task"]))
    log(f"Previous session hung on {hang['task']} / {hang['setup']} (hang #{count})")
    if count >= 2:
        row = {**hang, "status": "error", "hung": True, "resolved": False,
               "error": f"hung twice (killed after the hang limit both times)"}
        with attempts_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    pending.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=int, required=True, help="Run number (0 = smoke test)")
    p.add_argument("--tasks", nargs="*", help="Task IDs (default: all in selected_tasks.txt)")
    p.add_argument("--setups", nargs="*", default=["A", "B", "C"], choices=list(SETUPS))
    for k, v in KAGGLE.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v)
    p.add_argument("--skip-install", action="store_true", help="Harness already installed")
    p.add_argument("--hang-minutes", type=float, default=15,
                   help="Kill the process if one attempt runs longer (budget is 4 min + ~1.5 min checks)")
    p.add_argument("--startup-minutes", type=float, default=35,
                   help="Kill the process if installing and loading the model takes longer (vLLM's own limit is 20)")
    p.add_argument("--deadline", type=float,
                   help="Unix time; don't start an attempt that can't finish by then (exit code 3)")
    p.add_argument("--attempt-minutes", type=float, default=8,
                   help="Time to reserve per attempt when checking the deadline (measured: ~5.3)")
    p.add_argument("--resume-from", help="Results of an earlier session of this run, copied in if this one has none")
    args = p.parse_args()

    os.environ.update(ENV)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("LOCAL_API_KEY", "EMPTY")
    data_dir, work_dir = Path(args.data_dir), Path(args.work_dir)

    # Startup guard: a hang before the first attempt (install, model load) also ends the process
    faulthandler.dump_traceback_later(args.startup_minutes * 60, exit=True)
    if not args.skip_install:
        install_harness(args.wheelhouse)
    from importlib.metadata import version
    versions = {pkg: version(pkg) for pkg in ("swegemma", "adk_submission", "google-adk", "vllm")}
    assert versions["swegemma"] == HARNESS_VERSION, (
        f"Harness is {versions['swegemma']}, plan uses {HARNESS_VERSION}. Stop and check before running.")
    from swegemma.models import load_tasks
    from swegemma.results import append_task_result

    task_ids = args.tasks or (HERE / "selected_tasks.txt").read_text().split()
    by_id = {t.instance_id: t for t in load_tasks(data_dir / "tasks.jsonl")}
    missing = [t for t in task_ids if t not in by_id]
    assert not missing, f"Unknown task IDs: {missing}"
    order = sorted(task_ids)
    random.Random(ORDER_SEED + args.run).shuffle(order)  # different, repeatable task order per run

    out = work_dir / "results" / f"run{args.run}"
    if args.resume_from and not (out / "attempts.jsonl").exists():
        shutil.copytree(args.resume_from, out, dirs_exist_ok=True)
        log(f"Resumed from {args.resume_from}")
    out.mkdir(parents=True, exist_ok=True)
    empty_dir = work_dir / "no_graphs"
    empty_dir.mkdir(exist_ok=True)
    attempts_path = out / "attempts.jsonl"
    handle_previous_hang(out, attempts_path)
    already = done_attempts(attempts_path)

    agents = prepare_agents(data_dir, work_dir)
    server, models, adapters, gpus = start_server(agents["A"], args.model_dir)
    faulthandler.cancel_dump_traceback_later()
    evaluators = {}
    for key in args.setups:
        evaluators[key] = make_evaluator(key, agents[key], data_dir, out / f"setup_{key}",
                                         models, adapters, empty_dir)

    if (out / "run_config.json").exists():  # keep each session's config
        n = len(list(out.glob("run_config*.json")))
        (out / "run_config.json").rename(out / f"run_config.session{n}.json")
    (out / "run_config.json").write_text(json.dumps({
        "run": args.run, "started": datetime.now(timezone.utc).isoformat(), "versions": versions,
        "model": MODEL, "gpus": gpus, "sandbox": "subprocess", "task_order": order,
        "order_seed": ORDER_SEED + args.run,
        "setups": {k: {"folder": SETUPS[k], "budget": evaluators[k][1], "graph_dir": evaluators[k][2]}
                   for k in evaluators},
    }, indent=2))
    log(f"Run {args.run}: {len(order)} tasks x {len(evaluators)} setups -> {out}")

    stopped_at_deadline = False
    try:
        for pos, task_id in enumerate(order):
            pair = list(evaluators) if pos % 2 == 0 else list(reversed(list(evaluators)))
            for key in pair:
                if (key, task_id) in already:
                    log(f"skip {task_id} / {key} (already recorded)")
                    continue
                if args.deadline and time.time() + args.attempt_minutes * 60 > args.deadline:
                    log(f"Deadline: not starting {task_id} / {key}; stopping so the results are saved")
                    stopped_at_deadline = True
                    return
                evaluator = evaluators[key][0]
                log(f"[{pos + 1}/{len(order)}] {task_id} / setup {key}")
                started = time.time()
                row = {"run": args.run, "setup": key, "task": task_id, "position": pos,
                       "started": datetime.now(timezone.utc).isoformat()}
                (out / "pending.json").write_text(json.dumps(row))
                # Hard limit outside the event loop: the harness's 4-minute timer can't fire if a
                # call never returns. Dumps all stacks (for diagnosis) and exits; see docstring.
                faulthandler.dump_traceback_later(args.hang_minutes * 60, exit=True)
                try:
                    r = asyncio.run(evaluator.evaluate_task(task=by_id[task_id], task_index=pos + 1,
                                                            total_tasks=len(order)))
                    append_task_result(r, out / f"setup_{key}")  # harness's own writer: patch + test output
                    row.update(status="ok", resolved=bool(r.resolved), test_exit_code=r.test_exit_code,
                               tool_calls=r.tool_calls, duration_seconds=round(r.duration_seconds, 1),
                               patch_chars=len((r.agent_patch or "").strip()),
                               agent_error=r.error_message)  # e.g. "Agent exceeded session timeout"
                except Exception as e:  # record and keep going; one broken task mustn't end the run
                    row.update(status="error", error=f"{type(e).__name__}: {e}")
                faulthandler.cancel_dump_traceback_later()
                row["wall_seconds"] = round(time.time() - started, 1)
                with attempts_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                (out / "pending.json").unlink()
                log(f"  -> {row.get('status')} resolved={row.get('resolved')} "
                    f"calls={row.get('tool_calls')} {row['wall_seconds']}s")
    finally:
        faulthandler.dump_traceback_later(180, exit=True)  # stopping the server mustn't hang either
        server.stop()
        faulthandler.cancel_dump_traceback_later()
        if stopped_at_deadline:
            sys.exit(DEADLINE_EXIT)
    log("Done.")


if __name__ == "__main__":
    main()
