# Given a Map, the Agent Doesn't Look

Code and results for a paper-track entry to Kaggle's **Gemma 4 Developer Agent** competition. Do the harness's code-graph tools help a small local model (Gemma 4 31B, 4-bit) fix real GitHub issues?

**Answer: no.** We ran 4 setups × 26 tasks × 3 runs, for 312 attempts. No setup beat grep-only search, and every gap was smaller than the run-to-run spread. The agent rarely called the graph tools unless it was forced to. Finding the code wasn't the bottleneck: fixing it was.

- **Paper:** [paper/paper.md](paper/paper.md)
- **Experiment plan:** [paper/experiment-plan.md](paper/experiment-plan.md). The hypotheses and decision rules were locked before any run, and the plan keeps a dated change log.

## What's here
| Path | What |
|---|---|
| `experiment/run_experiment.py` | Runs every task under each setup, with hang guards and resume |
| `experiment/kaggle/run.ipynb` | Kaggle notebook that calls the run script (4× L4 GPUs) |
| `experiment/validate_tasks.py`, `kaggle_validate/` | Checks that each task is solvable in the sandbox (the reference fix passes) |
| `setups/*.diff` | The 4 agent setups, as diffs against the official sample agent |
| `analysis/select_tasks.py` | Selects the tasks (seed 20261016) |
| `analysis/analyze_traces.py` | Turns traces into per-attempt rows and failure categories |
| `analysis/week5_results.py` | Builds every table and figure in the paper |
| `analysis/output/runs1-3/` | One row per attempt (312), plus the hypothesis sheets |
| `analysis/output/week5/` | Results tables and figures |

## What's not here, and why
Competition data may not be shared with non-participants. These are left out:
- **Raw traces:** they contain task problem statements and hidden tests.
- **The official sample agent and harness manual:** both come from the competition's Data tab.
- **One per-attempt column:** the `evidence` column of `attempts.csv` is dropped because it names hidden tests.

Participants can rebuild everything from the competition data.

## Reproduce
1. Join the competition and download its data, including `sample_submission/` and `tasks.jsonl`.
2. Build the setups:
   ```
   sh setups/make_setups.sh <path>/sample_submission
   ```
3. Run on Kaggle: `experiment/build_bundle.sh`, then push `experiment/kaggle/run.ipynb` (set `RUNS` and `SETUPS` in cell 1).
4. Analyze:
   ```
   python analysis/analyze_traces.py <results folder>/run<N>   # one per run
   python analysis/week5_results.py analysis/output/runs1-3/attempts.csv --out analysis/output/week5
   ```

Harness: swegemma 0.2.11. Model: `gemma-4-31b-it-qat-w4a16-ct`, temperature 0.2. Budget per attempt: 4 minutes and 100 tool calls.

## License
Apache 2.0.
