# Experiment Plan

**Status: 🔒 Locked Oct 17, 2026**, before any experiment runs. Built from your Week 2 homework; reviewed and approved by you.

**Rule:** once locked, changes are allowed only with a dated note saying what changed and why. That record is what makes the analysis "decided in advance".

---

## 1. Question

> When do code-graph tools help a small coding agent fix bugs, and when they don't, why not?

## 2. Hypotheses

| # | Hypothesis | Measure | Supported if | Against it if |
|---|---|---|---|---|
| H1 | Graph tools help **more on multi-file fixes** than on single-file fixes | **Gain** = graph's resolution rate − grep's, per group. Multi-file = the reference fix changes 2+ library files (not tests, docs examples or scripts) | Multi-file gain > single-file gain **in all 3 runs** | Multi-file gain is no bigger, or the order flips between runs |
| H2 | Graph tools **hurt on single-file fixes** through over-exploration | On single-file tasks: resolution rate (harm), plus average tool calls and minutes per task (mechanism), graph vs. grep | Graph's resolution rate is **lower** than grep's in all 3 runs, **and** graph uses more calls and time | Graph's resolution rate is equal or higher |
| H3a | Failures fall into **a few repeating types** | Each failed attempt (both setups) gets 1 of 6 categories below, by script; 10 hand-checked | ≥ 80% of failures fit the 6 categories, in every run | "Other" is > 20% in any run |
| H3b | Graph tools **shift the failure mix** | Share of failures per category, per setup, per run | With graph, "couldn't find the code" (categories 2 + 4) is a **smaller** share **and** "found it, never edited" (3) a **larger** share, in all 3 runs | Graph's mix looks like grep's, or the direction flips between runs |
| H4 | The agent **misuses `search_similar_code`** with sentences, and it hurts | Setup B only (the tool's only home). Share of calls that are sentences (query has a space); zero-result rate for sentences vs. code names, per run | **More than half** of calls are sentences, **and** their zero-result rate is **≥ 20 points higher** than names', in all 3 runs | Most calls use code names, or the zero-result rates are within 20 points |

- **Why "multi-file" and not graph distance (H1):** the graph has blind spots (missed `self.` calls, validators), so distance is sometimes wrong or missing. Files changed comes straight from the reference fix and is always reliable. Graph distance is reported as an extra, along with how often it's missing.
- **H3 categories** (decided Oct 16, before any runs). Checked in order; first match wins:
  1. **Crash or tool error:** run ended in an error, or tool calls were broken
  2. **Never found the code:** never opened a file the reference fix changes
  3. **Found it, never edited:** opened a reference file, made no edit
  4. **Edited the wrong place:** edits, but none in a reference file
  5. **Right place, wrong fix:** edited a reference file; fail-to-pass tests still fail
  6. **Fixed it, broke something:** fail-to-pass pass; pass-to-pass fail
  - **Other:** none of the above. **"Hit the time or call limit"** is a yes/no flag on every failure, not a category (it overlaps all of them).
- **H4 minimum:** if setup B makes **fewer than 20** `search_similar_code` calls over 3 runs, H4 is reported as "not testable", with counts.
- **Time is reported, not judged:** a setup that fixes more bugs but takes longer isn't a failure of H1.

## 3. Tasks

- **26 tasks:** 13 multi-file + 13 single-file, picked on purpose so the two groups are equal. **The same 26 tasks for both setups and all runs.** (Was 15 + 15; changed Oct 21, see change log.)
- **Repos:** the pool is FastAPI 67, Rich 48, Requests 13, HTTPX 1. Picked in roughly those proportions; HTTPX is too small to report on its own.
- **Exclusion rule (applied before any runs, from issue text only):** exclude a task if its description is empty or template-only, **or** it names no file, folder, function, class or error message **and** has fewer than 10 words of real text. Links don't count as text. Also exclude tasks whose reference fix changes no library file. **Added Oct 21:** exclude tasks that aren't solvable in the sandbox, meaning their tests don't fail with no patch or don't pass with the reference fix (validation of all 129).
- **How chosen:** [`analysis/select_tasks.py`](../analysis/select_tasks.py), seed **20261016**. Within each group, repos get seats in proportion to their eligible pool. Every task's features and exclusion reason: [`task_features.csv`](../analysis/output/task_features.csv); the 26: [`selected_tasks.txt`](../analysis/output/selected_tasks.txt).
- **Result (Oct 16):** 25 excluded (20 Rich PRs that say little more than "Fixes <link>"; 1 seven-word FastAPI PR; 3 docs-example fixes; 1 empty template). Pool 104. Picked: multi = FastAPI 10, Rich 4, Requests 1; single = FastAPI 9, Rich 4, Requests 2. ✅ Hand-checked Oct 17: you agreed with all 5 borderline decisions, so the rule stands unchanged.
- **Result (Oct 21, re-selection):** only 50 of the 104 are solvable: multi = Rich 7, FastAPI 6; single = Rich 18, FastAPI 19; Requests and HTTPX 0. Picked: all 13 multi; single = FastAPI 7, Rich 6.

## 4. Setups

| Setup | What the agent has | How it's built |
|---|---|---|
| **A: grep only** (baseline) | All tools except the 3 graph tools | Remove graph tools from `agent.yaml` and `sub_agents/code_analyzer.yaml`; remove graph mentions from `prompts/system.md` and `prompts/analyzer.md`; run on a copy of the data **without** `graphs/` and `embeddings/` (so the harness doesn't advertise them) |
| **B: grep + graph** | The sample agent as provided | Unchanged sample submission, with graph files present |
| **C: grep + graph, prompted** (added Oct 24; see change log) | Same as B, plus line 11 of `system.md` | Exactly the official sample, with our 4-min / 100-call budget |
| **D: graph required** (added Oct 25, exploratory; see change log) | Same as C, but step 1 of `system.md` **requires** `search_similar_code` as the first call and `get_code_neighbors` as the second | Copy of C; only step 1 of `system.md` changed. Run alone (`SETUPS = ["D"]`), 3 runs |

- **The one difference:** access to graph tools.
- **Same in both:** model, LoRA adapters, budget (4 min, 100 tool calls), hardware (4× L4), harness version.
- ✅ **Line 11 of `system.md`** tells the agent to prefer `search_similar_code` *over* `grep`. **Decided Oct 16: remove it from both setups**, so the setups differ only in tools and H4 misuse can't be blamed on the prompt. Setup B then differs from the official sample by this one line; the paper says so.
- **"Graph only" is not a setup:** `grep` lives inside `run_command`, which the agent also needs for tests and Python. It can't be removed cleanly.

## 5. Runs and budget

- **3 runs per setup**, each run covering all 26 tasks: 26 × 3 × 3 = **234 attempts** (A and B: 156; C: 78).
- **Compute (measured on the Oct 19 smoke test):** ~5.3 min per attempt including setup and tests, plus ~9 min to load the model per session. One run = 52 attempts ≈ **4.75 GPU hours ≈ 9.5 Kaggle quota hours** (4× L4 counts double). 3 runs ≈ 29 quota hours.
- **All 3 runs on Kaggle (changed Oct 20; see change log).** Same machine type for every run. Runs 1–2 in one weekly quota, run 3 after the weekly reset. Setups alternate within each run.
- **Setup D (added Oct 25):** 26 × 3 = 78 more attempts, run alone. Measured: one setup × 26 tasks ≈ 1.7 h (≈ 3.5 min per attempt + model load) ≈ **3.4 quota hours per run**, ~10 for all 3.
- **One task at a time** unless we measure that running 2 at once doesn't slow the agent.
- **Cost:** $0. **Risk:** Kaggle queues (2+ hours seen Sep 27; none on Oct 19). The run script resumes where it stopped, so a cut-off session costs time, not data.
- **Fallback if Kaggle can't finish run 3 by Oct 31:** report 2 runs (the Week 3 rule), or rent AWS `g6.12xlarge` (4× L4, the same GPU), noted in the change log.

## 6. Metrics

- **Primary:** resolution rate (share of tasks whose hidden tests pass).
- **Secondary:** average minutes per task; average tool calls per task; graph tool calls per task; failure category (H3); `search_similar_code` sentence share and zero-result rates (H4).

## 7. Analysis plan (decided now)

- **Paired comparison:** the same task under both setups. Per run, compute each setup's resolution rate, overall and per group.
- **A difference counts as real** only if it points the same way in **all 3 runs** and is bigger than the spread between runs of the same setup.
- **Report every run,** not just the average. Show the per-task table in an appendix.
- **Sensitivity check:** headline results with and without the excluded tasks.
- **Blind spots:** report how many fixes sit in functions with no graph links.

## 8. Limitations we already know

- 2 repos (FastAPI and Rich); Python only. Requests and HTTPX tasks can't be solved in the official sandbox (missing test server), so none are in our list.
- Over half the benchmark (54 of 104 eligible tasks) is unsolvable in the official sandbox, mostly from missing test libraries. We report this; our 26 are all checked solvable.
- One model (Gemma 4 31B, 4-bit) and a tight 4-minute budget.
- The graph only records calls; it misses `self.` calls and validators.
- We removed one line of the organizers' prompt (it steered the agent toward `search_similar_code`), so setup B is not exactly the official sample.
- Kaggle assigns a machine per session; runs may land on different physical machines of the same type. Run-to-run spread includes that.

## 9. If graph tools make no difference

The paper still reports a clear result: *as provided, graph tools don't help a small agent under a tight time budget*. The trace analysis explains why (blind spots, tool misuse, time spent thinking), and the analyzer is released as a resource. A clean null result with an explanation is publishable.

---

## Change log (after lock)

Every change from here on gets a dated line: what changed, why, and whether any results had been seen yet.

| Date | Change | Why | Results seen? |
|---|---|---|---|
| Oct 19 | **Sandbox = `subprocess` on both Kaggle and Google Cloud** (the plan didn't name one). Corrected the Google Cloud rationale, which said the harness needs Docker | The organizers' notebook uses `subprocess`, since Kaggle has no Docker. Using it everywhere keeps runs 1–3 comparable | No |
| Oct 20 | **All 3 runs on Kaggle;** Google Cloud dropped. §5 rewritten with measured times | Google auto-denied both GPU quota requests (Oct 17; new account). Kaggle's weekly quota fits 3 runs over 2 weeks, at no cost, on one machine type, which is better for comparing runs | No (only the smoke test) |
| Oct 25 | **Setup D added (exploratory): the prompt requires graph tools first.** Decided before any D results. **Check first:** did the agent comply? Measure = share of D attempts whose first call is `search_similar_code` (column `first_calls`); if under 50% in run 1, stop and report "prompting can't make this model use the tools". **Then:** D vs. A, paired by task, over 3 runs: D helps if it resolves more than A in all 3 runs; H1 and H3b are re-checked with D in place of B. H4's sentence share isn't meaningful in D (the prompt asks for code names); only the zero-result rate is reported. **Limits:** D runs in separate sessions from A, B, C (a week later, same harness, model and tasks); 14 of 26 tasks were never solved by any setup, so only ~12 tasks can show a difference | Runs 1–3 of A, B, C: 8 graph calls in 156 attempts, so A vs. B and A vs. C can't say whether the tools help when used | **Yes:** all of runs 1–3 (A 29, B 29, C 26 of 78). The paper labels D as a follow-up |
| Oct 24 | **Setup C added: B with line 11 restored (= the official sample).** Comparisons: A vs. C (do graph tools help when used?), B vs. C (effect of the prompt line). A vs. B is kept and reported as two setups that behave the same (a noise floor). C's run 1 is a separate session after A/B's; runs 2–3 run A, B, C together. H4 is measured on B and C | In run 1, B made **0 graph tool calls in 26 attempts**: without line 11 the agent never uses the graph tools, so A vs. B can't test H1 or H4 | **Yes:** run 1 of A and B (A 8/26, B 10/26, 0 graph calls). The paper labels C as a follow-up added after this result |
| Oct 21 | **Tasks re-selected: 26 (13 multi + 13 single), only from tasks solvable in the sandbox.** Same rule and seed, plus a solvability check; group size capped by the 13 solvable multi-file tasks. Run 1 restarted on the new list; its first 8 attempts (old list) archived in `runs/2026-10-20-pilot-old30/` and not used | Validation found 16 of the old 30 (and 54 of 104 overall) can't pass even with the reference fix. Those are guaranteed failures in both setups and would hide any difference | Partly: 8 attempts of run 1 (4 tasks). The new list comes from the seed and the validation check alone, not from those results |
| Oct 19 | **Harness pinned to `swegemma` 0.2.11** (wheelhouse of Oct 12); the run script stops if the version differs | "Same harness version" in §4 needs a number to check against | No |
