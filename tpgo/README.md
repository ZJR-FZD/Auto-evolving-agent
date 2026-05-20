# TPGO Reflection Module

This folder contains a self-contained TPGO-oriented add-on for the Harness
agent project. It does not change the original runner by default.

## What It Adds

- Trajectory efficiency metrics: steps, search calls, duplicate searches,
  low-signal tool results, critic BAD frequency, approximate context tokens,
  runtime.
- Reflection memory extraction from failed trajectories.
- Mermaid trajectory graphs for debugging and presentation.
- Initial Textual Parameter Graph (TPG) config for the current ReAct harness.
- Soft online constraint guidance used by `plan_react_negcrit`: constraint
  ledger prompting, critic alignment rules, reflection additions, answer-role
  checks, and duplicate-query pivot handling.
- A small benchmark ablation runner with hard timeouts.

## Quick Commands

Analyze existing trajectories:

```powershell
python -m tpgo.tpgo_tools analyze --traj-dir trajectories\benchmark\plan_react_negcrit\20260520_015253 --out-dir tpgo\outputs\negcrit_100
```

Render one trajectory graph:

```powershell
python -m tpgo.tpgo_tools graph --traj trajectories\benchmark\plan_react_negcrit\20260520_015253\bench_095.jsonl --out tpgo\outputs\bench_095.mmd
```

Initialize a TPG config:

```powershell
python -m tpgo.tpgo_tools init-tpg --out tpgo\current_tpg.json
```

Run a benchmark smoke ablation on the first 3 rows:

```powershell
python -m tpgo.ablation_runner --dataset ..\datasets\benchmark.csv --end 3 --timeout-seconds 480 --modes basic plan_react plan_react_negcrit
```

For a faster harness smoke test, cap step budget:

```powershell
python -m tpgo.ablation_runner --dataset ..\datasets\benchmark.csv --end 3 --timeout-seconds 480 --max-steps 8 --modes plan_react_negcrit
```

The ablation runner writes `ablation_report.json` even when the dataset,
LLM server, or critic server is unavailable. This makes CI/smoke-test
failure modes explicit.

Expected runtime dependencies for live benchmark runs:

- `..\datasets\benchmark.csv`
- main OpenAI-compatible model endpoint, default `http://127.0.0.1:8000/v1`
- for `plan_react_negcrit`, critic endpoint, default `http://127.0.0.1:8001/v1`
- search/browser service environment variables required by the original harness

## GitHub Submission Notes

Recommended files to commit:

- `tpgo/__init__.py`
- `tpgo/tpgo_tools.py`
- `tpgo/ablation_runner.py`
- `tpgo/README.md`

Generated files under `tpgo/outputs/` and `tpgo/ablation_runs/` are ignored.
