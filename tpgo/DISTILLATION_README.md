# TPGO Distillation Compliance README

## Status

Current implementation is compliant with the stated restriction:

> Do not directly distill on the 200 SimpleVQA, 2wiki, or leaderboard benchmark items.

This repository currently uses **static, generic strategy distillation guidance** only. It does not perform online distillation, does not train on benchmark examples, and does not persist benchmark-derived memory.

## What This Is

The current distillation-related module is:

- `tpgo/distilled_strategy.py`

It contains a generic search-process prompt for text web-search QA tasks. The prompt describes broad tactics such as:

- identify the requested answer role;
- split a question into clue chains;
- search rare source-side facts first;
- verify candidates against constraints;
- avoid overlong all-clue queries;
- output a short answer span near the step limit.

The guidance is **task-agnostic** and does not contain:

- benchmark questions;
- benchmark answers;
- benchmark trajectories;
- dataset-specific entities;
- SimpleVQA examples;
- 2wiki examples;
- leaderboard problem content;
- learned memory from a prior benchmark run.

## What This Is Not

This is not supervised fine-tuning, preference tuning, or example-level distillation on the benchmark set.

The current implementation does **not**:

- call a larger model with any of the 200 benchmark questions;
- call a larger model with benchmark trajectories;
- call a larger model with benchmark answers;
- generate labels, rationales, hints, or memories from the benchmark set;
- reuse information learned from one benchmark run to improve a later run on the same benchmark;
- write any benchmark-derived strategy memory to disk;
- expose image tasks or leaderboard content to a stronger model.

## Compliance Boundary

The key compliance rule is:

| Action | Allowed? | Reason |
|---|---:|---|
| Use a static generic search-strategy prompt | Yes | It is not derived from the 200 test items and contains no benchmark content. |
| Use larger-model guidance produced from public, non-test development examples | Yes, if documented | The examples must not overlap with SimpleVQA, 2wiki, or leaderboard test data. |
| Use 32B to judge or rewrite trajectories from the 200 benchmark items | No | This would expose benchmark trajectories and become test-set distillation/evolution. |
| Use 32B to generate hints for the 200 benchmark questions | No | This would directly use test items for distillation. |
| Store memory learned from one benchmark run and reuse it on the same benchmark | No | This violates the one-round test-time evolution boundary. |
| Tune prompts using observed labels or gold answers from the benchmark | No | This is direct benchmark overfitting. |

## Runtime Isolation

The mixed benchmark runner keeps text and image tasks separated:

- `run_benchmark_parallel_mixed.py`

Default split:

| Index Range | Runner | Distillation Guidance | TPGO Router | Online 32B Critic |
|---|---|---:|---:|---:|
| `0-49` text tasks | `task_runner_plan_react_negcrit.py` | Optional static prompt only | Yes | Should be disabled for compliant distillation runs |
| `50-99` image tasks | `task_runner_plan_react.py` | No | No | No |

For compliant static-distillation runs, use:

```bash
TPGO_DISTILLED_STRATEGY=1
NEG_CRITIC_ENABLED=0
```

`NEG_CRITIC_ENABLED=0` ensures the stronger critic model is not called on benchmark questions or trajectories.

## Recommended Compliant Run Command

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210830/Auto-evolving-agent

TPGO_DISTILLED_STRATEGY=1 \
NEG_CRITIC_ENABLED=0 \
MAX_STEPS=14 \
NEG_FORCE_ANSWER_STEP=13 \
MAX_SEARCH_CALLS=10 \
MAX_BLOCKED_SEARCHES=3 \
BAD_STREAK_REPLAN=2 \
ROUTE_REPLAN_COOLDOWN=2 \
NEG_EVAL_AFTER_TOOL=0 \
NEG_EVAL_EVERY_STEPS=2 \
NEG_EVAL_MIN_STEP=2 \
python run_benchmark_parallel_mixed.py \
  --group 7 \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/benchmark.csv \
  --start 0 \
  --end 100 \
  --image-start 50 \
  -c 5
```

## Audit Checklist

Before submitting a run that uses `TPGO_DISTILLED_STRATEGY=1`, verify:

1. `NEG_CRITIC_ENABLED=0` was set.
2. No process called a larger model with benchmark questions, trajectories, answers, screenshots, or image content.
3. No benchmark-derived memory file was written or reused.
4. `tpgo/distilled_strategy.py` still contains only generic process guidance.
5. The output directory is from `mixed_negcrit_text_planreact_image`, confirming image tasks used the isolated old runner.
6. There are no answer-format artifacts such as `<answer>`, `[LOW_CONFIDENCE]`, `Correction Note`, or `KEY FINDINGS` in `group_7.csv`.

Suggested format check:

```bash
LATEST=$(ls -dt results/benchmark/mixed_negcrit_text_planreact_image/* | head -1)
export LATEST

python - <<'PY'
import csv, os, re, sys
csv.field_size_limit(sys.maxsize)
p = os.path.join(os.environ["LATEST"], "group_7.csv")
rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
bad = []
empty = []
for i, row in enumerate(rows):
    ans = (row.get("answer") or "").strip()
    if not ans:
        empty.append(i)
    if re.search(
        r"LOW_CONFIDENCE|<answer|Correction Note|Revised Plan|Constraint Ledger|"
        r"Unable|insufficient|I need to|not definitively|cannot|KEY FINDINGS|"
        r"Based on my search|Internal Correction",
        ans,
        re.I,
    ):
        bad.append((i, ans[:180].replace("\n", " ")))
print("result_dir:", os.environ["LATEST"])
print("rows:", len(rows))
print("empty:", len(empty), empty[:20])
print("bad_format_or_prose:", len(bad))
for item in bad[:30]:
    print(item)
PY
```

## Future Distillation Rules

If stronger-model distillation is added later, it must satisfy all of the following:

1. Training or prompt-distillation data must come from a separate development set or synthetic tasks, not from the 200 SimpleVQA, 2wiki, or leaderboard benchmark items.
2. The larger model must not see benchmark questions, images, trajectories, intermediate tool results, final answers, or labels.
3. Distilled artifacts must be frozen before benchmark evaluation.
4. No benchmark-run-derived memory may be reused in a later benchmark run.
5. Any distilled prompt, rule, or model checkpoint must include a documented data source and exclusion statement.

## Summary

The current `TPGO_DISTILLED_STRATEGY=1` implementation is compliant because it is a static, generic strategy prompt and not benchmark-derived distillation. It should be used with `NEG_CRITIC_ENABLED=0` for benchmark runs to ensure no larger model participates in answering or trajectory evaluation.
