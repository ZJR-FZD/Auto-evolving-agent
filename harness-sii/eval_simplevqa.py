"""
Batch evaluation entry point for the local SimpleVQA subset.

This script wraps task_runner.run_task, which evaluates one agent task at a
time. It writes one JSON object per sample to the predictions file and prints a
small aggregate summary at the end.

python harness-sii/eval_simplevqa.py \
    --dataset datasets/simpleVQA/SimpleVQA.jsonl \
    --data-root datasets/simpleVQA \
    --output-root harness-sii/runs \
    --run-name simplevqa_raw \
    --llm-url http://127.0.0.1:8000/v1 \
    --model Qwen3.5-9B \
    --max-steps 20 \
    --concurrency 2 \
    --result-format minimal \
    --overwrite
"""

import argparse
import base64
import concurrent.futures
import re
import time
from pathlib import Path

from eval_utils import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_ROOT,
    append_jsonl,
    concat_trajectories,
    format_output_row,
    load_done_by_index,
    patch_task_runner_tool_args,
    prepare_run_dir,
    read_jsonl,
    run_paths,
    write_jsonl,
    write_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    REPO_ROOT / "datasets" / "simpleVQA"
    if (REPO_ROOT / "datasets" / "simpleVQA").exists()
    else REPO_ROOT / "simpleVQA"
)
DEFAULT_DATA_FILE = DEFAULT_DATA_ROOT / "SimpleVQA.jsonl"


def _normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》【】]", "", text)
    return text


def _is_exact(row: dict) -> bool:
    return _normalize_answer(row.get("pred", row.get("prediction", ""))) == _normalize_answer(
        row.get("answer", "")
    )


def _contains_gold(row: dict) -> bool:
    norm_pred = _normalize_answer(row.get("pred", row.get("prediction", "")))
    norm_gold = _normalize_answer(row.get("answer", ""))
    return bool(norm_gold and norm_gold in norm_pred)


def _build_instruction(sample: dict) -> str:
    question = sample["question"]
    language = sample.get("language", "")
    if language == "EN":
        return (
            "Answer the visual question. You may use the available tools when external "
            "knowledge or image search is needed. Return only the final short answer.\n"
            f"Question: {question}"
        )
    return (
        "请回答这道视觉问答题。需要外部知识或图搜时可以调用工具。最终只输出简短答案，不要解释。\n"
        f"问题：{question}"
    )


def _sample_metadata(sample: dict) -> dict:
    vqa_category = sample.get("vqa_category")
    if not isinstance(vqa_category, dict):
        vqa_category = {}
    return {
        "language": sample.get("language", ""),
        "source": sample.get("source", ""),
        "task_category": vqa_category.get("task_category", ""),
        "subject_category": vqa_category.get("subject_category", ""),
        "entity_class": vqa_category.get("entity_class", ""),
        "original_category": sample.get("original_category", ""),
    }


def _sample_to_task(sample: dict, data_root: Path, config: dict | None = None) -> dict:
    data_id = int(sample["data_id"])
    image_path = data_root / sample["image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for data_id={data_id}: {image_path}")

    with image_path.open("rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    config = config or {}
    enable_evolution = str(config.get("agent_mode", "evolved")).lower() != "raw"
    allow_gold_feedback = bool(config.get("allow_gold_feedback", False))
    task = {
        "id": f"simplevqa_{data_id}",
        "instruction": _build_instruction(sample),
        "image_b64": image_b64,
        "image_url": sample.get("image_url"),
        "task_family": "simplevqa",
        "metadata": _sample_metadata(sample),
        "memory_dir": config.get("memory_dir"),
        "memory_update_mode": "disabled" if not enable_evolution else config.get("memory_update_mode", "heuristic"),
        "enable_memory": enable_evolution,
        "enable_reflection": enable_evolution and bool(config.get("enable_reflection", True)),
        "allow_gold_feedback": allow_gold_feedback,
    }
    if allow_gold_feedback:
        task["gold_answer"] = sample.get("answer", "")
    return task


def _result_row(sample: dict, result: dict) -> dict:
    prediction = result["answer"]
    gold = sample.get("answer", "")
    norm_pred = _normalize_answer(prediction)
    norm_gold = _normalize_answer(gold)
    data_id = int(sample["data_id"])
    instruction = _build_instruction(sample)
    return {
        "index": data_id,
        "instruction": instruction,
        "image": sample.get("image"),
        "answer": gold,
        "pred": prediction,
        "data_id": data_id,
        "task_id": f"simplevqa_{data_id}",
        "question": sample.get("question"),
        "prediction": prediction,
        "exact_match": norm_pred == norm_gold,
        "contains_gold": bool(norm_gold and norm_gold in norm_pred),
        "steps": result["steps"],
        "trajectory_path": result["trajectory_path"],
        "stats": result.get("stats", {}),
        "signals": result.get("signals", {}),
        "memory_recalled": result.get("memory_recalled", []),
        "memory_written": result.get("memory_written", []),
        "image_url": sample.get("image_url"),
        "source": sample.get("source"),
        "language": sample.get("language"),
    }


def _error_row(sample: dict, error: str) -> dict:
    data_id = int(sample["data_id"])
    instruction = _build_instruction(sample)
    return {
        "index": data_id,
        "instruction": instruction,
        "image": sample.get("image"),
        "answer": sample.get("answer", ""),
        "pred": "",
        "data_id": data_id,
        "task_id": f"simplevqa_{data_id}",
        "question": sample.get("question"),
        "prediction": "",
        "exact_match": False,
        "contains_gold": False,
        "steps": 0,
        "trajectory_path": "",
        "image_url": sample.get("image_url"),
        "source": sample.get("source"),
        "language": sample.get("language"),
        "error": error,
    }


def _remove_stale_trajectory(traj_dir: Path, task_id: str) -> None:
    traj_path = traj_dir / f"{task_id}.jsonl"
    if traj_path.exists():
        traj_path.unlink()


def _clear_pending_trajectories(traj_dir: Path, pending: list[tuple[int, dict]]) -> None:
    for _, sample in pending:
        _remove_stale_trajectory(traj_dir, f"simplevqa_{int(sample['data_id'])}")


def _run_one_sample(payload: tuple[int, int, dict, dict]) -> tuple[int, dict]:
    """Run one sample. Kept top-level so ProcessPoolExecutor can pickle it."""
    index, total, sample, config = payload
    data_id = int(sample["data_id"])

    import task_runner
    from agent_memory import ShortTermMemory

    patch_task_runner_tool_args(task_runner)

    task = _sample_to_task(sample, Path(config["data_root"]), config)
    _remove_stale_trajectory(Path(config["traj_dir"]), task["id"])
    print(f"[{index}/{total}] run data_id={data_id} task_id={task['id']}", flush=True)

    enable_retry = config.get("enable_confidence_retry", False)
    if enable_retry:
        result = task_runner.run_task_with_retry(
            task,
            max_steps=int(config["max_steps"]),
            llm_base_url=str(config["llm_url"]),
            model_name=str(config["model"]),
            trajectory_dir=str(config["traj_dir"]),
        )
    else:
        result = task_runner.run_task(
            task,
            max_steps=int(config["max_steps"]),
            llm_base_url=str(config["llm_url"]),
            model_name=str(config["model"]),
            trajectory_dir=str(config["traj_dir"]),
        )
    return index, _result_row(sample, result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SimpleVQA with harness-sii/task_runner.py")
    parser.add_argument("--dataset", "--data-file", dest="data_file", type=Path, default=DEFAULT_DATA_FILE, help="SimpleVQA JSONL file")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for local images")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for timestamped run outputs")
    parser.add_argument("--run-name", default=None, help="Run directory name; defaults to simplevqa_YYYYMMDD_HHMMSS")
    parser.add_argument("--out", type=Path, default=None, help="Deprecated: explicit prediction JSONL path")
    parser.add_argument("--traj-dir", type=Path, default=None, help="Deprecated: explicit trajectory directory")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_BASE_URL, help="SGLang OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Served model name")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum agent loop steps per sample")
    parser.add_argument("--start", type=int, default=None, help="Start index in the dataset")
    parser.add_argument("--end", type=int, default=None, help="End index in the dataset, exclusive")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N samples")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument(
        "--agent-mode",
        choices=("raw", "evolved"),
        default="evolved",
        help="raw disables reflection and memory; evolved enables the self-evolution modules",
    )
    parser.add_argument("--memory-dir", type=Path, default=None, help="Long-term memory directory/path; default is <run_dir>/memory")
    parser.add_argument(
        "--memory-update-mode",
        choices=("heuristic", "gold", "disabled"),
        default="heuristic",
        help="heuristic uses failure signals only; gold may use labels and is for offline training/open eval only",
    )
    parser.add_argument("--allow-gold-feedback", action="store_true", help="Expose gold answer to the memory updater for offline evolution")
    parser.add_argument("--disable-reflection", action="store_true", help="Disable reflection hints while keeping memory recall if evolved")
    parser.add_argument(
        "--result-format",
        choices=("full", "minimal"),
        default="minimal",
        help="full keeps metrics/debug fields; minimal writes only index/instruction/image/answer/pred",
    )
    parser.add_argument("--enable-confidence-retry", action="store_true", help="Enable LLM-as-Judge confidence retry for low-confidence answers")
    parser.add_argument("--resume", action="store_true", help="Resume an existing --run-name by skipping indices already in predictions.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing --run-name before running")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started_at = time.time()

    all_samples = read_jsonl(args.data_file)
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(all_samples)
    else:
        start = args.offset
        end = len(all_samples) if args.limit is None else args.offset + args.limit
    samples = all_samples[start:end]

    run_dir = prepare_run_dir(
        "simplevqa",
        args.output_root,
        args.run_name,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    paths = run_paths(run_dir)
    out_path = args.out or paths["predictions"]
    progress_path = paths["progress"]
    traj_dir = args.traj_dir or paths["trajectory_dir"]
    traj_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and args.overwrite:
        out_path.unlink()
    elif out_path.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {out_path}. Use --resume or --overwrite.")

    if progress_path.exists() and args.overwrite:
        progress_path.unlink()

    metadata = {
        "dataset": "simplevqa",
        "dataset_path": args.data_file,
        "data_root": args.data_root,
        "run_dir": run_dir,
        "prediction_path": out_path,
        "progress_path": progress_path,
        "trajectory_dir": traj_dir,
        "trajectory_jsonl": paths["trajectories"],
        "selection": {"start": start, "end": end, "total_dataset_rows": len(all_samples)},
        "llm_url": args.llm_url,
        "model": args.model,
        "max_steps": args.max_steps,
        "concurrency": max(1, int(args.concurrency)),
        "agent_mode": args.agent_mode,
        "memory_dir": args.memory_dir or (run_dir / "memory"),
        "memory_update_mode": args.memory_update_mode,
        "allow_gold_feedback": bool(args.allow_gold_feedback),
        "enable_reflection": not args.disable_reflection,
        "result_format": args.result_format,
        "started_at": started_at,
        "args": args,
    }
    write_metadata(paths["metadata"], metadata)

    done = {}
    if args.resume:
        done = load_done_by_index(progress_path) or load_done_by_index(out_path)
    new_results: list[dict] = []
    pending: list[tuple[int, dict]] = []

    for index, sample in enumerate(samples, start=1):
        data_id = int(sample["data_id"])
        if data_id in done:
            print(f"[{index}/{len(samples)}] skip data_id={data_id} (already done)")
            continue
        pending.append((index, sample))

    if args.overwrite:
        _clear_pending_trajectories(traj_dir, pending)

    config = {
        "data_root": str(args.data_root),
        "traj_dir": str(traj_dir),
        "llm_url": args.llm_url,
        "model": args.model,
        "max_steps": args.max_steps,
        "agent_mode": args.agent_mode,
        "memory_dir": str(args.memory_dir or (run_dir / "memory")),
        "memory_update_mode": args.memory_update_mode,
        "allow_gold_feedback": bool(args.allow_gold_feedback),
        "enable_reflection": not args.disable_reflection,
        "enable_confidence_retry": bool(args.enable_confidence_retry),
    }

    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        for index, sample in pending:
            try:
                _, row = _run_one_sample((index, len(samples), sample, config))
            except Exception as exc:  # noqa: BLE001
                row = _error_row(sample, f"{type(exc).__name__}: {exc}")
                print(f"[{index}/{len(samples)}] error data_id={sample.get('data_id')}: {row['error']}")
            append_jsonl(progress_path, row)
            append_jsonl(out_path, format_output_row(row, args.result_format))
            new_results.append(row)
    else:
        print(f"Running with concurrency={concurrency}")
        payloads = [
            (index, len(samples), sample, config)
            for index, sample in pending
        ]
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=concurrency)
        future_to_sample: dict[concurrent.futures.Future, dict] = {}
        shutdown_called = False
        try:
            future_to_sample = {
                executor.submit(_run_one_sample, payload): payload[2]
                for payload in payloads
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    _, row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = _error_row(sample, f"{type(exc).__name__}: {exc}")
                    print(f"error data_id={sample.get('data_id')}: {row['error']}")
                append_jsonl(progress_path, row)
                append_jsonl(out_path, format_output_row(row, args.result_format))
                new_results.append(row)
        except KeyboardInterrupt:
            print("Interrupted, cancelling pending tasks...")
            for future in future_to_sample:
                future.cancel()
            for process in getattr(executor, "_processes", {}).values():
                process.terminate()
            shutdown_called = True
            executor.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(130)
        finally:
            if not shutdown_called:
                executor.shutdown(wait=True, cancel_futures=True)

    all_rows = list(done.values()) + new_results
    all_rows.sort(key=lambda row: row.get("data_id", row.get("index", 0)))
    write_jsonl(out_path, (format_output_row(row, args.result_format) for row in all_rows))
    concat_trajectories(
        traj_dir,
        (row.get("task_id", f"simplevqa_{row.get('data_id', row.get('index'))}") for row in all_rows),
        paths["trajectories"],
    )

    total = len(all_rows)
    exact = sum(1 for row in all_rows if row.get("exact_match", _is_exact(row)))
    contains = sum(1 for row in all_rows if row.get("contains_gold", _contains_gold(row)))
    errors = sum(1 for row in all_rows if row.get("error"))
    summary = {
        "total": total,
        "errors": errors,
        "exact_match": exact,
        "contains_gold": contains,
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
    }
    metadata["summary"] = summary
    write_metadata(paths["metadata"], metadata)

    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Output:        {out_path}")
    print(f"Progress:      {progress_path}")
    print(f"Trajectories:  {traj_dir}")
    print(f"Trajectory all:{paths['trajectories']}")
    print(f"Metadata:      {paths['metadata']}")
    print(f"Total:         {total}")
    print(f"Errors:        {errors}")
    print(f"Exact match:   {exact}/{total} = {exact / total:.2%}" if total else "Exact match:   n/a")
    print(f"Contains gold: {contains}/{total} = {contains / total:.2%}" if total else "Contains gold: n/a")


if __name__ == "__main__":
    main()
