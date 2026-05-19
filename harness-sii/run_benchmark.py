"""
Benchmark runner for the submission dataset.

python harness-sii/run_benchmark.py \
    --dataset datasets/benchmark.csv \
    --output-root harness-sii/runs \
    --run-name benchmark_group7 \
    --llm-url http://127.0.0.1:8000/v1 \
    --model Qwen3.5-9B \
    --max-steps 20 \
    --concurrency 2 \
    --result-format minimal \
    --group 7 \
    --overwrite
    
Writes a timestamped run directory:
  harness-sii/runs/benchmark_YYYYMMDD_HHMMSS/
    predictions.jsonl
    final_results.jsonl
    progress.jsonl
    trajectories/
    trajectories.jsonl
    group_<group>.csv
    group_<group>.json
    metadata.json
    submission_group_<group>.zip
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import sys
import time
import zipfile
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
    remove_stale_trajectory,
    run_paths,
    write_jsonl,
    write_metadata,
)


csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.benchmark")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPO_ROOT / "datasets" / "benchmark.csv"
    if (REPO_ROOT / "datasets" / "benchmark.csv").exists()
    else REPO_ROOT / "benchmark.csv"
)


def load_benchmark(path: Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def _result_row(index: int, item: dict, result: dict, elapsed: float) -> dict:
    pred = result.get("answer", "")
    return {
        "index": index,
        "instruction": item["problem"],
        "image": item.get("image", "").strip() or "",
        "answer": item.get("answer", ""),
        "pred": pred,
        "task_id": f"bench_{index:03d}",
        "prediction": pred,
        "steps": result.get("steps", 0),
        "trajectory_path": result.get("trajectory_path", ""),
        "elapsed_seconds": elapsed,
    }


def _error_row(index: int, item: dict, error: str) -> dict:
    return {
        "index": index,
        "instruction": item.get("problem", ""),
        "image": item.get("image", "").strip() or "",
        "answer": item.get("answer", ""),
        "pred": "",
        "task_id": f"bench_{index:03d}",
        "prediction": "",
        "steps": 0,
        "trajectory_path": "",
        "elapsed_seconds": 0.0,
        "error": error,
    }


def _run_one(payload: tuple[int, int, dict, dict]) -> tuple[int, dict]:
    index, total, item, config = payload

    import task_runner

    patch_task_runner_tool_args(task_runner)

    problem = item["problem"]
    image_b64 = item.get("image", "").strip() or None
    task = {
        "id": f"bench_{index:03d}",
        "instruction": problem,
        "image_b64": image_b64,
        "image_url": None,
    }

    remove_stale_trajectory(Path(config["traj_dir"]), task["id"])
    logger.info("[%d/%d] %s", index + 1, total, problem[:80])
    started_at = time.time()
    result = task_runner.run_task(
        task,
        max_steps=int(config["max_steps"]),
        llm_base_url=str(config["llm_url"]),
        model_name=str(config["model"]),
        trajectory_dir=str(config["traj_dir"]),
    )
    return index, _result_row(index, item, result, time.time() - started_at)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark CSV with harness-sii/task_runner.py")
    parser.add_argument("--group", "-g", default="7", help="Group ID used in the submission zip filenames")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Benchmark CSV file")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for timestamped run outputs")
    parser.add_argument("--run-name", default=None, help="Run directory name; defaults to benchmark_YYYYMMDD_HHMMSS")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_BASE_URL, help="SGLang OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Served model name")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum agent loop steps per sample")
    parser.add_argument("--start", type=int, default=0, help="Start index in the dataset")
    parser.add_argument("--end", type=int, default=None, help="End index in the dataset, exclusive")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument(
        "--result-format",
        choices=("full", "minimal"),
        default="minimal",
        help="full keeps debug fields; minimal writes only index/instruction/image/answer/pred",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing --run-name by skipping indices already in predictions.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing --run-name before running")
    return parser.parse_args()


def _run_payloads(
    payloads: list[tuple[int, int, dict, dict]],
    *,
    concurrency: int,
    progress_path: Path,
    out_path: Path,
    result_format: str,
) -> list[dict]:
    new_results: list[dict] = []
    if concurrency == 1:
        for payload in payloads:
            index, _, item, _ = payload
            try:
                _, row = _run_one(payload)
            except Exception as exc:  # noqa: BLE001
                row = _error_row(index, item, f"{type(exc).__name__}: {exc}")
                logger.exception("run_task failed for idx=%d", index)
            append_jsonl(progress_path, row)
            append_jsonl(out_path, format_output_row(row, result_format))
            new_results.append(row)
        return new_results

    logger.info("Running with concurrency=%d", concurrency)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=concurrency)
    future_to_payload: dict[concurrent.futures.Future, tuple[int, int, dict, dict]] = {}
    shutdown_called = False
    try:
        future_to_payload = {executor.submit(_run_one, payload): payload for payload in payloads}
        for future in concurrent.futures.as_completed(future_to_payload):
            index, _, item, _ = future_to_payload[future]
            try:
                _, row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = _error_row(index, item, f"{type(exc).__name__}: {exc}")
                logger.exception("run_task failed for idx=%d", index)
            append_jsonl(progress_path, row)
            append_jsonl(out_path, format_output_row(row, result_format))
            new_results.append(row)
    except KeyboardInterrupt:
        logger.warning("Interrupted, cancelling pending tasks")
        for future in future_to_payload:
            future.cancel()
        for process in getattr(executor, "_processes", {}).values():
            process.terminate()
        shutdown_called = True
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130)
    finally:
        if not shutdown_called:
            executor.shutdown(wait=True, cancel_futures=True)
    return new_results


def _write_submission_csv(
    dataset: list[dict],
    rows: list[dict],
    output_path: Path,
) -> None:
    answers = {
        int(row.get("index", 0)): row.get("pred", row.get("prediction", ""))
        for row in rows
    }
    fieldnames = list(dataset[0].keys()) if dataset else ["problem", "image", "answer"]
    if "answer" not in fieldnames:
        fieldnames.append("answer")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(dataset):
            rec = {key: item.get(key, "") for key in fieldnames}
            rec["answer"] = answers.get(index, "")
            writer.writerow(rec)


def _write_final_results_jsonl(rows: list[dict], output_path: Path) -> None:
    write_jsonl(
        output_path,
        (
            {
                "index": row.get("index"),
                "instruction": row.get("instruction", ""),
                "image": row.get("image", ""),
                "answer": row.get("answer", ""),
                "pred": row.get("pred", row.get("prediction", "")),
            }
            for row in rows
        ),
    )


def _write_submission_trajectory_json(
    trajectory_dir: Path,
    rows: list[dict],
    output_path: Path,
) -> None:
    trajectories = []
    for row in rows:
        index = int(row.get("index", 0))
        task_id = row.get("task_id", f"bench_{index:03d}")
        traj_file = Path(trajectory_dir) / f"{task_id}.jsonl"
        steps = []
        if traj_file.exists():
            with traj_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        trajectories.append(
            {
                "index": index,
                "task_id": task_id,
                "trajectory": steps,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_submission_zip(run_dir: Path, group: str, answer_csv: Path, trajectory_json: Path) -> Path:
    zip_path = run_dir / f"submission_group_{group}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(answer_csv, f"group_{group}.csv")
        zf.write(trajectory_json, f"group_{group}.json")
    return zip_path


def main() -> None:
    args = _parse_args()
    started_at = time.time()

    dataset = load_benchmark(args.dataset)
    end = args.end if args.end is not None else len(dataset)
    selected = list(enumerate(dataset[args.start:end], start=args.start))
    run_dir = prepare_run_dir(
        "benchmark",
        args.output_root,
        args.run_name,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    paths = run_paths(run_dir)

    if paths["predictions"].exists() and args.overwrite:
        paths["predictions"].unlink()
    elif paths["predictions"].exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {paths['predictions']}. Use --resume or --overwrite.")
    if paths["progress"].exists() and args.overwrite:
        paths["progress"].unlink()

    concurrency = max(1, int(args.concurrency))
    metadata = {
        "dataset": "benchmark",
        "dataset_path": args.dataset,
        "group": args.group,
        "run_dir": run_dir,
        "prediction_path": paths["predictions"],
        "progress_path": paths["progress"],
        "trajectory_dir": paths["trajectory_dir"],
        "trajectory_jsonl": paths["trajectories"],
        "selection": {"start": args.start, "end": end, "total_dataset_rows": len(dataset)},
        "llm_url": args.llm_url,
        "model": args.model,
        "max_steps": args.max_steps,
        "concurrency": concurrency,
        "result_format": args.result_format,
        "started_at": started_at,
        "args": args,
    }
    write_metadata(paths["metadata"], metadata)

    done = {}
    if args.resume:
        done = load_done_by_index(paths["progress"]) or load_done_by_index(paths["predictions"])
    payloads: list[tuple[int, int, dict, dict]] = []
    config = {
        "traj_dir": str(paths["trajectory_dir"]),
        "llm_url": args.llm_url,
        "model": args.model,
        "max_steps": args.max_steps,
    }

    for index, item in selected:
        if index in done:
            logger.info("[%d/%d] Already done, skipping", index + 1, end)
            continue
        payloads.append((index, len(dataset), item, config))

    new_results = _run_payloads(
        payloads,
        concurrency=concurrency,
        progress_path=paths["progress"],
        out_path=paths["predictions"],
        result_format=args.result_format,
    )

    all_rows = list(done.values()) + new_results
    all_rows.sort(key=lambda row: int(row.get("index", 0)))
    write_jsonl(paths["predictions"], (format_output_row(row, args.result_format) for row in all_rows))
    final_results_path = run_dir / "final_results.jsonl"
    _write_final_results_jsonl(all_rows, final_results_path)
    concat_trajectories(
        paths["trajectory_dir"],
        (row.get("task_id", f"bench_{int(row.get('index', 0)):03d}") for row in all_rows),
        paths["trajectories"],
    )
    submission_csv = run_dir / f"group_{args.group}.csv"
    submission_json = run_dir / f"group_{args.group}.json"
    _write_submission_csv(dataset, all_rows, submission_csv)
    _write_submission_trajectory_json(paths["trajectory_dir"], all_rows, submission_json)
    zip_path = _write_submission_zip(run_dir, args.group, submission_csv, submission_json)

    errors = sum(1 for row in all_rows if row.get("error"))
    summary = {
        "total": len(all_rows),
        "errors": errors,
        "final_results_jsonl": final_results_path,
        "submission_csv": submission_csv,
        "submission_trajectory_json": submission_json,
        "submission_zip": zip_path,
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
    }
    metadata["summary"] = summary
    write_metadata(paths["metadata"], metadata)

    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Output:        {paths['predictions']}")
    print(f"Final JSONL:   {final_results_path}")
    print(f"Progress:      {paths['progress']}")
    print(f"Trajectories:  {paths['trajectory_dir']}")
    print(f"Trajectory all:{paths['trajectories']}")
    print(f"Metadata:      {paths['metadata']}")
    print(f"Submit CSV:    {submission_csv}")
    print(f"Submit JSON:   {submission_json}")
    print(f"Submission:    {zip_path}")
    print(f"Total:         {len(all_rows)}")
    print(f"Errors:        {errors}")


if __name__ == "__main__":
    main()
