"""
2Wiki evaluation runner.

Writes a timestamped run directory:
  harness-sii/runs/2wiki_YYYYMMDD_HHMMSS/
    predictions.jsonl
    progress.jsonl
    trajectories/
    trajectories.jsonl
    metadata.json
    
python harness-sii/run_2wiki.py \
    --dataset datasets/2wiki.jsonl \
    --output-root harness-sii/runs \
    --run-name 2wiki_raw \
    --llm-url http://127.0.0.1:8000/v1 \
    --model Qwen3.5-9B \
    --max-steps 20 \
    --concurrency 2 \
    --result-format minimal \
    --overwrite
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
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
    remove_stale_trajectory,
    run_paths,
    write_jsonl,
    write_metadata,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.2wiki")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPO_ROOT / "datasets" / "2wiki.jsonl"
    if (REPO_ROOT / "datasets" / "2wiki.jsonl").exists()
    else REPO_ROOT / "2wiki.jsonl"
)


def _dataset_fingerprint(path: Path) -> dict[str, object]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _source_id(item: dict) -> object:
    return item.get("_id", item.get("id"))


def _validate_resume_dataset(metadata_path: Path, current_fingerprint: dict[str, object]) -> None:
    if not metadata_path.exists():
        return

    with metadata_path.open("r", encoding="utf-8") as f:
        existing_metadata = json.load(f)

    previous_fingerprint = existing_metadata.get("dataset_fingerprint")
    if not previous_fingerprint:
        raise ValueError(
            "Cannot safely resume this 2Wiki run because its metadata has no "
            "dataset_fingerprint. The 2Wiki dataset has changed, so reusing old "
            "index-based results may mix predictions from different questions. "
            "Use a new --run-name, --overwrite, or pass --allow-dataset-mismatch "
            "if you intentionally want to resume anyway."
        )

    if previous_fingerprint.get("sha256") != current_fingerprint.get("sha256"):
        raise ValueError(
            "Cannot resume this 2Wiki run because the dataset content differs "
            f"from the original run. previous_sha256={previous_fingerprint.get('sha256')} "
            f"current_sha256={current_fingerprint.get('sha256')}. Use a new "
            "--run-name, --overwrite, or pass --allow-dataset-mismatch if this "
            "is intentional."
        )


def load_dataset(path: Path) -> list[dict]:
    return read_jsonl(path)


def _result_row(index: int, item: dict, result: dict, elapsed: float) -> dict:
    pred = result.get("answer", "")
    return {
        "index": index,
        "instruction": item["question"],
        "image": "",
        "answer": item.get("answer", ""),
        "pred": pred,
        "task_id": f"2wiki_{index:03d}",
        "source_id": _source_id(item),
        "question_type": item.get("type"),
        "prediction": pred,
        "steps": result.get("steps", 0),
        "trajectory_path": result.get("trajectory_path", ""),
        "elapsed_seconds": elapsed,
    }


def _error_row(index: int, item: dict, error: str) -> dict:
    return {
        "index": index,
        "instruction": item.get("question", ""),
        "image": "",
        "answer": item.get("answer", ""),
        "pred": "",
        "task_id": f"2wiki_{index:03d}",
        "source_id": _source_id(item),
        "question_type": item.get("type"),
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

    question = item["question"]
    task = {
        "id": f"2wiki_{index:03d}",
        "instruction": question,
        "image_b64": None,
        "image_url": None,
    }

    remove_stale_trajectory(Path(config["traj_dir"]), task["id"])
    logger.info("[%d/%d] q=%s", index + 1, total, question[:80])
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
    parser = argparse.ArgumentParser(description="Evaluate 2Wiki with harness-sii/task_runner.py")
    parser.add_argument("--group", "-g", default="7", help="Group ID kept in metadata for compatibility")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="2Wiki JSONL file")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for timestamped run outputs")
    parser.add_argument("--run-name", default=None, help="Run directory name; defaults to 2wiki_YYYYMMDD_HHMMSS")
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
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Allow --resume even when the existing run metadata was created from a different 2Wiki dataset",
    )
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


def main() -> None:
    args = _parse_args()
    started_at = time.time()

    dataset = load_dataset(args.dataset)
    dataset_fingerprint = _dataset_fingerprint(args.dataset)
    end = args.end if args.end is not None else len(dataset)
    selected = list(enumerate(dataset[args.start:end], start=args.start))
    run_dir = prepare_run_dir(
        "2wiki",
        args.output_root,
        args.run_name,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    paths = run_paths(run_dir)

    if args.resume and not args.allow_dataset_mismatch:
        _validate_resume_dataset(paths["metadata"], dataset_fingerprint)
    elif args.resume and args.allow_dataset_mismatch:
        logger.warning("Skipping 2Wiki dataset fingerprint validation because --allow-dataset-mismatch was passed")

    if paths["predictions"].exists() and args.overwrite:
        paths["predictions"].unlink()
    elif paths["predictions"].exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {paths['predictions']}. Use --resume or --overwrite.")
    if paths["progress"].exists() and args.overwrite:
        paths["progress"].unlink()

    concurrency = max(1, int(args.concurrency))
    metadata = {
        "dataset": "2wiki",
        "dataset_path": args.dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "resume_dataset_mismatch_allowed": bool(args.allow_dataset_mismatch),
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
    concat_trajectories(
        paths["trajectory_dir"],
        (row.get("task_id", f"2wiki_{int(row.get('index', 0)):03d}") for row in all_rows),
        paths["trajectories"],
    )

    errors = sum(1 for row in all_rows if row.get("error"))
    summary = {
        "total": len(all_rows),
        "errors": errors,
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
    }
    metadata["summary"] = summary
    write_metadata(paths["metadata"], metadata)

    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Output:        {paths['predictions']}")
    print(f"Progress:      {paths['progress']}")
    print(f"Trajectories:  {paths['trajectory_dir']}")
    print(f"Trajectory all:{paths['trajectories']}")
    print(f"Metadata:      {paths['metadata']}")
    print(f"Total:         {len(all_rows)}")
    print(f"Errors:        {errors}")


if __name__ == "__main__":
    main()
