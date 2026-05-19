"""
Batch evaluation entry point for the local SimpleVQA subset.

This script wraps task_runner.run_task, which evaluates one agent task at a
time. It writes one JSON object per sample to the predictions file and prints a
small aggregate summary at the end.
"""

import argparse
import base64
import concurrent.futures
import json
import os
import re
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "simpleVQA"
DEFAULT_DATA_FILE = DEFAULT_DATA_ROOT / "SimpleVQA.jsonl"
DEFAULT_OUT_FILE = Path(__file__).resolve().parent / "outputs" / "simplevqa_predictions.jsonl"
DEFAULT_TRAJ_DIR = Path(__file__).resolve().parent / "trajectories_simplevqa"
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-9B")
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _load_done(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    for row in _read_jsonl(path):
        data_id = row.get("data_id", row.get("index"))
        if data_id is not None:
            done[int(data_id)] = row
    return done


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


def _sample_to_task(sample: dict, data_root: Path) -> dict:
    data_id = int(sample["data_id"])
    image_path = data_root / sample["image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for data_id={data_id}: {image_path}")

    with image_path.open("rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    return {
        "id": f"simplevqa_{data_id}",
        "instruction": _build_instruction(sample),
        "image_b64": image_b64,
        "image_url": sample.get("image_url"),
    }


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


def _format_output_row(row: dict, result_format: str) -> dict:
    if result_format == "minimal":
        return {
            "index": row.get("index", row.get("data_id")),
            "instruction": row.get("instruction", ""),
            "image": row.get("image", ""),
            "answer": row.get("answer", ""),
            "pred": row.get("pred", row.get("prediction", "")),
        }
    return row


def _patch_task_runner_tool_args(task_runner_module) -> None:
    """Normalize declared tool-schema args to the existing tool implementation.

    task_runner exposes search_image(image_url=...) to the model, while
    tools.search_tool.search_image currently accepts image=.... Keeping this
    adapter in the eval wrapper avoids editing the baseline task_runner file.
    """
    search_image = task_runner_module.search_image

    def _call_search_image(args: dict):
        normalized = dict(args)
        if "image" not in normalized and "image_url" in normalized:
            normalized["image"] = normalized.pop("image_url")
        return search_image(**normalized)

    task_runner_module.TOOL_FN_MAP["search_image"] = _call_search_image


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

    _patch_task_runner_tool_args(task_runner)

    task = _sample_to_task(sample, Path(config["data_root"]))
    _remove_stale_trajectory(Path(config["traj_dir"]), task["id"])
    print(f"[{index}/{total}] run data_id={data_id} task_id={task['id']}", flush=True)
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
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="SimpleVQA JSONL file")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for local images")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE, help="Prediction JSONL output path")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="Trajectory output directory")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_BASE_URL, help="SGLang OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Served model name")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum agent loop steps per sample")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N samples")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument(
        "--result-format",
        choices=("full", "minimal"),
        default="full",
        help="full keeps metrics/debug fields; minimal writes only index/instruction/image/answer/pred",
    )
    parser.add_argument("--resume", action="store_true", help="Skip data_id values already present in --out")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --out before running")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    samples = _read_jsonl(args.data_file)
    samples = samples[args.offset:]
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.overwrite and args.out.exists():
        args.out.unlink()

    done = _load_done(args.out) if args.resume else {}
    new_results: list[dict] = []
    pending: list[tuple[int, dict]] = []

    for index, sample in enumerate(samples, start=1):
        data_id = int(sample["data_id"])
        if data_id in done:
            print(f"[{index}/{len(samples)}] skip data_id={data_id} (already done)")
            continue
        pending.append((index, sample))

    if args.overwrite:
        _clear_pending_trajectories(args.traj_dir, pending)

    config = {
        "data_root": str(args.data_root),
        "traj_dir": str(args.traj_dir),
        "llm_url": args.llm_url,
        "model": args.model,
        "max_steps": args.max_steps,
    }

    concurrency = max(1, int(args.concurrency))
    if concurrency == 1:
        for index, sample in pending:
            try:
                _, row = _run_one_sample((index, len(samples), sample, config))
            except Exception as exc:  # noqa: BLE001
                row = _error_row(sample, f"{type(exc).__name__}: {exc}")
                print(f"[{index}/{len(samples)}] error data_id={sample.get('data_id')}: {row['error']}")
            _append_jsonl(args.out, _format_output_row(row, args.result_format))
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
                _append_jsonl(args.out, _format_output_row(row, args.result_format))
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
    _write_jsonl(args.out, (_format_output_row(row, args.result_format) for row in all_rows))

    total = len(all_rows)
    exact = sum(1 for row in all_rows if row.get("exact_match", _is_exact(row)))
    contains = sum(1 for row in all_rows if row.get("contains_gold", _contains_gold(row)))
    errors = sum(1 for row in all_rows if row.get("error"))
    print("=" * 60)
    print(f"Output:        {args.out}")
    print(f"Trajectories:  {args.traj_dir}")
    print(f"Total:         {total}")
    print(f"Errors:        {errors}")
    print(f"Exact match:   {exact}/{total} = {exact / total:.2%}" if total else "Exact match:   n/a")
    print(f"Contains gold: {contains}/{total} = {contains / total:.2%}" if total else "Contains gold: n/a")


if __name__ == "__main__":
    main()
