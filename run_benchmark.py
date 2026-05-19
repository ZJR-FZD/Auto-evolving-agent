"""
Benchmark Runner for Competition
================================

Reads benchmark.csv, runs each problem through task_runner.run_task, writes
competition outputs and keeps progress for resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from answer_utils import clean_pred_for_submit

csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("eval.benchmark")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
BENCHMARK_DEFAULT = PROJECT_ROOT.parent / "datasets" / "benchmark.csv"
DEFAULT_GROUP = "7"


def probe_openai_models(base_url: str, timeout: int = 5) -> list[str]:
    models_url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(models_url, timeout=timeout) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"/models returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_pred(raw: Any, question: str = "") -> str:
    return clean_pred_for_submit(raw, question)


def check_gpu() -> None:
    try:
        proc = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"nvidia-smi is not executable here ({exc}). Please run benchmark on the GPU instance.") from None
    smi = proc.stdout or ""
    print(smi)
    if proc.returncode != 0 or "CUDA Version" not in smi:
        raise RuntimeError("GPU/CUDA is not visible. Refusing to run benchmark.")
    if "NVIDIA H200" not in smi:
        logger.warning("CUDA is visible, but this does not look like the expected H200 GPU instance.")
    if not any(name in smi.lower() for name in ("sglang", "vllm", "python")):
        logger.warning(
            "No obvious local model-serving process is shown by nvidia-smi. "
            "run_benchmark.py is only an API client; the GPU should be used by "
            "the service behind LLM_BASE_URL."
        )
    else:
        logger.info("GPU is visible and nvidia-smi shows a local process using it.")


def check_services(llm_base_url: str) -> None:
    logger.info("Main agent endpoint: %s", llm_base_url)
    try:
        main_models = probe_openai_models(llm_base_url)
        logger.info("Main agent models: %s", ", ".join(main_models) or "(none)")
    except Exception as exc:  # noqa: BLE001
        models_url = llm_base_url.rstrip("/") + "/models"
        raise RuntimeError(f"LLM_BASE_URL is not reachable at {models_url}: {exc}") from exc

    if os.getenv("ENABLE_REFLECTION", "0") == "1" and os.getenv("REFLECTION_USE_LLM", "0") == "1":
        critic_base_url = os.getenv("REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1")
        critic_model = os.getenv("REFLECTION_MODEL", "Qwen3-30B-A3B")
        logger.info("Reflection critic endpoint: %s", critic_base_url)
        try:
            critic_models = probe_openai_models(critic_base_url)
            logger.info("Reflection critic models: %s", ", ".join(critic_models) or "(none)")
            if critic_model not in critic_models:
                message = f"REFLECTION_MODEL={critic_model} not listed by critic endpoint."
                if os.getenv("REQUIRE_REFLECTION_CRITIC", "0") == "1":
                    raise RuntimeError(message)
                logger.warning(message)
        except Exception as exc:  # noqa: BLE001
            if os.getenv("REQUIRE_REFLECTION_CRITIC", "0") == "1":
                raise RuntimeError(
                    f"Required reflection critic is not reachable at "
                    f"{critic_base_url.rstrip('/')}/models: {exc}"
                ) from exc
            logger.warning(
                "Reflection critic is not reachable at %s; this run will use rule_fallback. Error: %s",
                critic_base_url.rstrip("/") + "/models",
                exc,
            )
            os.environ["REFLECTION_USE_LLM"] = "0"

    try:
        from tools.search_tool import SEARCH_PROXY_URL
    except Exception:  # noqa: BLE001
        SEARCH_PROXY_URL = os.getenv("SEARCH_PROXY_URL", "")
    if SEARCH_PROXY_URL:
        logger.info("Search proxy active: %s", SEARCH_PROXY_URL)
    else:
        logger.warning("SEARCH_PROXY_URL is not set; search tools may fail unless direct keys/network are available.")

    try:
        from sandbox_client import SANDBOX_BASE_URL
    except Exception:  # noqa: BLE001
        SANDBOX_BASE_URL = os.getenv("SANDBOX_BASE_URL", "")
    if SANDBOX_BASE_URL:
        logger.info("Browser sandbox active: %s", SANDBOX_BASE_URL)
    else:
        logger.warning("SANDBOX_BASE_URL is not set; browser tools may be unavailable.")


def load_done(progress_path: Path) -> tuple[set[int], dict[int, str], list[dict[str, Any]]]:
    done: set[int] = set()
    answers: dict[int, str] = {}
    trajectories = []
    if not progress_path.exists():
        return done, answers, trajectories
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = int(rec["index"])
            done.add(idx)
            answers[idx] = str(rec.get("answer", ""))
            trajectories.append(rec)
    return done, answers, trajectories


def read_traj(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_benchmark(
    dataset: list[dict[str, Any]],
    group_id: str,
    output_dir: str = "results",
    traj_dir: str = "trajectories/benchmark",
    start: int = 0,
    end: int | None = None,
    max_steps: int | None = None,
    time_budget_sec: float | None = None,
) -> str:
    run_started_at = time.time()
    end = min(end or len(dataset), len(dataset))
    subset = dataset[start:end]
    output_path = Path(output_dir)
    traj_path = Path(traj_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    traj_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"group_{group_id}.json"
    csv_path = output_path / f"group_{group_id}.csv"
    zip_path = output_path / f"group_{group_id}.zip"
    progress_path = output_path / f"group_{group_id}_progress.jsonl"
    raw_jsonl = output_path / f"group_{group_id}_raw_results.jsonl"
    final_jsonl = output_path / f"group_{group_id}_final_results.jsonl"

    done_indices, answers_by_idx, all_trajectories = load_done(progress_path)
    logger.info("Running benchmark: %d items [%d:%d], resumed=%d", len(subset), start, end, len(done_indices))
    from task_runner import run_task  # noqa: PLC0415

    for i, item in enumerate(subset):
        if time_budget_sec and time.time() - run_started_at >= time_budget_sec:
            logger.info("Run time budget reached: %.1fs. Stop cleanly; rerun the same command to resume.", time_budget_sec)
            break
        idx = start + i
        if idx in done_indices:
            logger.info("[%d/%d] Already done, skipping", idx, end)
            continue

        problem = item.get("problem") or item.get("instruction") or item.get("question") or ""
        image_b64 = (item.get("image", "") or "").strip() or None
        has_image = image_b64 is not None
        task_type = "benchmark_multimodal" if idx >= 50 or has_image else "benchmark_text"
        if has_image:
            instruction = (
                f"{problem}\n\n"
                "If no online image_url is available, first identify visual entities/scenes from the image, "
                "then use text search/browser verification. Do not call search_image without an image_url."
            )
        else:
            instruction = problem

        task = {
            "id": f"bench_{idx:03d}",
            "instruction": instruction,
            "image_b64": image_b64,
            "image_url": None,
            "task_type": task_type,
            "image_info": {"has_image": has_image, "image_url": "", "benchmark_index": idx},
        }

        logger.info("[%d/%d] %s", idx, end, problem[:80])
        t0 = time.time()
        try:
            run_kwargs = {"trajectory_dir": str(traj_path)}
            if max_steps is not None:
                run_kwargs["max_steps"] = max_steps
            result = run_task(task, **run_kwargs)
            raw_answer = result.get("raw_answer", result.get("answer", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("run_task failed for idx=%d: %s", idx, exc)
            result = {"error": str(exc), "steps": -1}
            raw_answer = ""
        final_answer = clean_pred(raw_answer, problem)
        elapsed = time.time() - t0
        traj_file = traj_path / f"bench_{idx:03d}.jsonl"
        traj_data = read_traj(traj_file)

        answers_by_idx[idx] = final_answer
        entry = {
            "index": idx,
            "problem": problem,
            "has_image": has_image,
            "answer": final_answer,
            "raw_answer": raw_answer,
            "steps": result.get("steps", -1),
            "elapsed_s": round(elapsed, 1),
            "trajectory": traj_data,
        }
        all_trajectories.append(entry)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"index": idx, "answer": final_answer, "raw_answer": raw_answer, "trajectory": traj_data}, ensure_ascii=False) + "\n")
        logger.info("  => answer=%s  %.1fs", final_answer[:80], elapsed)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for j, row in enumerate(dataset):
            writer.writerow({"problem": row.get("problem", ""), "image": row.get("image", ""), "answer": answers_by_idx.get(j, "")})

    raw_rows = []
    final_rows = []
    for j, row in enumerate(dataset):
        raw = next((x.get("raw_answer", "") for x in all_trajectories if x.get("index") == j), answers_by_idx.get(j, ""))
        base = {"index": j, "instruction": row.get("problem", ""), "image": row.get("image", ""), "answer": row.get("answer", "")}
        raw_rows.append({**base, "pred": raw})
        final_rows.append({**base, "pred": clean_pred(raw, row.get("problem", ""))})
    write_jsonl(raw_jsonl, raw_rows)
    write_jsonl(final_jsonl, final_rows)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, f"group_{group_id}.json")
        zf.write(csv_path, f"group_{group_id}.csv")
    logger.info("Saved %s, %s, %s", json_path, csv_path, zip_path)
    return str(zip_path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner for competition")
    parser.add_argument("--group", "-g", required=True, help="Group ID number")
    parser.add_argument("--dataset", default=str(BENCHMARK_DEFAULT))
    parser.add_argument("--output-dir", "-o", default="results")
    parser.add_argument("--traj-dir", default="trajectories/benchmark")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override task_runner max_steps for this benchmark run.")
    parser.add_argument("--time-budget-sec", type=float, default=None, help="Stop cleanly after this many seconds; progress is resumable.")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip GPU/LLM checks for local dry validation only.")
    args = parser.parse_args()

    if not args.skip_env_check:
        check_gpu()
        check_services(os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"benchmark dataset not found: {dataset_path}")
    dataset = load_benchmark(dataset_path)
    zip_path = run_benchmark(
        dataset=dataset,
        group_id=args.group,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        start=args.start,
        end=args.end if args.end is not None else len(dataset),
        max_steps=args.max_steps,
        time_budget_sec=args.time_budget_sec,
    )
    print(f"\nDone! Submission file: {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
