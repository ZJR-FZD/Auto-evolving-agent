"""Run SimpleVQA or 2Wiki evaluation and export required JSONL files.

Outputs:
  - {dataset_name}_results.jsonl
    {"index":, "instruction":, "image":, "answer":, "pred":}
  - {dataset_name}_trajectories.jsonl
    concatenated trajectory entries from all evaluated items
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.dataset")


def _import_runner(name: str):
    """Import a task runner by short name."""
    if name == "plan_react":
        from task_runner_plan_react import run_task
    elif name == "plan_react_negcrit":
        from task_runner_plan_react_negcrit import run_task
    elif name == "basic":
        from task_runner import run_task
    else:
        raise ValueError(f"unknown runner: {name}")
    return run_task


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL dataset."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _clean_pred(text: str) -> str:
    """Normalize answer wrappers while preserving content."""
    pred = str(text or "").strip()
    match = re.search(r"<answer(?:\s+[^>]*)?>(.*?)</answer>", pred, flags=re.I | re.S)
    if match:
        pred = match.group(1).strip()
    pred = re.sub(r"^\[LOW_CONFIDENCE\]\s*", "", pred, flags=re.I).strip()
    return re.sub(r"\s+", " ", pred).strip()


def _load_image_b64(path: Path | None) -> str | None:
    """Load an image as base64 if the file exists."""
    if not path or not path.exists() or not path.is_file():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _simplevqa_item(row: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    """Convert one SimpleVQA row to a runner task and result metadata."""
    root = dataset_path.parent
    rel_image = str(row.get("image") or "").strip()
    image_path = root / rel_image if rel_image else None
    question = str(row.get("question") or "").strip()
    image_url = str(row.get("image_url") or "").strip() or None
    instruction = question
    if image_url:
        instruction += f"\nimage_url: {image_url}"
    return {
        "instruction": instruction,
        "image": rel_image,
        "image_b64": _load_image_b64(image_path),
        "image_url": image_url,
        "answer": row.get("answer", ""),
    }


def _format_2wiki_context(context: Any) -> str:
    """Render 2Wiki context without using gold answers/support labels."""
    if not isinstance(context, list):
        return str(context or "")
    parts: list[str] = []
    for item in context:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = str(item[0])
        sentences = item[1]
        if isinstance(sentences, list):
            text = " ".join(str(s) for s in sentences)
        else:
            text = str(sentences)
        parts.append(f"[{title}] {text}")
    return "\n".join(parts)


def _2wiki_item(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one 2Wiki row to a runner task and result metadata."""
    question = str(row.get("question") or "").strip()
    context = _format_2wiki_context(row.get("context"))
    instruction = (
        "Answer the question using the provided context. "
        "Output only the final answer in <answer>...</answer>.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    return {
        "instruction": instruction,
        "image": "",
        "image_b64": None,
        "image_url": None,
        "answer": row.get("answer", ""),
    }


def _build_item(dataset_name: str, row: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    """Convert a raw dataset row to common evaluation fields."""
    name = dataset_name.lower()
    if name in {"simplevqa", "simple_vqa"}:
        return _simplevqa_item(row, dataset_path)
    if name in {"2wiki", "2wikimultihopqa", "2wiki_multihop"}:
        return _2wiki_item(row)
    raise ValueError(f"unsupported dataset-name: {dataset_name}")


def run_eval(
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    traj_dir: str,
    runner_name: str,
    start: int,
    end: int | None,
    concurrency: int,
) -> Path:
    """Run dataset evaluation and write JSONL result/trajectory files."""
    dataset_file = Path(dataset_path)
    raw_rows = _read_jsonl(dataset_file)
    end = min(end if end is not None else len(raw_rows), len(raw_rows))
    rows = raw_rows[start:end]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / timestamp
    traj_out_dir = Path(traj_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"{dataset_name}_results.jsonl"
    trajectories_path = out_dir / f"{dataset_name}_trajectories.jsonl"
    progress_path = out_dir / f"{dataset_name}_progress.jsonl"
    lock = threading.Lock()
    run_task = _import_runner(runner_name)

    logger.info(
        "Running dataset=%s runner=%s items=%d [%d:%d]",
        dataset_name,
        runner_name,
        len(rows),
        start,
        end,
    )

    def _run_one(local_i: int, row: dict[str, Any]) -> dict[str, Any]:
        idx = start + local_i
        item = _build_item(dataset_name, row, dataset_file)
        task = {
            "id": f"{dataset_name}_{idx:03d}",
            "instruction": item["instruction"],
            "image_b64": item.get("image_b64"),
            "image_url": item.get("image_url"),
        }
        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=str(traj_out_dir))
            pred = _clean_pred(result.get("answer", ""))
        except Exception as exc:
            logger.error("run_task failed idx=%d: %s", idx, exc)
            pred = ""
        elapsed = time.time() - t0
        rec = {
            "index": idx,
            "instruction": item["instruction"],
            "image": item.get("image", ""),
            "answer": item.get("answer", ""),
            "pred": pred,
        }
        with lock:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("[%d] pred=%s %.1fs", idx, pred[:80], elapsed)
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, i, row) for i, row in enumerate(rows)]
        for future in as_completed(futures):
            future.result()

    records: dict[int, dict[str, Any]] = {}
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[int(rec["index"])] = rec

    with open(results_path, "w", encoding="utf-8") as out:
        for idx in range(start, end):
            if idx in records:
                out.write(json.dumps(records[idx], ensure_ascii=False) + "\n")

    with open(trajectories_path, "w", encoding="utf-8") as out:
        for idx in range(start, end):
            traj_file = traj_out_dir / f"{dataset_name}_{idx:03d}.jsonl"
            if not traj_file.exists():
                continue
            with open(traj_file, "r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line)

    logger.info("Results saved to %s", results_path)
    logger.info("Trajectories saved to %s", trajectories_path)
    return out_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run SimpleVQA/2Wiki dataset eval")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--runner", choices=["basic", "plan_react", "plan_react_negcrit"], default="plan_react")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrency", "-c", type=int, default=5)
    args = parser.parse_args()

    out_dir = run_eval(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        runner_name=args.runner,
        start=args.start,
        end=args.end,
        concurrency=args.concurrency,
    )
    print(f"\nDone! Output dir: {out_dir}")


if __name__ == "__main__":
    main()
