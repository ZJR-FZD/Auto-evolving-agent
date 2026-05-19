"""Shared helpers for harness evaluation entry points."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-9B")
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip("-._")
    return value or "run"


def prepare_run_dir(
    dataset_name: str,
    output_root: Path,
    run_name: str | None = None,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    """Create or reuse a run directory.

    New runs are named {dataset}_{timestamp}; fixed run names may be resumed or
    overwritten explicitly to avoid accidentally mixing outputs from old runs.
    """
    output_root = Path(output_root)
    dataset_slug = slugify(dataset_name)
    requested_name = run_name or f"{dataset_slug}_{timestamp()}"
    run_dir = output_root / slugify(requested_name)

    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    elif run_dir.exists() and resume:
        pass
    elif run_dir.exists() and run_name:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --resume or --overwrite."
        )
    elif run_dir.exists():
        base = run_dir
        suffix = 2
        while run_dir.exists():
            run_dir = Path(f"{base}_{suffix}")
            suffix += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    return run_dir


def run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "predictions": run_dir / "predictions.jsonl",
        "progress": run_dir / "progress.jsonl",
        "trajectory_dir": run_dir / "trajectories",
        "trajectories": run_dir / "trajectories.jsonl",
        "metadata": run_dir / "metadata.json",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_by_index(path: Path) -> dict[int, dict[str, Any]]:
    if not Path(path).exists():
        return {}

    done: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(Path(path)):
        index = row.get("data_id", row.get("index"))
        if index is not None:
            done[int(index)] = row
    return done


def format_output_row(row: dict[str, Any], result_format: str) -> dict[str, Any]:
    if result_format == "minimal":
        return {
            "index": row.get("index", row.get("data_id")),
            "instruction": row.get("instruction", ""),
            "image": row.get("image", ""),
            "answer": row.get("answer", ""),
            "pred": row.get("pred", row.get("prediction", "")),
        }
    return row


def concat_trajectories(
    trajectory_dir: Path,
    task_ids: Iterable[str],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for task_id in task_ids:
            traj_file = Path(trajectory_dir) / f"{task_id}.jsonl"
            if not traj_file.exists():
                continue
            with traj_file.open("r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line)


def remove_stale_trajectory(trajectory_dir: Path, task_id: str) -> None:
    traj_path = Path(trajectory_dir) / f"{task_id}.jsonl"
    if traj_path.exists():
        traj_path.unlink()


def patch_task_runner_tool_args(task_runner_module) -> None:
    """Adapt search_image(image_url=...) schema to search_image(image=...).

    This keeps the compatibility shim in the eval wrappers instead of changing
    task_runner.py itself.
    """
    search_image = task_runner_module.search_image

    def _call_search_image(args: dict[str, Any]):
        normalized = dict(args)
        if "image" not in normalized and "image_url" in normalized:
            normalized["image"] = normalized.pop("image_url")
        return search_image(**normalized)

    task_runner_module.TOOL_FN_MAP["search_image"] = _call_search_image


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {k: jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    return value


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(metadata), f, ensure_ascii=False, indent=2)
        f.write("\n")
