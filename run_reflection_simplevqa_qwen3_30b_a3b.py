"""
Run SimpleVQA with Qwen3.5-9B as main multimodal agent and Qwen3-30B-A3B as
open-source reflection critic. The critic sees only trajectory/image metadata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_simpleqa import DATASET_DEFAULT, IMAGE_DIR_DEFAULT, check_gpu, run_batch


def configure_env() -> None:
    os.environ["ENABLE_REFLECTION"] = "1"
    os.environ["REFLECTION_USE_LLM"] = "1"
    os.environ["REFLECTION_MODEL"] = "Qwen3-30B-A3B"
    os.environ.setdefault("REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1")
    os.environ.setdefault("REFLECTION_API_KEY", "EMPTY")
    os.environ["MODEL_NAME"] = "Qwen3.5-9B"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_DEFAULT))
    parser.add_argument("--image-dir", default=str(IMAGE_DIR_DEFAULT))
    parser.add_argument("--output", default="results/reflection_qwen3_30b_a3b_simplevqa_full.jsonl")
    parser.add_argument("--traj-dir", default="trajectories_reflection_qwen3_30b_a3b_simplevqa_full")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    check_gpu()
    configure_env()
    from run_simpleqa import load_dataset  # noqa: PLC0415

    dataset = load_dataset(Path(args.dataset))
    end = args.end if args.end is not None else args.start + args.limit
    run_batch(dataset, args.image_dir, args.output, args.start, end, args.traj_dir, args.overwrite)


if __name__ == "__main__":
    main()
