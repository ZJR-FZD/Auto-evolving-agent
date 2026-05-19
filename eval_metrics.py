"""
Evaluation Metrics Calculator
==============================
Computes Exact Match, Contains Match, and Token-level F1
for SimpleVQA and 2Wiki result files.

Usage:
    python eval_metrics.py --result results/group_7_simpleqa.jsonl -o results/group_7_simpleqa_metrics.json
    python eval_metrics.py --result results/group_7_2wiki.jsonl -o results/group_7_2wiki_metrics.json
    python eval_metrics.py --all --group 7   # evaluate both datasets
"""

import argparse
import json
import re
import string
import os
from collections import Counter


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation/articles."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the|的|了|是|在)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + '。，、；：！？""''（）【】'))
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text into words/characters for F1 calculation."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    # For Chinese text, split by character; for English, split by space
    tokens = []
    for char in normalized:
        if '一' <= char <= '鿿':
            tokens.append(char)
        elif char == ' ':
            continue
        else:
            tokens.append(char)
    # Also split by whitespace for English words
    words = normalized.split()
    # Use whichever gives more tokens (handles mixed language)
    if len(words) > len(tokens):
        return words
    return tokens if tokens else words


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(pred: str, answer: str) -> bool:
    return normalize_text(pred) == normalize_text(answer)


def contains_match(pred: str, answer: str) -> bool:
    """Check if answer is contained in pred (or pred in answer)."""
    pred_n = normalize_text(pred)
    ans_n = normalize_text(answer)
    if not ans_n:
        return False
    return ans_n in pred_n or pred_n in ans_n


def token_f1(pred: str, answer: str) -> tuple[float, float, float]:
    """Compute token-level precision, recall, F1."""
    pred_tokens = tokenize(pred)
    ans_tokens = tokenize(answer)

    if not pred_tokens and not ans_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ans_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ans_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ans_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Evaluate a result file
# ---------------------------------------------------------------------------

def evaluate_file(result_path: str) -> dict:
    """Evaluate a result JSONL file and return metrics."""
    records = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    em_count = 0
    contains_count = 0
    f1_scores = []
    details = []

    for rec in records:
        answer = rec.get("answer", "")
        pred = rec.get("pred", "")

        if not answer:
            continue

        em = exact_match(pred, answer)
        contains = contains_match(pred, answer)
        _, _, f1 = token_f1(pred, answer)

        if em:
            em_count += 1
        if contains:
            contains_count += 1
        f1_scores.append(f1)

        details.append({
            "index": rec.get("index"),
            "answer": answer,
            "pred": pred[:200],
            "exact_match": em,
            "contains_match": contains,
            "f1": round(f1, 4),
        })

    evaluated = len(f1_scores)
    metrics = {
        "total": total,
        "evaluated": evaluated,
        "exact_match": round(em_count / evaluated * 100, 2) if evaluated else 0,
        "contains_match": round(contains_count / evaluated * 100, 2) if evaluated else 0,
        "avg_f1": round(sum(f1_scores) / evaluated * 100, 2) if evaluated else 0,
        "em_count": em_count,
        "contains_count": contains_count,
    }
    return {"metrics": metrics, "details": details}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate QA results (EM / Contains / F1)")
    p.add_argument("--result", "-r", help="Path to result JSONL file")
    p.add_argument("--output", "-o", help="Output metrics JSON path")
    p.add_argument("--all", action="store_true", help="Evaluate both simpleqa and 2wiki")
    p.add_argument("--group", "-g", default="7", help="Group ID (used with --all)")
    p.add_argument("--result-dir", default="results", help="Results directory")
    args = p.parse_args()

    if args.all:
        for name in ["simpleqa", "2wiki"]:
            rpath = os.path.join(args.result_dir, f"group_{args.group}_{name}.jsonl")
            opath = os.path.join(args.result_dir, f"group_{args.group}_{name}_metrics.json")
            if not os.path.exists(rpath):
                print(f"[SKIP] {rpath} not found")
                continue
            result = evaluate_file(rpath)
            with open(opath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            m = result["metrics"]
            print(f"[{name}] EM={m['exact_match']}%  Contains={m['contains_match']}%  F1={m['avg_f1']}%  ({m['em_count']}/{m['evaluated']})")
            print(f"  -> {opath}")
    else:
        if not args.result:
            p.error("--result is required when not using --all")
        opath = args.output or args.result.replace(".jsonl", "_metrics.json")
        result = evaluate_file(args.result)
        with open(opath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        m = result["metrics"]
        print(f"EM={m['exact_match']}%  Contains={m['contains_match']}%  F1={m['avg_f1']}%  ({m['em_count']}/{m['evaluated']})")
        print(f"Metrics saved to: {opath}")


if __name__ == "__main__":
    main()
