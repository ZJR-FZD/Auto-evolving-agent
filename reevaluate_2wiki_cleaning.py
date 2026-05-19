"""Offline re-evaluation for 2Wiki prediction cleaning.

The cleaning function is gold-free. Gold answers are used only to compute
offline EM / contains / F1 and to diagnose whether formatting caused errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from answer_utils import clean_pred_for_submit
from run_2wiki_text_eval import contains_match, exact_match, token_f1


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    gold_rows = [r for r in rows if str(r.get("gold", "") or "").strip()]
    if not gold_rows:
        return {"gold_count": 0, "exact_match": None, "contains_accuracy": None, "avg_f1": None}
    ems = [exact_match(str(r.get(key, "") or ""), str(r.get("gold", "") or "")) for r in gold_rows]
    contains = [contains_match(str(r.get(key, "") or ""), str(r.get("gold", "") or "")) for r in gold_rows]
    f1s = [token_f1(str(r.get(key, "") or ""), str(r.get("gold", "") or "")) for r in gold_rows]
    f1s = [x for x in f1s if x is not None]
    return {
        "gold_count": len(gold_rows),
        "exact_match": round(sum(ems) / len(ems), 4),
        "contains_accuracy": round(sum(contains) / len(contains), 4),
        "avg_f1": round(mean(f1s), 4) if f1s else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    changed = []
    improved_em = []
    regressed_em = []
    contains_but_not_em = []
    long_before = []
    long_after = []

    for row in rows:
        raw_pred = str(row.get("prediction") or row.get("raw_prediction") or "")
        question = str(row.get("question") or "")
        gold = str(row.get("gold") or "")
        cleaned = clean_pred_for_submit(raw_pred, question)
        row["prediction_original_for_cleaning"] = raw_pred
        row["prediction_cleaned"] = cleaned

        if cleaned != raw_pred:
            changed.append(row.get("idx"))
        if len(raw_pred) > 80:
            long_before.append(row.get("idx"))
        if len(cleaned) > 80:
            long_after.append(row.get("idx"))

        if gold:
            before_em = exact_match(raw_pred, gold)
            after_em = exact_match(cleaned, gold)
            before_contains = contains_match(raw_pred, gold)
            if not before_em and before_contains:
                contains_but_not_em.append(row.get("idx"))
            if not before_em and after_em:
                improved_em.append(row.get("idx"))
            if before_em and not after_em:
                regressed_em.append(row.get("idx"))

    report = {
        "input": args.results,
        "out": args.out,
        "original": evaluate(rows, "prediction_original_for_cleaning"),
        "cleaned": evaluate(rows, "prediction_cleaned"),
        "changed_count": len(changed),
        "changed_indices": changed,
        "improved_em_count": len(improved_em),
        "improved_em_indices": improved_em,
        "regressed_em_count": len(regressed_em),
        "regressed_em_indices": regressed_em,
        "contains_but_not_em_count": len(contains_but_not_em),
        "contains_but_not_em_indices": contains_but_not_em,
        "long_before_count": len(long_before),
        "long_before_indices": long_before,
        "long_after_count": len(long_after),
        "long_after_indices": long_after,
        "compliance_note": (
            "prediction_cleaned was produced from prediction/question only. "
            "Gold was used only for offline scoring and diagnostics."
        ),
    }

    write_jsonl(args.out, rows)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

