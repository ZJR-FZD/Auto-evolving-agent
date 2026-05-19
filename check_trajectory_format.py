import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict


REQUIRED_KEYS = {"timestamp", "step_id", "role", "content", "tool_call_id"}
VALID_ROLES = {"system", "user", "assistant", "tool"}


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def short(x, n=160):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def validate_tool_call(tc):
    errors = []
    if not isinstance(tc, dict):
        return [f"tool_call is not dict: {short(tc)}"]

    if "id" not in tc:
        errors.append("tool_call missing id")
    elif not isinstance(tc["id"], str):
        errors.append("tool_call.id should be str")

    if "type" in tc and tc["type"] != "function":
        errors.append(f"tool_call.type should be function, got {tc['type']}")

    fn = tc.get("function")
    if not isinstance(fn, dict):
        errors.append("tool_call.function missing or not dict")
    else:
        if not isinstance(fn.get("name"), str) or not fn.get("name"):
            errors.append("tool_call.function.name missing or not str")
        args = fn.get("arguments")
        if not isinstance(args, str):
            errors.append("tool_call.function.arguments should be JSON string")
        else:
            try:
                json.loads(args)
            except Exception:
                errors.append("tool_call.function.arguments is not valid JSON string")

    return errors


def validate_record(obj, file_path, line_no):
    errors = []
    warnings = []

    if not isinstance(obj, dict):
        return [f"line is not JSON object: {short(obj)}"], warnings

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")

    extra_keys = set(obj.keys()) - (
        REQUIRED_KEYS
        | {
            "tool_calls",
            "reasoning_content",
            "finish_reason",
            "total_tokens",
            "fn_name",
            "fn_args",
            "reflection_trigger",
            "reflection_mode",
            "failure_type",
            "root_cause",
            "correction_strategy",
            "memory_used",
            "memory_hits",
            "memory_written",
            "critic_confidence",
        }
    )
    if extra_keys:
        warnings.append(f"extra keys: {sorted(extra_keys)}")

    if "timestamp" in obj and not is_number(obj["timestamp"]):
        errors.append("timestamp should be int/float")

    if "step_id" in obj:
        if not isinstance(obj["step_id"], int) or isinstance(obj["step_id"], bool):
            errors.append("step_id should be int")
        elif obj["step_id"] < 0:
            errors.append("step_id should be >= 0")

    role = obj.get("role")
    if role not in VALID_ROLES:
        errors.append(f"role should be one of {sorted(VALID_ROLES)}, got {role!r}")

    if "content" in obj:
        # 官方示例里 user.content 可以是 list；system/assistant/tool 通常是 str。
        if obj["content"] is None:
            errors.append("content should not be null")
        elif role in {"system", "assistant", "tool"} and not isinstance(obj["content"], str):
            warnings.append(f"{role}.content is usually str, got {type(obj['content']).__name__}")
        elif role == "user" and not isinstance(obj["content"], (str, list)):
            warnings.append(f"user.content is usually str or list, got {type(obj['content']).__name__}")

    if "tool_call_id" in obj:
        if obj["tool_call_id"] is not None and not isinstance(obj["tool_call_id"], str):
            errors.append("tool_call_id should be str or null")

    if role == "assistant":
        if "tool_calls" in obj:
            if not isinstance(obj["tool_calls"], list):
                errors.append("assistant.tool_calls should be list")
            else:
                for i, tc in enumerate(obj["tool_calls"]):
                    for e in validate_tool_call(tc):
                        errors.append(f"tool_calls[{i}]: {e}")

        if "reasoning_content" in obj and not isinstance(obj["reasoning_content"], str):
            errors.append("reasoning_content should be str")

        if "total_tokens" in obj:
            if not isinstance(obj["total_tokens"], (int, str)):
                warnings.append("total_tokens is usually int or str")

    if role == "tool":
        if obj.get("tool_call_id") is None:
            warnings.append("tool role usually should have non-null tool_call_id")

        if "fn_name" in obj and not isinstance(obj["fn_name"], str):
            errors.append("fn_name should be str")

        if "fn_args" in obj and not isinstance(obj["fn_args"], dict):
            errors.append("fn_args should be dict")

    return errors, warnings


def validate_file(path):
    errors = []
    warnings = []
    role_counter = Counter()
    step_counter = Counter()
    line_count = 0
    last_timestamp = None
    pending_tool_ids = set()
    seen_tool_ids = set()

    first_roles = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                warnings.append((line_no, "blank line"))
                continue

            line_count += 1

            try:
                obj = json.loads(raw)
            except Exception as e:
                errors.append((line_no, f"invalid JSON: {e}"))
                continue

            rec_errors, rec_warnings = validate_record(obj, path, line_no)
            for e in rec_errors:
                errors.append((line_no, e))
            for w in rec_warnings:
                warnings.append((line_no, w))

            role = obj.get("role")
            if role:
                role_counter[role] += 1
                if len(first_roles) < 3:
                    first_roles.append(role)

            if isinstance(obj.get("step_id"), int):
                step_counter[obj["step_id"]] += 1

            ts = obj.get("timestamp")
            if is_number(ts):
                if last_timestamp is not None and ts < last_timestamp:
                    warnings.append((line_no, "timestamp decreases compared with previous line"))
                last_timestamp = ts

            if role == "assistant" and isinstance(obj.get("tool_calls"), list):
                for tc in obj["tool_calls"]:
                    if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                        pending_tool_ids.add(tc["id"])

            if role == "tool":
                tid = obj.get("tool_call_id")
                if isinstance(tid, str):
                    seen_tool_ids.add(tid)

    if line_count == 0:
        errors.append((0, "empty jsonl file"))

    # 每个轨迹通常前两行应该是 system + user
    if line_count >= 2:
        if len(first_roles) >= 1 and first_roles[0] != "system":
            warnings.append((1, f"first role is usually system, got {first_roles[0]}"))
        if len(first_roles) >= 2 and first_roles[1] != "user":
            warnings.append((2, f"second role is usually user, got {first_roles[1]}"))

    if role_counter["assistant"] == 0:
        warnings.append((0, "no assistant record found"))

    unmatched = pending_tool_ids - seen_tool_ids
    if unmatched:
        warnings.append((0, f"assistant tool_calls not followed by tool result: {sorted(list(unmatched))[:5]}"))

    orphan = seen_tool_ids - pending_tool_ids
    if orphan:
        warnings.append((0, f"tool result has no matching assistant tool_call id: {sorted(list(orphan))[:5]}"))

    return {
        "path": str(path),
        "line_count": line_count,
        "role_counter": dict(role_counter),
        "step_count": len(step_counter),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, help="trajectory jsonl directory")
    ap.add_argument("--report", default="trajectory_format_report.json", help="output report json")
    ap.add_argument("--max-print", type=int, default=30)
    ap.add_argument("--merge-out", default=None, help="optional: merge all valid jsonl lines into one jsonl")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(f"traj-dir not found: {traj_dir}")

    files = sorted(traj_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl files found under: {traj_dir}")

    all_reports = []
    total_lines = 0
    total_errors = 0
    total_warnings = 0
    bad_files = []

    for p in files:
        rep = validate_file(p)
        all_reports.append(rep)
        total_lines += rep["line_count"]
        total_errors += len(rep["errors"])
        total_warnings += len(rep["warnings"])
        if rep["errors"]:
            bad_files.append(str(p))

    summary = {
        "traj_dir": str(traj_dir),
        "num_files": len(files),
        "total_lines": total_lines,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "bad_files": bad_files[:100],
    }

    out = {
        "summary": summary,
        "files": all_reports,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Trajectory Format Check")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {args.report}")

    printed = 0
    for rep in all_reports:
        if not rep["errors"] and not rep["warnings"]:
            continue

        print("\n" + "-" * 80)
        print(rep["path"])
        print(f"lines={rep['line_count']} roles={rep['role_counter']}")

        for line_no, msg in rep["errors"]:
            if printed >= args.max_print:
                break
            print(f"[ERROR] line {line_no}: {msg}")
            printed += 1

        for line_no, msg in rep["warnings"]:
            if printed >= args.max_print:
                break
            print(f"[WARN ] line {line_no}: {msg}")
            printed += 1

        if printed >= args.max_print:
            print(f"\nOnly printed first {args.max_print} issues. See full report json.")
            break

    if args.merge_out:
        merge_path = Path(args.merge_out)
        merge_path.parent.mkdir(parents=True, exist_ok=True)

        merged = 0
        skipped = 0
        with open(merge_path, "w", encoding="utf-8") as w:
            for p in files:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        raw = line.strip()
                        if not raw:
                            skipped += 1
                            continue
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            skipped += 1
                            continue
                        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        merged += 1

        print(f"\nMerged trajectory jsonl saved to: {merge_path}")
        print(f"merged_lines={merged}, skipped_lines={skipped}")

    if total_errors == 0:
        print("\nPASS: 没有发现硬性格式错误。")
    else:
        print("\nFAIL: 存在硬性格式错误，请查看 report。")


if __name__ == "__main__":
    main()
