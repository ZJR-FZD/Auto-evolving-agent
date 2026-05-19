import json
from pathlib import Path

from answer_utils import clean_pred_for_submit

DATA_PATH = Path("/inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/2wiki.jsonl")
TRAJ_DIR = Path("/inspire/qb-ilm2/project/26summer-camp-01/26210830/harness-sii/trajectories_baseline_2wiki_full")
OUT_DIR = Path("/inspire/qb-ilm2/project/26summer-camp-01/26210830/harness-sii/submit_2wiki_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUT_DIR / "2wiki_raw_results.jsonl"
FINAL_OUT = OUT_DIR / "2wiki_final_results.jsonl"
TRAJ_MERGED_OUT = OUT_DIR / "2wiki_trajectories.jsonl"
REPORT_OUT = OUT_DIR / "2wiki_submit_check_report.json"

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"{path} line {line_no} JSON error: {e}")
    return rows

def get_id(ex, idx):
    for k in ["id", "_id", "qid", "question_id", "task_id", "index"]:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return str(idx)

def get_instruction(ex):
    for k in ["instruction", "question", "query", "input", "text"]:
        if k in ex and ex[k]:
            return str(ex[k])
    return ""

def get_answer(ex):
    for k in ["answer", "gold", "label", "target", "output"]:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return ""

def get_image(ex):
    for k in ["image", "image_path", "image_url"]:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return ""

def clean_pred(raw, question=""):
    return clean_pred_for_submit(raw, question)

def read_final_assistant(traj_path):
    rows = load_jsonl(traj_path)
    assistants = [r for r in rows if r.get("role") == "assistant" and str(r.get("content", "")).strip()]
    if not assistants:
        return ""
    return assistants[-1].get("content", "")

# 1. 读数据集
dataset = load_jsonl(DATA_PATH)

# 2. 建立 task_id -> dataset record
id_to_data = {}
for idx, ex in enumerate(dataset):
    tid = get_id(ex, idx)
    id_to_data[tid] = (idx, ex)

# 3. 建立 trajectory stem -> final assistant raw output
traj_files = sorted(TRAJ_DIR.glob("*.jsonl"))
traj_pred = {}
for p in traj_files:
    traj_pred[p.stem] = read_final_assistant(p)

# 4. 生成结果
raw_rows = []
final_rows = []
missing_traj = []
matched = 0

for idx, ex in enumerate(dataset):
    tid = get_id(ex, idx)
    raw_pred = traj_pred.get(tid, "")

    # 如果 task_id 对不上，尝试用顺序兜底：第 idx 个轨迹对应第 idx 条数据
    if not raw_pred and idx < len(traj_files):
        raw_pred = read_final_assistant(traj_files[idx])

    if raw_pred:
        matched += 1
    else:
        missing_traj.append({"index": idx, "id": tid})

    base = {
        "index": idx,
        "instruction": get_instruction(ex),
        "image": get_image(ex),  # 2Wiki 一般为空
        "answer": get_answer(ex),
    }

    raw_rows.append({**base, "pred": raw_pred})
    final_rows.append({**base, "pred": clean_pred(raw_pred, base["instruction"])})

with open(RAW_OUT, "w", encoding="utf-8") as w:
    for r in raw_rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(FINAL_OUT, "w", encoding="utf-8") as w:
    for r in final_rows:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

# 5. 合并轨迹
with open(TRAJ_MERGED_OUT, "w", encoding="utf-8") as w:
    for p in traj_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    obj = json.loads(s)
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 6. 检查格式
def check_result_format(path):
    bad = []
    n = 0
    required = {"index", "instruction", "image", "answer", "pred"}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            n += 1
            obj = json.loads(line)
            if set(obj.keys()) != required:
                bad.append((line_no, sorted(obj.keys())))
    return n, bad

raw_n, raw_bad = check_result_format(RAW_OUT)
final_n, final_bad = check_result_format(FINAL_OUT)

traj_lines = sum(1 for _ in open(TRAJ_MERGED_OUT, "r", encoding="utf-8"))

report = {
    "dataset_path": str(DATA_PATH),
    "trajectory_dir": str(TRAJ_DIR),
    "dataset_count": len(dataset),
    "trajectory_file_count": len(traj_files),
    "matched_prediction_count": matched,
    "missing_prediction_count": len(missing_traj),
    "missing_prediction_examples": missing_traj[:10],
    "raw_result_file": str(RAW_OUT),
    "final_result_file": str(FINAL_OUT),
    "merged_trajectory_file": str(TRAJ_MERGED_OUT),
    "raw_result_lines": raw_n,
    "final_result_lines": final_n,
    "merged_trajectory_lines": traj_lines,
    "raw_format_errors": raw_bad[:20],
    "final_format_errors": final_bad[:20],
}

with open(REPORT_OUT, "w", encoding="utf-8") as w:
    json.dump(report, w, ensure_ascii=False, indent=2)

print("=" * 80)
print("2Wiki submit files built")
print("=" * 80)
print(json.dumps(report, ensure_ascii=False, indent=2))

if raw_bad or final_bad:
    print("\nNOT PASS: 结果文件字段不完全符合格式。")
elif len(dataset) != len(traj_files):
    print("\nWARNING: 数据集条数和轨迹文件数不一致，可能没跑满。")
elif len(dataset) != 200:
    print("\nWARNING: 当前不是 200 条。请确认官方 2Wiki 评测集是否就是这个条数。")
else:
    print("\nPASS: 原始结果、最终结果、轨迹文件已生成。")
