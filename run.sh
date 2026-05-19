#!/bin/bash
# Unified runner script
# Usage:
#   ./run.sh [dataset] [mode] [start] [end]
#
# Arguments:
#   dataset:  benchmark | simpleqa | 2wiki | all  (default: benchmark)
#   mode:     basic | plan_react                   (default: basic)
#   start:    start index (optional)
#   end:      end index (optional)
#
# Examples:
#   ./run.sh benchmark plan_react          # 打榜全量, plan_react 范式
#   ./run.sh simpleqa basic 0 5            # SimpleVQA 前5题, 基础范式
#   ./run.sh all plan_react                # 全部数据集, plan_react 范式
#   ./run.sh benchmark basic 0 3           # benchmark 前3题测试

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-benchmark}"
MODE="${2:-basic}"
START="${3:-}"
END="${4:-}"
GROUP="${GROUP_ID:-7}"

RANGE_ARGS=""
if [ -n "$START" ]; then
    RANGE_ARGS="--start $START"
fi
if [ -n "$END" ]; then
    RANGE_ARGS="$RANGE_ARGS --end $END"
fi

echo "=========================================="
echo "  Dataset:  $DATASET"
echo "  Mode:     $MODE"
echo "  Group:    $GROUP"
echo "  Range:    ${START:-0} ~ ${END:-full}"
echo "=========================================="

run_benchmark() {
    echo "[*] Running benchmark (打榜数据集)..."
    python run_benchmark.py --group "$GROUP" --mode "$MODE" $RANGE_ARGS
}

run_simpleqa() {
    echo "[*] Running SimpleVQA (评测集)..."
    python run_simpleqa.py --group "$GROUP" --mode "$MODE" $RANGE_ARGS
}

run_2wiki() {
    echo "[*] Running 2Wiki (评测集)..."
    python run_2wiki.py --group "$GROUP" --mode "$MODE" $RANGE_ARGS
}

case "$DATASET" in
    benchmark)
        run_benchmark
        ;;
    simpleqa)
        run_simpleqa
        ;;
    2wiki)
        run_2wiki
        ;;
    all)
        run_simpleqa
        run_2wiki
        run_benchmark
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Usage: ./run.sh [benchmark|simpleqa|2wiki|all] [basic|plan_react] [start] [end]"
        exit 1
        ;;
esac

echo ""
echo "All done!"
