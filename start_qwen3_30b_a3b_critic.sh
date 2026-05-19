#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${CRITIC_MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3-30B-A3B-Instruct-2507}"
GPU_ID="${CRITIC_CUDA_VISIBLE_DEVICES:-1}"
PORT="${REFLECTION_CRITIC_PORT:-8001}"
HOST="${REFLECTION_CRITIC_HOST:-127.0.0.1}"
SERVED_NAME="${REFLECTION_MODEL:-Qwen3-30B-A3B}"
CONTEXT_LENGTH="${REFLECTION_CRITIC_CONTEXT_LENGTH:-32768}"
TP_SIZE="${REFLECTION_CRITIC_TP_SIZE:-1}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Qwen3-30B-A3B model config not found: ${MODEL_PATH}/config.json" >&2
  echo "Set CRITIC_MODEL_PATH to the real local model directory." >&2
  exit 2
fi

echo "[critic] model path: ${MODEL_PATH}"
echo "[critic] served name: ${SERVED_NAME}"
echo "[critic] endpoint: http://${HOST}:${PORT}/v1"
echo "[critic] CUDA_VISIBLE_DEVICES=${GPU_ID}"
nvidia-smi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if command -v sglang >/dev/null 2>&1; then
  exec sglang serve \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --context-length "${CONTEXT_LENGTH}"
fi

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}"
