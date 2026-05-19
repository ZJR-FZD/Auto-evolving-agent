export CODEX_HOME=/inspire/qb-ilm2/project/26summer-camp-01/26210893/codex
codex

export CLAUDE_CONFIG_DIR=/inspire/qb-ilm2/project/26summer-camp-01/26210893/claude
claude

conda activate pegp
export CUDA_VISIBLE_DEVICES=0,1

python -m sglang.launch_server \
--model-path /inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B \
--port 8000 \
--tp-size 2 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--served-model-name Qwen3.5-9B


export CUDA_VISIBLE_DEVICES=0,1
export FLASHINFER_CACHE_DIR=/inspire/qb-ilm2/project/26summer-camp-01/26210893/flashinfer_cache

python -m sglang.launch_server \
--model-path /inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B \
--port 8000 \
--tp-size 2 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--served-model-name Qwen3.5-9B \
--enforce-disable-flashinfer-allreduce-fusion \
--disable-custom-all-reduce

cd /inspire/qb-ilm2/project/26summer-camp-01/26210893
conda activate pegp

python harness-sii/eval_simplevqa.py \
--data-file simpleVQA/SimpleVQA.jsonl \
--data-root simpleVQA \
--llm-url http://127.0.0.1:8000/v1 \
--model Qwen3.5-9B \
--max-steps 20 \
--concurrency 2 \
--result-format minimal \
--overwrite