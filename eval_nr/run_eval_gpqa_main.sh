#!/bin/bash

set -euo pipefail

MODEL_PATH=${1:-${MODEL_PATH:-}}
GPU_ID=${2:-${GPU_ID:-0}}

if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: $0 <model_path> [gpu_id]"
    echo "or: MODEL_PATH=/path/to/model GPU_ID=0 bash $0"
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH"
    exit 1
fi
eval "$(conda shell.bash hook)"
CONDA_ENV=${CONDA_ENV:-ccae}
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROMPT_TYPE=${PROMPT_TYPE:-qwen3-train}
EVAL_SCRIPT="${SCRIPT_DIR}/eval_mmlu_pro_gpqa.py"
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results/gpqa_main_sampling"}
mkdir -p "${RESULTS_DIR}"

STEP_NAME=$(basename "${MODEL_PATH}")
MODEL_DIR_NAME=$(basename "$(dirname "${MODEL_PATH}")")
MODEL_FULL_NAME="${MODEL_DIR_NAME}_${STEP_NAME}"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${RESULTS_DIR}/${MODEL_FULL_NAME}_${RUN_STAMP}"
SUMMARY_FILE="${RESULTS_DIR}/${MODEL_FULL_NAME}_${RUN_STAMP}.csv"
LOG_FILE="${RESULTS_DIR}/${MODEL_FULL_NAME}_${RUN_STAMP}.log"

BASE_TOKENIZER_DIR=${BASE_TOKENIZER_DIR:-}
TOKENIZER_FILES=(
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "special_tokens_map.json"
    "added_tokens.json"
    "chat_template.jinja"
)

ensure_tokenizer_files() {
    local model_path=$1
    if [ -z "${BASE_TOKENIZER_DIR}" ] || [ ! -d "${BASE_TOKENIZER_DIR}" ]; then
        return
    fi
    for f in "${TOKENIZER_FILES[@]}"; do
        if [ ! -f "${model_path}/${f}" ] && [ -f "${BASE_TOKENIZER_DIR}/${f}" ]; then
            cp -f "${BASE_TOKENIZER_DIR}/${f}" "${model_path}/"
        fi
    done
}

format_duration() {
    local secs=$1
    if [ "${secs}" -ge 3600 ]; then
        printf "%dh%02dm%02ds" $((secs / 3600)) $((secs % 3600 / 60)) $((secs % 60))
    elif [ "${secs}" -ge 60 ]; then
        printf "%dm%02ds" $((secs / 60)) $((secs % 60))
    else
        printf "%ds" "${secs}"
    fi
}

TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
N_SAMPLING=${N_SAMPLING:-16}
MAX_TOKENS=${MAX_TOKENS:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] config.json not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[ERROR] eval script not found: ${EVAL_SCRIPT}"
    exit 1
fi

ensure_tokenizer_files "${MODEL_PATH}"
mkdir -p "${OUTPUT_DIR}"

echo "Model: ${MODEL_FULL_NAME} | GPU: ${GPU_ID} | Run: ${RUN_STAMP}"
echo "Mode: Sampling (T=${TEMPERATURE}, N=${N_SAMPLING}, Top-P=${TOP_P})"
echo "---"

append_score() {
    local result_file="${OUTPUT_DIR}/gpqa_main/test_${PROMPT_TYPE}_-1_seed0_t${TEMPERATURE}_s0_e-1.jsonl"
    if [ -f "${result_file}" ]; then
        local scores
        scores=$(python3 -c "
import json, numpy as np
with open('${result_file}') as f:
    results = [json.loads(line) for line in f]
score_rows = [r.get('score', []) for r in results if r.get('score')]
if score_rows:
    p1 = np.mean([np.mean(s) for s in score_rows]) * 100
    pk = np.mean([1 if any(s) else 0 for s in score_rows]) * 100
    print(f'{p1:.2f},{pk:.2f}')
else:
    print('N/A,N/A')
" 2>/dev/null || echo "N/A,N/A")
        echo "gpqa_main,${scores}" >> "${SUMMARY_FILE}"
    else
        echo "gpqa_main,FAILED,FAILED" >> "${SUMMARY_FILE}"
    fi
}

GLOBAL_START=$(date +%s)
echo "dataset,pass@1,pass@${N_SAMPLING}" > "${SUMMARY_FILE}"

CUR_LOG="${LOG_FILE%.log}_gpqa_main.log"
CUDA_VISIBLE_DEVICES=${GPU_ID} \
TOKENIZERS_PARALLELISM=false \
python3 -u "${EVAL_SCRIPT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_name gpqa_main \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature "${TEMPERATURE}" \
    --n_sampling "${N_SAMPLING}" \
    --top_p "${TOP_P}" \
    --max_tokens_per_call "${MAX_TOKENS}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --enable_prefix_caching \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \
    > "${CUR_LOG}" 2>&1 &
EVAL_PID=$!

PHASE_START=$(date +%s)
while kill -0 "${EVAL_PID}" 2>/dev/null; do
    sleep 3
    [ ! -f "${CUR_LOG}" ] && continue

    prog_line=$(tr '\r' '\n' < "${CUR_LOG}" | grep 'Processed prompts:' | tail -1 || true)
    if [ -n "${prog_line}" ]; then
        pct=$(echo "${prog_line}" | grep -oP '\d+(?=%)' | head -1 || true)
        counts=$(echo "${prog_line}" | grep -oP '\d+/\d+' | head -1 || true)
        eta=$(echo "${prog_line}" | grep -oP '(?<=<)[0-9:]+' | head -1 || true)
        printf "\r%-22s inference %3s%% (%s), ETA %s          " "gpqa_main:" "${pct}" "${counts}" "${eta:-...}"
    elif tr '\r' '\n' < "${CUR_LOG}" | grep -q 'Evaluate:'; then
        printf "\r%-22s scoring...                                    " "gpqa_main:"
    elif tr '\r' '\n' < "${CUR_LOG}" | grep -q 'Loading'; then
        printf "\r%-22s loading model...                               " "gpqa_main:"
    else
        printf "\r%-22s starting...                                    " "gpqa_main:"
    fi
done
wait "${EVAL_PID}"

ELAPSED=$(( $(date +%s) - PHASE_START ))
printf "\r%-22s done (%s)                                      \n" "gpqa_main:" "$(format_duration "${ELAPSED}")"
append_score

TOTAL_ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo ""
echo "---"
echo "Total time: $(format_duration "${TOTAL_ELAPSED}")"
echo ""

column -t -s',' "${SUMMARY_FILE}"
echo "---"
echo "Saved: ${SUMMARY_FILE}"
echo "Log dir: ${RESULTS_DIR}"
