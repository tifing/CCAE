#!/bin/bash

set -euo pipefail

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
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
EVAL_SCRIPT="${SCRIPT_DIR}/eval_mmlu_pro_gpqa.py"
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results/mmlu_pro_by_cat"}
mkdir -p "${RESULTS_DIR}"

if [ -n "${MODEL_PATHS:-}" ]; then
    read -r -a MODELS <<< "${MODEL_PATHS}"
elif [ -n "${MODEL_PATH:-}" ]; then
    MODELS=("${MODEL_PATH}")
else
    echo "Set MODEL_PATH or MODEL_PATHS to the checkpoint path(s) to evaluate."
    exit 1
fi

GPU_STEM1=${GPU_STEM1:-0}
GPU_STEM2=${GPU_STEM2:-1}
GPU_STEM3=${GPU_STEM3:-2}
GPU_HUM=${GPU_HUM:-3}
GPU_SOC=${GPU_SOC:-4}
GPU_OTH=${GPU_OTH:-5}

TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
N_SAMPLING=${N_SAMPLING:-16}
MAX_TOKENS=${MAX_TOKENS:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:---enable_prefix_caching}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}

declare -A TASK_GPUS
declare -A TASK_SUBJECTS

TASK_GPUS[STEM1]=${GPU_STEM1}
TASK_SUBJECTS[STEM1]="math computer_science"
TASK_GPUS[STEM2]=${GPU_STEM2}
TASK_SUBJECTS[STEM2]="physics biology"
TASK_GPUS[STEM3]=${GPU_STEM3}
TASK_SUBJECTS[STEM3]="chemistry engineering"
TASK_GPUS[Humanities]=${GPU_HUM}
TASK_SUBJECTS[Humanities]="history philosophy law"
TASK_GPUS[SocialSciences]=${GPU_SOC}
TASK_SUBJECTS[SocialSciences]="economics psychology"
TASK_GPUS[Other]=${GPU_OTH}
TASK_SUBJECTS[Other]="other business health"

ALL_TASKS="STEM1 STEM2 STEM3 Humanities SocialSciences Other"
TASKS_TO_RUN=${TASKS_TO_RUN:-${ALL_TASKS}}
AGGREGATE_ONLY=${AGGREGATE_ONLY:-0}

parse_model_info() {
    local model_path=$1
    local step_name
    local model_name
    step_name=$(basename "${model_path}")
    model_name=$(basename "$(dirname "${model_path}")")
    echo "${model_name},${step_name}"
}

eval_task() {
    local model_path=$1
    local gpu_id=$2
    local task_name=$3
    local subjects=$4

    local info
    info=$(parse_model_info "${model_path}")
    local model_name
    local step_name
    model_name=$(echo "${info}" | cut -d',' -f1)
    step_name=$(echo "${info}" | cut -d',' -f2)

    local output_dir="${RESULTS_DIR}/${model_name}/${step_name}"
    local log_file="${RESULTS_DIR}/${model_name}_${step_name}_${task_name}.log"

    if [ ! -f "${model_path}/config.json" ]; then
        echo "[GPU${gpu_id}] Skipping: ${model_name}/${step_name} @ ${task_name} (config.json not found)"
        return
    fi

    echo "[GPU${gpu_id}] Starting eval: ${model_name}/${step_name} @ ${task_name}"
    local start_time
    start_time=$(date +%s)
    mkdir -p "${output_dir}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    TOKENIZERS_PARALLELISM=false \
    python3 -u "${EVAL_SCRIPT}" \
        --model_name_or_path "${model_path}" \
        --data_name mmlu_pro \
        --data_dir "${DATA_DIR}" \
        --output_dir "${output_dir}/${task_name}" \
        --prompt_type "${PROMPT_TYPE}" \
        --subjects ${subjects} \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --seed 0 \
        --temperature "${TEMPERATURE}" \
        --n_sampling "${N_SAMPLING}" \
        --top_p "${TOP_P}" \
        --max_tokens_per_call "${MAX_TOKENS}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --max_num_seqs "${MAX_NUM_SEQS}" \
        ${ENABLE_PREFIX_CACHING} \
        --start 0 \
        --end -1 \
        --save_outputs \
        > "${log_file}" 2>&1

    local duration=$(( $(date +%s) - start_time ))
    echo "[GPU${gpu_id}] Finished: ${model_name}/${step_name} @ ${task_name} (${duration}s)"
}

aggregate_results() {
    local model_path=$1
    local info
    info=$(parse_model_info "${model_path}")
    local model_name
    local step_name
    model_name=$(echo "${info}" | cut -d',' -f1)
    step_name=$(echo "${info}" | cut -d',' -f2)
    local output_dir="${RESULTS_DIR}/${model_name}/${step_name}"

    _AGG_OUTPUT_DIR="${output_dir}" \
    _AGG_MODEL_NAME="${model_name}" \
    _AGG_STEP_NAME="${step_name}" \
    _AGG_N_SAMPLING="${N_SAMPLING}" \
    python3 <<'PY'
import json
import os
import sys

import numpy as np

output_dir = os.environ["_AGG_OUTPUT_DIR"]
model_name = os.environ["_AGG_MODEL_NAME"]
step_name = os.environ["_AGG_STEP_NAME"]
n_sampling = int(os.environ["_AGG_N_SAMPLING"])

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["economics", "psychology"],
    "Other": ["other", "business", "health"],
}
subj_to_cat = {subject: category for category, subjects in categories.items() for subject in subjects}

task_dirs = ["STEM1", "STEM2", "STEM3", "Humanities", "SocialSciences", "Other"]
all_samples = []
missing_tasks = []
for task_dir in task_dirs:
    jsonl_dir = os.path.join(output_dir, task_dir, "mmlu_pro")
    if not os.path.isdir(jsonl_dir):
        missing_tasks.append(task_dir)
        continue
    for fname in os.listdir(jsonl_dir):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(jsonl_dir, fname)) as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

if missing_tasks:
    print(f"WARNING: Missing task results: {', '.join(missing_tasks)}")
if not all_samples:
    print("ERROR: No result files found.")
    sys.exit(1)

subject_data = {}
for sample in all_samples:
    subject = sample.get("category", sample.get("type", "unknown"))
    subject_data.setdefault(subject, []).append(sample["score"])

cat_data = {}
for subject, scores in subject_data.items():
    category = subj_to_cat.get(subject, "Other")
    cat_data.setdefault(category, []).extend(scores)

all_scores = [sample["score"] for sample in all_samples]
overall_p1 = np.mean([np.mean(s) for s in all_scores]) * 100
overall_pk = np.mean([1.0 if any(s) else 0.0 for s in all_scores]) * 100

print(f"===== Summary: {model_name}/{step_name} =====")
print(f"Total samples loaded: {len(all_samples)} / 12032")
for subject in sorted(subject_data):
    scores = subject_data[subject]
    p1 = np.mean([np.mean(s) for s in scores]) * 100
    pk = np.mean([1.0 if any(s) else 0.0 for s in scores]) * 100
    print(f"{subject:<20s} Pass@1={p1:>6.2f}% Pass@{n_sampling}={pk:>6.2f}% Count={len(scores)}")
for category in ["STEM", "Humanities", "Social Sciences", "Other"]:
    if category not in cat_data:
        continue
    scores = cat_data[category]
    p1 = np.mean([np.mean(s) for s in scores]) * 100
    pk = np.mean([1.0 if any(s) else 0.0 for s in scores]) * 100
    print(f"{category:<20s} Pass@1={p1:>6.2f}% Pass@{n_sampling}={pk:>6.2f}% Count={len(scores)}")
print(f"{'Overall':<20s} Pass@1={overall_p1:>6.2f}% Pass@{n_sampling}={overall_pk:>6.2f}% Count={len(all_samples)}")

agg_metrics = {
    "model": f"{model_name}/{step_name}",
    "num_samples": len(all_samples),
    "complete": len(all_samples) >= 12032,
    "overall_pass1": round(overall_p1, 2),
    f"overall_pass{n_sampling}": round(overall_pk, 2),
    "subject_pass1": {k: round(np.mean([np.mean(s) for s in v]) * 100, 2) for k, v in sorted(subject_data.items())},
    f"subject_pass{n_sampling}": {k: round(np.mean([1.0 if any(s) else 0.0 for s in v]) * 100, 2) for k, v in sorted(subject_data.items())},
    "cat_pass1": {k: round(np.mean([np.mean(s) for s in v]) * 100, 2) for k, v in sorted(cat_data.items())},
    f"cat_pass{n_sampling}": {k: round(np.mean([1.0 if any(s) else 0.0 for s in v]) * 100, 2) for k, v in sorted(cat_data.items())},
}
agg_file = os.path.join(output_dir, "mmlu_pro_aggregated_metrics.json")
with open(agg_file, "w") as f:
    json.dump(agg_metrics, f, indent=2, ensure_ascii=False)
print(f"Aggregated metrics saved to {agg_file}")
PY
}

export -f eval_task parse_model_info
export RESULTS_DIR PROMPT_TYPE TEMPERATURE N_SAMPLING TOP_P MAX_TOKENS NUM_TEST_SAMPLE
export GPU_MEMORY_UTILIZATION MAX_NUM_SEQS ENABLE_PREFIX_CACHING
export DATA_DIR EVAL_SCRIPT

if [ "${AGGREGATE_ONLY}" = "1" ]; then
    for model_path in "${MODELS[@]}"; do
        aggregate_results "${model_path}"
    done
    exit 0
fi

echo "MMLU-Pro evaluation by category (one GPU per group)"
echo "Tasks on this node: ${TASKS_TO_RUN}"

for model_path in "${MODELS[@]}"; do
    info=$(parse_model_info "${model_path}")
    model_name=$(echo "${info}" | cut -d',' -f1)
    step_name=$(echo "${info}" | cut -d',' -f2)
    echo "Evaluating model: ${model_name}/${step_name}"

    ALL_PIDS=()
    for task in ${TASKS_TO_RUN}; do
        eval_task "${model_path}" "${TASK_GPUS[$task]}" "${task}" "${TASK_SUBJECTS[$task]}" &
        ALL_PIDS+=($!)
    done

    for pid in "${ALL_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null
    done

    aggregate_results "${model_path}"
done

echo "Results directory: ${RESULTS_DIR}"
