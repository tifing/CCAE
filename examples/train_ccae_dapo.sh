#!/bin/bash
set -x

export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1

export SYMPY_POOL_SIZE=64
export SYMPY_POOL_MAXTASKS=200
export SYMPY_POOL_START=spawn
export REWARD_POOL_MAXTASKS=200
export MATH_VERIFY_POOL_MAXTASKS=50

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$HOME/data/DAPO-Math-14k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
OUTPUT_DIR="./output/CCAE-DAPO-Math-14k-$(date +%Y%m%d-%H%M%S)"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ccae \
    algorithm.ccae_tau=1 \
    data.train_files=${TRAIN_DATA_DIR}/train.parquet \
    data.train_batch_size=128 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.96 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=4096 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=ccae \
    trainer.experiment_name=ccae-dapo-qwen3-4b \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${OUTPUT_DIR} 2>&1 | tee ccae_dapo_$(date +%Y%m%d-%H%M%S).log
