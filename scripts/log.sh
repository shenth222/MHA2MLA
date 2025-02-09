#################### 环境变量 ####################
export WANDB_API_KEY="4bdf7e746b5e3bb922d6c3675da23e2a1d2b642f"

#################### 函数定义 ####################

log_lighteval_to_wandb() {
    python log_lighteval_to_wandb.py \
        --eval-path $1 \
        --wandb-project mla_smollm_evaluation \
        --wandb-name $2
}

#################### 任务执行 ####################

export MODEL_NAME="1.7B_rope_v4_top4_svd_method7_rank32"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"
