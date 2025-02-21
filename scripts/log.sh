#################### 环境变量 ####################
export WANDB_API_KEY=""

#################### 函数定义 ####################

log_lighteval_to_wandb() {
    python log_lighteval_to_wandb.py \
        --eval-path $1 \
        --wandb-project mla_smollm_evaluation \
        --wandb-name $2
}

#################### 任务执行 ####################

export MODEL_NAME="v1_2_rope"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"
