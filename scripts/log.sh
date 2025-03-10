#################### 环境变量 ####################
<<<<<<< HEAD
export WANDB_API_KEY=""
=======
# export WANDB_API_KEY=""
>>>>>>> feature/low-rank-approx

#################### 函数定义 ####################

log_lighteval_to_wandb() {
    python log_lighteval_to_wandb.py \
        --eval-path $1 \
        --wandb-project mla_smollm_evaluation \
        --wandb-name $2
}

#################### 任务执行 ####################

<<<<<<< HEAD
export MODEL_NAME="v1_2_rope"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"
=======
export MODEL_NAME="1.7B_rope_v4_top4_svd_method7_rank8"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"

export MODEL_NAME="1.7B_rope_v4_top4_svd_method2_rank8"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"
>>>>>>> feature/low-rank-approx
