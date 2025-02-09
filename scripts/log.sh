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

<<<<<<< Updated upstream
export MODEL_NAME="1.7B_rope_v4_top4_svd_method7_rank32"

log_lighteval_to_wandb ../eval_results/${MODEL_NAME}/results "${MODEL_NAME}"
=======
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v1_4_rope/results v1_4_rope
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v1_8_rope/results v1_8_rope

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v2_start0_step8_rope/results v2_start0_step8_rope
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v2_start0_step4_rope/results v2_start0_step4_rope

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v3_top2_last2_rope/results v3_top2_last2_rope
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v3_top4_last4_rope/results v3_top4_last4_rope

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v4_topk4_rope/results v4_topk4_rope
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v4_topk8_rope/results v4_topk8_rope

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v5_last4_rope/results v5_last4_rope
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/v5_last8_rope/results v5_last8_rope

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v2_start0_step8_svd_method2_rank8/results rope_v2_start0_step8_svd_method2_rank8
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v2_start0_step8_svd_method5_rank8/results rope_v2_start0_step8_svd_method5_rank8
# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v2_start0_step8_svd_method7_rank8/results rope_v2_start0_step8_svd_method7_rank8

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v0_svd_v_rank8/results rope_v0_svd_v_rank8

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v2_start0_step8_svd_method2_rank4/results rope_v2_start0_step8_svd_method2_rank4

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v2_start0_step8_svd_method2_rank16/results rope_v2_start0_step8_svd_method2_rank16

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v4_topk4_svd_method2_rank4/results rope_v4_topk4_svd_method2_rank4

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v4_topk4_svd_method2_rank16/results rope_v4_topk4_svd_method2_rank16

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v4_topk4_svd_method7_rank16/results rope_v4_topk4_svd_method7_rank16

# log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v4_topk4_svd_method7_rank32/results rope_v4_topk4_svd_method7_rank32
log_lighteval_to_wandb ../eval_results/rope_v2_start0_step8_svd_method7_rank8/results rope_v2_start0_step8_svd_method7_rank8
>>>>>>> Stashed changes
