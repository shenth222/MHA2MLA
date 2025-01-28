#################### 函数定义 ####################

log_lighteval_to_wandb() {
    python log_lighteval_to_wandb.py \
        --eval-path $1 \
        --wandb-project smollm_evaluation \
        --wandb-name $2
}

#################### 任务执行 ####################

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

log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v0_svd_v_method2_rank8_silu/results rope_v0_svd_v_method2_rank8_silu

log_lighteval_to_wandb /home/binguo/data/MLA-FT/eval_results/rope_v0_svd_v_method3_rank8/results rope_v0_svd_v_method3_rank8

