#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="2"
export HF_HOME="~/data/hf-home_new"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export model_name_or_path="~/data/models/HuggingFaceTB/SmolLM-135M"
export PYTHONPATH=..:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

#################### 任务执行 ####################

set -e

# python -m src.evaluation.perplexity \
#     --output_dir ../perplexity/outputs \
#     --title log_perplexity

# python -m src.evaluation.perplexity \
#     --experiment rope_v2_start0_step8_svd_method2_rank8 \
#     --cache_implementation dynamic \
#     --model_name_or_path "../checkpoints/rope_v2_start0_step8_svd_method2_rank8/18000_hf" \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --is_mla