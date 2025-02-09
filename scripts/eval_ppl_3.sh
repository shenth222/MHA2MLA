#################### 环境变量 ####################

export HF_HOME="~/data/hf-home_new"
export CUDA_VISIBLE_DEVICES="1"
export model_name_or_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-135M"
export PYTHONPATH=..:$PYTHONPATH

#################### 任务执行 ####################

set -e

# python -m src.evaluation.perplexity \
#     --experiment bf16 \
#     --cache_implementation dynamic \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite

# python -m src.evaluation.perplexity \
#     --experiment quanto_int4 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend quanto \
#     --nbits 4

# python -m src.evaluation.perplexity \
#     --experiment quanto_int2 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend quanto \
#     --nbits 2

# python -m src.evaluation.perplexity \
#     --experiment HQQ_int4 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend HQQ \
#     --nbits 4

