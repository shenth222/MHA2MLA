#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="~/data/hf-home_new"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export model_name_or_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-135M"
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
#     --model_name_or_path "/home/binguo/data/MLA-FT/checkpoints/rope_v2_start0_step8_svd_method2_rank8/18000_hf" \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --is_mla

# python -m src.evaluation.perplexity \
#     --experiment rope_v2_start0_step8_svd_method7_rank8 \
#     --cache_implementation dynamic \
#     --model_name_or_path "/home/binguo/data/MLA-FT/checkpoints/rope_v2_start0_step8_svd_method7_rank8/18000_hf" \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --is_mla

# python -m src.evaluation.perplexity \
#     --experiment rope_v4_topk4_svd_method2_rank8 \
#     --cache_implementation dynamic \
#     --model_name_or_path "/home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method2_rank8/18000_hf" \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --is_mla

# python -m src.evaluation.perplexity \
#     --experiment rope_v4_topk4_svd_method7_rank8 \
#     --cache_implementation dynamic \
#     --model_name_or_path "/home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank8/18000_hf" \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --is_mla


# python -m src.evaluation.perplexity \
#     --experiment HQQ_int2 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend HQQ \
#     --nbits 2 &
# PID1=$!

# python -m src.evaluation.perplexity \
#     --experiment HQQ_int8 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend HQQ \
#     --nbits 8 &
# PID2=$!

# python -m src.evaluation.perplexity \
#     --experiment quanto_int8 \
#     --cache_implementation quantized \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite \
#     --backend HQQ \
#     --nbits 8 &
# PID3=$!

# wait $PID1 $PID2 $PID3


# python -m src.evaluation.perplexity \
#     --experiment bf16 \
#     --cache_implementation dynamic \
#     --model_name_or_path ${model_name_or_path} \
#     --dataset_name smollm1_corpus \
#     --dtype bf16 \
#     --num_tokens 2048 \
#     --num_samples 100 \
#     --output_dir ../perplexity/outputs \
#     --overwrite &
# PID1=$!

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
#     --nbits 4 &
# PID2=$!

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
#     --nbits 2 &
# PID3=$!

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
#     --nbits 4 &
# PID4=$!

# wait $PID1 $PID2 $PID3 $PID4