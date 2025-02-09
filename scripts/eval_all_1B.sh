#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2

    torchrun --nproc_per_node=1 --master_port 25675 \
        -m src.original_conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name /cpfs01/shared/llm_ddd/doushihan/taoji/code/MLA-FT/checkpoints/meta-llama/Llama-2-7b-hf
        # --tokenizer_name ../checkpoints/HuggingFaceTB/SmolLM-1.7B

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        -m lighteval accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 50 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

eval_all() {
    local model_name_path=$1
    local output_dir=$2

    # eval所有检查点
    matching_directories=$(find "$model_name_path" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+')

    echo $matching_directories

    for dir in $matching_directories; do
        echo "Evaluating $dir"
        eval_one_ckpt $dir $output_dir
    done
}

#################### 任务执行 ####################

set -e
# eval_all ../checkpoints/1.7B_1 "1.7B_1"
# eval_one_ckpt ../checkpoints/meta-llama/Llama-2-7b-hf "Llama-2-7b-hf"
# eval_one_ckpt ../checkpoints/1.7B_v1_topk4/12000 "1.7B_v1_topk4"
# eval_one_ckpt ../checkpoints/1.7B_v2_start0_step8/12000 "1.7B_v2_start0_step8"
# eval_one_ckpt ../checkpoints/1.7B_v4_topk4/12000 "1.7B_v4_topk4"
# eval_one_ckpt ../checkpoints/1.7B_v5_last4/12000 "1.7B_v5_last4"

eval_one_ckpt ../checkpoints/7B_1/12000 "7B_1"
