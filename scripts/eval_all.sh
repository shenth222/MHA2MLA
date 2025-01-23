#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/home/binguo/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2

    torchrun --nproc_per_node=1 \
        ../modules/nanotron/examples/llama/convert_nanotron_to_hf.py \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name /home/binguo/data/models/HuggingFaceTB/SmolLM-135M

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} --main_process_port 25675 \
        -m  lighteval \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
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


eval_one_ckpt_hf() {
    local model_name_or_path=$1
    local output_dir=$2

    lighteval accelerate \
        --model_args "pretrained=${model_name_or_path},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048,data_parallel_size=${NUM_GPUS}" \
        --custom_tasks "../src/evaluation/smollm1_tasks.py" \
        --tasks "../src/evaluation/smollm1_base.txt" \
        --output_dir "../eval_results/${output_dir}"
}

eval_all_hf() {
    local model_name_path=$1
    local output_dir=$2

    # eval所有检查点
    matching_directories=$(find "$model_name_path" -type d -name "checkpoint-[0-9]*")
    echo $matching_directories

    for dir in $matching_directories; do
        echo "Evaluating $dir"
        eval_one_ckpt_hf $dir $output_dir
    done
}


#################### 任务执行 ####################


eval_all /home/binguo/data/MLA-FT/checkpoints/lr_1e-4_cooldown_10 lr_1e-4_cooldown_10
