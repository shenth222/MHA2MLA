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

    lighteval accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048,data_parallel_size=${NUM_GPUS}" \
        --custom_tasks "../src/evaluation/smollm1_tasks.py" \
        --tasks "../src/evaluation/smollm1_base.txt" \
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


eval_all_hf "/home/binguo/data/MLA-FT/checkpoints/test" "hf_test"

# torchrun --standalone --nnodes=1 --nproc-per-node=3  \
#     -m lighteval nanotron \
#     --checkpoint_config_path /home/binguo/data/MLA-FT/checkpoints/test/0/config.yaml \
#     --lighteval_config_path /home/binguo/data/MLA-FT/configs/eval/lighteval_config.yaml


# lighteval accelerate \
#   --model_args "pretrained=HuggingFaceTB/SmolLM2-1.7B,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
#   --custom_tasks "tasks.py" --tasks "smollm2_base.txt" --output_dir "./evals" --save_details

# lighteval accelerate \
#     --model_args "pretrained=/home/binguo/data/models/HuggingFaceTB/SmolLM-360M,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048,data_parallel_size=${NUM_GPUS}" \
#     --custom_tasks "../src/evaluation/smollm1_tasks.py" \
#     --tasks "custom|mmlu_cloze:abstract_algebra|0|1,custom|mmlu_cloze:anatomy|0|1,custom|mmlu_cloze:astronomy|0|1,custom|mmlu_cloze:business_ethics|0|1,custom|mmlu_cloze:clinical_knowledge|0|1,custom|mmlu_cloze:college_biology|0|1,custom|mmlu_cloze:college_chemistry|0|1,custom|mmlu_cloze:college_computer_science|0|1,custom|mmlu_cloze:college_mathematics|0|1,custom|mmlu_cloze:college_medicine|0|1,custom|mmlu_cloze:college_physics|0|1,custom|mmlu_cloze:computer_security|0|1,custom|mmlu_cloze:conceptual_physics|0|1,custom|mmlu_cloze:econometrics|0|1,custom|mmlu_cloze:electrical_engineering|0|1,custom|mmlu_cloze:elementary_mathematics|0|1,custom|mmlu_cloze:formal_logic|0|1,custom|mmlu_cloze:global_facts|0|1,custom|mmlu_cloze:high_school_biology|0|1,custom|mmlu_cloze:high_school_chemistry|0|1,custom|mmlu_cloze:high_school_computer_science|0|1,custom|mmlu_cloze:high_school_european_history|0|1,custom|mmlu_cloze:high_school_geography|0|1,custom|mmlu_cloze:high_school_government_and_politics|0|1,custom|mmlu_cloze:high_school_macroeconomics|0|1,custom|mmlu_cloze:high_school_mathematics|0|1,custom|mmlu_cloze:high_school_microeconomics|0|1,custom|mmlu_cloze:high_school_physics|0|1,custom|mmlu_cloze:high_school_psychology|0|1,custom|mmlu_cloze:high_school_statistics|0|1,custom|mmlu_cloze:high_school_us_history|0|1,custom|mmlu_cloze:high_school_world_history|0|1,custom|mmlu_cloze:human_aging|0|1,custom|mmlu_cloze:human_sexuality|0|1,custom|mmlu_cloze:international_law|0|1,custom|mmlu_cloze:jurisprudence|0|1,custom|mmlu_cloze:logical_fallacies|0|1,custom|mmlu_cloze:machine_learning|0|1,custom|mmlu_cloze:management|0|1,custom|mmlu_cloze:marketing|0|1,custom|mmlu_cloze:medical_genetics|0|1,custom|mmlu_cloze:miscellaneous|0|1,custom|mmlu_cloze:moral_disputes|0|1,custom|mmlu_cloze:moral_scenarios|0|1,custom|mmlu_cloze:nutrition|0|1,custom|mmlu_cloze:philosophy|0|1,custom|mmlu_cloze:prehistory|0|1,custom|mmlu_cloze:professional_accounting|0|1,custom|mmlu_cloze:professional_law|0|1,custom|mmlu_cloze:professional_medicine|0|1,custom|mmlu_cloze:professional_psychology|0|1,custom|mmlu_cloze:public_relations|0|1,custom|mmlu_cloze:security_studies|0|1,custom|mmlu_cloze:sociology|0|1,custom|mmlu_cloze:us_foreign_policy|0|1,custom|mmlu_cloze:virology|0|1,custom|mmlu_cloze:world_religions|0|1" \
#     --output_dir "../eval_results/"
    # --tasks "../modules/smollm/evaluation/smollm1_base.txt" \
    # --save_details


