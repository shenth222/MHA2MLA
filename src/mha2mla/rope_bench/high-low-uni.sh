# python partial_rope.py --model_name 135m --task winogrande --rope_method high --output_path ./res/high/SmolLM/135m/winogrande/
# python partial_rope.py --model_name 135m --task mmlu --rope_method high --output_path ./res/high/SmolLM/135m/mmlu/
# python partial_rope.py --model_name 135m --task arc_challenge --rope_method high --output_path ./res/high/SmolLM/135m/arc_challenge/
# python partial_rope.py --model_name 135m --task hellaswag --rope_method high --output_path ./res/high/SmolLM/135m/hellaswag/
# python partial_rope.py --model_name 135m --task openbookqa --rope_method high --output_path ./res/high/SmolLM/135m/openbookqa/
# python partial_rope.py --model_name 135m --task piqa --rope_method high --output_path ./res/high/SmolLM/135m/piqa/

# python partial_rope.py --model_name 135m --task winogrande --rope_method uniform --output_path ./res/uniform/SmolLM/135m/winogrande/
# python partial_rope.py --model_name 135m --task mmlu --rope_method uniform --output_path ./res/uniform/SmolLM/135m/mmlu/
# python partial_rope.py --model_name 135m --task arc_challenge --rope_method uniform --output_path ./res/uniform/SmolLM/135m/arc_challenge/
# python partial_rope.py --model_name 135m --task hellaswag --rope_method uniform --output_path ./res/uniform/SmolLM/135m/hellaswag/
# python partial_rope.py --model_name 135m --task openbookqa --rope_method uniform --output_path ./res/uniform/SmolLM/135m/openbookqa/
# python partial_rope.py --model_name 135m --task piqa --rope_method uniform --output_path ./res/uniform/SmolLM/135m/piqa/

#!/bin/bash

# filepath: /data/shenth/work/MHA2MLA/src/mha2mla/rope_bench/run_all.sh

# 定义参数列表
# MODEL_NAMES=("360m" "2-7b" "1b")
MODEL_NAMES=("1b")
TASKS=("winogrande" "mmlu" "arc_challenge" "hellaswag" "openbookqa" "piqa")
# TASKS=("mmlu" "arc_challenge" "hellaswag" "openbookqa" "piqa")
# ROPE_METHODS=("high" "uniform" "low" "high-low" "full-rope")
ROPE_METHODS=("accumulate")
BASE_OUTPUT_PATH="./res"
MAX_COMPONENT=false

# 遍历所有组合
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for ROPE_METHOD in "${ROPE_METHODS[@]}"; do
            # 构造输出路径
            if [ "$ROPE_METHOD" == "accumulate" ]; then
                OUTPUT_PATH="${BASE_OUTPUT_PATH}/${MODEL_NAME}/${ROPE_METHOD}-"$MAX_COMPONENT"/${TASK}/"
            else
                OUTPUT_PATH="${BASE_OUTPUT_PATH}/${MODEL_NAME}/${ROPE_METHOD}/${TASK}/"
            fi

            # 创建输出目录（如果不存在）
            # mkdir -p "$OUTPUT_PATH"

            # 执行程序
            echo "Running: model_name=${MODEL_NAME}, task=${TASK}, rope_method=${ROPE_METHOD}, output_path=${OUTPUT_PATH}"
            CUDA_VISIBLE_DEVICES=0 python partial_rope.py --model_name "$MODEL_NAME" --task "$TASK" --rope_method "$ROPE_METHOD" --output_path "$OUTPUT_PATH" --max_component "$MAX_COMPONENT"
        done
    done
done