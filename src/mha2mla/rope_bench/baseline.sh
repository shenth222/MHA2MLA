lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks mmlu \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/mmlu/ \
    --cache_requests true \
    --show_config

lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks arc_challenge \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/arc_challenge/ \
    --cache_requests true \
    --show_config

lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks hellaswag \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/hellaswag/ \
    --cache_requests true \
    --show_config

lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks openbookqa \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/openbookqa/ \
    --cache_requests true \
    --show_config

lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks piqa \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/piqa/ \
    --cache_requests true \
    --show_config

lm_eval --model hf \
    --model_args pretrained=/data/shenth/models/SmolLM/1b \
    --tasks winogrande \
    --device cuda:1 \
    --batch_size auto:4 \
    --output_path ./res/SmolLM/1b/winogrande/ \
    --cache_requests true \
    --show_config

lm_eval --model hf --model_args pretrained=/data/shenth/models/llama/2-7b-hf --tasks arc_challenge --device cuda:1 --batch_size auto:4 --output_path ./res/full/llama/arc_challenge/ --cache_requests true