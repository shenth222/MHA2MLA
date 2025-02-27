# MHA2MLA

This repo contains the code for the paper ["Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs"](https://arxiv.org/abs/2502.14837).


![alt text](img/overview.png)


## News

* [2025.02.21] The paper of MHA2MLA is publicly available: https://arxiv.org/abs/2502.14837
* [2025.02.19] Released the first version of the MHA2MLA code, providing usage code for Llama fine-tuning and evaluating.

## TO-DO

- [ ] ~~Provide the code for incorporating the projection matrix and inference.~~
- [ ] Thanks to DeepSeek for open-sourcing the [FlashMLA](https://github.com/deepseek-ai/FlashMLA) inference framework. It’s theoretically possible to save more GPU memory usage using this framework. Let’s see how economical MHA2MLA + FlashMLA (+ KV quanto) can be!
- [ ] Release the code of MHA2MLA based on HuggingFace `Transformers`


## Models

* SmolLM: https://huggingface.co/blog/smollm

* Llama-2-7b-hf: https://huggingface.co/meta-llama/Llama-2-7b-hf

We use framework [nanotron](https://github.com/huggingface/nanotron) to train the model and [transformers](https://github.com/huggingface/transformers/) to eval the model, so it is necessary to convert the model to the required format.  

For models with the original structure, the following command can be used for model conversion.

```bash
# hf2nanotron 
torchrun --nproc_per_node=1  \
    -m src.original_conversation.convert_hf_to_nanotron \
    --checkpoint_path meta-llama/Llama-2-7b-hf \
    --save_path meta-llama/Llama-2-7b-nt

# nanotron2hf
torchrun --nproc_per_node=1  \
    -m src.original_conversation.convert_nanotron_to_hf \
    --checkpoint_path meta-llama/Llama-2-7b-nt \
    --tokenizer_path meta-llama/Llama-2-7b-hf \
    --save_path meta-llama/Llama-2-7b
```

For MLA models, the following command can be used for model conversion.

```bash
# hf2nanotron 
torchrun --nproc_per_node=1  \
    -m src.conversation.convert_hf_to_nanotron \
    --checkpoint_path meta-llama/Llama-2-7b-hf \
    --save_path meta-llama/Llama-2-7b-nt \
    --is_mla

# nanotron2hf
torchrun --nproc_per_node=1  \
    -m src.conversation.convert_nanotron_to_hf \
    --checkpoint_path meta-llama/Llama-2-7b-nt \
    --tokenizer_path meta-llama/Llama-2-7b-hf \
    --save_path meta-llama/Llama-2-7b \
    --is_mla
```

## Datasets

First download the datasets.

* smollm-corpus(fineweb-edu-dedup, cosmopedia-v2, python-edu): https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

* open-web-math: https://huggingface.co/datasets/open-web-math/open-web-math

* stackoverflow: https://huggingface.co/datasets/bigcode/stackoverflow-clean

Secondly, process the datasets according to https://github.com/huggingface/nanotron/blob/main/docs/nanoset.md.

## Environment

Install pytorch and other packages.

```bash
conda create -n mla-ft python=3.11
pip install -r requirements.txt
```

## Partial-RoPE Fine-Tuning

Once the checkpoint in nanotron format is ready, you can use the following command for partial-RoPE fine-tuning (FT). The config file can refer to [general configuration](./configs/rope/v4_topk4_cfg.yaml) and the [partial-RoPE configuration](./configs/mla/v4_topk4_rope.yaml).


```bash
torchrun --nproc_per_node 2 \
    -m src.run_train \
    --config-file configs/rope/v5_last8_cfg.yaml \
    --rope-cfg configs/rope/v5_last8_rope.yaml
```

> If you want to use the partial-RoPE version 4, you should get the `qk_tensor` first.
> Using the following command, you can get the `qk_tensor`:
> ```bash
>torchrun --nproc_per_node 1 \
>    src/test/test_2_norm.py \
>    --config-file configs/test/1B_2norm.yaml
>    --output-dir utils/ \
>    --sample-size 1024
> ```

| Partial-RoPE version | Strategy |
| :----: | --- |
| 0    | full-RoPE  |
| 1    | $\mathcal{S}_{\text{high}}$ |
| 2    | $\mathcal{S}_{\text{uniform}}$ |
| 3    | $\mathcal{S}_{\text{middle}}$ |
| 4    | $\mathcal{S}_{\text{2-norm}}$ |
| 5    | $\mathcal{S}_{\text{low}}$ |

## Multiple-Head Latent Attention Fine-Tuning

Use the following command for MLA fine-tuning:

```bash
torchrun --nproc_per_node 4 \
    -m src.mla_train_nt \
    --config-file ../configs/mla/rope_v4_topk4_svd_method7_rank16.yaml
```

| SVD version | Strategy |
| :----: | --- |
| 2 |  $SVD_{split}$ |
| 7 |  $SVD_{joint}$ |

## Lighteval Evaluation

For the partial-RoPE model, use the following command:

```bash
export model_name_or_path=""
export output_dir=""
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
    -m lighteval accelerate \
    --model_args "pretrained=${model_name_or_path},revision=main,dtype=bfloat16,max_length=2048" \
    --override_batch_size 50 \
    --custom_tasks "src/evaluation/tasks.py" \
    --tasks "src/evaluation/smollm1_base_v2.txt" \
    --output_dir "eval_results/${output_dir}"
```

For the MLA evaluation, you can use the following command:
```bash
export model_name_or_path=""
export output_dir=""
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export cfg_RoPE="configs/rope/v5_last8_rope.yaml"

accelerate launch --num_processes=${NUM_GPUS} \
    -m src.evaluation.eval_mla --cfg_RoPE ${cfg_RoPE} \
    accelerate \
    --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
    --override_batch_size 200 \
    --custom_tasks "src/evaluation/tasks.py" \
    --tasks "src/evaluation/smollm1_base_v2.txt" \
    --output_dir "eval_results/${output_dir}"
```


## LongBench Evaluation


For the baseline evaluation, you can use the following command:
```bash
export model_name_or_path=""
export output_dir=""
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export cfg_RoPE="configs/rope/v5_last8_rope.yaml"

torchrun --nproc_per_node=${NUM_GPUS} \
    -m src.evaluation.longbench \
    --model_path ${model_name_or_path} \
    --tokenizer_path ${model_name_or_path} \
    --longbench True \
    --lb_max_tokens 2048 \
    --lb_batch_size 16 \
    --output_dir /longbench/bf16 \
    --dtype "bfloat16"
```

For the MLA model, you should add the parameter `--is_mla` to the command.

If you want to use the quantized KV cache, you can use the following command:
```bash
export model_name_or_path=""
export output_dir=""
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

torchrun --nproc_per_node=${NUM_GPUS} \
    -m src.evaluation.longbench \
    --model_path ${model_name_or_path} \
    --tokenizer_path ${model_name_or_path} \
    --longbench True \
    --lb_max_tokens 2048 \
    --lb_batch_size 16 \
    --output_dir /longbench/${model_name_or_path}_hqq_int4 \
    --dtype "bfloat16" \
    --cache_implementation "quantized" \
    --backend "HQQ" \
    --nbits 4 \
    --residual_length 128 \
```

## Citation
```
@misc{ji2025economicalinferenceenablingdeepseeks,
      title={Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs}, 
      author={Tao Ji and Bin Guo and Yuanbin Wu and Qipeng Guo and Lixing Shen and Zhan Chen and Xipeng Qiu and Qi Zhang and Tao Gui},
      year={2025},
      eprint={2502.14837},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14837}, 
}
```
