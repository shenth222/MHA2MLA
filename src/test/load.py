from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    checkpoint = "/home/binguo/data/MLA-FT/checkpoints/7B_rope_v4_topk8_svd_method7_rank16"
    tokenizer_path = checkpoint
    import json, os

    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)
    from ..mla.mla_patch_hf import mla_patch_hf
    mla_patch_hf(config["RoPE"])

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(dtype=dtype).cuda()
    tokenizer.pad_token = tokenizer.eos_token

if __name__ == "__main__":
    main()
