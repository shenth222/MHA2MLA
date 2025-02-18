from transformers import AutoModelForCausalLM
import torch
import os
import argparse

from .init import AttnForTraing

def merge(model_name_path:str):
    original_model_path = f"{model_name_path}/0_hf"
    model_name_path = f"{model_name_path}_init/checkpoint-2000"
    from .patch_func_hf import ae_patch_func_hf
    import json
    with open(os.path.join(model_name_path,"config.json"),"r") as f:
        config = json.load(f)
    ae_patch_func_hf(config["RoPE"])
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
    model = AttnForTraing.from_pretrained(model_name_path)
    with torch.no_grad():
        for layer_idx, layer in enumerate(original_model.model.layers):
            attn = model.model[layer_idx]
            layer.self_attn.auto_encoder.W_down_k.weight.data[:] = attn.auto_encoder.W_down_k.weight.detach()
            layer.self_attn.auto_encoder.W_up_k.weight.data[:] = attn.auto_encoder.W_up_k.weight.detach()
            assert hasattr(layer.self_attn.auto_encoder,"W_down_v") == hasattr(attn.auto_encoder,"W_down_v")
            if hasattr(layer.self_attn.auto_encoder,"W_down_v"):
                layer.self_attn.auto_encoder.W_down_v.weight.data[:] = attn.auto_encoder.W_down_v.weight.detach()
                layer.self_attn.auto_encoder.W_up_v.weight.data[:] = attn.auto_encoder.W_up_v.weight.detach()
    original_model.save_pretrained(original_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path",type=str,required=True)
    args = parser.parse_args()
    merge(args.model_name_path)

if __name__ =="__main__":
    main()