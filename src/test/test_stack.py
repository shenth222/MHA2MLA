import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.llama import modeling_llama
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect

def apply_rotary_pos_emb_test(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    stack = inspect.stack()
    for frame in stack:
        caller_instance = frame.frame.f_locals.get('self', None)
        if caller_instance:
            print(caller_instance.layer_idx)
            break
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def main():
    modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_test
    inputs= "Hello, What is your name?"
    model_name_or_path="~/data/models/HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    inputs = tokenizer(inputs, return_tensors="pt")
    ouputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(ouputs[0], skip_special_tokens=True))
    

if __name__ == '__main__':
    main()