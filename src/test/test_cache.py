import torch
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import argparse

def calculate_loss_direct(model, input_ids, attention_mask):
    """Calculate the loss of the sequence directly."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids 
        )
    return outputs.loss.item()

def calculate_loss_with_kvcache(model, input_ids, attention_mask):
    """Calculate the loss of the sequence using past_key_values."""
    total_loss = 0
    past_key_values = None
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        for i in range(0, seq_len-1):  
            current_input_ids = input_ids[:, i:i+1]  
            current_attention_mask = attention_mask[:, :i+1] 
            
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask[:, :i+1],  
                past_key_values=past_key_values,
                labels=None,  
                use_cache=True
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            labels = input_ids[:,i+1:i+2]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fn(logits, labels.view(-1))
            past_key_values = outputs.past_key_values
            total_loss += loss.item()
    
    return total_loss / (seq_len - 1) 


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The model name or path")
    parser.add_argument("--is_mla", action="store_true", help="Whether the model is a MLA model")
    args = parser.parse_args()
    # Monkey Patching
    if args.is_mla:
        import json,os
        from ..mla.mla_patch_hf import mla_patch_hf
        with open(os.path.join(args.model_name,"config.json"),"r") as f:
            model_config = json.load(f)
        mla_patch_hf(model_config["RoPE"])

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    text = "Hello, this is a test sentence to verify the loss calculation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    direct_loss = calculate_loss_direct(model, inputs["input_ids"], inputs["attention_mask"])
    kvcache_loss = calculate_loss_with_kvcache(model, inputs["input_ids"], inputs["attention_mask"])
    
    print(f"Direct Loss: {direct_loss:.6f}")
    print(f"KV Cache Loss: {kvcache_loss:.6f}")
    print(f"Difference: {abs(direct_loss - kvcache_loss):.6f}")




def estimate_kvcache_memory(sentence: str, model_hidden_size: int, num_heads: int, dtype_size: int = 4):
    token_kvcache_size = (model_hidden_size * 2) / num_heads 
    token_kvcache_size *= dtype_size 

    num_tokens = len(sentence)

    total_memory_bytes = token_kvcache_size * num_heads * num_tokens

    total_memory_mb = total_memory_bytes / (1024 * 1024)

    return total_memory_mb

def main2():
    from transformers.cache_utils import DynamicCache
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The model name or path")
    parser.add_argument("--is_mla", action="store_true", help="Whether the model is a MLA model")
    args = parser.parse_args()
    # Monkey Patching
    if args.is_mla:
        import json,os
        from ..mla.mla_patch_hf import mla_patch_hf
        with open(os.path.join(args.model_name,"config.json"),"r") as f:
            model_config = json.load(f)
        mla_patch_hf(model_config["RoPE"])    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    input_text = "Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation.Hello, this is a test sentence to verify the loss calculation."
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    past_key_values = DynamicCache()
    generation_kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 128, "min_new_tokens": 20,"use_cache":True,"past_key_values":past_key_values}
    begin_mem = torch.cuda.memory_allocated()
    out_fp16 = model.generate(**inputs, **generation_kwargs)
    generated_text = tokenizer.batch_decode(out_fp16)
    print(past_key_values.key_cache[0].shape)
    print(past_key_values.value_cache[0].shape)

    end_mem = torch.cuda.max_memory_allocated()
    print(f"{(end_mem - begin_mem) / 1024 ** 2} MiB VRAM used")

if __name__ == "__main__":
    main2()