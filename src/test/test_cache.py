import torch
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import argparse

def calculate_loss_direct(model, input_ids, attention_mask):
    """直接计算整句话的loss"""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 因为是自回归任务，labels就是输入
        )
    return outputs.loss.item()

def calculate_loss_with_kvcache(model, input_ids, attention_mask):
    """使用KV cache逐token计算loss"""
    total_loss = 0
    past_key_values = None
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        for i in range(0, seq_len-1):  # 从第二个token开始计算
            # 当前位置之前的输入
            current_input_ids = input_ids[:, i:i+1]  # 只取当前token
            current_attention_mask = attention_mask[:, :i+1]  # 更新attention mask
            
            # 使用past_key_values进行计算
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask[:, :i+1],  # 保持attention mask的长度一致
                past_key_values=past_key_values,
                labels=None,  # labels只包含当前token
                use_cache=True
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            labels = input_ids[:,i+1:i+2]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fn(logits, labels.view(-1))
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            # 累加loss
            total_loss += loss.item()
    
    # 计算平均loss
    return total_loss / (seq_len - 1)  # 除以预测的token数量


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

    # 加载模型和分词器
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    
    # 将模型移至GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 准备输入文本
    text = "Hello, this is a test sentence to verify the loss calculation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 计算两种方式的loss
    direct_loss = calculate_loss_direct(model, inputs["input_ids"], inputs["attention_mask"])
    kvcache_loss = calculate_loss_with_kvcache(model, inputs["input_ids"], inputs["attention_mask"])
    
    print(f"Direct Loss: {direct_loss:.6f}")
    print(f"KV Cache Loss: {kvcache_loss:.6f}")
    print(f"Difference: {abs(direct_loss - kvcache_loss):.6f}")




def estimate_kvcache_memory(sentence: str, model_hidden_size: int, num_heads: int, dtype_size: int = 4):
    """
    估算KV Cache的显存占用情况。

    参数:
        sentence (str): 输入的句子。
        model_hidden_size (int): 模型的隐藏层维度大小。
        num_heads (int): 模型的注意力头数量。
        dtype_size (int): 数据类型大小（默认为4字节，即float32）。
    
    返回:
        float: KV Cache的显存占用（以MB为单位）。
    """
    # 每个token的KV Cache大小
    token_kvcache_size = (model_hidden_size * 2) / num_heads  # Key和Value各占一半
    token_kvcache_size *= dtype_size  # 转换为字节

    # 估算句子的token数量（简单假设每个字符为一个token）
    num_tokens = len(sentence)

    # 总显存占用（字节）
    total_memory_bytes = token_kvcache_size * num_heads * num_tokens

    # 转换为MB
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
# cache_implementation="quantized",
        # cache_config = {
        #     "backend": "HQQ",
        #     "nbits": 4,
        #     "residual_length": 0,
        #     "compute_dtype": torch.bfloat16,
        #     "device": model.device,
        # }
    begin_mem = torch.cuda.memory_allocated()
    out_fp16 = model.generate(**inputs, **generation_kwargs)
    generated_text = tokenizer.batch_decode(out_fp16)
    print(past_key_values.key_cache[0].shape)
    print(past_key_values.value_cache[0].shape)

    end_mem = torch.cuda.max_memory_allocated()
    print(f"{(end_mem - begin_mem) / 1024 ** 2} MiB VRAM used")

if __name__ == "__main__":
    main2()